"""
Kairos Thesis Validity — Signal-Validity Decision Engine

The premise of Exit Architecture v2 is that hold/sell decisions should be driven
by whether the ORIGINAL ENTRY SIGNAL is still supported by current data — NOT by
how many days have elapsed. Time windows are backstops only (handled in
kairos_thesis_review.py), never primary triggers.

This module answers one question per open position: "Is the thesis still valid?"
It does so by re-deriving each signal's validity from live data:

    HOT-INSIDER   — re-check SEC EDGAR for fresh insider SALES since entry
    HOT-CONGRESS  — re-check House/Senate Stock Watcher for congressional sales
    HOT-EARNINGS  — has the move played out (>25%)? is the next print imminent?
    HOT-REVERSION — has price reverted to its 30-day mean (win condition)?
    HOT-CHAIN     — is the upstream Tier-1 still elevated and the gap still open?
    HOT-CATALYST  — has the event_calendar catalyst passed, and how did it resolve?
    IPO_MOMENTUM  — is the stock still above its offering price?
    DEFAULT       — fall back to a price-proximity check (±15% / +30%)

Each per-signal verdict is combined with sector health (sector ETF 10-day return)
and macro regime alignment (kairos.db regime_log) into a single 0-100 thesis
score and a HOLD / WATCH / SELL action.

Graceful degradation is mandatory: yfinance, SEC EDGAR and the stock-watcher
feeds may all be unavailable. Every external call is wrapped; when a check
cannot run it returns a neutral default rather than raising. The signal-validity
component is weighted highest (60%), so a confident SELL signal still drives the
verdict even when sector/regime data is missing.

Public surface:
    check_signal_validity(ticker, signal_type, entry_date, entry_price,
                          current_price, signals_fired) -> dict
    check_sector_health(ticker, sector) -> dict
    check_regime_alignment(signal_type) -> dict
    compute_thesis_score(ticker, signal_validity, sector_health, regime) -> dict
    get_composite_verdict(ticker, signal_type, entry_date, entry_price,
                          current_price, signals_fired, sector) -> dict
"""

import json
import os
import re
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")

# SEC requires a descriptive User-Agent with a contact address.
_SEC_UA = {"User-Agent": "Kairos Trading Research jason@meridiangroup.llc"}
_SEC_TIMEOUT = 12
_CONGRESS_TIMEOUT = 15

# Sector → sector-ETF map for the 10-day sector-health check.
SECTOR_ETF = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Consumer": "XLY",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication": "XLC",
    "Communication Services": "XLC",
}
DEFAULT_ETF = "SPY"

# Thesis-score action thresholds (compute_thesis_score).
SCORE_HOLD_MIN = 65   # >= 65 → HOLD
SCORE_WATCH_MIN = 45  # 45-64 → WATCH ; < 45 → SELL


# ── small utilities ───────────────────────────────────────────────────

def _yf():
    """Return the yfinance module, or None if it isn't installed."""
    try:
        import yfinance as yf
        return yf
    except Exception:
        return None


def _config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _parse_date(raw) -> "datetime | None":
    """Parse the many timestamp shapes Kairos persists into a naive UTC datetime."""
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None)
    s = str(raw).strip().replace(" UTC", "").replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    # last resort: leading date token
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _result(valid, confidence, reason, action, signal_type) -> dict:
    return {
        "valid": bool(valid),
        "confidence": round(float(max(0.0, min(1.0, confidence))), 3),
        "reason": reason,
        "action": action,
        "signal_type": signal_type,
    }


def _close_series(ticker: str, period: str = "40d"):
    """Return a list of recent closes (oldest→newest) via yfinance, or None."""
    yf = _yf()
    if yf is None or not ticker:
        return None
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df is None or df.empty:
            return None
        closes = [float(c) for c in df["Close"].dropna().tolist()]
        return closes or None
    except Exception:
        return None


def _pct_change_since(ticker: str, entry_date: str):
    """% change of `ticker` from its close on/near entry_date to latest close."""
    yf = _yf()
    dt = _parse_date(entry_date)
    if yf is None or dt is None or not ticker:
        return None
    try:
        end = _now()
        span_days = max(7, (end - dt).days + 5)
        df = yf.Ticker(ticker).history(period=f"{span_days}d", auto_adjust=False)
        if df is None or df.empty:
            return None
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        if first <= 0:
            return None
        return (last - first) / first * 100.0
    except Exception:
        return None


# ── per-signal validity ───────────────────────────────────────────────

def check_signal_validity(ticker: str, signal_type: str, entry_date: str,
                          entry_price: float, current_price: float,
                          signals_fired: dict) -> dict:
    """Re-derive whether the entry signal is still supported by live data.

    Returns {valid, confidence, reason, action, signal_type} where action is
    one of HOLD / SELL / WATCH. Never raises — every branch degrades to a
    sensible default and dispatch is wrapped.
    """
    st = (signal_type or "").strip().upper()
    try:
        ep = float(entry_price) if entry_price else 0.0
        cp = float(current_price) if current_price else 0.0
    except (TypeError, ValueError):
        ep = cp = 0.0
    gain_pct = ((cp - ep) / ep * 100.0) if ep > 0 else 0.0

    try:
        if st == "HOT-INSIDER":
            return _validity_insider(ticker, st, entry_date, ep, cp)
        if st == "HOT-CONGRESS":
            return _validity_congress(ticker, st, entry_date)
        if st == "HOT-EARNINGS":
            return _validity_earnings(ticker, st, gain_pct)
        if st == "HOT-REVERSION":
            return _validity_reversion(ticker, st, cp)
        if st == "HOT-CHAIN":
            return _validity_chain(ticker, st, entry_date)
        if st == "HOT-CATALYST":
            return _validity_catalyst(ticker, st, entry_date, ep, cp)
        if st == "IPO_MOMENTUM":
            return _validity_ipo(ticker, st, cp, entry_date, signals_fired)
        return _validity_default(st or "DEFAULT", gain_pct)
    except Exception as exc:
        # Engine-level safety net: never let a single signal kill the review.
        return _result(True, 0.5, f"validity check errored ({exc}); defaulting to HOLD",
                       "HOLD", st or "DEFAULT")


def _validity_insider(ticker, st, entry_date, ep, cp) -> dict:
    sale, detail = _insider_sale_since(ticker, entry_date)
    if sale:
        return _result(False, 0.3,
                       f"insider SALE filed since entry ({detail}) — thesis reversed",
                       "SELL", st)
    if ep > 0 and cp >= ep * 3.0:
        return _result(False, 0.6,
                       f"price up {((cp-ep)/ep*100):.0f}% (>=3x) — insider thesis "
                       f"complete, lock in",
                       "SELL", st)
    if ep > 0 and cp >= ep * 2.5:
        return _result(True, 0.5,
                       f"price up {((cp-ep)/ep*100):.0f}% (approaching 3x cap) — "
                       f"no insider sales; hold but watch",
                       "WATCH", st)
    return _result(True, 1.0,
                   "no insider sales filed since entry — buy thesis intact",
                   "HOLD", st)


def _validity_congress(ticker, st, entry_date) -> dict:
    sold, detail = _congress_sale_since(ticker, entry_date)
    if sold:
        return _result(False, 0.1,
                       f"congressional SALE disclosed since entry ({detail})",
                       "SELL", st)
    return _result(True, 0.9,
                   "no congressional sale disclosed — follow-the-money thesis intact",
                   "HOLD", st)


def _validity_earnings(ticker, st, gain_pct) -> dict:
    if gain_pct >= 25.0:
        return _result(False, 0.85,
                       f"price up {gain_pct:.1f}% (>=25%) — earnings move has played "
                       f"out, thesis complete",
                       "SELL", st)
    days = _days_to_next_earnings(ticker)
    if days is not None and 0 <= days <= 14:
        return _result(True, 0.55,
                       f"next earnings in ~{days}d — catalyst approaching, re-evaluate "
                       f"holding through vs. exiting",
                       "WATCH", st)
    return _result(True, 0.9,
                   f"move not yet played out ({gain_pct:+.1f}%) and no earnings "
                   f"imminent — thesis active",
                   "HOLD", st)


def _validity_reversion(ticker, st, cp) -> dict:
    closes = _close_series(ticker, period="45d")
    if not closes or cp <= 0:
        # Can't compute the mean — fall back to neutral hold.
        return _result(True, 0.5,
                       "30d SMA unavailable — cannot confirm reversion; hold",
                       "HOLD", st)
    window = closes[-30:] if len(closes) >= 30 else closes
    sma = sum(window) / len(window)
    if sma <= 0:
        return _result(True, 0.5, "invalid SMA — hold", "HOLD", st)
    ratio = cp / sma
    if ratio >= 0.98:
        # Reverted to (or above) the mean — this is the win condition.
        return _result(False, 0.85,
                       f"price ${cp:.2f} reverted to 30d SMA ${sma:.2f} "
                       f"({(ratio-1)*100:+.1f}%) — reversion complete, take the win",
                       "SELL", st)
    # Still below the mean: more upside to the reversion. Confidence scales
    # with remaining distance (further below mean → more conviction to hold).
    distance = (0.98 - ratio)  # how far below the 0.98 trigger, 0..~0.5
    confidence = max(0.5, min(1.0, 0.6 + distance * 4))
    return _result(True, confidence,
                   f"price ${cp:.2f} still {((ratio-1)*100):.1f}% below 30d SMA "
                   f"${sma:.2f} — reversion in progress",
                   "HOLD", st)


def _validity_chain(ticker, st, entry_date) -> dict:
    try:
        from kairos_signals_chain import get_upstream_tickers
        upstream = get_upstream_tickers(ticker)
    except Exception:
        upstream = []
    if not upstream:
        return _result(True, 0.5,
                       "no upstream chain ticker resolved — hold on default",
                       "HOLD", st)

    up = upstream[0]
    up_move = _pct_change_since(up, entry_date)        # upstream since entry
    down_move = _pct_change_since(ticker, entry_date)  # this name since entry

    if up_move is None or down_move is None:
        return _result(True, 0.5,
                       f"chain prices unavailable (upstream {up}) — hold on default",
                       "HOLD", st)

    upstream_elevated = up_move > 2.0
    gap = up_move - down_move          # how much catch-up is left
    gap_closed = gap < 1.0             # downstream has repriced to upstream

    if gap_closed:
        return _result(False, 0.8,
                       f"downstream repriced (gap {gap:+.1f}% vs upstream {up} "
                       f"{up_move:+.1f}%) — catch-up complete",
                       "SELL", st)
    if upstream_elevated:
        return _result(True, 0.85,
                       f"upstream {up} still elevated ({up_move:+.1f}%) and gap open "
                       f"({gap:+.1f}%) — catch-up thesis active",
                       "HOLD", st)
    # Upstream faded but gap not closed → thesis weakening.
    return _result(True, 0.45,
                   f"upstream {up} faded ({up_move:+.1f}%); gap {gap:+.1f}% — "
                   f"weakening, watch",
                   "WATCH", st)


def _validity_catalyst(ticker, st, entry_date, ep, cp) -> dict:
    event = _catalyst_event(ticker)
    if not event:
        return _result(True, 0.5,
                       "no matching event_calendar catalyst — hold on default",
                       "HOLD", st)
    ev_date = event["date"]
    entry_dt = _parse_date(entry_date) or _now()
    now = _now()
    name = event.get("event", "event")

    if ev_date and ev_date < now and ev_date >= entry_dt:
        # Catalyst has fired since entry → evaluate the outcome.
        if ep > 0 and cp > ep * 1.05:
            return _result(True, 0.7,
                           f"{name} passed; price up {((cp-ep)/ep*100):.1f}% — WIN, "
                           f"hold for post-event momentum",
                           "HOLD", st)
        if ep > 0 and cp < ep:
            return _result(False, 0.3,
                           f"{name} passed; price below entry "
                           f"({((cp-ep)/ep*100):.1f}%) — catalyst failed",
                           "SELL", st)
        return _result(True, 0.5,
                       f"{name} passed; price roughly flat — neutral outcome, watch",
                       "WATCH", st)
    if ev_date and ev_date >= now:
        return _result(True, 0.85,
                       f"{name} still upcoming (~{(ev_date - now).days}d) — "
                       f"catalyst thesis active",
                       "HOLD", st)
    return _result(True, 0.5,
                   f"{name} timing indeterminate — hold on default",
                   "HOLD", st)


def _validity_ipo(ticker, st, cp, entry_date, signals_fired) -> dict:
    offer = _ipo_offering_price(ticker)
    if offer is None or offer <= 0:
        return _result(True, 0.5,
                       "offering price unavailable — hold on default",
                       "HOLD", st)
    if cp < offer * 0.95:
        return _result(False, 0.2,
                       f"price ${cp:.2f} broke below offering ${offer:.2f} "
                       f"(<0.95x) — bail",
                       "SELL", st)
    if cp <= offer:
        return _result(True, 0.45,
                       f"price ${cp:.2f} at/just under offering ${offer:.2f} — "
                       f"fragile, watch",
                       "WATCH", st)
    # Above offering and holding. If aged past 30d with no fresh catalyst, watch.
    entry_dt = _parse_date(entry_date)
    days_held = (_now() - entry_dt).days if entry_dt else 0
    fresh = bool(signals_fired) and any(
        str(s).upper() not in ("", "IPO_MOMENTUM")
        for s in (signals_fired.get("current", []) if isinstance(signals_fired, dict)
                  else signals_fired or [])
    )
    if days_held > 30 and not fresh:
        return _result(True, 0.5,
                       f"above offering ${offer:.2f} but {days_held}d in with no new "
                       f"catalyst — watch",
                       "WATCH", st)
    return _result(True, 0.9,
                   f"price ${cp:.2f} above offering ${offer:.2f} — momentum intact",
                   "HOLD", st)


def _validity_default(st, gain_pct) -> dict:
    if gain_pct < -15.0:
        return _result(False, 0.3,
                       f"down {gain_pct:.1f}% (>15% loss) — safety-net stop territory",
                       "SELL", st)
    if gain_pct > 30.0:
        return _result(True, 0.5,
                       f"up {gain_pct:.1f}% (>30%) — consider locking in",
                       "WATCH", st)
    return _result(True, 0.7,
                   f"price within range ({gain_pct:+.1f}%) — no thesis to invalidate, "
                   f"hold",
                   "HOLD", st)


# ── external-data helpers (all best-effort) ───────────────────────────

_SEC_TICKER_MAP = None  # cached ticker→CIK map for the process lifetime


def _sec_cik(ticker: str):
    """Resolve a ticker to a zero-padded 10-digit CIK via SEC's mapping file."""
    global _SEC_TICKER_MAP
    if not ticker:
        return None
    try:
        import requests
        if _SEC_TICKER_MAP is None:
            resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                                headers=_SEC_UA, timeout=_SEC_TIMEOUT)
            data = resp.json()
            _SEC_TICKER_MAP = {
                str(v.get("ticker", "")).upper(): int(v.get("cik_str"))
                for v in data.values() if v.get("ticker")
            }
        cik = _SEC_TICKER_MAP.get(ticker.upper())
        return f"{cik:010d}" if cik is not None else None
    except Exception:
        return None


def _insider_sale_since(ticker: str, entry_date: str) -> "tuple[bool, str]":
    """Best-effort: was any insider SALE (Form 4, code 'S') filed since entry?

    Returns (sale_detected, detail). On any failure returns (False, reason) so
    the caller treats the thesis as intact rather than force-selling on a
    transient EDGAR error.
    """
    cik = _sec_cik(ticker)
    if cik is None:
        return False, "EDGAR CIK unresolved"
    entry_dt = _parse_date(entry_date)
    try:
        import requests
        resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                            headers=_SEC_UA, timeout=_SEC_TIMEOUT)
        recent = resp.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        checked = 0
        for i, form in enumerate(forms):
            if str(form).strip() != "4":
                continue
            fdate = _parse_date(dates[i]) if i < len(dates) else None
            if entry_dt and fdate and fdate < entry_dt:
                continue  # filed before we entered
            if checked >= 8:
                break
            checked += 1
            # Pull the Form 4 primary doc and look for an open-market sale.
            try:
                acc = accs[i].replace("-", "") if i < len(accs) else ""
                doc = docs[i] if i < len(docs) else ""
                if not acc or not doc:
                    continue
                cik_int = int(cik)
                url = (f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                       f"{acc}/{doc}")
                body = requests.get(url, headers=_SEC_UA,
                                    timeout=_SEC_TIMEOUT).text
                codes = re.findall(r"<transactionCode>\s*([A-Z])\s*</transactionCode>",
                                   body)
                if "S" in codes:  # 'S' = open-market sale
                    return True, f"Form 4 filed {dates[i]} (code S)"
            except Exception:
                continue
        return False, f"no sale in {checked} Form 4 filing(s) since entry"
    except Exception as exc:
        return False, f"EDGAR unavailable ({exc})"


def _congress_sale_since(ticker: str, entry_date: str) -> "tuple[bool, str]":
    """Best-effort: any House/Senate SELL disclosure for ticker since entry."""
    if not ticker:
        return False, "no ticker"
    entry_dt = _parse_date(entry_date)
    feeds = (
        "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
        "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
    )
    try:
        import requests
        for url in feeds:
            try:
                rows = requests.get(url, timeout=_CONGRESS_TIMEOUT).json()
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for r in rows:
                if str(r.get("ticker", "")).upper() != ticker.upper():
                    continue
                ttype = str(r.get("type") or r.get("transaction_type") or "").lower()
                if "sale" not in ttype and "sell" not in ttype:
                    continue
                tdate = _parse_date(r.get("transaction_date") or r.get("disclosure_date"))
                if entry_dt and tdate and tdate < entry_dt:
                    continue
                who = r.get("representative") or r.get("senator") or "member"
                return True, f"{who} {ttype} {r.get('transaction_date', '?')}"
        return False, "no congressional sale since entry"
    except Exception as exc:
        return False, f"stock-watcher unavailable ({exc})"


def _days_to_next_earnings(ticker: str):
    """Days until the next earnings date via yfinance, or None."""
    yf = _yf()
    if yf is None or not ticker:
        return None
    try:
        t = yf.Ticker(ticker)
        cal = None
        try:
            cal = t.get_earnings_dates(limit=8)
        except Exception:
            cal = None
        now = _now()
        if cal is not None and not getattr(cal, "empty", True):
            future = [d for d in cal.index.tolist()]
            for d in sorted(future):
                try:
                    dd = datetime(d.year, d.month, d.day)
                except Exception:
                    continue
                if dd >= now:
                    return (dd - now).days
        # Fallback: the .calendar attribute
        info_cal = getattr(t, "calendar", None)
        if isinstance(info_cal, dict):
            ed = info_cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                dd = _parse_date(str(ed[0]))
                if dd:
                    return (dd - now).days
        return None
    except Exception:
        return None


def _catalyst_event(ticker: str):
    """Most relevant event_calendar entry for ticker with a resolved date."""
    if not ticker:
        return None
    cfg = _config()
    events = cfg.get("event_calendar", [])
    if not isinstance(events, list):
        return None
    now = _now()
    best = None
    for e in events:
        if not isinstance(e, dict):
            continue
        if str(e.get("ticker", "")).upper() != ticker.upper():
            continue
        month = e.get("month")
        day = e.get("day_estimate", 15)
        if not month:
            continue
        # Resolve the event to the nearest occurrence (this year or next).
        try:
            cand = datetime(now.year, int(month), int(day))
        except Exception:
            continue
        # If this year's date is far in the past, the next occurrence is next year.
        if (now - cand).days > 60:
            try:
                cand = datetime(now.year + 1, int(month), int(day))
            except Exception:
                pass
        entry = dict(e)
        entry["date"] = cand
        if best is None or abs((cand - now).days) < abs((best["date"] - now).days):
            best = entry
    return best


def _ipo_offering_price(ticker: str):
    """Offering price from kairos_signals_ipo context / ipo cache, or None."""
    try:
        from kairos_signals_ipo import get_ipo_context
        ctx = get_ipo_context(ticker) or {}
        price = ctx.get("ipo_price")
        if price:
            return float(price)
    except Exception:
        pass
    # Fallback: scan the IPO cache directly for an offering/offer price.
    try:
        cache_file = os.path.join(SCRIPT_DIR, "kairos_ipo_cache.json")
        with open(cache_file) as f:
            cache = json.load(f)
        node = (cache.get("detected", {}) or {}).get(ticker.upper(), {})
        for key in ("offering_price", "ipo_price", "offer_price", "price"):
            if node.get(key):
                return float(node[key])
    except Exception:
        pass
    return None


# ── sector health ─────────────────────────────────────────────────────

def check_sector_health(ticker: str, sector: str) -> dict:
    """10-day return of the relevant sector ETF.

    healthy if return > -5%, unhealthy if < -8%. Degrades to a neutral
    "healthy/unknown" verdict when yfinance is unavailable so it never
    dominates the score on missing data.
    """
    sec = (sector or "").strip()
    if not sec:
        sec = _resolve_sector(ticker) or ""
    etf = SECTOR_ETF.get(sec, DEFAULT_ETF)
    closes = _close_series(etf, period="15d")
    if not closes or len(closes) < 2:
        return {"healthy": True, "sector_return_10d": None, "etf": etf,
                "note": "sector data unavailable — assumed neutral"}
    window = closes[-11:] if len(closes) >= 11 else closes
    first, last = window[0], window[-1]
    ret = ((last - first) / first * 100.0) if first > 0 else 0.0
    healthy = ret > -5.0
    return {
        "healthy": healthy,
        "sector_return_10d": round(ret, 2),
        "etf": etf,
        "unhealthy": ret < -8.0,
    }


def _resolve_sector(ticker: str):
    """Best-effort sector lookup via yfinance .info, or None."""
    yf = _yf()
    if yf is None or not ticker:
        return None
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") if isinstance(info, dict) else None
    except Exception:
        return None


# ── regime alignment ──────────────────────────────────────────────────

def check_regime_alignment(signal_type: str) -> dict:
    """Read the latest market regime and decide whether the thesis aligns.

    NORMAL / CAUTION → aligned for most signals. RISK-OFF / EXTREME-FEAR →
    not aligned, theses should be re-evaluated — EXCEPT HOT-REVERSION, which
    is *more* valid in EXTREME-FEAR (mean-reversion opportunity).
    """
    st = (signal_type or "").strip().upper()
    regime = None
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        row = conn.execute(
            "SELECT regime FROM regime_log ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            regime = (row["regime"] if hasattr(row, "keys") else row[0])
    except Exception:
        regime = None

    if regime is None:
        return {"aligned": True, "regime": "UNKNOWN",
                "reason": "regime unavailable — assumed aligned"}

    reg = str(regime).strip().upper()
    risk_off = reg in ("RISK-OFF", "RISK_OFF", "EXTREME-FEAR", "EXTREME_FEAR",
                       "EXTREME FEAR")
    extreme_fear = "EXTREME" in reg and "FEAR" in reg

    if st == "HOT-REVERSION" and extreme_fear:
        return {"aligned": True, "regime": reg,
                "reason": "EXTREME-FEAR amplifies mean-reversion edge"}
    if risk_off:
        return {"aligned": False, "regime": reg,
                "reason": f"{reg} regime — re-evaluate thesis under risk-off"}
    return {"aligned": True, "regime": reg,
            "reason": f"{reg} regime supports holding"}


# ── composite scoring ─────────────────────────────────────────────────

def compute_thesis_score(ticker: str, signal_validity: dict,
                         sector_health: dict, regime: dict) -> dict:
    """Blend signal validity (60%), sector health (25%) and regime (15%) into a
    0-100 thesis score, then map to a HOLD / WATCH / SELL action.

    The signal's own action is authoritative (it can decide a high-confidence
    "thesis complete" SELL/WATCH that the raw score wouldn't surface); sector
    and regime can only make the verdict MORE cautious, never less.
    """
    sv = signal_validity or {}
    confidence = float(sv.get("confidence", 0.5))
    valid = bool(sv.get("valid", True))
    sig_action = (sv.get("action") or "HOLD").upper()

    # Signal component (0..1). An invalidated signal is capped low.
    sig_factor = confidence if valid else min(confidence, 0.3)

    # Sector component (0..1).
    if sector_health.get("sector_return_10d") is None:
        sec_factor = 0.6  # unknown → mildly positive default
    elif sector_health.get("unhealthy"):
        sec_factor = 0.0
    elif sector_health.get("healthy"):
        sec_factor = 1.0
    else:
        sec_factor = 0.5

    # Regime component (0..1).
    reg_factor = 1.0 if regime.get("aligned", True) else 0.0
    if regime.get("regime") == "UNKNOWN":
        reg_factor = 0.6

    score = int(round(sig_factor * 60 + sec_factor * 25 + reg_factor * 15))
    score = max(0, min(100, score))

    # Signal action is primary; macro can only escalate caution.
    action = sig_action if sig_action in ("HOLD", "WATCH", "SELL") else "HOLD"
    if action == "HOLD":
        if score < SCORE_WATCH_MIN:
            action = "SELL"
        elif score < SCORE_HOLD_MIN:
            action = "WATCH"
    elif action == "WATCH" and score < 30:
        action = "SELL"

    rationale = sv.get("reason", "no signal rationale")
    macro_bits = []
    if sector_health.get("sector_return_10d") is not None:
        macro_bits.append(
            f"sector {sector_health['etf']} {sector_health['sector_return_10d']:+.1f}%/10d"
            + ("" if sector_health.get("healthy") else " (weak)")
        )
    if regime.get("regime") and regime.get("regime") != "UNKNOWN":
        macro_bits.append(
            f"regime {regime['regime']}"
            + ("" if regime.get("aligned") else " (misaligned)")
        )
    if macro_bits:
        rationale = f"{rationale}; {', '.join(macro_bits)}"

    return {"score": score, "action": action, "rationale": rationale}


def get_composite_verdict(ticker: str, signal_type: str, entry_date: str,
                          entry_price: float, current_price: float,
                          signals_fired: dict, sector: str) -> dict:
    """Run all three checks + scoring and return a full verdict.

    Graceful degradation: any sub-check that fails falls back to a neutral
    default rather than raising, so the verdict is always well-formed.
    """
    try:
        sv = check_signal_validity(ticker, signal_type, entry_date,
                                   entry_price, current_price, signals_fired)
    except Exception as exc:
        sv = _result(True, 0.5, f"signal check failed ({exc})", "HOLD",
                     (signal_type or "DEFAULT"))
    try:
        sh = check_sector_health(ticker, sector)
    except Exception as exc:
        sh = {"healthy": True, "sector_return_10d": None, "etf": DEFAULT_ETF,
              "note": f"sector check failed ({exc})"}
    try:
        rg = check_regime_alignment(signal_type)
    except Exception as exc:
        rg = {"aligned": True, "regime": "UNKNOWN", "reason": f"regime failed ({exc})"}

    scored = compute_thesis_score(ticker, sv, sh, rg)

    return {
        "ticker": ticker,
        "signal_type": (signal_type or "DEFAULT"),
        "action": scored["action"],
        "score": scored["score"],
        "rationale": scored["rationale"],
        "valid": sv.get("valid"),
        "confidence": sv.get("confidence"),
        "signal_reason": sv.get("reason"),
        "signal_action": sv.get("action"),
        "sector_health": sh,
        "regime": rg,
    }


# ── Cycle-scoped validity cache ───────────────────────────────────────

def _primary_signal_from(signals_fired, ticker: str):
    """Resolve the primary (thesis-leading) signal for a position.

    Accepts signals_fired as a list, a {"entry": [...]} dict, or None; falls
    back to the recorded entry signals in kairos.db. Uses the ml_thesis
    priority ranking when available.
    """
    sigs = []
    if isinstance(signals_fired, dict):
        sigs = list(signals_fired.get("entry") or signals_fired.get("current") or [])
    elif isinstance(signals_fired, (list, tuple)):
        sigs = list(signals_fired)
    if not sigs:
        try:
            from kairos_exits import _get_entry_signals
            sigs = _get_entry_signals(ticker) or []
        except Exception:
            sigs = []
    try:
        from kairos_ml_thesis import pick_primary_signal
        return pick_primary_signal(sigs)
    except Exception:
        return sigs[0] if sigs else None


def _best_effort_price(ticker: str):
    """Cheap live-price fetch for cache warm-up (Finnhub, then yfinance). None
    if neither is available — caller treats a missing price as neutral."""
    try:
        import requests
        api_key = os.environ.get("FINNHUB_API_KEY")
        if api_key:
            resp = requests.get("https://finnhub.io/api/v1/quote",
                                params={"symbol": ticker, "token": api_key},
                                timeout=8)
            c = resp.json().get("c", 0)
            if c and c > 0:
                return float(c)
    except Exception:
        pass
    closes = _close_series(ticker, period="5d")  # yfinance; None when absent
    return closes[-1] if closes else None


def warm_validity_cache(tickers: list, holdings: list) -> dict:
    """Score validity for all open positions once per cycle.

    Returns {ticker: {score, action, rationale, signal_type, confidence}}.

    Each holding is scored independently inside try/except so one failure never
    blocks the rest. When no live price can be obtained for a position the entry
    is stored as NEUTRAL (score 60) so conviction decay falls back to its
    existing time-based behavior rather than being skewed by a fabricated gain.
    """
    import time
    start = time.time()
    cache: dict = {}

    for h in (holdings or []):
        ticker = None
        try:
            if not isinstance(h, dict):
                h = dict(h)
            ticker = h.get("ticker") or h.get("symbol")
            if not ticker:
                continue

            entry_price = h.get("avg_cost") or h.get("entry_price") or 0.0
            try:
                entry_price = float(entry_price)
            except (TypeError, ValueError):
                entry_price = 0.0
            entry_date = (h.get("earliest_entry") or h.get("entry_date")
                          or h.get("entry_ts") or "")
            sector = h.get("sector") or ""
            signals_fired = h.get("signals_fired") or h.get("entry_signals")
            signal_type = _primary_signal_from(signals_fired, ticker)

            # Live price: provided → best-effort fetch → neutral fallback.
            current_price = h.get("current_price")
            price_is_real = bool(current_price and current_price > 0)
            if not price_is_real:
                fetched = _best_effort_price(ticker)
                if fetched and fetched > 0:
                    current_price, price_is_real = fetched, True
            if not price_is_real:
                current_price = entry_price  # neutral gain (0%) for the verdict

            verdict = get_composite_verdict(
                ticker=ticker, signal_type=signal_type, entry_date=entry_date,
                entry_price=entry_price, current_price=current_price,
                signals_fired=(signals_fired if isinstance(signals_fired, dict)
                               else {"entry": signals_fired or [], "current": []}),
                sector=sector,
            )

            if price_is_real:
                cache[ticker] = {
                    "score": verdict["score"],
                    "action": verdict["action"],
                    "rationale": verdict["rationale"],
                    "signal_type": verdict["signal_type"],
                    "confidence": verdict.get("confidence"),
                }
            else:
                cache[ticker] = {
                    "score": 60, "action": "HOLD",
                    "rationale": "no live price — neutral (time-based decay preserved)",
                    "signal_type": signal_type or "DEFAULT",
                    "confidence": 0.5,
                }
        except Exception as exc:
            print(f"  Validity cache: {ticker or '?'} failed ({exc})")
            continue

    elapsed = time.time() - start
    print(f"  Validity cache: scored {len(cache)} positions ({elapsed:.1f}s)")
    return cache


def get_cached_validity(ticker: str, cache: "dict | None" = None) -> dict:
    """Look up a position's cached validity, with safe neutral defaults.

    Missing cache or missing ticker → {score 60, HOLD} so any consumer
    (e.g. conviction decay) degrades to its baseline behavior.
    """
    default = {"score": 60, "action": "HOLD", "rationale": "no cache",
               "confidence": 0.5}
    if not cache or ticker not in cache:
        return default
    return cache.get(ticker) or default


# ── CLI / self-test ───────────────────────────────────────────────────

def _demo(ticker, signal_type, entry_date, entry_price, current_price,
          signals_fired=None, sector=""):
    verdict = get_composite_verdict(
        ticker=ticker, signal_type=signal_type, entry_date=entry_date,
        entry_price=entry_price, current_price=current_price,
        signals_fired=signals_fired or {}, sector=sector,
    )
    gain = (current_price - entry_price) / entry_price * 100 if entry_price else 0
    print(f"\n  {ticker} [{signal_type}] entry {entry_date} "
          f"${entry_price:.2f}→${current_price:.2f} ({gain:+.2f}%)")
    print(f"    ACTION : {verdict['action']}   (score {verdict['score']}/100)")
    print(f"    SIGNAL : {verdict['signal_action']} — {verdict['signal_reason']}")
    print(f"    SECTOR : {verdict['sector_health']}")
    print(f"    REGIME : {verdict['regime']}")
    print(f"    WHY    : {verdict['rationale']}")
    return verdict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kairos thesis validity engine")
    parser.add_argument("--ticker", default="AMZN")
    parser.add_argument("--signal", default="HOT-EARNINGS")
    parser.add_argument("--entry-date")
    parser.add_argument("--entry-price", type=float, default=200.0)
    parser.add_argument("--current-price", type=float)
    parser.add_argument("--held-days", type=int, default=5)
    parser.add_argument("--gain-pct", type=float, default=-3.74)
    args = parser.parse_args()

    from datetime import timedelta
    entry_date = args.entry_date or (
        _now() - timedelta(days=args.held_days)).strftime("%Y-%m-%d")
    current_price = args.current_price or round(
        args.entry_price * (1 + args.gain_pct / 100.0), 2)

    print("=" * 72)
    print("  THESIS VALIDITY ENGINE — verdict demo")
    print("=" * 72)
    _demo(args.ticker, args.signal, entry_date, args.entry_price, current_price)
