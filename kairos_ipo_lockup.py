"""
Kairos HOT-IPO Lock-Up Expiration Tracker (Phase 3)

180-day lock-up expirations create predictable selling pressure for
overvalued IPOs. This module tracks lock-up dates for recently priced
IPOs, scores a short thesis 30 days out, and — when the thesis is strong
enough — ARMS it in the ipo_lockup_tracker table. The catalyst detector
(kairos_catalyst_signals._detect_lockup) reads armed rows each cycle and
emits long-PUT signals, so the HOT-CATALYST options engine picks them up
automatically. No options orders are placed here.

Sources for priced IPOs + Day-1 date (priority order):
  1. ipo_reservations status='converted' — IPOs we actually traded.
     converted_at = Day 1, converted_avg_price = IPO price. Most reliable.
  2. kairos_ipo_pipeline.json entries that have a ticker (priced).
  3. Renaissance Capital /basic-priced/PricedIPOs (when API key present).
  4. Renaissance CompanyIpoDate / EDGAR as a per-ticker fallback.

Lock-up = Day 1 + lockup_days (default 180 calendar days).

Short-thesis scoring (0-10), only inside the scoring window:
  performance / valuation vs IPO price (0-6) — still elevated = higher
  insider concentration from S-1 (0-3)            — best-effort EDGAR
  timing within window (0-1)
A perf gate (min_perf_since_ipo_pct) blocks arming a flat/broken IPO.

Two-lock dry-run, mirroring the rest of the IPO pipeline: arming a real
signal requires BOTH config hot_ipo.dry_run=false AND --no-dry-run. In
dry-run the cycle scores and reports but never flips a row to 'armed'.
(Even when armed, the options executor's own hot_catalyst.dry_run still
gates whether a real put is placed — a third safety layer.)

Public surface:
    load_lockup_config()                 -> dict
    discover_priced_ipos()               -> list[dict]
    compute_lockup_date(ipo_date)        -> str | None
    score_short_thesis(candidate, cfg)   -> dict
    run_lockup_cycle(dry_run=True)       -> dict
"""

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
PIPELINE_FILE = os.path.join(SCRIPT_DIR, "kairos_ipo_pipeline.json")

W = 72
SEC_HEADERS = {
    "User-Agent": "Kairos Trading System jelmore@kairos.local",
    "Accept": "application/json, text/html",
}
FETCH_TIMEOUT = 20

# Defaults (overridden by kairos_config.json["hot_ipo"] / ["hot_ipo"]["lockup"])
DEFAULT_LOCKUP_DAYS = 180
DEFAULT_SCORING_WINDOW_DAYS = 30
DEFAULT_SCORE_THRESHOLD = 7
DEFAULT_MIN_PERF_PCT = 20.0
SLACK_CHANNEL = "alerts"


# ── Config ───────────────────────────────────────────────────────────

def load_lockup_config() -> dict:
    """Read hot_ipo + hot_ipo.lockup from config, with defaults.

    Returns a flat dict: {dry_run, enabled, lockup_days, scoring_window_days,
    score_threshold, min_perf_since_ipo_pct}.
    """
    cfg = {
        "dry_run": True,
        "enabled": True,
        "lockup_days": DEFAULT_LOCKUP_DAYS,
        "scoring_window_days": DEFAULT_SCORING_WINDOW_DAYS,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "min_perf_since_ipo_pct": DEFAULT_MIN_PERF_PCT,
    }
    try:
        with open(CONFIG_FILE) as f:
            raw = json.load(f)
        hot_ipo = raw.get("hot_ipo", {})
        if isinstance(hot_ipo, dict):
            cfg["dry_run"] = bool(hot_ipo.get("dry_run", True))
            lk = hot_ipo.get("lockup", {})
            if isinstance(lk, dict):
                cfg["enabled"] = bool(lk.get("enabled", True))
                cfg["lockup_days"] = int(lk.get("lockup_days", DEFAULT_LOCKUP_DAYS))
                cfg["scoring_window_days"] = int(
                    lk.get("scoring_window_days", DEFAULT_SCORING_WINDOW_DAYS))
                cfg["score_threshold"] = float(
                    lk.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
                cfg["min_perf_since_ipo_pct"] = float(
                    lk.get("min_perf_since_ipo_pct", DEFAULT_MIN_PERF_PCT))
    except (IOError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return cfg


# ── Small helpers ────────────────────────────────────────────────────

def banner(title: str) -> str:
    return "\n" + "━" * W + f"\n  {title}\n" + "━" * W


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_only(value) -> Optional[str]:
    """Coerce a date/datetime string to YYYY-MM-DD; None if unusable."""
    if not value:
        return None
    s = str(value).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    for fmt in ("%m/%d/%Y", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


def compute_lockup_date(ipo_date, lockup_days: int = DEFAULT_LOCKUP_DAYS) -> Optional[str]:
    """Day 1 + lockup_days, as YYYY-MM-DD. None if ipo_date unparseable."""
    d = _date_only(ipo_date)
    if not d:
        return None
    try:
        base = datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        return None
    return (base + timedelta(days=int(lockup_days))).strftime("%Y-%m-%d")


def _days_until(date_str: str) -> Optional[int]:
    d = _date_only(date_str)
    if not d:
        return None
    try:
        target = datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (target - datetime.now(timezone.utc).date()).days


def _current_price(ticker: str) -> Optional[float]:
    """Latest close via yfinance. None on any failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if hist is None or len(hist) == 0:
            return None
        px = float(hist["Close"].iloc[-1])
        return px if px > 0 else None
    except Exception:
        return None


# ── Discovery ────────────────────────────────────────────────────────

def _discover_from_reservations() -> list[dict]:
    """Converted IPO reservations: the IPOs we actually traded on Day 1."""
    out: list[dict] = []
    try:
        import kairos_ipo_capital as cap
        from kairos_log_db import get_connection
        cap.init_schema()
        conn = get_connection()
        rows = conn.execute(
            "SELECT ticker, company_name, converted_at, converted_avg_price "
            "FROM ipo_reservations WHERE status = 'converted' "
            "AND converted_at IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"  WARNING: reservation discovery failed: {exc}")
        return out
    for r in rows:
        ipo_date = _date_only(r["converted_at"])
        if not ipo_date:
            continue
        out.append({
            "ticker": (r["ticker"] or "").upper(),
            "company_name": r["company_name"],
            "cik": None,
            "ipo_date": ipo_date,
            "ipo_price": r["converted_avg_price"],
            "source": "reservation",
        })
    return out


def _pipeline_ipo_price(entry: dict) -> Optional[float]:
    """Midpoint of the most recent filing's price range, if any."""
    filings = entry.get("filings") or []
    for f in reversed(filings):
        pr = (f or {}).get("price_range") or {}
        lo, hi = pr.get("low"), pr.get("high")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi > 0:
            return round((lo + hi) / 2.0, 4)
    return None


def _discover_from_pipeline() -> list[dict]:
    """Pipeline entries that carry a ticker (i.e. have priced / are public)."""
    out: list[dict] = []
    try:
        with open(PIPELINE_FILE) as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError):
        return out
    pipeline = data.get("pipeline", {}) if isinstance(data, dict) else {}
    for entry in pipeline.values():
        if not isinstance(entry, dict):
            continue
        ticker = (entry.get("ticker") or "").strip().upper()
        if not ticker:
            continue  # not yet priced / no symbol
        ipo_date = _date_only(entry.get("expected_pricing_date"))
        if not ipo_date:
            continue
        out.append({
            "ticker": ticker,
            "company_name": entry.get("company"),
            "cik": entry.get("cik"),
            "ipo_date": ipo_date,
            "ipo_price": _pipeline_ipo_price(entry),
            "source": "pipeline",
        })
    return out


def _extract_priced_fields(item: dict) -> Optional[dict]:
    """Defensive field extraction from a Renaissance PricedIPOs item."""
    if not isinstance(item, dict):
        return None
    def pick(keys):
        for k in keys:
            if k in item and item[k] not in (None, ""):
                return item[k]
        return None
    ticker = pick(("Ticker", "ticker", "Symbol", "symbol", "TickerSymbol"))
    if not ticker:
        return None
    date = pick(("PricingDate", "OfferDate", "offerDate", "ipoDate",
                 "IPODate", "Date", "TradeDate"))
    price = pick(("OfferPrice", "offerPrice", "IPOPrice", "Price", "PriceUSD"))
    company = pick(("Company", "CompanyName", "company", "name", "Name"))
    try:
        price = float(price) if price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    return {
        "ticker": str(ticker).strip().upper(),
        "company_name": company,
        "cik": None,
        "ipo_date": _date_only(date),
        "ipo_price": price,
        "source": "renaissance",
    }


def _discover_from_renaissance(lockup_days: int, window_days: int) -> list[dict]:
    """Priced IPOs whose lock-up could be approaching (best-effort)."""
    out: list[dict] = []
    try:
        from kairos_renaissance import get_priced_ipos, _load_api_key
    except ImportError:
        return out
    try:
        if not _load_api_key():
            return out  # no key -> skip silently
    except Exception:
        return out
    # IPOs priced ~ (lockup_days) ago have lock-ups expiring ~now. Pull a
    # trailing window that brackets that, padded by the scoring window.
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lockup_days + window_days + 15)
    try:
        items = get_priced_ipos(start.strftime("%Y-%m-%d"),
                                end.strftime("%Y-%m-%d"))
    except Exception as exc:
        print(f"  WARNING: Renaissance PricedIPOs failed: {exc}")
        return out
    for it in items or []:
        rec = _extract_priced_fields(it)
        if rec and rec["ticker"] and rec["ipo_date"]:
            out.append(rec)
    return out


def discover_priced_ipos(lockup_days: int = DEFAULT_LOCKUP_DAYS,
                         window_days: int = DEFAULT_SCORING_WINDOW_DAYS) -> list[dict]:
    """Merge all sources into a deduped list keyed by ticker.

    Priority for filling fields: reservation > renaissance > pipeline.
    """
    merged: dict[str, dict] = {}
    ordered = (
        _discover_from_reservations()
        + _discover_from_renaissance(lockup_days, window_days)
        + _discover_from_pipeline()
    )
    for rec in ordered:
        t = rec.get("ticker")
        if not t:
            continue
        if t not in merged:
            merged[t] = dict(rec)
        else:
            # Fill only missing fields; keep the higher-priority source.
            for k, v in rec.items():
                if merged[t].get(k) in (None, "") and v not in (None, ""):
                    merged[t][k] = v
    return list(merged.values())


# ── Insider concentration (best-effort EDGAR) ────────────────────────

def _extract_insider_pct_from_edgar(cik: Optional[str]) -> Optional[float]:
    """Best-effort: aggregate insider ownership % from the S-1/424B4.

    Parses the 'beneficial ownership' table for the 'all executive officers
    and directors as a group' percentage. Returns None on any failure — the
    scorer treats None as an unknown (component 0), so this never blocks.
    """
    if not cik:
        return None
    try:
        import requests
    except ImportError:
        return None
    cik10 = str(cik).strip().lstrip("CIK").lstrip(":").zfill(10)
    if not cik10.isdigit():
        return None
    try:
        sub = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik10}.json",
            headers=SEC_HEADERS, timeout=FETCH_TIMEOUT,
        )
        sub.raise_for_status()
        recent = sub.json().get("filings", {}).get("recent", {})
    except Exception:
        return None

    forms = recent.get("form", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    target_idx = None
    for i, form in enumerate(forms):
        if form in ("424B4", "424B1", "S-1", "S-1/A"):
            target_idx = i
            break
    if target_idx is None:
        return None
    try:
        adsh = accns[target_idx].replace("-", "")
        doc = docs[target_idx]
        url = (f"https://www.sec.gov/Archives/edgar/data/"
               f"{int(cik10)}/{adsh}/{doc}")
        resp = requests.get(url, headers=SEC_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)
    except Exception:
        return None

    # Look for "...officers and directors as a group..." then the next %.
    m = re.search(
        r"(?:executive officers and directors|directors and (?:executive )?officers)"
        r"[^%]{0,400}?(\d{1,3}(?:\.\d+)?)\s*%",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    try:
        pct = float(m.group(1))
    except ValueError:
        return None
    return pct if 0.0 < pct <= 100.0 else None


# ── Scoring ──────────────────────────────────────────────────────────

def _score_performance(perf_pct: Optional[float]) -> int:
    if perf_pct is None:
        return 0
    if perf_pct >= 100:
        return 6
    if perf_pct >= 50:
        return 5
    if perf_pct >= 30:
        return 4
    if perf_pct >= 20:
        return 3
    if perf_pct >= 0:
        return 1
    return 0


def _score_insider(insider_pct: Optional[float]) -> int:
    if insider_pct is None:
        return 0
    if insider_pct >= 60:
        return 3
    if insider_pct >= 40:
        return 2
    if insider_pct >= 25:
        return 1
    return 0


def score_short_thesis(candidate: dict, cfg: dict) -> dict:
    """Score the post-lock-up short thesis for one candidate.

    Returns {short_score, current_price, perf_since_ipo_pct, insider_pct,
    score_components}. Degrades gracefully: missing price -> perf component
    0; missing/unparsable insider data -> insider component 0.
    """
    ticker = candidate["ticker"]
    ipo_price = candidate.get("ipo_price")
    current = _current_price(ticker)

    perf_pct = None
    if isinstance(ipo_price, (int, float)) and ipo_price > 0 and current:
        perf_pct = round((current - ipo_price) / ipo_price * 100.0, 2)

    insider_pct = _extract_insider_pct_from_edgar(candidate.get("cik"))

    perf_pts = _score_performance(perf_pct)
    insider_pts = _score_insider(insider_pct)
    timing_pts = 1  # only scored inside the window
    total = perf_pts + insider_pts + timing_pts

    components = {
        "performance": perf_pts,
        "insider": insider_pts,
        "timing": timing_pts,
        "perf_since_ipo_pct": perf_pct,
        "insider_pct": insider_pct,
        "ipo_price": ipo_price,
        "current_price": current,
        "insider_note": "unavailable" if insider_pct is None else "edgar",
        "perf_note": "unavailable" if perf_pct is None else "ok",
    }
    return {
        "short_score": float(total),
        "current_price": current,
        "perf_since_ipo_pct": perf_pct,
        "insider_pct": insider_pct,
        "score_components": components,
    }


# ── Slack ────────────────────────────────────────────────────────────

def _slack(msg: str) -> None:
    try:
        from kairos_alerts import alert_pipeline_event
        alert_pipeline_event(msg, channel=SLACK_CHANNEL)
    except Exception as exc:
        print(f"  WARNING: Slack post failed: {exc}")


def _slack_armed(*, ticker, score, perf_pct, insider_pct, lockup_date, days) -> None:
    perf_s = f"{perf_pct:+.1f}%" if isinstance(perf_pct, (int, float)) else "n/a"
    ins_s = f"{insider_pct:.0f}%" if isinstance(insider_pct, (int, float)) else "n/a"
    _slack(
        f":lock: *HOT-IPO lock-up short ARMED:* {ticker} — short thesis "
        f"{score:.0f}/10\n"
        f"  Lock-up expires {lockup_date} ({days}d)  |  since IPO {perf_s}  "
        f"|  insiders {ins_s}\n"
        f"  Routed to HOT-CATALYST as a long-PUT setup (defined risk, "
        f"known catalyst)."
    )


# ── Cycle ────────────────────────────────────────────────────────────

def run_lockup_cycle(dry_run: bool = True) -> dict:
    """Discover priced IPOs, track lock-ups, score in-window theses, arm.

    Returns {tracked, scored, armed, expired, skipped, actions}.
    In dry-run nothing is flipped to 'armed' (signals are never emitted).
    """
    from kairos_log_db import (
        init_db, upsert_lockup_row, get_lockup_rows, set_lockup_status,
    )

    cfg = load_lockup_config()
    init_db()

    if not cfg["enabled"]:
        print("  Lock-up tracker disabled (hot_ipo.lockup.enabled=false).")
        return {"tracked": 0, "scored": 0, "armed": 0, "expired": 0,
                "skipped": 0, "actions": []}

    lockup_days = cfg["lockup_days"]
    window = cfg["scoring_window_days"]
    threshold = cfg["score_threshold"]
    min_perf = cfg["min_perf_since_ipo_pct"]

    # 1. Discover + upsert tracking rows.
    print(banner("Discovering Priced IPOs"))
    candidates = discover_priced_ipos(lockup_days, window)
    tracked = 0
    for c in candidates:
        lockup_date = compute_lockup_date(c.get("ipo_date"), lockup_days)
        if not lockup_date:
            continue
        upsert_lockup_row(
            ticker=c["ticker"], lockup_expiration_date=lockup_date,
            company_name=c.get("company_name"), cik=c.get("cik"),
            ipo_date=_date_only(c.get("ipo_date")), ipo_price=c.get("ipo_price"),
            source=c.get("source"),
        )
        tracked += 1
    print(f"  {len(candidates)} candidate(s) from sources; {tracked} tracked.")

    # 2. Re-evaluate all non-expired rows.
    rows = [r for r in get_lockup_rows() if r["status"] != "expired"]
    print(banner("Scoring Lock-Up Theses"))
    scored = armed = expired = skipped = 0
    actions: list[dict] = []

    for row in rows:
        ticker = row["ticker"]
        lockup_date = row["lockup_expiration_date"]
        days = _days_until(lockup_date)

        if days is None:
            continue
        if days <= 0:
            set_lockup_status(row["id"], "expired", signal_emitted=0,
                              last_scored_at=_now_iso())
            expired += 1
            actions.append({"ticker": ticker, "status": "expired"})
            print(f"  {ticker:<8} lock-up passed ({lockup_date}) — expired")
            continue
        if days > window:
            # Too far out to score yet; keep tracking quietly.
            continue

        scored += 1
        result = score_short_thesis(row, cfg)
        score = result["short_score"]
        perf = result["perf_since_ipo_pct"]
        insider = result["insider_pct"]

        base_fields = {
            "current_price": result["current_price"],
            "perf_since_ipo_pct": perf,
            "insider_pct": insider,
            "short_score": score,
            "score_components": result["score_components"],
            "last_scored_at": _now_iso(),
        }

        perf_ok = isinstance(perf, (int, float)) and perf >= min_perf
        meets = score >= threshold and perf_ok

        perf_s = f"{perf:+.1f}%" if isinstance(perf, (int, float)) else "n/a"
        ins_s = f"{insider:.0f}%" if isinstance(insider, (int, float)) else "n/a"

        if meets and not dry_run:
            set_lockup_status(row["id"], "armed", signal_emitted=1,
                              armed_at=_now_iso(), simulated=0, **base_fields)
            armed += 1
            actions.append({"ticker": ticker, "status": "armed",
                            "score": score, "perf_pct": perf})
            print(f"  {ticker:<8} score {score:.0f}/10  since-IPO {perf_s}  "
                  f"insiders {ins_s}  ARMED (expires {lockup_date}, {days}d)")
            _slack_armed(ticker=ticker, score=score, perf_pct=perf,
                         insider_pct=insider, lockup_date=lockup_date, days=days)
        elif meets and dry_run:
            # Would arm, but dry-run never flips status -> no signal emitted.
            set_lockup_status(row["id"], "tracking", signal_emitted=0,
                              simulated=1, notes="would_arm", **base_fields)
            skipped += 1
            actions.append({"ticker": ticker, "status": "would_arm",
                            "score": score, "perf_pct": perf})
            print(f"  {ticker:<8} score {score:.0f}/10  since-IPO {perf_s}  "
                  f"insiders {ins_s}  WOULD ARM [dry-run] "
                  f"(expires {lockup_date}, {days}d)")
        else:
            # Below threshold or perf gate failed: keep tracking, disarm if
            # it had previously been armed.
            why = ("perf<%g%%" % min_perf if not perf_ok
                   else "score<%g" % threshold)
            set_lockup_status(row["id"], "tracking", signal_emitted=0,
                              notes=why, **base_fields)
            skipped += 1
            actions.append({"ticker": ticker, "status": "below_threshold",
                            "score": score, "perf_pct": perf, "why": why})
            print(f"  {ticker:<8} score {score:.0f}/10  since-IPO {perf_s}  "
                  f"insiders {ins_s}  no-arm ({why})")

    return {"tracked": tracked, "scored": scored, "armed": armed,
            "expired": expired, "skipped": skipped, "actions": actions}


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kairos HOT-IPO lock-up expiration tracker")
    parser.add_argument("--list", action="store_true",
                        help="List tracked lock-up rows and exit")
    parser.add_argument("--score", metavar="TICKER",
                        help="Score one ticker's short thesis ad hoc and exit")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Allow arming (also requires config dry_run=false)")
    args = parser.parse_args()

    cfg = load_lockup_config()

    if args.list:
        from kairos_log_db import init_db, get_lockup_rows
        init_db()
        rows = get_lockup_rows()
        print(banner("IPO Lock-Up Tracker"))
        if not rows:
            print("  (none)")
            return
        for r in rows:
            sc = f"{r['short_score']:.0f}" if r["short_score"] is not None else "—"
            print(f"  {r['ticker']:<8} {r['status']:<9} expires "
                  f"{r['lockup_expiration_date']}  score {sc}  "
                  f"src={r['source']}  emitted={r['signal_emitted']}")
        return

    if args.score:
        from kairos_log_db import init_db
        init_db()
        ticker = args.score.strip().upper()
        cand = next((c for c in discover_priced_ipos(cfg["lockup_days"],
                                                      cfg["scoring_window_days"])
                     if c["ticker"] == ticker), None)
        if not cand:
            cand = {"ticker": ticker, "ipo_price": None, "cik": None}
            print(f"  {ticker} not in discovered sources — scoring with "
                  f"no IPO price (perf will be n/a).")
        res = score_short_thesis(cand, cfg)
        print(json.dumps(res, indent=2, default=str))
        return

    allow_real = (not cfg["dry_run"]) and args.no_dry_run
    dry_run = not allow_real

    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS HOT-IPO LOCK-UP TRACKER — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")
    mode = "LIVE ARM" if allow_real else "DRY-RUN (score + report)"
    print(f"  Mode: {mode}   (config.dry_run={cfg['dry_run']}, "
          f"--no-dry-run={args.no_dry_run})")

    summary = run_lockup_cycle(dry_run=dry_run)

    print(banner(f"LOCK-UP TRACKER COMPLETE  ({mode})"))
    print(f"  Tracked: {summary['tracked']}  Scored: {summary['scored']}  "
          f"Armed: {summary['armed']}  Expired: {summary['expired']}  "
          f"Skipped: {summary['skipped']}")
    print("━" * W)


if __name__ == "__main__":
    main()
