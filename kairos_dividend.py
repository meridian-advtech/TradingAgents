"""
Kairos HOT-DIVIDEND — Dividend Core Maintenance + Event Signals

Two completely separate mechanisms (per the architecture doc):

  1. DIVIDEND CORE (background, quiet)
       - DRIP reconciliation: keep Kairos share counts / cost basis in
         sync with IBKR's automatic dividend reinvestment.
       - Protected position review: alert (no auto-action) when a
         protected holding falls >20% from cost basis.
       - Config→DB flag sync: mirror the config-authoritative protected /
         drip_enabled lists onto the holdings table.

  2. HOT-DIVIDEND SIGNALS (event-driven, layered on top)
       - Sub-1  Dividend increase >=5%  → bullish, registers HOT-DIVIDEND
       - Sub-2  Dividend cut >=20%/susp  → bearish HOT-CATALYST put setup
                (guard: skip names already down >=30%)
       - Sub-3  Ex-div drop >1.5x amount → HOT-REVERSION tactical long

Signals feed confluence scoring by registering tags into
kairos_signal_summary.json, exactly like HOT-OPTIONS / HOT-CATALYST.

Public API:
    run_dividend_cycle(ib=None, dry_run=False)   Orchestrator (daily)
    is_dividend_due(now_et=None)                  Once-per-ET-day gate
    reconcile_drip(ib)                            DRIP component
    review_protected_positions(ib)               Protected-review component
    detect_dividend_signals()                     HOT-DIVIDEND component
    sync_flags_from_config()                      Config→DB flag mirror

Everything degrades gracefully: missing config/token/network returns
empty results, and nothing here raises to its caller.
"""

import json
import os
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
STATE_FILE = os.path.join(SCRIPT_DIR, "kairos_dividend_state.json")
SUMMARY_FILE = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")

FETCH_TIMEOUT = 20

# Tag names registered into signal_tags for confluence awareness.
TAG_INCREASE = "HOT-DIVIDEND"
TAG_CUT = "HOT-CATALYST"
TAG_REVERSION = "HOT-REVERSION"

# Defaults — overridden by config["dividend"]["thresholds"].
DEFAULT_THRESHOLDS = {
    "increase_pct": 5.0,
    "cut_pct": 20.0,
    "exdiv_drop_multiple": 1.5,
    "cut_already_down_guard_pct": 30.0,
    "protected_loss_alert_pct": 20.0,
    "signal_fresh_days": 14,
}

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = None


# ── Config / state I/O ───────────────────────────────────────────────

def _load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _dividend_config() -> dict:
    return _load_config().get("dividend", {}) or {}


def _thresholds() -> dict:
    out = dict(DEFAULT_THRESHOLDS)
    out.update(_dividend_config().get("thresholds", {}) or {})
    return out


def _protected_tickers() -> list[str]:
    return [str(t).upper() for t in _dividend_config().get("protected_tickers", []) if t]


def _drip_tickers() -> list[str]:
    return [str(t).upper() for t in _dividend_config().get("drip_enabled_tickers", []) if t]


def _signal_scan_tickers() -> list[str]:
    """Tickers scanned for HOT-DIVIDEND signals.

    Config `dividend.signal_scan_tickers` if present, else the union of the
    protected + drip core (so the module is useful out of the box).
    """
    explicit = _dividend_config().get("signal_scan_tickers")
    if isinstance(explicit, list) and explicit:
        return [str(t).upper() for t in explicit if t]
    return sorted(set(_protected_tickers()) | set(_drip_tickers()))


def _load_finnhub_token():
    try:
        token = _load_config().get("finnhub", {}).get("api_key", "")
        if token:
            return token.strip()
    except Exception:
        pass
    return (os.environ.get("FINNHUB_API_KEY") or "").strip() or None


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: dividend state write failed: {exc}")


def _today_et() -> str:
    if _ET is not None:
        return datetime.now(_ET).strftime("%Y-%m-%d")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Daily gate ───────────────────────────────────────────────────────

def is_dividend_due(now_et=None) -> bool:
    """True at most once per ET calendar day (mirrors the IPO tracker gate)."""
    today = (now_et.strftime("%Y-%m-%d") if now_et is not None else _today_et())
    return (_load_state().get("last_run") or "").strip() != today


# ── Price / dividend data (yfinance + Finnhub) ───────────────────────

def _recent_close(ticker: str):
    """Most recent daily close via yfinance, or None."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        closes = df["Close"].dropna()
        return float(closes.iloc[-1]) if len(closes) else None
    except Exception:
        return None


def _trailing_return_pct(ticker: str, days: int = 90):
    """Trailing %% price change over ~`days` calendar days, or None."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        period = "6mo" if days <= 130 else "1y"
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        last = float(closes.iloc[-1])
        # nearest bar ~`days` ago
        idx = max(0, len(closes) - 1 - days)
        ref = float(closes.iloc[idx])
        if ref <= 0:
            return None
        return (last - ref) / ref * 100.0
    except Exception:
        return None


def _close_on_or_before(ticker: str, date_str: str):
    """Close on `date_str`, else the most recent close before it. (price, date)."""
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        start = (target - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (target + timedelta(days=2)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        closes = df["Close"].dropna()
        if not len(closes):
            return None, None
        chosen = None
        chosen_date = None
        for ts, val in closes.items():
            d = ts.date()
            if d <= target:
                chosen = float(val)
                chosen_date = d.strftime("%Y-%m-%d")
        if chosen is None:
            chosen = float(closes.iloc[0])
            chosen_date = closes.index[0].date().strftime("%Y-%m-%d")
        return chosen, chosen_date
    except Exception:
        return None, None


def _fetch_dividend_history(ticker: str, token: str) -> list[dict]:
    """Finnhub stock/dividend over the trailing ~14 months, sorted by ex-date.

    Each item: {date (ex-date), amount, payDate, declarationDate, ...}.
    Returns [] on any failure.
    """
    if not token:
        return []
    try:
        import requests
    except ImportError:
        return []
    today_dt = datetime.now(timezone.utc)
    start = (today_dt - timedelta(days=430)).strftime("%Y-%m-%d")
    end = today_dt.strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/dividend",
            params={"symbol": ticker, "from": start, "to": end,
                    "token": token},
            timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  WARNING: Finnhub dividend fetch failed for {ticker}: {exc}")
        return []
    if not isinstance(data, list):
        return []
    rows = [d for d in data if isinstance(d, dict) and d.get("date")]
    rows.sort(key=lambda d: d.get("date", ""))
    return rows


# ── IBKR positions ───────────────────────────────────────────────────

def _ibkr_positions(ib) -> dict:
    """{SYM: {"qty": float, "avg_cost": float}} for stock positions, or {}."""
    if ib is None:
        return {}
    out: dict[str, dict] = {}
    try:
        for p in ib.positions():
            try:
                if getattr(p.contract, "secType", "STK") != "STK":
                    continue
                sym = p.contract.symbol.upper()
                out[sym] = {"qty": float(p.position),
                            "avg_cost": round(float(p.avgCost), 2)}
            except Exception:
                continue
    except Exception as exc:
        print(f"  WARNING: IBKR positions fetch failed: {exc}")
    return out


# ── Signal-tag registration (feeds confluence scoring) ───────────────

def _register_signal_tags(tags_by_ticker: dict) -> None:
    """Merge {ticker: tag} into kairos_signal_summary.json signal_tags."""
    if not tags_by_ticker or not os.path.exists(SUMMARY_FILE):
        return
    try:
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)
        for ticker, tag in tags_by_ticker.items():
            bucket = summary.setdefault("signal_tags", {}).setdefault(ticker, [])
            if tag not in bucket:
                bucket.append(tag)
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"  WARNING: signal_tags merge failed: {exc}")


def _get_entry_rationale(ticker: str) -> str:
    """Most recent BUY rationale for a ticker from the decisions log."""
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        row = conn.execute(
            "SELECT rationale FROM decisions WHERE ticker = ? AND action = 'BUY' "
            "AND rationale IS NOT NULL ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        conn.close()
        return (row["rationale"] if row else "") or ""
    except Exception:
        return ""


def _active_signal_tags(ticker: str) -> list[str]:
    try:
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)
        return summary.get("signal_tags", {}).get(ticker, []) or []
    except Exception:
        return []


# ── Component: config→DB flag sync ───────────────────────────────────

def sync_flags_from_config() -> dict:
    """Mirror the config-authoritative protected / drip lists onto holdings.

    Config is the source of truth (Decision 2). Open lots of listed tickers
    get the flag set; lots of unlisted tickers get it cleared, so removing a
    ticker from config un-protects it on the next run.
    """
    summary = {"protected_set": 0, "drip_set": 0}
    try:
        from kairos_log_db import set_holding_flags, get_connection
        protected = set(_protected_tickers())
        drip = set(_drip_tickers())

        conn = get_connection()
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM holdings WHERE sold_date IS NULL"
        ).fetchall()
        conn.close()
        tickers = {r["ticker"].upper() for r in rows} | protected | drip

        for t in tickers:
            n = set_holding_flags(
                t,
                drip_enabled=(t in drip),
                protected=(t in protected),
            )
            if t in protected and n:
                summary["protected_set"] += 1
            if t in drip and n:
                summary["drip_set"] += 1
    except Exception as exc:
        print(f"  WARNING: flag sync failed: {exc}")
    return summary


# ── Component 1: DRIP reconciliation ─────────────────────────────────

def reconcile_drip(ib, dry_run: bool = False) -> list[dict]:
    """Reconcile IBKR share counts vs Kairos for drip_enabled tickers.

    A positive delta with no corresponding Kairos trade is treated as a DRIP
    reinvestment: logged as a DRIP decision and added as a new cost-basis lot
    at the dividend payment-date close (Decision 3). Returns list of deltas.
    """
    applied: list[dict] = []
    try:
        from kairos_log_db import get_open_holdings, insert_drip
    except Exception as exc:
        print(f"  WARNING: DRIP reconcile import failed: {exc}")
        return applied

    drip = _drip_tickers()
    if not drip:
        return applied

    ibkr = _ibkr_positions(ib)
    if not ibkr:
        print("  DRIP reconcile: no IBKR positions available — skipping.")
        return applied

    token = _load_finnhub_token()
    today = _today_et()

    for ticker in drip:
        ibkr_qty = ibkr.get(ticker, {}).get("qty")
        if ibkr_qty is None:
            continue
        kairos_qty = sum(float(l["quantity"]) for l in get_open_holdings(ticker))
        delta = round(ibkr_qty - kairos_qty, 6)
        if delta <= 0.0001:
            continue  # only reinvestment-style positive drift is a DRIP

        # Price the new lot at the most recent dividend payment-date close.
        price = None
        pay_date = today
        hist = _fetch_dividend_history(ticker, token)
        if hist:
            latest = hist[-1]
            pay_date = (latest.get("payDate") or latest.get("date") or today)
            price, _ = _close_on_or_before(ticker, pay_date)
        if price is None:
            price = _recent_close(ticker)
        if price is None:
            print(f"  DRIP {ticker}: +{delta} sh but no price — deferring.")
            continue

        print(f"  DRIP {ticker}: IBKR {ibkr_qty} vs Kairos {kairos_qty} "
              f"→ +{delta} sh @ ${price:.2f} (pay {pay_date})")
        if not dry_run:
            insert_drip(ticker, delta, price, pay_date)
        applied.append({"ticker": ticker, "shares": delta,
                        "price": price, "pay_date": pay_date,
                        "dry_run": dry_run})
    return applied


# ── Component 2: protected position review ───────────────────────────

def review_protected_positions(ib, dry_run: bool = False) -> list[dict]:
    """Alert (no auto-action) when a protected position is down > threshold.

    Max one alert per ticker per week. Posts to #kairos-alerts with P&L,
    entry rationale, and active signals. Returns list of alerts fired.
    """
    fired: list[dict] = []
    protected = _protected_tickers()
    if not protected:
        return fired

    ibkr = _ibkr_positions(ib)
    if not ibkr:
        print("  Protected review: no IBKR positions available — skipping.")
        return fired

    thr = _thresholds()["protected_loss_alert_pct"]
    state = _load_state()
    sent = state.get("protected_alerts_sent", {})
    if not isinstance(sent, dict):
        sent = {}
    today = _today_et()
    today_d = datetime.strptime(today, "%Y-%m-%d").date()

    for ticker in protected:
        pos = ibkr.get(ticker)
        if not pos or pos["qty"] <= 0:
            continue
        avg = pos["avg_cost"]
        cur = _recent_close(ticker)
        if avg is None or avg <= 0 or cur is None:
            continue
        loss_pct = (cur - avg) / avg * 100.0
        if loss_pct > -abs(thr):
            continue  # not down enough

        # Weekly dedup
        last = sent.get(ticker)
        if last:
            try:
                if (today_d - datetime.strptime(last, "%Y-%m-%d").date()).days < 7:
                    continue
            except ValueError:
                pass

        qty = pos["qty"]
        unreal = (cur - avg) * qty
        rationale = _get_entry_rationale(ticker) or "(no recorded entry rationale)"
        active = _active_signal_tags(ticker)
        msg = (
            f":warning: *Protected Position Review: {ticker}* is down "
            f"{loss_pct:.1f}% from cost basis\n"
            f"Entry: ${avg:.2f} | Current: ${cur:.2f} | "
            f"Unrealized: ${unreal:,.2f} ({loss_pct:.1f}%)\n"
            f"Active signals: {', '.join(active) if active else 'none'}\n"
            f"Original entry rationale: {rationale}\n"
            f"_No action taken — human review required._"
        )
        print(f"  PROTECTED REVIEW {ticker}: {loss_pct:.1f}% — alerting.")
        if not dry_run:
            try:
                from kairos_alerts import post_message
                post_message("alerts", msg)
            except Exception as exc:
                print(f"  WARNING: protected alert post failed: {exc}")
            sent[ticker] = today
        fired.append({"ticker": ticker, "loss_pct": round(loss_pct, 2),
                      "unrealized": round(unreal, 2), "dry_run": dry_run})

    if not dry_run:
        state["protected_alerts_sent"] = sent
        _save_state(state)
    return fired


# ── Component 3: HOT-DIVIDEND signal detection ───────────────────────

def detect_dividend_signals(dry_run: bool = False) -> list[dict]:
    """Scan dividend tickers for the three HOT-DIVIDEND sub-signals.

    Registers confluence tags into kairos_signal_summary.json and dedups so
    each (ticker, signal, dividend-date) fires at most once. Returns signals.
    """
    signals: list[dict] = []
    tickers = _signal_scan_tickers()
    if not tickers:
        return signals

    token = _load_finnhub_token()
    if not token:
        print("  HOT-DIVIDEND: no Finnhub token — skipping detection.")
        return signals

    thr = _thresholds()
    fresh_days = int(thr.get("signal_fresh_days", 14))
    today = _today_et()
    today_d = datetime.strptime(today, "%Y-%m-%d").date()

    state = _load_state()
    seen = state.get("dividend_signals_sent", {})
    if not isinstance(seen, dict):
        seen = {}

    tags_to_register: dict[str, str] = {}

    for ticker in tickers:
        hist = _fetch_dividend_history(ticker, token)
        if len(hist) < 1:
            continue
        latest = hist[-1]
        amount = _to_float(latest.get("amount"))
        ex_date = (latest.get("date") or "").strip()
        if not ex_date:
            continue
        try:
            ex_d = datetime.strptime(ex_date, "%Y-%m-%d").date()
        except ValueError:
            continue

        # Only act on a dividend event that is recent.
        decl = (latest.get("declarationDate") or ex_date).strip()
        try:
            decl_d = datetime.strptime(decl, "%Y-%m-%d").date()
        except ValueError:
            decl_d = ex_d
        recent_announce = (today_d - decl_d).days <= fresh_days

        prev_amount = _to_float(hist[-2].get("amount")) if len(hist) >= 2 else None

        # ── Sub-1 / Sub-2: increase or cut vs prior dividend ─────────
        if recent_announce and prev_amount and prev_amount > 0 and amount is not None:
            change_pct = (amount - prev_amount) / prev_amount * 100.0
            key = f"{ticker}:DIVCHANGE:{ex_date}"
            if change_pct >= thr["increase_pct"] and seen.get(key) != "1":
                signals.append({
                    "ticker": ticker, "signal": TAG_INCREASE, "kind": "increase",
                    "change_pct": round(change_pct, 1), "amount": amount,
                    "prev_amount": prev_amount, "ex_date": ex_date,
                    "hold_days": 14, "direction": "bullish",
                })
                tags_to_register[ticker] = TAG_INCREASE
                seen[key] = "1"
                print(f"  HOT-DIVIDEND {ticker}: increase +{change_pct:.1f}% "
                      f"(${prev_amount}→${amount}) — bullish")
            elif change_pct <= -thr["cut_pct"] and seen.get(key) != "1":
                trail = _trailing_return_pct(ticker, days=90)
                guard = thr["cut_already_down_guard_pct"]
                if trail is not None and trail <= -abs(guard):
                    print(f"  HOT-DIVIDEND {ticker}: cut {change_pct:.1f}% but "
                          f"already down {trail:.1f}% — guard skip.")
                    seen[key] = "1"
                else:
                    signals.append({
                        "ticker": ticker, "signal": TAG_CUT, "kind": "cut",
                        "change_pct": round(change_pct, 1), "amount": amount,
                        "prev_amount": prev_amount, "ex_date": ex_date,
                        "hold_days": 10, "direction": "bearish",
                        "put_setup": True,
                    })
                    tags_to_register[ticker] = TAG_CUT
                    seen[key] = "1"
                    print(f"  HOT-DIVIDEND {ticker}: cut {change_pct:.1f}% "
                          f"(${prev_amount}→${amount}) — bearish put setup")

        # ── Sub-3: ex-dividend reversion ─────────────────────────────
        days_since_ex = (today_d - ex_d).days
        if amount and 0 <= days_since_ex <= 2:
            key = f"{ticker}:EXDIV:{ex_date}"
            if seen.get(key) != "1":
                prior_close, _ = _close_on_or_before(
                    ticker, (ex_d - timedelta(days=1)).strftime("%Y-%m-%d"))
                cur_close = _recent_close(ticker)
                if prior_close and cur_close:
                    drop = prior_close - cur_close
                    if drop > thr["exdiv_drop_multiple"] * amount:
                        signals.append({
                            "ticker": ticker, "signal": TAG_REVERSION,
                            "kind": "exdiv_reversion",
                            "drop": round(drop, 2), "amount": amount,
                            "ex_date": ex_date, "hold_days": 5,
                            "direction": "bullish",
                            "stop_below": round(2 * amount, 2),
                        })
                        tags_to_register[ticker] = TAG_REVERSION
                        seen[key] = "1"
                        print(f"  HOT-DIVIDEND {ticker}: ex-div drop ${drop:.2f} "
                              f"> 1.5x div ${amount} — reversion long")

    if not dry_run and tags_to_register:
        _register_signal_tags(tags_to_register)
    if not dry_run:
        state["dividend_signals_sent"] = seen
        _save_state(state)
    return signals


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Orchestrator ─────────────────────────────────────────────────────

def run_dividend_cycle(ib=None, dry_run: bool = False) -> dict:
    """Daily dividend maintenance + signal detection. Never raises."""
    summary = {"flags": {}, "drip": [], "protected": [], "signals": []}
    try:
        summary["flags"] = sync_flags_from_config()
    except Exception as exc:
        print(f"  WARNING: dividend flag sync failed: {exc}")
    try:
        summary["drip"] = reconcile_drip(ib, dry_run=dry_run)
    except Exception as exc:
        print(f"  WARNING: DRIP reconciliation failed: {exc}")
    try:
        summary["protected"] = review_protected_positions(ib, dry_run=dry_run)
    except Exception as exc:
        print(f"  WARNING: protected review failed: {exc}")
    try:
        summary["signals"] = detect_dividend_signals(dry_run=dry_run)
    except Exception as exc:
        print(f"  WARNING: dividend signal detection failed: {exc}")

    if not dry_run:
        state = _load_state()
        state["last_run"] = _today_et()
        _save_state(state)

    print(f"  Dividend cycle: drip={len(summary['drip'])} "
          f"protected_alerts={len(summary['protected'])} "
          f"signals={len(summary['signals'])}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kairos HOT-DIVIDEND module")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect/report without writing or posting")
    args = parser.parse_args()
    result = run_dividend_cycle(ib=None, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
