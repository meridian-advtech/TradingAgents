"""
Kairos Event-Driven Seasonality Signals

Tracks a small calendar of recurring product events (AAPL WWDC, NVDA GTC,
Amazon Prime Day, etc.) and emits a tradable signal when today falls
inside a tracked window. Two historical patterns are scored:

  dip_then_rip — common for the big-tech keynotes (AAPL, NVDA, MSFT,
                 GOOGL, META). Pre-announcement profit-taking creates a
                 ~3-5% dip from the 30-day high; mean-reversion lifts
                 it back through the event.
  buildup      — retail / commerce events (AMZN Prime Day, WMT/TGT
                 holiday season). Steady accumulation into the catalyst,
                 no dip required.

Calendar entries live in kairos_config.json under "event_calendar". A
"day_estimate" is fine — the window_days_before / window_days_after
fields give us slack on the exact date.

Public surface:
    load_event_calendar() -> list[dict]
    get_active_events(today=None) -> list[dict]
    get_event_tickers() -> set[str]
    score_event_signal(ticker, current_price, high_30d) -> dict
    format_event_prompt_block(active_events) -> str
    run_event_scan() -> dict     # daily entry: Slack alerts on PRE-EVENT-DIP

State persistence (kairos_ipo_cache.json — shared file, distinct keys):
    "event_forced_screens": {"AAPL:WWDC:2026-06-09": 2}
    "event_alerts_sent":    {"AAPL:WWDC:2026-06-09": "2026-05-29T..."}
"""

import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

SCRIPT_DIR = "/Users/jelmore/Kairos"
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
EVENT_CACHE_FILE = os.path.join(SCRIPT_DIR, "kairos_ipo_cache.json")

EVENT_FORCED_MAX_CYCLES = 3
SLACK_CHANNEL = "reports"


# ─────────────────────────────────────────────────────────────────────
# Calendar
# ─────────────────────────────────────────────────────────────────────

def load_event_calendar() -> list[dict]:
    """Read event_calendar from kairos_config.json. Returns [] on failure."""
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except (IOError, json.JSONDecodeError):
        return []
    calendar = cfg.get("event_calendar", [])
    if not isinstance(calendar, list):
        return []
    out: list[dict] = []
    for e in calendar:
        if not isinstance(e, dict):
            continue
        if not (e.get("ticker") and e.get("event")
                and e.get("month") and e.get("day_estimate")):
            continue
        out.append(e)
    return out


def _estimate_event_date(month: int, day_estimate: int, year: int) -> Optional[date]:
    try:
        return date(int(year), int(month), int(day_estimate))
    except (TypeError, ValueError):
        return None


def get_active_events(today: Optional[date] = None) -> list[dict]:
    """Return events whose `today` falls inside [event-before, event+after].

    Each event tries the current calendar year first, then next year (so
    a December-1 event with a 30-day before-window is found from Nov 1).
    Past events whose post-window has closed are skipped.
    """
    if today is None:
        today = datetime.now(timezone.utc).date()
    out: list[dict] = []
    for event in load_event_calendar():
        before = int(event.get("window_days_before", 0) or 0)
        after  = int(event.get("window_days_after", 0) or 0)
        for year in (today.year, today.year + 1):
            event_date = _estimate_event_date(
                event["month"], event["day_estimate"], year
            )
            if event_date is None:
                continue
            window_start = event_date - timedelta(days=before)
            window_end   = event_date + timedelta(days=after)
            if window_start <= today <= window_end:
                phase = "pre" if today < event_date else "post"
                days_until = max(0, (event_date - today).days)
                days_since = max(0, (today - event_date).days)
                out.append({
                    "ticker":             event["ticker"],
                    "event":              event["event"],
                    "event_date":         event_date.isoformat(),
                    "days_until_event":   days_until,
                    "days_since_event":   days_since,
                    "phase":              phase,
                    "pattern":            event.get("pattern", "dip_then_rip"),
                    "window_days_before": before,
                })
                break  # only one active occurrence per calendar entry
    return out


def get_event_tickers() -> set[str]:
    """Tickers that currently have at least one active event window."""
    return {e["ticker"].upper() for e in get_active_events()}


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def score_event_signal(ticker: str, current_price: float, high_30d: float) -> dict:
    """Score `ticker` against its active event windows.

    Returns:
        {
          "signal": "PRE-EVENT-DIP" | "PRE-EVENT-BUILDUP"
                    | "POST-EVENT-REBOUND" | None,
          "conviction_boost":  float,
          "event":             str | None,
          "event_date":        str | None,
          "days_until_event":  int  | None,
          "dip_from_high_pct": float,
          "pattern":           str | None,
          "event_context":     dict | None,    # echo of active event row
        }

    Neutral payload (signal=None, boost=1.0) when the ticker has no
    active event so callers can apply the result unconditionally.
    """
    ticker_u = (ticker or "").strip().upper()
    if not ticker_u:
        return _neutral()

    matches = [e for e in get_active_events()
               if e["ticker"].upper() == ticker_u]
    if not matches:
        return _neutral()
    event = matches[0]   # pick the first active window per ticker

    pattern    = event.get("pattern", "dip_then_rip")
    phase      = event.get("phase", "pre")
    days_until = event.get("days_until_event", 0)
    win_before = event.get("window_days_before", 0)

    try:
        dip_pct = ((float(high_30d) - float(current_price))
                   / float(high_30d) * 100.0) if high_30d else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        dip_pct = 0.0

    signal: Optional[str] = None
    boost = 1.0

    if phase == "pre":
        if pattern == "dip_then_rip":
            if dip_pct >= 3.0 and days_until <= win_before:
                signal = "PRE-EVENT-DIP"
                boost = 1.5 if dip_pct >= 5.0 else 1.3
                if days_until <= 5:
                    boost = round(boost + 0.2, 2)
        elif pattern == "buildup":
            signal = "PRE-EVENT-BUILDUP"
            boost = 1.2
    elif phase == "post":
        signal = "POST-EVENT-REBOUND"
        boost = 1.1

    if signal is None:
        return {
            **_neutral(),
            "event": event["event"],
            "event_date": event["event_date"],
            "days_until_event": days_until,
            "dip_from_high_pct": round(dip_pct, 2),
            "pattern": pattern,
            "event_context": event,
        }

    return {
        "signal":            signal,
        "conviction_boost":  round(boost, 2),
        "event":             event["event"],
        "event_date":        event["event_date"],
        "days_until_event":  days_until,
        "dip_from_high_pct": round(dip_pct, 2),
        "pattern":           pattern,
        "event_context":     event,
    }


def _neutral() -> dict:
    return {
        "signal":            None,
        "conviction_boost":  1.0,
        "event":             None,
        "event_date":        None,
        "days_until_event":  None,
        "dip_from_high_pct": 0.0,
        "pattern":           None,
        "event_context":     None,
    }


# ─────────────────────────────────────────────────────────────────────
# Prompt block builder
# ─────────────────────────────────────────────────────────────────────

def format_event_prompt_block(active_events: list[dict]) -> str:
    """Build the event-context block to prepend to the reasoning prompt.

    Empty string when nothing's active. Pulls live prices via yfinance
    so the prompt carries the actual dip / boost numbers — falls back
    to phase-only language if yfinance is unavailable.
    """
    if not active_events:
        return ""

    lines: list[str] = []
    for event in active_events:
        ticker = event["ticker"].upper()
        cur, hi = _fetch_price_and_high(ticker)
        scored = None
        if cur is not None and hi is not None:
            scored = score_event_signal(ticker, cur, hi)

        if event["phase"] == "pre":
            time_str = (f"{event['days_until_event']} days from "
                        f"{event['event']} ({event['event_date']})")
        else:
            time_str = (f"{event['days_since_event']} days after "
                        f"{event['event']} ({event['event_date']})")

        line = f"EVENT CALENDAR ALERT: {ticker} is {time_str}"

        if scored and scored.get("signal"):
            boost = scored["conviction_boost"]
            dip = scored["dip_from_high_pct"]
            if scored["signal"] == "PRE-EVENT-DIP":
                line += (f" and currently down {dip:.1f}% from its "
                         f"30-day high — matches historical pre-announcement "
                         f"dip pattern. Consider pre-event positioning. "
                         f"Apply ~{boost:.1f}x conviction boost "
                         f"(encode by raising the conviction value in your JSON).")
            elif scored["signal"] == "PRE-EVENT-BUILDUP":
                line += (f" — buildup pattern active. "
                         f"Apply ~{boost:.1f}x conviction boost "
                         f"(encode by raising the conviction value in your JSON).")
            elif scored["signal"] == "POST-EVENT-REBOUND":
                line += (f" — post-event rebound window. "
                         f"Apply ~{boost:.1f}x conviction boost "
                         f"(encode by raising the conviction value in your JSON).")
        else:
            line += "."
        lines.append(line)

    header = (
        "================================================================\n"
        "EVENT-DRIVEN SEASONALITY SIGNAL\n"
        "================================================================\n"
        "The following tickers fall inside a tracked product-event window. "
        "Where a signal is named (PRE-EVENT-DIP / PRE-EVENT-BUILDUP / "
        "POST-EVENT-REBOUND), raise the conviction value on any BUY for "
        "that ticker by the indicated multiplier.\n"
    )
    return header + "\n".join(lines) + "\n\n"


# ─────────────────────────────────────────────────────────────────────
# Price helpers (yfinance, graceful)
# ─────────────────────────────────────────────────────────────────────

def _fetch_price_and_high(ticker: str) -> tuple[Optional[float], Optional[float]]:
    """yfinance lookup → (current_close, 30d_high). (None, None) on failure."""
    if not ticker:
        return None, None
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    try:
        hist = yf.Ticker(ticker).history(period="1mo", auto_adjust=False)
    except Exception:
        return None, None
    if hist is None or len(hist) == 0:
        return None, None
    try:
        current = float(hist["Close"].iloc[-1])
        high_30d = float(hist["High"].max())
        if current <= 0 or high_30d <= 0:
            return None, None
        return current, high_30d
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────
# Shared cache I/O (kairos_ipo_cache.json — reused with distinct keys)
# ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not os.path.exists(EVENT_CACHE_FILE):
        return {}
    try:
        with open(EVENT_CACHE_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(EVENT_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: event cache write failed: {exc}")


def event_key(active_event: dict) -> str:
    """Stable key for an active event: 'TICKER:Event:YYYY-MM-DD'."""
    return f"{active_event['ticker'].upper()}:{active_event['event']}:{active_event['event_date']}"


def get_event_force_count(key: str) -> int:
    cache = _load_cache()
    bucket = cache.get("event_forced_screens") or {}
    try:
        return int(bucket.get(key, 0))
    except (TypeError, ValueError):
        return 0


def bump_event_force_count(key: str) -> int:
    cache = _load_cache()
    bucket = cache.setdefault("event_forced_screens", {})
    if not isinstance(bucket, dict):
        bucket = {}
        cache["event_forced_screens"] = bucket
    new_val = int(bucket.get(key, 0) or 0) + 1
    bucket[key] = new_val
    _save_cache(cache)
    return new_val


# ─────────────────────────────────────────────────────────────────────
# Daily scan
# ─────────────────────────────────────────────────────────────────────

def run_event_scan() -> dict:
    """Daily event scan: fetch prices, score, Slack on fresh PRE-EVENT-DIP."""
    summary = {
        "active_events":  0,
        "signals_fired":  0,
        "new_alerts":     0,
        "skipped_no_price": 0,
        "errors":         [],
    }
    active = get_active_events()
    summary["active_events"] = len(active)
    if not active:
        return summary

    cache = _load_cache()
    alerts = cache.setdefault("event_alerts_sent", {})
    if not isinstance(alerts, dict):
        alerts = {}
        cache["event_alerts_sent"] = alerts
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for event in active:
        ticker = event["ticker"].upper()
        try:
            cur, hi = _fetch_price_and_high(ticker)
            if cur is None or hi is None:
                summary["skipped_no_price"] += 1
                continue
            sig = score_event_signal(ticker, cur, hi)
            if not sig.get("signal"):
                continue
            summary["signals_fired"] += 1

            if sig["signal"] != "PRE-EVENT-DIP":
                continue   # alerts are reserved for the strongest signal

            key = event_key(event)
            prior = alerts.get(key, "")
            if prior.startswith(today_iso):
                continue   # already alerted today

            days = sig["days_until_event"]
            dip  = sig["dip_from_high_pct"]
            try:
                from kairos_alerts import alert_pipeline_event
                msg = (
                    f":calendar: *EVENT SIGNAL:* `${ticker}` — "
                    f"{event['event']} in {days} days, "
                    f"down {dip:.1f}% from 30d high. "
                    f"PRE-EVENT-DIP signal active "
                    f"(boost ~{sig['conviction_boost']:.1f}x)."
                )
                alert_pipeline_event(msg, channel=SLACK_CHANNEL)
            except Exception as exc:
                summary["errors"].append(f"slack({ticker}): {exc}")

            alerts[key] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            summary["new_alerts"] += 1
        except Exception as exc:
            summary["errors"].append(f"{ticker}: {exc}")

    _save_cache(cache)
    print(
        f"  Event scan: active={summary['active_events']} "
        f"signals={summary['signals_fired']} "
        f"new_alerts={summary['new_alerts']} "
        f"no_price={summary['skipped_no_price']}"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _print_json(label: str, payload) -> None:
    print(f"\n── {label} ──")
    if payload in (None, [], {}, set()):
        print("  (no data)")
        return
    if isinstance(payload, set):
        payload = sorted(payload)
    print(json.dumps(payload, indent=2, default=str))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kairos event seasonality signals")
    parser.add_argument("--calendar", action="store_true",
                        help="Print the configured event calendar")
    parser.add_argument("--active", action="store_true",
                        help="Show currently-active events")
    parser.add_argument("--tickers", action="store_true",
                        help="Show tickers with active events")
    parser.add_argument("--score", nargs=3,
                        metavar=("TICKER", "PRICE", "HIGH_30D"),
                        help="Score TICKER given current price + 30d high")
    parser.add_argument("--run", action="store_true",
                        help="Run the daily scan + post Slack alerts")
    args = parser.parse_args()

    did_any = False
    if args.calendar:
        _print_json("EventCalendar", load_event_calendar())
        did_any = True
    if args.active:
        _print_json("ActiveEvents", get_active_events())
        did_any = True
    if args.tickers:
        _print_json("ActiveEventTickers", get_event_tickers())
        did_any = True
    if args.score:
        t, p, h = args.score
        _print_json(f"ScoreEventSignal {t}",
                    score_event_signal(t, float(p), float(h)))
        did_any = True
    if args.run:
        _print_json("RunEventScan", run_event_scan())
        did_any = True
    if not did_any:
        parser.print_help()


if __name__ == "__main__":
    main()
