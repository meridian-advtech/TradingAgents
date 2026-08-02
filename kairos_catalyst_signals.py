"""
Kairos HOT-CATALYST Signal Detector — long options only

Detects three catalyst setups and emits directional long-option signals
(buy calls / buy puts). It NEVER proposes selling premium. Detection runs
on price + IV data (yfinance) so it is fully usable while OPRA / IBKR
Level 2 options permissions are still pending — the executor
(kairos_options_execute.py) is the only piece that touches the live chain.

Three setups (config under kairos_config.json["hot_catalyst"]):

  SETUP1  Pre-Catalyst Underpriced Volatility
          A ticker sits 7-21 days ahead of a tracked product event
          (kairos_config.json["event_calendar"]) AND its IV rank is at or
          below setup1.iv_rank_max (vol is cheap before the catalyst).
          Direction follows the event's historical pattern — the calendar
          events are bullish catalysts (dip_then_rip / buildup) so the
          play is long CALLS into the move.

  SETUP2  Post-Crush Reversion (thesis-driven direction)
          Underlying has dropped >= setup2.drop_pct_min over the last
          setup2.drop_lookback_days. Direction is decided by price action:
            intact  — reclaiming the pre-crush level / holding above the
                      gap low (no lower low) -> long CALLS (bounce)
            broken  — failure to hold / making lower lows -> long PUTS
                      (continuation)
          Ambiguous tape (neither clearly intact nor broken) is skipped.

  SETUP3  Unusual Flow Confirmation
          A ticker already flagged HOT-OPTIONS (kairos_signals_options)
          with directional flow, gated so we do NOT enter when earnings is
          within setup3.no_earnings_within_days (IV-crush risk). Direction
          follows the flow; uses the tighter "flow" delta band.

IV snapshot cache (kairos_iv_history.json):
    A rolling per-ticker daily ATM-IV history. compute_iv_rank() returns a
    true percentile once enough samples exist, falling back to a coarse
    heuristic for cold-start tickers.

Public surface:
    record_iv_snapshot(ticker, iv)         -> None
    compute_iv_rank(ticker, current_iv)    -> float (0..1)
    detect_catalyst_signals(tickers=None)  -> list[dict]   # the signals

Each signal dict is self-contained for the executor + logger:
    {
      "ticker", "setup", "right" ("C"/"P"), "direction" ("bullish"/"bearish"),
      "delta_band" ("default"/"reversion"/"flow"),
      "iv", "iv_rank", "spot", "rationale",
      "signals_fired": [tags/sub-signals], "context": {...}
    }
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
IV_HISTORY_FILE = os.path.join(SCRIPT_DIR, "kairos_iv_history.json")
SIGNAL_SUMMARY_FILE = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")

W = 72
IV_HISTORY_MAX = 252          # ~1 trading year of daily snapshots
IV_RANK_MIN_SAMPLES = 20      # below this, fall back to the coarse heuristic
IV_HEURISTIC_FLOOR = 0.15
IV_HEURISTIC_CEIL = 0.80


# ── Config ───────────────────────────────────────────────────────────

def load_hot_catalyst_config() -> dict:
    """Read the hot_catalyst block from kairos_config.json (with defaults)."""
    defaults = {
        "dry_run": True,
        "max_options_portfolio_pct": 0.15,
        "max_single_position_pct": 0.025,
        "stop_loss_pct": 0.50,
        "take_profit_pct": 1.00,
        "close_at_dte": 21,
        "dte_min_entry": 30,
        "dte_max_entry": 60,
        "no_entry_on_earnings_day": True,
        "delta_bands": {
            "default": [0.30, 0.40],
            "reversion": [0.40, 0.50],
            "flow": [0.25, 0.35],
        },
        "setup1": {"enabled": True, "window_days_before_min": 7,
                   "window_days_before_max": 21, "iv_rank_max": 0.50},
        "setup2": {"enabled": True, "drop_pct_min": 10, "drop_lookback_days": 3},
        "setup3": {"enabled": True, "no_earnings_within_days": 5},
    }
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        block = cfg.get("hot_catalyst", {})
        if isinstance(block, dict):
            for k, v in block.items():
                if k == "_comment":
                    continue
                defaults[k] = v
    except (IOError, json.JSONDecodeError):
        pass
    return defaults


def load_signal_tags() -> dict[str, list[str]]:
    """Load {ticker: [HOT-*]} from kairos_signal_summary.json."""
    try:
        with open(SIGNAL_SUMMARY_FILE) as f:
            data = json.load(f)
        tags = data.get("signal_tags", {})
        return tags if isinstance(tags, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


# ── IV snapshot cache ────────────────────────────────────────────────

def _load_iv_history() -> dict:
    if not os.path.exists(IV_HISTORY_FILE):
        return {}
    try:
        with open(IV_HISTORY_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _save_iv_history(history: dict) -> None:
    try:
        with open(IV_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: IV history write failed: {exc}")


def record_iv_snapshot(ticker: str, iv: float, today: Optional[str] = None) -> None:
    """Append today's ATM IV for a ticker (one sample per day, capped)."""
    if not ticker or iv is None or iv <= 0:
        return
    sym = ticker.upper()
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history = _load_iv_history()
    series = history.get(sym, [])
    if not isinstance(series, list):
        series = []
    # One snapshot per calendar day — replace if today already recorded.
    series = [s for s in series if s.get("date") != today]
    series.append({"date": today, "iv": round(float(iv), 4)})
    series = series[-IV_HISTORY_MAX:]
    history[sym] = series
    _save_iv_history(history)


def compute_iv_rank(ticker: str, current_iv: float) -> float:
    """True percentile rank of current_iv within the cached history.

    Returns the fraction of historical samples <= current_iv. Falls back to
    the coarse (iv-0.15)/(0.80-0.15) heuristic when the cache holds fewer
    than IV_RANK_MIN_SAMPLES — same behavior as the legacy HOT-OPTIONS path.
    """
    if current_iv is None or current_iv <= 0:
        return 0.0
    series = _load_iv_history().get((ticker or "").upper(), [])
    ivs = [s["iv"] for s in series if isinstance(s, dict) and s.get("iv", 0) > 0]
    if len(ivs) >= IV_RANK_MIN_SAMPLES:
        at_or_below = sum(1 for v in ivs if v <= current_iv)
        return round(at_or_below / len(ivs), 4)
    rank = (current_iv - IV_HEURISTIC_FLOOR) / (IV_HEURISTIC_CEIL - IV_HEURISTIC_FLOOR)
    return round(max(0.0, min(1.0, rank)), 4)


# ── yfinance data helpers (graceful) ─────────────────────────────────

def _fetch_history(ticker: str, period: str = "1mo") -> Optional[list[dict]]:
    """Daily OHLC, oldest->newest. None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    except Exception:
        return None
    if hist is None or len(hist) == 0:
        return None
    try:
        return [
            {"high": float(r["High"]), "low": float(r["Low"]),
             "close": float(r["Close"])}
            for _, r in hist.iterrows()
            if r["Close"] == r["Close"] and r["Close"] > 0
        ]
    except Exception:
        return None


def _fetch_atm_iv(ticker: str) -> tuple[Optional[float], Optional[float]]:
    """(current_spot, atm_iv) via yfinance nearest expiry. (None, None) on failure.

    atm_iv = mean implied vol of the ~ATM call and put on the nearest expiry.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", auto_adjust=False)
        if hist is None or len(hist) == 0:
            return None, None
        spot = float(hist["Close"].iloc[-1])
        if spot <= 0 or not tk.options:
            return spot if spot > 0 else None, None
        chain = tk.option_chain(tk.options[0])
        ivs = []
        for df in (chain.calls, chain.puts):
            if df is None or df.empty:
                continue
            atm = df.iloc[(df["strike"] - spot).abs().argsort()[:1]]
            for v in atm["impliedVolatility"].tolist():
                if v == v and v > 0:
                    ivs.append(float(v))
        atm_iv = round(sum(ivs) / len(ivs), 4) if ivs else None
        return spot, atm_iv
    except Exception:
        return None, None


# ── Setup 1: Pre-Catalyst Underpriced Volatility ─────────────────────

def _detect_setup1(cfg: dict) -> list[dict]:
    s1 = cfg.get("setup1", {})
    if not s1.get("enabled", True):
        return []
    try:
        from kairos_signals_events import get_active_events
    except ImportError:
        return []

    win_min = int(s1.get("window_days_before_min", 7))
    win_max = int(s1.get("window_days_before_max", 21))
    iv_rank_max = float(s1.get("iv_rank_max", 0.50))

    signals: list[dict] = []
    for event in get_active_events():
        if event.get("phase") != "pre":
            continue
        days_until = int(event.get("days_until_event", 0))
        if not (win_min <= days_until <= win_max):
            continue

        ticker = event["ticker"].upper()
        spot, iv = _fetch_atm_iv(ticker)
        if iv is not None:
            record_iv_snapshot(ticker, iv)
        iv_rank = compute_iv_rank(ticker, iv) if iv else None
        if iv_rank is None or iv_rank > iv_rank_max:
            continue  # vol not (yet) underpriced — no edge

        # Calendar events are bullish catalysts -> long calls into the move.
        signals.append({
            "ticker": ticker,
            "setup": "SETUP1",
            "right": "C",
            "direction": "bullish",
            "delta_band": "default",
            "iv": iv,
            "iv_rank": iv_rank,
            "spot": spot,
            "rationale": (
                f"{ticker} is {days_until}d before {event['event']} "
                f"({event['event_date']}) with IV rank {iv_rank:.0%} "
                f"<= {iv_rank_max:.0%} — underpriced vol into a bullish "
                f"catalyst. Long calls (Setup 1)."
            ),
            "signals_fired": ["HOT-CATALYST", "SETUP1-PRECATALYST"],
            "context": {
                "event": event["event"],
                "event_date": event["event_date"],
                "days_until_event": days_until,
                "pattern": event.get("pattern"),
            },
        })
    return signals


# ── Setup 2: Post-Crush Reversion (thesis-driven direction) ──────────

def _detect_setup2(tickers: list[str], cfg: dict) -> list[dict]:
    s2 = cfg.get("setup2", {})
    if not s2.get("enabled", True):
        return []

    drop_pct_min = float(s2.get("drop_pct_min", 10))
    lookback = int(s2.get("drop_lookback_days", 3))

    signals: list[dict] = []
    for ticker in tickers:
        sym = ticker.upper()
        hist = _fetch_history(sym, period="1mo")
        if not hist or len(hist) < lookback + 2:
            continue

        closes = [d["close"] for d in hist]
        highs = [d["high"] for d in hist]
        lows = [d["low"] for d in hist]

        last_close = closes[-1]
        prior_low = lows[-2]
        # Pre-crush peak: highest high over the crush window (incl. the day
        # just before it). gap_low: the crush low over the lookback window.
        recent_high = max(highs[-(lookback + 1):])
        gap_low = min(lows[-lookback:])
        if recent_high <= 0:
            continue
        drop_pct = (recent_high - last_close) / recent_high * 100.0
        if drop_pct < drop_pct_min:
            continue

        # Thesis-driven direction (locked heuristic):
        #   holding above the gap low and not making a lower low -> intact -> CALL
        #   failure to hold / lower low -> broken -> PUT
        #   otherwise ambiguous -> skip (stay flat).
        holding = last_close >= gap_low * 1.02 and last_close >= prior_low
        broken = last_close <= gap_low or last_close < prior_low

        if holding and not broken:
            right, direction, thesis = "C", "bullish", "intact"
        elif broken and not holding:
            right, direction, thesis = "P", "bearish", "broken"
        else:
            continue  # ambiguous tape — no trade

        spot, iv = _fetch_atm_iv(sym)
        if iv is not None:
            record_iv_snapshot(sym, iv)
        iv_rank = compute_iv_rank(sym, iv) if iv else None

        signals.append({
            "ticker": sym,
            "setup": "SETUP2",
            "right": right,
            "direction": direction,
            "delta_band": "reversion",
            "iv": iv,
            "iv_rank": iv_rank,
            "spot": spot if spot else last_close,
            "rationale": (
                f"{sym} dropped {drop_pct:.1f}% over ~{lookback}d "
                f"(pre-crush {recent_high:.2f} -> {last_close:.2f}, gap low "
                f"{gap_low:.2f}). Tape reads {thesis} -> long "
                f"{'calls' if right == 'C' else 'puts'} (Setup 2)."
            ),
            "signals_fired": ["HOT-CATALYST", f"SETUP2-{thesis.upper()}"],
            "context": {
                "drop_pct": round(drop_pct, 2),
                "pre_crush_high": round(recent_high, 2),
                "gap_low": round(gap_low, 2),
                "last_close": round(last_close, 2),
                "thesis": thesis,
            },
        })
    return signals


# ── Setup 3: Unusual Flow Confirmation ───────────────────────────────

def _detect_setup3(cfg: dict, signal_tags: dict) -> list[dict]:
    s3 = cfg.get("setup3", {})
    if not s3.get("enabled", True):
        return []
    try:
        from kairos_signals_options import load_options_activity
    except ImportError:
        return []

    activity = load_options_activity()
    if not activity:
        return []

    signals: list[dict] = []
    for sym, data in activity.items():
        sym = sym.upper()
        # Earnings gate: HOT-EARNINGS means earnings is imminent -> skip
        # (IV crush would wreck a long-premium flow follow).
        if "HOT-EARNINGS" in [t.upper() for t in signal_tags.get(sym, [])]:
            continue

        direction = data.get("direction", "neutral")
        if direction in ("bullish", "leaning_bullish"):
            right, dirn = "C", "bullish"
        elif direction == "bearish":
            right, dirn = "P", "bearish"
        else:
            continue  # no clear directional flow to follow

        triggered = data.get("triggered_by", [])
        spot, iv = _fetch_atm_iv(sym)
        if iv is not None:
            record_iv_snapshot(sym, iv)
        iv_rank = data.get("iv_rank")
        if iv_rank is None and iv:
            iv_rank = compute_iv_rank(sym, iv)

        signals.append({
            "ticker": sym,
            "setup": "SETUP3",
            "right": right,
            "direction": dirn,
            "delta_band": "flow",
            "iv": iv,
            "iv_rank": iv_rank,
            "spot": spot,
            "rationale": (
                f"{sym} shows unusual options flow ({', '.join(triggered)}; "
                f"C/P {data.get('call_put_ratio', 0)}x, {direction}) and no "
                f"earnings within {s3.get('no_earnings_within_days', 5)}d — "
                f"follow the flow with long "
                f"{'calls' if right == 'C' else 'puts'} (Setup 3)."
            ),
            "signals_fired": ["HOT-CATALYST", "HOT-OPTIONS", "SETUP3-FLOW"],
            "context": {
                "triggered_by": triggered,
                "call_put_ratio": data.get("call_put_ratio"),
                "vol_oi_ratio": data.get("vol_oi_ratio"),
                "front_expiry": data.get("front_expiry"),
            },
        })
    return signals


# ── Setup LOCKUP: IPO Lock-Up Expiration (bearish, known catalyst) ───

def _detect_lockup(cfg: dict) -> list[dict]:
    """Emit long-PUT signals for armed IPO lock-up short theses.

    The ipo_lockup_tracker table is the source of truth: kairos_ipo_lockup
    scores the short thesis ~30 days before a 180-day lock-up expiration and
    flips a row to status='armed' (two-lock gated). Here we read those armed
    rows and surface them as bearish long-put setups so the options engine
    enters them — a defined-risk play on a known, predictably-timed catalyst.
    """
    try:
        from kairos_ipo_lockup import load_lockup_config
        from kairos_log_db import get_armed_lockup_signals
    except ImportError:
        return []
    if not load_lockup_config().get("enabled", True):
        return []

    signals: list[dict] = []
    for row in get_armed_lockup_signals():
        ticker = (row.get("ticker") or "").upper()
        if not ticker:
            continue

        lockup_date = row.get("lockup_expiration_date")
        days_left = None
        if lockup_date:
            try:
                d = datetime.strptime(str(lockup_date)[:10], "%Y-%m-%d").date()
                days_left = (d - datetime.now(timezone.utc).date()).days
            except ValueError:
                days_left = None

        spot, iv = _fetch_atm_iv(ticker)
        if iv is not None:
            record_iv_snapshot(ticker, iv)
        iv_rank = compute_iv_rank(ticker, iv) if iv else None
        if spot is None:
            spot = row.get("current_price")

        perf = row.get("perf_since_ipo_pct")
        insider = row.get("insider_pct")
        score = row.get("short_score")
        perf_s = f"{perf:+.1f}%" if isinstance(perf, (int, float)) else "n/a"
        ins_s = f"{insider:.0f}%" if isinstance(insider, (int, float)) else "n/a"

        signals.append({
            "ticker": ticker,
            "setup": "LOCKUP",
            "right": "P",
            "direction": "bearish",
            "delta_band": "default",
            "iv": iv,
            "iv_rank": iv_rank,
            "spot": spot,
            "rationale": (
                f"{ticker} 180-day IPO lock-up expires {lockup_date}"
                + (f" ({days_left}d)" if days_left is not None else "")
                + f"; up {perf_s} since IPO, insider concentration {ins_s}, "
                f"short thesis {score}/10. Predictable supply unlock — long "
                f"puts (Lock-Up)."
            ),
            "signals_fired": ["HOT-CATALYST", "LOCKUP-EXPIRY"],
            "context": {
                "lockup_date": lockup_date,
                "days_to_expiration": days_left,
                "perf_since_ipo_pct": perf,
                "insider_pct": insider,
                "short_score": score,
                "ipo_price": row.get("ipo_price"),
            },
        })
    return signals


# ── Public entry ─────────────────────────────────────────────────────

def detect_catalyst_signals(
    tickers: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> list[dict]:
    """Run all three setups and return long-option signals.

    `tickers` scopes the Setup 2 crush scan (defaults to the tickers in
    kairos_signal_summary.json). Setup 1 is driven by the event calendar
    and Setup 3 by cached HOT-OPTIONS activity, so both ignore `tickers`.

    The returned list contains BUY_CALL / BUY_PUT candidates only — this
    detector never emits a short/write signal.
    """
    cfg = config or load_hot_catalyst_config()
    signal_tags = load_signal_tags()
    if tickers is None:
        tickers = list(signal_tags.keys())
    tickers = [t.upper() for t in tickers]

    signals: list[dict] = []
    signals.extend(_detect_setup1(cfg))
    signals.extend(_detect_setup2(tickers, cfg))
    signals.extend(_detect_setup3(cfg, signal_tags))
    signals.extend(_detect_lockup(cfg))

    # Long-only invariant guard: drop anything that isn't a clean C/P buy.
    clean = [s for s in signals if s.get("right") in ("C", "P")]

    # HARD-pause block (point 3 of the pause-mode design). This engine is the
    # route that let options trades fire while HOT-OPTIONS was "paused" (the
    # generation chokepoint in kairos_signals.py never saw it). If HOT-OPTIONS
    # or HOT-CATALYST is HARD-paused, drop the corresponding candidates here so
    # a hard-killed signal genuinely cannot produce a trade. SOFT does NOT block
    # generation of a confirmer-style flow signal — only hard cuts the engine.
    try:
        from kairos_signals import signal_pause_mode
        blocked = {sig for sig in ("HOT-OPTIONS", "HOT-CATALYST")
                   if signal_pause_mode(sig) == "hard"}
    except Exception:
        blocked = set()
    if blocked:
        before = len(clean)
        def _sig_type(s: dict) -> str:
            # SETUP3 follows unusual options flow -> HOT-OPTIONS; the other
            # setups are catalyst-driven -> HOT-CATALYST.
            return "HOT-OPTIONS" if s.get("setup") == "SETUP3" else "HOT-CATALYST"
        clean = [s for s in clean if _sig_type(s) not in blocked]
        dropped = before - len(clean)
        if dropped:
            print(f"  [PAUSED:hard] catalyst engine dropped {dropped} candidate(s) "
                  f"for blocked signal(s): {', '.join(sorted(blocked))}")

    return clean


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kairos HOT-CATALYST detector")
    parser.add_argument("--tickers", default="",
                        help="Comma-separated tickers for the Setup 2 scan "
                             "(default: kairos_signal_summary.json)")
    parser.add_argument("--ivrank", nargs=2, metavar=("TICKER", "IV"),
                        help="Print the cached IV rank for TICKER given IV")
    parser.add_argument("--record-iv", nargs=2, metavar=("TICKER", "IV"),
                        help="Record an IV snapshot for TICKER")
    args = parser.parse_args()

    if args.record_iv:
        record_iv_snapshot(args.record_iv[0], float(args.record_iv[1]))
        print(f"  Recorded IV snapshot: {args.record_iv[0].upper()} "
              f"iv={args.record_iv[1]}")
        return
    if args.ivrank:
        rank = compute_iv_rank(args.ivrank[0], float(args.ivrank[1]))
        print(f"  IV rank {args.ivrank[0].upper()} @ {args.ivrank[1]}: "
              f"{rank:.0%}")
        return

    print("=" * W)
    print("  KAIROS HOT-CATALYST DETECTOR  (long options only)")
    print("=" * W)

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else None
    )
    signals = detect_catalyst_signals(tickers=tickers)

    if not signals:
        print("  No catalyst setups detected.")
    else:
        print(f"  {len(signals)} signal(s):\n")
        for s in signals:
            side = "BUY_CALL" if s["right"] == "C" else "BUY_PUT"
            ivr = f"{s['iv_rank']:.0%}" if s.get("iv_rank") is not None else "n/a"
            print(f"  [{s['setup']}] {s['ticker']}: {side} "
                  f"({s['direction']}, band={s['delta_band']}, IVrank={ivr})")
            print(f"      {s['rationale']}")
    print("=" * W)


if __name__ == "__main__":
    main()
