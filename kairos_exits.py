"""
Kairos Exit Engine — Exit Architecture v2 (Profit Maximization)

The unified five-condition exit engine. Core principle: **time is never an
exit trigger for a winning position.** A position is held as long as it is
advantageous; exits happen only when one of five conditions is true.

This module OWNS conditions 1, 2 and 5 and runs once per cycle in place of
the old flat regime stop-loss (kairos_stoploss.run_stoploss):

  1. HARD STOP-LOSS   — signal-aware closing stop + 1.5x intraday backstop,
                        regime threshold applied as a tighter-of-the-two floor.
  2. TRAILING STOP    — peak-gain tiers 15/30/50% trail 8/10/12%, closing
                        price + intraday backstop.
  5. HOT-REVERSION
     TIME GATE        — reversion entry held >= N days → exit (the ONLY
                        legitimate time-based exit).

It also exposes three SHARED OVERLAY helpers imported by the qualitative and
capital-redeployment gates (which own conditions 3 and 4):

  - tax_gate_blocks_exit(...)   — delays non-stop exits near the 12-month
                                  long-term-capital-gains anniversary.
  - conviction_decay_score(...) — aging signals lower a position's conviction;
                                  below the threshold it becomes liberation-eligible.
  - reentry_guard_blocks(...)   — no-higher-price re-entry unless a genuinely
                                  new signal fired.

Closing-price evaluation: there is no dedicated end-of-day run, so the cycle
landing in the configured close window (close_window_et) is treated as the
"close." The intraday backstop runs on live price every cycle.

Usage:
    from kairos_exits import run_exit_engine
    result = run_exit_engine(ib=conn, regime="NORMAL")

    python kairos_exits.py --dry-run     # old-vs-new comparison, no orders
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
REGIME_STATE_FILE = os.path.join(SCRIPT_DIR, ".kairos_regime_state.json")
W = 72

# Signal → stop-table key. Reversion takes precedence (tightest stop + the
# only signal with a time gate), then insider/congress, else standard.
_SIGNAL_PRIORITY = ["HOT-REVERSION", "HOT-INSIDER", "HOT-CONGRESS"]

# Defaults mirror the kairos_config.json "exits" block; used if config absent.
_DEFAULT_EXITS = {
    "stop_loss": {
        "HOT-REVERSION": {"closing": -6, "intraday_mult": 1.5},
        "HOT-INSIDER":   {"closing": -8, "intraday_mult": 1.5},
        "HOT-CONGRESS":  {"closing": -8, "intraday_mult": 1.5},
        "STANDARD":      {"closing": -8, "intraday_mult": 1.5},
    },
    "trailing_stop": {"enabled": True, "tiers": [[15, 8], [30, 10], [50, 12]]},
    "tax_gate": {"enabled": True, "days_before_anniversary": 30,
                 "deep_retreat_override_pct": 20},
    "conviction_decay": {
        "decay_days": {"HOT-EARNINGS": 10, "HOT-INSIDER": 365, "HOT-CONGRESS": 90,
                       "HOT-OPTIONS": 7, "HOT-REVERSION": 5, "DEFAULT": 7},
        "liberation_threshold": 0.5,
    },
    "reentry": {"no_higher_price_guard": True, "require_new_signal_above_exit": True},
    "hot_reversion_time_gate": {"days": 7},
    "close_window_et": {"start": "15:45", "end": "16:00"},
}


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Config + small shared utilities ──────────────────────────────────

def _load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}


def _exits_config() -> dict:
    """Return the 'exits' config block merged over defaults (shallow)."""
    cfg = _load_config().get("exits", {})
    merged = json.loads(json.dumps(_DEFAULT_EXITS))  # deep copy of defaults
    for key, val in cfg.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key].update(val)
        else:
            merged[key] = val
    return merged


def _get_regime() -> str:
    try:
        with open(REGIME_STATE_FILE) as f:
            return json.load(f).get("regime", "NORMAL")
    except (IOError, json.JSONDecodeError):
        return "NORMAL"


def _parse_entry_date(entry_date: str) -> datetime | None:
    """Parse a holdings.entry_date ('2026-05-19 14:12:41 UTC') to aware UTC."""
    if not entry_date:
        return None
    s = entry_date.replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


_IPO_TICKERS_CACHE = None


def _is_recent_ipo(ticker: str) -> bool:
    """True if ticker is in the IPO detection cache (kairos_ipo_cache.json).

    IPO identity is read from the cache, NOT the position's signal tag: the
    entry signal (IPO_MOMENTUM) is frequently lost by the time a position is
    held (conviction trades store empty confluence signals), so get_position_signal
    returns STANDARD for held IPOs. The cache's 'detected' set is the reliable
    source. Fails safe to False (no widening) if the cache is missing/unreadable.
    The HOLDING-day window check is applied by the caller, not here.
    """
    global _IPO_TICKERS_CACHE
    if _IPO_TICKERS_CACHE is None:
        try:
            import os
            cache_path = os.path.join(SCRIPT_DIR, "kairos_ipo_cache.json")
            with open(cache_path) as f:
                data = json.load(f)
            _IPO_TICKERS_CACHE = {t.upper() for t in (data.get("detected") or {})}
        except Exception:
            _IPO_TICKERS_CACHE = set()
    return ticker.upper() in _IPO_TICKERS_CACHE


def _get_entry_signals(ticker: str) -> list[str]:
    """Signals active at entry (most recent filled BUY) from kairos.db."""
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        row = conn.execute(
            "SELECT data_inputs FROM decisions "
            "WHERE ticker = ? AND action = 'BUY' AND execution_status = 'Filled' "
            "ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        conn.close()
        if row and row["data_inputs"]:
            di = json.loads(row["data_inputs"])
            conf = di.get("confluence", {})
            if conf.get("signals"):
                return conf["signals"]
            if di.get("signals"):
                return di["signals"]
    except Exception:
        pass
    return []


def _get_current_signals(ticker: str) -> list[str]:
    """Currently-active signal tags for a ticker (best-effort)."""
    try:
        from kairos_confluence import get_ticker_signals
        return get_ticker_signals(ticker) or []
    except Exception:
        return []


def get_position_signal(ticker: str, entry_signals: list[str] | None = None) -> str:
    """Pick the dominant entry signal that selects the stop-table row.

    Reversion > insider > congress > standard. Returns a stop_loss config key.
    """
    sigs = entry_signals if entry_signals is not None else _get_entry_signals(ticker)
    sigset = {str(s).upper() for s in sigs}
    for s in _SIGNAL_PRIORITY:
        if s in sigset:
            return s
    return "STANDARD"


# ── Condition 1 + 2: stop threshold math ─────────────────────────────

def hard_stop_threshold(signal: str, regime: str, cfg: dict | None = None) -> tuple[float, float]:
    """Return (closing_pct, intraday_pct) for the hard stop, both negative.

    Signal-aware closing stop, tightened by the regime floor (tighter-of-the-two
    — can only move the stop closer to 0, never looser). Intraday backstop is
    the regime-effective closing stop times the configured multiplier.
    """
    cfg = cfg or _exits_config()
    sl = cfg["stop_loss"].get(signal, cfg["stop_loss"]["STANDARD"])
    closing = float(sl["closing"])
    mult = float(sl.get("intraday_mult", 1.5))

    # Regime floor: -threshold% (e.g. RISK-OFF -6, EXTREME-FEAR -5). Take the
    # tighter (less negative) of the signal stop and the regime stop.
    from kairos_stoploss import STOP_LOSS_THRESHOLDS
    regime_stop = -abs(STOP_LOSS_THRESHOLDS.get(regime, 10.0))
    effective_closing = max(closing, regime_stop)
    intraday = effective_closing * mult
    return effective_closing, intraday


def trailing_stop_threshold(peak_gain_pct: float, cfg: dict | None = None) -> float | None:
    """Return the active trail % for the current peak, or None if no tier hit.

    Tiers are [[min_gain, trail_pct], ...]; the highest tier whose min_gain is
    reached wins (50% peak trails 12%, etc.).
    """
    cfg = cfg or _exits_config()
    ts = cfg.get("trailing_stop", {})
    if not ts.get("enabled", True):
        return None
    active = None
    for tier in ts.get("tiers", []):
        min_gain, trail = float(tier[0]), float(tier[1])
        if peak_gain_pct >= min_gain:
            active = trail
    return active


def evaluate_position(
    ticker: str,
    avg_cost: float,
    current_price: float,
    peak_gain_pct: float,
    holding_days: int,
    signal: str,
    regime: str,
    is_close_eval: bool,
    cfg: dict | None = None,
    entry_signals: list[str] | None = None,
) -> str | None:
    """Apply conditions 1, 2 and 5. Return a SELL reason, or None to hold.

    Order: hard intraday backstop (any cycle) → hard closing stop (close-eval
    only) → trailing intraday backstop → trailing closing stop (close-eval
    only) → HOT-REVERSION time gate.
    """
    cfg = cfg or _exits_config()
    if avg_cost is None or avg_cost <= 0:
        return None

    gain_pct = (current_price - avg_cost) / avg_cost * 100.0
    closing_stop, intraday_stop = hard_stop_threshold(signal, regime, cfg)

    # ── Condition 1: Hard stop-loss (always fires) ───────────────────
    if gain_pct <= intraday_stop:
        return (f"STOP-LOSS: {gain_pct:.1f}% <= intraday backstop "
                f"{intraday_stop:.1f}% ({signal}, {regime})")
    if is_close_eval and gain_pct <= closing_stop:
        return (f"STOP-LOSS: {gain_pct:.1f}% <= closing stop "
                f"{closing_stop:.1f}% ({signal}, {regime})")

    # ── Condition 2: Trailing stop (profit capture) ──────────────────
    trail = trailing_stop_threshold(peak_gain_pct, cfg)
    if trail is not None:
        # IPO widening: a young IPO whip-saws on day-1/2 noise, and the standard
        # trail fires a profit-capture exit on that noise. For the first
        # ipo_window_days HOLDING days of a detected IPO, widen the trail by
        # ipo_multiplier so the position can breathe. Identity comes from the IPO
        # cache (not the signal tag, which is lost on held conviction trades).
        ts_cfg = cfg.get("trailing_stop", {})
        ipo_window = int(ts_cfg.get("ipo_window_days", 0))
        ipo_mult = float(ts_cfg.get("ipo_multiplier", 1.0))
        ipo_widened = False
        if (ipo_mult > 1.0 and holding_days <= ipo_window
                and _is_recent_ipo(ticker)):
            trail = trail * ipo_mult
            ipo_widened = True
        retreat = peak_gain_pct - gain_pct  # how far off the high-water mark
        intraday_mult = float(cfg["stop_loss"].get(signal, cfg["stop_loss"]["STANDARD"])
                              .get("intraday_mult", 1.5))
        _tag = " [IPO-widened]" if ipo_widened else ""
        if retreat >= trail * intraday_mult:
            return (f"TRAILING-STOP: retreated {retreat:.1f}% from peak "
                    f"{peak_gain_pct:.1f}% (intraday backstop {trail * intraday_mult:.1f}%{_tag})")
        if is_close_eval and retreat >= trail:
            return (f"TRAILING-STOP: retreated {retreat:.1f}% from peak "
                    f"{peak_gain_pct:.1f}% (trail {trail:.1f}%, gain {gain_pct:+.1f}%{_tag})")

    # ── Condition 5: HOT-REVERSION validity check (replaced time gate) ──
    # Sell when price reverts to 30d SMA (thesis complete), not on elapsed time.
    # But only let REVERSION-COMPLETE govern when NO longer-horizon co-signal was
    # part of the entry: get_position_signal resolves any confluence containing a
    # reversion tag to HOT-REVERSION (reversion-first priority), so a position
    # entered on HOT-INSIDER/HOT-CONGRESS + HOT-REVERSION would otherwise be
    # force-sold at the 30d SMA, truncating the longer thesis.
    LONGER_HORIZON = {"HOT-INSIDER", "HOT-CONGRESS"}
    entry_set = {str(s).upper() for s in (entry_signals or [])}
    reversion_governs = (signal == "HOT-REVERSION") and not (entry_set & LONGER_HORIZON)
    if reversion_governs:
        try:
            from kairos_thesis_validity import check_signal_validity
            _entry_date = (datetime.now(timezone.utc) - timedelta(days=holding_days)).strftime("%Y-%m-%d")
            validity = check_signal_validity(
                ticker=ticker, signal_type="HOT-REVERSION",
                entry_date=_entry_date, entry_price=avg_cost,
                current_price=current_price, signals_fired={}
            )
            if validity.get("action") == "SELL":
                return (f"REVERSION-COMPLETE: {validity.get('reason', 'price reverted to mean')} "
                        f"(gain {gain_pct:+.1f}%)")
        except Exception:
            # Fallback: if validity check fails, use 30-day backstop
            backstop = int(cfg.get("thesis_validity", {}).get("backstop_days", {}).get("HOT-REVERSION", 30))
            if holding_days >= backstop:
                return (f"REVERSION-BACKSTOP: held {holding_days}d >= {backstop}d, "
                        f"validity check unavailable (gain {gain_pct:+.1f}%)")

    return None


# ── Shared overlay: Tax gate (delays conditions 3 and 4) ─────────────

def tax_gate_blocks_exit(
    ticker: str,
    entry_date: str,
    current_price: float,
    avg_cost: float,
    peak_gain_pct: float = 0.0,
    cfg: dict | None = None,
    alert: bool = True,
) -> bool:
    """True when a non-stop exit should be delayed for long-term-gains treatment.

    Blocks only when the position is profitable AND within
    days_before_anniversary of the 365-day entry anniversary. Exempt (returns
    False) when the position has retreated more than deep_retreat_override_pct
    from its peak — protecting a big gain wins over a tax deferral.

    NEVER call this from the hard/trailing stop path; it is an overlay on
    thesis-invalidation (3) and capital liberation (4) only.
    """
    cfg = cfg or _exits_config()
    tg = cfg.get("tax_gate", {})
    if not tg.get("enabled", True):
        return False
    if avg_cost is None or avg_cost <= 0 or current_price <= avg_cost:
        return False  # only profitable positions are delayed

    entry_dt = _parse_entry_date(entry_date)
    if entry_dt is None:
        return False

    now = datetime.now(timezone.utc)
    anniversary = entry_dt + timedelta(days=365)
    days_to_anniv = (anniversary - now).days
    window = int(tg.get("days_before_anniversary", 30))
    if not (0 <= days_to_anniv <= window):
        return False

    # Deep-retreat exemption: don't tax-delay if we've given back the gain.
    gain_pct = (current_price - avg_cost) / avg_cost * 100.0
    retreat = peak_gain_pct - gain_pct
    if retreat > float(tg.get("deep_retreat_override_pct", 20)):
        return False

    if alert:
        try:
            from kairos_alerts import post_message
            # Exit-gating explanation (no trade occurred, not a health item) → #log.
            post_message(
                "log",
                f":hourglass_flowing_sand: *{ticker} exit delayed {days_to_anniv}d "
                f"for long-term capital gains treatment* — current gain {gain_pct:+.1f}%")
        except Exception as exc:
            print(f"  tax-gate delay Slack post failed: {exc}")
    return True


# ── Shared overlay: Conviction decay (feeds condition 4) ─────────────

def get_liberation_threshold(cfg: dict | None = None) -> float:
    cfg = cfg or _exits_config()
    return float(cfg.get("conviction_decay", {}).get("liberation_threshold", 0.5))


def conviction_decay_score(
    ticker: str,
    entry_signals: list[str] | None = None,
    entry_date: str | None = None,
    cfg: dict | None = None,
    validity_cache: dict | None = None,
) -> float:
    """Decayed conviction for an open position.

    Each entry signal contributes its confluence base points, linearly decayed
    over its per-signal horizon: base * max(0, 1 - age/decay_days). A signal
    still active today does NOT decay (conviction refreshed). Any NEW signal
    active now but not at entry adds full base points. Below the liberation
    threshold the position is eligible for capital redeployment.

    Validity-based decay (Exit Architecture v2): when a cycle-scoped
    validity_cache is supplied (see kairos_thesis_validity.warm_validity_cache),
    decay_days become MAXIMUM windows that the thesis-validity score stretches
    or compresses — a still-sound thesis (high validity) decays slower so a
    fading signal doesn't prematurely liberate it, a failing thesis decays
    faster so capital is released sooner. validity_cache=None (every existing
    caller) preserves the original purely time-based decay unchanged.
    """
    cfg = cfg or _exits_config()
    decay_days = cfg.get("conviction_decay", {}).get("decay_days", {})
    default_days = float(decay_days.get("DEFAULT", 7))
    signal_points = _load_config().get("confluence", {}).get("signal_points", {})

    entry_sigs = entry_signals if entry_signals is not None else _get_entry_signals(ticker)
    current_sigs = set(_get_current_signals(ticker))

    age_days = 0.0
    if entry_date:
        dt = _parse_entry_date(entry_date)
        if dt is not None:
            age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)

    score = 0.0
    counted = set()
    for sig in entry_sigs:
        base = float(signal_points.get(sig, 1))
        counted.add(sig)
        if sig in current_sigs:
            score += base  # refreshed — no decay
        else:
            dd = float(decay_days.get(sig, default_days)) or default_days
            # Modulate the decay window by thesis validity. High validity →
            # slower decay (thesis sound despite the signal fading); low
            # validity → faster decay (thesis failing, release capital sooner).
            # 45-64 is neutral and leaves dd unchanged. decay_days are MAXIMUM
            # windows; validity stretches them up to 2.5x or compresses to 0.2x.
            if validity_cache is not None:
                try:
                    from kairos_thesis_validity import get_cached_validity
                    v_score = get_cached_validity(ticker, validity_cache).get("score", 60)
                    if v_score >= 75:
                        dd *= 2.5      # very strong thesis — decay ~40% of normal
                    elif v_score >= 65:
                        dd *= 1.5      # strong thesis — decay ~67% of normal
                    elif v_score < 35:
                        dd *= 0.2      # failing thesis — decay ~5x normal
                    elif v_score < 45:
                        dd *= 0.4      # weak thesis — decay ~2.5x normal
                    # 45-64: neutral — default decay rate unchanged
                except Exception:
                    pass  # graceful: fall back to pure time-based decay
            score += base * max(0.0, 1.0 - age_days / dd)

    # Fresh signals fired after entry add full conviction.
    for sig in current_sigs:
        if sig not in counted:
            score += float(signal_points.get(sig, 1))

    return round(score, 3)


# ── Shared overlay: Re-entry guard (gates condition-1/2 re-entries) ──

def reentry_guard_blocks(
    ticker: str,
    proposed_price: float,
    current_signals: list[str] | None = None,
    cfg: dict | None = None,
) -> tuple[bool, str]:
    """Block re-buying a ticker ABOVE its last exit price unless a new signal fired.

    No time cooldown. Re-entry at or below the exit price is always allowed.
    Above the exit price requires a current signal that was NOT present at the
    last exit (a genuinely new setup, not the same stale thesis).
    Returns (blocked, reason).
    """
    cfg = cfg or _exits_config()
    re = cfg.get("reentry", {})
    if not re.get("no_higher_price_guard", True):
        return False, ""

    from kairos_log_db import get_position_exit
    rec = get_position_exit(ticker)
    if not rec or rec.get("exit_price") is None:
        return False, ""

    exit_price = float(rec["exit_price"])
    if proposed_price <= exit_price:
        return False, ""

    if re.get("require_new_signal_above_exit", True):
        cur = set(current_signals if current_signals is not None
                  else _get_current_signals(ticker))
        exit_sigs = set(rec.get("exit_signals") or [])
        new_signals = cur - exit_sigs
        if new_signals:
            return False, ""  # genuinely new signal justifies the higher entry

    reason = (f"RE-ENTRY GUARD: ${proposed_price:.2f} > last exit ${exit_price:.2f} "
              f"({rec.get('exit_reason', 'prior exit')}) with no new signal")
    return True, reason


# ── Engine ────────────────────────────────────────────────────────────

def _in_close_window(cfg: dict | None = None) -> bool:
    """True if the current ET wall-clock is inside the configured close window."""
    cfg = cfg or _exits_config()
    cw = cfg.get("close_window_et", {})
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        sh, sm = (int(x) for x in cw.get("start", "15:45").split(":"))
        eh, em = (int(x) for x in cw.get("end", "16:00").split(":"))
        start_min, end_min, cur_min = sh * 60 + sm, eh * 60 + em, now.hour * 60 + now.minute
        return start_min <= cur_min <= end_min
    except Exception:
        return False


def _load_open_holdings() -> list[dict]:
    """Open positions aggregated by ticker, with peak and earliest entry."""
    from kairos_log_db import get_connection
    conn = get_connection()
    rows = conn.execute("""
        SELECT ticker,
               SUM(quantity) AS total_qty,
               ROUND(SUM(entry_price * quantity) / SUM(quantity), 2) AS avg_cost,
               MIN(entry_date) AS earliest_entry,
               MAX(peak_gain_pct) AS peak_gain_pct,
               CAST(julianday(datetime('now')) - julianday(REPLACE(MIN(entry_date), ' UTC', '')) AS INTEGER) AS holding_days
        FROM holdings
        WHERE sold_date IS NULL
        GROUP BY ticker
        ORDER BY ticker
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _old_verdict(gain_pct: float, holding_days: int, current_signals: list[str],
                 regime: str) -> str:
    """What the legacy logic (regime flat stop + stale-thesis) would have done."""
    from kairos_stoploss import (STOP_LOSS_THRESHOLDS, TAX_OVERRIDE_MIN_DAYS,
                                 TAX_OVERRIDE_MAX_DAYS, TAX_OVERRIDE_DRAWDOWN_CAP)
    threshold = STOP_LOSS_THRESHOLDS.get(regime, 10.0)
    if gain_pct < -threshold:
        if (TAX_OVERRIDE_MIN_DAYS <= holding_days <= TAX_OVERRIDE_MAX_DAYS
                and abs(gain_pct) < TAX_OVERRIDE_DRAWDOWN_CAP):
            return "HOLD (tax-override)"
        return f"STOP-LOSS (regime {threshold:.0f}%)"
    # Legacy stale-thesis (removed in v2): 30d + <2% + no signals
    if holding_days >= 30 and gain_pct < 2.0 and not current_signals:
        return "STALE-THESIS (30d)"
    return "HOLD"


def run_exit_engine(ib=None, regime: str | None = None, dry_run: bool = False) -> dict:
    """Run the five-condition exit engine across all open equity positions.

    Conditions 1 (hard stop), 2 (trailing stop) and 5 (reversion time gate) are
    evaluated here every cycle; closing-price stops fire only inside the close
    window, the intraday backstop fires any cycle. Peak gain is updated first.

    Args:
        ib:       optional existing IBKR connection to reuse for live prices.
        regime:   override regime (defaults to .kairos_regime_state.json).
        dry_run:  evaluate + print old-vs-new verdicts; place no orders, write
                  no rows, update no peaks.

    Returns {"checked", "triggered", "sells": [...], "rows": [...]}.
    """
    cfg = _exits_config()
    if regime is None:
        regime = _get_regime()
    is_close = _in_close_window(cfg)

    print(f"  Regime: {regime}  |  close-window eval: {is_close}"
          f"{'  |  DRY RUN' if dry_run else ''}")

    holdings = _load_open_holdings()
    if not holdings:
        print("  No open holdings — nothing to evaluate")
        return {"checked": 0, "triggered": 0, "sells": [], "rows": []}

    print(f"  Evaluating {len(holdings)} positions")

    # IBKR connection for live prices (reuse if a connected one was passed in).
    from kairos_stoploss import _get_ibkr_price, _place_market_sell, _log_sell, _alert_stoploss
    owns_conn = False
    if not dry_run:
        connected = False
        try:
            connected = ib is not None and ib.isConnected()
        except Exception:
            connected = False
        if not connected:
            from ib_insync import IB
            import random
            ib = IB()
            try:
                ib.connect("127.0.0.1", 7497, clientId=random.randint(20, 29), timeout=10)
                owns_conn = True
            except Exception as exc:
                print(f"  ERROR: IBKR connection failed: {exc}")
                return {"checked": 0, "triggered": 0, "sells": [], "rows": [],
                        "error": str(exc)}

    result = {"checked": 0, "triggered": 0, "sells": [], "rows": []}

    for h in holdings:
        ticker = h["ticker"]
        total_qty = int(h["total_qty"])
        avg_cost = h["avg_cost"]
        holding_days = h["holding_days"] or 0
        stored_peak = float(h["peak_gain_pct"] or 0.0)
        result["checked"] += 1

        # Price — live IBKR, or Finnhub fallback in dry-run when IBKR is off.
        current_price = _get_ibkr_price(ib, ticker) if ib is not None else None
        if current_price is None and dry_run:
            current_price = _finnhub_price(ticker)
        if current_price is None:
            print(f"    {ticker}: no price — skipped")
            continue
        if avg_cost is None or avg_cost == 0:
            print(f"    {ticker}: invalid avg_cost — skipped")
            continue

        gain_pct = (current_price - avg_cost) / avg_cost * 100.0
        peak_gain_pct = max(stored_peak, gain_pct)

        # Update the high-water mark (skip in dry-run).
        if not dry_run and peak_gain_pct > stored_peak:
            try:
                from kairos_log_db import update_peak_gain
                update_peak_gain(ticker, round(peak_gain_pct, 3))
            except Exception as exc:
                print(f"    WARNING: peak update failed for {ticker}: {exc}")

        entry_signals = _get_entry_signals(ticker)
        signal = get_position_signal(ticker, entry_signals)
        current_signals = _get_current_signals(ticker)

        reason = evaluate_position(
            ticker, avg_cost, current_price, peak_gain_pct, holding_days,
            signal, regime, is_close, cfg, entry_signals=entry_signals)

        row = {
            "ticker": ticker, "avg_cost": avg_cost, "price": round(current_price, 2),
            "gain_pct": round(gain_pct, 1), "peak_pct": round(peak_gain_pct, 1),
            "holding_days": holding_days, "signal": signal,
            "new": (reason.split(":")[0] if reason else "HOLD"),
            "old": _old_verdict(gain_pct, holding_days, current_signals, regime),
        }
        result["rows"].append(row)

        if dry_run:
            print(f"    {ticker:6s} {gain_pct:+6.1f}% peak {peak_gain_pct:+6.1f}% "
                  f"{holding_days:>3}d {signal:13s}  OLD={row['old']:22s} NEW={row['new']}"
                  + (f"  ← {reason}" if reason else ""))
            if reason:
                result["triggered"] += 1
            continue

        if not reason:
            continue

        # ── Execute the exit ─────────────────────────────────────────
        print(f"    {ticker}: {gain_pct:+.1f}% → SELL {total_qty} — {reason}")
        execution = _place_market_sell(ib, ticker, total_qty)
        sell_price = execution.get("fill_price") or current_price

        _log_sell(ticker, total_qty, avg_cost, sell_price, holding_days, reason, execution)
        _alert_stoploss(ticker, avg_cost, sell_price, gain_pct, holding_days, reason)

        # Record the exit for the re-entry guard.
        try:
            from kairos_log_db import upsert_position_exit
            upsert_position_exit(
                ticker,
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                sell_price, reason, current_signals)
        except Exception as exc:
            print(f"    WARNING: position_exit upsert failed: {exc}")

        # ML close.
        try:
            from kairos_ml_outcomes import init_db as ml_init, write_trade_close, find_open_trade
            ml_init()
            tid = find_open_trade(ticker, "BUY")
            if tid:
                write_trade_close(tid, sell_price, timestamp_exit=None)
        except Exception as exc:
            print(f"    WARNING: ML outcomes close failed: {exc}")

        # Wash-sale violation check on loss sells.
        if sell_price < avg_cost:
            try:
                from kairos_wash_sale import check_wash_sale_violation, log_wash_sale_event
                ws = check_wash_sale_violation(ticker, sell_price, avg_cost)
                if ws["violation"]:
                    sell_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    blocked_until = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
                    loss_amt = round((avg_cost - sell_price) * total_qty, 2)
                    log_wash_sale_event(ticker, sell_date_str, sell_price, avg_cost,
                                        loss_amt, ws["repurchase_date"], blocked_until)
                    print(f"    ⚠ WASH SALE: ${loss_amt:,.2f} disallowed — "
                          f"repurchased {ws['repurchase_date']}")
            except Exception as exc:
                print(f"    WARNING: wash sale check failed: {exc}")

        result["triggered"] += 1
        result["sells"].append({
            "ticker": ticker, "qty": total_qty, "entry_price": avg_cost,
            "exit_price": sell_price, "gain_pct": round(gain_pct, 1),
            "holding_days": holding_days, "reason": reason,
            "status": execution.get("status", "?"),
        })

    if owns_conn:
        try:
            ib.disconnect()
        except Exception:
            pass

    print(f"\n  Exit engine: {result['checked']} checked, {result['triggered']} triggered"
          f"{' (dry run)' if dry_run else ''}")
    return result


def _finnhub_price(ticker: str) -> float | None:
    """Best-effort Finnhub quote — used only as a dry-run fallback when IBKR is off."""
    try:
        import requests
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            return None
        resp = requests.get("https://finnhub.io/api/v1/quote",
                            params={"symbol": ticker, "token": api_key}, timeout=10)
        c = resp.json().get("c", 0)
        return float(c) if c and c > 0 else None
    except Exception:
        return None


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos Exit Engine (Exit Architecture v2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate + print old-vs-new verdicts; no orders, no writes")
    parser.add_argument("--regime", help="Override regime (NORMAL/CAUTION/RISK-OFF/EXTREME-FEAR)")
    args = parser.parse_args()

    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS EXIT ENGINE — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    res = run_exit_engine(regime=args.regime, dry_run=args.dry_run)
    if args.dry_run and res.get("rows"):
        changed = [r for r in res["rows"] if r["old"] != r["new"]]
        print(banner("Old vs New — positions where the verdict changed"))
        if not changed:
            print("  (none — every position gets the same verdict under both regimes)")
        for r in changed:
            print(f"    {r['ticker']:6s} {r['gain_pct']:+6.1f}%  "
                  f"OLD={r['old']:24s} → NEW={r['new']}")
