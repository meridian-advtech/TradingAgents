"""
Kairos Capital Reallocation Evaluator

When a high-conviction trade is blocked by cash constraints, evaluates
whether selling the weakest existing position to fund it would be
NPV-positive after tax and transaction costs.

Activation criteria (Exit Architecture v2 — condition 4, capital liberation):
  - New opportunity has conviction >= 5 (absolute floor; the delta gate below
    does the real work of demanding a "much better" replacement)
  - The weakest existing position's DECAYED conviction has fallen below the
    liberation threshold (aging signals lower conviction — kairos_exits
    .conviction_decay_score). This replaces the old hold-window "runway" gate.
  - Conviction delta >= 4.5 between new opportunity and weakest position
    (~half the 1-10 scale — this is the definition of "much better")
  - Position has been held >= 5 TRADING days (anti-churn guard)
  - Tax gate: not within 30 days of the weakest position's 12-month anniversary
    while profitable (defers to long-term capital gains)
  - After-tax exit cost is not prohibitive (tax savings < $1,000)
  - No wash sale conflicts on either leg

Usage:
    from kairos_reallocation import evaluate_reallocation
    result = evaluate_reallocation(new_ticker, new_conviction, new_signals, ib, nlv)
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")

SCHEMA_REALLOCATION_EVENTS = """
CREATE TABLE IF NOT EXISTS reallocation_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_date         TEXT NOT NULL,
    ticker_exited     TEXT,
    ticker_entered    TEXT NOT NULL,
    conviction_delta  INTEGER,
    tax_cost          REAL,
    signals_at_exit   TEXT,
    reason            TEXT NOT NULL,
    recommended       INTEGER NOT NULL DEFAULT 0,
    executed          INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Absolute conviction floor to even consider reallocation. Lowered 7 -> 5: this
# is a floor stacked in front of the RELATIVE delta gate, and a 7-floor blocked
# the exact case we WANT — dumping a conviction 1-2.5 dog for a "merely good"
# 5.5-6.9 replacement. 5 keeps the replacement above-average; the delta does the
# real work. Overridable from kairos_config.json (see _load_reallocation_config).
MIN_CONVICTION = 5
# Minimum gap between new conviction and weakest position score. This IS the
# definition of "much better" (~half the 1-10 scale) and the primary anti-churn
# brake — kept at 4.5 deliberately.
MIN_CONVICTION_DELTA = 4.5
# Max tax savings from waiting before we refuse to exit (prohibitive threshold)
MAX_TAX_SAVINGS_FOR_EXIT = 1000
# Minimum hold time in TRADING days before a position can be reallocated
# (anti-churn guard). 5 trading days (~1 week) is a brake against reflexive flips
# without letting dogs rot; deep losers (> -15%) still route to the stop-loss
# path via the unrealized-loss guard, so true dogs are not delayed by this.
MIN_HOLD_DAYS_FOR_REALLOCATION = 5
# Maximum unrealized loss percentage before blocking reallocation (-15% = block sells down >15%)
MAX_REALLOCATION_LOSS_PCT = -15.0
# Maximum age of thesis review in hours (default 24)
MAX_THESIS_REVIEW_AGE_HOURS = 24
# Minimum remaining runway days to block reallocation (default 0 = any positive runway blocks)
MIN_REMAINING_RUNWAY_DAYS = 0
# Signal hold windows (loaded from config)
SIGNAL_HOLD_WINDOWS = {}


def _load_reallocation_config() -> dict:
    """Load reallocation configuration from kairos_config.json."""
    try:
        import json
        import os
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kairos_config.json")
        with open(config_file) as f:
            config = json.load(f)
        realloc_config = config.get("reallocation", {})

        # Use config values if available, otherwise use defaults. NOTE: the three
        # reallocation GATES (MIN_CONVICTION, MIN_CONVICTION_DELTA,
        # MIN_HOLD_DAYS_FOR_REALLOCATION) were previously omitted from this global
        # statement, so config values for them were dead/decorative and the module
        # constants were the real source of truth. They are now config-authoritative.
        global MAX_REALLOCATION_LOSS_PCT, MAX_THESIS_REVIEW_AGE_HOURS, MIN_REMAINING_RUNWAY_DAYS, SIGNAL_HOLD_WINDOWS
        global MIN_CONVICTION, MIN_CONVICTION_DELTA, MIN_HOLD_DAYS_FOR_REALLOCATION
        MAX_REALLOCATION_LOSS_PCT = realloc_config.get("MAX_REALLOCATION_LOSS_PCT", -15.0)
        MAX_THESIS_REVIEW_AGE_HOURS = realloc_config.get("MAX_THESIS_REVIEW_AGE_HOURS", 24)
        MIN_REMAINING_RUNWAY_DAYS = realloc_config.get("MIN_REMAINING_RUNWAY_DAYS", 0)
        SIGNAL_HOLD_WINDOWS = realloc_config.get("signal_hold_windows", {})
        MIN_CONVICTION = realloc_config.get("MIN_CONVICTION", 5)
        MIN_CONVICTION_DELTA = realloc_config.get("MIN_CONVICTION_DELTA", 4.5)
        MIN_HOLD_DAYS_FOR_REALLOCATION = realloc_config.get("MIN_HOLD_DAYS_FOR_REALLOCATION", 5)

        return realloc_config
    except Exception:
        # Config file missing or invalid - use hardcoded defaults
        return {
            "MAX_REALLOCATION_LOSS_PCT": -15.0,
            "MAX_THESIS_REVIEW_AGE_HOURS": 24,
            "MIN_REMAINING_RUNWAY_DAYS": 0,
            "signal_hold_windows": {},
            "MIN_CONVICTION": 5,
            "MIN_CONVICTION_DELTA": 4.5,
            "MIN_HOLD_DAYS_FOR_REALLOCATION": 5,
            "MAX_TAX_SAVINGS_FOR_EXIT": 1000
        }


# Apply config overrides at import so the module constants reflect
# kairos_config.json from the very first call (evaluate_reallocation re-applies
# them defensively, but this guarantees correct values even for direct readers).
_load_reallocation_config()


def _load_protected_tickers() -> set[str]:
    """Tickers protected from capital liberation (config-authoritative).

    Reads dividend.protected_tickers from kairos_config.json. The reallocation
    engine must never liquidate these. HOT-IPO is exempt because it runs
    through a separate path and never calls evaluate_reallocation().
    """
    try:
        config_file = os.path.join(SCRIPT_DIR, "kairos_config.json")
        with open(config_file) as f:
            config = json.load(f)
        tickers = config.get("dividend", {}).get("protected_tickers", [])
        return {str(t).upper() for t in tickers if t}
    except Exception:
        return set()


def _get_connection() -> sqlite3.Connection:
    from kairos_log_db import get_connection
    return get_connection()


def _trading_days_between(start: datetime, end: datetime) -> int:
    """Count weekdays (Mon-Fri) between two datetimes, inclusive of partial days.

    A simple proxy for trading days (ignores market holidays). Used by the
    anti-churn minimum-hold guard.
    """
    if end < start:
        return 0
    days = 0
    cur = start.date()
    last = end.date()
    while cur <= last:
        if cur.weekday() < 5:
            days += 1
        cur += timedelta(days=1)
    # Subtract 1 so "entered today" counts as 0 trading days held.
    return max(0, days - 1)


def init_reallocation_tables() -> None:
    """Create the reallocation_events table."""
    conn = _get_connection()
    conn.executescript(SCHEMA_REALLOCATION_EVENTS)
    conn.commit()
    conn.close()


def _log_reallocation_event(
    ticker_entered: str,
    ticker_exited: str | None,
    conviction_delta: int | None,
    tax_cost: float | None,
    signals_at_exit: list[str] | None,
    reason: str,
    recommended: bool,
    executed: bool = False,
) -> int:
    """Write one row to reallocation_events. Returns row ID."""
    init_reallocation_tables()
    conn = _get_connection()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        """INSERT INTO reallocation_events
           (eval_date, ticker_exited, ticker_entered, conviction_delta,
            tax_cost, signals_at_exit, reason, recommended, executed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (today, ticker_exited, ticker_entered, conviction_delta,
         round(tax_cost, 2) if tax_cost is not None else None,
         json.dumps(signals_at_exit) if signals_at_exit else None,
         reason, 1 if recommended else 0, 1 if executed else 0),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def _get_entry_signals(ticker: str) -> list[str]:
    """Get signals that were active when this position was opened."""
    try:
        conn = _get_connection()
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


def _compute_thesis_freshness(ticker: str) -> tuple[int, list[str], list[str]]:
    """Compute thesis freshness score for an open position.

    Score = count of entry signals still active - count of reversed signals.
    Returns (score, current_signals, entry_signals).
    """
    from kairos_confluence import get_ticker_signals

    entry_signals = _get_entry_signals(ticker)
    current_signals = get_ticker_signals(ticker)

    if not entry_signals:
        # No recorded entry signals — treat as weak (score 0)
        return 0, current_signals, entry_signals

    # Count how many entry signals are still active
    still_active = sum(1 for s in entry_signals if s in current_signals)
    # Count reversed: entry signals no longer present
    reversed_count = sum(1 for s in entry_signals if s not in current_signals)

    score = still_active - reversed_count
    return score, current_signals, entry_signals


def _check_thesis_validity(ticker: str) -> bool:
    """Check thesis_reviews table for recent HOLD assessment on a ticker.
    
    Returns True if reallocation should be blocked (recent HOLD review exists),
    False if reallocation can proceed (no recent review or review triggered sell).
    """
    try:
        conn = _get_connection()
        now = datetime.now(timezone.utc)
        hours = MAX_THESIS_REVIEW_AGE_HOURS
        # Calculate cutoff time
        cutoff = now - timedelta(hours=hours)
        # Query for most recent non-dry-run review within the configured age
        row = conn.execute(
            """SELECT timestamp, trigger_type, sell_triggered
               FROM thesis_reviews
               WHERE ticker = ?
                 AND dry_run = 0
                 AND datetime(timestamp) >= datetime(?)
               ORDER BY timestamp DESC LIMIT 1""",
            (ticker, cutoff.strftime("%Y-%m-%d %H:%M:%S")),
        ).fetchone()
        conn.close()
        
        if row and row["sell_triggered"] == 0:
            # Recent review exists and sell_triggered = 0 (HOLD assessment)
            return True
    except Exception:
        pass
    return False


# NOTE: _compute_remaining_runway (the hold-window "runway" gate) was removed in
# Exit Architecture v2. Liberation eligibility is now decided purely by decayed
# conviction (see Gate 2 in evaluate_reallocation). SIGNAL_HOLD_WINDOWS remains
# loaded for the arbiter / ML thesis checkpoints / commander, which read it
# directly, but it no longer gates capital liberation.


def evaluate_reallocation(
    new_ticker: str,
    new_conviction: int,
    new_signals: list[str],
    ib,
    nlv: float,
) -> dict:
    """Evaluate whether to reallocate from the weakest position to fund a new trade.

    Returns:
        {
            "recommended": bool,
            "exit_ticker": str | None,
            "exit_qty": int,
            "exit_price": float,
            "conviction_delta": int,
            "tax_cost": float,
            "reason": str,
        }
    """
    init_reallocation_tables()

    # Re-apply config overrides so a live edit to kairos_config.json takes effect
    # without a process restart, then surface the effective gate values so the
    # engine's configuration is visible in the scheduler log every cycle.
    _load_reallocation_config()
    print(f"    [REALLOC] engine init: MIN_CONVICTION={MIN_CONVICTION}, "
          f"MIN_CONVICTION_DELTA={MIN_CONVICTION_DELTA}, "
          f"MIN_HOLD_DAYS_FOR_REALLOCATION={MIN_HOLD_DAYS_FOR_REALLOCATION}")

    # Gate 1: conviction must be high
    if new_conviction < MIN_CONVICTION:
        reason = f"Conviction {new_conviction} < {MIN_CONVICTION} minimum for reallocation"
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, None, None, None, None,
                                reason, recommended=False)
        return {"recommended": False, "reason": reason}

    # Get all open positions with freshness scores
    conn = _get_connection()
    holdings = conn.execute("""
        SELECT ticker,
               SUM(quantity) AS total_qty,
               ROUND(SUM(entry_price * quantity) / SUM(quantity), 2) AS avg_cost,
               MIN(entry_date) AS earliest_entry,
               CAST(julianday(datetime('now')) - julianday(MIN(entry_date)) AS INTEGER) AS holding_days
        FROM holdings
        WHERE sold_date IS NULL
        GROUP BY ticker
        ORDER BY ticker
    """).fetchall()
    conn.close()

    if not holdings:
        reason = "No open positions to reallocate from"
        _log_reallocation_event(new_ticker, None, None, None, None,
                                reason, recommended=False)
        return {"recommended": False, "reason": reason}

    # Score each position by DECAYED conviction (Exit Architecture v2):
    # aging signals lower a position's conviction; below the liberation
    # threshold it becomes eligible for redeployment. This replaces the old
    # thesis-freshness ranking + hold-window runway gate.
    from kairos_exits import conviction_decay_score, get_liberation_threshold
    lib_threshold = get_liberation_threshold()

    # Warm the validity cache once per evaluation (cycle-scoped, not per ticker)
    # so conviction decay can stretch/compress each position's decay window by
    # its live thesis-validity score. Degrades gracefully: a None/empty cache
    # leaves conviction_decay_score on its original time-based behavior.
    validity_cache = {}
    try:
        from kairos_thesis_validity import warm_validity_cache
        open_holdings = [dict(h) for h in holdings]
        open_tickers = [oh.get("ticker") for oh in open_holdings]
        validity_cache = warm_validity_cache(open_tickers, open_holdings)
    except Exception as exc:
        print(f"    Reallocation: validity cache warm-up failed ({exc}) — "
              f"using time-based decay")

    protected = _load_protected_tickers()
    scored = []
    for h in holdings:
        hticker = h["ticker"]
        if hticker == new_ticker:
            continue  # don't sell what we're trying to buy
        if hticker.upper() in protected:
            print(f"    Reallocation: skipping {hticker} — protected position "
                  f"(capital liberation disabled)")
            continue
        freshness, current_sigs, entry_sigs = _compute_thesis_freshness(hticker)
        conviction = conviction_decay_score(hticker, entry_sigs, h["earliest_entry"],
                                            validity_cache=validity_cache)
        scored.append({
            "ticker": hticker,
            "score": conviction,          # decayed conviction (ranking key)
            "freshness": freshness,        # retained for context/logging
            "total_qty": int(h["total_qty"]),
            "avg_cost": h["avg_cost"],
            "holding_days": h["holding_days"] or 0,
            "earliest_entry": h["earliest_entry"],
            "current_signals": current_sigs,
            "entry_signals": entry_sigs,
        })

    if not scored:
        reason = "No eligible positions to exit (all are the target ticker)"
        _log_reallocation_event(new_ticker, None, None, None, None,
                                reason, recommended=False)
        return {"recommended": False, "reason": reason}

    # Rank by decayed conviction ascending (weakest first)
    scored.sort(key=lambda x: x["score"])
    weakest = scored[0]

    print(f"    Reallocation eval: weakest={weakest['ticker']} "
          f"(decayed conviction={weakest['score']}, freshness={weakest['freshness']}, "
          f"signals={weakest['current_signals'] or 'none'})")

    # Gate 2: weakest must have decayed below the liberation threshold
    if weakest["score"] >= lib_threshold:
        reason = (f"Weakest position {weakest['ticker']} decayed conviction "
                  f"{weakest['score']} >= liberation threshold {lib_threshold} "
                  f"(conviction has not decayed enough to liberate)")
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], None, None,
                                weakest["current_signals"], reason, recommended=False)
        return {"recommended": False, "reason": reason}

    # Gate 2.4: minimum hold time — MIN_HOLD_DAYS_FOR_REALLOCATION trading days (anti-churn guard)
    entry_dt = None
    if weakest.get("earliest_entry"):
        try:
            entry_dt = datetime.strptime(
                weakest["earliest_entry"].replace(" UTC", "").strip(),
                "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            entry_dt = None
    if entry_dt is not None:
        trading_days_held = _trading_days_between(entry_dt, datetime.now(timezone.utc))
        if trading_days_held < MIN_HOLD_DAYS_FOR_REALLOCATION:
            reason = (f"Position {weakest['ticker']} held only {trading_days_held} trading "
                      f"days < {MIN_HOLD_DAYS_FOR_REALLOCATION} trading-day minimum "
                      f"(anti-churn guard)")
            print(f"    Reallocation: {reason}")
            _log_reallocation_event(new_ticker, weakest["ticker"], None, None,
                                    weakest["current_signals"], reason, recommended=False)
            return {"recommended": False, "reason": reason}

    # Gate 2.5: thesis validity check
    if _check_thesis_validity(weakest["ticker"]):
        reason = (f"Thesis review assessed {weakest['ticker']} as HOLD within last "
                  f"{MAX_THESIS_REVIEW_AGE_HOURS} hours — deferring to thesis review module")
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], None, None,
                                weakest["current_signals"], reason, recommended=False)
        return {"recommended": False, "reason": reason}

    # (Gate 2.6 "remaining thesis runway" REMOVED in Exit Architecture v2 —
    #  hold-window runway no longer gates liberation; conviction decay (Gate 2)
    #  is now the sole eligibility test.)

    # Gate 3: conviction delta
    conviction_delta = new_conviction - weakest["score"]
    if conviction_delta < MIN_CONVICTION_DELTA:
        reason = (f"Conviction delta {conviction_delta} < {MIN_CONVICTION_DELTA} "
                  f"(new={new_conviction}, weakest={weakest['score']})")
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                None, weakest["current_signals"], reason,
                                recommended=False)
        return {"recommended": False, "reason": reason}

    # Gate 3.5: unrealized loss guard (block selling deep losses)
    from kairos_execute import get_reference_price
    exit_price = get_reference_price(ib, weakest["ticker"])
    if not exit_price:
        reason = f"No price for {weakest['ticker']} — cannot evaluate unrealized loss"
        _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                None, weakest["current_signals"], reason,
                                recommended=False)
        return {"recommended": False, "reason": reason}
    
    # Calculate unrealized loss percentage
    avg_cost = weakest.get("avg_cost", 0)
    if avg_cost is None or avg_cost == 0:
        reason = f"Invalid avg_cost ({avg_cost}) for {weakest['ticker']} — cannot calculate unrealized P&L"
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                None, weakest["current_signals"], reason,
                                recommended=False)
        return {"recommended": False, "reason": reason}
    unrealized_pct = ((exit_price - avg_cost) / avg_cost) * 100
    if unrealized_pct < MAX_REALLOCATION_LOSS_PCT:
        reason = (f"Unrealized loss {unrealized_pct:.1f}% < {MAX_REALLOCATION_LOSS_PCT:.1f}% threshold "
                  f"for {weakest['ticker']} — deferring to stop-loss module")
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                None, weakest["current_signals"], reason,
                                recommended=False)
        return {"recommended": False, "reason": reason}
    
    # Gate 3.6: tax gate — defer a profitable exit near the 12-month anniversary
    # (Exit Architecture v2 overlay on capital liberation). Posts a Slack alert
    # on activation. Exempt when the position has retreated >20% from its peak.
    try:
        from kairos_exits import tax_gate_blocks_exit
        from kairos_log_db import get_peak_gain
        peak = get_peak_gain(weakest["ticker"])
        if tax_gate_blocks_exit(weakest["ticker"], weakest.get("earliest_entry"),
                                exit_price, avg_cost, peak):
            reason = (f"Tax gate: {weakest['ticker']} profitable and within "
                      f"long-term-gains window — liberation deferred to anniversary")
            print(f"    Reallocation: {reason}")
            _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                    None, weakest["current_signals"], reason,
                                    recommended=False)
            return {"recommended": False, "reason": reason}
    except Exception as tax_exc:
        print(f"    WARNING: tax gate check failed: {tax_exc}")

    # Gate 4: after-tax exit cost

    from kairos_tax_efficiency import calculate_aftertax_npv
    npv = calculate_aftertax_npv(
        weakest["avg_cost"], exit_price, weakest["total_qty"],
        weakest["holding_days"])

    tax_cost = npv["tax_savings"]  # what we'd forfeit by selling now instead of waiting
    if tax_cost > MAX_TAX_SAVINGS_FOR_EXIT:
        reason = (f"Tax cost prohibitive: ${tax_cost:,.0f} savings forfeited "
                  f"(>{MAX_TAX_SAVINGS_FOR_EXIT:,} threshold) for {weakest['ticker']}")
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                tax_cost, weakest["current_signals"], reason,
                                recommended=False)
        return {"recommended": False, "reason": reason}

    # Gate 5: wash sale checks on both legs
    try:
        from kairos_wash_sale import check_wash_sale_risk, check_wash_sale_violation

        # Check BUY leg: would buying new_ticker trigger wash sale?
        ws_buy = check_wash_sale_risk(new_ticker)
        if ws_buy["risk"]:
            reason = (f"Wash sale risk on BUY {new_ticker}: "
                      f"blocked until {ws_buy['blocked_until']}")
            print(f"    Reallocation: {reason}")
            _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                    tax_cost, weakest["current_signals"], reason,
                                    recommended=False)
            return {"recommended": False, "reason": reason}

        # Check SELL leg: would selling weakest at a loss create violation?
        if exit_price < weakest["avg_cost"]:
            ws_sell = check_wash_sale_violation(
                weakest["ticker"], exit_price, weakest["avg_cost"])
            if ws_sell["violation"]:
                reason = (f"Wash sale violation on SELL {weakest['ticker']}: "
                          f"repurchased on {ws_sell['repurchase_date']}")
                print(f"    Reallocation: {reason}")
                _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                                        tax_cost, weakest["current_signals"], reason,
                                        recommended=False)
                return {"recommended": False, "reason": reason}
    except Exception as ws_exc:
        print(f"    WARNING: Wash sale check failed: {ws_exc}")

    # All gates passed — recommend reallocation
    reason = (f"Reallocating {weakest['ticker']} (decayed conviction={weakest['score']}) "
              f"to fund {new_ticker} (conviction={new_conviction}, "
              f"delta=+{conviction_delta}, tax cost=${tax_cost:,.0f})")
    print(f"    Reallocation: RECOMMENDED — {reason}")

    _log_reallocation_event(new_ticker, weakest["ticker"], conviction_delta,
                            tax_cost, weakest["current_signals"], reason,
                            recommended=True)

    return {
        "recommended": True,
        "exit_ticker": weakest["ticker"],
        "exit_qty": weakest["total_qty"],
        "exit_price": exit_price,
        "exit_avg_cost": weakest["avg_cost"],
        "exit_holding_days": weakest["holding_days"],
        "exit_signals": weakest["current_signals"],
        "conviction_delta": conviction_delta,
        "tax_cost": tax_cost,
        "reason": reason,
        # NLV at evaluation time — used by execute_reallocation to cap the buy
        # leg at the single-position limit when sizing off freed capital.
        "nlv": nlv,
    }


def _size_reallocation_buy(freed_capital: float, ref_price: float,
                           nlv: float | None) -> tuple[int, str]:
    """Size a reallocation BUY off the freed capital, capped at the single-position limit.

    The reallocation decision is made on conviction DELTA, so once we've chosen to
    swap INTO the target we deploy the freed capital directly — independent of the
    from-scratch confluence nlv_pct (which can be ~0 for a weak-but-higher-delta
    target, and was exactly what collapsed the GD buy to 0). Quantity is
    floor(freed_capital / ref_price), then capped at max_single_position_pct of NLV
    as an upper bound. Returns (qty, human-readable detail). qty < 1 signals the
    caller to ABORT the swap (never sell when the buy can't be sized).
    """
    if ref_price is None or ref_price <= 0 or freed_capital is None or freed_capital <= 0:
        return 0, f"unsizable (freed=${freed_capital or 0:,.2f}, ref={ref_price})"
    base_qty = int(freed_capital // ref_price)
    detail = f"floor(${freed_capital:,.2f} / ${ref_price:,.2f}) = {base_qty} sh"
    # Single-position cap (max % NLV) as an upper bound when NLV is known.
    if nlv and nlv > 0:
        try:
            from kairos_confluence import load_guardrails
            max_pct = float(load_guardrails().get("max_single_position_pct", 0.10))
        except Exception as exc:
            max_pct = 0.10
            detail += f" [guardrail load failed: {exc}; default {max_pct:.0%}]"
        cap_qty = int((nlv * max_pct) // ref_price)
        if cap_qty >= 1 and base_qty > cap_qty:
            return cap_qty, detail + f", capped to {cap_qty} sh ({max_pct:.0%} of ${nlv:,.0f} NLV)"
    return base_qty, detail


def _alert_realloc_abort(exit_ticker: str, new_ticker: str, exit_qty,
                         exit_price, reason: str) -> None:
    """Loudly flag a reallocation aborted BEFORE selling (atomic invariant held)."""
    est = (exit_qty or 0) * (exit_price or 0)
    print(f"    *** REALLOCATION ABORTED ({exit_ticker} → {new_ticker}): {reason} "
          f"— NO SELL executed; position left intact.")
    try:
        from kairos_alerts import post_message
        post_message("alerts",
            f":rotating_light: *Reallocation ABORTED: {exit_ticker} → {new_ticker}*\n"
            f"Buy leg could not be sized to a valid order — *NO SELL executed* "
            f"(atomic swap invariant).\n"
            f"Reason: {reason}.\n"
            f"Would have freed ≈${est:,.0f} from {exit_ticker}; position left intact.")
    except Exception as exc:
        print(f"    reallocation abort alert failed: {exc}")


def execute_reallocation(
    realloc: dict,
    new_trade: dict,
    new_qty: int,
    decision: dict,
    ib,
) -> tuple[dict | None, dict | None]:
    """Execute both legs of a reallocation as an ATOMIC swap: SELL then BUY.

    Atomic invariant: the BUY leg is sized and validated (>= 1 share, off the
    capital the SELL would free) BEFORE any SELL is placed. If the buy cannot be
    sized, the entire reallocation aborts and NO sell happens — so a swap can
    never strand liberated cash (the 2026-06-18 ACET→GD failure mode).

    Returns (sell_execution, buy_execution); (None, None) when aborted pre-sell.
    """
    from kairos_stoploss import _place_market_sell
    from kairos_execute import execute_order, log_execution, get_reference_price

    exit_ticker = realloc["exit_ticker"]
    exit_qty = realloc["exit_qty"]
    exit_price = realloc["exit_price"]
    new_ticker = new_trade["ticker"]
    nlv = realloc.get("nlv")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Pre-flight: SIZE + VALIDATE the BUY leg BEFORE any SELL ────
    # (atomic invariant — never sell unless the buy is a valid order)
    print(f"\n    ── Reallocation pre-flight: sizing BUY {new_ticker} before any SELL ──")
    new_ref_price = get_reference_price(ib, new_ticker)
    if not new_ref_price:
        _alert_realloc_abort(exit_ticker, new_ticker, exit_qty, exit_price,
                             f"could not get reference price for {new_ticker}")
        return None, None

    est_freed = exit_qty * exit_price   # estimate; refined from the actual fill below
    planned_qty, size_detail = _size_reallocation_buy(est_freed, new_ref_price, nlv)
    print(f"    BUY sizing (pre-sell): {size_detail} → planned {planned_qty} sh "
          f"(est. freed ${est_freed:,.2f} @ ${new_ref_price:,.2f})")
    if planned_qty < 1:
        _alert_realloc_abort(
            exit_ticker, new_ticker, exit_qty, exit_price,
            f"buy sizes to {planned_qty} sh (est. freed ${est_freed:,.2f}, "
            f"{new_ticker} @ ${new_ref_price:,.2f} > freed capital)")
        return None, None

    # ── Leg 1: SELL the weak position (buy leg already validated) ──
    print(f"\n    ── Reallocation Leg 1: SELL {exit_qty} {exit_ticker} ──")
    sell_reason = (f"REALLOCATION: Reallocating to higher-conviction opportunity "
                   f"{new_ticker} (conviction delta: +{realloc['conviction_delta']})")

    sell_execution = _place_market_sell(ib, exit_ticker, exit_qty)
    if sell_execution.get("oversell_blocked"):
        # Guard refused the SELL — no order was sent, so the atomic invariant
        # holds: abort the whole swap before the buy leg, log nothing.
        _alert_realloc_abort(exit_ticker, new_ticker, exit_qty, exit_price,
                             sell_execution.get("reason", "oversell prevented"))
        return None, None
    sell_price = sell_execution.get("fill_price", realloc["exit_price"])
    if sell_price is None:
        sell_price = realloc["exit_price"]
    
    # Debug: Calculate freed capital from the sell
    freed_capital = exit_qty * sell_price
    print(f"    DEBUG: Freed capital from SELL: ${freed_capital:,.2f}")

    # Log the sell via the shared path
    from kairos_log_db import init_db, insert_decision, sell_holdings
    init_db()

    exit_avg_cost = realloc.get("exit_avg_cost", 0)
    if exit_avg_cost is None or exit_avg_cost == 0:
        print(f"    WARNING: Invalid exit_avg_cost for {exit_ticker} — using 0")
        exit_avg_cost = 0
    pnl_dollars = (sell_price - exit_avg_cost) * exit_qty
    pnl_pct = ((sell_price - exit_avg_cost) / exit_avg_cost * 100
               if exit_avg_cost > 0 else 0)

    sell_decision_id = insert_decision(
        timestamp=ts,
        ticker=exit_ticker,
        action="SELL",
        quantity=exit_qty,
        rationale=sell_reason,
        data_inputs=json.dumps({
            "sell_trigger": "REALLOCATION",
            "entry_price": realloc["exit_avg_cost"],
            "sell_price": sell_price,
            "pnl_dollars": round(pnl_dollars, 2),
            "pnl_pct": round(pnl_pct, 2),
            "conviction_delta": realloc["conviction_delta"],
            "new_ticker": new_ticker,
        }),
        execution_price=sell_price,
        execution_status=sell_execution.get("status", "Submitted"),
        commission=sell_execution.get("commission"),
    )

    sell_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    closed_lots = sell_holdings(exit_ticker, exit_qty, sell_date, sell_price,
                                sell_reason)
    print(f"    SELL {exit_ticker}: {sell_execution.get('status', '?')} "
          f"@ ${sell_price:,.2f} (P&L: ${pnl_dollars:+,.0f})")

    # Ledger entry for the sell
    try:
        from kairos_reason import record_ledger_entry, classify_signals
        sigs = classify_signals(None)
        record_ledger_entry(
            date=sell_date[:10],
            ticker=exit_ticker,
            action="SELL",
            signals=sigs,
            pnl_pct=pnl_pct,
            buy_signals=["REALLOCATION"],
        )
    except Exception as ledger_exc:
        print(f"    WARNING: reallocation ledger entry failed: {ledger_exc}")

    # Log to trade_outcomes for ML learning
    try:
        from kairos_ml_outcomes import init_db, write_trade_close, find_open_trade
        init_db()
        open_tid = find_open_trade(exit_ticker, "BUY")
        if open_tid:
            result = write_trade_close(open_tid, sell_price, timestamp_exit=sell_date)
            print(f"    ML Outcomes: recorded {exit_ticker} reallocation exit "
                  f"→ {result['outcome_label']} ({result['pnl_pct']:+.2f}%)")
        else:
            print(f"    ML Outcomes: no open {exit_ticker} BUY trade to close")
    except Exception as ml_exc:
        print(f"    WARNING: ML Outcomes logging failed: {ml_exc}")

    # ── Leg 2: BUY the new ticker — re-size off ACTUAL freed capital ──
    # The pre-flight already validated the buy off the estimated freed capital;
    # here we re-size off what the SELL actually freed (sell_price may differ from
    # the pre-sell estimate) and re-apply the single-position cap.
    final_qty, final_detail = _size_reallocation_buy(freed_capital, new_ref_price, nlv)
    print(f"\n    ── Reallocation Leg 2: BUY {new_ticker} ──")
    print(f"    BUY sizing (post-sell): {final_detail} → final {final_qty} sh "
          f"(actual freed ${freed_capital:,.2f} @ ${new_ref_price:,.2f})")
    if final_qty < 1:
        # Validated >=1 pre-sell; only an adverse price move between sizing and the
        # fill could drop this below 1. The capital is ALREADY freed, so aborting now
        # would strand it — fall back to the validated pre-sell qty and complete the buy.
        print(f"    WARNING: post-sell qty {final_qty} < 1 (price moved after sizing); "
              f"falling back to validated pre-sell qty {planned_qty}")
        final_qty = planned_qty
    print(f"    FINAL BUY QTY: {final_qty} shares")

    buy_execution = execute_order(ib, new_ticker, "BUY", final_qty)

    # Log the buy
    buy_trade = dict(new_trade)
    buy_trade["rationale"] = (
        f"Funded by REALLOCATION from {exit_ticker} "
        f"(+{realloc['conviction_delta']} conviction delta)"
    )
    # WRITE-BACK FIX: log_execution writes the holdings row from
    # trade["quantity"], but new_trade still carries the ORIGINAL pre-reallocation
    # size (often ~0 — the cash-breach round-down that triggered reallocation in
    # the first place). The order we actually placed/filled was final_qty, so
    # stamp the actual filled quantity here. Prefer the broker's reported filled
    # position; fall back to final_qty. Without this the holding was written with
    # quantity=0 despite a real fill (e.g. GL/T on 2026-06-17).
    filled_qty = final_qty
    new_pos = buy_execution.get("new_position") if buy_execution else None
    if isinstance(new_pos, dict) and new_pos.get("quantity"):
        try:
            filled_qty = int(float(new_pos["quantity"]))
        except (TypeError, ValueError):
            filled_qty = final_qty
    buy_trade["quantity"] = filled_qty
    log_execution(decision, buy_trade, buy_execution)

    print(f"    BUY {new_ticker}: {buy_execution.get('status', '?')} "
          f"@ ${buy_execution.get('fill_price', 0):,.2f}")

    # Update reallocation_events with executed=1. SQLite's UPDATE ... ORDER BY ...
    # LIMIT is rejected unless compiled with SQLITE_ENABLE_UPDATE_DELETE_LIMIT (it is
    # NOT, here) — the old form raised and was silently swallowed, so the flag never
    # flipped. Target the row id via a subquery instead, and log both paths.
    try:
        conn = _get_connection()
        cur = conn.execute(
            "UPDATE reallocation_events SET executed = 1 "
            "WHERE id = ("
            "    SELECT id FROM reallocation_events "
            "    WHERE ticker_entered = ? AND ticker_exited = ? AND recommended = 1 "
            "    ORDER BY id DESC LIMIT 1"
            ")",
            (new_ticker, exit_ticker),
        )
        conn.commit()
        matched = cur.rowcount
        conn.close()
        if matched == 1:
            print(f"    reallocation_events: executed=1 set for {exit_ticker}→{new_ticker}")
        else:
            print(f"    WARNING: reallocation_events executed-flag update matched "
                  f"{matched} row(s) for {exit_ticker}→{new_ticker} (expected 1)")
    except Exception as exc:
        print(f"    WARNING: reallocation_events executed-flag update failed: {exc}")

    # ── Slack: ONE combined #trades message for both legs ─────────────
    # A reallocation is a POSITION CHANGE, so it belongs in #trades (not
    # #alerts). Report the ACTUAL order quantity (filled_qty / final_qty), NOT
    # the stale pre-realloc new_qty — that is what produced the bogus
    # "BUY 0 ... (Skipped)" line. If the BUY leg did NOT fill, the capital from
    # the SELL leg has been liberated but not redeployed: flag it loudly here
    # AND raise a #alerts attention item (stranded cash is attention-required).
    try:
        from kairos_alerts import post_message
        sell_status = sell_execution.get("status", "?") if sell_execution else "?"
        buy_status = buy_execution.get("status", "?") if buy_execution else "Skipped"
        buy_price = (buy_execution.get("fill_price") or 0) if buy_execution else 0
        freed = exit_qty * sell_price
        buy_filled = bool(buy_execution) and buy_status == "Filled" and buy_price > 0

        msg = (
            f":arrows_counterclockwise: *Capital Reallocation — {exit_ticker} → {new_ticker}*\n"
            f"*Leg 1 — SELL:* {exit_qty} {exit_ticker} @ ${sell_price:,.2f} "
            f"({sell_status}) | P&L: ${pnl_dollars:+,.0f} ({pnl_pct:+.1f}%)\n"
            f"*Leg 2 — BUY:* {filled_qty} {new_ticker} @ ${buy_price:,.2f} "
            f"({buy_status})\n"
            f"Conviction delta: +{realloc['conviction_delta']} | "
            f"Tax cost: ${realloc['tax_cost']:,.0f}"
        )
        if not buy_filled:
            msg += (f"\n:rotating_light: *BUY leg did NOT fill* — ≈${freed:,.0f} freed "
                    f"from {exit_ticker} is now sitting in cash, NOT redeployed into "
                    f"{new_ticker} (buy returned {buy_status}, qty={filled_qty}). "
                    f"Manual review required.")
        post_message("trades", msg)

        # Stranded capital is an attention-required condition → #alerts too.
        if not buy_filled:
            post_message("alerts",
                f":rotating_light: *Reallocation BUY leg failed: {new_ticker}*\n"
                f"SELL {exit_ticker} filled ({exit_qty} @ ${sell_price:,.2f}, "
                f"≈${freed:,.0f} liberated) but the {new_ticker} BUY returned "
                f"{buy_status} (qty={filled_qty}). Capital freed but NOT redeployed "
                f"— review reallocation buy-leg sizing.")
    except Exception as exc:
        print(f"  reallocation: Slack post failed: {exc}")

    return sell_execution, buy_execution
