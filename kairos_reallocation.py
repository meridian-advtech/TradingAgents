"""
Kairos Capital Reallocation Evaluator

When a high-conviction trade is blocked by cash constraints, evaluates
whether selling the weakest existing position to fund it would be
NPV-positive after tax and transaction costs.

Activation criteria:
  - New opportunity has conviction >= 7
  - An existing position has thesis_freshness_score <= 1
  - Conviction delta >= 4.5 between new opportunity and weakest position
  - Position has been held >= 5 days (anti-churn guard)
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

# Minimum conviction to even consider reallocation (high bar)
MIN_CONVICTION = 7
# Minimum gap between new conviction and weakest position score
MIN_CONVICTION_DELTA = 4.5  # Increased by 50% from 3 to 4.5 to reduce churn
# Max tax savings from waiting before we refuse to exit (prohibitive threshold)
MAX_TAX_SAVINGS_FOR_EXIT = 1000
# Minimum hold time in days before a position can be reallocated (anti-churn guard)
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
        
        # Use config values if available, otherwise use defaults
        global MAX_REALLOCATION_LOSS_PCT, MAX_THESIS_REVIEW_AGE_HOURS, MIN_REMAINING_RUNWAY_DAYS, SIGNAL_HOLD_WINDOWS
        MAX_REALLOCATION_LOSS_PCT = realloc_config.get("MAX_REALLOCATION_LOSS_PCT", -15.0)
        MAX_THESIS_REVIEW_AGE_HOURS = realloc_config.get("MAX_THESIS_REVIEW_AGE_HOURS", 24)
        MIN_REMAINING_RUNWAY_DAYS = realloc_config.get("MIN_REMAINING_RUNWAY_DAYS", 0)
        SIGNAL_HOLD_WINDOWS = realloc_config.get("signal_hold_windows", {})
        
        return realloc_config
    except Exception:
        # Config file missing or invalid - use hardcoded defaults
        return {
            "MAX_REALLOCATION_LOSS_PCT": -15.0,
            "MAX_THESIS_REVIEW_AGE_HOURS": 24,
            "MIN_REMAINING_RUNWAY_DAYS": 0,
            "signal_hold_windows": {},
            "MIN_CONVICTION": 7,
            "MIN_CONVICTION_DELTA": 4.5,
            "MIN_HOLD_DAYS_FOR_REALLOCATION": 5,
            "MAX_TAX_SAVINGS_FOR_EXIT": 1000
        }


def _get_connection() -> sqlite3.Connection:
    from kairos_log_db import get_connection
    return get_connection()


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


def _compute_remaining_runway(weakest: dict) -> tuple[int, int]:
    """Compute remaining thesis runway for a position.
    
    Returns (remaining_runway, freshness_score) where:
    - remaining_runway: days remaining based on longest entry signal window
    - freshness_score: current thesis freshness score
    
    If entry_signals is empty or unavailable, returns (0, freshness_score).
    """
    entry_signals = weakest.get("entry_signals", [])
    if not entry_signals:
        # No entry signals - skip this gate, allow reallocation
        freshness_score = _compute_thesis_freshness(weakest["ticker"])[0]
        return 0, freshness_score
    
    # Get the maximum hold window across all entry signals
    max_window = 0
    for signal in entry_signals:
        window = SIGNAL_HOLD_WINDOWS.get(signal, SIGNAL_HOLD_WINDOWS.get("DEFAULT", 7))
        max_window = max(max_window, window)
    
    # Calculate remaining runway
    holding_days = weakest.get("holding_days", 0)
    remaining_runway = max_window - holding_days
    
    # Get current thesis freshness score
    freshness_score = _compute_thesis_freshness(weakest["ticker"])[0]
    
    return remaining_runway, freshness_score


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

    # Score each position
    scored = []
    for h in holdings:
        hticker = h["ticker"]
        if hticker == new_ticker:
            continue  # don't sell what we're trying to buy
        score, current_sigs, entry_sigs = _compute_thesis_freshness(hticker)
        scored.append({
            "ticker": hticker,
            "score": score,
            "total_qty": int(h["total_qty"]),
            "avg_cost": h["avg_cost"],
            "holding_days": h["holding_days"] or 0,
            "current_signals": current_sigs,
            "entry_signals": entry_sigs,
        })

    if not scored:
        reason = "No eligible positions to exit (all are the target ticker)"
        _log_reallocation_event(new_ticker, None, None, None, None,
                                reason, recommended=False)
        return {"recommended": False, "reason": reason}

    # Rank by thesis freshness ascending (weakest first)
    scored.sort(key=lambda x: x["score"])
    weakest = scored[0]

    print(f"    Reallocation eval: weakest={weakest['ticker']} "
          f"(freshness={weakest['score']}, signals={weakest['current_signals'] or 'none'})")

    # Gate 2: weakest must be a genuinely weak hold
    if weakest["score"] > 1:
        reason = (f"Weakest position {weakest['ticker']} has freshness score "
                  f"{weakest['score']} > 1 (not weak enough to exit)")
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], None, None,
                                weakest["current_signals"], reason, recommended=False)
        return {"recommended": False, "reason": reason}

    # Gate 2.4: minimum hold time (anti-churn guard)
    from kairos_execute import init_db
    init_db()
    import sqlite3
    conn = sqlite3.connect("kairos.db")
    cursor = conn.cursor()
    
    # Get the buy date for this position
    cursor.execute("""
        SELECT entry_date FROM holdings 
        WHERE ticker = ? AND sold_date IS NULL
        ORDER BY entry_date DESC LIMIT 1
    """, (weakest["ticker"],))
    
    buy_date_row = cursor.fetchone()
    conn.close()
    
    if buy_date_row:
        buy_date = datetime.strptime(buy_date_row[0], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
        current_date = datetime.now(timezone.utc)
        hold_days = (current_date - buy_date).days
        
        if hold_days < MIN_HOLD_DAYS_FOR_REALLOCATION:
            reason = (f"Position {weakest['ticker']} held only {hold_days} days "
                      f"< {MIN_HOLD_DAYS_FOR_REALLOCATION} day minimum (anti-churn guard)")
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

    # Gate 2.6: remaining thesis runway
    remaining_runway, freshness = _compute_remaining_runway(weakest)
    if remaining_runway > MIN_REMAINING_RUNWAY_DAYS and freshness > 0:
        reason = (
            f"Position {weakest['ticker']} is mid-thesis: "
            f"{remaining_runway} days of runway remaining, "
            f"freshness score {freshness} — reallocation blocked"
        )
        print(f"    Reallocation: {reason}")
        _log_reallocation_event(new_ticker, weakest["ticker"], None, None,
                                weakest["current_signals"], reason, recommended=False)
        return {"recommended": False, "reason": reason}

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
    reason = (f"Reallocating {weakest['ticker']} (freshness={weakest['score']}) "
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
    }


def execute_reallocation(
    realloc: dict,
    new_trade: dict,
    new_qty: int,
    decision: dict,
    ib,
) -> tuple[dict | None, dict | None]:
    """Execute both legs of a reallocation: SELL exit_ticker, BUY new_ticker.

    Returns (sell_execution, buy_execution) — either may be None on failure.
    """
    from kairos_stoploss import _place_market_sell
    from kairos_execute import execute_order, log_execution

    exit_ticker = realloc["exit_ticker"]
    exit_qty = realloc["exit_qty"]
    new_ticker = new_trade["ticker"]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Leg 1: SELL the weak position ─────────────────────────────
    print(f"\n    ── Reallocation Leg 1: SELL {exit_qty} {exit_ticker} ──")
    sell_reason = (f"REALLOCATION: Reallocating to higher-conviction opportunity "
                   f"{new_ticker} (conviction delta: +{realloc['conviction_delta']})")

    sell_execution = _place_market_sell(ib, exit_ticker, exit_qty)
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
    closed_lots = sell_holdings(exit_ticker, exit_qty, sell_date, sell_price)
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
    except Exception:
        pass

    # Log to trade_outcomes for ML learning
    try:
        from kairos_ml_outcomes import init_db, write_trade_close
        init_db()
        
        # Get the original buy timestamp from decisions table
        import sqlite3
        conn = sqlite3.connect("kairos.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp FROM decisions 
            WHERE ticker = ? AND action = 'BUY'
            ORDER BY timestamp DESC LIMIT 1
        """, (exit_ticker,))
        buy_timestamp = cursor.fetchone()
        conn.close()
        
        timestamp_entry = buy_timestamp[0] if buy_timestamp else sell_date
        
        # Write to trade_outcomes table
        write_trade_close(
            decision_id=None,  # No matching decision ID for reallocations
            fill_price=sell_price,
            timestamp_exit=sell_date,
            timestamp_entry=timestamp_entry,
            ticker=exit_ticker,
            action="SELL",
            quantity=exit_qty,
            price_entry=avg_cost,
            price_exit=sell_price,
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            signals_fired=json.dumps(["REALLOCATION"]),
            confluence_score=0,  # Reallocations don't have confluence scores
            outcome_label="REALLOCATION"
        )
        print(f"    ML Outcomes: recorded {exit_ticker} reallocation exit")
    except Exception as ml_exc:
        print(f"    WARNING: ML Outcomes logging failed: {ml_exc}")

    # ── Leg 2: BUY the new ticker ─────────────────────────────────
    print(f"\n    ── Reallocation Leg 2: BUY {new_qty} {new_ticker} ──")
    
    # Debug: Show the new_qty calculation
    print(f"    DEBUG: new_qty = {new_qty} (calculated from old NLV)")
    print(f"    DEBUG: This should be calculated from freed_capital = ${freed_capital:,.2f}")
    
    # FIX: Recalculate buy quantity using actual freed capital from the sell
    # The original new_qty was calculated from pre-sell NLV, which doesn't include
    # the freed capital. We need to recalculate based on what we actually freed.
    
    from kairos_confluence import compute_confluence, compute_position_size
    
    # Get the reference price for the new ticker
    from kairos_execute import get_reference_price
    new_ref_price = get_reference_price(ib, new_ticker)
    if not new_ref_price:
        print(f"    ERROR: Cannot get reference price for {new_ticker}")
        return sell_execution, None
    
    # Get signals for the new trade to compute confluence
    signal_tags = new_trade.get("signals", [])
    if not signal_tags and new_trade.get("conviction", 0) >= 7:
        # This is a conviction trade - use the conviction-based sizing
        # Use 1% of freed capital (matching Mode C logic)
        conviction_spend = freed_capital * 0.01
        recalculated_qty = max(1, int(conviction_spend / new_ref_price))
        print(f"    RECALCULATED: Conviction trade → 1% of ${freed_capital:,.2f} = ${conviction_spend:,.2f} → {recalculated_qty} shares")
    else:
        # This is a confluence-based trade
        confluence = compute_confluence(signal_tags)
        # Use the freed capital as the effective NLV for sizing
        recalculated_qty = compute_position_size(confluence, freed_capital, new_ref_price)
        print(f"    RECALCULATED: Confluence {confluence.get('tier', '?')} → {recalculated_qty} shares")
    
    # Use the recalculated quantity (but don't let it be smaller than original if that was valid)
    final_qty = max(new_qty, recalculated_qty) if recalculated_qty > 0 else new_qty
    print(f"    FINAL BUY QTY: {final_qty} shares (was {new_qty}, recalculated {recalculated_qty})")
    
    buy_execution = execute_order(ib, new_ticker, "BUY", final_qty)

    # Log the buy
    buy_trade = dict(new_trade)
    buy_trade["rationale"] = (
        f"Funded by REALLOCATION from {exit_ticker} "
        f"(+{realloc['conviction_delta']} conviction delta)"
    )
    log_execution(decision, buy_trade, buy_execution)

    print(f"    BUY {new_ticker}: {buy_execution.get('status', '?')} "
          f"@ ${buy_execution.get('fill_price', 0):,.2f}")

    # Update reallocation_events with executed=True
    try:
        conn = _get_connection()
        conn.execute(
            "UPDATE reallocation_events SET executed = 1 "
            "WHERE ticker_entered = ? AND ticker_exited = ? "
            "AND recommended = 1 ORDER BY id DESC LIMIT 1",
            (new_ticker, exit_ticker),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    # ── Slack alert for both legs ─────────────────────────────────
    try:
        from kairos_alerts import post_message
        sell_status = sell_execution.get("status", "?")
        buy_status = buy_execution.get("status", "?")
        buy_price = buy_execution.get("fill_price", 0)
        post_message("alerts",
            f":arrows_counterclockwise: *Capital Reallocation*\n"
            f"*Leg 1 — SELL:* {exit_qty} {exit_ticker} @ ${sell_price:,.2f} "
            f"({sell_status}) | P&L: ${pnl_dollars:+,.0f} ({pnl_pct:+.1f}%)\n"
            f"*Leg 2 — BUY:* {new_qty} {new_ticker} @ ${buy_price:,.2f} "
            f"({buy_status})\n"
            f"Conviction delta: +{realloc['conviction_delta']} | "
            f"Tax cost: ${realloc['tax_cost']:,.0f}")
    except Exception:
        pass

    return sell_execution, buy_execution
