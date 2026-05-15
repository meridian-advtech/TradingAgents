"""
Kairos Stop-Loss Monitor — Phase 0.1

Runs at the start of every equity cycle, after regime detection.
For each open position, checks drawdown against regime-adjusted
thresholds and places immediate market SELL orders if breached.

Tax-aware: overrides stop-loss if position is 340-365 days old
and drawdown is under 15% (near long-term capital gains threshold).

Thresholds by regime:
    NORMAL:        10%
    CAUTION:        8%
    RISK-OFF:       6%
    EXTREME-FEAR:   5%

Usage:
    from kairos_stoploss import run_stoploss
    result = run_stoploss()  # uses current regime from state file
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

REGIME_STATE_FILE = os.path.join(SCRIPT_DIR, ".kairos_regime_state.json")
W = 72

STOP_LOSS_THRESHOLDS = {
    "NORMAL":        10.0,
    "CAUTION":        8.0,
    "RISK-OFF":       6.0,
    "EXTREME-FEAR":   5.0,
}

# Tax efficiency override: don't sell if within 25 days of long-term
# threshold (365 days) unless drawdown exceeds this hard cap
TAX_OVERRIDE_MIN_DAYS = 340
TAX_OVERRIDE_MAX_DAYS = 365
TAX_OVERRIDE_DRAWDOWN_CAP = 15.0  # override only if drawdown < 15%


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


def _get_regime() -> str:
    try:
        with open(REGIME_STATE_FILE) as f:
            return json.load(f).get("regime", "NORMAL")
    except (IOError, json.JSONDecodeError):
        return "NORMAL"


def _get_ibkr_price(ib, ticker: str) -> float | None:
    """Get current price for a ticker via an existing IBKR connection."""
    from ib_insync import Stock
    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqMarketDataType(4)
        mkt = ib.reqMktData(contract)
        ib.sleep(2)
        price = None
        for attr in ("last", "close", "bid", "ask"):
            val = getattr(mkt, attr, None)
            if val is not None and val == val and val > 0:
                price = round(float(val), 2)
                break
        ib.cancelMktData(contract)
        return price
    except Exception:
        return None


def _place_market_sell(ib, ticker: str, qty: int) -> dict:
    """Place an immediate market SELL order."""
    from ib_insync import Stock, MarketOrder
    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("SELL", qty)
        trade = ib.placeOrder(contract, order)
        print(f"    Order ID: {trade.order.orderId}")

        timeout = 30
        start = time.time()
        while time.time() - start < timeout:
            ib.sleep(1)
            if trade.isDone():
                break

        result = {"status": trade.orderStatus.status, "order_id": trade.order.orderId}

        if trade.fills:
            fill = trade.fills[0]
            result["status"] = "Filled"
            result["fill_price"] = fill.execution.price
            result["commission"] = sum(
                f.commissionReport.commission for f in trade.fills
                if f.commissionReport.commission < 1e6
            )
            print(f"    Filled @ ${fill.execution.price:.2f}")
        else:
            result["reason"] = f"Order status: {trade.orderStatus.status}"
            print(f"    Status: {trade.orderStatus.status} — no fill yet")

        return result
    except Exception as exc:
        return {"status": "Cancelled", "reason": str(exc)}


def _log_sell(ticker: str, qty: int, entry_price: float, sell_price: float,
              holding_days: int, reason: str, execution: dict) -> None:
    """Log stop-loss sell to kairos.db (decisions + holdings)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sell_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    pnl_dollars = (sell_price - entry_price) * qty
    pnl_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    tax_class = "long-term" if holding_days >= 365 else "short-term"

    # Determine trigger type from reason string
    if "STOP-LOSS" in reason:
        trigger = "STOP-LOSS"
    elif "THESIS-INVALID" in reason:
        trigger = "THESIS-INVALID"
    elif "TAKE-PROFIT" in reason:
        trigger = "TAKE-PROFIT"
    elif "STALE-THESIS" in reason:
        trigger = "STALE-THESIS"
    else:
        trigger = "MANUAL"

    try:
        from kairos_log_db import insert_decision, sell_holdings
        decision_id = insert_decision(
            timestamp=ts,
            ticker=ticker,
            action="SELL",
            quantity=qty,
            rationale=reason,
            data_inputs=json.dumps({
                "sell_trigger": trigger,
                "entry_price": entry_price,
                "sell_price": sell_price,
                "pnl_dollars": round(pnl_dollars, 2),
                "pnl_pct": round(pnl_pct, 2),
                "holding_days": holding_days,
                "tax_classification": tax_class,
            }),
            execution_price=sell_price,
            execution_status=execution.get("status", "Submitted"),
            commission=execution.get("commission"),
        )

        closed_lots = sell_holdings(ticker, qty, sell_date, sell_price)
        print(f"    DB: decision #{decision_id}, {len(closed_lots)} lot(s) closed")

        # Ledger entry — record the closed trade for pattern analysis
        try:
            from kairos_reason import record_ledger_entry, classify_signals
            sigs = classify_signals(json.loads(
                json.dumps({"sell_trigger": trigger, "reason": reason})
            ))
            # Add trigger as a signal tag (e.g. STOP-LOSS, TAKE-PROFIT)
            record_ledger_entry(
                date=sell_date[:10],
                ticker=ticker,
                action="SELL",
                signals=sigs,
                pnl_pct=pnl_pct,
                buy_signals=[trigger],
            )
            verdict = "PASS" if pnl_pct > 0 else "FAIL"
            print(f"    Ledger: {ticker} {pnl_pct:+.2f}% {verdict} ({trigger})")
        except Exception as ledger_exc:
            print(f"    WARNING: Ledger entry failed: {ledger_exc}")
    except Exception as exc:
        print(f"    WARNING: DB logging failed: {exc}")


def _alert_stoploss(ticker: str, entry_price: float, sell_price: float,
                    drawdown_pct: float, holding_days: int, reason: str) -> None:
    """Send stop-loss alert to #kairos-alerts."""
    pnl = (sell_price - entry_price)
    tax_class = "long-term" if holding_days >= 365 else "short-term"
    try:
        from kairos_alerts import post_message
        post_message("alerts",
            f":octagonal_sign: *Stop-Loss Triggered: {ticker}*\n"
            f"Entry: ${entry_price:.2f} → Exit: ${sell_price:.2f} "
            f"({drawdown_pct:+.1f}%)\n"
            f"Loss: ${pnl:+,.2f} | Held {holding_days}d ({tax_class})\n"
            f"Reason: {reason}"
        )
    except Exception as exc:
        print(f"    WARNING: Slack alert failed: {exc}")


def run_stoploss(regime: str | None = None, ib=None) -> dict:
    """Run stop-loss check across all open equity positions.

    Args:
        regime: override regime (defaults to reading .kairos_regime_state.json)
        ib: optional existing IBKR connection to reuse (avoids opening new connection)

    Returns:
        {"checked": int, "triggered": int, "overridden": int, "sells": [...]}
    """
    if regime is None:
        regime = _get_regime()

    threshold = STOP_LOSS_THRESHOLDS.get(regime, 10.0)

    print(f"  Regime: {regime} → stop-loss threshold: {threshold}%")

    # Load open holdings
    from kairos_log_db import get_connection
    conn = get_connection()
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
        print("  No open holdings — nothing to check")
        return {"checked": 0, "triggered": 0, "overridden": 0, "sells": []}

    print(f"  Checking {len(holdings)} positions against {threshold}% threshold")

    # Connect to IBKR for live prices (reuse existing connection if provided)
    from ib_insync import IB
    import random
    
    # Use existing connection if provided and connected
    use_existing_connection = False
    if ib is not None:
        try:
            # Check if connection is active
            if hasattr(ib, 'isConnected') and ib.isConnected():
                use_existing_connection = True
                print(f"  Using existing IBKR connection (clientId={getattr(ib, 'clientId', 'unknown')})")
            else:
                print(f"  Existing IBKR connection provided but not connected — opening new connection")
        except Exception:
            print(f"  Existing IBKR connection check failed — opening new connection")
    
    if not use_existing_connection:
        ib = IB()
        try:
            ib.connect("127.0.0.1", 7497, clientId=random.randint(20, 29), timeout=10)
        except Exception as exc:
            print(f"  ERROR: IBKR connection failed: {exc}")
            return {"checked": 0, "triggered": 0, "overridden": 0, "sells": [],
                    "error": str(exc)}

    result = {"checked": 0, "triggered": 0, "overridden": 0, "sells": []}

    for h in holdings:
        ticker = h["ticker"]
        total_qty = int(h["total_qty"])
        avg_cost = h["avg_cost"]
        holding_days = h["holding_days"] or 0
        result["checked"] += 1

        current_price = _get_ibkr_price(ib, ticker)
        if current_price is None:
            print(f"    {ticker}: no price available — skipped")
            continue

        if avg_cost is None or avg_cost == 0:
            print(f"    {ticker}: invalid avg_cost ({avg_cost}) — skipped")
            continue

        drawdown_pct = (current_price - avg_cost) / avg_cost * 100

        if drawdown_pct >= -threshold:
            # No breach — position is fine
            continue

        abs_drawdown = abs(drawdown_pct)

        # Tax efficiency override: near long-term threshold
        if (TAX_OVERRIDE_MIN_DAYS <= holding_days <= TAX_OVERRIDE_MAX_DAYS
                and abs_drawdown < TAX_OVERRIDE_DRAWDOWN_CAP):
            days_to_lt = 365 - holding_days
            reason = (f"STOP-LOSS overridden: near long-term threshold "
                      f"({holding_days}d held, {days_to_lt}d to go, "
                      f"{drawdown_pct:.1f}% < {TAX_OVERRIDE_DRAWDOWN_CAP}% cap)")
            print(f"    {ticker}: {drawdown_pct:.1f}% drawdown BUT {reason}")
            result["overridden"] += 1
            continue

        # Threshold breached — execute stop-loss sell
        reason = f"STOP-LOSS: {abs_drawdown:.1f}% drawdown (threshold {threshold}% in {regime})"
        print(f"    {ticker}: {drawdown_pct:.1f}% drawdown → SELL {total_qty} shares")

        execution = _place_market_sell(ib, ticker, total_qty)
        sell_price = execution.get("fill_price", current_price)
        if sell_price is None:
            sell_price = current_price

        _log_sell(ticker, total_qty, avg_cost, sell_price, holding_days,
                  reason, execution)
        _alert_stoploss(ticker, avg_cost, sell_price, drawdown_pct,
                        holding_days, reason)

        # Wash sale violation check — loss sell with recent repurchase
        if sell_price < avg_cost:
            try:
                from kairos_wash_sale import (check_wash_sale_violation,
                                              log_wash_sale_event, flag_decision_wash_sale)
                ws = check_wash_sale_violation(ticker, sell_price, avg_cost)
                if ws["violation"]:
                    from datetime import timedelta
                    sell_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    blocked_until = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
                    loss_amt = round((avg_cost - sell_price) * total_qty, 2)
                    log_wash_sale_event(ticker, sell_date_str, sell_price, avg_cost,
                                       loss_amt, ws["repurchase_date"], blocked_until)
                    print(f"    ⚠ WASH SALE: loss of ${loss_amt:,.2f} disallowed — "
                          f"repurchased on {ws['repurchase_date']}")
                    try:
                        from kairos_alerts import post_message
                        post_message("alerts",
                            f":warning: *Wash Sale Violation: {ticker}*\n"
                            f"Loss of ${loss_amt:,.2f} is disallowed for tax purposes\n"
                            f"Recent repurchase on {ws['repurchase_date']}\n"
                            f"Sell still executed — loss cannot be claimed until {blocked_until}")
                    except Exception:
                        pass
            except Exception as ws_exc:
                print(f"    WARNING: Wash sale check failed: {ws_exc}")

        result["triggered"] += 1
        result["sells"].append({
            "ticker": ticker,
            "qty": total_qty,
            "entry_price": avg_cost,
            "exit_price": sell_price,
            "drawdown_pct": round(drawdown_pct, 1),
            "holding_days": holding_days,
            "status": execution.get("status", "?"),
        })

    ib.disconnect()

    print(f"\n  Stop-loss: {result['checked']} checked, "
          f"{result['triggered']} triggered, {result['overridden']} tax-overridden")

    return result


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kairos Stop-Loss Monitor")
    parser.add_argument("--regime", choices=list(STOP_LOSS_THRESHOLDS.keys()),
                        help="Override regime for testing")
    args = parser.parse_args()

    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS STOP-LOSS MONITOR — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    result = run_stoploss(regime=args.regime)
    print(f"\n  Result: {json.dumps(result, indent=2)}")
