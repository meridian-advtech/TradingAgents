"""
Kairos Executor — Milestone 5

Reads the latest decision from kairos_decisions.log, applies risk guards,
places the order via IBKR TWS (port 7497 paper account), confirms
execution, and logs the result back to the decisions log.

Risk guards:
  - Maximum 20 shares per trade
  - Maximum $5,000 per single trade
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from ib_insync import IB, Stock, LimitOrder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
LOG_FILE = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
LAST_DECISION_FILE = os.path.join(SCRIPT_DIR, "kairos_last_decision.txt")
TICKER = "AAPL"
W = 72

# Risk limits (base values — confluence scoring may scale these)
MAX_SHARES = 20
MAX_DOLLARS = 5_000


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Step 1: Read latest decision ────────────────────────────────────

def read_latest_decision() -> dict:
    """Parse the most recent decision from the log file.

    Prefers raw decision entries (written by Claude Code) over EXECUTION
    entries (written by the executor). Falls back to unwrapping the
    nested .decision field from EXECUTION entries if no raw decision exists.

    Validates that required fields are present and prints diagnostics.
    """
    read_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(LOG_FILE, "r") as f:
        content = f.read()

    # Extract all top-level JSON objects
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(content[start : i + 1]))
                except json.JSONDecodeError:
                    pass

    if not objects:
        raise RuntimeError("No decision found in kairos_decisions.log")

    raw_count = sum(1 for o in objects if o.get("type") != "EXECUTION" and o.get("action"))
    exec_count = sum(1 for o in objects if o.get("type") == "EXECUTION")
    print(f"  Decision log read at: {read_ts}")
    print(f"  Log entries: {len(objects)} total ({raw_count} raw decisions, {exec_count} executions)")

    decision = None
    source = None

    # Prefer the latest non-EXECUTION entry (raw decision from Claude Code)
    for obj in reversed(objects):
        if obj.get("type") != "EXECUTION" and obj.get("action"):
            decision = obj
            source = "raw decision (Claude Code reasoning)"
            break

    # Fallback: unwrap from the latest EXECUTION entry's nested .decision
    if decision is None:
        for obj in reversed(objects):
            if obj.get("type") == "EXECUTION" and "decision" in obj:
                decision = obj["decision"]
                source = f"EXECUTION wrapper (timestamp: {obj.get('timestamp', '?')})"
                # Carry over any extra fields from the wrapper
                if obj.get("execution"):
                    decision["_last_execution"] = obj["execution"]
                break

    # Last resort: return the last object as-is
    if decision is None:
        decision = objects[-1]
        source = "last entry (fallback)"

    # ── Validate and extract required fields ──────────────────────────
    required_fields = ("action", "ticker", "rationale")
    optional_fields = ("quantity", "tickers_evaluated", "runner_up")

    print(f"  Decision source: {source}")
    print(f"  Parsed decision:")
    print(f"    action:            {decision.get('action', '*** MISSING ***')}")
    print(f"    ticker:            {decision.get('ticker', '*** MISSING ***')}")
    print(f"    quantity:          {decision.get('quantity', 0)}")
    rationale = decision.get("rationale", "")
    print(f"    rationale:         {rationale[:120]}{'...' if len(rationale) > 120 else ''}")
    print(f"    tickers_evaluated: {decision.get('tickers_evaluated', [])}")
    print(f"    runner_up:         {decision.get('runner_up', '(none)')}")

    for field in required_fields:
        if not decision.get(field):
            print(f"  WARNING: Required field '{field}' is missing or empty!")

    # Ensure defaults for optional fields so downstream code doesn't break
    decision.setdefault("quantity", 0)
    decision.setdefault("tickers_evaluated", [])
    decision.setdefault("runner_up", "")

    return decision


# ── Step 2: Risk guards ────────────────────────────────────────────

def apply_risk_guards(decision: dict, ref_price: float) -> dict:
    """Enforce confluence-aware position limits. Returns adjusted decision.

    Loads the winning ticker's signal tags from kairos_signal_summary.json,
    computes a confluence score, and uses the tier's sizing multiplier to
    set max shares / max dollars.  Falls back to static limits if confluence
    data is unavailable.
    """
    action = decision.get("action", "HOLD").upper()
    qty = decision.get("quantity", 0)
    ticker = decision.get("ticker", TICKER)

    if action == "HOLD":
        return decision

    # ── Confluence scoring ────────────────────────────────────────────
    try:
        from kairos_confluence import (
            compute_confluence, apply_confluence_sizing,
            get_ticker_signals, format_confluence_for_display,
        )

        signal_tags = get_ticker_signals(ticker)
        confluence = compute_confluence(signal_tags)

        print(f"  Signals for {ticker}: {signal_tags or '(none)'}")
        print(f"  Confluence: {format_confluence_for_display(confluence)}")

        decision = apply_confluence_sizing(decision, confluence, ref_price)
        # Store confluence for downstream alert
        decision["_confluence"] = confluence

    except Exception as exc:
        print(f"  WARNING: Confluence scoring failed ({exc}) — using static limits")

        # Fallback: static risk guards
        original_qty = qty

        if qty > MAX_SHARES:
            qty = MAX_SHARES
            print(f"  RISK GUARD: Capped quantity from {original_qty} to {MAX_SHARES} shares")

        if ref_price and ref_price * qty > MAX_DOLLARS:
            qty = int(MAX_DOLLARS / ref_price)
            if qty < 1:
                print(f"  RISK GUARD: ${ref_price:.2f}/share exceeds ${MAX_DOLLARS} limit. Blocking.")
                decision["action"] = "HOLD"
                decision["quantity"] = 0
                decision["risk_guard"] = f"Single share ${ref_price:.2f} exceeds ${MAX_DOLLARS} limit"
                return decision
            print(f"  RISK GUARD: Capped to {qty} shares (${ref_price * qty:,.2f} < ${MAX_DOLLARS} limit)")

        decision["quantity"] = qty
        if qty != original_qty:
            decision["risk_guard"] = f"Adjusted from {original_qty} to {qty} shares"

    return decision


# ── Step 3: Place order via IBKR ────────────────────────────────────

def execute_order(decision: dict) -> dict:
    """Connect to IBKR, place order, wait for fill, return execution details."""
    action = decision.get("action", "HOLD").upper()
    qty = decision.get("quantity", 0)
    ticker = decision.get("ticker", TICKER)

    if action == "HOLD" or qty == 0:
        return {
            "status": "NO_ACTION",
            "reason": "HOLD decision — no order placed",
        }

    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=3, timeout=10)

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Get reference price for limit order
    ib.reqMarketDataType(4)
    mkt = ib.reqMktData(contract)
    ib.sleep(3)

    ref_price = None
    for attr in ("last", "close", "bid", "ask"):
        val = getattr(mkt, attr, None)
        if val is not None and val == val:
            ref_price = round(val, 2)
            break
    ib.cancelMktData(contract)

    # Fallback: estimate from existing positions
    if ref_price is None:
        for p in ib.positions():
            if p.contract.symbol == ticker:
                ref_price = round(p.avgCost, 2)
                break

    if ref_price is None:
        ib.disconnect()
        return {"status": "FAILED", "reason": "Could not determine reference price"}

    # Set limit slightly favorable to ensure fill
    if action == "BUY":
        limit_price = round(ref_price * 1.005, 2)  # 0.5% above
    else:
        limit_price = round(ref_price * 0.995, 2)  # 0.5% below

    print(f"  Ref price: ${ref_price}  Limit: ${limit_price}")
    print(f"  Placing {action} {qty} {ticker} @ limit ${limit_price}")

    order = LimitOrder(action, qty, limit_price, tif="GTC", outsideRth=True)
    order.overridePercentageConstraints = True
    trade = ib.placeOrder(contract, order)
    print(f"  Order ID: {trade.order.orderId}")

    # Wait for fill (up to 45 seconds)
    timeout = 45
    start = time.time()
    while time.time() - start < timeout:
        ib.sleep(1)
        if trade.isDone():
            break

    status = trade.orderStatus.status
    result = {"status": status, "order_id": trade.order.orderId}

    if trade.fills:
        fill = trade.fills[0]
        result["fill_price"] = fill.execution.price
        result["fill_time"] = str(fill.execution.time)
        result["commission"] = sum(f.commissionReport.commission for f in trade.fills
                                   if f.commissionReport.commission < 1e6)
        print(f"  Filled @ ${fill.execution.price:.2f}")
    else:
        result["reason"] = f"Order status: {status} (no fill within {timeout}s)"
        print(f"  Status: {status} — no fill yet")

    # Grab updated positions
    ib.sleep(1)
    for p in ib.positions():
        if p.contract.symbol == ticker:
            result["new_position"] = {
                "quantity": float(p.position),
                "avg_cost": round(p.avgCost, 2),
            }
            print(f"  Updated position: {p.position} shares @ ${p.avgCost:.2f}")
            break

    # Account balance after
    summary = ib.accountSummary()
    for v in summary:
        if v.tag == "NetLiquidation":
            result["net_liquidation_after"] = v.value
            break

    ib.disconnect()
    return result


# ── Step 4: Log execution result ────────────────────────────────────

def log_execution(decision: dict, execution: dict):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    entry = {
        "type": "EXECUTION",
        "timestamp": timestamp,
        "decision": {
            "action": decision.get("action", "HOLD"),
            "ticker": decision.get("ticker", TICKER),
            "quantity": decision.get("quantity", 0),
            "rationale": decision.get("rationale", ""),
            "tickers_evaluated": decision.get("tickers_evaluated", []),
            "runner_up": decision.get("runner_up", ""),
        },
        "execution": execution,
    }

    # Log to flat file
    with open(LOG_FILE, "a") as f:
        f.write("\n" + json.dumps(entry, indent=2))
        f.write("\n" + "=" * W + "\n")
    print(f"  Logged to {LOG_FILE}")

    # Write human-readable summary
    _write_last_decision(timestamp, decision, execution)

    # Log to SQLite database (include tickers_evaluated)
    try:
        from kairos_log_db import init_db, insert_decision
        init_db()  # ensure tables exist

        data_inputs = decision.get("input_summary")
        if not data_inputs and decision.get("reasoning_chain"):
            data_inputs = {"reasoning_chain": decision["reasoning_chain"]}
        if not data_inputs:
            data_inputs = {}
        # Always record which tickers were evaluated
        tickers_eval = decision.get("tickers_evaluated", [])
        if tickers_eval:
            data_inputs["tickers_evaluated"] = tickers_eval
        runner_up = decision.get("runner_up", "")
        if runner_up:
            data_inputs["runner_up"] = runner_up

        net_liq = None
        if execution.get("net_liquidation_after"):
            net_liq = float(execution["net_liquidation_after"])

        row_id = insert_decision(
            timestamp=timestamp,
            ticker=decision.get("ticker", TICKER),
            action=decision.get("action", "HOLD"),
            quantity=decision.get("quantity", 0),
            rationale=decision.get("rationale", ""),
            data_inputs=data_inputs,
            execution_price=execution.get("fill_price"),
            execution_status=execution.get("status"),
            commission=execution.get("commission"),
            net_liq_after=net_liq,
            position_after=execution.get("new_position"),
        )
        print(f"  Logged to kairos.db (decision #{row_id})")

        # Update holdings table for filled orders
        if execution.get("status") == "Filled" and execution.get("fill_price"):
            from kairos_log_db import insert_holding, sell_holdings
            action_upper = decision.get("action", "").upper()
            fill_price = execution["fill_price"]
            qty = decision.get("quantity", 0)
            tkr = decision.get("ticker", TICKER)

            if action_upper == "BUY":
                h_id = insert_holding(tkr, timestamp, fill_price, qty)
                print(f"  Holdings: recorded BUY lot #{h_id}")

                # ── ML Outcomes: record trade open ────────────────
                try:
                    from kairos_ml_outcomes import init_db as ml_init, write_trade_open
                    ml_init()
                    # Gather signal tags for this ticker
                    ml_signals = None
                    try:
                        from kairos_confluence import get_ticker_signals
                        ml_signals = get_ticker_signals(tkr)
                    except Exception:
                        pass
                    ml_confluence = None
                    if decision.get("_confluence"):
                        ml_confluence = decision["_confluence"].get("score")
                    # Sector lookup (best-effort)
                    ml_sector = _lookup_sector(tkr)
                    ml_trade_id = write_trade_open(
                        ticker=tkr,
                        action=action_upper,
                        quantity=qty,
                        price_entry=fill_price,
                        timestamp_entry=timestamp,
                        signals_fired=ml_signals if ml_signals else None,
                        confluence_score=ml_confluence,
                        sector=ml_sector,
                    )
                    print(f"  ML Outcomes: recorded trade open {ml_trade_id[:8]}...")
                except Exception as ml_exc:
                    print(f"  WARNING: ML outcomes (open) failed: {ml_exc}")

            elif action_upper == "SELL":
                closed = sell_holdings(tkr, qty, timestamp, fill_price)
                for lot in closed:
                    days = lot["holding_days"]
                    rate = "long-term" if days >= 365 else "short-term"
                    print(f"  Holdings: closed {lot['quantity']} shares "
                          f"(held {days}d, {rate})")

                    # Write ledger entry for each closed lot
                    entry_price = lot["entry_price"]
                    if entry_price and entry_price > 0:
                        pnl_pct = (fill_price - entry_price) / entry_price * 100
                        from kairos_reason import record_ledger_entry, classify_signals
                        raw_inputs = decision.get("input_summary") or decision.get("reasoning_chain")
                        if not raw_inputs and decision.get("data_inputs"):
                            raw_inputs = decision["data_inputs"]
                            if isinstance(raw_inputs, str):
                                try:
                                    raw_inputs = json.loads(raw_inputs)
                                except (json.JSONDecodeError, TypeError):
                                    raw_inputs = None
                        sigs = classify_signals(raw_inputs)
                        record_ledger_entry(
                            date=timestamp[:10],
                            ticker=tkr,
                            action="SELL",
                            signals=sigs,
                            pnl_pct=pnl_pct,
                        )
                        verdict = "PASS" if pnl_pct > 0 else "FAIL"
                        print(f"  Ledger: {tkr} {pnl_pct:+.2f}% {verdict}")

                # ── ML Outcomes: close matching open trade(s) ─────
                try:
                    from kairos_ml_outcomes import init_db as ml_init, write_trade_close, find_open_trade
                    ml_init()
                    open_tid = find_open_trade(tkr, "BUY")
                    if open_tid:
                        result = write_trade_close(open_tid, fill_price, timestamp_exit=timestamp)
                        print(f"  ML Outcomes: closed {open_tid[:8]}... → "
                              f"{result['outcome_label']} ({result['pnl_pct']:+.2f}%)")
                    else:
                        print(f"  ML Outcomes: no open BUY trade found for {tkr}")
                except Exception as ml_exc:
                    print(f"  WARNING: ML outcomes (close) failed: {ml_exc}")

    except Exception as e:
        print(f"  WARNING: DB logging failed: {e}")


def _write_last_decision(timestamp: str, decision: dict, execution: dict):
    """Overwrite kairos_last_decision.txt with a human-readable summary."""
    action = decision.get("action", "HOLD")
    ticker = decision.get("ticker", TICKER)
    qty = decision.get("quantity", 0)
    rationale = decision.get("rationale", "(none)")
    tickers_eval = decision.get("tickers_evaluated", [])
    runner_up = decision.get("runner_up", "(none)")

    # Load signal tags from screen result
    signal_lines = ""
    screen_file = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")
    signal_file = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
    signal_tags = {}
    if os.path.exists(screen_file):
        try:
            with open(screen_file) as f:
                sr = json.load(f)
            # Collect all signal categories
            for key in ("hot_reversion", "hot_earnings", "hot_rsi",
                        "hot_insider", "hot_congress", "hot"):
                for t in sr.get(key, []):
                    signal_tags.setdefault(t, []).append(key.upper().replace("_", "-"))
        except (json.JSONDecodeError, IOError):
            pass
    if os.path.exists(signal_file):
        try:
            with open(signal_file) as f:
                ss = json.load(f)
            for t, tags in ss.get("signal_tags", {}).items():
                for tag in tags:
                    signal_tags.setdefault(t, [])
                    if tag not in signal_tags[t]:
                        signal_tags[t].append(tag)
        except (json.JSONDecodeError, IOError):
            pass

    if signal_tags:
        lines = []
        for t in sorted(signal_tags):
            lines.append(f"    {t:<6} → {', '.join(signal_tags[t])}")
        signal_lines = "\n".join(lines)
    else:
        signal_lines = "    (no signal data available)"

    # Execution status
    exec_status = execution.get("status", "N/A")
    fill_price = execution.get("fill_price")
    exec_detail = f"Status: {exec_status}"
    if fill_price:
        exec_detail += f"  |  Fill price: ${fill_price:.2f}"

    summary = f"""KAIROS LAST DECISION — {timestamp}
{'=' * 72}

DECISION:      {action} {qty} {ticker}
RATIONALE:     {rationale}

EXECUTION:     {exec_detail}

{'─' * 72}
TICKERS EVALUATED ({len(tickers_eval)}):
    {', '.join(tickers_eval) if tickers_eval else '(none recorded)'}

RUNNER-UP:     {runner_up}

{'─' * 72}
SIGNALS FIRED:
{signal_lines}

{'=' * 72}
"""

    try:
        with open(LAST_DECISION_FILE, "w") as f:
            f.write(summary)
        print(f"  Summary → {LAST_DECISION_FILE}")
    except IOError as e:
        print(f"  WARNING: Could not write last decision summary: {e}")


# ── Sector lookup (best-effort from kairos_universe.json) ──────────

def _lookup_sector(ticker: str) -> str | None:
    """Return GICS sector for a ticker from kairos_universe.json, or None."""
    universe_file = os.path.join(SCRIPT_DIR, "kairos_universe.json")
    if not os.path.exists(universe_file):
        return None
    try:
        with open(universe_file) as f:
            universe = json.load(f)
        for entry in universe if isinstance(universe, list) else []:
            if entry.get("ticker") == ticker or entry.get("symbol") == ticker:
                return entry.get("sector") or entry.get("gics_sector")
        # Also check if it's a dict keyed by ticker
        if isinstance(universe, dict) and ticker in universe:
            item = universe[ticker]
            if isinstance(item, dict):
                return item.get("sector") or item.get("gics_sector")
    except (json.JSONDecodeError, IOError):
        pass
    return None


# ── main ────────────────────────────────────────────────────────────

def main():
    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS EXECUTOR — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    # Read decision
    print(banner("Reading Latest Decision"))
    decision = read_latest_decision()
    action = decision.get("action", "HOLD").upper()
    qty = decision.get("quantity", 0)
    ticker = decision.get("ticker", TICKER)
    print(f"  Action:   {action}")
    print(f"  Ticker:   {ticker}")
    print(f"  Quantity: {qty}")
    print(f"  Rationale: {decision.get('rationale', 'N/A')[:100]}...")
    eval_list = decision.get("tickers_evaluated", [])
    if eval_list:
        print(f"  Evaluated: {', '.join(eval_list)}")
    runner = decision.get("runner_up", "")
    if runner:
        print(f"  Runner-up: {runner[:80]}")

    # Record chosen ticker for bias detection
    try:
        from kairos_reason import record_chosen_ticker
        record_chosen_ticker(ticker)
        print(f"  Bias history: recorded {ticker}")
    except Exception as e:
        print(f"  WARNING: Could not record bias history: {e}")

    if action == "HOLD" or qty == 0:
        print(banner("HOLD — No Action"))
        execution = {"status": "NO_ACTION", "reason": "HOLD decision"}
        log_execution(decision, execution)
        print("\n" + "━" * W)
        print("  Kairos executor complete (no trade).")
        print("━" * W)
        return

    # Get a rough price for risk guard calculation
    print(banner("Connecting to IBKR for Price Check"))
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=4, timeout=10)
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    ib.reqMarketDataType(4)
    mkt = ib.reqMktData(contract)
    ib.sleep(3)
    ref_price = None
    for attr in ("last", "close", "bid", "ask"):
        val = getattr(mkt, attr, None)
        if val is not None and val == val:
            ref_price = round(val, 2)
            break
    if ref_price is None:
        for p in ib.positions():
            if p.contract.symbol == ticker:
                ref_price = round(p.avgCost, 2)
                break
    ib.cancelMktData(contract)
    ib.disconnect()

    if ref_price:
        print(f"  {ticker} reference price: ${ref_price}")
    else:
        print(f"  WARNING: No price found for {ticker}, using $250 estimate")
        ref_price = 250.0

    # Apply risk guards
    print(banner("Applying Risk Guards"))
    print(f"  Max shares/trade: {MAX_SHARES}")
    print(f"  Max $/trade:      ${MAX_DOLLARS:,}")
    decision = apply_risk_guards(decision, ref_price)
    action = decision.get("action", "HOLD").upper()
    qty = decision.get("quantity", 0)
    print(f"  Final: {action} {qty} {ticker}")

    if action == "HOLD" or qty == 0:
        print(banner("BLOCKED by Risk Guard"))
        execution = {"status": "BLOCKED", "reason": decision.get("risk_guard", "Risk guard triggered")}
        log_execution(decision, execution)
        # Slack alert: trade blocked
        try:
            from kairos_alerts import alert_trade_blocked
            alert_trade_blocked(decision, decision.get("risk_guard", "Risk guard"), decision.get("_confluence"))
        except Exception:
            pass
        return

    # Execute
    print(banner(f"Executing: {action} {qty} {ticker}"))
    execution = execute_order(decision)

    # Log result
    print(banner("Logging Execution"))
    log_execution(decision, execution)

    # Slack alert: trade executed (every fill — full audit trail)
    try:
        from kairos_alerts import alert_trade_executed
        from kairos_confluence import get_ticker_signals
        runner_up_text = decision.get("runner_up", "")
        # Try to get runner-up ticker's signals for context
        runner_up_signals = None
        if runner_up_text:
            # Extract ticker symbol from runner-up text (first word before ':' or space)
            runner_ticker = runner_up_text.split(":")[0].split()[0].strip(",") if runner_up_text else ""
            if runner_ticker and runner_ticker.isalpha():
                runner_up_signals = get_ticker_signals(runner_ticker)
        alert_trade_executed(
            decision, execution,
            confluence=decision.get("_confluence"),
            runner_up_signals=runner_up_signals,
        )
    except Exception as exc:
        print(f"  WARNING: Slack trade alert failed: {exc}")

    print("\n" + "━" * W)
    status = execution.get("status", "UNKNOWN")
    if execution.get("fill_price"):
        print(f"  Kairos executor complete. {action} {qty} {ticker} filled @ ${execution['fill_price']:.2f}")
    else:
        print(f"  Kairos executor complete. Status: {status}")
    print("━" * W)


if __name__ == "__main__":
    main()
