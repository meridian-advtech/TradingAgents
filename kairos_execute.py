"""
Kairos Executor — Multi-Trade, NLV-Based Position Sizing

Reads ALL recommended trades from kairos_decisions.log, applies
portfolio-level guardrails before each order, and executes via IBKR
TWS (port 7497 paper account).

Portfolio guardrails (checked before EACH order):
  - Cash reserve:       1% of NLV must remain as cash (commissions + settlement timing)
  - Sector max:         25% of NLV in any single sector
  - Single position max: 10% of NLV in any single stock

Position sizing is conviction-based (% of NLV):
  Score 1     → 1.0% NLV   (~$10K at $1M)
  Score 2–3   → 1.5% NLV   (~$15K)
  Score 4–5   → 2.0% NLV   (~$20K)
  Score 6+    → 2.5% NLV   (~$25K)
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone

from ib_insync import IB, Stock, LimitOrder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
LOG_FILE = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
LAST_DECISION_FILE = os.path.join(SCRIPT_DIR, "kairos_last_decision.txt")
REGIME_STATE_FILE = os.path.join(SCRIPT_DIR, ".kairos_regime_state.json")
W = 72


# ── Regime guardrail loader ──────────────────────────────────────────

def _load_regime_guardrails() -> dict:
    """Load current regime guardrails from .kairos_regime_state.json.

    Returns the guardrails dict for the current regime, or NORMAL defaults
    if the state file is missing or unreadable.
    """
    from kairos_regime import REGIMES

    try:
        with open(REGIME_STATE_FILE) as f:
            state = json.load(f)
        regime = state.get("regime", "NORMAL")
        if regime in REGIMES:
            return {"regime": regime, **REGIMES[regime]["guardrails"]}
    except (IOError, json.JSONDecodeError, KeyError):
        pass

    return {"regime": "NORMAL", **REGIMES["NORMAL"]["guardrails"]}


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Step 1: Read trades ────────────────────────────────────────────

def read_latest_decision() -> dict:
    """Parse the most recent raw decision from the log file.

    Now expects a JSON object with a "trades" array (multi-trade format)
    or falls back to single-trade format for backwards compatibility.
    """
    read_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(LOG_FILE, "r") as f:
        content = f.read()

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

    raw_count = sum(1 for o in objects if o.get("type") != "EXECUTION" and (o.get("action") or o.get("trades")))
    exec_count = sum(1 for o in objects if o.get("type") == "EXECUTION")
    print(f"  Decision log read at: {read_ts}")
    print(f"  Log entries: {len(objects)} total ({raw_count} raw decisions, {exec_count} executions)")

    decision = None
    source = None

    # Prefer latest non-EXECUTION entry
    for obj in reversed(objects):
        if obj.get("type") != "EXECUTION" and (obj.get("action") or obj.get("trades")):
            decision = obj
            source = "raw decision (Claude Code reasoning)"
            break

    # Fallback: unwrap from EXECUTION wrapper
    if decision is None:
        for obj in reversed(objects):
            if obj.get("type") == "EXECUTION" and "decision" in obj:
                decision = obj["decision"]
                source = f"EXECUTION wrapper (timestamp: {obj.get('timestamp', '?')})"
                break

    if decision is None:
        decision = objects[-1]
        source = "last entry (fallback)"

    print(f"  Decision source: {source}")

    # Ensure defaults
    decision.setdefault("quantity", 0)
    decision.setdefault("tickers_evaluated", [])
    decision.setdefault("runner_up", "")

    return decision


def extract_trades(decision: dict) -> list[dict]:
    """Extract the list of trades from a decision.

    Supports both multi-trade format ({"trades": [...]}) and
    legacy single-trade format ({"action": "BUY", "ticker": "X", ...}).
    """
    if "trades" in decision and isinstance(decision["trades"], list):
        return decision["trades"]

    # Legacy single-trade format
    if decision.get("action") and decision.get("ticker"):
        return [{
            "action": decision["action"],
            "ticker": decision["ticker"],
            "quantity": decision.get("quantity", 0),
            "rationale": decision.get("rationale", ""),
            "sector": decision.get("sector", ""),
            "conviction": decision.get("conviction", ""),
        }]

    return []


# ── Step 2: Portfolio state ────────────────────────────────────────

def fetch_portfolio_state(ib: IB) -> dict:
    """Fetch current NLV, cash, positions, and sector exposures from IBKR.

    Returns:
        {
            "nlv": float,
            "cash": float,
            "positions": {ticker: {"qty": float, "avg_cost": float, "market_value": float}},
            "sector_exposure": {sector: float},  # total market value per sector
        }
    """
    from kairos_confluence import lookup_sector

    summary = ib.accountSummary()
    acct = {v.tag: float(v.value) for v in summary
            if v.tag in ("NetLiquidation", "TotalCashValue")}

    nlv = acct.get("NetLiquidation", 0)
    cash = acct.get("TotalCashValue", 0)

    positions: dict[str, dict] = {}
    sector_exposure: dict[str, float] = {}

    for p in ib.positions():
        sym = p.contract.symbol
        qty = float(p.position)
        avg = float(p.avgCost)
        mkt_val = abs(qty * avg)  # approximate market value
        positions[sym] = {"qty": qty, "avg_cost": round(avg, 2), "market_value": round(mkt_val, 2)}

        sector = lookup_sector(sym)
        sector_exposure[sector] = sector_exposure.get(sector, 0) + mkt_val

    return {
        "nlv": nlv,
        "cash": cash,
        "positions": positions,
        "sector_exposure": sector_exposure,
    }


def get_reference_price(ib: IB, ticker: str) -> float | None:
    """Get current price for a ticker via IBKR."""
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
                price = round(val, 2)
                break
        ib.cancelMktData(contract)
        return price
    except Exception:
        return None


# ── Step 3: Pre-execution guardrail check ──────────────────────────

def check_guardrails(
    ticker: str,
    proposed_spend: float,
    portfolio: dict,
) -> tuple[bool, list[str]]:
    """Run all portfolio guardrails before placing an order.

    Returns (all_passed, list_of_reasons).
    Reasons are empty if all passed, otherwise each failed check is listed.
    """
    from kairos_confluence import (
        lookup_sector, load_guardrails,
        check_cash_reserve, check_sector_concentration, check_single_position,
    )

    nlv = portfolio["nlv"]
    cash = portfolio["cash"]
    guardrails = load_guardrails()
    failures: list[str] = []

    # 1. Cash reserve
    ok, reason = check_cash_reserve(cash, nlv, proposed_spend, guardrails)
    if not ok:
        failures.append(reason)

    # 2. Sector concentration
    sector = lookup_sector(ticker)
    current_sector_val = portfolio["sector_exposure"].get(sector, 0)
    ok, reason = check_sector_concentration(sector, current_sector_val, proposed_spend, nlv, guardrails)
    if not ok:
        failures.append(reason)

    # 3. Single position
    current_pos_val = portfolio["positions"].get(ticker, {}).get("market_value", 0)
    ok, reason = check_single_position(ticker, current_pos_val, proposed_spend, nlv, guardrails)
    if not ok:
        failures.append(reason)

    return len(failures) == 0, failures


# ── Step 4: Place order ────────────────────────────────────────────

def execute_order(ib: IB, ticker: str, action: str, qty: int) -> dict:
    """Place a single order via IBKR. Returns execution details."""
    if action == "HOLD" or qty == 0:
        return {"status": "Skipped", "reason": "HOLD decision"}

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    ib.reqMarketDataType(4)
    mkt = ib.reqMktData(contract)
    ib.sleep(2)

    ref_price = None
    for attr in ("last", "close", "bid", "ask"):
        val = getattr(mkt, attr, None)
        if val is not None and val == val and val > 0:
            ref_price = round(val, 2)
            break
    ib.cancelMktData(contract)

    if not ref_price:
        return {"status": "Cancelled", "reason": "Could not determine reference price"}

    if action == "BUY":
        limit_price = round(ref_price * 1.005, 2)
    else:
        limit_price = round(ref_price * 0.995, 2)

    print(f"    Ref: ${ref_price}  Limit: ${limit_price}")
    print(f"    Placing {action} {qty} {ticker} @ limit ${limit_price}")

    order = LimitOrder(action, qty, limit_price, tif="GTC", outsideRth=True)
    order.overridePercentageConstraints = True
    trade = ib.placeOrder(contract, order)
    print(f"    Order ID: {trade.order.orderId}")

    timeout = 45
    start = time.time()
    while time.time() - start < timeout:
        ib.sleep(1)
        if trade.isDone():
            break

    result = {"status": trade.orderStatus.status, "order_id": trade.order.orderId}

    if trade.fills:
        fill = trade.fills[0]
        result["fill_price"] = fill.execution.price
        result["fill_time"] = str(fill.execution.time)
        result["commission"] = sum(f.commissionReport.commission for f in trade.fills
                                   if f.commissionReport.commission < 1e6)
        print(f"    Filled @ ${fill.execution.price:.2f}")
    else:
        result["reason"] = f"Order status: {trade.orderStatus.status} (no fill within {timeout}s)"
        print(f"    Status: {trade.orderStatus.status} — no fill yet")

    # Grab updated position
    ib.sleep(1)
    for p in ib.positions():
        if p.contract.symbol == ticker:
            result["new_position"] = {
                "quantity": float(p.position),
                "avg_cost": round(p.avgCost, 2),
            }
            break

    summary = ib.accountSummary()
    for v in summary:
        if v.tag == "NetLiquidation":
            result["net_liquidation_after"] = v.value
            break

    return result


# ── Step 5: Logging ────────────────────────────────────────────────

def log_execution(decision: dict, trade: dict, execution: dict,
                   conviction_trade: bool = False):
    """Log a single trade execution to file, DB, and Slack."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ticker = trade.get("ticker", "?")
    action = trade.get("action", "HOLD")
    qty = trade.get("quantity", 0)

    entry = {
        "type": "EXECUTION",
        "timestamp": timestamp,
        "decision": {
            "action": action,
            "ticker": ticker,
            "quantity": qty,
            "rationale": trade.get("rationale", ""),
            "sector": trade.get("sector", ""),
            "tickers_evaluated": decision.get("tickers_evaluated", []),
            "conviction_only": trade.get("conviction_only", False),
        },
        "execution": execution,
    }

    with open(LOG_FILE, "a") as f:
        f.write("\n" + json.dumps(entry, indent=2))
        f.write("\n" + "=" * W + "\n")
    print(f"    Logged to {LOG_FILE}")

    # SQLite
    closed_lots = []  # populated by SELL branch for Slack P&L details
    try:
        from kairos_log_db import init_db, insert_decision, insert_holding, sell_holdings
        init_db()

        conf = trade.get("_confluence", {})
        data_inputs = {"sector": trade.get("sector", ""), "confluence": conf}

        net_liq = None
        if execution.get("net_liquidation_after"):
            net_liq = float(execution["net_liquidation_after"])

        row_id = insert_decision(
            timestamp=timestamp,
            ticker=ticker,
            action=action,
            quantity=qty,
            rationale=trade.get("rationale", ""),
            data_inputs=data_inputs,
            execution_price=execution.get("fill_price"),
            execution_status=execution.get("status"),
            commission=execution.get("commission"),
            net_liq_after=net_liq,
            position_after=execution.get("new_position"),
            conviction_trade=conviction_trade,
        )
        print(f"    DB: decision #{row_id}")

        if execution.get("status") == "Filled" and execution.get("fill_price"):
            fill_price = execution["fill_price"]
            action_upper = action.upper()
            if action_upper == "BUY":
                h_id = insert_holding(ticker, timestamp, fill_price, qty)
                print(f"    Holdings: BUY lot #{h_id}")

                # ML Outcomes: record trade open
                try:
                    from kairos_ml_outcomes import init_db as ml_init, write_trade_open
                    ml_init()
                    ml_signals = None
                    try:
                        from kairos_confluence import get_ticker_signals
                        ml_signals = get_ticker_signals(ticker)
                    except Exception:
                        pass
                    ml_confluence = conf.get("score") if conf else None
                    ml_sector = trade.get("sector", _lookup_sector(ticker))
                    ml_trade_id = write_trade_open(
                        ticker=ticker,
                        action=action_upper,
                        quantity=qty,
                        price_entry=fill_price,
                        timestamp_entry=timestamp,
                        signals_fired=ml_signals if ml_signals else None,
                        confluence_score=ml_confluence,
                        sector=ml_sector,
                    )
                    print(f"    ML Outcomes: trade open {ml_trade_id[:8]}...")
                except Exception as ml_exc:
                    print(f"    WARNING: ML outcomes (open) failed: {ml_exc}")

            elif action_upper == "SELL":
                closed_lots = sell_holdings(ticker, qty, timestamp, fill_price)
                closed = closed_lots
                for lot in closed:
                    days = lot["holding_days"]
                    rate = "long-term" if days >= 365 else "short-term"
                    print(f"    Holdings: closed {lot['quantity']} shares (held {days}d, {rate})")

                # Ledger entry for pattern analysis
                if closed_lots:
                    try:
                        from kairos_reason import record_ledger_entry, classify_signals
                        raw_inputs = data_inputs
                        sigs = classify_signals(raw_inputs)
                        # Include buy_signals from confluence if available
                        buy_sigs = conf.get("signals", []) if conf else []
                        
                        # Record individual ledger entry for each lot (not aggregated)
                        for i, lot in enumerate(closed_lots):
                            entry_price = lot["entry_price"]
                            pnl_pct = (fill_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                            record_ledger_entry(
                                date=timestamp[:10],
                                ticker=ticker,
                                action="SELL",
                                signals=sigs,
                                pnl_pct=pnl_pct,
                                buy_signals=buy_sigs,
                            )
                            verdict = "PASS" if pnl_pct > 0 else "FAIL"
                            print(f"    Ledger: {ticker} lot {i+1}/{len(closed_lots)} {pnl_pct:+.2f}% {verdict}")
                    except Exception as ledger_exc:
                        print(f"    WARNING: Ledger entry failed: {ledger_exc}")

                # ML Outcomes: close matching trade
                try:
                    from kairos_ml_outcomes import init_db as ml_init, write_trade_close, find_open_trade
                    ml_init()
                    open_tid = find_open_trade(ticker, "BUY")
                    if open_tid:
                        result = write_trade_close(open_tid, fill_price, timestamp_exit=timestamp)
                        print(f"    ML Outcomes: closed {open_tid[:8]}... → "
                              f"{result['outcome_label']} ({result['pnl_pct']:+.2f}%)")
                except Exception as ml_exc:
                    print(f"    WARNING: ML outcomes (close) failed: {ml_exc}")

    except Exception as e:
        print(f"    WARNING: DB logging failed: {e}")

    # Slack alert — only for trades that actually execute, not skips
    exec_status = (execution.get("status") or "").lower()
    if exec_status in ("filled", "submitted"):
        try:
            from kairos_alerts import alert_trade_executed
            alert_decision = {"action": action, "ticker": ticker, "quantity": qty,
                              "rationale": trade.get("rationale", ""),
                              "runner_up": "",
                              "conviction": trade.get("conviction", 0)}
            alert_trade_executed(
                decision=alert_decision,
                execution=execution,
                confluence=trade.get("_confluence"),
                conviction_trade=conviction_trade,
                closed_lots=closed_lots if action.upper() == "SELL" else None,
            )
        except Exception:
            pass


# ── Sector lookup (best-effort from kairos_universe.json) ──────────

def _lookup_sector(ticker: str) -> str | None:
    """Return sector for a ticker from kairos_universe.json, or None."""
    try:
        from kairos_confluence import lookup_sector
        s = lookup_sector(ticker)
        return s if s != "unknown" else None
    except Exception:
        return None


# ── main ────────────────────────────────────────────────────────────

def main():
    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS EXECUTOR — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    # ── Read decision ─────────────────────────────────────────────
    print(banner("Reading Decision"))
    decision = read_latest_decision()
    trades = extract_trades(decision)

    if not trades:
        print("  No actionable trades in decision.")
        return

    buy_trades = [t for t in trades if t.get("action", "").upper() == "BUY"]
    sell_trades = [t for t in trades if t.get("action", "").upper() == "SELL"]
    hold_trades = [t for t in trades if t.get("action", "").upper() == "HOLD"]

    print(f"  Trades: {len(buy_trades)} BUY, {len(sell_trades)} SELL, {len(hold_trades)} HOLD")
    for t in trades:
        action = t.get("action", "?")
        ticker = t.get("ticker", "?")
        sector = t.get("sector", "?")
        conviction = t.get("conviction", "?")
        print(f"    {action:<4} {ticker:<6} sector={sector:<20} conviction={conviction}")

    actionable = [t for t in trades if t.get("action", "").upper() in ("BUY", "SELL")]
    if not actionable:
        print(banner("HOLD — No Action"))
        execution = {"status": "Skipped", "reason": "All trades are HOLD"}
        log_execution(decision, trades[0] if trades else {"action": "HOLD", "ticker": "NONE"}, execution)
        return

    # ── Regime guardrails (hard backstop) ────────────────────────
    regime_g = _load_regime_guardrails()
    regime_name = regime_g["regime"]
    if regime_name != "NORMAL":
        print(f"\n  Regime: {regime_name} — guardrails active")

    # Gate 1: EXTREME-FEAR equity buy halt — reject all BUYs immediately
    if not regime_g["equity_buys_allowed"]:
        halted_buys = [t for t in actionable if t.get("action", "").upper() == "BUY"]
        if halted_buys:
            print(banner(f"Regime Gate: {regime_name} — Equity Buy Halt"))
            reason = f"Regime gate: {regime_name} equity halt"
            for t in halted_buys:
                ticker = t.get("ticker", "?")
                print(f"    REJECTED: BUY {ticker} — {reason}")
                log_execution(decision, t, {"status": "Skipped", "reason": reason})
            # Remove halted buys from actionable list
            actionable = [t for t in actionable if t.get("action", "").upper() != "BUY"]
            if not actionable:
                print("  All BUY trades halted by regime — nothing to execute.")
                return

    # ── Connect to IBKR ──────────────────────────────────────────
    print(banner("Connecting to IBKR"))
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=random.randint(50, 59), timeout=10)

    # ── Fetch portfolio state ─────────────────────────────────────
    portfolio = fetch_portfolio_state(ib)
    nlv = portfolio["nlv"]
    cash = portfolio["cash"]
    print(f"  NLV:  ${nlv:>14,.2f}")
    print(f"  Cash: ${cash:>14,.2f}")
    print(f"  Positions: {len(portfolio['positions'])}")
    if portfolio["sector_exposure"]:
        print(f"  Sector exposure:")
        for sector, val in sorted(portfolio["sector_exposure"].items(), key=lambda x: -x[1]):
            pct = val / nlv * 100 if nlv > 0 else 0
            print(f"    {sector:<25} ${val:>12,.0f}  ({pct:.1f}%)")

    # ── Execute each trade with pre-execution guardrail check ─────
    executed_count = 0
    skipped_count = 0
    total_spent = 0.0
    skip_reasons: list[dict] = []  # [{ticker, reason}] for cycle summary
    conviction_taken = False          # Mode C: only one conviction trade per cycle
    conviction_sectors: set[str] = set()  # Mode C: one per sector per cycle

    for i, trade in enumerate(actionable, 1):
        action = trade.get("action", "HOLD").upper()
        ticker = trade.get("ticker", "?")

        print(banner(f"Trade {i}/{len(actionable)}: {action} {ticker}"))

        # Get reference price
        ref_price = get_reference_price(ib, ticker)
        if not ref_price:
            print(f"    SKIP: No price available for {ticker}")
            skipped_count += 1
            skip_reasons.append({"ticker": ticker, "reason": "No price available"})
            log_execution(decision, trade, {"status": "Skipped", "reason": "No price available"})
            continue

        print(f"    Price: ${ref_price}")

        # Compute confluence-based quantity
        from kairos_confluence import (
            get_ticker_signals, compute_confluence, compute_position_size,
            format_confluence_for_display, lookup_sector,
        )

        signal_tags = get_ticker_signals(ticker)
        confluence = compute_confluence(signal_tags)
        qty = compute_position_size(confluence, nlv, ref_price)

        # Gate 2: Apply max_position_mult from regime guardrails
        pos_mult = regime_g.get("max_position_mult", 1.0)
        if action == "BUY" and pos_mult < 1.0 and qty > 0:
            original_qty = qty
            qty = max(1, int(qty * pos_mult))
            if qty != original_qty:
                print(f"    Regime sizing: {regime_name} → {int(pos_mult*100)}% cap "
                      f"({original_qty} → {qty} shares)")

        trade["_confluence"] = {
            "score": confluence["score"],
            "tier": confluence["tier"],
            "nlv_pct": confluence["nlv_pct"],
            "signals": list(confluence["signals_detail"].keys()),
        }
        trade["sector"] = trade.get("sector") or lookup_sector(ticker)

        print(f"    Signals: {signal_tags or '(none)'}")
        print(f"    Confluence: {format_confluence_for_display(confluence)}")
        print(f"    Conviction size: {qty} shares (${qty * ref_price:,.0f})")

        # Gate 3: Blocked sector check from regime guardrails
        if action == "BUY":
            blocked = regime_g.get("blocked_sectors", [])
            trade_sector = (trade.get("sector") or "").lower().replace(" ", "_")
            if blocked and trade_sector in [s.lower() for s in blocked]:
                reason = f"Regime gate: sector {trade['sector']!r} blocked in {regime_name}"
                print(f"    REJECTED: {reason}")
                skipped_count += 1
                skip_reasons.append({"ticker": ticker, "reason": reason})
                log_execution(decision, trade, {"status": "Skipped", "reason": reason})
                continue

        # ── Mode C: Conviction trade path (zero signals, high conviction) ──
        is_conviction_trade = False
        if qty < 1:
            conviction_score = trade.get("conviction", 0)
            if isinstance(conviction_score, str):
                try:
                    conviction_score = int(conviction_score)
                except (ValueError, TypeError):
                    conviction_score = 0
            sector = trade.get("sector") or ""

            # Gate 4: Mode C disabled by regime
            mode_c_allowed = regime_g.get("mode_c_allowed", True)
            mode_c_min = regime_g.get("mode_c_min_conviction", 4)

            can_conviction = (
                mode_c_allowed
                and conviction_score >= max(7, mode_c_min)
                and confluence["score"] == 0
                and not conviction_taken
                and sector.lower() not in conviction_sectors
            )

            if can_conviction:
                # Mode C: size at 1% NLV max
                conviction_spend = nlv * 0.01
                qty = max(1, int(conviction_spend / ref_price))
                is_conviction_trade = True
                trade["conviction_only"] = True
                trade["_confluence"]["conviction_trade"] = True
                print(f"    CONVICTION-ONLY: No signals, but conviction={conviction_score}/10")
                print(f"    Mode C sizing: 1% NLV = ${conviction_spend:,.0f} → {qty} shares")
            else:
                reason = "Quantity rounds to 0"
                if not mode_c_allowed and conviction_score >= 7:
                    reason = f"Regime gate: Mode C disabled in {regime_name}"
                elif conviction_score >= 7 and conviction_score < mode_c_min:
                    reason = f"Regime gate: conviction {conviction_score} < {mode_c_min} min in {regime_name}"
                elif conviction_score >= 7 and conviction_taken:
                    reason = "Conviction trade already taken this cycle"
                elif conviction_score >= 7 and sector.lower() in conviction_sectors:
                    reason = f"Conviction trade already taken in sector {sector}"
                print(f"    SKIP: {reason}")
                skipped_count += 1
                skip_reasons.append({"ticker": ticker, "reason": reason})
                log_execution(decision, trade, {"status": "Skipped", "reason": reason})
                continue

        proposed_spend = qty * ref_price

        # ── Pre-execution portfolio check ─────────────────────────
        # Refresh portfolio state before each order
        portfolio = fetch_portfolio_state(ib)

        passed, failures = check_guardrails(ticker, proposed_spend, portfolio)

        if not passed:
            print(f"    GUARDRAIL BREACH — skipping {ticker}:")
            for reason in failures:
                print(f"      • {reason}")

            # ── Reallocation check: cash reserve breach on a BUY ─────
            has_cash_breach = any("cash reserve" in f.lower() for f in failures)
            new_conviction = trade.get("conviction", 0)
            if isinstance(new_conviction, str):
                try:
                    new_conviction = int(new_conviction)
                except (ValueError, TypeError):
                    new_conviction = 0

            if action == "BUY" and has_cash_breach and new_conviction >= 7:
                try:
                    from kairos_reallocation import evaluate_reallocation, execute_reallocation
                    realloc = evaluate_reallocation(
                        ticker, new_conviction, signal_tags, ib, nlv)

                    if realloc["recommended"]:
                        sell_exec, buy_exec = execute_reallocation(
                            realloc, trade, qty, decision, ib)
                        if buy_exec and buy_exec.get("status") in ("Filled", "Submitted"):
                            executed_count += 1
                            if buy_exec.get("fill_price"):
                                total_spent += qty * buy_exec["fill_price"]
                            continue  # skip the normal skip path — trade was reallocated
                        else:
                            print(f"    Reallocation BUY leg failed — falling through to skip")
                except Exception as realloc_exc:
                    print(f"    WARNING: Reallocation eval failed: {realloc_exc}")

            skipped_count += 1
            skip_reasons.append({"ticker": ticker, "reason": "; ".join(failures)})
            log_execution(decision, trade, {
                "status": "Skipped",
                "reason": "; ".join(failures),
            })
            try:
                from kairos_alerts import alert_trade_blocked
                alert_trade_blocked(
                    {"action": action, "ticker": ticker, "quantity": qty},
                    "; ".join(failures),
                    trade.get("_confluence"),
                )
            except Exception:
                pass
            continue

        print(f"    Guardrails: ALL PASSED")

        # ── Wash sale check (BUY only) ───────────────────────────
        if action == "BUY":
            try:
                from kairos_wash_sale import check_wash_sale_risk
                ws = check_wash_sale_risk(ticker)
                if ws["risk"]:
                    reason = (f"WASH-SALE: repurchase within 30 days of loss sale "
                              f"on {ticker}, blocked until {ws['blocked_until']}")
                    print(f"    BLOCKED: {reason}")
                    print(f"    Loss sale: ${ws['entry_price']:.2f} → ${ws['sell_price']:.2f} "
                          f"on {ws['sell_date']} (loss ${ws['loss_amount']:,.2f})")
                    skipped_count += 1
                    skip_reasons.append({"ticker": ticker, "reason": reason})
                    log_execution(decision, trade, {"status": "Skipped", "reason": reason})
                    try:
                        from kairos_alerts import post_message
                        post_message("alerts",
                            f":no_entry: *Wash Sale Block: {ticker}*\n"
                            f"BUY blocked — loss sale of ${ws['loss_amount']:,.2f} on {ws['sell_date']}\n"
                            f"Blocked until: {ws['blocked_until']}")
                    except Exception:
                        pass
                    continue
            except Exception as ws_exc:
                print(f"    WARNING: Wash sale check failed: {ws_exc}")

        # ── Cash floor check (BUY only) ───────────────────────────
        if action == "BUY":
            available_cash = portfolio["cash"]
            order_cost = qty * ref_price
            if order_cost > available_cash:
                # Try to reduce quantity to fit within available cash
                max_affordable = int(available_cash // ref_price)
                if max_affordable >= 1:
                    original_qty = qty
                    qty = max_affordable
                    print(f"    CASH FLOOR: Reduced quantity from {original_qty} to {qty} shares to fit cash")
                else:
                    reason = f"SKIPPED {ticker}: order cost ${order_cost:.0f} exceeds available cash ${available_cash:.0f}"
                    print(f"    {reason}")
                    skipped_count += 1
                    skip_reasons.append({"ticker": ticker, "reason": "INSUFFICIENT_CASH"})
                    log_execution(decision, trade, {"status": "Skipped", "reason": "INSUFFICIENT_CASH"})
                    continue

        if is_conviction_trade:
            print(f"    ★ CONVICTION-ONLY trade — Mode C (1% NLV cap, no signals)")

        # ── Place order ───────────────────────────────────────────
        trade["quantity"] = qty
        execution = execute_order(ib, ticker, action, qty)
        log_execution(decision, trade, execution, conviction_trade=is_conviction_trade)
        executed_count += 1

        # Track conviction state for cycle-level limits
        if is_conviction_trade:
            conviction_taken = True
            sector = (trade.get("sector") or "").lower()
            if sector:
                conviction_sectors.add(sector)

        if execution.get("fill_price"):
            total_spent += qty * execution["fill_price"]

    # ── Disconnect and summary ────────────────────────────────────
    ib.disconnect()

    print("\n" + "━" * W)
    print(f"  EXECUTOR COMPLETE")
    print(f"  Executed: {executed_count}  Skipped: {skipped_count}  Total: {len(actionable)}")
    if total_spent > 0:
        print(f"  Capital deployed: ${total_spent:,.0f}")
    print("━" * W)

    # ── End-of-cycle summary → #kairos-reports ────────────────────
    try:
        from kairos_alerts import post_message

        # Bucket skip reasons into three categories for the summary line.
        # cash_reserve: guardrail blocked due to insufficient cash
        # no_signal:    position size computed to 0 (no HOT signals)
        # sizing:       everything else (sector/position limits, regime gates, etc.)
        cash_reserve_n = sum(
            1 for s in skip_reasons if "cash reserve" in s["reason"].lower()
        )
        no_signal_n = sum(
            1 for s in skip_reasons if s["reason"] == "Quantity rounds to 0"
        )
        sizing_n = len(skip_reasons) - cash_reserve_n - no_signal_n

        evaluated = decision.get("tickers_evaluated", [])
        summary_text = (
            f":clipboard: *Kairos Execution Cycle Summary*\n"
            f"Evaluated: {len(evaluated)} ticker(s)\n"
            f"Executed: {executed_count}  |  Skipped: {skipped_count}  |  "
            f"Total actionable: {len(actionable)}\n"
        )
        if total_spent > 0:
            summary_text += f"Capital deployed: ${total_spent:,.0f}\n"
        if skip_reasons:
            summary_text += (
                f"Skipped: {skipped_count} trades "
                f"(cash reserve: {cash_reserve_n}, sizing: {sizing_n}, no signal: {no_signal_n})\n"
            )
        post_message("reports", summary_text)

        # Full per-ticker skip detail → #kairos-log only
        if skip_reasons:
            skip_lines = "\n".join(
                f"  \u2022 {s['ticker']}: {s['reason']}" for s in skip_reasons
            )
            post_message("log", f":memo: *Skipped trades detail:*\n{skip_lines}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
