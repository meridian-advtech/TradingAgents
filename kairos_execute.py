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
from datetime import datetime, timedelta, timezone

from ib_insync import IB, Stock, LimitOrder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
LOG_FILE = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
LAST_DECISION_FILE = os.path.join(SCRIPT_DIR, "kairos_last_decision.txt")
REGIME_STATE_FILE = os.path.join(SCRIPT_DIR, ".kairos_regime_state.json")
W = 72

# Cash-failure markers. These MUST stay in sync with the failure strings emitted
# by kairos_confluence.check_cash_reserve(), which is the source of truth. It
# emits three distinct phrasings — negative cash, insufficient cash, and the
# soft reserve-floor breach — and in a near-zero-cash regime trades fail at the
# first two long before the reserve-floor string is ever reached. Matching only
# "cash reserve" (the old behavior) meant reallocation almost never triggered.
_CASH_FAIL_MARKERS = ("cash reserve", "insufficient cash", "cash negative")


def _reallocation_min_conviction() -> int:
    """Conviction floor that gates a reallocation ATTEMPT on the execute side.

    Reads reallocation.MIN_CONVICTION from kairos_config.json so the execute-side
    attempt-gate moves in lockstep with the engine-side floor in
    kairos_reallocation. Never hardcode this — if the two drift, the execute gate
    pre-filters out exactly the candidates the engine floor is meant to accept.
    """
    try:
        cfg_path = os.path.join(SCRIPT_DIR, "kairos_config.json")
        with open(cfg_path) as f:
            return int(json.load(f).get("reallocation", {}).get("MIN_CONVICTION", 5))
    except Exception:
        return 5


def _try_reallocation(ticker, new_conviction, signal_tags, ib, nlv,
                      trade, qty, decision, path):
    """Evaluate and (if recommended) execute a capital reallocation to fund a
    blocked BUY. Shared by BOTH the cash-breach path and the rounds-to-0 path so
    the two never drift.

    `path` is "cash_breach" or "rounds_to_0" and is logged so the silent-failure
    pattern (reallocation firing zero times) cannot recur — every evaluation
    emits a single [REALLOC] line whether or not it fires.

    Returns (executed: bool, spent: float).
    """
    print(f"    [REALLOC] Triggered for {ticker} (conv={new_conviction}, path={path})")
    try:
        from kairos_reallocation import (
            evaluate_reallocation, execute_reallocation, _load_protected_tickers,
        )
        realloc = evaluate_reallocation(ticker, new_conviction, signal_tags, ib, nlv)

        # Belt-and-suspenders: never liberate a protected position, even if the
        # reallocation engine's own filter missed it.
        exit_tk = (realloc.get("exit_ticker") or "").upper()
        if realloc.get("recommended") and exit_tk in _load_protected_tickers():
            print(f"    Reallocation BLOCKED: exit candidate {exit_tk} "
                  f"is a protected position — refusing to liberate.")
            realloc = {"recommended": False, "reason": f"{exit_tk} is protected"}

        if not realloc.get("recommended"):
            print(f"    [REALLOC] {ticker}: no exit candidate cleared MIN_CONVICTION_DELTA "
                  f"-> skipped ({realloc.get('reason', 'no reason')})")
            return False, 0.0

        sell_exec, buy_exec = execute_reallocation(realloc, trade, qty, decision, ib)
        if buy_exec and buy_exec.get("status") in ("Filled", "Submitted"):
            fill = buy_exec.get("fill_price")
            spent = qty * fill if fill else 0.0
            print(f"    [REALLOC] {ticker}: recommended exit={exit_tk} "
                  f"delta={realloc.get('conviction_delta')} -> EXECUTED")
            return True, spent
        print(f"    [REALLOC] {ticker}: exit={exit_tk} recommended but BUY leg "
              f"failed -> falling through to skip")
        return False, 0.0
    except Exception as realloc_exc:
        print(f"    WARNING: Reallocation eval failed for {ticker} "
              f"(path={path}): {realloc_exc}")
        return False, 0.0


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


def _is_paused() -> bool:
    """Return True when kairos_config.json sets paused=true.

    Toggled by kairos_commander.py's !pause / !resume commands.
    """
    try:
        with open(os.path.join(SCRIPT_DIR, "kairos_config.json")) as f:
            return bool(json.load(f).get("paused", False))
    except (IOError, json.JSONDecodeError):
        return False


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
        # Use live IBKR market value if available, fall back to qty × avg_cost
        if hasattr(p, 'marketValue') and p.marketValue is not None and p.marketValue != 0:
            mkt_val = abs(float(p.marketValue))
        else:
            mkt_val = abs(qty * avg)
        positions[sym] = {"qty": qty, "avg_cost": round(avg, 2), "market_value": round(mkt_val, 2)}

        sector = lookup_sector(sym)
        sector_exposure[sector] = sector_exposure.get(sector, 0) + mkt_val

    return {
        "nlv": nlv,
        "cash": cash,
        "positions": positions,
        "sector_exposure": sector_exposure,
    }


def get_open_order_tickers(ib: IB) -> set[str]:
    """Return the set of ticker symbols that currently have an active
    (open/pending) order at IBKR.

    Used to prevent stacking a duplicate BUY on a ticker that already has
    a live order waiting to fill or be cancelled.
    """
    try:
        ib.reqAllOpenOrders()
    except Exception:
        pass

    active_statuses = {
        "ApiPending", "PendingSubmit", "PendingCancel",
        "PreSubmitted", "Submitted",
    }
    tickers: set[str] = set()
    for trade in ib.openTrades():
        try:
            status = trade.orderStatus.status
            if status in active_statuses:
                tickers.add(trade.contract.symbol)
        except Exception:
            continue
    return tickers


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

    # 2b. SHADOW ONLY — record what this same check would decide using real
    # sectors (security_master) instead of universe screening buckets. The
    # verdict above is NOT touched: lookup_sector returns size/style cohorts,
    # so the live threshold was calibrated against a number that understates
    # true sector exposure, and swapping the measurement without re-deciding
    # the limit would silently tighten the gate. Evidence first, then J sets
    # the limit that belongs with real sectors. Failure here is swallowed —
    # observation must never be able to block a trade.
    try:
        from kairos_sector_shadow import log_shadow_decision
        note = log_shadow_decision(
            ticker, proposed_spend, portfolio, guardrails,
            bucket_passed=ok, bucket_sector=sector)
        if note:
            print(f"    {note}")
    except Exception:
        pass

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

    # ── Never-oversell guard (long-only) ──────────────────────────
    # Last line of defence before submission: no SELL may exceed the shares
    # actually held. See kairos_sell_guard for the contract.
    if action == "SELL":
        from kairos_sell_guard import clamp_sell_quantity
        qty, clamp_note = clamp_sell_quantity(ticker, qty, ib,
                                              context="execute_order")
        if qty <= 0:
            return {"status": "Cancelled", "oversell_blocked": True,
                    "reason": clamp_note or "oversell prevented"}

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

    order = LimitOrder(action, qty, limit_price, tif="DAY", outsideRth=True)
    order.overridePercentageConstraints = True
    trade = ib.placeOrder(contract, order)
    print(f"    Order ID: {trade.order.orderId}")

    timeout = 45
    start = time.time()
    while time.time() - start < timeout:
        ib.sleep(1)
        if trade.isDone():
            break

    result = {
        "status": trade.orderStatus.status,
        "order_id": trade.order.orderId,
        "limit_price": limit_price,   # intended price — fallback entry price if no fill price
    }

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

        # Phase C tag: the active learned-calibration weights that were in the
        # reasoning prompt this cycle, captured to kairos_axis_snapshot.json by
        # kairos_reason.py at prompt-build time. We stamp them onto this decision
        # row (axis_weights_snapshot column, added by the Phase C migration) for
        # Phase D efficacy analysis. A missing/empty/unparseable sidecar tags NULL
        # and must NEVER block or alter execution.
        axis_weights_snapshot = None
        try:
            _snap_path = os.path.join(SCRIPT_DIR, "kairos_axis_snapshot.json")
            if os.path.exists(_snap_path):
                with open(_snap_path) as _snap_f:
                    _snap = json.load(_snap_f)
                # Only a non-empty {axis: weight} map is a real tag; {} means no
                # active weight cleared the deadband this cycle → leave NULL.
                if isinstance(_snap, dict) and _snap:
                    axis_weights_snapshot = json.dumps(_snap)
        except Exception as _snap_err:
            axis_weights_snapshot = None
            print(f"    NOTE: axis weights snapshot unavailable "
                  f"({_snap_err}) — tagging NULL")

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

        # Stamp the Phase C weights snapshot onto the row just inserted. Done as a
        # follow-up UPDATE (insert_decision's signature is fixed) and fully guarded
        # so a DB hiccup here can never block or alter the trade. Skipped when the
        # sidecar gave us nothing → column stays NULL.
        if axis_weights_snapshot is not None:
            try:
                from kairos_log_db import get_connection
                _tag_conn = get_connection()
                try:
                    _tag_conn.execute(
                        "UPDATE decisions SET axis_weights_snapshot = ? WHERE id = ?",
                        (axis_weights_snapshot, row_id),
                    )
                    _tag_conn.commit()
                finally:
                    _tag_conn.close()
                print(f"    DB: tagged decision #{row_id} with axis weights "
                      f"{axis_weights_snapshot}")
            except Exception as _tag_err:
                print(f"    NOTE: could not tag decision #{row_id} with axis "
                      f"weights ({_tag_err}) — continuing")

        exec_filled = execution.get("status") == "Filled"
        fill_price = execution.get("fill_price")
        action_upper = action.upper()

        # ── BUY entry logging — robust to a missing/late fill price ──────
        # A BUY that filled at the broker MUST create the ML record, even if the
        # fill price has not yet arrived — otherwise the trade is lost from the
        # training corpus (this orphaned 13 positions historically). Derive the
        # best entry price available and mark the row for reconciliation when it
        # is only an estimate.
        if exec_filled and action_upper == "BUY":
            entry_price, price_provisional = _best_entry_price(execution, trade)
            if entry_price is None:
                # No price knowable anywhere — can't create a meaningful row.
                _alert_ml_log_failure(
                    ticker, qty,
                    "Filled BUY with no derivable entry price — ML row NOT created")
            else:
                if price_provisional:
                    print(f"    NOTE: no immediate fill price — logging entry at "
                          f"${entry_price:.2f} (provisional, flagged for reconcile)")
                h_id = insert_holding(ticker, timestamp, entry_price, qty)
                print(f"    Holdings: BUY lot #{h_id}")

                # Signals that actually triggered THIS trade — sourced from the
                # trade's own decision context first, falling back to the JSON
                # screener. Recovers reversion/score-route entries (HOT-REVERSION,
                # HOT-IPO, reversion-route HOT-CATALYST) that get_ticker_signals
                # alone returns [] for.
                ml_signals, ml_attr_source = _derive_entry_signals_with_source(
                    trade, ticker)

                # Confluence: prefer the score computed during sizing; if that is
                # 0/None (reversion route never scored from these tags), recompute
                # from the real signals rather than logging a hard 0.
                ml_confluence = conf.get("score") if conf else None
                if not ml_confluence and ml_signals:
                    try:
                        from kairos_confluence import compute_confluence
                        ml_confluence = compute_confluence(ml_signals)["score"]
                    except Exception:
                        pass
                ml_sector = trade.get("sector") or _lookup_sector(ticker)

                # Own-data capture at entry (added 2026-08-19): conviction is
                # Kairos's own confidence in THIS trade; the ml_* fields are
                # the model's prediction, stored so it can be scored against
                # the realized outcome later; regime enables per-regime
                # performance analysis. All were previously discarded.
                ml_conviction = trade.get("conviction")
                try:
                    ml_conviction = int(ml_conviction) if ml_conviction not in (None, "") else None
                except (TypeError, ValueError):
                    ml_conviction = None
                ml_pred_conf, ml_pred_signal, ml_pred_trained_on = _ml_prediction_for(ticker)
                ml_regime = _current_market_regime()

                # ML Outcomes: record trade open. A trade that executes but fails
                # to log is a corpus-integrity event — alert, never swallow.
                ml_trade_id = None
                try:
                    from kairos_ml_outcomes import init_db as ml_init, write_trade_open
                    ml_init()
                    ml_trade_id = write_trade_open(
                        ticker=ticker,
                        action=action_upper,
                        quantity=qty,
                        price_entry=entry_price,
                        timestamp_entry=timestamp,
                        signals_fired=ml_signals if ml_signals else None,
                        confluence_score=ml_confluence,
                        conviction=ml_conviction,
                        ml_confidence_at_entry=ml_pred_conf,
                        ml_signal_at_entry=ml_pred_signal,
                        ml_trained_on_at_entry=ml_pred_trained_on,
                        market_regime=ml_regime,
                        sector=ml_sector,
                        entry_price_provisional=price_provisional,
                        signal_attribution_source=ml_attr_source,
                    )
                    print(f"    ML Outcomes: trade open {ml_trade_id[:8]}... "
                          f"(signals={ml_signals or 'none'}, "
                          f"source={ml_attr_source}, "
                          f"confluence={ml_confluence}, "
                          f"conviction={ml_conviction}, "
                          f"ml_pred={ml_pred_signal or 'n/a'}"
                          f"{f'/{ml_pred_conf}' if ml_pred_conf is not None else ''}, "
                          f"regime={ml_regime or 'n/a'})")
                except Exception as ml_exc:
                    _alert_ml_log_failure(
                        ticker, qty, f"write_trade_open failed: {ml_exc}")
                    ml_trade_id = None

                # Thesis prediction: capture Claude's expected move at entry.
                # Fields are sourced from the per-trade JSON; signal_type now
                # comes from the trade's REAL signals (no longer None on the
                # reversion/score path).
                if ml_trade_id:
                    try:
                        from kairos_ml_thesis import (
                            write_thesis_prediction, pick_primary_signal,
                        )
                        thesis_signal = pick_primary_signal(ml_signals)
                        write_thesis_prediction(
                            decision_id=ml_trade_id,
                            ticker=ticker,
                            timestamp_entry=timestamp,
                            predicted_direction=trade.get("predicted_direction"),
                            predicted_timeframe_days=trade.get("predicted_timeframe_days"),
                            predicted_return_pct=trade.get("predicted_return_pct"),
                            key_conditions=trade.get("key_conditions"),
                            signal_type=thesis_signal,
                            conviction_score=trade.get("conviction"),
                            invalidation_conditions=trade.get("invalidation_conditions"),
                        )
                        print(f"    Thesis prediction recorded "
                              f"({trade.get('predicted_direction','?')} "
                              f"{trade.get('predicted_return_pct','?')}% / "
                              f"{trade.get('predicted_timeframe_days','?')}d, "
                              f"signal={thesis_signal or 'n/a'})")
                    except Exception as th_exc:
                        print(f"    WARNING: thesis prediction failed: {th_exc}")

        # ── SELL close logging — needs a real fill price for PnL ─────────
        elif exec_filled and fill_price and action_upper == "SELL":
            # Exit reason: the SELL decision's own rationale (covers manual
            # sells, reallocation buy-legs, and the oversized+underwater TRIM,
            # which all route through here). sell_holdings records it so the
            # close is never silent.
            sell_reason = (trade.get("rationale") or "").strip() or "SELL (unspecified)"
            closed_lots = sell_holdings(ticker, qty, timestamp, fill_price, sell_reason)
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


# Screen-result route lists → the signal tag they represent. These are the
# entry routes that get_ticker_signals (kairos_signal_summary.json only) does
# NOT see — chiefly the reversion route, which is how IBM/FICO/META/GD enter.
_SCREEN_ROUTE_TAGS = {
    "hot_reversion": "HOT-REVERSION",
    "hot_earnings":  "HOT-EARNINGS",
    "hot_rsi":       "HOT-RSI",
    "hot_insider":   "HOT-INSIDER",
    "hot_congress":  "HOT-CONGRESS",
}

# Tags we'll recognise if Claude embedded them in the trade rationale (the
# reasoning layer is told to "Tag REVERSION trades in the rationale").
_KNOWN_RATIONALE_TAGS = (
    "HOT-EARNINGS", "HOT-RSI", "HOT-INSIDER", "HOT-CONGRESS",
    "HOT-REVERSION", "HOT-KALSHI", "HOT-OPTIONS", "HOT-CATALYST", "HOT-IPO",
)


def _current_market_regime() -> str | None:
    """Regime label for stamping onto a trade at entry.

    Reads kairos.db regime_log (written fresh by Phase 0R every cycle) rather
    than .kairos_regime_state.json, which was last written 2026-04-23 and is
    NOT a reliable current-regime source. Returns None rather than guessing —
    an unstamped trade is honest; a wrong regime label would poison exactly
    the per-regime performance analysis this field exists to enable.
    """
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(SCRIPT_DIR, "kairos.db"))
        row = conn.execute(
            "SELECT regime FROM regime_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _ml_prediction_for(ticker: str) -> tuple[float | None, str | None, int | None]:
    """This cycle's ML prediction for a ticker, as (confidence, signal, trained_on).

    Captured at ENTRY so the model's prediction can later be scored against the
    realized outcome. Without this the ML loop is open — predictions are made
    every cycle and then discarded, so there is no way to answer "do STRONG
    picks actually win more often than WEAK ones", i.e. no way to validate or
    calibrate the model against reality. Returns (None, None, None) when the
    ticker wasn't scored this cycle (e.g. exit-engine or IPO-route entries that
    never passed through Phase 0.7).
    """
    try:
        path = os.path.join(SCRIPT_DIR, "kairos_ml_result.json")
        if not os.path.exists(path):
            return None, None, None
        with open(path) as f:
            data = json.load(f)
        for cand in data.get("candidates", []):
            if str(cand.get("ticker", "")).upper() == ticker.upper():
                return (cand.get("ml_confidence"),
                        cand.get("ml_signal"),
                        cand.get("ml_trained_on"))
    except Exception:
        pass
    return None, None, None


def _derive_entry_signals_with_source(trade: dict, ticker: str) -> tuple[list[str], str]:
    """Signals that actually triggered THIS trade, plus how we know.

    Returns (tags, source) where source is one of ATTRIBUTION_SOURCES.

    The sources are NOT equivalent, and treating them as though they were is
    what contaminated per-signal P&L. They fall into three tiers, and the FIRST
    tier that yields anything wins outright — later tiers are not merged in:

      TIER A — trade-level, causal. What triggered *this trade*.
        1. Explicit signal context on the trade dict (signals_fired / signals /
           signal_tags), set by entry paths that know their own trigger.
        2. Confluence tags attached during sizing (trade["_confluence"]).

      TIER B — ticker-level, CONTEXTUAL. What was firing for *this ticker*
        that day, which is not the same claim at all.
        3. get_ticker_signals(ticker) — kairos_signal_summary.json.
        4. kairos_screen_result.json per-ticker tags + route-list membership
           (recovers the reversion / score-based route).

      TIER C — inferred from prose. A guess.
        5. Tags mentioned in the trade's rationale text.

    The previous implementation documented this priority but did not implement
    it: every source was unioned unconditionally, so a reversion trade in a
    ticker that also had congressional activity was recorded as a HOT-CONGRESS
    trade too. 38% of the corpus carries multiple tags as a result, and
    HOT-CONGRESS's population became mostly trades it never drove. Attribution
    is now single-tier, and the tier is recorded so downstream consumers can
    tell causation from coincidence.

    Tags are de-duplicated and upper-cased. An empty list with source "none" is
    a real answer — a genuinely context-free trade, which we do NOT fabricate.
    """
    def _norm(seq) -> list[str]:
        out: list[str] = []
        for s in (seq or []):
            t = str(s).strip().upper()
            if t and t not in out:
                out.append(t)
        return out

    # ── TIER A.1: explicit on the trade dict ─────────────────────────
    explicit: list[str] = []
    for key in ("signals_fired", "signals", "signal_tags"):
        val = trade.get(key)
        if isinstance(val, str):
            val = [val]
        for t in _norm(val):
            if t not in explicit:
                explicit.append(t)
    if explicit:
        return _finalize_attribution(explicit, "explicit", ticker)

    # ── TIER A.2: confluence captured during sizing ──────────────────
    conf_tags = _norm((trade.get("_confluence") or {}).get("signals"))
    if conf_tags:
        return _finalize_attribution(conf_tags, "confluence", ticker)

    # ── TIER B: ticker-level context (signal_summary + screen_result) ─
    ctx: list[str] = []
    try:
        from kairos_confluence import get_ticker_signals
        ctx.extend(t for t in _norm(get_ticker_signals(ticker)) if t not in ctx)
    except Exception:
        pass
    try:
        screen_file = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")
        if os.path.exists(screen_file):
            with open(screen_file) as f:
                screen = json.load(f)
            for t in _norm((screen.get("signal_tags") or {}).get(ticker)):
                if t not in ctx:
                    ctx.append(t)
            for route, tag in _SCREEN_ROUTE_TAGS.items():
                if ticker in (screen.get(route) or []) and tag not in ctx:
                    ctx.append(tag)
    except Exception:
        pass
    if ctx:
        return _finalize_attribution(ctx, "ticker_context", ticker)

    # ── TIER C: parsed out of the rationale prose ────────────────────
    rationale = (trade.get("rationale") or "").upper()
    inferred: list[str] = []
    if rationale:
        for tag in _KNOWN_RATIONALE_TAGS:
            if tag in rationale and tag not in inferred:
                inferred.append(tag)
        # Bare keyword without the HOT- prefix (the reasoning layer often writes
        # just "REVERSION play"). Deliberately last, and only when nothing
        # structured exists anywhere — substring matching cannot see negation,
        # so "not a reversion setup" would read as HOT-REVERSION.
        if "HOT-REVERSION" not in inferred and "REVERSION" in rationale:
            inferred.append("HOT-REVERSION")
    if inferred:
        return _finalize_attribution(inferred, "rationale_text", ticker)

    return [], "none"


def _finalize_attribution(tags: list[str], source: str, ticker: str) -> tuple[list[str], str]:
    """Apply the HARD-pause attribution strip to a resolved tag list.

    A hard-killed signal must never reach trade_outcomes.signals_fired, so it
    cannot distort the dashboard or the Arbiter's per-signal analysis going
    forward. FORWARD-ONLY (existing rows are left intact) and strips ONLY tags
    whose mode is exactly 'hard' — co-firing healthy signals and soft-paused
    confirmers are preserved, so a confluence trade driven by a healthy signal
    keeps its honest attribution.
    """
    try:
        from kairos_signals import signal_pause_mode
        stripped = [t for t in tags if signal_pause_mode(t) == "hard"]
        if stripped:
            tags = [t for t in tags if signal_pause_mode(t) != "hard"]
            print(f"    [PAUSED:hard] {ticker}: excluded {', '.join(stripped)} "
                  f"from ML attribution (kept: {tags or 'none'})")
    except Exception:
        pass
    return tags, ("none" if not tags else source)


def _derive_entry_signals(trade: dict, ticker: str) -> list[str]:
    """Tag list only — see _derive_entry_signals_with_source for provenance."""
    return _derive_entry_signals_with_source(trade, ticker)[0]


def _alert_ml_log_failure(ticker: str, qty, detail: str) -> None:
    """A BUY executed at the broker but failed to log to the ML ledger.

    This is a corpus-integrity event: the trade happened but won't be in the
    training data. Print loudly AND notify #kairos-alerts. Never raises.
    """
    msg = f"ML-LEDGER GAP: {ticker} BUY x{qty} executed but did NOT log — {detail}"
    print(f"    ALERT: {msg}")
    try:
        from kairos_alerts import post_message
        post_message(
            "alerts",
            f":rotating_light: *Trade-log integrity*\n{msg}\n"
            f"This executed BUY is missing from kairos_ml_outcomes.db — "
            f"reconcile manually so the corpus stays complete.",
        )
    except Exception as exc:
        print(f"    (could not post ML-log-failure alert: {exc})")


def _best_entry_price(execution: dict, trade: dict) -> tuple[float | None, bool]:
    """Pick the best available entry price for ML logging.

    Returns (price, provisional). provisional is True when the price is an
    estimate (not the actual fill price) and should be reconciled later.
    Order: real fill price → broker avg cost after fill → order limit price →
    the decision's intended price. Returns (None, ...) only if nothing is known.
    """
    fill_price = execution.get("fill_price")
    if fill_price:
        return float(fill_price), False

    # Broker's average cost after the fill — confirmed by IBKR, reconcilable.
    new_pos = execution.get("new_position") or {}
    avg_cost = new_pos.get("avg_cost")
    if avg_cost:
        return float(avg_cost), True

    # Intended price for THIS order, then the decision's intended price.
    for candidate in (execution.get("limit_price"),
                      trade.get("intended_price"), trade.get("ref_price")):
        if candidate:
            return float(candidate), True

    return None, True


# ── Trim trigger ──────────────────────────────────────────────────────

def check_trim_triggers(portfolio: dict, ib: IB, nlv: float) -> list[dict]:
    """Scan positions for oversized + underwater holdings and generate partial SELLs.

    A position triggers a trim if:
      - It exceeds 8% of NLV, AND
      - Current price is below avg_cost (underwater)

    Trims bring the position down to 6% of NLV.
    Returns list of executed trim results for logging.
    """
    trims_executed = []
    if nlv <= 0:
        return trims_executed

    for sym, pos_data in portfolio["positions"].items():
        mkt_val = pos_data["market_value"]
        weight = mkt_val / nlv
        avg_cost = pos_data["avg_cost"]

        if weight <= 0.08:
            continue

        # Fetch live price to confirm underwater
        current_price = get_reference_price(ib, sym)
        if current_price is None:
            continue

        if current_price >= avg_cost:
            continue  # not underwater

        # Trim to 6% of NLV
        target_value = nlv * 0.06
        current_value = pos_data["qty"] * current_price
        excess_value = current_value - target_value
        if excess_value <= 0:
            continue

        trim_qty = int(excess_value / current_price)
        if trim_qty < 1:
            continue

        reason = (f"TRIM: oversized + underwater — {sym} at {weight:.1%} of NLV, "
                  f"price ${current_price:.2f} < avg ${avg_cost:.2f}, trimming {trim_qty} shares to ~6%")
        print(f"  {reason}")

        # Execute the trim
        execution = execute_order(ib, sym, "SELL", trim_qty)
        trims_executed.append({"ticker": sym, "qty": trim_qty, "execution": execution, "reason": reason})

        # Log to decisions log
        trim_trade = {
            "action": "SELL",
            "ticker": sym,
            "quantity": trim_qty,
            "rationale": reason,
            "sector": "",
        }
        log_execution({"trades": [], "tickers_evaluated": []}, trim_trade, execution)

        # Slack alert
        try:
            from kairos_alerts import post_message
            status = execution.get("status", "Unknown")
            fill = f" @ ${execution['fill_price']:.2f}" if execution.get("fill_price") else ""
            post_message("trades",
                f":scissors: *{reason}*\n"
                f"Sold {trim_qty} shares{fill} — status: {status}")
        except Exception as exc:
            print(f"  trim Slack post failed: {exc}")

    return trims_executed


# ── Stale order cleanup ─────────────────────────────────────────────

def cancel_stale_orders(max_age_hours: int = 8) -> int:
    """Cancel live IBKR orders older than ``max_age_hours``.

    Connects to TWS on port 7497 with a dedicated clientId, pulls all open
    orders, matches each to its original Submitted row in kairos.db
    (by ticker + quantity), and cancels any whose original submit timestamp
    is older than the threshold. The matching decisions row is updated to
    execution_status='Cancelled', and a summary is posted to #kairos-alerts.

    Returns the number of orders cancelled.
    """
    from kairos_log_db import get_connection, init_db

    init_db()

    ib = IB()
    try:
        ib.connect("127.0.0.1", 7497, clientId=6, timeout=10)
    except Exception as exc:
        print(f"  cancel_stale_orders: IBKR connect failed: {exc}")
        return 0

    cancelled = 0
    try:
        try:
            ib.reqAllOpenOrders()
            ib.sleep(2)
        except Exception as exc:
            print(f"  cancel_stale_orders: reqAllOpenOrders failed: {exc}")

        active_statuses = {
            "ApiPending", "PendingSubmit", "PendingCancel",
            "PreSubmitted", "Submitted",
        }

        now = datetime.now(timezone.utc)
        threshold = timedelta(hours=max_age_hours)
        conn = get_connection()

        for trade in ib.openTrades():
            try:
                status = trade.orderStatus.status
                if status not in active_statuses:
                    continue
                ticker = trade.contract.symbol
                qty = int(trade.order.totalQuantity)
            except Exception:
                continue

            row = conn.execute(
                "SELECT id, timestamp FROM decisions "
                "WHERE ticker = ? AND quantity = ? AND execution_status = 'Submitted' "
                "ORDER BY id DESC LIMIT 1",
                (ticker, qty),
            ).fetchone()

            if row is None:
                print(f"  cancel_stale_orders: no Submitted DB row for "
                      f"{ticker} qty={qty} — skipping age check")
                continue

            decision_id = row["id"]
            ts_str = row["timestamp"]
            try:
                submitted_at = datetime.strptime(
                    ts_str, "%Y-%m-%d %H:%M:%S UTC"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                print(f"  cancel_stale_orders: unparseable timestamp "
                      f"{ts_str!r} on decision #{decision_id} — skipping")
                continue

            age = now - submitted_at
            if age < threshold:
                continue

            age_h = age.total_seconds() / 3600
            try:
                ib.cancelOrder(trade.order)
                conn.execute(
                    "UPDATE decisions SET execution_status = 'Cancelled' "
                    "WHERE id = ?",
                    (decision_id,),
                )
                conn.commit()
                cancelled += 1
                print(f"  Cancelled stale order: {ticker} qty={qty} "
                      f"age={age_h:.1f}h (decision #{decision_id})")
            except Exception as exc:
                print(f"  cancel_stale_orders: cancel failed for "
                      f"{ticker} qty={qty}: {exc}")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    if cancelled > 0:
        try:
            from kairos_alerts import post_message
            # Order housekeeping, not a position change → #log (verbose detail).
            post_message(
                "log",
                f":wastebasket: Cancelled {cancelled} stale orders older than {max_age_hours}h",
            )
        except Exception as exc:
            print(f"  cancel_stale_orders: Slack post failed: {exc}")

    return cancelled


def _looks_like_possible_fill(row) -> bool:
    """True if a Submitted row carries evidence of a possible (mislogged) fill:
    a non-null execution_price AND a position_after whose quantity is positive.

    Reconciliation uses this as a safety brake — it must never overwrite a row
    that might actually represent a real fill (only the execution status would be
    wrong, and that is a manual-review case, not an auto-expire case).
    """
    if row["execution_price"] is None:
        return False
    pos = row["position_after"]
    if not pos:
        return False
    try:
        qty_after = json.loads(pos).get("quantity")
        return qty_after is not None and float(qty_after) > 0
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def reconcile_submitted_orders(grace_minutes: int = 30) -> int:
    """Reconcile DB rows stuck at 'Submitted' whose IBKR order is no longer live.

    For each decisions row with execution_status='Submitted' older than
    grace_minutes: if a matching order (ticker+quantity) is NOT in IBKR's current
    open orders AND no fill was recorded, mark it 'Expired'. This catches DAY
    orders that expired at market close and fell off the open-orders list.

    This is the DB-side INVERSE of cancel_stale_orders: that routine walks
    ib.openTrades() (orders still live at IBKR) and can never see an order that
    has already expired off the broker; this one walks the stuck DB rows and
    trues up the ones whose broker order has already vanished.

    Returns the number of rows reconciled.
    """
    from kairos_log_db import get_connection, init_db

    init_db()

    ib = IB()
    try:
        # Dedicated clientId=7, distinct from cancel_stale_orders' clientId=6,
        # so the two can run back-to-back in the same cycle without colliding.
        ib.connect("127.0.0.1", 7497, clientId=7, timeout=10)
    except Exception as exc:
        print(f"  reconcile_submitted_orders: IBKR connect failed: {exc}")
        return 0

    reconciled = 0
    ml_reconciled = 0  # provisional ML entry prices trued up to broker avg cost
    flagged = []  # rows that look like a possible mislogged fill — skipped
    try:
        try:
            ib.reqAllOpenOrders()
            ib.sleep(2)
        except Exception as exc:
            print(f"  reconcile_submitted_orders: reqAllOpenOrders failed: {exc}")

        active_statuses = {
            "ApiPending", "PendingSubmit", "PendingCancel",
            "PreSubmitted", "Submitted",
        }

        # Build the set of (symbol, qty) currently live at IBKR.
        live_orders = set()
        for trade in ib.openTrades():
            try:
                if trade.orderStatus.status in active_statuses:
                    live_orders.add(
                        (trade.contract.symbol, int(trade.order.totalQuantity))
                    )
            except Exception:
                continue

        now = datetime.now(timezone.utc)
        grace = timedelta(minutes=grace_minutes)
        conn = get_connection()

        rows = conn.execute(
            "SELECT id, timestamp, ticker, quantity, execution_price, "
            "position_after, rationale FROM decisions "
            "WHERE execution_status = 'Submitted' ORDER BY id"
        ).fetchall()

        for row in rows:
            decision_id = row["id"]
            ticker = row["ticker"]
            qty = row["quantity"]
            ts_str = row["timestamp"]

            try:
                submitted_at = datetime.strptime(
                    ts_str, "%Y-%m-%d %H:%M:%S UTC"
                ).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                print(f"  reconcile_submitted_orders: unparseable timestamp "
                      f"{ts_str!r} on decision #{decision_id} — skipping")
                continue

            age = now - submitted_at
            if age < grace:
                continue  # may still be working this session

            if (ticker, int(qty)) in live_orders:
                continue  # genuinely still open — let cancel_stale_orders age it out

            # Safety: never expire a row that looks like a possible mislogged
            # fill (recorded price + a positive resulting position). Flag for
            # manual review instead of destroying a real fill record.
            if _looks_like_possible_fill(row):
                flagged.append((decision_id, ticker, qty))
                continue

            note = (f" [AUTO-RECONCILED {now.strftime('%Y-%m-%d')}: DAY order no "
                    f"longer live at IBKR, no fill recorded]")
            new_rationale = (row["rationale"] or "") + note
            conn.execute(
                "UPDATE decisions SET execution_status = 'Expired', rationale = ? "
                "WHERE id = ?",
                (new_rationale, decision_id),
            )
            conn.commit()
            reconciled += 1
            age_h = age.total_seconds() / 3600
            print(f"  Reconciled expired order: {ticker} qty={qty} "
                  f"age={age_h:.1f}h (decision #{decision_id}) -> Expired")

        # ── Backfill provisional ML entry prices from the broker ─────────
        # BUY rows opened before their fill price was known carry a best-effort
        # estimate (entry_price_provisional=1). Now that we're connected, true
        # them up against IBKR's confirmed average cost per position.
        try:
            from kairos_ml_outcomes import (
                init_db as ml_init, list_provisional_entries,
                reconcile_provisional_entries,
            )
            ml_init()
            pending = list_provisional_entries()
            if pending:
                broker_costs = {}
                for p in ib.positions():
                    if p.position and p.avgCost:
                        broker_costs[p.contract.symbol] = round(p.avgCost, 4)
                fixed = reconcile_provisional_entries(broker_costs)
                if fixed:
                    ml_reconciled = fixed
                    print(f"  ML entry prices reconciled: {fixed} provisional "
                          f"row(s) trued up to broker avg cost")
        except Exception as exc:
            print(f"  reconcile_submitted_orders: ML entry-price reconcile failed: {exc}")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    if flagged:
        print(f"  WARNING: reconcile_submitted_orders skipped {len(flagged)} row(s) "
              f"with a recorded price + positive position (possible mislogged "
              f"fill) — manual review needed:")
        for did, tk, q in flagged:
            print(f"    • decision #{did}: {tk} qty={q}")

    if reconciled > 0 or ml_reconciled > 0:
        try:
            from kairos_alerts import post_message
            parts = []
            if reconciled > 0:
                parts.append(
                    f":broom: Reconciled {reconciled} expired DAY order(s) "
                    f"stuck at 'Submitted' >{grace_minutes}m and no longer live at IBKR")
            if ml_reconciled > 0:
                parts.append(
                    f":abacus: Backfilled {ml_reconciled} provisional ML entry "
                    f"price(s) from broker avg cost")
            post_message("alerts", "\n".join(parts))
        except Exception as exc:
            print(f"  reconcile_submitted_orders: Slack post failed: {exc}")

    return reconciled


# ── Position reconciliation (IBKR = source of truth) ────────────────

def _backup_kairos_db(tag: str = "RECON") -> str | None:
    """Timestamped copy of kairos.db before a reconciliation write pass.

    Returns the backup path, or None if the source is missing / copy fails.
    """
    import shutil
    src = os.path.join(SCRIPT_DIR, "kairos.db")
    if not os.path.exists(src):
        print("  reconcile: kairos.db not found — cannot back up")
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(SCRIPT_DIR, f"kairos_{tag}_BACKUP_{ts}.db")
    try:
        shutil.copy2(src, dst)
        print(f"  Backed up kairos.db -> {os.path.basename(dst)}")
        return dst
    except Exception as exc:
        print(f"  WARNING: kairos.db backup failed: {exc}")
        return None


def _most_recent_buy_date(conn, ticker: str) -> str | None:
    """Best-effort entry_date for an orphan: timestamp of its most recent BUY."""
    try:
        row = conn.execute(
            "SELECT timestamp FROM decisions WHERE ticker = ? AND action = 'BUY' "
            "ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        return row["timestamp"] if row else None
    except Exception:
        return None


def reconcile_positions_against_broker(dry_run: bool = True) -> dict:
    """Reconcile kairos.db open holdings against the live IBKR account.

    IBKR is the SOURCE OF TRUTH. A fill-callback gap once dropped many fills,
    leaving holdings that drifted from the broker (orphans with no stop/thesis
    management, qty/cost mismatches, and lot fragmentation). This routine trues
    the DB up to the broker on a schedule so that can't silently recur.

    Per-ticker resolution:
      • ORPHAN       — broker holds it, no open DB lot → CREATE one consolidated
                       row at broker qty + avgCost. entry_date from the most
                       recent BUY decision, else stamped with a [RECON] marker.
                       peak_gain_pct seeded from current gain if positive, else 0.
      • QTY MISMATCH — broker qty != summed DB qty → consolidate to ONE row at
                       broker qty + avgCost; extra lots marked sold [RECON-merged].
      • CONSOLIDATE  — qty already matches but >1 open lot (lot fragmentation,
                       e.g. DE/KEYS/ONON) → collapse to one blended row anyway.
      • PHANTOM      — DB shows open, broker is flat → mark the DB lot(s) sold
                       (a real exit went unrecorded). Logged loudly.

    Broker access is READ-ONLY (ib.positions() only — never places or cancels an
    order). Writes go ONLY to kairos.db holdings, and ONLY when dry_run is False.
    Backs up kairos.db before any write pass. Returns a dict of what changed
    (proposed changes when dry_run).
    """
    from collections import defaultdict
    from kairos_log_db import get_connection, init_db

    init_db()

    QTY_TOL = 0.5        # sub-share tolerance — ignore fractional DRIP drift
    COST_TOL_PCT = 1.0   # only true-up cost basis when it differs by >1%
    # ── Safety guards (surfaced by live observation 2026-06-18) ──────
    # A flaky/partial IBKR response can return an empty or short position
    # list; without a guard, every open DB holding would look PHANTOM and
    # get mass-marked sold. And a snapshot taken mid-fill (e.g. APC during
    # a 730-share stop-loss) shows a transient qty that is not real drift.
    MAX_PHANTOM_FRACTION = 0.34  # abort writes if >34% of open tickers look phantom
    MIN_DB_FOR_GUARD = 5         # only apply the fraction guard once book is non-trivial

    summary: dict = {
        "dry_run": dry_run,
        "created": [],   # ORPHAN -> new consolidated row
        "updated": [],   # QTY / COST mismatch -> trued up to broker
        "merged": [],    # lot-fragmentation consolidations
        "phantom": [],   # DB open, broker flat -> marked sold (SERIOUS)
        "broker_positions": 0,
        "db_open_lots": 0,
        "errors": [],
    }
    mode = "DRY-RUN" if dry_run else "WRITE"
    print(banner(f"Position Reconciliation ({mode}) — IBKR = source of truth"))

    ib = IB()
    try:
        # Dedicated clientId=8 — distinct from cancel_stale_orders (6) and
        # reconcile_submitted_orders (7) so all three run back-to-back cleanly.
        ib.connect("127.0.0.1", 7497, clientId=8, timeout=10)
    except Exception as exc:
        msg = f"IBKR connect failed: {exc}"
        print(f"  reconcile_positions_against_broker: {msg}")
        summary["errors"].append(msg)
        return summary

    try:
        # ── Broker truth: STK positions only ──────────────────────────
        broker: dict = {}
        try:
            ib.reqPositions()
            ib.sleep(1)
            for p in ib.positions():
                try:
                    if p.contract.secType != "STK":
                        continue
                    sym = p.contract.symbol
                    qty = float(p.position)
                    if abs(qty) < 1e-9:
                        continue
                    broker[sym] = (qty, round(float(p.avgCost), 4))
                except Exception as exc:
                    summary["errors"].append(f"broker position parse: {exc}")
                    print(f"  WARNING: could not parse a broker position: {exc}")
        except Exception as exc:
            msg = f"ib.positions() failed: {exc}"
            print(f"  reconcile_positions_against_broker: {msg}")
            summary["errors"].append(msg)
            return summary

        summary["broker_positions"] = len(broker)
        print(f"  Broker STK positions: {len(broker)}")

        # ── Long-only sanity check: NEGATIVE broker positions ─────────
        # Distinct from the drift checks below, which validate DB↔broker SYNC.
        # A short STK position is never valid for Kairos regardless of whether
        # the DB agrees with it (on 2026-07-29 DB and broker agreed on ETN -16
        # / EME -8, so drift reconciliation stayed silent). This fires every
        # run until the position is flat. It never auto-trades — flattening is
        # a human decision.
        shorts = {sym: q for sym, (q, _c) in broker.items() if q < 0}
        summary["short_positions"] = shorts
        if shorts:
            detail = ", ".join(f"{s} {q:g}" for s, q in sorted(shorts.items()))
            print(f"  *** LONG-ONLY VIOLATION: negative broker position(s): {detail}")
            summary["errors"].append(f"negative broker position(s): {detail}")
            try:
                from kairos_alerts import post_message
                lines = "\n".join(
                    f"• *{s}*: {q:g} shares short" for s, q in sorted(shorts.items()))
                post_message("alerts",
                    f":rotating_light: *LONG-ONLY VIOLATION — short position at broker*\n"
                    f"{lines}\n"
                    f"Kairos is long-only. Buy-to-cover to flatten; this alert "
                    f"repeats every reconciliation run until the position is flat. "
                    f"No automatic action has been taken.")
            except Exception as exc:
                print(f"  short-position alert failed: {exc}")

        # In-flight-order guard: a ticker with a working order at IBKR has a
        # quantity that is legitimately mid-change (partial fill in progress).
        # Reconciling it would "true-up" to a transient qty and create fresh
        # drift the moment the order completes. Skip such tickers this pass.
        active_order_syms = set()
        try:
            _ACTIVE = {"ApiPending", "PendingSubmit", "PreSubmitted",
                       "Submitted", "PendingCancel"}
            # MUST use reqAllOpenOrders(): the engine's orders are placed on a
            # DIFFERENT clientId, and ib.openTrades() only returns orders from
            # THIS session (clientId=8, which places none). reqAllOpenOrders()
            # returns working orders across all clients, which is what we need.
            ib.reqAllOpenOrders()
            ib.sleep(1)
            for tr in ib.openTrades():
                try:
                    st = tr.orderStatus.status if tr.orderStatus else None
                    if st in _ACTIVE and getattr(tr.contract, "symbol", None):
                        active_order_syms.add(tr.contract.symbol)
                except Exception:
                    continue
            if active_order_syms:
                print(f"  In-flight orders (skipped): {sorted(active_order_syms)}")
        except Exception as exc:
            # If we cannot determine open orders, be conservative: log it, but
            # do NOT skip everything (that would just disable reconciliation).
            summary["errors"].append(f"openTrades() failed: {exc}")
            print(f"  WARNING: could not read open orders: {exc}")

        # ── DB truth: open holdings aggregated by ticker ──────────────
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, ticker, entry_date, entry_price, quantity, peak_gain_pct "
            "FROM holdings WHERE sold_date IS NULL OR sold_date = '' "
            "ORDER BY ticker, id"
        ).fetchall()
        db_lots: dict = defaultdict(list)
        for r in rows:
            db_lots[r["ticker"]].append(dict(r))
        db_qty = {tk: sum(l["quantity"] for l in lots) for tk, lots in db_lots.items()}
        summary["db_open_lots"] = len(rows)
        print(f"  DB open holdings: {len(rows)} lot(s) across {len(db_lots)} ticker(s)")

        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # ── Plan the changes (no writes here) ─────────────────────────
        plan: list[dict] = []
        for ticker in sorted(set(broker) | set(db_lots)):
            if ticker in active_order_syms:
                print(f"  {ticker}: skipped — working order in flight (qty mid-change)")
                continue
            bqty, bcost = broker.get(ticker, (0.0, None))
            lots = db_lots.get(ticker, [])
            dqty = db_qty.get(ticker, 0.0)

            if bqty > 0 and not lots:
                # ORPHAN — broker holds it, no open DB lot.
                entry_date = _most_recent_buy_date(conn, ticker)
                marked_date = entry_date if entry_date else f"{now_ts} [RECON]"
                cur = get_reference_price(ib, ticker)
                peak = 0.0
                if cur and bcost and bcost > 0:
                    g = (cur - bcost) / bcost * 100.0
                    peak = round(g, 2) if g > 0 else 0.0
                if cur is None:
                    print(f"  {ticker}: ORPHAN — current price unavailable, peak seeded 0")
                plan.append({
                    "kind": "ORPHAN", "ticker": ticker, "qty": bqty, "cost": bcost,
                    "entry_date": marked_date, "peak": peak, "current": cur,
                })

            elif bqty <= 0 and lots:
                # PHANTOM — DB shows open, broker is flat. A sell went unrecorded.
                plan.append({
                    "kind": "PHANTOM", "ticker": ticker, "db_qty": dqty,
                    "lot_ids": [l["id"] for l in lots],
                })

            elif bqty > 0 and lots:
                qty_mismatch = abs(bqty - dqty) > QTY_TOL
                survivor = min(lots, key=lambda l: l["id"])
                other_ids = [l["id"] for l in lots if l["id"] != survivor["id"]]
                peak = round(max([l.get("peak_gain_pct") or 0.0 for l in lots] + [0.0]), 2)
                s_cost = survivor.get("entry_price")
                cost_mismatch = bool(
                    s_cost and bcost and bcost > 0
                    and abs(s_cost - bcost) / bcost * 100.0 > COST_TOL_PCT
                )
                if qty_mismatch or len(lots) > 1 or cost_mismatch:
                    plan.append({
                        "kind": "MISMATCH" if qty_mismatch else "CONSOLIDATE",
                        "ticker": ticker, "old_qty": dqty, "new_qty": bqty,
                        "old_cost": s_cost, "new_cost": bcost,
                        "survivor_id": survivor["id"], "other_ids": other_ids,
                        "peak": peak, "n_lots": len(lots),
                        "cost_mismatch": cost_mismatch,
                    })
                # else: single lot, qty + cost both agree → no action.

        if not plan:
            print("  ✓ No discrepancies — kairos.db agrees with the broker.")

        # ── Empty / implausible-response guard ────────────────────────
        # A real account does not lose all (or most) of its positions in a
        # single cycle; an empty/short broker response is far more likely a
        # flaky connection. In that case the plan would mass-mark phantoms.
        # Refuse to write and alert instead — broker snapshot is untrusted.
        phantom_ct = sum(1 for it in plan if it["kind"] == "PHANTOM")
        n_db_tickers = len(db_lots)
        bad_snapshot = False
        if len(broker) == 0 and n_db_tickers > 0:
            bad_snapshot = True
            reason = (f"broker returned 0 positions while DB holds {n_db_tickers} "
                      f"open tickers — treating snapshot as untrusted")
        elif (n_db_tickers >= MIN_DB_FOR_GUARD
              and phantom_ct > MAX_PHANTOM_FRACTION * n_db_tickers):
            bad_snapshot = True
            reason = (f"{phantom_ct}/{n_db_tickers} open tickers would be marked "
                      f"phantom (> {MAX_PHANTOM_FRACTION:.0%}) — implausible mass "
                      f"closure, treating snapshot as untrusted")
        if bad_snapshot:
            msg = f"SAFETY ABORT: {reason}. No writes performed."
            print(f"  *** {msg}")
            summary["errors"].append(msg)
            try:
                from kairos_alerts import post_message
                post_message("alerts",
                    f":rotating_light: *Position reconciliation SAFETY ABORT*\n{reason}. "
                    f"No DB writes were made. Likely a flaky IBKR response — "
                    f"will retry next cycle.")
            except Exception as exc:
                print(f"  (alert post failed: {exc})")
            conn.close()
            return summary

        # ── Back up before any write pass ─────────────────────────────
        if plan and not dry_run:
            _backup_kairos_db()

        # ── Apply / preview each planned change ───────────────────────
        for item in plan:
            kind = item["kind"]
            ticker = item["ticker"]
            try:
                if kind == "ORPHAN":
                    print(f"  {'[would create]' if dry_run else '[create]'} ORPHAN {ticker}: "
                          f"qty={item['qty']} @ ${item['cost']} cost, peak_seed={item['peak']}%, "
                          f"entry_date={item['entry_date']}")
                    if not dry_run:
                        conn.execute(
                            "INSERT INTO holdings (ticker, entry_date, entry_price, "
                            "quantity, peak_gain_pct) VALUES (?, ?, ?, ?, ?)",
                            (ticker, item["entry_date"], item["cost"],
                             item["qty"], item["peak"]),
                        )
                        conn.commit()
                    summary["created"].append({
                        "ticker": ticker, "qty": item["qty"], "avg_cost": item["cost"],
                    })

                elif kind in ("MISMATCH", "CONSOLIDATE"):
                    # Prefer broker truth; log the overwrite (a DB row losing qty
                    # may have held an unrecorded fill — broker still wins, loudly).
                    label = "QTY MISMATCH" if kind == "MISMATCH" else "CONSOLIDATE"
                    print(f"  {'[would true-up]' if dry_run else '[true-up]'} {label} {ticker}: "
                          f"db_qty={item['old_qty']} -> broker_qty={item['new_qty']}, "
                          f"cost ${item['old_cost']} -> ${item['new_cost']}, "
                          f"collapsing {item['n_lots']} lot(s) -> 1 "
                          f"(survivor #{item['survivor_id']}, "
                          f"merge {len(item['other_ids'])} lot(s))")
                    if item["old_qty"] > item["new_qty"]:
                        print(f"    NOTE: DB qty exceeded broker for {ticker} — "
                              f"overwriting to broker truth (possible unrecorded fill).")
                    if not dry_run:
                        conn.execute(
                            "UPDATE holdings SET quantity = ?, entry_price = ?, "
                            "peak_gain_pct = ? WHERE id = ?",
                            (item["new_qty"], item["new_cost"], item["peak"],
                             item["survivor_id"]),
                        )
                        for oid in item["other_ids"]:
                            # Mark merged lots sold at their own cost (zero realized
                            # P&L — no real economic exit) so they leave the open set
                            # without polluting the ledger. Do NOT delete.
                            conn.execute(
                                "UPDATE holdings SET sold_date = ?, "
                                "sold_price = entry_price WHERE id = ?",
                                (f"{now_ts} [RECON-merged]", oid),
                            )
                        conn.commit()
                    summary["updated"].append({
                        "ticker": ticker, "old_qty": item["old_qty"],
                        "new_qty": item["new_qty"], "old_cost": item["old_cost"],
                        "new_cost": item["new_cost"], "reason": label,
                    })
                    if item["other_ids"]:
                        summary["merged"].append({
                            "ticker": ticker, "lots_merged": len(item["other_ids"]),
                            "final_qty": item["new_qty"],
                        })

                elif kind == "PHANTOM":
                    print(f"  *** PHANTOM {ticker}: DB shows {item['db_qty']} open shares "
                          f"but broker is FLAT — a SELL went UNRECORDED. "
                          f"{'[would mark]' if dry_run else '[marking]'} "
                          f"{len(item['lot_ids'])} lot(s) sold. ***")
                    if not dry_run:
                        for lid in item["lot_ids"]:
                            # Broker is truth: the position is gone. Exit price is
                            # unknown (the sell wasn't recorded) → leave sold_price NULL.
                            conn.execute(
                                "UPDATE holdings SET sold_date = ? WHERE id = ?",
                                (f"{now_ts} [RECON-phantom]", lid),
                            )
                        conn.commit()
                    summary["phantom"].append({
                        "ticker": ticker, "db_qty": item["db_qty"],
                        "lots": len(item["lot_ids"]),
                    })
            except Exception as exc:
                msg = f"{kind} {ticker} write failed: {exc}"
                print(f"  ERROR: {msg}")
                summary["errors"].append(msg)

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    # ── Summary + Slack ──────────────────────────────────────────────
    n_created = len(summary["created"])
    n_updated = len(summary["updated"])
    n_phantom = len(summary["phantom"])
    n_merged = sum(m["lots_merged"] for m in summary["merged"])
    print(f"  Reconciliation {mode} complete: created={n_created} "
          f"updated={n_updated} phantom={n_phantom} merged_lots={n_merged} "
          f"errors={len(summary['errors'])}")

    # Post to #kairos-alerts on real (write) runs. Manual dry-runs print only
    # (they write nothing and should not page the channel).
    if not dry_run:
        try:
            from kairos_alerts import post_message
            lines = [
                ":scales: *Position Reconciliation* (IBKR = source of truth)",
                f"Broker STK positions: {summary['broker_positions']}  |  "
                f"created: {n_created}  updated: {n_updated}  "
                f"merged lots: {n_merged}  phantom: {n_phantom}",
            ]
            if summary["phantom"]:
                lines.append(":rotating_light: *PHANTOM (unrecorded exits — investigate):*")
                for ph in summary["phantom"]:
                    lines.append(f"  • {ph['ticker']}: DB had {ph['db_qty']} sh "
                                 f"open, broker flat ({ph['lots']} lot(s) closed)")
            if summary["created"]:
                lines.append("*Created (orphans now managed):* "
                             + ", ".join(f"{c['ticker']}×{c['qty']:g}" for c in summary["created"]))
            if summary["errors"]:
                lines.append(f":warning: {len(summary['errors'])} error(s) during reconcile")
            if not (summary["phantom"] or summary["created"]
                    or summary["updated"] or summary["merged"]):
                lines.append("✓ No discrepancies — DB agrees with broker.")
            post_message("alerts", "\n".join(lines))
        except Exception as exc:
            print(f"  reconcile_positions_against_broker: Slack post failed: {exc}")

    return summary


# ── NLV snapshots (daily account-value time series) ─────────────────

def _realized_pnl_cumulative(conn) -> float:
    """Cumulative realized P&L — single source of truth: the ML trade ledger.

    PRIMARY: SUM(pnl_dollar) over closed rows in kairos_ml_outcomes.db
    (trade_outcomes), opened READ-ONLY so this can never lock the ML DB against
    the writers in kairos_ml_outcomes.py. That ledger is the reconciled record
    of every attributable round trip, and each row's P&L is frozen at exit.

    FALLBACK (ML DB missing / locked / malformed): the legacy kairos.db lot sum
    of (sold_price - entry_price) * quantity over closed holdings, via the
    `conn` argument. This path is UNRELIABLE and exists only so callers such as
    write_nlv_snapshot never crash: the daily position reconciler rewrites and
    merges closed lots AT BROKER COST, so those lots contribute ~$0 and the sum
    silently SHRINKS over time (observed: nlv_snapshots.realized_pnl_cum fell
    34,956 on 2026-07-14 to 14,499 on 2026-07-29 while real P&L rose). Treat a
    fallback value as a floor, not a measurement.
    """
    ml_db = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
    if os.path.exists(ml_db):
        try:
            import sqlite3
            ml = sqlite3.connect(f"file:{ml_db}?mode=ro", uri=True, timeout=2)
            try:
                ml.execute("PRAGMA busy_timeout = 2000")
                row = ml.execute(
                    "SELECT COALESCE(SUM(pnl_dollar), 0.0) FROM trade_outcomes "
                    "WHERE timestamp_exit IS NOT NULL AND pnl_dollar IS NOT NULL"
                ).fetchone()
            finally:
                ml.close()
            return round(float(row[0] or 0.0), 2)
        except Exception as exc:
            print(f"  _realized_pnl_cumulative: ML ledger unavailable ({exc}) "
                  f"— falling back to kairos.db closed lots (understated)")

    row = conn.execute(
        "SELECT COALESCE(SUM((sold_price - entry_price) * quantity), 0.0) AS realized "
        "FROM holdings "
        "WHERE sold_date IS NOT NULL AND sold_date <> '' AND sold_price IS NOT NULL"
    ).fetchone()
    try:
        return round(float(row["realized"] or 0.0), 2)
    except (TypeError, ValueError, IndexError, KeyError):
        return 0.0


def nlv_snapshot_exists(snapshot_date: str) -> bool:
    """True if an nlv_snapshots row already exists for the given ET date."""
    from kairos_log_db import get_connection, init_db
    init_db()
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT 1 FROM nlv_snapshots WHERE snapshot_date = ?", (snapshot_date,)
        ).fetchone() is not None
    finally:
        conn.close()


def write_nlv_snapshot() -> dict | None:
    """Capture today's account value from IBKR into kairos.db nlv_snapshots.

    IBKR is the source of truth (consistent with the position reconciler).
    Connects READ-ONLY on a dedicated clientId=9 (distinct from 6/7/8), pulls
    NetLiquidation / TotalCashValue / UnrealizedPnL and counts open STK
    positions, and UPSERTs one row keyed on TODAY's ET date — so re-running the
    same day overwrites rather than duplicates. realized_pnl_cum comes from
    _realized_pnl_cumulative() — the ML trade ledger, with the kairos.db
    closed-lot sum only as a fallback. Returns the written row, or None on failure.
    """
    from zoneinfo import ZoneInfo
    from kairos_log_db import get_connection, init_db

    init_db()
    today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ib = IB()
    try:
        # Dedicated clientId=9 — distinct from cancel_stale_orders (6),
        # reconcile_submitted_orders (7), reconcile_positions (8).
        ib.connect("127.0.0.1", 7497, clientId=9, timeout=10)
    except Exception as exc:
        print(f"  write_nlv_snapshot: IBKR connect failed: {exc}")
        return None

    try:
        wanted = {"NetLiquidation", "TotalCashValue", "UnrealizedPnL"}
        acct: dict = {}
        try:
            for v in ib.accountSummary():
                if v.tag in wanted:
                    try:
                        acct[v.tag] = float(v.value)
                    except (TypeError, ValueError):
                        pass
        except Exception as exc:
            print(f"  write_nlv_snapshot: accountSummary failed: {exc}")

        nlv = acct.get("NetLiquidation")
        total_cash = acct.get("TotalCashValue")
        unrealized = acct.get("UnrealizedPnL")

        if nlv is None:
            print("  write_nlv_snapshot: NetLiquidation unavailable — aborting, no row written")
            return None

        num_positions = 0
        try:
            ib.reqPositions()
            ib.sleep(1)
            for p in ib.positions():
                if p.contract.secType == "STK" and abs(float(p.position)) > 1e-9:
                    num_positions += 1
        except Exception as exc:
            print(f"  write_nlv_snapshot: position count failed: {exc}")

        invested = round(nlv - total_cash, 2) if total_cash is not None else None

        conn = get_connection()
        realized_cum = _realized_pnl_cumulative(conn)
        try:
            conn.execute(
                "INSERT INTO nlv_snapshots (snapshot_date, nlv, total_cash, invested, "
                "num_positions, unrealized_pnl, realized_pnl_cum, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(snapshot_date) DO UPDATE SET "
                "nlv=excluded.nlv, total_cash=excluded.total_cash, "
                "invested=excluded.invested, num_positions=excluded.num_positions, "
                "unrealized_pnl=excluded.unrealized_pnl, "
                "realized_pnl_cum=excluded.realized_pnl_cum, created_at=excluded.created_at",
                (today_et, round(nlv, 2),
                 round(total_cash, 2) if total_cash is not None else None,
                 invested, num_positions,
                 round(unrealized, 2) if unrealized is not None else None,
                 realized_cum, created_at),
            )
            conn.commit()
        finally:
            conn.close()

        snap = {
            "snapshot_date": today_et,
            "nlv": round(nlv, 2),
            "total_cash": round(total_cash, 2) if total_cash is not None else None,
            "invested": invested,
            "num_positions": num_positions,
            "unrealized_pnl": round(unrealized, 2) if unrealized is not None else None,
            "realized_pnl_cum": realized_cum,
            "created_at": created_at,
        }
        cash_str = f"${total_cash:,.2f}" if total_cash is not None else "n/a"
        inv_str = f"${invested:,.2f}" if invested is not None else "n/a"
        print(f"  NLV snapshot written for {today_et}: NLV=${nlv:,.2f} "
              f"cash={cash_str} invested={inv_str} positions={num_positions} "
              f"realized_cum=${realized_cum:,.2f}")
        return snap
    except Exception as exc:
        print(f"  write_nlv_snapshot: FAILED: {exc}")
        return None
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


# ── main ────────────────────────────────────────────────────────────

def _get_conviction_boost(ticker: str) -> float:
    """Read a ticker's conviction_boost from kairos_screen_result.json.

    Returns 1.0 (no boost) when the file/key is absent or unreadable.
    Populated by the screener for HOT-CHAIN tickers (Tier 2 → 1.2x,
    Tier 3 → 1.3x).
    """
    path = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")
    if not os.path.exists(path):
        return 1.0
    try:
        with open(path) as f:
            boosts = json.load(f).get("conviction_boosts") or {}
        return float(boosts.get(ticker, 1.0))
    except (json.JSONDecodeError, IOError, TypeError, ValueError):
        return 1.0


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

    # ── Trim triggers (oversized + underwater positions) ──────────
    print(banner("Trim Check"))
    trims = check_trim_triggers(portfolio, ib, nlv)
    if trims:
        print(f"  Executed {len(trims)} trim(s)")
        # Refresh portfolio state after trims
        portfolio = fetch_portfolio_state(ib)
        nlv = portfolio["nlv"]
        cash = portfolio["cash"]
    else:
        print("  No trim triggers.")

    # ── Execute each trade with pre-execution guardrail check ─────
    executed_count = 0
    skipped_count = 0
    total_spent = 0.0
    skip_reasons: list[dict] = []  # [{ticker, reason}] for cycle summary
    conviction_taken = False          # Mode C: only one conviction trade per cycle
    conviction_sectors: set[str] = set()  # Mode C: one per sector per cycle

    paused = _is_paused()

    for i, trade in enumerate(actionable, 1):
        action = trade.get("action", "HOLD").upper()
        ticker = trade.get("ticker", "?")

        print(banner(f"Trade {i}/{len(actionable)}: {action} {ticker}"))

        # ── Pause flag (commander !pause) — block BUYs only ──────
        # SELLs and HOLDs still execute so stop-loss + risk management
        # remain effective while the pipeline is "paused".
        if action == "BUY" and paused:
            reason = "Pipeline paused via !pause — BUY blocked"
            print(f"    SKIP: {reason}")
            skipped_count += 1
            skip_reasons.append({"ticker": ticker, "reason": reason})
            log_execution(decision, trade, {"status": "Skipped", "reason": reason})
            continue

        # ── Duplicate-order guard (BUY only) ──────────────────────
        if action == "BUY" and ticker in get_open_order_tickers(ib):
            msg = f"SKIP {ticker}: open order already pending"
            print(f"    {msg}")
            skipped_count += 1
            skip_reasons.append({"ticker": ticker, "reason": "open order already pending"})
            log_execution(decision, trade, {"status": "Skipped", "reason": "open order already pending"})
            continue

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

        # Use the trade's REAL triggering signals (HOT-REVERSION / score-route
        # entries are NOT in kairos_signal_summary.json, so the bare
        # get_ticker_signals returns [] for them → confluence 0 → qty 0 → Skipped,
        # leaving cash idle). _derive_entry_signals falls back to
        # get_ticker_signals internally (priority 3), so normal screener trades
        # are unaffected. This makes sizing consistent with the logging path.
        signal_tags = _derive_entry_signals(trade, ticker)
        confluence = compute_confluence(signal_tags)

        # ── ROOT SIZING: entry sizing is for ENTRIES only ────────────
        # compute_position_size() answers "how much should we BUY" — it is a
        # function of confluence tier x NLV x price and knows nothing about
        # what is held. Applying it to a SELL is a category error, and it
        # produced both failure modes seen live:
        #   • oversell — 2026-07-29 ETN sold 44 vs 28 held, EME 23 vs 15
        #     (1.5% NLV / price), leaving the long-only account SHORT;
        #   • dropped exits — 10 Council SELLs sized to 0 shares (confluence 0)
        #     and were skipped as "Quantity rounds to 0", so the exit never ran.
        # A Council SELL is a full exit of the CURRENT position, which is what
        # every other exit path in the system already does (total_qty).
        if action == "SELL":
            from kairos_sell_guard import get_held_quantity
            held, held_source = get_held_quantity(ticker, ib)
            if held is None:
                print(f"    SKIP: cannot determine held quantity for {ticker} "
                      f"— SELL not sized")
                skipped_count += 1
                skip_reasons.append({"ticker": ticker,
                                     "reason": "held quantity unavailable"})
                log_execution(decision, trade, {
                    "status": "Skipped",
                    "reason": "held quantity unavailable",
                })
                continue
            qty = int(held)
            print(f"    SELL sizing: full exit of {qty} held shares "
                  f"(source: {held_source})")
            if qty < 1:
                # Nothing held — never fall through to Mode C (a BUY path).
                reason = f"SELL skipped: no open position in {ticker} (held {held:g})"
                print(f"    SKIP: {reason}")
                skipped_count += 1
                skip_reasons.append({"ticker": ticker, "reason": reason})
                log_execution(decision, trade, {"status": "Skipped", "reason": reason})
                continue
        else:
            qty = compute_position_size(confluence, nlv, ref_price)

        # ── Re-entry guard (BUY only) ─────────────────────────────
        # Don't re-buy a ticker ABOVE its last exit price unless a genuinely
        # new signal has fired since that exit (Exit Architecture v2). Re-entry
        # at/below the exit price is always allowed; there is no time cooldown.
        if action == "BUY":
            try:
                from kairos_exits import reentry_guard_blocks
                blocked, reentry_reason = reentry_guard_blocks(ticker, ref_price, signal_tags)
                if blocked:
                    print(f"    BLOCKED: {reentry_reason}")
                    skipped_count += 1
                    skip_reasons.append({"ticker": ticker, "reason": reentry_reason})
                    log_execution(decision, trade, {"status": "Skipped", "reason": reentry_reason})
                    try:
                        from kairos_alerts import post_message
                        # Routine BUY-gating decision (a skip), not a health item → #log.
                        post_message("log",
                            f":no_entry_sign: *Re-Entry Guard: {ticker}*\n{reentry_reason}")
                    except Exception as exc:
                        print(f"  re-entry guard Slack post failed: {exc}")
                    continue
            except Exception as reentry_exc:
                print(f"    WARNING: re-entry guard check failed: {reentry_exc}")

        # Chain conviction_boost (HOT-CHAIN: Tier 2 → 1.2x, Tier 3 → 1.3x),
        # applied like the regime / weight multipliers below.
        conviction_boost = _get_conviction_boost(ticker)
        if action == "BUY" and conviction_boost > 1.0 and qty > 0:
            original_qty = qty
            qty = max(1, int(qty * conviction_boost))
            if qty != original_qty:
                print(f"    Chain boost: {conviction_boost:.1f}x conviction_boost "
                      f"({original_qty} → {qty} shares)")

        # Gate 2: Apply max_position_mult from regime guardrails
        pos_mult = regime_g.get("max_position_mult", 1.0)
        if action == "BUY" and pos_mult < 1.0 and qty > 0:
            original_qty = qty
            qty = max(1, int(qty * pos_mult))
            if qty != original_qty:
                print(f"    Regime sizing: {regime_name} → {int(pos_mult*100)}% cap "
                      f"({original_qty} → {qty} shares)")

        # ── Unrealized loss gate (BUY adds only) ─────────────────
        if action == "BUY" and ticker in portfolio["positions"]:
            existing_avg = portfolio["positions"][ticker]["avg_cost"]
            if existing_avg > 0 and ref_price is not None:
                unrealized_pnl_pct = (ref_price - existing_avg) / existing_avg
                if unrealized_pnl_pct < -0.05:
                    reason = (f"Unrealized loss gate: {ticker} at {unrealized_pnl_pct:.1%} "
                              f"(< -5%), blocking add entirely")
                    print(f"    BLOCKED: {reason}")
                    skipped_count += 1
                    skip_reasons.append({"ticker": ticker, "reason": reason})
                    log_execution(decision, trade, {"status": "Skipped", "reason": reason})
                    continue
                elif unrealized_pnl_pct < -0.02:
                    # Between -2% and -5%: require confluence score ≥ 4
                    if confluence["score"] < 4:
                        reason = (f"Unrealized loss gate: {ticker} at {unrealized_pnl_pct:.1%}, "
                                  f"confluence {confluence['score']} < 4 required")
                        print(f"    BLOCKED: {reason}")
                        skipped_count += 1
                        skip_reasons.append({"ticker": ticker, "reason": reason})
                        log_execution(decision, trade, {"status": "Skipped", "reason": reason})
                        continue
                    else:
                        print(f"    Unrealized loss: {unrealized_pnl_pct:.1%} but confluence "
                              f"{confluence['score']} ≥ 4, proceeding")

        # ── Portfolio-weight penalty on adds (BUY only) ───────────
        if action == "BUY" and qty > 0 and nlv > 0:
            current_pos_val = portfolio["positions"].get(ticker, {}).get("market_value", 0)
            current_weight = current_pos_val / nlv
            if current_weight > 0.06:
                # Linear scale: 6%→full, 8%→50%, 10%→10%
                if current_weight >= 0.10:
                    weight_mult = 0.10
                elif current_weight >= 0.08:
                    # Linear from 0.50 at 8% to 0.10 at 10%
                    weight_mult = 0.50 - (current_weight - 0.08) / 0.02 * 0.40
                else:
                    # Linear from 1.0 at 6% to 0.50 at 8%
                    weight_mult = 1.0 - (current_weight - 0.06) / 0.02 * 0.50
                original_qty = qty
                qty = max(1, int(qty * weight_mult))
                if qty != original_qty:
                    print(f"    Weight penalty: position at {current_weight:.1%} of NLV → "
                          f"{weight_mult:.0%} sizing ({original_qty} → {qty} shares)")

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

                # A 0-share result on a worthwhile BUY is EXACTLY when capital
                # should be liberated: low cash collapsed sizing to nothing. Route
                # genuine "rounds to 0" cases into the SAME reallocation evaluation
                # the cash-breach path uses (not the regime-gate skips above).
                if (action == "BUY" and reason == "Quantity rounds to 0"
                        and conviction_score >= _reallocation_min_conviction()):
                    executed, spent = _try_reallocation(
                        ticker, conviction_score, signal_tags, ib, nlv,
                        trade, qty, decision, path="rounds_to_0")
                    if executed:
                        executed_count += 1
                        total_spent += spent
                        continue  # trade was reallocated — skip the normal skip path

                print(f"    SKIP: {reason}")
                skipped_count += 1
                skip_reasons.append({"ticker": ticker, "reason": reason})
                log_execution(decision, trade, {"status": "Skipped", "reason": reason})
                continue

        proposed_spend = qty * ref_price

        # ── Pre-execution portfolio check ─────────────────────────
        # Refresh portfolio state before each order
        portfolio = fetch_portfolio_state(ib)

        # check_guardrails is an ENTRY gate: all three checks model
        # proposed_spend as capital being ADDED (cash reserve after spending,
        # current sector value + spend, current position value + spend). A SELL
        # frees capital and shrinks the position, so those checks are backwards
        # for it. This was latent while SELLs carried a small entry-sized
        # spend; now that a SELL is sized at the full held position, running
        # them would let a concentration limit BLOCK an exit. Entry gates apply
        # to entries only.
        passed, failures = True, []
        if action == "BUY":
            passed, failures = check_guardrails(ticker, proposed_spend, portfolio)

        if not passed:
            print(f"    GUARDRAIL BREACH — skipping {ticker}:")
            for reason in failures:
                print(f"      • {reason}")

            # ── Reallocation check: any cash-related breach on a BUY ─────
            # Match ALL cash-failure phrasings (see _CASH_FAIL_MARKERS, kept in
            # sync with kairos_confluence.check_cash_reserve), not just the soft
            # reserve-floor string — in a near-zero-cash regime trades fail at the
            # negative/insufficient-cash strings first.
            has_cash_breach = any(
                any(m in f.lower() for m in _CASH_FAIL_MARKERS) for f in failures
            )
            new_conviction = trade.get("conviction", 0)
            if isinstance(new_conviction, str):
                try:
                    new_conviction = int(new_conviction)
                except (ValueError, TypeError):
                    new_conviction = 0

            if action == "BUY" and has_cash_breach and new_conviction >= _reallocation_min_conviction():
                executed, spent = _try_reallocation(
                    ticker, new_conviction, signal_tags, ib, nlv,
                    trade, qty, decision, path="cash_breach")
                if executed:
                    executed_count += 1
                    total_spent += spent
                    continue  # skip the normal skip path — trade was reallocated

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
                        # Routine BUY-gating decision that PREVENTS a wash sale (a skip,
                        # not a realized tax event) → #log. (Contrast: an actual wash-sale
                        # VIOLATION on a sell is a distinct attention item kept in #alerts.)
                        post_message("log",
                            f":no_entry: *Wash Sale Block: {ticker}*\n"
                            f"BUY blocked — loss sale of ${ws['loss_amount']:,.2f} on {ws['sell_date']}\n"
                            f"Blocked until: {ws['blocked_until']}")
                    except Exception as exc:
                        print(f"  wash-sale block Slack post failed: {exc}")
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
    import argparse
    parser = argparse.ArgumentParser(description="Kairos Executor")
    parser.add_argument("--reconcile-positions", action="store_true",
                        help="Reconcile kairos.db holdings against IBKR instead of "
                             "running the executor (broker = source of truth)")
    parser.add_argument("--execute", action="store_true",
                        help="With --reconcile-positions: WRITE changes to kairos.db. "
                             "Omitted = dry-run (print only, the safe default).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Explicit dry-run for --reconcile-positions (default).")
    parser.add_argument("--write-nlv-snapshot", action="store_true",
                        help="Capture today's account value from IBKR into "
                             "kairos.db nlv_snapshots (read-only on IBKR) and exit.")
    cli_args, _ = parser.parse_known_args()

    if cli_args.write_nlv_snapshot:
        write_nlv_snapshot()
    elif cli_args.reconcile_positions:
        # Dry-run is the default; only a deliberate --execute writes.
        reconcile_positions_against_broker(dry_run=not cli_args.execute)
    else:
        main()
