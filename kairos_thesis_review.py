"""
Kairos Thesis Review — Daily Thesis Invalidation + Take-Profit + Stale Check

Runs daily at 9:35am ET (scheduled via kairos_run.py). For each open position:

  1. THESIS-INVALID: Sends position context to Claude Code and asks whether
     the original entry thesis has been invalidated. If YES → SELL.
  2. TAKE-PROFIT: If unrealized gain >15% with no fresh signals in 7 days → SELL.
  3. STALE-THESIS: If held >30 days with return <2% and no active signals → SELL.

All sells are logged to kairos.db with trigger type, P&L, holding days,
and tax classification.

Usage:
    python kairos_thesis_review.py              # Run full review
    python kairos_thesis_review.py --dry-run    # Evaluate but don't sell
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

W = 72

TAKE_PROFIT_PCT = 15.0       # unrealized gain threshold
TAKE_PROFIT_SIGNAL_DAYS = 7  # no fresh signals within this window
STALE_DAYS = 30              # minimum hold period for stale check
STALE_RETURN_PCT = 2.0       # cumulative return below this = stale

# Per-run audit log. Every reviewed (run_id, ticker) gets a row here,
# regardless of whether a sell triggered, so the table is also a record
# of "we looked and decided to hold". Sells additionally go through
# kairos_stoploss._log_sell into the decisions table — unchanged.
SCHEMA_THESIS_REVIEWS = """
CREATE TABLE IF NOT EXISTS thesis_reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    avg_cost        REAL,
    current_price   REAL,
    return_pct      REAL,
    holding_days    INTEGER,
    current_signals TEXT,
    sell_triggered  INTEGER NOT NULL DEFAULT 0,
    trigger_type    TEXT,
    sell_reason     TEXT,
    dry_run         INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_thesis_reviews_run_id
    ON thesis_reviews(run_id);
CREATE INDEX IF NOT EXISTS idx_thesis_reviews_ticker
    ON thesis_reviews(ticker);
"""


def _init_thesis_reviews_table() -> None:
    """Idempotently create the thesis_reviews table + indexes."""
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        conn.executescript(SCHEMA_THESIS_REVIEWS)
        conn.commit()
        conn.close()
    except Exception as exc:
        # Surface but don't kill the run — review can still execute and
        # log via stoploss path even if our audit table can't be created.
        print(f"  WARNING: thesis_reviews table init failed: {exc}")


def _log_review_row(run_id: str, ticker: str, avg_cost: float,
                    current_price: float, return_pct: float,
                    holding_days: int, current_signals: list[str],
                    sell_reason: str | None, dry_run: bool) -> None:
    """Insert one audit row per (run, ticker) reviewed."""
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        trigger_type = (sell_reason.split(":", 1)[0] if sell_reason else None)
        conn.execute(
            "INSERT INTO thesis_reviews "
            "(timestamp, run_id, ticker, avg_cost, current_price, "
            " return_pct, holding_days, current_signals, "
            " sell_triggered, trigger_type, sell_reason, dry_run) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                run_id, ticker, avg_cost, current_price,
                round(return_pct, 4), holding_days,
                json.dumps(current_signals or []),
                1 if sell_reason else 0,
                trigger_type,
                sell_reason,
                1 if dry_run else 0,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"    WARNING: thesis_reviews insert failed for {ticker}: {exc}")


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


def _get_ibkr_price(ib, ticker: str) -> float | None:
    """Get current price via an existing IBKR connection."""
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


def _get_current_signals(ticker: str) -> list[str]:
    """Load current signal tags for a ticker from kairos_signal_summary.json."""
    summary_file = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
    try:
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                data = json.load(f)
            return list(data.get("signal_tags", {}).get(ticker, []))
    except (IOError, json.JSONDecodeError):
        pass
    return []


def _get_entry_signals(ticker: str) -> list[str]:
    """Extract entry signals from the most recent BUY decision in kairos.db."""
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
            # Entry signals may be in confluence.signals or at top level
            conf = di.get("confluence", {})
            if conf.get("signals"):
                return conf["signals"]
            if di.get("signals"):
                return di["signals"]
    except Exception:
        pass
    return []


def _get_entry_rationale(ticker: str) -> str:
    """Get the entry rationale from the most recent BUY decision."""
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        row = conn.execute(
            "SELECT rationale FROM decisions "
            "WHERE ticker = ? AND action = 'BUY' AND execution_status = 'Filled' "
            "ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        conn.close()
        if row and row["rationale"]:
            return row["rationale"]
    except Exception:
        pass
    return "(no rationale recorded)"


def _ask_claude_thesis(ticker: str, entry_rationale: str,
                       entry_signals: list[str], current_signals: list[str],
                       current_price: float, entry_price: float,
                       holding_days: int) -> tuple[bool, str]:
    """Ask Claude Code whether the entry thesis has been invalidated.

    Returns (invalidated: bool, explanation: str).
    """
    if entry_price is None or entry_price == 0:
        pnl_pct = 0
    else:
        pnl_pct = (current_price - entry_price) / entry_price * 100

    prompt = (
        f"You are reviewing an open equity position for thesis invalidation.\n\n"
        f"Ticker: {ticker}\n"
        f"Entry price: ${entry_price:.2f}\n"
        f"Current price: ${current_price:.2f} ({pnl_pct:+.1f}%)\n"
        f"Holding period: {holding_days} days\n"
        f"Entry signals: {', '.join(entry_signals) if entry_signals else 'none recorded'}\n"
        f"Current signals: {', '.join(current_signals) if current_signals else 'none active'}\n"
        f"Original entry thesis: {entry_rationale}\n\n"
        f"Has the original entry thesis been invalidated? "
        f"Respond with EXACTLY one line: YES or NO followed by a one-sentence explanation.\n"
        f"Example: YES — Insider selling pressure has reversed the bullish thesis.\n"
        f"Example: NO — Original momentum thesis remains intact with rising volume."
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=[{
                "type": "text",
                "text": (
                    "You are reviewing open equity positions for thesis invalidation. "
                    "Respond with EXACTLY one line: YES or NO followed by a one-sentence explanation."
                ),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        text = response.content[0].text.strip()
        if not text:
            return False, "Claude returned empty response"

        first_line = text.split("\n")[0].strip()
        if first_line.upper().startswith("YES"):
            explanation = first_line[3:].strip().lstrip("—–- ").strip()
            return True, explanation or "Thesis invalidated (no explanation)"
        else:
            return False, first_line

    except Exception as exc:
        return False, f"Claude API error: {exc}"


def _execute_sell(ib, ticker: str, qty: int) -> dict:
    """Place a market SELL order via IBKR."""
    from kairos_stoploss import _place_market_sell
    return _place_market_sell(ib, ticker, qty)


def _log_and_alert(ticker: str, qty: int, entry_price: float, sell_price: float,
                   holding_days: int, reason: str, execution: dict) -> None:
    """Log sell to DB and alert Slack."""
    from kairos_stoploss import _log_sell, _alert_stoploss
    drawdown_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

    _log_sell(ticker, qty, entry_price, sell_price, holding_days, reason, execution)

    pnl = (sell_price - entry_price) * qty
    tax_class = "long-term" if holding_days >= 365 else "short-term"

    # Determine emoji based on trigger
    if "TAKE-PROFIT" in reason:
        emoji = ":moneybag:"
    elif "STALE" in reason:
        emoji = ":hourglass:"
    elif "THESIS" in reason:
        emoji = ":mag:"
    else:
        emoji = ":chart_with_downwards_trend:"

    try:
        from kairos_alerts import post_message
        post_message("alerts",
            f"{emoji} *Thesis Review Sell: {ticker}*\n"
            f"Entry: ${entry_price:.2f} → Exit: ${sell_price:.2f} ({drawdown_pct:+.1f}%)\n"
            f"P&L: ${pnl:+,.2f} | Held {holding_days}d ({tax_class})\n"
            f"Reason: {reason}"
        )
    except Exception as exc:
        print(f"    WARNING: Slack alert failed: {exc}")


def run_thesis_review(dry_run: bool = False, no_claude: bool = False) -> dict:
    """Run thesis review across all open equity positions.

    Args:
        dry_run: Evaluate but do not place SELL orders.
        no_claude: Skip the Claude thesis-invalidation check entirely.
            Useful when (a) Claude CLI is unavailable, or (b) you want a
            fast verification run that exercises only the deterministic
            take-profit / stale checks. Take-profit and stale-thesis
            triggers are still applied normally.

    Returns summary dict with counts and sell details. Per-ticker review
    rows are persisted to the `thesis_reviews` table regardless of
    whether a sell triggered, so the table is a complete audit trail.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS THESIS REVIEW — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    if dry_run:
        print("  *** DRY RUN — will evaluate but not sell ***")
    if no_claude:
        print("  *** NO-CLAUDE — skipping thesis-invalidation check ***")

    # Idempotent table init — first run creates thesis_reviews + indexes.
    _init_thesis_reviews_table()

    # Load open holdings grouped by ticker
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
        print("  No open positions — nothing to review")
        return {"reviewed": 0, "sells": 0, "details": []}

    print(f"  Reviewing {len(holdings)} open positions")

    # Connect to IBKR
    ib = None
    if not dry_run:
        try:
            from ib_insync import IB
            import random
            ib = IB()
            ib.connect("127.0.0.1", 7497, clientId=random.randint(30, 39), timeout=10)
        except Exception as exc:
            print(f"  WARNING: IBKR unavailable ({exc}) — will evaluate but can't execute")
            ib = None

    result = {"reviewed": 0, "sells": 0, "details": []}
    harvest_candidates_data = []  # Positions for tax loss harvest evaluation

    for h in holdings:
        ticker = h["ticker"]
        total_qty = int(h["total_qty"])
        avg_cost = h["avg_cost"]
        holding_days = h["holding_days"] or 0
        result["reviewed"] += 1

        # Get current price
        current_price = None
        if ib:
            current_price = _get_ibkr_price(ib, ticker)

        if current_price is None:
            # Fallback: try Finnhub
            try:
                import requests
                api_key = os.environ.get("FINNHUB_API_KEY")
                if api_key:
                    resp = requests.get(
                        "https://finnhub.io/api/v1/quote",
                        params={"symbol": ticker, "token": api_key},
                        timeout=10,
                    )
                    data = resp.json()
                    if data.get("c", 0) > 0:
                        current_price = float(data["c"])
            except Exception:
                pass

        if current_price is None:
            print(f"    {ticker}: no price — skipped")
            continue

        if avg_cost is None or avg_cost == 0:
            print(f"    {ticker}: invalid avg_cost ({avg_cost}) — skipped")
            continue

        return_pct = (current_price - avg_cost) / avg_cost * 100
        current_signals = _get_current_signals(ticker)

        print(f"\n  {ticker}: ${avg_cost:.2f}→${current_price:.2f} ({return_pct:+.1f}%) "
              f"held {holding_days}d, signals={current_signals or 'none'}")

        sell_reason = None

        # ── Check 1: Take-profit ─────────────────────────────────
        if return_pct is not None and return_pct >= TAKE_PROFIT_PCT and not current_signals:
            sell_reason = (f"TAKE-PROFIT: {return_pct:.1f}% gain with no fresh signals "
                          f"(threshold {TAKE_PROFIT_PCT}%)")
            print(f"    → {sell_reason}")

        # ── Check 2: Stale thesis ────────────────────────────────
        if sell_reason is None and holding_days >= STALE_DAYS:
            if return_pct is not None and return_pct < STALE_RETURN_PCT and not current_signals:
                sell_reason = (f"STALE-THESIS: {return_pct:.1f}% return after {holding_days}d "
                              f"with no active signals")
                print(f"    → {sell_reason}")

        # ── Check 3: Claude thesis invalidation ──────────────────
        if sell_reason is None and not no_claude:
            entry_signals = _get_entry_signals(ticker)
            entry_rationale = _get_entry_rationale(ticker)
            print(f"    Asking Claude: thesis still valid?")

            invalidated, explanation = _ask_claude_thesis(
                ticker, entry_rationale, entry_signals, current_signals,
                current_price, avg_cost, holding_days)

            if invalidated:
                sell_reason = f"THESIS-INVALID: {explanation}"
                print(f"    → Claude: YES — {explanation}")
            else:
                print(f"    → Claude: NO — {explanation[:80]}")
        elif sell_reason is None and no_claude:
            print(f"    Skipping Claude check (--no-claude)")

        # ── Audit log: one row per (run, ticker) regardless of outcome ──
        _log_review_row(
            run_id=run_id, ticker=ticker, avg_cost=avg_cost,
            current_price=current_price, return_pct=return_pct,
            holding_days=holding_days, current_signals=current_signals,
            sell_reason=sell_reason, dry_run=dry_run,
        )

        # ── Collect data for tax loss harvest evaluation ──────────
        if sell_reason is None and current_price is not None and return_pct is not None and return_pct < 0:
            harvest_candidates_data.append({
                "ticker": ticker,
                "total_qty": total_qty,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "holding_days": holding_days,
            })

        # ── Execute sell if triggered ────────────────────────────
        if sell_reason:
            if dry_run:
                print(f"    DRY RUN — would sell {total_qty} {ticker}")
                result["sells"] += 1
                result["details"].append({
                    "ticker": ticker,
                    "reason": sell_reason,
                    "qty": total_qty,
                    "return_pct": round(return_pct, 1),
                    "holding_days": holding_days,
                    "action": "DRY_RUN",
                })
            elif ib:
                print(f"    SELLING {total_qty} {ticker}")
                execution = _execute_sell(ib, ticker, total_qty)
                sell_price = execution.get("fill_price", current_price)
                _log_and_alert(ticker, total_qty, avg_cost, sell_price,
                              holding_days, sell_reason, execution)

                # Wash sale violation check — loss sell with recent repurchase
                if sell_price < avg_cost:
                    try:
                        from kairos_wash_sale import (check_wash_sale_violation,
                                                      log_wash_sale_event)
                        from datetime import timedelta
                        ws = check_wash_sale_violation(ticker, sell_price, avg_cost)
                        if ws["violation"]:
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
                                    f"Loss of ${loss_amt:,.2f} disallowed (thesis review sell)\n"
                                    f"Recent repurchase on {ws['repurchase_date']}\n"
                                    f"Sell executed — loss cannot be claimed until {blocked_until}")
                            except Exception:
                                pass
                    except Exception as ws_exc:
                        print(f"    WARNING: Wash sale check failed: {ws_exc}")

                result["sells"] += 1
                result["details"].append({
                    "ticker": ticker,
                    "reason": sell_reason,
                    "qty": total_qty,
                    "exit_price": sell_price,
                    "return_pct": round(return_pct, 1),
                    "holding_days": holding_days,
                    "status": execution.get("status", "?"),
                })
            else:
                print(f"    IBKR unavailable — sell logged but not executed")
                result["details"].append({
                    "ticker": ticker,
                    "reason": sell_reason,
                    "qty": total_qty,
                    "action": "IBKR_UNAVAILABLE",
                })

    # ── Tax loss harvest evaluation ──────────────────────────────
    if harvest_candidates_data:
        try:
            from kairos_tax_harvest import evaluate_harvest_candidates, execute_harvest
            harvests = evaluate_harvest_candidates(harvest_candidates_data, dry_run=dry_run)
            for rec in harvests:
                execution = execute_harvest(rec, ib, dry_run=dry_run)
                if execution and execution.get("status") in ("Filled", "Submitted"):
                    result["sells"] += 1
                    result["details"].append({
                        "ticker": rec["ticker"],
                        "reason": rec["reason"],
                        "qty": rec["total_qty"],
                        "exit_price": execution.get("fill_price", rec["current_price"]),
                        "return_pct": round(-rec["loss_pct"], 1),
                        "holding_days": rec["holding_days"],
                        "status": execution.get("status", "?"),
                    })
        except Exception as exc:
            import traceback
            print(f"  WARNING: Tax loss harvest failed: {exc}")
            print(traceback.format_exc())

    # ── Reallocation outcome tracking ────────────────────────────────
    try:
        from kairos_reallocation_tracker import evaluate_reallocation_outcomes, send_weekly_summary
        outcomes = evaluate_reallocation_outcomes(ib=ib)
        if outcomes:
            print(f"\n  Reallocation outcomes tracked: {len(outcomes)}")
            for o in outcomes:
                print(f"    {o['ticker_exited']}→{o['ticker_entered']} "
                      f"({o['days_since']}d): delta={o['perf_delta']:+.1f}% {o['verdict']}")
        # Sunday weekly summary
        send_weekly_summary(outcomes if outcomes else None)
    except Exception as exc:
        print(f"  WARNING: Reallocation tracking failed: {exc}")

    if ib:
        ib.disconnect()

    print(banner("Thesis Review Summary"))
    print(f"  Reviewed: {result['reviewed']}")
    print(f"  Sells triggered: {result['sells']}")
    for d in result["details"]:
        trigger = d["reason"].split(":")[0]
        print(f"    {d['ticker']:6s} {trigger} ({d.get('return_pct', '?'):+}%)")

    return result


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kairos Thesis Review")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate but don't execute sells")
    parser.add_argument("--no-claude", action="store_true",
                        help="Skip the Claude thesis-invalidation check "
                             "(fast path: only deterministic take-profit "
                             "and stale-thesis triggers apply)")
    args = parser.parse_args()

    result = run_thesis_review(dry_run=args.dry_run, no_claude=args.no_claude)
