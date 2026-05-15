"""
Kairos Reallocation Outcome Tracker

Evaluates whether past reallocations were the right call by comparing:
  - Counterfactual: what would the exited ticker have returned?
  - Actual: what has the entered ticker returned?

Runs inside kairos_thesis_review.py's daily 9:35 AM cycle.
After 30 days, writes a ledger entry with the combined verdict.
Sends a weekly summary to #kairos-reports every Sunday.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")

SCHEMA_REALLOCATION_OUTCOMES = """
CREATE TABLE IF NOT EXISTS reallocation_outcomes (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    reallocation_event_id           INTEGER NOT NULL,
    ticker_exited                   TEXT NOT NULL,
    ticker_entered                  TEXT NOT NULL,
    days_since_reallocation         INTEGER NOT NULL,
    exit_ticker_counterfactual_return REAL NOT NULL,
    entry_ticker_actual_return      REAL NOT NULL,
    performance_delta               REAL NOT NULL,
    verdict                         TEXT NOT NULL,
    ledger_written                  INTEGER NOT NULL DEFAULT 0,
    eval_date                       TEXT NOT NULL,
    created_at                      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Minimum days before we start tracking outcomes
MIN_TRACKING_DAYS = 14
# Days after which we write the final ledger entry
LEDGER_WRITE_DAYS = 30


def _get_connection():
    from kairos_log_db import get_connection
    return get_connection()


def init_outcome_tables() -> None:
    """Create the reallocation_outcomes table."""
    conn = _get_connection()
    conn.executescript(SCHEMA_REALLOCATION_OUTCOMES)
    conn.commit()
    conn.close()


def _get_price(ib, ticker: str) -> float | None:
    """Get current price via IBKR (if connected) or Finnhub fallback."""
    if ib:
        try:
            from kairos_thesis_review import _get_ibkr_price
            price = _get_ibkr_price(ib, ticker)
            if price:
                return price
        except Exception:
            pass

    # Finnhub fallback
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
                return float(data["c"])
    except Exception:
        pass

    return None


def _get_exit_price_at_reallocation(event) -> float | None:
    """Get the price at which the exit ticker was sold.

    Uses the execution_price from the decisions table for the SELL
    that matches this reallocation event.
    """
    try:
        conn = _get_connection()
        row = conn.execute(
            """SELECT execution_price FROM decisions
               WHERE ticker = ? AND action = 'SELL'
               AND rationale LIKE '%REALLOCATION%'
               AND date(timestamp) = date(?)
               ORDER BY id DESC LIMIT 1""",
            (event["ticker_exited"], event["eval_date"]),
        ).fetchone()
        conn.close()
        if row and row["execution_price"]:
            return float(row["execution_price"])
    except Exception:
        pass
    return None


def _get_entry_price_at_reallocation(event) -> float | None:
    """Get the price at which the entry ticker was bought."""
    try:
        conn = _get_connection()
        row = conn.execute(
            """SELECT execution_price FROM decisions
               WHERE ticker = ? AND action = 'BUY'
               AND date(timestamp) = date(?)
               ORDER BY id DESC LIMIT 1""",
            (event["ticker_entered"], event["eval_date"]),
        ).fetchone()
        conn.close()
        if row and row["execution_price"]:
            return float(row["execution_price"])
    except Exception:
        pass
    return None


def evaluate_reallocation_outcomes(ib=None) -> list[dict]:
    """Evaluate all mature reallocation pairs (≥14 days old).

    For each pair:
      1. Fetch current prices for both tickers
      2. Calculate counterfactual return (exit ticker) and actual return (entry ticker)
      3. Compute performance delta
      4. Write to reallocation_outcomes table
      5. At 30+ days, write final ledger entry

    Returns list of outcome dicts for reporting.
    """
    init_outcome_tables()

    conn = _get_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=MIN_TRACKING_DAYS)).strftime("%Y-%m-%d")

    # Find executed reallocations ≥14 days old
    events = conn.execute(
        """SELECT id, eval_date, ticker_exited, ticker_entered, conviction_delta
           FROM reallocation_events
           WHERE executed = 1
             AND recommended = 1
             AND ticker_exited IS NOT NULL
             AND date(eval_date) <= date(?)
           ORDER BY eval_date""",
        (cutoff,),
    ).fetchall()
    conn.close()

    if not events:
        return []

    outcomes = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for event in events:
        event_id = event["id"]
        ticker_exited = event["ticker_exited"]
        ticker_entered = event["ticker_entered"]
        eval_date = event["eval_date"][:10]

        # Calculate days since reallocation
        try:
            realloc_dt = datetime.strptime(eval_date, "%Y-%m-%d")
        except ValueError:
            continue
        days_since = (datetime.now(timezone.utc).replace(tzinfo=None) - realloc_dt).days

        # Check if we already have an outcome for today
        conn = _get_connection()
        existing = conn.execute(
            """SELECT id FROM reallocation_outcomes
               WHERE reallocation_event_id = ? AND eval_date = ?""",
            (event_id, today),
        ).fetchone()
        conn.close()
        if existing:
            continue  # already tracked today

        # Get prices at time of reallocation
        exit_price_at_realloc = _get_exit_price_at_reallocation(event)
        entry_price_at_realloc = _get_entry_price_at_reallocation(event)

        if not exit_price_at_realloc or not entry_price_at_realloc:
            continue

        # Get current prices
        exit_price_now = _get_price(ib, ticker_exited)
        entry_price_now = _get_price(ib, ticker_entered)

        if not exit_price_now or not entry_price_now:
            print(f"    Outcome tracking: skipping {ticker_exited}→{ticker_entered} "
                  f"(missing current prices)")
            continue

        # Counterfactual: what would exit ticker have returned?
        exit_return_pct = ((exit_price_now - exit_price_at_realloc)
                           / exit_price_at_realloc * 100)

        # Actual: what has entry ticker returned?
        entry_return_pct = ((entry_price_now - entry_price_at_realloc)
                            / entry_price_at_realloc * 100)

        # Performance delta: positive = reallocation was right
        perf_delta = entry_return_pct - exit_return_pct
        verdict = "VALIDATED" if perf_delta >= 0 else "UNDERPERFORMED"

        print(f"    Reallocation outcome ({days_since}d): "
              f"{ticker_exited}→{ticker_entered} "
              f"exit={exit_return_pct:+.1f}% entry={entry_return_pct:+.1f}% "
              f"delta={perf_delta:+.1f}% → {verdict}")

        # Write outcome row
        conn = _get_connection()
        conn.execute(
            """INSERT INTO reallocation_outcomes
               (reallocation_event_id, ticker_exited, ticker_entered,
                days_since_reallocation, exit_ticker_counterfactual_return,
                entry_ticker_actual_return, performance_delta, verdict,
                eval_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, ticker_exited, ticker_entered, days_since,
             round(exit_return_pct, 2), round(entry_return_pct, 2),
             round(perf_delta, 2), verdict, today),
        )
        conn.commit()
        conn.close()

        outcome = {
            "event_id": event_id,
            "ticker_exited": ticker_exited,
            "ticker_entered": ticker_entered,
            "days_since": days_since,
            "exit_return": round(exit_return_pct, 2),
            "entry_return": round(entry_return_pct, 2),
            "perf_delta": round(perf_delta, 2),
            "verdict": verdict,
        }
        outcomes.append(outcome)

        # At 30+ days, write final ledger entry (once)
        if days_since >= LEDGER_WRITE_DAYS:
            conn = _get_connection()
            already_written = conn.execute(
                """SELECT id FROM reallocation_outcomes
                   WHERE reallocation_event_id = ? AND ledger_written = 1""",
                (event_id,),
            ).fetchone()
            conn.close()

            if not already_written:
                try:
                    from kairos_reason import record_ledger_entry
                    record_ledger_entry(
                        date=today,
                        ticker=f"{ticker_exited}→{ticker_entered}",
                        action="REALLOC",
                        signals={},
                        pnl_pct=perf_delta,
                        buy_signals=[verdict],
                    )
                    # Mark as written
                    conn = _get_connection()
                    conn.execute(
                        """UPDATE reallocation_outcomes SET ledger_written = 1
                           WHERE reallocation_event_id = ? AND eval_date = ?""",
                        (event_id, today),
                    )
                    conn.commit()
                    conn.close()
                    print(f"    Ledger entry written for {ticker_exited}→{ticker_entered} "
                          f"({perf_delta:+.1f}% {verdict})")
                except Exception as exc:
                    print(f"    WARNING: Ledger write failed: {exc}")

    return outcomes


def send_weekly_summary(outcomes: list[dict] | None = None) -> bool:
    """Send reallocation outcomes summary to #kairos-reports.

    Only sends on Sundays. Queries all reallocation outcomes from the
    last 90 days if no outcomes list is provided.
    """
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))
    if et_now.weekday() != 6:  # 6 = Sunday
        return False

    if outcomes is None:
        # Load recent outcomes from DB
        init_outcome_tables()
        conn = _get_connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
        rows = conn.execute(
            """SELECT DISTINCT reallocation_event_id,
                      ticker_exited, ticker_entered,
                      MAX(days_since_reallocation) AS days_since,
                      exit_ticker_counterfactual_return AS exit_return,
                      entry_ticker_actual_return AS entry_return,
                      performance_delta AS perf_delta,
                      verdict
               FROM reallocation_outcomes
               WHERE eval_date >= ?
               GROUP BY reallocation_event_id
               ORDER BY eval_date DESC""",
            (cutoff,),
        ).fetchall()
        conn.close()
        outcomes = [dict(r) for r in rows]

    if not outcomes:
        return False

    validated = sum(1 for o in outcomes if o["verdict"] == "VALIDATED")
    underperformed = sum(1 for o in outcomes if o["verdict"] == "UNDERPERFORMED")
    avg_delta = sum(o["perf_delta"] for o in outcomes) / len(outcomes)

    lines = []
    for o in outcomes:
        emoji = ":white_check_mark:" if o["verdict"] == "VALIDATED" else ":x:"
        lines.append(
            f"  {emoji} {o['ticker_exited']}→{o['ticker_entered']} "
            f"({o['days_since']}d) | "
            f"exited: {o['exit_return']:+.1f}% | "
            f"entered: {o['entry_return']:+.1f}% | "
            f"delta: {o['perf_delta']:+.1f}%")

    summary = (
        f":bar_chart: *Weekly Reallocation Outcomes*\n"
        f"Pairs tracked: {len(outcomes)} | "
        f"Validated: {validated} | Underperformed: {underperformed} | "
        f"Avg delta: {avg_delta:+.1f}%\n\n"
        + "\n".join(lines)
    )

    try:
        from kairos_alerts import post_message
        return post_message("reports", summary)
    except Exception:
        return False
