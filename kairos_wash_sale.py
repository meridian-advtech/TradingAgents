"""
Kairos Wash Sale Prevention — IRS 30-Day Rule

The IRS wash sale rule (Section 1091) disallows claiming a capital loss
if a substantially identical security is purchased within 30 days before
or after the loss sale.

Two checks:
  1. check_wash_sale_risk(ticker)      — before BUY: was this ticker sold
     at a loss in the last 30 days?  If yes, block the BUY.
  2. check_wash_sale_violation(ticker)  — before loss SELL: was this ticker
     purchased in the last 30 days?  If yes, flag the loss as disallowed
     but still execute (the trade happens, the tax deduction does not).

Events are logged to the wash_sale_log table for audit.
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")

SCHEMA_WASH_SALE_LOG = """
CREATE TABLE IF NOT EXISTS wash_sale_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    sell_date       TEXT NOT NULL,
    sell_price      REAL NOT NULL,
    entry_price     REAL NOT NULL,
    loss_amount     REAL NOT NULL,
    repurchase_date TEXT,
    blocked_until   TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _get_connection() -> sqlite3.Connection:
    from kairos_log_db import get_connection
    return get_connection()


def init_wash_sale_tables() -> None:
    """Create the wash_sale_log table and add wash_sale_flag to decisions."""
    conn = _get_connection()
    conn.executescript(SCHEMA_WASH_SALE_LOG)
    # Add wash_sale_flag column to decisions if missing
    try:
        conn.execute("SELECT wash_sale_flag FROM decisions LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            "ALTER TABLE decisions ADD COLUMN wash_sale_flag INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()
    conn.close()


def check_wash_sale_risk(ticker: str) -> dict:
    """Check if buying this ticker would trigger a wash sale.

    Looks for any SELL of the same ticker at a loss in the last 30 days.
    If found, the BUY should be blocked until 30 days after that loss sale.

    Returns:
        {"risk": True, "blocked_until": "YYYY-MM-DD", "loss_amount": float,
         "sell_date": str, "sell_price": float, "entry_price": float}
        or {"risk": False}
    """
    init_wash_sale_tables()

    conn = _get_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

    # Check holdings table for loss sells in the last 30 days
    rows = conn.execute(
        """SELECT sold_date, sold_price, entry_price, quantity
           FROM holdings
           WHERE ticker = ? AND sold_date IS NOT NULL
             AND sold_price < entry_price
             AND date(sold_date) >= date(?)
           ORDER BY sold_date DESC""",
        (ticker, cutoff),
    ).fetchall()
    conn.close()

    if not rows:
        return {"risk": False}

    # Use the most recent loss sale
    row = rows[0]
    sold_date_str = row["sold_date"][:10]
    loss_per_share = row["entry_price"] - row["sold_price"]
    loss_amount = round(loss_per_share * row["quantity"], 2)

    # Parse the sell date and compute blocked_until (30 days after)
    try:
        sell_dt = datetime.strptime(sold_date_str, "%Y-%m-%d")
    except ValueError:
        sell_dt = datetime.strptime(row["sold_date"][:19], "%Y-%m-%d %H:%M:%S")
    blocked_until = (sell_dt + timedelta(days=30)).strftime("%Y-%m-%d")

    return {
        "risk": True,
        "blocked_until": blocked_until,
        "loss_amount": loss_amount,
        "sell_date": sold_date_str,
        "sell_price": row["sold_price"],
        "entry_price": row["entry_price"],
    }


def check_wash_sale_violation(ticker: str, sell_price: float,
                              entry_price: float) -> dict:
    """Check if selling this ticker at a loss creates a wash sale violation.

    A violation occurs when the same ticker was purchased in the last
    30 days AND the current sale is at a loss.

    Returns:
        {"violation": True, "repurchase_date": str, "loss_disallowed": float}
        or {"violation": False}
    """
    if sell_price >= entry_price:
        return {"violation": False}

    init_wash_sale_tables()

    conn = _get_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

    # Check holdings for BUY lots entered in the last 30 days
    rows = conn.execute(
        """SELECT entry_date, entry_price, quantity
           FROM holdings
           WHERE ticker = ? AND sold_date IS NULL
             AND date(entry_date) >= date(?)
           ORDER BY entry_date DESC""",
        (ticker, cutoff),
    ).fetchall()
    conn.close()

    if not rows:
        return {"violation": False}

    repurchase_date = rows[0]["entry_date"][:10]
    loss_per_share = entry_price - sell_price
    # Loss disallowed is capped at the number of shares repurchased
    repurchase_qty = sum(r["quantity"] for r in rows)
    # (For simplicity, flag the full loss — IRS rules are per-share but
    #  this gives a conservative upper bound for the warning.)
    loss_disallowed = round(loss_per_share * repurchase_qty, 2)

    return {
        "violation": True,
        "repurchase_date": repurchase_date,
        "loss_disallowed": loss_disallowed,
    }


def log_wash_sale_event(ticker: str, sell_date: str, sell_price: float,
                        entry_price: float, loss_amount: float,
                        repurchase_date: str | None,
                        blocked_until: str) -> int:
    """Write an event to the wash_sale_log table. Returns row ID."""
    init_wash_sale_tables()

    conn = _get_connection()
    cur = conn.execute(
        """INSERT INTO wash_sale_log
           (ticker, sell_date, sell_price, entry_price, loss_amount,
            repurchase_date, blocked_until)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticker, sell_date, sell_price, entry_price, loss_amount,
         repurchase_date, blocked_until),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def flag_decision_wash_sale(decision_id: int) -> None:
    """Set wash_sale_flag = 1 on a decisions row."""
    init_wash_sale_tables()

    conn = _get_connection()
    conn.execute(
        "UPDATE decisions SET wash_sale_flag = 1 WHERE id = ?",
        (decision_id,),
    )
    conn.commit()
    conn.close()
