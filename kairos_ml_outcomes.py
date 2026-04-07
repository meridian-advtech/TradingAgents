"""
Kairos ML Outcomes — Machine-Learning-Ready Trade Outcome Database

Defines and manages a SQLite table `trade_outcomes` with 22 fields
designed for ML feature extraction.  Replaces the human-readable
kairos_ledger.txt for analytical purposes (ledger kept in parallel
until this schema is validated).

DB location: ~/TradingAgents/kairos_ml_outcomes.db

Usage:
    from kairos_ml_outcomes import init_db, write_trade_open, write_trade_close, read_outcomes_for_ml

    init_db()
    trade_id = write_trade_open(...)
    write_trade_close(trade_id, ...)
    df = read_outcomes_for_ml()
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")


# ── Schema ───────────────────────────────────────────────────────────

SCHEMA_TRADE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS trade_outcomes (
    trade_id                TEXT PRIMARY KEY,
    timestamp_entry         TEXT NOT NULL,
    timestamp_exit          TEXT,
    ticker                  TEXT NOT NULL,
    action                  TEXT NOT NULL,
    quantity                INTEGER NOT NULL,
    price_entry             REAL NOT NULL,
    price_exit              REAL,
    pnl_dollar              REAL,
    pnl_pct                 REAL,
    hold_duration_mins      INTEGER,
    signals_fired           TEXT,
    confluence_score        INTEGER,
    council_member_1_rec    TEXT,
    council_member_1_confidence REAL,
    council_member_2_rec    TEXT,
    council_member_2_confidence REAL,
    council_agreement       INTEGER,
    arbiter_invoked         INTEGER,
    arbiter_rec             TEXT,
    market_regime           TEXT,
    sector                  TEXT,
    outcome_label           TEXT
);
"""


# ── Database connection ──────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create the trade_outcomes table if it doesn't exist."""
    conn = get_connection()
    conn.executescript(SCHEMA_TRADE_OUTCOMES)
    conn.commit()
    conn.close()


# ── Write: trade open ────────────────────────────────────────────────

def write_trade_open(
    ticker: str,
    action: str,
    quantity: int,
    price_entry: float,
    timestamp_entry: Optional[str] = None,
    signals_fired: Optional[list[str]] = None,
    confluence_score: Optional[int] = None,
    council_member_1_rec: Optional[str] = None,
    council_member_1_confidence: Optional[float] = None,
    council_member_2_rec: Optional[str] = None,
    council_member_2_confidence: Optional[float] = None,
    council_agreement: Optional[int] = None,
    arbiter_invoked: Optional[int] = None,
    arbiter_rec: Optional[str] = None,
    market_regime: Optional[str] = None,
    sector: Optional[str] = None,
    trade_id: Optional[str] = None,
) -> str:
    """Record a new trade at open. Returns the trade_id (UUID)."""
    if trade_id is None:
        trade_id = str(uuid.uuid4())
    if timestamp_entry is None:
        timestamp_entry = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    signals_json = json.dumps(signals_fired) if signals_fired else None

    conn = get_connection()
    conn.execute(
        """INSERT INTO trade_outcomes
           (trade_id, timestamp_entry, ticker, action, quantity, price_entry,
            signals_fired, confluence_score,
            council_member_1_rec, council_member_1_confidence,
            council_member_2_rec, council_member_2_confidence,
            council_agreement, arbiter_invoked, arbiter_rec,
            market_regime, sector)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trade_id, timestamp_entry, ticker, action, quantity, price_entry,
            signals_json, confluence_score,
            council_member_1_rec, council_member_1_confidence,
            council_member_2_rec, council_member_2_confidence,
            council_agreement, arbiter_invoked, arbiter_rec,
            market_regime, sector,
        ),
    )
    conn.commit()
    conn.close()
    return trade_id


# ── Write: trade close ───────────────────────────────────────────────

def write_trade_close(
    trade_id: str,
    price_exit: float,
    timestamp_exit: Optional[str] = None,
) -> dict:
    """Close an existing trade: compute PnL, duration, and outcome label.

    Returns a dict with the computed fields.
    """
    if timestamp_exit is None:
        timestamp_exit = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM trade_outcomes WHERE trade_id = ?", (trade_id,)
    ).fetchone()

    if row is None:
        conn.close()
        raise ValueError(f"trade_id {trade_id!r} not found in trade_outcomes")

    price_entry = row["price_entry"]
    action = row["action"]
    quantity = row["quantity"]
    timestamp_entry = row["timestamp_entry"]

    # PnL calculation: direction-aware
    if action == "BUY":
        pnl_dollar = (price_exit - price_entry) * quantity
    else:  # SELL (short)
        pnl_dollar = (price_entry - price_exit) * quantity

    pnl_pct = ((price_exit - price_entry) / price_entry * 100) if price_entry else 0.0
    if action == "SELL":
        pnl_pct = -pnl_pct

    # Duration
    hold_duration_mins = _compute_duration_mins(timestamp_entry, timestamp_exit)

    # Outcome label
    if abs(pnl_dollar) < 0.01:
        outcome_label = "SCRATCH"
    elif pnl_dollar > 0:
        outcome_label = "WIN"
    else:
        outcome_label = "LOSS"

    conn.execute(
        """UPDATE trade_outcomes
           SET timestamp_exit = ?,
               price_exit = ?,
               pnl_dollar = ?,
               pnl_pct = ?,
               hold_duration_mins = ?,
               outcome_label = ?
           WHERE trade_id = ?""",
        (timestamp_exit, price_exit, round(pnl_dollar, 4),
         round(pnl_pct, 4), hold_duration_mins, outcome_label, trade_id),
    )
    conn.commit()
    conn.close()

    return {
        "trade_id": trade_id,
        "pnl_dollar": round(pnl_dollar, 4),
        "pnl_pct": round(pnl_pct, 4),
        "hold_duration_mins": hold_duration_mins,
        "outcome_label": outcome_label,
    }


def _compute_duration_mins(ts_entry: str, ts_exit: str) -> int:
    """Parse ISO timestamps and return duration in minutes."""
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%d %H:%M:%S",
    ]
    dt_entry = _parse_ts(ts_entry, fmts)
    dt_exit = _parse_ts(ts_exit, fmts)
    if dt_entry and dt_exit:
        delta = dt_exit - dt_entry
        return max(0, int(delta.total_seconds() / 60))
    return 0


def _parse_ts(ts: str, fmts: list[str]) -> Optional[datetime]:
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


# ── Read: ML export ──────────────────────────────────────────────────

def read_outcomes_for_ml(closed_only: bool = True) -> list[dict]:
    """Return trade outcomes as a list of dicts ready for ML ingestion.

    If closed_only=True (default), only returns trades with a non-null
    outcome_label (i.e., closed trades with PnL computed).

    Each dict has all 22 fields.  signals_fired is deserialized from
    JSON back to a Python list.
    """
    conn = get_connection()
    if closed_only:
        rows = conn.execute(
            "SELECT * FROM trade_outcomes WHERE outcome_label IS NOT NULL ORDER BY timestamp_entry"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trade_outcomes ORDER BY timestamp_entry"
        ).fetchall()
    conn.close()

    results = []
    for row in rows:
        d = dict(row)
        # Deserialize signals_fired JSON
        if d.get("signals_fired"):
            try:
                d["signals_fired"] = json.loads(d["signals_fired"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


# ── Utility: find open trade by ticker ───────────────────────────────

def find_open_trade(ticker: str, action: str = "BUY") -> Optional[str]:
    """Find the most recent open (unclosed) trade_id for a ticker+action.

    Returns trade_id or None.
    """
    conn = get_connection()
    row = conn.execute(
        """SELECT trade_id FROM trade_outcomes
           WHERE ticker = ? AND action = ? AND outcome_label IS NULL
           ORDER BY timestamp_entry DESC LIMIT 1""",
        (ticker, action),
    ).fetchone()
    conn.close()
    return row["trade_id"] if row else None


# ── main (standalone init) ───────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print(f"ML outcomes database ready: {DB_PATH}")

    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
    open_count = conn.execute(
        "SELECT COUNT(*) FROM trade_outcomes WHERE outcome_label IS NULL"
    ).fetchone()[0]
    closed_count = conn.execute(
        "SELECT COUNT(*) FROM trade_outcomes WHERE outcome_label IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    print(f"  Total: {count}  Open: {open_count}  Closed: {closed_count}")
