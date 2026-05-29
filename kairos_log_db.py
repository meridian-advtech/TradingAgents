"""
Kairos Log Database — Milestone 6

Creates kairos.db with two tables (decisions + outcomes), provides a
shared API for inserting/querying, and migrates existing
kairos_decisions.log entries into the database.

Usage:
  python kairos_log_db.py           # Create DB + migrate existing logs
  python kairos_log_db.py --reset   # Drop and recreate tables, then migrate
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
LOG_FILE = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
W = 72


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Schema ───────────────────────────────────────────────────────────

SCHEMA_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    action          TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    rationale       TEXT,
    data_inputs     TEXT,
    execution_price REAL,
    execution_status TEXT,
    commission      REAL,
    net_liq_after   REAL,
    position_after  TEXT,
    conviction_trade INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL REFERENCES decisions(id),
    close_price REAL,
    pnl         REAL,
    hold_pnl    REAL,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_HOLDINGS = """
CREATE TABLE IF NOT EXISTS holdings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    entry_date  TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity    REAL NOT NULL,
    sold_date   TEXT,
    sold_price  REAL
);
"""

SCHEMA_CRYPTO_DECISIONS = """
CREATE TABLE IF NOT EXISTS crypto_decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    asset            TEXT NOT NULL,
    action           TEXT NOT NULL,
    trade_usd        REAL NOT NULL DEFAULT 0,
    quantity         REAL,
    rationale        TEXT,
    execution_price  REAL,
    execution_status TEXT,
    commission       REAL,
    confluence_score INTEGER,
    confluence_tier  TEXT,
    simulated        INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_CRYPTO_HOLDINGS = """
CREATE TABLE IF NOT EXISTS crypto_holdings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset       TEXT NOT NULL,
    entry_date  TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity    REAL NOT NULL,
    sold_date   TEXT,
    sold_price  REAL
);
"""

SCHEMA_OPTIONS_DECISIONS = """
CREATE TABLE IF NOT EXISTS options_decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    setup            TEXT NOT NULL,
    action           TEXT NOT NULL,
    right            TEXT,
    strike           REAL,
    expiry           TEXT,
    contracts        INTEGER NOT NULL DEFAULT 0,
    target_delta     REAL,
    limit_price      REAL,
    rationale        TEXT,
    signals_fired    TEXT,
    conviction       INTEGER,
    execution_status TEXT,
    fill_price       REAL,
    commission       REAL,
    degraded         INTEGER NOT NULL DEFAULT 0,
    simulated        INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_OPTIONS_POSITIONS = """
CREATE TABLE IF NOT EXISTS options_positions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id      INTEGER REFERENCES options_decisions(id),
    ticker           TEXT NOT NULL,
    setup            TEXT,
    right            TEXT NOT NULL,
    strike           REAL NOT NULL,
    expiry           TEXT NOT NULL,
    dte_at_entry     INTEGER,
    delta_at_entry   REAL,
    contracts        INTEGER NOT NULL,
    entry_premium    REAL NOT NULL,
    entry_underlying REAL,
    entry_iv         REAL,
    cost_basis       REAL,
    direction        TEXT,
    status           TEXT NOT NULL DEFAULT 'open',
    opened_at        TEXT NOT NULL,
    closed_at        TEXT,
    close_premium    REAL,
    close_reason     TEXT,
    realized_pnl     REAL,
    simulated        INTEGER NOT NULL DEFAULT 0
);
"""

SCHEMA_IPO_LOCKUP = """
CREATE TABLE IF NOT EXISTS ipo_lockup_tracker (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                 TEXT NOT NULL,
    company_name           TEXT,
    cik                    TEXT,
    ipo_date               TEXT,
    ipo_price              REAL,
    lockup_expiration_date TEXT NOT NULL,
    source                 TEXT,
    current_price          REAL,
    perf_since_ipo_pct     REAL,
    insider_pct            REAL,
    short_score            REAL,
    score_components       TEXT,
    status                 TEXT NOT NULL DEFAULT 'tracking',
    signal_emitted         INTEGER NOT NULL DEFAULT 0,
    armed_at               TEXT,
    last_scored_at         TEXT,
    notes                  TEXT,
    simulated              INTEGER NOT NULL DEFAULT 0,
    created_at             TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, lockup_expiration_date)
);
"""

# holding_days is computed at query time, not stored, to avoid
# SQLite's restriction on non-deterministic generated columns.
HOLDINGS_SELECT = """
    *, CAST(julianday(COALESCE(sold_date, datetime('now'))) - julianday(entry_date) AS INTEGER) AS holding_days
"""


# ── Database connection ──────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(reset: bool = False):
    """Create tables (optionally drop first)."""
    conn = get_connection()
    if reset:
        conn.execute("DROP TABLE IF EXISTS outcomes")
        conn.execute("DROP TABLE IF EXISTS holdings")
        conn.execute("DROP TABLE IF EXISTS decisions")
    if reset:
        conn.execute("DROP TABLE IF EXISTS options_positions")
        conn.execute("DROP TABLE IF EXISTS options_decisions")
    conn.executescript(
        SCHEMA_DECISIONS + SCHEMA_OUTCOMES + SCHEMA_HOLDINGS
        + SCHEMA_CRYPTO_DECISIONS + SCHEMA_CRYPTO_HOLDINGS
        + SCHEMA_OPTIONS_DECISIONS + SCHEMA_OPTIONS_POSITIONS
        + SCHEMA_IPO_LOCKUP
    )
    # Migrate: add conviction_trade column if missing (existing DBs)
    try:
        conn.execute("SELECT conviction_trade FROM decisions LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE decisions ADD COLUMN conviction_trade INTEGER NOT NULL DEFAULT 0")
    # Migrate: add simulated column to crypto_decisions if missing
    try:
        conn.execute("SELECT simulated FROM crypto_decisions LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE crypto_decisions ADD COLUMN simulated INTEGER NOT NULL DEFAULT 0")
    # Migrate: add degraded column to options_decisions if missing
    try:
        conn.execute("SELECT degraded FROM options_decisions LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE options_decisions ADD COLUMN degraded INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    conn.close()


# ── Public API (used by kairos_execute.py and kairos_report.py) ──────

def insert_decision(
    timestamp: str,
    ticker: str,
    action: str,
    quantity: int,
    rationale: str,
    data_inputs: dict | None = None,
    execution_price: float | None = None,
    execution_status: str | None = None,
    commission: float | None = None,
    net_liq_after: float | None = None,
    position_after: dict | None = None,
    conviction_trade: bool = False,
) -> int:
    """Insert a decision row and return its id."""
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO decisions
           (timestamp, ticker, action, quantity, rationale, data_inputs,
            execution_price, execution_status, commission, net_liq_after,
            position_after, conviction_trade)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            timestamp,
            ticker,
            action,
            quantity,
            rationale,
            json.dumps(data_inputs) if data_inputs else None,
            execution_price,
            execution_status,
            commission,
            net_liq_after,
            json.dumps(position_after) if position_after else None,
            1 if conviction_trade else 0,
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def insert_outcome(
    decision_id: int,
    close_price: float | None = None,
    pnl: float | None = None,
    hold_pnl: float | None = None,
) -> int:
    """Insert an outcome row and return its id."""
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO outcomes (decision_id, close_price, pnl, hold_pnl)
           VALUES (?, ?, ?, ?)""",
        (decision_id, close_price, pnl, hold_pnl),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_all_decisions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM decisions ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_outcomes() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT o.*, d.ticker, d.action, d.quantity, d.execution_price, d.rationale
           FROM outcomes o JOIN decisions d ON o.decision_id = d.id
           ORDER BY o.id"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Decision History API ─────────────────────────────────────────────

def get_decision_history(limit: int = 20) -> list[dict]:
    """Return last N decisions with any associated outcome data."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT d.*,
                  o.pnl, o.hold_pnl, o.close_price AS outcome_close
           FROM decisions d
           LEFT JOIN outcomes o ON o.decision_id = d.id
           ORDER BY d.id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    # Return in chronological order (oldest first)
    results = [dict(r) for r in rows]
    results.reverse()
    return results


# ── Holdings API ─────────────────────────────────────────────────────

def insert_holding(ticker: str, entry_date: str, entry_price: float, quantity: float) -> int:
    """Record a new holding lot (called on BUY fills)."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO holdings (ticker, entry_date, entry_price, quantity) VALUES (?, ?, ?, ?)",
        (ticker, entry_date, entry_price, quantity),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def sell_holdings(ticker: str, qty_to_sell: float, sold_date: str, sold_price: float) -> list[dict]:
    """Mark oldest open lots as sold (FIFO). Returns list of closed lots."""
    conn = get_connection()
    lots = conn.execute(
        f"SELECT {HOLDINGS_SELECT} FROM holdings WHERE ticker = ? AND sold_date IS NULL ORDER BY entry_date ASC",
        (ticker,),
    ).fetchall()

    closed = []
    remaining = qty_to_sell
    for lot in lots:
        if remaining <= 0:
            break
        lot_qty = lot["quantity"]
        sell_qty = min(lot_qty, remaining)

        if sell_qty >= lot_qty:
            # Close entire lot
            conn.execute(
                "UPDATE holdings SET sold_date = ?, sold_price = ? WHERE id = ?",
                (sold_date, sold_price, lot["id"]),
            )
        else:
            # Partial sell: reduce lot, create closed split
            conn.execute(
                "UPDATE holdings SET quantity = ? WHERE id = ?",
                (lot_qty - sell_qty, lot["id"]),
            )
            conn.execute(
                "INSERT INTO holdings (ticker, entry_date, entry_price, quantity, sold_date, sold_price) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticker, lot["entry_date"], lot["entry_price"], sell_qty, sold_date, sold_price),
            )
        closed.append({
            "entry_date": lot["entry_date"],
            "entry_price": lot["entry_price"],
            "quantity": sell_qty,
            "holding_days": lot["holding_days"],
        })
        remaining -= sell_qty

    conn.commit()
    conn.close()
    return closed


def get_open_holdings(ticker: str) -> list[dict]:
    """Return all unsold lots for a ticker."""
    conn = get_connection()
    rows = conn.execute(
        f"SELECT {HOLDINGS_SELECT} FROM holdings WHERE ticker = ? AND sold_date IS NULL ORDER BY entry_date ASC",
        (ticker,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_sells(ticker: str, within_days: int = 30) -> list[dict]:
    """Return lots sold within the last N days (for wash-sale detection)."""
    conn = get_connection()
    rows = conn.execute(
        f"SELECT {HOLDINGS_SELECT} FROM holdings WHERE ticker = ? AND sold_date IS NOT NULL "
        "AND julianday(datetime('now')) - julianday(sold_date) <= ? "
        "ORDER BY sold_date DESC",
        (ticker, within_days),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_tax_context(ticker: str) -> dict:
    """Build a full tax context summary for a ticker's open holdings."""
    open_lots = get_open_holdings(ticker)
    recent_sells = get_recent_sells(ticker, within_days=30)

    if not open_lots:
        return {
            "has_position": False,
            "lots": [],
            "total_quantity": 0,
            "wash_sale_risk": len(recent_sells) > 0,
            "recent_sells": [
                {
                    "entry_date": s["entry_date"],
                    "sold_date": s["sold_date"],
                    "quantity": s["quantity"],
                    "entry_price": s["entry_price"],
                    "sold_price": s["sold_price"],
                }
                for s in recent_sells
            ],
        }

    lots_detail = []
    total_qty = 0
    for lot in open_lots:
        days = lot["holding_days"] or 0
        lots_detail.append({
            "entry_date": lot["entry_date"],
            "entry_price": lot["entry_price"],
            "quantity": lot["quantity"],
            "holding_days": days,
            "tax_rate": "long-term" if days >= 365 else "short-term",
            "days_to_long_term": max(0, 365 - days),
        })
        total_qty += lot["quantity"]

    any_short_term = any(l["tax_rate"] == "short-term" for l in lots_detail)
    wash_sale_risk = len(recent_sells) > 0

    return {
        "has_position": True,
        "lots": lots_detail,
        "total_quantity": total_qty,
        "any_short_term": any_short_term,
        "wash_sale_risk": wash_sale_risk,
        "recent_sells": [
            {
                "entry_date": s["entry_date"],
                "sold_date": s["sold_date"],
                "quantity": s["quantity"],
                "entry_price": s["entry_price"],
                "sold_price": s["sold_price"],
            }
            for s in recent_sells
        ],
    }


# ── Crypto API ────────────────────────────────────────────────────────

def insert_crypto_decision(
    timestamp: str,
    asset: str,
    action: str,
    trade_usd: float = 0,
    quantity: float | None = None,
    rationale: str = "",
    execution_price: float | None = None,
    execution_status: str | None = None,
    commission: float | None = None,
    confluence_score: int | None = None,
    confluence_tier: str | None = None,
    simulated: bool = False,
) -> int:
    """Insert a crypto decision and return its row ID."""
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO crypto_decisions
           (timestamp, asset, action, trade_usd, quantity, rationale,
            execution_price, execution_status, commission,
            confluence_score, confluence_tier, simulated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (timestamp, asset, action, trade_usd, quantity, rationale,
         execution_price, execution_status, commission,
         confluence_score, confluence_tier, 1 if simulated else 0),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def insert_crypto_holding(
    asset: str, entry_date: str, entry_price: float, quantity: float
) -> int:
    """Insert a crypto holding lot (BUY). Returns row ID."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO crypto_holdings (asset, entry_date, entry_price, quantity) VALUES (?, ?, ?, ?)",
        (asset, entry_date, entry_price, quantity),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_crypto_decisions(limit: int = 20) -> list[dict]:
    """Return recent crypto decisions."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM crypto_decisions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_crypto_holdings(asset: str | None = None) -> list[dict]:
    """Return open crypto holdings, optionally filtered by asset."""
    conn = get_connection()
    if asset:
        rows = conn.execute(
            "SELECT * FROM crypto_holdings WHERE asset = ? AND sold_date IS NULL ORDER BY entry_date",
            (asset,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM crypto_holdings WHERE sold_date IS NULL ORDER BY entry_date",
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def sell_crypto_holdings(asset: str, qty_to_sell: float, sold_date: str, sold_price: float) -> list[dict]:
    """Mark oldest open crypto lots as sold (FIFO). Returns list of closed lots."""
    conn = get_connection()
    lots = conn.execute(
        f"SELECT {HOLDINGS_SELECT} FROM crypto_holdings WHERE asset = ? AND sold_date IS NULL ORDER BY entry_date ASC",
        (asset,),
    ).fetchall()

    closed = []
    remaining = qty_to_sell
    for lot in lots:
        if remaining <= 0:
            break
        lot_qty = lot["quantity"]
        sell_qty = min(lot_qty, remaining)

        if sell_qty >= lot_qty:
            conn.execute(
                "UPDATE crypto_holdings SET sold_date = ?, sold_price = ? WHERE id = ?",
                (sold_date, sold_price, lot["id"]),
            )
        else:
            conn.execute(
                "UPDATE crypto_holdings SET quantity = ? WHERE id = ?",
                (lot_qty - sell_qty, lot["id"]),
            )
            conn.execute(
                "INSERT INTO crypto_holdings (asset, entry_date, entry_price, quantity, sold_date, sold_price) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (asset, lot["entry_date"], lot["entry_price"], sell_qty, sold_date, sold_price),
            )
        closed.append({
            "entry_date": lot["entry_date"],
            "entry_price": lot["entry_price"],
            "quantity": sell_qty,
            "holding_days": lot["holding_days"],
        })
        remaining -= sell_qty

    conn.commit()
    conn.close()
    return closed


# ── Options API (HOT-CATALYST, long-only) ────────────────────────────

def insert_options_decision(
    timestamp: str,
    ticker: str,
    setup: str,
    action: str,
    right: str | None = None,
    strike: float | None = None,
    expiry: str | None = None,
    contracts: int = 0,
    target_delta: float | None = None,
    limit_price: float | None = None,
    rationale: str = "",
    signals_fired: list | None = None,
    conviction: int | None = None,
    execution_status: str | None = None,
    fill_price: float | None = None,
    commission: float | None = None,
    degraded: bool = False,
    simulated: bool = False,
) -> int:
    """Insert a HOT-CATALYST options decision and return its row ID.

    action is BUY_CALL / BUY_PUT / CLOSE only — long options exclusively.
    degraded=True flags rows where greeks came from the Black-Scholes
    fallback (OPRA subscription unavailable).
    """
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO options_decisions
           (timestamp, ticker, setup, action, right, strike, expiry,
            contracts, target_delta, limit_price, rationale, signals_fired,
            conviction, execution_status, fill_price, commission,
            degraded, simulated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            timestamp, ticker, setup, action, right, strike, expiry,
            contracts, target_delta, limit_price, rationale,
            json.dumps(signals_fired) if signals_fired else None,
            conviction, execution_status, fill_price, commission,
            1 if degraded else 0, 1 if simulated else 0,
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def open_options_position(
    decision_id: int | None,
    ticker: str,
    setup: str,
    right: str,
    strike: float,
    expiry: str,
    contracts: int,
    entry_premium: float,
    opened_at: str,
    dte_at_entry: int | None = None,
    delta_at_entry: float | None = None,
    entry_underlying: float | None = None,
    entry_iv: float | None = None,
    direction: str | None = None,
    simulated: bool = False,
) -> int:
    """Open a long options position. cost_basis = premium * contracts * 100."""
    cost_basis = entry_premium * contracts * 100
    status = "simulated" if simulated else "open"
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO options_positions
           (decision_id, ticker, setup, right, strike, expiry, dte_at_entry,
            delta_at_entry, contracts, entry_premium, entry_underlying,
            entry_iv, cost_basis, direction, status, opened_at, simulated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            decision_id, ticker, setup, right, strike, expiry, dte_at_entry,
            delta_at_entry, contracts, entry_premium, entry_underlying,
            entry_iv, cost_basis, direction, status, opened_at,
            1 if simulated else 0,
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def close_options_position(
    position_id: int,
    close_premium: float,
    close_reason: str,
    closed_at: str,
    realized_pnl: float | None = None,
) -> None:
    """Mark an options position closed. close_reason: stop_50/tp_100/dte_21/manual.

    If realized_pnl is None it is computed as
    (close_premium - entry_premium) * contracts * 100.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT entry_premium, contracts, simulated FROM options_positions WHERE id = ?",
        (position_id,),
    ).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"options_positions id {position_id} not found")
    if realized_pnl is None:
        realized_pnl = (close_premium - row["entry_premium"]) * row["contracts"] * 100
    status = "simulated" if row["simulated"] else "closed"
    conn.execute(
        """UPDATE options_positions
           SET status = ?, closed_at = ?, close_premium = ?,
               close_reason = ?, realized_pnl = ?
           WHERE id = ?""",
        (status, closed_at, close_premium, close_reason, realized_pnl, position_id),
    )
    conn.commit()
    conn.close()


def get_open_options_positions(ticker: str | None = None) -> list[dict]:
    """Return positions not yet closed (status open or simulated)."""
    conn = get_connection()
    if ticker:
        rows = conn.execute(
            "SELECT * FROM options_positions "
            "WHERE closed_at IS NULL AND ticker = ? ORDER BY opened_at",
            (ticker,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM options_positions WHERE closed_at IS NULL ORDER BY opened_at"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_options_decisions(limit: int = 20) -> list[dict]:
    """Return recent options decisions, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM options_decisions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── HOT-IPO lock-up expiration tracker ───────────────────────────────

def upsert_lockup_row(
    ticker: str,
    lockup_expiration_date: str,
    company_name: str | None = None,
    cik: str | None = None,
    ipo_date: str | None = None,
    ipo_price: float | None = None,
    source: str | None = None,
    notes: str | None = None,
) -> int:
    """Insert a lock-up tracking row (or refresh static fields if it exists).

    Dedup key is (ticker, lockup_expiration_date). Scoring/status fields are
    left untouched on update — those move through set_lockup_status. Returns
    the row id.
    """
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO ipo_lockup_tracker
           (ticker, company_name, cik, ipo_date, ipo_price,
            lockup_expiration_date, source, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker, lockup_expiration_date) DO UPDATE SET
            company_name = COALESCE(excluded.company_name, company_name),
            cik          = COALESCE(excluded.cik, cik),
            ipo_date     = COALESCE(excluded.ipo_date, ipo_date),
            ipo_price    = COALESCE(excluded.ipo_price, ipo_price),
            source       = COALESCE(excluded.source, source),
            notes        = COALESCE(excluded.notes, notes)""",
        (ticker, company_name, cik, ipo_date, ipo_price,
         lockup_expiration_date, source, notes),
    )
    row_id = cur.lastrowid
    if not row_id:
        row = conn.execute(
            "SELECT id FROM ipo_lockup_tracker "
            "WHERE ticker = ? AND lockup_expiration_date = ?",
            (ticker, lockup_expiration_date),
        ).fetchone()
        row_id = row["id"] if row else None
    conn.commit()
    conn.close()
    return row_id


def get_lockup_rows(status: str | None = None) -> list[dict]:
    """Return lock-up tracker rows, optionally filtered by status."""
    conn = get_connection()
    if status:
        rows = conn.execute(
            "SELECT * FROM ipo_lockup_tracker WHERE status = ? "
            "ORDER BY lockup_expiration_date",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM ipo_lockup_tracker ORDER BY lockup_expiration_date"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_lockup_status(row_id: int, status: str, **fields) -> None:
    """Update a lock-up row's status and any scalar fields passed by name.

    Accepts: current_price, perf_since_ipo_pct, insider_pct, short_score,
    score_components (dict -> JSON), signal_emitted, armed_at, last_scored_at,
    simulated, notes. Unknown keys are ignored.
    """
    allowed = {
        "current_price", "perf_since_ipo_pct", "insider_pct", "short_score",
        "score_components", "signal_emitted", "armed_at", "last_scored_at",
        "simulated", "notes",
    }
    sets = ["status = ?"]
    vals: list = [status]
    for key, val in fields.items():
        if key not in allowed:
            continue
        if key == "score_components" and isinstance(val, (dict, list)):
            val = json.dumps(val)
        if key in ("signal_emitted", "simulated") and isinstance(val, bool):
            val = 1 if val else 0
        sets.append(f"{key} = ?")
        vals.append(val)
    vals.append(row_id)
    conn = get_connection()
    conn.execute(
        f"UPDATE ipo_lockup_tracker SET {', '.join(sets)} WHERE id = ?",
        vals,
    )
    conn.commit()
    conn.close()


def get_armed_lockup_signals() -> list[dict]:
    """Armed lock-up short theses the catalyst detector should emit as puts."""
    return get_lockup_rows(status="armed")


# ── Migration from kairos_decisions.log ──────────────────────────────

def parse_log_objects(content: str) -> list[dict]:
    """Extract all top-level JSON objects from the log file."""
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
    return objects


def migrate_log():
    """Read kairos_decisions.log and insert entries into the database."""
    if not os.path.exists(LOG_FILE):
        print("  No kairos_decisions.log found — nothing to migrate.")
        return 0

    with open(LOG_FILE, "r") as f:
        content = f.read()

    objects = parse_log_objects(content)
    if not objects:
        print("  No JSON entries found in log.")
        return 0

    # Pair decisions with their executions
    decisions = [o for o in objects if o.get("type") != "EXECUTION"]
    executions = [o for o in objects if o.get("type") == "EXECUTION"]

    migrated = 0
    conn = get_connection()

    # Check if we already have data (avoid double migration)
    existing = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    if existing > 0:
        print(f"  Database already has {existing} decisions — skipping migration.")
        conn.close()
        return 0

    for dec in decisions:
        # Find matching execution
        ticker = dec.get("ticker", "AAPL")
        action = dec.get("action", "HOLD")
        exec_match = None
        for ex in executions:
            ex_dec = ex.get("decision", {})
            if ex_dec.get("ticker") == ticker and ex_dec.get("action") == action:
                exec_match = ex
                break

        exec_data = exec_match.get("execution", {}) if exec_match else {}
        data_inputs = {}
        if "input_summary" in dec:
            data_inputs = dec["input_summary"]
        if "reasoning_chain" in dec:
            data_inputs["reasoning_chain"] = dec["reasoning_chain"]

        ts = dec.get("timestamp", exec_match.get("timestamp", "") if exec_match else "")

        conn.execute(
            """INSERT INTO decisions
               (timestamp, ticker, action, quantity, rationale, data_inputs,
                execution_price, execution_status, commission, net_liq_after,
                position_after)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                ticker,
                action,
                dec.get("quantity", 0),
                dec.get("rationale", ""),
                json.dumps(data_inputs) if data_inputs else None,
                exec_data.get("fill_price"),
                exec_data.get("status"),
                exec_data.get("commission"),
                float(exec_data["net_liquidation_after"]) if exec_data.get("net_liquidation_after") else None,
                json.dumps(exec_data.get("new_position")) if exec_data.get("new_position") else None,
            ),
        )
        migrated += 1

    conn.commit()
    conn.close()

    # Seed holdings from migrated BUY decisions
    _seed_holdings_from_decisions()

    return migrated


def _seed_holdings_from_decisions():
    """Create holding lots for any BUY decisions that don't have matching holdings."""
    conn = get_connection()
    existing_holdings = conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]
    if existing_holdings > 0:
        conn.close()
        return

    buys = conn.execute(
        "SELECT timestamp, ticker, quantity, execution_price "
        "FROM decisions WHERE action = 'BUY' AND execution_status = 'Filled'"
    ).fetchall()

    for b in buys:
        # Parse date from timestamp like "2026-03-25 15:08:07 UTC"
        entry_date = b["timestamp"].replace(" UTC", "").strip()
        conn.execute(
            "INSERT INTO holdings (ticker, entry_date, entry_price, quantity) VALUES (?, ?, ?, ?)",
            (b["ticker"], entry_date, b["execution_price"], b["quantity"]),
        )

    conn.commit()
    conn.close()


# ── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kairos Log Database")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS LOG DATABASE".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    print(banner("Initializing Database"))
    init_db(reset=args.reset)
    print(f"  Database: {DB_PATH}")
    if args.reset:
        print("  Tables dropped and recreated.")
    else:
        print("  Tables created (if not exists).")

    print(banner("Migrating Existing Logs"))
    count = migrate_log()
    print(f"  Migrated {count} decision(s).")

    # Verify
    print(banner("Verification"))
    conn = get_connection()
    dec_count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    out_count = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    hold_count = conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]
    open_count = conn.execute("SELECT COUNT(*) FROM holdings WHERE sold_date IS NULL").fetchone()[0]
    opt_dec = conn.execute("SELECT COUNT(*) FROM options_decisions").fetchone()[0]
    opt_pos = conn.execute("SELECT COUNT(*) FROM options_positions").fetchone()[0]
    conn.close()
    print(f"  Decisions: {dec_count}")
    print(f"  Outcomes:  {out_count}")
    print(f"  Holdings:  {hold_count} ({open_count} open)")
    print(f"  Options:   {opt_dec} decision(s), {opt_pos} position(s)")

    print("\n" + "━" * W)
    print("  Database ready.")
    print("━" * W)


if __name__ == "__main__":
    main()
