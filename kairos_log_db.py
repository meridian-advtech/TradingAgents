"""
Kairos Log Database — Milestone 6

Creates kairos.db, provides the shared read API, and migrates existing
kairos_decisions.log entries into the database.

The equity TRADE RECORD is no longer written here. Since the fills-ledger
migration (2026-09-28) holdings / trade_outcomes are read-only views over
broker fills (kairos_ledger); decisions are written by
kairos_ledger.record_decision together with their annotations. The readers
below (get_open_holdings, get_tax_context, entry_signals_before,
get_position_exit, get_peak_gain, …) keep their old contracts.

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

# holdings / outcomes / position_exits / position_exits_history were replaced
# by the fills ledger (kairos_ledger) on 2026-09-28; the tables survive as
# *_legacy (read-only, drop on/after 2026-10-28) and holdings is now a view.

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

# Daily time-series of account value, captured from IBKR (broker = source of
# truth) once per ET trading day. Enables daily/weekly P&L, drawdown, and
# rolling-return trend metrics that are otherwise impossible without history.
# Keyed on the ET date so re-running the same day UPSERTs (overwrites, no dupes).
SCHEMA_NLV_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS nlv_snapshots (
    snapshot_date    TEXT PRIMARY KEY,   -- 'YYYY-MM-DD' ET
    nlv              REAL,               -- NetLiquidation from IBKR
    total_cash       REAL,               -- TotalCashValue from IBKR
    invested         REAL,               -- nlv - total_cash
    num_positions    INTEGER,            -- count of open equity (STK) positions at broker
    unrealized_pnl   REAL,               -- UnrealizedPnL from IBKR account summary
    realized_pnl_cum REAL,               -- cumulative realized from kairos.db closed lots
    created_at       TEXT
);
"""

# ── Arbiter → council feedback loop (Phase A: capture findings only) ──
# axis_weights holds the three learning axes the Arbiter scores each run. Phase A
# only seeds them (weight 0.0, status 'active'); nothing reads the weights yet, so
# trading decisions are unaffected. positive_means pins the sign convention so the
# model's score signs are interpretable.
SCHEMA_AXIS_WEIGHTS = """
CREATE TABLE IF NOT EXISTS axis_weights (
    axis            TEXT PRIMARY KEY,
    weight          REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'active',
    positive_means  TEXT,
    updated_at      TEXT
);
"""

# arbiter_findings is the structured capture of each Arbiter run: one row per axis
# score plus one row per qualitative observation. run_id matches the report
# filename stem (e.g. "2026-06-19_daily").
SCHEMA_ARBITER_FINDINGS = """
CREATE TABLE IF NOT EXISTS arbiter_findings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    mode             TEXT NOT NULL,
    axis             TEXT,
    score            REAL,
    sample_size      INTEGER,
    trade_ids        TEXT,
    category         TEXT,
    observation_text TEXT,
    confidence       REAL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_arbiter_findings_run_id
    ON arbiter_findings(run_id);
"""

# axis_weight_history is the audit trail + approval queue for Phase B weight
# proposals: one row per proposed update to an axis weight, carrying the computed
# statistic, the evidence, and the human decision. Nothing reads axis_weights.weight
# into trading yet — an approved weight is observed only.
SCHEMA_AXIS_WEIGHT_HISTORY = """
CREATE TABLE IF NOT EXISTS axis_weight_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    axis            TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    computed_score  REAL,
    sample_size     INTEGER,
    evidence        TEXT,
    prior_weight    REAL,
    proposed_delta  REAL,
    new_weight      REAL,
    status          TEXT NOT NULL DEFAULT 'proposed',
    created_at      TEXT NOT NULL,
    decided_at      TEXT,
    decided_by      TEXT
);
CREATE INDEX IF NOT EXISTS idx_axis_weight_history_axis_status
    ON axis_weight_history(axis, status);
"""

# The three axes seeded into axis_weights, with the exact sign convention the
# Arbiter prompt is given verbatim so its score signs stay consistent.
AXIS_POSITIVE_MEANS = {
    "exit_timing":
        "Positive = we have been exiting too LATE (giving back in-hold peaks); "
        "correction is to exit earlier.",
    "reallocation_aggressiveness":
        "Positive = we have been reallocating too EAGERLY (rotating before theses "
        "mature); correction is to hold longer before rotating.",
    "conviction_calibration":
        "Positive = conviction scores have been OVER-confident (high-conviction "
        "trades underperformed); correction is to discount conviction.",
}


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
    """Create tables (optionally drop first). The equity trade-record schema
    (fills, annotations, derived trades, views) comes from kairos_ledger."""
    conn = get_connection()
    if reset:
        conn.execute("DROP TABLE IF EXISTS options_positions")
        conn.execute("DROP TABLE IF EXISTS options_decisions")
    conn.executescript(
        SCHEMA_DECISIONS
        + SCHEMA_CRYPTO_DECISIONS + SCHEMA_CRYPTO_HOLDINGS
        + SCHEMA_OPTIONS_DECISIONS + SCHEMA_OPTIONS_POSITIONS
        + SCHEMA_IPO_LOCKUP + SCHEMA_NLV_SNAPSHOTS
        + SCHEMA_AXIS_WEIGHTS + SCHEMA_ARBITER_FINDINGS
        + SCHEMA_AXIS_WEIGHT_HISTORY
    )
    # Seed the three learning axes (idempotent — INSERT OR IGNORE leaves any
    # already-tuned weight/status untouched on re-run).
    seeded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for axis, positive_means in AXIS_POSITIVE_MEANS.items():
        conn.execute(
            "INSERT OR IGNORE INTO axis_weights "
            "(axis, weight, status, positive_means, updated_at) "
            "VALUES (?, 0.0, 'active', ?, ?)",
            (axis, positive_means, seeded_at),
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
    # Migrate: add axis_weights_snapshot column to decisions if missing (Phase C —
    # JSON {axis: weight} of the active learned-calibration weights in context when
    # the decision's reasoning prompt was built; feeds Phase D efficacy analysis).
    try:
        conn.execute("SELECT axis_weights_snapshot FROM decisions LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE decisions ADD COLUMN axis_weights_snapshot TEXT")
    # ATR-scaled armed trail (2026-09-09). Append-only record of what the
    # engine MEASURED when a position's trail armed: the ATR, the raw and
    # clamped trail, and which of the three parameters the clamp handed the
    # decision to (see kairos_atr_trail.BIND_STATES).
    #
    # Its own table rather than columns on the position, because the close
    # path reads this AFTER the position is flat, and because one row per
    # arming is the audit trail — the trail applied to a live position has to
    # be reconstructable after the fact, not inferred from today's config.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS armed_trail_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            armed_at TEXT NOT NULL,
            atr_pct REAL,
            raw_trail_pct REAL,
            trail_pct REAL,
            bind_state TEXT,
            atr_mult REAL,
            trail_lo_pct REAL,
            trail_hi_pct REAL,
            fallback_trail_pct REAL,
            atr_enabled INTEGER NOT NULL DEFAULT 0,
            peak_gain_pct REAL,
            target_pct REAL,
            created_at TEXT
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_armed_trail_ticker "
                 "ON armed_trail_context(ticker, armed_at)")
    conn.commit()
    import kairos_ledger
    if not kairos_ledger.is_migrated(conn) and \
            kairos_ledger._object_type(conn, "holdings") == "table":
        print("  WARNING: kairos.db still has the pre-ledger holdings TABLE — run "
              "kairos_migrate_fills.py before trading on this code.")
        kairos_ledger.ensure_schema(conn, views=False)
    else:
        kairos_ledger.ensure_schema(conn, views=True)
    conn.close()


# ── Public API (used by kairos_execute.py and kairos_report.py) ──────

def get_all_decisions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM decisions ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Decision History API ─────────────────────────────────────────────

def get_decision_history(limit: int = 20) -> list[dict]:
    """Return last N decisions. The pnl / hold_pnl / outcome_close keys are kept
    (always NULL) for callers of the old decisions⋈outcomes shape — the
    outcomes table never received a row and is now outcomes_legacy."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT d.*,
                  NULL AS pnl, NULL AS hold_pnl, NULL AS outcome_close
           FROM decisions d
           ORDER BY d.id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    # Return in chronological order (oldest first)
    results = [dict(r) for r in rows]
    results.reverse()
    return results


# ── Holdings API (read-only over the fills ledger) ───────────────────

def insert_drip(
    ticker: str,
    shares: float,
    price: float,
    date: str,
    net_liq_after: float | None = None,
) -> int:
    """Record a DRIP reinvestment DECISION (audit trail only).

    No lot is written: the reinvested shares reach the ledger the same way
    every other share does — as broker executions (Flex nightly / broker
    check). Writing a synthetic lot here is exactly the kind of second source
    of truth the ledger exists to remove. Returns the decision id.
    """
    import kairos_ledger
    decision_id, _ = kairos_ledger.record_decision(
        timestamp=date,
        ticker=ticker,
        action="DRIP",
        quantity=int(shares),
        rationale=f"DRIP reinvestment: {shares:.6f} sh @ ${price:.2f} on {date}",
        data_inputs={"drip_shares": shares, "drip_price": price},
        execution_price=price,
        execution_status="Filled",
        net_liq_after=net_liq_after,
    )
    return decision_id


def _open_state_tickers(flag: str) -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT DISTINCT ps.ticker FROM position_state ps "
            f"JOIN holdings h ON h.ticker = ps.ticker AND h.sold_date IS NULL "
            f"WHERE ps.{flag} = 1 ORDER BY ps.ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]
    finally:
        conn.close()


def get_protected_tickers() -> list[str]:
    """Distinct open tickers flagged protected (position_state)."""
    return _open_state_tickers("protected")


def get_drip_tickers() -> list[str]:
    """Distinct open tickers flagged drip_enabled (position_state)."""
    return _open_state_tickers("drip_enabled")


def is_protected(ticker: str) -> bool:
    """True if an open position in `ticker` is flagged protected."""
    return ticker in get_protected_tickers()


def entry_signals_before(ticker: str, before_ts: str | None = None, conn=None) -> list[str]:
    """Signals that drove the most recent BUY of `ticker` at or before
    `before_ts` (latest overall if None).

    Source is entry_annotations (written with the BUY decision, in the same
    transaction) — deliberately NOT a signals list reconstructed at exit time.
    Only BUYs that actually produced fills count, i.e. a real trade.

    Exists because 282 of 331 position_exits_history rows (85%, measured
    2026-09-28) carried empty exit_signals, so the re-entry guard compared new
    signals against nothing and waved everything through.
    """
    own = conn is None
    if own:
        conn = get_connection()
    try:
        sql = ("SELECT ea.signals FROM decisions d "
               "JOIN entry_annotations ea ON ea.trade_id = d.trade_id "
               "WHERE d.ticker = ? AND d.action = 'BUY' "
               "AND EXISTS (SELECT 1 FROM trades t WHERE t.trade_id = d.trade_id) ")
        args: list = [ticker]
        if before_ts:
            sql += "AND substr(d.timestamp,1,19) <= substr(?,1,19) "
            args.append(before_ts)
        row = conn.execute(sql + "ORDER BY d.id DESC LIMIT 1", args).fetchone()
        if not row or not row[0]:
            return []
        return [str(s) for s in json.loads(row[0]) if s]
    except Exception:
        return []
    finally:
        if own:
            conn.close()


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


# ── Exit Architecture v2 — peak tracking + last-exit (readers) ───────

def get_peak_gain(ticker: str) -> float:
    """The position's high-water mark (position_state), 0 if none.

    position_state is per TICKER and outlives a position, so a peak recorded
    before the ticker last went flat must not leak into a re-entry: a peak
    whose updated_at predates the oldest currently-open lot is stale and
    reads as 0. (Writer: kairos_exits.update_peak_gain.)
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT ps.peak_gain_pct, ps.updated_at, "
            "       (SELECT MIN(t.timestamp_entry) FROM trades t "
            "         WHERE t.ticker_current = ps.ticker AND t.action = 'BUY' "
            "           AND t.qty_open > 1e-6) AS opened_at "
            "FROM position_state ps WHERE ps.ticker = ?", (ticker,)).fetchone()
    finally:
        conn.close()
    if not row or row["opened_at"] is None or row["peak_gain_pct"] is None:
        return 0.0
    if row["updated_at"] < row["opened_at"]:
        return 0.0
    return float(row["peak_gain_pct"])


def get_position_exit(ticker: str) -> dict | None:
    """The MOST-RECENT exit for a ticker, or None if never exited.

    Reads exit_annotations (one row per SELL decision) joined to its decision
    for the date and to its fills for the realized price. Only SELLs that
    actually produced fills count — an order that never filled is not an exit. Contract unchanged:
    {ticker, exit_date, exit_price, exit_reason, exit_signals: list[str]}.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT xa.ticker, d.timestamp AS exit_date, "
            "       COALESCE((SELECT SUM(f.quantity * f.price) / SUM(f.quantity) "
            "                   FROM fills f WHERE f.decision_id = d.id), "
            "                d.execution_price) AS exit_price, "
            "       xa.exit_reason, xa.exit_signals "
            "FROM exit_annotations xa JOIN decisions d ON d.id = xa.decision_id "
            "WHERE xa.ticker = ? AND EXISTS (SELECT 1 FROM fills f WHERE f.decision_id = d.id) "
            "ORDER BY d.id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    rec = dict(row)
    try:
        rec["exit_signals"] = json.loads(rec["exit_signals"]) if rec["exit_signals"] else []
    except (json.JSONDecodeError, TypeError):
        rec["exit_signals"] = []
    return rec


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
    return migrated


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
    hold_count = conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]
    open_count = conn.execute("SELECT COUNT(*) FROM holdings WHERE sold_date IS NULL").fetchone()[0]
    opt_dec = conn.execute("SELECT COUNT(*) FROM options_decisions").fetchone()[0]
    opt_pos = conn.execute("SELECT COUNT(*) FROM options_positions").fetchone()[0]
    conn.close()
    print(f"  Decisions: {dec_count}")
    print(f"  Holdings:  {hold_count} ({open_count} open)")
    print(f"  Options:   {opt_dec} decision(s), {opt_pos} position(s)")

    print("\n" + "━" * W)
    print("  Database ready.")
    print("━" * W)


if __name__ == "__main__":
    main()
