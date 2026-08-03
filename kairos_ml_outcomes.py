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
import re
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

# Thesis validation tables — predictions captured at entry, then checkpointed
# during the hold, finally scored at close.  decision_id links to
# trade_outcomes.trade_id (UUID created in write_trade_open).
SCHEMA_THESIS_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS thesis_predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                  TEXT NOT NULL,
    decision_id             TEXT NOT NULL,
    timestamp_entry         TEXT NOT NULL,
    predicted_direction     TEXT,
    predicted_timeframe_days INTEGER,
    predicted_return_pct    REAL,
    key_conditions          TEXT,
    signal_type             TEXT,
    conviction_score        REAL,
    invalidation_conditions TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_thesis_pred_decision_id
    ON thesis_predictions(decision_id);
CREATE INDEX IF NOT EXISTS idx_thesis_pred_ticker
    ON thesis_predictions(ticker);
"""

SCHEMA_THESIS_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS thesis_checkpoints (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                  TEXT NOT NULL,
    decision_id             TEXT NOT NULL,
    checkpoint_day          INTEGER NOT NULL,
    timestamp_checked       TEXT NOT NULL,
    price_at_entry          REAL,
    price_at_checkpoint     REAL,
    pct_move_actual         REAL,
    direction_correct       INTEGER,
    thesis_conditions_intact INTEGER,
    notes                   TEXT,
    checkpoint_score        REAL,
    UNIQUE(decision_id, checkpoint_day)
);
CREATE INDEX IF NOT EXISTS idx_thesis_chk_decision_id
    ON thesis_checkpoints(decision_id);
"""

# Per-signal aggregate view. signal_type comes from thesis_predictions; if a
# trade has no prediction (legacy / non-BUY), it is grouped under 'UNKNOWN'.
SCHEMA_SIGNAL_PERFORMANCE_VIEW = """
DROP VIEW IF EXISTS signal_performance;
CREATE VIEW signal_performance AS
SELECT
    COALESCE(tp.signal_type, 'UNKNOWN') AS signal_type,
    COUNT(*) AS total_trades,
    AVG(CASE WHEN trade_outcomes.pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
    AVG(trade_outcomes.pnl_pct) AS avg_return_pct,
    AVG(trade_outcomes.hold_duration_mins) / 1440.0 AS avg_hold_days,
    AVG(trade_outcomes.prediction_accuracy) AS avg_prediction_accuracy
FROM trade_outcomes
LEFT JOIN thesis_predictions AS tp
    ON tp.decision_id = trade_outcomes.trade_id
WHERE trade_outcomes.outcome_label IS NOT NULL
GROUP BY COALESCE(tp.signal_type, 'UNKNOWN');
"""

# Columns to ensure exist on trade_outcomes (idempotent ALTER TABLE).
_TRADE_OUTCOMES_EXTRA_COLUMNS = [
    ("prediction_accuracy", "REAL"),
    ("thesis_score", "REAL"),
    # Set to 1 when the row was opened before the real fill price was known
    # (Filled BUY with no immediate fill_price). price_entry holds a best-effort
    # estimate until the position reconciler trues it up against the broker.
    ("entry_price_provisional", "INTEGER"),
    # B2a: closed-trade outcome features, populated post-hoc by
    # kairos_outcome_features.py (behaviour-neutral; not written by
    # write_trade_close). features_filled_at stays NULL until computed.
    ("mfe_pct", "REAL"),                 # max favorable excursion during hold
    ("give_back_pct", "REAL"),           # mfe_pct - pnl_pct (peak surrendered)
    ("post_exit_peak_pct", "REAL"),      # max favorable move AFTER exit (too-early signal)
    ("post_exit_window_days", "INTEGER"),
    ("exit_reason", "TEXT"),             # trigger, copied from kairos.db position_exits
    ("features_filled_at", "TEXT"),      # NULL until features computed
    # Provenance of signals_fired — how the attribution was determined, which
    # is the difference between "this signal drove the trade" and "this signal
    # happened to be firing for this ticker that day". See ATTRIBUTION_SOURCES.
    ("signal_attribution_source", "TEXT"),
]

# Provenance values for signal_attribution_source, ordered most → least
# trustworthy. Anything below 'confluence' is CONTEXT, not causation, and
# per-signal P&L should say so rather than quietly averaging them together.
ATTRIBUTION_SOURCES = (
    "explicit",        # the entry path named its own trigger — trade-level truth
    "confluence",      # confluence tags captured during sizing — trade-level
    "ticker_context",  # signals firing for the TICKER that day — not causation
    "rationale_text",  # tags parsed out of prose — a guess, last resort
    "none",            # nothing known anywhere; deliberately not fabricated
    "legacy_mixed",    # pre-2026-08-03 rows: union of ALL sources, unseparable
)

# Sources that support a causal claim about why a trade was entered.
TRUSTED_ATTRIBUTION_SOURCES = ("explicit", "confluence")

# Columns to ensure exist on thesis_checkpoints (idempotent ALTER TABLE).
_THESIS_CHECKPOINTS_EXTRA_COLUMNS = [
    ("ipo_metrics", "TEXT"),   # JSON blob populated for IPO_MOMENTUM trades
]


# ── Database connection ──────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create the trade_outcomes table + thesis tables/view if missing."""
    conn = get_connection()
    conn.executescript(SCHEMA_TRADE_OUTCOMES)
    conn.executescript(SCHEMA_THESIS_PREDICTIONS)
    conn.executescript(SCHEMA_THESIS_CHECKPOINTS)

    # Add prediction_accuracy / thesis_score columns idempotently.
    existing_cols = {row["name"] for row in conn.execute(
        "PRAGMA table_info(trade_outcomes)"
    ).fetchall()}
    for col_name, col_type in _TRADE_OUTCOMES_EXTRA_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(
                f"ALTER TABLE trade_outcomes ADD COLUMN {col_name} {col_type}"
            )

    # Idempotent migration for thesis_checkpoints extra columns
    existing_chk_cols = {row["name"] for row in conn.execute(
        "PRAGMA table_info(thesis_checkpoints)"
    ).fetchall()}
    for col_name, col_type in _THESIS_CHECKPOINTS_EXTRA_COLUMNS:
        if col_name not in existing_chk_cols:
            conn.execute(
                f"ALTER TABLE thesis_checkpoints ADD COLUMN {col_name} {col_type}"
            )

    # signal_performance view depends on the new columns — rebuild it.
    conn.executescript(SCHEMA_SIGNAL_PERFORMANCE_VIEW)
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
    entry_price_provisional: bool = False,
    signal_attribution_source: Optional[str] = None,
) -> str:
    """Record a new trade at open. Returns the trade_id (UUID).

    Set entry_price_provisional=True when price_entry is a best-effort estimate
    (Filled BUY whose fill price was not yet known); the position reconciler
    backfills the true price later via reconcile_provisional_entries().
    """
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
            market_regime, sector, entry_price_provisional,
            signal_attribution_source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trade_id, timestamp_entry, ticker, action, quantity, price_entry,
            signals_json, confluence_score,
            council_member_1_rec, council_member_1_confidence,
            council_member_2_rec, council_member_2_confidence,
            council_agreement, arbiter_invoked, arbiter_rec,
            market_regime, sector, 1 if entry_price_provisional else 0,
            signal_attribution_source,
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

    # Score the original thesis prediction (if any) against the actual close.
    pred = conn.execute(
        "SELECT predicted_direction, predicted_timeframe_days, "
        "predicted_return_pct FROM thesis_predictions "
        "WHERE decision_id = ? ORDER BY id DESC LIMIT 1",
        (trade_id,),
    ).fetchone()

    prediction_accuracy: Optional[float] = None
    thesis_score: Optional[float] = None

    if pred is not None:
        prediction_accuracy = _score_prediction_accuracy(
            predicted_direction=pred["predicted_direction"],
            predicted_timeframe_days=pred["predicted_timeframe_days"],
            predicted_return_pct=pred["predicted_return_pct"],
            actual_pnl_pct=pnl_pct,
            hold_duration_mins=hold_duration_mins,
        )

        chk_rows = conn.execute(
            "SELECT checkpoint_score FROM thesis_checkpoints "
            "WHERE decision_id = ? AND checkpoint_score IS NOT NULL",
            (trade_id,),
        ).fetchall()
        scores = [row["checkpoint_score"] for row in chk_rows
                  if row["checkpoint_score"] is not None]
        if scores:
            thesis_score = round(sum(scores) / len(scores), 4)

        conn.execute(
            """UPDATE trade_outcomes
               SET prediction_accuracy = ?,
                   thesis_score = ?
               WHERE trade_id = ?""",
            (prediction_accuracy, thesis_score, trade_id),
        )

    conn.commit()
    conn.close()

    return {
        "trade_id": trade_id,
        "pnl_dollar": round(pnl_dollar, 4),
        "pnl_pct": round(pnl_pct, 4),
        "hold_duration_mins": hold_duration_mins,
        "outcome_label": outcome_label,
        "prediction_accuracy": prediction_accuracy,
        "thesis_score": thesis_score,
    }


def _score_prediction_accuracy(
    predicted_direction: Optional[str],
    predicted_timeframe_days: Optional[int],
    predicted_return_pct: Optional[float],
    actual_pnl_pct: float,
    hold_duration_mins: int,
) -> float:
    """Compute prediction_accuracy in [0.0, 1.0].

    Scoring (per spec):
        +0.4 direction correct
        +0.3 return within 50% of predicted (i.e. |actual - predicted| <= 0.5*|predicted|)
        +0.3 closed within predicted timeframe
    """
    score = 0.0

    direction_actual = (
        "UP" if actual_pnl_pct > 0.0
        else "DOWN" if actual_pnl_pct < 0.0
        else "NEUTRAL"
    )
    if predicted_direction:
        pred_dir = predicted_direction.strip().upper()
        if pred_dir == direction_actual:
            score += 0.4
        # NEUTRAL prediction matches a flat/scratch outcome
        elif pred_dir == "NEUTRAL" and abs(actual_pnl_pct) < 1.0:
            score += 0.4

    if predicted_return_pct is not None:
        pred_ret = float(predicted_return_pct)
        if pred_ret != 0.0:
            if abs(actual_pnl_pct - pred_ret) <= 0.5 * abs(pred_ret):
                score += 0.3
        else:
            # If predicted 0% (flat), be lenient: anything <1% absolute counts
            if abs(actual_pnl_pct) < 1.0:
                score += 0.3

    if predicted_timeframe_days:
        actual_days = max(0.0, hold_duration_mins / 1440.0)
        if actual_days <= float(predicted_timeframe_days):
            score += 0.3

    return round(score, 4)


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


def record_exit_outcome(
    ticker: str,
    timestamp_exit: str,
    price_exit: float,
    exit_reason: str,
) -> Optional[str]:
    """Populate exit fields on the oldest OPEN trade_outcomes row for a ticker.

    Called at the sell_holdings chokepoint (the single point every equity close
    funnels through) so the ML outcomes DB captures the exit_reason / realized
    PnL / give-back that were previously left NULL on all but a couple of rows.

    Match: ticker + timestamp_exit IS NULL; if several are open, the oldest by
    entry (FIFO) — mirroring how sell_holdings closes the underlying lots. Sets
    timestamp_exit, price_exit, pnl_pct, pnl_dollar, exit_reason, hold_duration_mins,
    and give_back_pct = mfe_pct - pnl_pct when mfe_pct has already been computed
    (by kairos_outcome_features.py). PnL is direction-aware, matching
    write_trade_close. outcome_label is deliberately left untouched so the
    downstream write_trade_close (matched by outcome_label IS NULL) still fires.

    Returns the trade_id updated, or None if no open row matched. Callers MUST
    wrap this so a DB failure never blocks or raises into the trade path.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT trade_id, action, quantity, price_entry, timestamp_entry, mfe_pct
               FROM trade_outcomes
               WHERE ticker = ? AND timestamp_exit IS NULL
               ORDER BY timestamp_entry ASC LIMIT 1""",
            (ticker,),
        ).fetchone()
        if row is None:
            return None

        price_entry = row["price_entry"]
        action = row["action"]
        quantity = row["quantity"] or 0

        # Direction-aware PnL (mirrors write_trade_close).
        if action == "BUY":
            pnl_dollar = (price_exit - price_entry) * quantity
        else:  # SELL (short)
            pnl_dollar = (price_entry - price_exit) * quantity
        pnl_pct = ((price_exit - price_entry) / price_entry * 100) if price_entry else 0.0
        if action == "SELL":
            pnl_pct = -pnl_pct

        hold_duration_mins = _compute_duration_mins(row["timestamp_entry"], timestamp_exit)

        give_back_pct = None
        if row["mfe_pct"] is not None:
            give_back_pct = round(row["mfe_pct"] - pnl_pct, 4)

        conn.execute(
            """UPDATE trade_outcomes
               SET timestamp_exit = ?, price_exit = ?, pnl_pct = ?, pnl_dollar = ?,
                   exit_reason = ?, give_back_pct = ?, hold_duration_mins = ?
               WHERE trade_id = ?""",
            (timestamp_exit, price_exit, round(pnl_pct, 4), round(pnl_dollar, 4),
             exit_reason, give_back_pct, hold_duration_mins, row["trade_id"]),
        )
        conn.commit()
        return row["trade_id"]
    finally:
        conn.close()


def get_thesis_target(ticker: str) -> Optional[float]:
    """Most recent logged predicted_return_pct for this ticker's thesis.

    Returns None if no thesis_predictions row exists (e.g. pre-logging-fix
    trades, or signal types that don't log a prediction) — caller must
    fall back to the existing flat trail in that case.
    """
    conn = get_connection()
    row = conn.execute(
        """SELECT predicted_return_pct FROM thesis_predictions
           WHERE ticker = ? ORDER BY timestamp_entry DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    conn.close()
    return row["predicted_return_pct"] if row and row["predicted_return_pct"] is not None else None


# ── Price-level thesis invalidation ──────────────────────────────────
# The Council logs free-text invalidation_conditions per thesis. Most are
# qualitative ("Breaks below pre-earnings support level") and out of scope for
# v1 — only an explicit, parseable PRICE LEVEL is usable mechanically. This
# reader is deliberately conservative: anything it cannot read cleanly returns
# None and the position falls through to the other exit conditions untouched.

# "below $165" / "above $12.50"
_INVAL_DIR_RE = re.compile(
    r"\b(below|above)\b[^$\d\n]{0,20}\$\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
# "$165 support" / "$12.50 resistance"
_INVAL_LEVEL_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(support|resistance)\b", re.IGNORECASE)

# Economic sanity band for a parsed level, as implied move from entry price.
# Rejects misparses in both directions: a level at or above entry (-0.5% floor)
# would fire instantly, and one more than 30% below entry is far past the hard
# stop and almost certainly a parse of an unrelated number.
_INVAL_MIN_MOVE_PCT = -30.0
_INVAL_MAX_MOVE_PCT = -0.5

_INVALIDATION_CACHE: dict = {}


def clear_invalidation_cache() -> None:
    """Reset the per-evaluation-run cache. Call once at the top of a run."""
    _INVALIDATION_CACHE.clear()


def _parse_invalidation_levels(text: str) -> list:
    """Extract BELOW-direction price levels from invalidation_conditions text.

    Returns a list of floats. "above"/"resistance" matches are dropped: for a
    long-only book, invalidation is a break DOWN through a level.
    """
    levels = []
    if not text:
        return levels

    for direction, raw in _INVAL_DIR_RE.findall(text):
        if direction.lower() != "below":
            continue
        try:
            levels.append(float(raw.replace(",", "")))
        except ValueError:
            continue

    for raw, kind in _INVAL_LEVEL_RE.findall(text):
        if kind.lower() != "support":
            continue
        try:
            levels.append(float(raw.replace(",", "")))
        except ValueError:
            continue

    return levels


def get_invalidation_level(ticker: str, entry_price: float) -> Optional[float]:
    """Parsed price-invalidation level for this ticker's most recent thesis.

    Reads thesis_predictions.invalidation_conditions (read-only connection),
    extracts below-direction price levels, keeps only those whose implied move
    from `entry_price` lands in the sanity band, and returns the one CLOSEST to
    entry (the first line that would be broken).

    Returns None when there is no thesis row, no parseable level, or nothing
    survives the sanity band — i.e. whenever a mechanical read is not safe.
    Never raises: any failure degrades to None.
    """
    key = (ticker, round(float(entry_price), 4) if entry_price else None)
    if key in _INVALIDATION_CACHE:
        return _INVALIDATION_CACHE[key]

    level = None
    try:
        if entry_price and entry_price > 0:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    """SELECT invalidation_conditions FROM thesis_predictions
                       WHERE ticker = ? ORDER BY timestamp_entry DESC LIMIT 1""",
                    (ticker,),
                ).fetchone()
            finally:
                conn.close()

            if row and row[0]:
                candidates = [
                    lvl for lvl in _parse_invalidation_levels(row[0])
                    if lvl > 0 and _INVAL_MIN_MOVE_PCT
                    <= (lvl - entry_price) / entry_price * 100.0
                    <= _INVAL_MAX_MOVE_PCT
                ]
                if candidates:
                    # Closest to entry == highest surviving level.
                    level = max(candidates)
    except Exception:
        level = None

    _INVALIDATION_CACHE[key] = level
    return level


# ── Reconcile provisional entry prices ───────────────────────────────

def list_provisional_entries() -> list[dict]:
    """Return open trades whose entry price is still a provisional estimate.

    Each dict: {trade_id, ticker, price_entry, quantity, timestamp_entry}.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT trade_id, ticker, price_entry, quantity, timestamp_entry
           FROM trade_outcomes
           WHERE entry_price_provisional = 1 AND outcome_label IS NULL
           ORDER BY timestamp_entry"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def reconcile_provisional_entries(prices: dict) -> int:
    """Backfill the true entry price on rows opened before the fill price was known.

    Args:
        prices: {ticker: confirmed_entry_price} — typically the broker's average
                cost for each currently-held position.

    For every open, provisional row whose ticker has a confirmed price, set
    price_entry to that price and clear the provisional flag. Returns the number
    of rows reconciled.
    """
    if not prices:
        return 0
    conn = get_connection()
    rows = conn.execute(
        """SELECT trade_id, ticker FROM trade_outcomes
           WHERE entry_price_provisional = 1 AND outcome_label IS NULL"""
    ).fetchall()
    reconciled = 0
    for r in rows:
        confirmed = prices.get(r["ticker"])
        if confirmed is None or confirmed <= 0:
            continue
        conn.execute(
            """UPDATE trade_outcomes
               SET price_entry = ?, entry_price_provisional = 0
               WHERE trade_id = ?""",
            (round(float(confirmed), 4), r["trade_id"]),
        )
        reconciled += 1
    if reconciled:
        conn.commit()
    conn.close()
    return reconciled


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
