"""
Kairos ML Outcomes — thesis tables, the exit-regime snapshot, and ML readers.

Since the fills-ledger migration (2026-09-28) this module WRITES NO TRADE
RECORD. trade_outcomes is a read-only view in kairos.db built by kairos_ledger
from broker fills + annotations; kairos_ml_outcomes.db is gone (its
thesis_predictions / thesis_checkpoints tables moved, unchanged, into kairos.db).

What lives here:
  * thesis_predictions / thesis_checkpoints schema (writer: kairos_ml_thesis)
  * build_exit_params_snapshot — the exit regime stamped on every SELL's
    exit_annotations row by kairos_ledger.record_decision
  * ATTRIBUTION_SOURCES / TRUSTED_ATTRIBUTION_SOURCES, FORGONE_HORIZONS
  * read_outcomes_for_ml, get_thesis_target, get_invalidation_level (readers)
  * _score_prediction_accuracy (used by the nightly features job)

Removed with the migration (their job is now done by kairos_ledger):
write_trade_open, write_trade_close, record_exit_outcome, _stamp_close,
select_rows_closed_by_sale, label_unlabeled_closes, list_provisional_entries,
reconcile_provisional_entries.
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Everything lives in kairos.db now (thesis tables + the trade_outcomes view).
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
KAIROS_DB_PATH = DB_PATH
CONFIG_PATH = os.path.join(SCRIPT_DIR, "kairos_config.json")


# ── Schema ───────────────────────────────────────────────────────────

# Thesis validation tables — predictions captured at entry, then checkpointed
# during the hold, finally scored at close.  decision_id links to
# trade_outcomes.trade_id (the BUY decision's trade_id, minted by
# kairos_ledger.record_decision).
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

# Post-exit horizons over which forgone gain is measured, in CALENDAR days:
# ~1 week, 2 weeks, 1 month, 2 months. Canonical definition for every horizon:
#
#     forgone_gain_Nd_pct = (max High in (exit, exit+N days] - price_exit)
#                           / price_exit * 100
#
# i.e. the best exit that was still available within N days of the one taken.
# Positive means we left money on the table (exited EARLY); negative means the
# position kept falling after the exit (exiting was right).
#
# NOTE: the pre-existing forgone_gain_5d_pct values were written by a script no
# longer present in the repo and disagreed with post_exit_peak_pct on 64 of 160
# rows, with no recoverable definition. kairos_outcome_features now recomputes
# all horizons — including 5d — from this one definition, so the objective runs
# on a measure that can be reproduced and audited.
FORGONE_HORIZONS = (5, 14, 30, 60)


def forgone_column(days: int) -> str:
    """Column name holding forgone gain at an N-day horizon."""
    return f"forgone_gain_{int(days)}d_pct"

# Provenance values for signal_attribution_source, ordered most → least
# trustworthy. Anything below 'confluence' is CONTEXT, not causation, and
# per-signal P&L should say so rather than quietly averaging them together.
ATTRIBUTION_SOURCES = (
    "explicit",        # the entry path named its own trigger — trade-level truth
    "confluence",      # confluence tags captured during sizing — trade-level
    "decision_record", # legacy row re-attributed from decisions.data_inputs —
                       # the same decision-time field 'confluence' is written
                       # from (validated 217/217 identical on 2026-09-28)
    "ticker_context",  # signals firing for the TICKER that day — not causation
    "rationale_text",  # tags parsed out of prose — a guess, last resort
    "none",            # nothing known anywhere; deliberately not fabricated
    "legacy_mixed",    # pre-2026-08-03 rows: union of ALL sources, unseparable
    "excluded",        # opening balance / corporate-action chain (e.g. SATS->ECHO):
                       # kept in positions and P&L, deliberately never learned from
)

# Sources that support a causal claim about why a trade was entered.
# 'decision_record' is trusted because it is read from exactly the field that
# 'confluence' rows are written from, and the two agreed on every one of 217
# rows checked. Adding it restored 239 historical trades that the legacy
# quarantine was discarding.
TRUSTED_ATTRIBUTION_SOURCES = ("explicit", "confluence", "decision_record")

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
    """Create the thesis tables (kairos.db) and the ledger schema if missing."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_THESIS_PREDICTIONS)
        conn.executescript(SCHEMA_THESIS_CHECKPOINTS)
        existing_chk_cols = {row["name"] for row in conn.execute(
            "PRAGMA table_info(thesis_checkpoints)"
        ).fetchall()}
        for col_name, col_type in _THESIS_CHECKPOINTS_EXTRA_COLUMNS:
            if col_name not in existing_chk_cols:
                conn.execute(
                    f"ALTER TABLE thesis_checkpoints ADD COLUMN {col_name} {col_type}"
                )
        conn.commit()
        import kairos_ledger
        kairos_ledger.ensure_schema(conn, views=kairos_ledger.is_migrated(conn)
                                    or kairos_ledger._object_type(conn, "holdings") is None)
    finally:
        conn.close()


# ── Exit-params snapshot (regime tagging) ────────────────────────────
# Every closed trade carries a record of the exit engine that produced it. The
# learning loop scores a parameter only on trades that closed under that
# parameter's CURRENT value (kairos_axis_weights._in_regime); a trade with no
# snapshot is not "assume it matches", it is excluded outright. So a close that
# is not stamped here is permanently invisible to the loop — the snapshot cannot
# be recovered later, because the config it describes has already moved on.
#
# Two shapes exist in the corpus and BOTH must stay readable by _in_regime,
# which only ever reads snap["params"][path] and snap["axis_weights"][axis]:
#
#   live (reconstructed=False)  params, axis_weights, reconstructed,
#                               trailing_stop (the whole config block, for
#                               forensics), captured_at = now
#   reconstructed (=True)       params, axis_weights, reconstructed,
#                               captured_at = the trade's exit timestamp
#
# The reconstructed shape carries no trailing_stop block on purpose: the
# historical config text is not recoverable, and inventing it would make a
# reconstruction indistinguishable from a real capture.

# Dotted config paths recorded in every snapshot. MUST stay in sync with
# kairos_axis_weights.PARAM_WHITELIST — for a path the loop can propose but the
# snapshot does not record, _snapshot_value returns None, so _in_regime is False
# for EVERY trade and that param sits gated at n=0 forever. It fails closed, but
# silently, and reads exactly like a regime that has not filled in yet.
# kairos_selftest_learning.py asserts every proposable path is recorded here.
#
# DUAL-STAMPING (2026-09-09). The ATR-scaled trail ships with atr_enabled=false,
# so behaviour is unchanged — but all four new paths are stamped from the moment
# this lands, NOT from the moment the behaviour flips. The trail's evidence pool
# refills at 3-8 closes a week. If atr_mult only started being stamped when
# atr_enabled went true, its effective_n would be 0 on day one and would stay
# there for weeks, which is precisely the starvation the weighted-evidence work
# fixed on 2026-09-09 — and proximity weighting CANNOT rescue it, because an
# unstamped close has no parameter distance to measure, only missing data.
# Stamping during the shadow period means the pool is already populated when
# the switch is flipped.
#
# atr_enabled is stamped for PROVENANCE, not as a regime key: _snapshot_value
# rejects booleans, so _in_regime is always False for it. That is correct — it
# is not learnable and must never be in PARAM_WHITELIST — but it is the only
# field that tells you whether a given close ran under the ATR trail or the
# fixed fallback, which the bind-state routing needs.
SNAPSHOT_PARAM_PATHS = (
    # Deprecated as the primary trail, still the fallback — keep stamping it.
    "exits.trailing_stop.target_armed.trail_pct",
    "exits.trailing_stop.profit_floor_pp",
    # ATR-scaled trail (shadow from 2026-09-09; behaviour off).
    "exits.trailing_stop.target_armed.atr_enabled",
    "exits.trailing_stop.target_armed.atr_mult",
    "exits.trailing_stop.target_armed.trail_lo_pct",
    "exits.trailing_stop.target_armed.trail_hi_pct",
)


def _get_dotted(cfg: dict, path: str):
    """Return (value, found) for a dotted path into a nested dict."""
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None, False
        cur = cur[key]
    return cur, True


def _load_live_config() -> dict:
    """kairos_config.json, or {} if unreadable. Never raises."""
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, IOError, OSError):
        return {}


def _live_axis_weights() -> dict:
    """{axis: weight} for active axes from kairos.db, or {} if unreadable.

    Read-only connection: this is a snapshot of primary state taken from the
    outcomes writer, and must never be able to lock or mutate kairos.db.
    """
    try:
        conn = sqlite3.connect(f"file:{KAIROS_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            return {r["axis"]: r["weight"] for r in conn.execute(
                "SELECT axis, weight FROM axis_weights WHERE status = 'active'")}
        finally:
            conn.close()
    except sqlite3.Error:
        return {}


def build_exit_params_snapshot(reconstructed: bool = False,
                               param_overrides: Optional[dict] = None,
                               weight_overrides: Optional[dict] = None,
                               as_of: Optional[str] = None,
                               ticker: Optional[str] = None,
                               entry_date: Optional[str] = None,
                               arm_context: Optional[dict] = None) -> dict:
    """The exit-engine regime to stamp on a closing trade.

    Live capture reads kairos_config.json + kairos.db. A backfill passes
    param_overrides / weight_overrides (the values reconstructed as in force at
    the trade's exit) plus as_of=<the exit timestamp>, and gets a snapshot that
    is explicitly flagged reconstructed=True.

    Never raises: an unreadable config or DB yields None values for what could
    not be read, which _in_regime treats as "cannot attribute" — the row is
    excluded from evidence rather than silently mis-attributed.
    """
    cfg = _load_live_config() if param_overrides is None else {}

    params = {}
    for path in SNAPSHOT_PARAM_PATHS:
        if param_overrides is not None:
            params[path] = param_overrides.get(path)
            continue
        val, found = _get_dotted(cfg, path)
        params[path] = val if found else None

    weights = dict(weight_overrides) if weight_overrides is not None \
        else _live_axis_weights()

    snap = {
        "params": params,
        "axis_weights": weights,
        "reconstructed": bool(reconstructed),
    }
    if not reconstructed:
        # Full block, for forensics: the params above say WHICH regime, this says
        # what the whole engine looked like (tiers, IPO widening, enabled flags).
        ts_block, found = _get_dotted(cfg or _load_live_config(),
                                      "exits.trailing_stop")
        if found and isinstance(ts_block, dict):
            snap["trailing_stop"] = ts_block
    # ── Clamp-bind attribution (2026-09-09) ─────────────────────────
    # Which of the three ATR parameters actually governed this position's
    # trail. Stamped OUTSIDE snapshot["params"] on purpose: params is the
    # numeric regime-key namespace _snapshot_value reads, and bind_state is a
    # string. kairos_axis_weights routes each close to exactly one parameter's
    # evidence pool on the strength of this block, so a close without it
    # contributes to none of the three — which is the correct outcome for a
    # close that predates the mechanism.
    if not reconstructed:
        try:
            import kairos_atr_trail as _atr
            ctx = arm_context
            if ctx is None and ticker:
                # persist=False: stamping a close must never CREATE an arming
                # record. If the position never armed, there is nothing to
                # attribute and the block is omitted.
                stored = _atr.latest_arm(ticker, since=entry_date)
                ctx = _atr.arm_context(ticker, since=entry_date,
                                       persist=False) if stored else None
            block = _atr.snapshot_block(
                ctx, bool(params.get(
                    "exits.trailing_stop.target_armed.atr_enabled")))
            if block:
                snap["armed_trail"] = block
        except Exception:
            # A close must never fail because attribution could not be built.
            pass
    snap["captured_at"] = as_of or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    return snap


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


# ── Read: ML export ──────────────────────────────────────────────────

def read_outcomes_for_ml(closed_only: bool = True) -> list[dict]:
    """Return trade outcomes as a list of dicts ready for ML ingestion.

    If closed_only=True (default), only returns trades with a non-null
    outcome_label (i.e., closed trades with PnL computed).

    signals_fired is deserialized from JSON back to a Python list.
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

# Direction form: "closes below $228.00", "trades under $120",
# "breaks below pre-earnings support levels (~$950)", "closes below 228.00".
# The 40-char gap absorbs the words filers put between the direction and the
# number; the spec's literal `(below|above)\s*\$` matches NONE of the 239
# theses actually logged, because every real phrasing has words in between.
#
# The `$` is optional, which is the risky half of this pattern: without a
# guard, "breaks below 50-day MA" parses 50 as a price. Two lookaheads close
# that. `(?![\d.])` forces the number to END where it matches — otherwise the
# engine happily takes "5" of "50" and finds nothing objectionable after it —
# and the second rejects period/ratio/percent units that are never prices.
_INVAL_DIR_RE = re.compile(
    r"\b(below|under)\b[^$\d\n]{0,40}?"
    r"(?:\$\s*)?"
    r"([\d,]+(?:\.\d+)?)"
    r"(?![\d.])"
    r"(?!\s*(?:-?\s*(?:day|week|month|yr|year)|d\b|%|bps|x\b|MA\b|SMA|EMA|DMA))",
    re.IGNORECASE)
# "$165 support" / "$75 technical support" / "$12.50 resistance"
_INVAL_LEVEL_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(technical support|support|resistance)\b",
    re.IGNORECASE)

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
        # Only downside breaks invalidate a long thesis.
        if direction.lower() not in ("below", "under"):
            continue
        try:
            levels.append(float(raw.replace(",", "")))
        except ValueError:
            continue

    for raw, kind in _INVAL_LEVEL_RE.findall(text):
        if "resistance" in kind.lower():
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
            conn.execute("PRAGMA busy_timeout = 2000")
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
