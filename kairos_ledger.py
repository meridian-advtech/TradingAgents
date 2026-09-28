"""
Kairos Ledger — the trade record, built from broker fills.

Design of record: "Kairos — Trade Record Architecture (Design Decision)" (Craft),
incl. the 2026-09-28 addendum. Equities only; crypto and options are untouched.

WHY THIS EXISTS
    Until 2026-09-28 the trade record was ~30 write sites mutating three
    overlapping stores (kairos.db holdings, position_exits_history, and
    kairos_ml_outcomes.db trade_outcomes), each reconstructing "what happened"
    from the caller's own idea of the fill. They disagreed: ghost rows, closes
    at the wrong price (DD's pre-split entry against a post-split exit), phantom
    closes of positions the broker still held (SATS → ECHO), and a reconciler
    that rewrote history at broker cost. Positions are now DERIVED from the one
    thing that cannot be wrong about what happened — the broker's executions —
    and everything else is an annotation on top.

TABLES (all in kairos.db) — each has exactly ONE writer
    fills             append-only   record_fills()
    fill_commissions  append-only   record_fills()   (late commission for a fill
                                                     first captured without one)
    position_events   append-only   record_position_events()
    entry_annotations               record_decision()  (BUY, same txn)
    exit_annotations                record_decision()  (SELL, same txn)
    trades            derived       rebuild_trades()
    trade_matches     derived       rebuild_trades()
    lots              derived       rebuild_trades()   (backs the holdings view)
    trade_features    derived       kairos_outcome_features (nightly)
    position_state    mutable       kairos_exits (the exit engine)
    decisions.trade_id / order_id / perm_id   record_decision()

VIEWS (read-only; same names/columns as the tables they replaced)
    holdings, trade_outcomes, positions, signal_performance

CONVENTIONS
    * fills.executed_at is UTC 'YYYY-MM-DDTHH:MM:SSZ' — one format, enforced by
      a CHECK constraint. Flex's US/Eastern 'yyyymmdd;hhmmss' is converted.
    * commission is stored POSITIVE (a cost). NULL means "not known yet" — never
      0 as a stand-in; IBKR's reqExecutions returns 0.0 before the commission
      report arrives, which is not the same as a free trade.
    * P&L (trades.pnl_dollar) is NET of commissions on both legs:
          pnl = exit proceeds − entry cost of the matched shares
                − entry commission (pro-rata to matched shares)
                − exit commission  (pro-rata share of each exit fill)
      An unknown commission counts as 0 and the trade is flagged
      commission_complete = 0. pnl_pct = pnl / entry cost (ex-commission) × 100.
      This is applied uniformly to every trade; the legacy ledger ignored
      commissions, so a small, systematic old-vs-new difference is expected.
    * One trade = one IBKR BUY order (adds are separate trades). A trade closed
      in pieces takes its exit_reason from its LAST exit; every exit keeps its
      own exit_annotations row.
    * FIFO per ticker, deterministic: rebuild_trades() always produces the same
      trades/trade_matches from the same fills + position_events.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
ET = ZoneInfo("America/New_York")

EPS = 1e-6          # share-quantity tolerance (IBKR reports 4 dp)
QTY_DP = 4          # quantities are rounded to IBKR's precision

FILL_SOURCES = ("live", "broker_check", "statement_import", "flex_nightly")
EVENT_TYPES = ("split", "symbol_change", "merger_cash", "spinoff", "opening_balance")

# Attribution provenance written onto trades that must never be learned from:
# opening balances (no entry decision exists) and the SATS→ECHO chain (the
# legacy ledger voided it, so its annotations are unverifiable). Deliberately
# NOT in kairos_ml_outcomes.TRUSTED_ATTRIBUTION_SOURCES.
EXCLUDED_ATTRIBUTION = "excluded"

# The Flex Web Service token expires on this date; the nightly pull names it in
# every failure alert so an expired token is never mistaken for "no trades".
FLEX_TOKEN_EXPIRES = "2027-09-28"
FLEX_BASE = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"


# ── Schema ───────────────────────────────────────────────────────────

def _append_only_triggers(table: str) -> str:
    return f"""
CREATE TRIGGER IF NOT EXISTS trg_{table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} is append-only: UPDATE refused'); END;
CREATE TRIGGER IF NOT EXISTS trg_{table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} is append-only: DELETE refused'); END;
"""


SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS fills (
    exec_id      TEXT PRIMARY KEY,
    perm_id      INTEGER,            -- IBKR API permId (unique per order, account-wide)
    order_id     TEXT,               -- Flex IBOrderID (statement/flex rows)
    decision_id  INTEGER REFERENCES decisions(id),
    link_method  TEXT NOT NULL CHECK (link_method IN
                   ('live','perm_id','order_sibling','nearest_session','unlinked')),
    ticker       TEXT NOT NULL,
    side         TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
    quantity     REAL NOT NULL CHECK (quantity > 0),
    price        REAL NOT NULL,
    commission   REAL,               -- positive cost; NULL = not known yet
    executed_at  TEXT NOT NULL CHECK (executed_at GLOB
                   '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'),
    source       TEXT NOT NULL CHECK (source IN
                   ('live','broker_check','statement_import','flex_nightly')),
    inserted_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_ticker ON fills(ticker, executed_at);
CREATE INDEX IF NOT EXISTS idx_fills_perm ON fills(perm_id);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_decision ON fills(decision_id);

CREATE TABLE IF NOT EXISTS fill_commissions (
    exec_id      TEXT PRIMARY KEY REFERENCES fills(exec_id),
    commission   REAL NOT NULL,
    source       TEXT NOT NULL,
    inserted_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS position_events (
    event_id       TEXT PRIMARY KEY,
    ticker         TEXT NOT NULL,
    event_type     TEXT NOT NULL CHECK (event_type IN
                     ('split','symbol_change','merger_cash','spinoff','opening_balance')),
    qty_change     REAL NOT NULL,     -- shares added (+) / removed (−) on `ticker`;
                                      -- symbol_change: shares MOVED to new_ticker (+)
    ratio          REAL,              -- split: new shares per old share
    new_ticker     TEXT,
    cash_per_share REAL,
    proceeds       REAL,
    effective_at   TEXT NOT NULL CHECK (effective_at GLOB
                     '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'),
    source         TEXT NOT NULL,
    note           TEXT,
    inserted_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entry_annotations (
    trade_id                  TEXT PRIMARY KEY,
    signals                   TEXT,
    signal_attribution_source TEXT,
    confluence_score          INTEGER,
    conviction                INTEGER,
    market_regime             TEXT,
    sector                    TEXT,
    ml_confidence_at_entry    REAL,
    ml_signal_at_entry        TEXT,
    ml_trained_on_at_entry    INTEGER,
    created_at                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exit_annotations (
    decision_id          INTEGER PRIMARY KEY REFERENCES decisions(id),
    ticker               TEXT NOT NULL,
    exit_reason          TEXT NOT NULL,
    exit_signals         TEXT NOT NULL CHECK (json_valid(exit_signals)
                           AND json_type(exit_signals) = 'array'),
    exit_signals_note    TEXT,        -- REQUIRED when exit_signals = '[]'
    exit_params_snapshot TEXT,
    created_at           TEXT NOT NULL,
    CHECK (exit_signals <> '[]' OR exit_signals_note IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_exit_ann_ticker ON exit_annotations(ticker);

CREATE TABLE IF NOT EXISTS trades (
    trade_no              INTEGER NOT NULL,
    trade_id              TEXT PRIMARY KEY,
    ticker                TEXT NOT NULL,     -- ticker at entry
    ticker_current        TEXT NOT NULL,     -- ticker now (open) / at last exit
    action                TEXT NOT NULL,     -- BUY (long) | SELL (short, long-only violation)
    origin                TEXT NOT NULL,     -- fills | opening_balance | spinoff
    order_key             TEXT,
    decision_id           INTEGER,
    quantity              REAL NOT NULL,     -- entry shares, entry units
    quantity_adj          REAL NOT NULL,     -- entry shares in current units (after splits)
    price_entry           REAL,              -- qty-weighted entry fill price, entry units
    price_entry_adj       REAL,              -- cost per current-unit share
    cost_basis            REAL,              -- Σ entry qty × price (NULL = unknown basis)
    entry_commission      REAL NOT NULL,
    timestamp_entry       TEXT NOT NULL,     -- first entry fill, UTC ISO-Z
    qty_open              REAL NOT NULL,
    qty_closed            REAL NOT NULL,
    price_exit            REAL,
    proceeds              REAL,
    exit_commission       REAL NOT NULL,
    timestamp_exit        TEXT,              -- last exit, only once fully closed
    pnl_dollar            REAL,              -- only once fully closed (see CONVENTIONS)
    pnl_pct               REAL,
    realized_pnl          REAL,              -- matched-so-far P&L (open or closed)
    hold_duration_mins    INTEGER,
    outcome_label         TEXT,
    last_exit_kind        TEXT,              -- fill | event
    last_exit_ref         TEXT,              -- exec_id | event_id
    last_exit_decision_id INTEGER,
    commission_complete   INTEGER NOT NULL,
    rebuilt_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker_current, qty_open);

CREATE TABLE IF NOT EXISTS trade_matches (
    match_no         INTEGER NOT NULL,
    trade_id         TEXT NOT NULL,
    exit_ref         TEXT NOT NULL,          -- exec_id (fill) or event_id (corporate action)
    exit_kind        TEXT NOT NULL,          -- fill | event
    ticker           TEXT NOT NULL,          -- ticker at exit
    qty              REAL NOT NULL,
    exit_price       REAL NOT NULL,
    entry_cost       REAL,                   -- cost basis of the matched shares
    entry_commission REAL NOT NULL,
    exit_commission  REAL NOT NULL,
    exit_at          TEXT NOT NULL,
    PRIMARY KEY (trade_id, exit_ref)
);
CREATE INDEX IF NOT EXISTS idx_matches_exit ON trade_matches(exit_ref);

-- One row per lot as the holdings view shows it: each open remainder of a long
-- trade (sold_date NULL) and each closed piece (a trade_matches row). Derived,
-- so the holdings view can be a single SELECT with the replaced table's exact
-- column types (a UNION ALL view loses REAL/INTEGER affinity in SQLite).
CREATE TABLE IF NOT EXISTS lots (
    lot_no      INTEGER PRIMARY KEY,
    trade_id    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    entry_date  TEXT NOT NULL,       -- 'YYYY-MM-DD HH:MM:SS UTC' (legacy spelling)
    entry_price REAL,                -- cost per current-unit share (NULL = unknown basis)
    quantity    REAL NOT NULL,
    sold_date   TEXT,                -- 'YYYY-MM-DD HH:MM:SS' (legacy spelling)
    sold_price  REAL
);
CREATE INDEX IF NOT EXISTS idx_lots_ticker ON lots(ticker, sold_date);

CREATE TABLE IF NOT EXISTS trade_features (
    trade_id              TEXT PRIMARY KEY,
    mfe_pct               REAL,
    give_back_pct         REAL,
    post_exit_peak_pct    REAL,
    post_exit_window_days INTEGER,
    forgone_gain_5d_pct   REAL,
    forgone_gain_14d_pct  REAL,
    forgone_gain_30d_pct  REAL,
    forgone_gain_60d_pct  REAL,
    features_filled_at    TEXT,
    forgone_filled_at     TEXT,
    prediction_accuracy   REAL,
    thesis_score          REAL,
    source                TEXT
);

CREATE TABLE IF NOT EXISTS position_state (
    ticker        TEXT PRIMARY KEY,
    peak_gain_pct REAL NOT NULL DEFAULT 0,
    protected     INTEGER NOT NULL DEFAULT 0,
    drip_enabled  INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT NOT NULL
);
""" + _append_only_triggers("fills") + _append_only_triggers("fill_commissions") \
    + _append_only_triggers("position_events")

# A BUY that went to the broker cannot be stored without its trade_id and entry
# annotations — this is what makes "an executed trade with no attribution"
# unrepresentable, rather than merely discouraged.
SCHEMA_DECISION_GUARD = """
CREATE TRIGGER IF NOT EXISTS trg_decisions_buy_needs_annotations
BEFORE INSERT ON decisions
WHEN NEW.action = 'BUY' AND COALESCE(NEW.execution_status, '') <> 'Skipped'
     AND (NEW.trade_id IS NULL OR NOT EXISTS
          (SELECT 1 FROM entry_annotations WHERE trade_id = NEW.trade_id))
BEGIN SELECT RAISE(ABORT, 'BUY decision requires trade_id + entry_annotations'); END;
CREATE INDEX IF NOT EXISTS idx_decisions_trade_id ON decisions(trade_id);
CREATE INDEX IF NOT EXISTS idx_decisions_perm_id ON decisions(perm_id);
"""

# ── Views ────────────────────────────────────────────────────────────
# Column names and ORDER reproduce the replaced tables exactly (verified by the
# rehearsal's schema test). Timestamp spellings are the ones readers already
# parse: holdings.entry_date / trade_outcomes.timestamp_entry are
# 'YYYY-MM-DD HH:MM:SS UTC', holdings.sold_date is 'YYYY-MM-DD HH:MM:SS',
# trade_outcomes.timestamp_exit is 'YYYY-MM-DDTHH:MM:SSZ'.

_LEGACY_TS = "replace(replace({c}, 'T', ' '), 'Z', ' UTC')"

VIEW_HOLDINGS = f"""
CREATE VIEW IF NOT EXISTS holdings AS
SELECT l.lot_no      AS id,
       l.ticker      AS ticker,
       l.entry_date  AS entry_date,
       l.entry_price AS entry_price,
       l.quantity    AS quantity,
       l.sold_date   AS sold_date,
       l.sold_price  AS sold_price,
       CAST(CASE WHEN l.sold_date IS NULL THEN COALESCE(ps.drip_enabled, 0) ELSE 0 END
            AS INTEGER) AS drip_enabled,
       CAST(CASE WHEN l.sold_date IS NULL THEN COALESCE(ps.protected, 0) ELSE 0 END
            AS INTEGER) AS protected,
       -- A peak recorded before the ticker last went flat is stale (position_state
       -- is per ticker and outlives positions): it reads as 0 for a re-entry.
       CAST(CASE WHEN l.sold_date IS NULL AND ps.updated_at >= (
                     SELECT MIN(t2.timestamp_entry) FROM trades t2
                      WHERE t2.ticker_current = l.ticker AND t2.action = 'BUY'
                        AND t2.qty_open > {EPS})
                 THEN ps.peak_gain_pct ELSE 0 END AS REAL) AS peak_gain_pct
  FROM lots l
  LEFT JOIN position_state ps ON ps.ticker = l.ticker;
"""

VIEW_TRADE_OUTCOMES = f"""
CREATE VIEW IF NOT EXISTS trade_outcomes AS
SELECT t.trade_id                                   AS trade_id,
       CAST({_LEGACY_TS.format(c='t.timestamp_entry')} AS TEXT) AS timestamp_entry,
       t.timestamp_exit                             AS timestamp_exit,
       t.ticker                                     AS ticker,
       t.action                                     AS action,
       CAST(t.quantity_adj AS INTEGER)              AS quantity,
       t.price_entry_adj                            AS price_entry,
       CAST(CASE WHEN t.timestamp_exit IS NULL THEN NULL
            ELSE t.price_exit END AS REAL)          AS price_exit,
       t.pnl_dollar                                 AS pnl_dollar,
       t.pnl_pct                                    AS pnl_pct,
       t.hold_duration_mins                         AS hold_duration_mins,
       ea.signals                                   AS signals_fired,
       ea.confluence_score                          AS confluence_score,
       CAST(NULL AS TEXT)                           AS council_member_1_rec,
       CAST(NULL AS REAL)                           AS council_member_1_confidence,
       CAST(NULL AS TEXT)                           AS council_member_2_rec,
       CAST(NULL AS REAL)                           AS council_member_2_confidence,
       CAST(NULL AS INTEGER)                        AS council_agreement,
       CAST(NULL AS INTEGER)                        AS arbiter_invoked,
       CAST(NULL AS TEXT)                           AS arbiter_rec,
       ea.market_regime                             AS market_regime,
       ea.sector                                    AS sector,
       t.outcome_label                              AS outcome_label,
       tf.prediction_accuracy                       AS prediction_accuracy,
       tf.thesis_score                              AS thesis_score,
       CAST(0 AS INTEGER)                           AS entry_price_provisional,
       tf.mfe_pct                                   AS mfe_pct,
       tf.give_back_pct                             AS give_back_pct,
       tf.post_exit_peak_pct                        AS post_exit_peak_pct,
       tf.post_exit_window_days                     AS post_exit_window_days,
       CAST(CASE WHEN t.timestamp_exit IS NULL THEN NULL
            WHEN t.last_exit_kind = 'event' THEN 'CORPORATE-ACTION'
            ELSE xa.exit_reason END AS TEXT)        AS exit_reason,
       tf.features_filled_at                        AS features_filled_at,
       tf.forgone_gain_5d_pct                       AS forgone_gain_5d_pct,
       CAST(CASE WHEN t.timestamp_exit IS NULL THEN NULL
            ELSE xa.exit_params_snapshot END AS TEXT) AS exit_params_snapshot,
       CAST(COALESCE(ea.signal_attribution_source, 'none') AS TEXT) AS signal_attribution_source,
       tf.forgone_gain_14d_pct                      AS forgone_gain_14d_pct,
       tf.forgone_gain_30d_pct                      AS forgone_gain_30d_pct,
       tf.forgone_gain_60d_pct                      AS forgone_gain_60d_pct,
       tf.forgone_filled_at                         AS forgone_filled_at,
       ea.conviction                                AS conviction,
       ea.ml_confidence_at_entry                    AS ml_confidence_at_entry,
       ea.ml_signal_at_entry                        AS ml_signal_at_entry,
       ea.ml_trained_on_at_entry                    AS ml_trained_on_at_entry,
       CAST(NULL AS TEXT)                           AS legacy_signals_fired
  FROM trades t
  LEFT JOIN entry_annotations ea ON ea.trade_id = t.trade_id
  LEFT JOIN trade_features tf    ON tf.trade_id = t.trade_id
  LEFT JOIN exit_annotations xa  ON xa.decision_id = t.last_exit_decision_id;
"""

# positions = Σ signed fills ± position_events, per ticker. A symbol change
# moves qty_change shares from `ticker` to `new_ticker`; a spinoff adds
# qty_change shares of `new_ticker` and leaves the parent untouched.
VIEW_POSITIONS = f"""
CREATE VIEW IF NOT EXISTS positions AS
SELECT ticker, ROUND(SUM(q), {QTY_DP}) AS qty FROM (
    SELECT ticker, CASE side WHEN 'BUY' THEN quantity ELSE -quantity END AS q FROM fills
    UNION ALL
    SELECT ticker, CASE WHEN event_type = 'symbol_change' THEN -qty_change
                        ELSE qty_change END FROM position_events
     WHERE event_type <> 'spinoff'
    UNION ALL
    SELECT new_ticker, qty_change FROM position_events
     WHERE event_type IN ('symbol_change', 'spinoff') AND new_ticker IS NOT NULL
)
GROUP BY ticker
HAVING ABS(SUM(q)) > {EPS};
"""

VIEW_SIGNAL_PERFORMANCE = """
CREATE VIEW IF NOT EXISTS signal_performance AS
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

LEDGER_TABLES = ("fills", "fill_commissions", "position_events", "entry_annotations",
                 "exit_annotations", "trades", "trade_matches", "lots", "trade_features",
                 "position_state")
LEDGER_VIEWS = ("holdings", "trade_outcomes", "positions", "signal_performance")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _object_type(conn, name: str) -> Optional[str]:
    row = conn.execute("SELECT type FROM sqlite_master WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def is_migrated(conn) -> bool:
    """True once holdings/trade_outcomes are views over the fills ledger."""
    return _object_type(conn, "holdings") == "view"


def ensure_decision_columns(conn) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    if not cols:
        from kairos_log_db import SCHEMA_DECISIONS
        conn.executescript(SCHEMA_DECISIONS)
        conn.execute("ALTER TABLE decisions ADD COLUMN axis_weights_snapshot TEXT")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    for col, typ in (("trade_id", "TEXT"), ("order_id", "INTEGER"), ("perm_id", "INTEGER")):
        if col not in cols:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {col} {typ}")


def ensure_schema(conn, views: bool = True) -> None:
    """Create ledger tables/triggers (idempotent). Views only when the legacy
    tables of the same names are gone (i.e. after kairos_migrate_fills)."""
    ensure_decision_columns(conn)
    conn.executescript(SCHEMA_TABLES)
    conn.executescript(SCHEMA_DECISION_GUARD)
    if views:
        for name, sql in (("holdings", VIEW_HOLDINGS),
                          ("trade_outcomes", VIEW_TRADE_OUTCOMES),
                          ("positions", VIEW_POSITIONS)):
            if _object_type(conn, name) in (None, "view"):
                conn.executescript(sql)
        if _object_type(conn, "thesis_predictions") == "table" and \
                _object_type(conn, "signal_performance") is None:
            conn.executescript(VIEW_SIGNAL_PERFORMANCE)
    conn.commit()


def _own(conn):
    """(conn, owned) — open a connection when the caller did not pass one."""
    if conn is not None:
        return conn, False
    return get_connection(), True


# ── Alerts ───────────────────────────────────────────────────────────

def alert(text: str) -> None:
    """Post to #kairos-alerts via the existing helper. Never raises; always prints.

    Every ledger alert is a come-look (a trade record could not be written, the
    ledger disagrees with the broker, a short exists, the Flex pull failed), so
    the channel is fixed — and literal, so the watch selftest's channel audit
    can see this module as an #alerts producer."""
    print(f"  LEDGER ALERT: {text}")
    try:
        from kairos_alerts import post_message
        post_message("alerts", text)
    except Exception as exc:
        print(f"  (ledger alert post failed: {exc})")


# ── Timestamp helpers ────────────────────────────────────────────────

def to_utc_iso(value) -> str:
    """Any fill timestamp → 'YYYY-MM-DDTHH:MM:SSZ' (UTC).

    Accepts datetime (naive = UTC), Flex 'yyyymmdd;hhmmss' (US/Eastern), and
    ISO-ish strings with or without 'Z' / ' UTC' / offset (naive = UTC).
    """
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    s = str(value).strip()
    m = re.fullmatch(r"(\d{8});(\d{6})", s)
    if m:
        dt = datetime.strptime(s, "%Y%m%d;%H%M%S").replace(tzinfo=ET)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if re.fullmatch(r"\d{8}", s):
        dt = datetime.strptime(s, "%Y%m%d").replace(tzinfo=ET)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    s2 = s.replace(" UTC", "").replace("Z", "+00:00")
    s2 = re.sub(r"\s*\[.*\]$", "", s2)
    dt = datetime.fromisoformat(s2.replace(" ", "T", 1) if "T" not in s2 else s2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _parse_decision_ts(s: str) -> Optional[datetime]:
    try:
        return _parse_iso(to_utc_iso(s))
    except Exception:
        return None


# ── 1. record_decision ───────────────────────────────────────────────

ENTRY_ANNOTATION_FIELDS = ("signals", "signal_attribution_source", "confluence_score",
                           "conviction", "market_regime", "sector",
                           "ml_confidence_at_entry", "ml_signal_at_entry",
                           "ml_trained_on_at_entry")


def record_decision(*, timestamp: str, ticker: str, action: str, quantity,
                    rationale: str = "", data_inputs=None,
                    execution_price=None, execution_status=None, commission=None,
                    net_liq_after=None, position_after=None,
                    conviction_trade: bool = False, axis_weights_snapshot=None,
                    order_id=None, perm_id=None,
                    entry: Optional[dict] = None, exit: Optional[dict] = None,
                    trade_id: Optional[str] = None, conn=None) -> tuple[int, Optional[str]]:
    """Write a decision and its annotations in ONE transaction.

    BUY sent to the broker (status != 'Skipped'): mints trade_id and writes
      entry_annotations. `entry` is REQUIRED — a dict of
      ENTRY_ANNOTATION_FIELDS (signals as a list). The decisions trigger refuses
      the row otherwise, so this cannot be bypassed by calling SQL directly.
    SELL: writes exit_annotations. `exit` is REQUIRED: {exit_reason,
      exit_signals (list), exit_signals_note (required if exit_signals is
      empty), exit_params_snapshot (dict|None)}.
    Returns (decision_id, trade_id|None).
    """
    act = (action or "").upper()
    sent = (execution_status or "") != "Skipped"
    if act == "BUY" and sent and entry is None:
        raise ValueError("record_decision: a BUY sent to the broker requires entry annotations")
    if act == "SELL" and sent and exit is None:
        raise ValueError("record_decision: a SELL requires exit annotations (exit_reason, exit_signals)")
    if exit is not None:
        if not exit.get("exit_reason"):
            raise ValueError("record_decision: exit_reason is required")
        if exit.get("exit_signals") is None:
            raise ValueError("record_decision: exit_signals is required (use [] with a note)")
        if not exit["exit_signals"] and not exit.get("exit_signals_note"):
            raise ValueError("record_decision: empty exit_signals needs exit_signals_note")

    conn, owned = _own(conn)
    try:
        now = _now_iso()
        with conn:
            if act == "BUY" and sent:
                trade_id = trade_id or str(uuid.uuid4())
                e = entry or {}
                sigs = e.get("signals")
                conn.execute(
                    "INSERT INTO entry_annotations (trade_id, signals, "
                    "signal_attribution_source, confluence_score, conviction, "
                    "market_regime, sector, ml_confidence_at_entry, ml_signal_at_entry, "
                    "ml_trained_on_at_entry, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (trade_id, json.dumps(sigs) if sigs else None,
                     e.get("signal_attribution_source"), e.get("confluence_score"),
                     e.get("conviction"), e.get("market_regime"), e.get("sector"),
                     e.get("ml_confidence_at_entry"), e.get("ml_signal_at_entry"),
                     e.get("ml_trained_on_at_entry"), timestamp))
            else:
                trade_id = None
            cur = conn.execute(
                """INSERT INTO decisions
                   (timestamp, ticker, action, quantity, rationale, data_inputs,
                    execution_price, execution_status, commission, net_liq_after,
                    position_after, conviction_trade, axis_weights_snapshot,
                    trade_id, order_id, perm_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (timestamp, ticker, action, quantity, rationale,
                 (data_inputs if isinstance(data_inputs, str)
                  else json.dumps(data_inputs) if data_inputs else None),
                 execution_price, execution_status, commission, net_liq_after,
                 (position_after if isinstance(position_after, str)
                  else json.dumps(position_after) if position_after else None),
                 1 if conviction_trade else 0, axis_weights_snapshot,
                 trade_id, order_id, perm_id))
            decision_id = cur.lastrowid
            if exit is not None:
                snap = exit.get("exit_params_snapshot")
                conn.execute(
                    "INSERT INTO exit_annotations (decision_id, ticker, exit_reason, "
                    "exit_signals, exit_signals_note, exit_params_snapshot, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (decision_id, ticker, exit["exit_reason"],
                     json.dumps(list(exit["exit_signals"])), exit.get("exit_signals_note"),
                     json.dumps(snap) if isinstance(snap, dict) else snap, now))
        return decision_id, trade_id
    finally:
        if owned:
            conn.close()


# ── 2. record_fills ──────────────────────────────────────────────────

def _norm_side(side: str) -> str:
    s = (side or "").upper()
    if s in ("BOT", "BUY", "B"):
        return "BUY"
    if s in ("SLD", "SELL", "S"):
        return "SELL"
    raise ValueError(f"unknown fill side {side!r}")


def normalize_fills(trade_or_fills) -> list[dict]:
    """ib_insync Trade | list[Fill] | list[dict] → list of canonical fill dicts.

    Non-STK fills are dropped (equities only). The commission is NULL unless the
    broker's CommissionReport for that exec actually arrived.
    """
    if trade_or_fills is None:
        return []
    items = getattr(trade_or_fills, "fills", trade_or_fills)
    out = []
    for f in items or []:
        if isinstance(f, dict):
            d = dict(f)
            d["side"] = _norm_side(d["side"])
            d["quantity"] = abs(float(d["quantity"]))
            d["price"] = float(d["price"])
            c = d.get("commission")
            d["commission"] = None if c in (None, "") else abs(float(c))
            d["executed_at"] = to_utc_iso(d["executed_at"])
            d["ticker"] = str(d["ticker"]).upper()
            out.append(d)
            continue
        con, ex = f.contract, f.execution
        if getattr(con, "secType", "STK") != "STK":
            continue
        cr = getattr(f, "commissionReport", None)
        comm = None
        if cr is not None and getattr(cr, "execId", "") and cr.commission is not None \
                and abs(cr.commission) < 1e6:
            comm = abs(float(cr.commission))
        out.append(dict(
            exec_id=ex.execId, perm_id=int(ex.permId) if ex.permId else None,
            order_id=None, ticker=con.symbol.upper(), side=_norm_side(ex.side),
            quantity=abs(float(ex.shares)), price=float(ex.price), commission=comm,
            executed_at=to_utc_iso(ex.time)))
    return out


def fills_as_dicts(trade_or_fills) -> list[dict]:
    """JSON-safe fill list for an execution result dict (logged verbatim)."""
    return normalize_fills(trade_or_fills)


def _order_key(f) -> str:
    if f.get("perm_id"):
        return f"P{int(f['perm_id'])}"
    if f.get("order_id"):
        return f"O{f['order_id']}"
    return f"X{f['exec_id']}"


def _link_orders(conn, orders: dict) -> dict:
    """{order_key: (decision_id|None, link_method)} for orders with no given decision.

    1. perm_id — decisions.perm_id recorded at execution time (exact).
    2. order_sibling — another fill of the same order is already linked.
    3. nearest_session — same ticker, same side, same US/Eastern session date,
       any status except 'Skipped', not already claimed by a different order;
       best = matching quantity first, then smallest |Δt|. One decision links to
       at most one order. Never invents a decision.
    """
    result = {}
    claimed = {r[0]: r[1] for r in conn.execute(
        "SELECT decision_id, MIN(COALESCE('P'||perm_id, 'O'||order_id, 'X'||exec_id)) "
        "FROM fills WHERE decision_id IS NOT NULL GROUP BY decision_id")}
    pending = []
    for key, o in orders.items():
        if o.get("perm_id"):
            row = conn.execute("SELECT id FROM decisions WHERE perm_id = ? ORDER BY id LIMIT 1",
                               (o["perm_id"],)).fetchone()
            if row:
                result[key] = (row[0], "perm_id")
                continue
        sib = None
        if o.get("perm_id"):
            sib = conn.execute("SELECT decision_id FROM fills WHERE perm_id = ? AND "
                               "decision_id IS NOT NULL LIMIT 1", (o["perm_id"],)).fetchone()
        if not sib and o.get("order_id"):
            sib = conn.execute("SELECT decision_id FROM fills WHERE order_id = ? AND "
                               "decision_id IS NOT NULL LIMIT 1", (o["order_id"],)).fetchone()
        if sib:
            result[key] = (sib[0], "order_sibling")
            continue
        pending.append(key)

    pairs = []
    for key in pending:
        o = orders[key]
        first = _parse_iso(o["first_at"])
        day = first.astimezone(ET).date()
        lo = datetime(day.year, day.month, day.day, tzinfo=ET).astimezone(timezone.utc)
        hi = lo + timedelta(days=1)
        cands = conn.execute(
            "SELECT id, timestamp, quantity FROM decisions WHERE ticker = ? AND action = ? "
            "AND COALESCE(execution_status,'') <> 'Skipped' "
            "AND substr(timestamp,1,19) >= ? AND substr(timestamp,1,19) < ?",
            (o["ticker"], o["side"], lo.strftime("%Y-%m-%d %H:%M:%S"),
             hi.strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
        for c in cands:
            ts = _parse_decision_ts(c["timestamp"])
            if ts is None or ts.astimezone(ET).date() != day:
                continue
            qty_mismatch = 0 if abs(float(c["quantity"] or 0) - o["qty"]) < EPS else 1
            pairs.append((qty_mismatch, abs((ts - first).total_seconds()), key, c["id"]))
    pairs.sort()
    taken_orders, taken_dec = set(), set()
    for _mm, _dt, key, did in pairs:
        if key in taken_orders or did in taken_dec:
            continue
        if did in claimed and claimed[did] != key:
            continue
        taken_orders.add(key)
        taken_dec.add(did)
        result[key] = (did, "nearest_session")
    for key in pending:
        result.setdefault(key, (None, "unlinked"))
    return result


def record_fills(trade_or_fills, decision_id: Optional[int] = None,
                 source: str = "live", conn=None) -> dict:
    """Append executions to `fills` (INSERT OR IGNORE on exec_id). Sole writer.

    Records EVERY entry in trade.fills (not just the first). With decision_id,
    fills are linked 'live'; without, each order is linked by perm_id, a linked
    sibling fill, or the nearest same-session decision (see _link_orders).
    A statement/flex row for an exec already stored WITHOUT a commission adds
    that commission to fill_commissions (fills itself is never updated).
    For flex/statement rows, a perm_id already known for the same IBOrderID is
    carried onto new rows so one order never splits into two trades.
    Returns {"inserted", "ignored", "commissions"}.
    """
    if source not in FILL_SOURCES:
        raise ValueError(f"record_fills: unknown source {source!r}")
    fills = normalize_fills(trade_or_fills)
    stats = {"inserted": 0, "ignored": 0, "commissions": 0}
    if not fills:
        return stats
    conn, owned = _own(conn)
    try:
        now = _now_iso()
        with conn:
            existing = {}
            for f in fills:
                r = conn.execute("SELECT perm_id, order_id, commission FROM fills WHERE exec_id = ?",
                                 (f["exec_id"],)).fetchone()
                if r:
                    existing[f["exec_id"]] = r
            # Carry perm_id across an IBOrderID group (flex/statement batches).
            perm_for_order = {}
            for f in fills:
                oid = f.get("order_id")
                if not oid:
                    continue
                r = existing.get(f["exec_id"])
                if r is not None and r["perm_id"]:
                    perm_for_order[oid] = r["perm_id"]
                elif f.get("perm_id"):
                    perm_for_order.setdefault(oid, f["perm_id"])
            for oid in {f.get("order_id") for f in fills if f.get("order_id")} - set(perm_for_order):
                r = conn.execute("SELECT perm_id FROM fills WHERE order_id = ? AND perm_id IS NOT NULL "
                                 "LIMIT 1", (oid,)).fetchone()
                if r:
                    perm_for_order[oid] = r[0]
            new = []
            for f in fills:
                if f["exec_id"] in existing:
                    stats["ignored"] += 1
                    r = existing[f["exec_id"]]
                    if r["commission"] is None and f.get("commission") is not None:
                        cur = conn.execute(
                            "INSERT OR IGNORE INTO fill_commissions (exec_id, commission, "
                            "source, inserted_at) VALUES (?,?,?,?)",
                            (f["exec_id"], f["commission"], source, now))
                        stats["commissions"] += cur.rowcount
                    continue
                if not f.get("perm_id") and f.get("order_id") in perm_for_order:
                    f["perm_id"] = perm_for_order[f["order_id"]]
                new.append(f)
            links = {}
            if decision_id is None and new:
                orders = {}
                for f in new:
                    k = _order_key(f)
                    o = orders.setdefault(k, dict(ticker=f["ticker"], side=f["side"], qty=0.0,
                                                  first_at=f["executed_at"],
                                                  perm_id=f.get("perm_id"),
                                                  order_id=f.get("order_id")))
                    o["qty"] += f["quantity"]
                    o["first_at"] = min(o["first_at"], f["executed_at"])
                links = _link_orders(conn, orders)
            for f in new:
                if decision_id is not None:
                    did, how = decision_id, "live"
                else:
                    did, how = links.get(_order_key(f), (None, "unlinked"))
                cur = conn.execute(
                    "INSERT OR IGNORE INTO fills (exec_id, perm_id, order_id, decision_id, "
                    "link_method, ticker, side, quantity, price, commission, executed_at, "
                    "source, inserted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f["exec_id"], f.get("perm_id"), f.get("order_id"), did, how,
                     f["ticker"], f["side"], round(f["quantity"], QTY_DP), f["price"],
                     f.get("commission"), f["executed_at"], source, now))
                stats["inserted"] += cur.rowcount
                stats["ignored"] += 1 - cur.rowcount
        return stats
    finally:
        if owned:
            conn.close()


# ── position_events writer ───────────────────────────────────────────

def record_position_events(events: Iterable[dict], conn=None) -> int:
    """Append corporate actions / manual balances. Sole writer. INSERT OR IGNORE."""
    conn, owned = _own(conn)
    n = 0
    try:
        now = _now_iso()
        with conn:
            for e in events:
                if e["event_type"] not in EVENT_TYPES:
                    raise ValueError(f"unknown event_type {e['event_type']!r}")
                cur = conn.execute(
                    "INSERT OR IGNORE INTO position_events (event_id, ticker, event_type, "
                    "qty_change, ratio, new_ticker, cash_per_share, proceeds, effective_at, "
                    "source, note, inserted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (e["event_id"], e["ticker"], e["event_type"], float(e["qty_change"]),
                     e.get("ratio"), e.get("new_ticker"), e.get("cash_per_share"),
                     e.get("proceeds"), to_utc_iso(e["effective_at"]), e["source"],
                     e.get("note"), now))
                n += cur.rowcount
        return n
    finally:
        if owned:
            conn.close()


# ── 4. rebuild_trades ────────────────────────────────────────────────

class _Lot:
    __slots__ = ("trade_id", "qty", "cost_per_unit", "comm_per_unit", "opened_at", "short")

    def __init__(self, trade_id, qty, cost_per_unit, comm_per_unit, opened_at, short=False):
        self.trade_id, self.qty = trade_id, qty
        self.cost_per_unit, self.comm_per_unit = cost_per_unit, comm_per_unit
        self.opened_at, self.short = opened_at, short


def _effective_commission(conn) -> dict:
    return {r[0]: r[1] for r in conn.execute(
        "SELECT f.exec_id, COALESCE(f.commission, c.commission) FROM fills f "
        "LEFT JOIN fill_commissions c ON c.exec_id = f.exec_id")}


def rebuild_trades(conn=None, verbose: bool = False) -> dict:
    """Deterministic FIFO rebuild of `trades` + `trade_matches`. Sole writer.

    Inputs are fills + position_events only (plus decisions.trade_id for naming
    a trade after its BUY decision). Corporate actions:
      split          open lots rescaled in qty, cost preserved
      symbol_change  open lots carried to new_ticker
      merger_cash    open lots closed at cash_per_share (exit_kind 'event',
                     trade exit_reason 'CORPORATE-ACTION' in the view)
      opening_balance a lot with unknown cost basis (P&L stays NULL)
      spinoff        a new lot in new_ticker with unknown basis
    A SELL beyond the open long quantity opens a short lot (a long-only
    violation, recorded as it happened); later BUYs cover it first.
    """
    conn, owned = _own(conn)
    try:
        comm = _effective_commission(conn)
        dec = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT id, action, trade_id FROM decisions WHERE id IN "
            "(SELECT DISTINCT decision_id FROM fills WHERE decision_id IS NOT NULL)")}
        fills = [dict(r) for r in conn.execute("SELECT * FROM fills ORDER BY executed_at, exec_id")]
        events = [dict(r) for r in conn.execute(
            "SELECT * FROM position_events ORDER BY effective_at, event_id")]

        # trade_id per BUY order: its BUY decision's trade_id, else synthetic.
        order_tid = {}
        for f in fills:
            if f["side"] != "BUY":
                continue
            k = _order_key(f)
            d = dec.get(f["decision_id"])
            if d and d[0] == "BUY" and d[1]:
                order_tid.setdefault(k, d[1])
        def tid_for(f):
            k = _order_key(f)
            return order_tid.get(k) or f"ORD-{k}"

        items = [(f["executed_at"], 1, f["exec_id"], "fill", f) for f in fills] + \
                [(e["effective_at"], 0, e["event_id"], "event", e) for e in events]
        items.sort(key=lambda x: (x[0], x[1], x[2]))
        # Splits arrive as a pair of rows (old line −, new line +) sharing an
        # effective time; apply the rescale once per (ticker, effective_at).
        split_groups = defaultdict(list)
        for e in events:
            if e["event_type"] == "split":
                split_groups[(e["ticker"], e["effective_at"])].append(e)
        applied_splits = set()

        longs = defaultdict(list)    # ticker -> [ _Lot ] FIFO
        shorts = defaultdict(list)
        T = {}                       # trade_id -> accumulator
        matches = {}                 # (trade_id, exit_ref) -> accumulator
        seq = [0]

        def trade(tid, ticker, action, origin, at, key=None, did=None):
            t = T.get(tid)
            if t is None:
                t = T[tid] = dict(trade_id=tid, ticker=ticker, ticker_current=ticker,
                                  action=action, origin=origin, order_key=key,
                                  decision_id=did, qty=0.0, cost=0.0, cost_known=True,
                                  entry_comm=0.0, comm_complete=True, first_at=at,
                                  first_seq=seq[0])
            return t

        def match(lot, ticker, qty, price, exit_comm, ref, kind, at, dec_id):
            seq[0] += 1
            key = (lot.trade_id, ref)
            m = matches.get(key)
            if m is None:
                m = matches[key] = dict(trade_id=lot.trade_id, exit_ref=ref, exit_kind=kind,
                                        ticker=ticker, qty=0.0, px_qty=0.0, entry_cost=0.0,
                                        cost_known=True, entry_comm=0.0, exit_comm=0.0,
                                        exit_at=at, seq=seq[0], dec_id=dec_id)
            m["qty"] += qty
            m["px_qty"] += qty * price
            if lot.cost_per_unit is None:
                m["cost_known"] = False
            else:
                m["entry_cost"] += qty * lot.cost_per_unit
            m["entry_comm"] += qty * lot.comm_per_unit
            m["exit_comm"] += exit_comm
            m["seq"] = seq[0]
            T[lot.trade_id]["ticker_current"] = ticker

        def consume(book, ticker, qty, price, comm_total, total_qty, ref, kind, at, dec_id):
            """FIFO-consume `qty` from book[ticker]; returns the unconsumed rest."""
            lots = book[ticker]
            while qty > EPS and lots:
                lot = lots[0]
                m = min(qty, lot.qty)
                match(lot, ticker, m, price, (comm_total or 0.0) * m / total_qty,
                      ref, kind, at, dec_id)
                lot.qty = round(lot.qty - m, QTY_DP + 2)
                qty = round(qty - m, QTY_DP + 2)
                if lot.qty <= EPS:
                    lots.pop(0)
            return qty

        for at, _pri, ref, kind, x in items:
            seq[0] += 1
            if kind == "fill":
                f = x
                tk, q, px = f["ticker"], float(f["quantity"]), float(f["price"])
                c = comm.get(f["exec_id"])
                c_known = c is not None
                d = dec.get(f["decision_id"])
                if f["side"] == "BUY":
                    rest = consume(shorts, tk, q, px, c, q, f["exec_id"], "fill", at,
                                   f["decision_id"])
                    if rest > EPS:
                        tid = tid_for(f)
                        t = trade(tid, tk, "BUY", "fills", at, _order_key(f),
                                  f["decision_id"] if d and d[0] == "BUY" else None)
                        t["qty"] += rest
                        t["cost"] += rest * px
                        t["entry_comm"] += (c or 0.0) * rest / q
                        t["comm_complete"] &= c_known
                        lot = next((l for l in longs[tk] if l.trade_id == tid), None)
                        if lot is None:
                            longs[tk].append(_Lot(tid, rest, px, (c or 0.0) / q, at))
                        else:
                            new_qty = lot.qty + rest
                            lot.cost_per_unit = (lot.cost_per_unit * lot.qty + px * rest) / new_qty
                            lot.comm_per_unit = (lot.comm_per_unit * lot.qty
                                                 + (c or 0.0) * rest / q) / new_qty
                            lot.qty = new_qty
                    if rest < q - EPS:
                        for m in matches.values():
                            if m["exit_ref"] == f["exec_id"]:
                                T[m["trade_id"]]["comm_complete"] &= c_known
                else:
                    rest = consume(longs, tk, q, px, c, q, f["exec_id"], "fill", at,
                                   f["decision_id"])
                    for m in matches.values():
                        if m["exit_ref"] == f["exec_id"]:
                            T[m["trade_id"]]["comm_complete"] &= c_known
                    if rest > EPS:
                        tid = f"SHORT-{_order_key(f)}"
                        t = trade(tid, tk, "SELL", "fills", at, _order_key(f),
                                  f["decision_id"] if d and d[0] == "SELL" else None)
                        t["qty"] += rest
                        t["cost"] += rest * px          # short sale proceeds
                        t["entry_comm"] += (c or 0.0) * rest / q
                        t["comm_complete"] &= c_known
                        lot = next((l for l in shorts[tk] if l.trade_id == tid), None)
                        if lot is None:
                            shorts[tk].append(_Lot(tid, rest, px, (c or 0.0) / q, at, True))
                        else:
                            new_qty = lot.qty + rest
                            lot.cost_per_unit = (lot.cost_per_unit * lot.qty + px * rest) / new_qty
                            lot.qty = new_qty
                continue

            e = x
            et, tk = e["event_type"], e["ticker"]
            if et == "opening_balance":
                q = float(e["qty_change"])
                t = trade(e["event_id"], tk, "BUY", "opening_balance", at)
                t["qty"] += q
                t["cost_known"] = False
                longs[tk].append(_Lot(e["event_id"], q, None, 0.0, at))
            elif et == "split":
                gk = (tk, e["effective_at"])
                if gk in applied_splits:
                    continue
                applied_splits.add(gk)
                grp = split_groups[gk]
                neg = -sum(float(g["qty_change"]) for g in grp if g["qty_change"] < 0)
                pos = sum(float(g["qty_change"]) for g in grp if g["qty_change"] > 0)
                ratio = (pos / neg) if neg > EPS else float(e["ratio"] or 1.0)
                lots = longs[tk]
                target = round(sum(l.qty for l in lots) * ratio, QTY_DP) if neg <= EPS else pos
                done = 0.0
                for i, lot in enumerate(lots):
                    new_q = (target - done) if i == len(lots) - 1 else round(lot.qty * ratio, QTY_DP)
                    if lot.cost_per_unit is not None:
                        lot.cost_per_unit = lot.cost_per_unit * lot.qty / new_q
                    lot.comm_per_unit = lot.comm_per_unit * lot.qty / new_q
                    lot.qty = new_q
                    done += new_q
            elif et == "symbol_change":
                nt = e["new_ticker"]
                for book in (longs, shorts):
                    moved = book.pop(tk, [])
                    for lot in moved:
                        T[lot.trade_id]["ticker_current"] = nt
                    book[nt] = sorted(book[nt] + moved, key=lambda l: l.opened_at)
            elif et == "merger_cash":
                q = -float(e["qty_change"])
                rest = consume(longs, tk, q, float(e["cash_per_share"]), 0.0, q,
                               e["event_id"], "event", at, None)
                if rest > EPS:
                    print(f"  rebuild_trades: {e['event_id']} closes {rest:g} more {tk} "
                          f"shares than were open")
            elif et == "spinoff":
                nt, q = e["new_ticker"], float(e["qty_change"])
                t = trade(f"EVT-{e['event_id']}", nt, "BUY", "spinoff", at)
                t["qty"] += q
                t["cost_known"] = False
                longs[nt].append(_Lot(t["trade_id"], q, None, 0.0, at))

        open_qty = defaultdict(float)
        for book in (longs, shorts):
            for tk, lots in book.items():
                for lot in lots:
                    open_qty[lot.trade_id] += lot.qty
                    T[lot.trade_id]["ticker_current"] = tk

        by_trade = defaultdict(list)
        for m in matches.values():
            by_trade[m["trade_id"]].append(m)

        rebuilt_at = _now_iso()
        trade_rows, match_rows = [], []
        for n, t in enumerate(sorted(T.values(), key=lambda t: (t["first_at"], t["first_seq"])), 1):
            ms = sorted(by_trade.get(t["trade_id"], []), key=lambda m: m["seq"])
            q_open = round(open_qty.get(t["trade_id"], 0.0), QTY_DP)
            q_closed = round(sum(m["qty"] for m in ms), QTY_DP)
            q_adj = round(q_open + q_closed, QTY_DP)
            cost = t["cost"] if t["cost_known"] else None
            px_entry = (t["cost"] / t["qty"]) if (t["cost_known"] and t["qty"] > EPS) else None
            px_adj = (cost / q_adj) if (cost is not None and q_adj > EPS) else None
            exit_comm = sum(m["exit_comm"] for m in ms)
            proceeds = sum(m["px_qty"] for m in ms) if ms else None
            px_exit = (proceeds / q_closed) if ms and q_closed > EPS else None
            realized = None
            if ms and all(m["cost_known"] for m in ms):
                matched_cost = sum(m["entry_cost"] for m in ms)
                entry_c = sum(m["entry_comm"] for m in ms)
                gross = (proceeds - matched_cost) if t["action"] == "BUY" else (matched_cost - proceeds)
                realized = round(gross - entry_c - exit_comm, 4)
            closed = bool(ms) and q_open <= EPS
            ts_exit = pnl = pnl_pct = hold = label = None
            last = ms[-1] if ms else None
            if closed:
                ts_exit = max(m["exit_at"] for m in ms)
                hold = max(0, int((_parse_iso(ts_exit) - _parse_iso(t["first_at"])).total_seconds() / 60))
                if realized is not None:
                    pnl = realized
                    base = sum(m["entry_cost"] for m in ms)
                    pnl_pct = round(pnl / base * 100, 4) if base else 0.0
                    label = "SCRATCH" if abs(pnl) < 0.01 else ("WIN" if pnl > 0 else "LOSS")
            last_dec = None
            if last and last["exit_kind"] == "fill":
                last_dec = last["dec_id"]
            trade_rows.append((
                n, t["trade_id"], t["ticker"], t["ticker_current"], t["action"], t["origin"],
                t["order_key"], t["decision_id"], round(t["qty"], QTY_DP), q_adj,
                round(px_entry, 6) if px_entry is not None else None,
                round(px_adj, 6) if px_adj is not None else None,
                round(cost, 4) if cost is not None else None, round(t["entry_comm"], 6),
                t["first_at"], q_open, q_closed,
                round(px_exit, 6) if px_exit is not None else None,
                round(proceeds, 4) if proceeds is not None else None, round(exit_comm, 6),
                ts_exit, pnl, pnl_pct, realized, hold, label,
                last["exit_kind"] if last else None, last["exit_ref"] if last else None,
                last_dec, 1 if t["comm_complete"] else 0, rebuilt_at))
            for m in ms:
                match_rows.append((
                    m["trade_id"], m["exit_ref"], m["exit_kind"], m["ticker"],
                    round(m["qty"], QTY_DP), round(m["px_qty"] / m["qty"], 6) if m["qty"] else 0.0,
                    round(m["entry_cost"], 4) if m["cost_known"] else None,
                    round(m["entry_comm"], 6), round(m["exit_comm"], 6), m["exit_at"], m["seq"]))
        match_rows.sort(key=lambda r: r[-1])
        legacy_ts = lambda iso: iso.replace("T", " ").replace("Z", " UTC")
        sold_ts = lambda iso: iso.replace("T", " ").replace("Z", "")
        first_at = {r[1]: r[14] for r in trade_rows}
        action = {r[1]: r[4] for r in trade_rows}
        lot_rows = [(r[1], r[3], legacy_ts(r[14]), r[11], r[15], None, None)
                    for r in trade_rows if r[4] == "BUY" and r[15] > EPS]
        # A CLOSED piece with unknown cost basis (an opening balance) is left out:
        # the replaced holdings table declared entry_price NOT NULL and readers
        # compute P&L from it. It stays in trades / trade_outcomes (P&L NULL).
        # OPEN lots are always shown — hiding one would hide a live position
        # from the exit engine.
        lot_rows += [(m[0], m[3], legacy_ts(first_at[m[0]]), m[6] / m[4],
                      m[4], sold_ts(m[9]), m[5])
                     for m in match_rows if action[m[0]] == "BUY" and m[6] is not None and m[4]]
        with conn:
            conn.execute("DELETE FROM lots")
            conn.execute("DELETE FROM trade_matches")
            conn.execute("DELETE FROM trades")
            conn.executemany(
                "INSERT INTO lots (lot_no, trade_id, ticker, entry_date, entry_price, quantity, "
                "sold_date, sold_price) VALUES (?,?,?,?,?,?,?,?)",
                [(i, *r) for i, r in enumerate(lot_rows, 1)])
            conn.executemany(
                "INSERT INTO trades (trade_no, trade_id, ticker, ticker_current, action, origin, "
                "order_key, decision_id, quantity, quantity_adj, price_entry, price_entry_adj, "
                "cost_basis, entry_commission, timestamp_entry, qty_open, qty_closed, "
                "price_exit, proceeds, exit_commission, timestamp_exit, pnl_dollar, pnl_pct, "
                "realized_pnl, hold_duration_mins, outcome_label, last_exit_kind, "
                "last_exit_ref, last_exit_decision_id, commission_complete, rebuilt_at) "
                "VALUES (" + ",".join("?" * 31) + ")", trade_rows)
            conn.executemany(
                "INSERT INTO trade_matches (match_no, trade_id, exit_ref, exit_kind, ticker, qty, "
                "exit_price, entry_cost, entry_commission, exit_commission, exit_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(i, *r[:-1]) for i, r in enumerate(match_rows, 1)])
        summary = dict(trades=len(trade_rows), matches=len(match_rows),
                       open=sum(1 for r in trade_rows if r[15] > EPS),
                       closed=sum(1 for r in trade_rows if r[20] is not None))
        if verbose:
            print(f"  rebuild_trades: {summary}")
        return summary
    finally:
        if owned:
            conn.close()


def rebuild_trades_safely(context: str = "") -> Optional[dict]:
    """rebuild_trades() for the trade path: never raises; alerts on failure."""
    try:
        return rebuild_trades()
    except Exception as exc:
        alert(f":rotating_light: *Trade record* — rebuild_trades failed"
              f"{f' ({context})' if context else ''}: {exc}. Positions view is still "
              f"correct (it reads fills directly); trades/holdings are stale until the "
              f"next rebuild.")
        return None


# ── Trade-path composition (decision → fills → rebuild) ─────────────

def build_exit_annotation(ticker: str, exit_reason: str,
                          exit_signals: Optional[list] = None) -> dict:
    """The exit_annotations payload for a SELL of `ticker`.

    exit_signals records the thesis being exited, not just what happened to be
    firing at the moment of sale: the caller's signals ∪ the signals that drove
    the entry (entry_annotations). The regime snapshot is bounded by the
    position's oldest open entry so a re-entered ticker cannot inherit the arm
    context of an earlier, already-closed position. Never raises.
    """
    try:
        from kairos_log_db import entry_signals_before
        entry_sigs = entry_signals_before(ticker)
    except Exception:
        entry_sigs = []
    recorded = list(dict.fromkeys([str(s) for s in (exit_signals or []) if s] + entry_sigs))
    snapshot = None
    try:
        from kairos_ml_outcomes import build_exit_params_snapshot
        snapshot = build_exit_params_snapshot(ticker=ticker,
                                              entry_date=oldest_open_entry(ticker))
    except Exception as exc:
        print(f"  build_exit_annotation: snapshot unavailable for {ticker}: {exc}")
    return dict(exit_reason=exit_reason or "SELL (unspecified)", exit_signals=recorded,
                exit_signals_note=None if recorded else
                "no signals firing at exit and no entry annotation for the position",
                exit_params_snapshot=snapshot)


def record_execution(execution: dict, *, context: str, **decision_kwargs) -> tuple:
    """Record one executed (or attempted) order: decision + annotations, then
    every fill of the order, then rebuild trades.

    A failed trade-record write NEVER blocks or raises into the order path — the
    order has already been placed. Each failure posts an alert instead.
    Returns (decision_id|None, trade_id|None).
    """
    decision_id = trade_id = None
    ticker = decision_kwargs.get("ticker", "?")
    try:
        decision_id, trade_id = record_decision(
            order_id=execution.get("order_id"), perm_id=execution.get("perm_id"),
            **decision_kwargs)
    except Exception as exc:
        alert(f":rotating_light: *Trade-record write failed* ({context}) — "
              f"{decision_kwargs.get('action')} {ticker}: record_decision: {exc}. "
              f"The order was placed; its fills will still be captured by the "
              f"end-of-cycle broker check, but the decision/annotations are missing.")
    fills = execution.get("fills") or []
    if fills:
        try:
            record_fills(fills, decision_id, source="live")
        except Exception as exc:
            alert(f":rotating_light: *Trade-record write failed* ({context}) — "
                  f"{ticker}: record_fills: {exc}. The broker check will backfill them.")
    rebuild_trades_safely(context)
    return decision_id, trade_id


# ── Readers used by the trade path ───────────────────────────────────

def closed_lots_for_decision(decision_id: int, conn=None) -> list[dict]:
    """Lots a SELL decision's fills closed: [{entry_date, entry_price, quantity,
    holding_days}] — the shape sell_holdings used to return (Slack P&L lines)."""
    conn, owned = _own(conn)
    try:
        rows = conn.execute(
            "SELECT t.timestamp_entry, m.entry_cost, m.qty, m.exit_at FROM trade_matches m "
            "JOIN trades t ON t.trade_id = m.trade_id "
            "JOIN fills f ON f.exec_id = m.exit_ref WHERE f.decision_id = ? "
            "ORDER BY m.match_no", (decision_id,)).fetchall()
        out = []
        for r in rows:
            days = (_parse_iso(r["exit_at"]) - _parse_iso(r["timestamp_entry"])).days
            out.append({"entry_date": r["timestamp_entry"].replace("T", " ").replace("Z", " UTC"),
                        "entry_price": (r["entry_cost"] / r["qty"]) if r["entry_cost"] is not None else 0.0,
                        "quantity": r["qty"], "holding_days": days})
        return out
    finally:
        if owned:
            conn.close()


def oldest_open_entry(ticker: str, conn=None) -> Optional[str]:
    """timestamp_entry (legacy spelling) of the oldest open long lot of ticker."""
    conn, owned = _own(conn)
    try:
        r = conn.execute("SELECT MIN(timestamp_entry) FROM trades WHERE ticker_current = ? "
                         "AND action = 'BUY' AND qty_open > ?", (ticker, EPS)).fetchone()
        return r[0].replace("T", " ").replace("Z", " UTC") if r and r[0] else None
    finally:
        if owned:
            conn.close()


def broker_positions(ib) -> dict:
    """{ticker: qty} for STK positions at the broker (non-zero only)."""
    out = {}
    for p in ib.positions():
        if getattr(p.contract, "secType", "") != "STK":
            continue
        q = round(float(p.position), QTY_DP)
        if abs(q) > EPS:
            out[p.contract.symbol.upper()] = out.get(p.contract.symbol.upper(), 0.0) + q
    return out


def compare_positions(conn, broker: dict) -> list[tuple]:
    """[(ticker, ledger_qty, broker_qty)] for every mismatch."""
    ledger = {r[0]: float(r[1]) for r in conn.execute("SELECT ticker, qty FROM positions")}
    bad = []
    for tk in sorted(set(ledger) | set(broker)):
        a, b = ledger.get(tk, 0.0), broker.get(tk, 0.0)
        if abs(a - b) > EPS:
            bad.append((tk, a, b))
    return bad


# ── 3. End-of-cycle broker check ─────────────────────────────────────

def broker_check(ib=None, client_id: int = 8, conn=None, post: bool = True,
                 record: bool = True) -> dict:
    """reqExecutions() → record missing fills (source 'broker_check') → rebuild →
    compare the positions view with ib.positions() per ticker.

    On any mismatch it posts a come-look alert and CHANGES NOTHING — the ledger
    is never "corrected" towards the broker. The broker is the source of truth
    for what happened, and a mismatch means a fill or a corporate action is
    missing from the ledger: that has to be found, not papered over.
    Replaces reconcile_positions_against_broker's rewrite of holdings.
    record=False compares only (no fills written). A negative (short) broker
    position is alerted every run — Kairos is long-only.
    """
    summary = {"new_fills": 0, "mismatches": [], "error": None}
    own_ib = ib is None
    try:
        if own_ib:
            from ib_insync import IB
            ib = IB()
            ib.connect("127.0.0.1", 7497, clientId=client_id, timeout=15, readonly=True)
        from ib_insync import ExecutionFilter
        execs = ib.reqExecutions(ExecutionFilter())
        ib.reqPositions()
        ib.sleep(1)
        broker = broker_positions(ib)
    except Exception as exc:
        summary["error"] = f"broker read failed: {exc}"
        print(f"  broker_check: {summary['error']}")
        return summary
    finally:
        if own_ib and ib is not None:
            try:
                ib.disconnect()
            except Exception:
                pass
    shorts = {tk: q for tk, q in broker.items() if q < 0}
    summary["short_positions"] = shorts
    if shorts and post:
        alert(":rotating_light: *LONG-ONLY VIOLATION — short position at broker*\n"
              + "\n".join(f"• *{tk}*: {q:g} shares short" for tk, q in sorted(shorts.items()))
              + "\nKairos is long-only. Buy-to-cover to flatten; this repeats every "
                "broker check until flat. No automatic action has been taken.")
    conn2, owned = _own(conn)
    try:
        if record:
            st = record_fills(execs, None, source="broker_check", conn=conn2)
            summary["new_fills"] = st["inserted"]
            if st["inserted"] or st["commissions"]:
                rebuild_trades(conn2)
        if not broker:
            summary["error"] = "broker returned 0 STK positions — comparison skipped"
            print(f"  broker_check: {summary['error']}")
            return summary
        summary["mismatches"] = compare_positions(conn2, broker)
    finally:
        if owned:
            conn2.close()
    print(f"  broker_check: {summary['new_fills']} missing fill(s) recorded; "
          f"{len(summary['mismatches'])} position mismatch(es)")
    if summary["mismatches"] and post:
        lines = "\n".join(f"• *{tk}*: ledger {a:g} vs broker {b:g} (diff {a - b:+g})"
                          for tk, a, b in summary["mismatches"][:25])
        alert(":mag: *Positions ledger ≠ broker — come look*\n" + lines +
              "\nNothing was changed. A fill or corporate action is missing from the "
              "ledger (fills / position_events); find it and record it — do not edit "
              "positions to match.")
    return summary


# ── Flex (statement + web service) parsing ───────────────────────────

def parse_flex_csv(text: str) -> dict:
    """Parse an IBKR Flex CSV (BOF/BOS/HEADER/DATA framing) into
    {section: [row dict, ...]}. Sections of interest: TRNT, POST, CORP."""
    rows = list(csv.reader(io.StringIO(text)))
    headers, out = {}, defaultdict(list)
    for r in rows:
        if len(r) < 2:
            continue
        if r[0] == "HEADER":
            headers[r[1]] = r
        elif r[0] == "DATA" and r[1] in headers:
            out[r[1]].append(dict(zip(headers[r[1]], r)))
    return dict(out)


def flex_trnt_to_fills(trnt_rows: list[dict], perm_by_exec: Optional[dict] = None) -> list[dict]:
    """TRNT rows → canonical fill dicts (STK only)."""
    out = []
    for x in trnt_rows:
        if x.get("AssetClass") and x["AssetClass"] != "STK":
            continue
        if x.get("Put/Call"):
            continue
        q = float(x["Quantity"])
        if abs(q) < EPS:
            continue
        exec_id = x["IBExecID"]
        out.append(dict(
            exec_id=exec_id,
            perm_id=(perm_by_exec or {}).get(exec_id),
            order_id=x.get("IBOrderID") or None,
            ticker=x["Symbol"].upper(),
            side=x.get("Buy/Sell") or ("BUY" if q > 0 else "SELL"),
            quantity=abs(q), price=float(x["TradePrice"]),
            commission=(abs(float(x["IBCommission"])) if x.get("IBCommission") not in (None, "")
                        else None),
            executed_at=x["DateTime"]))
    return out


def flex_corp_to_events(corp_rows: list[dict], source: str) -> list[dict]:
    """CORP DETAIL rows → position_events (keyed on TransactionID). SUMMARY rows
    duplicate DETAIL and are ignored. Supports splits (RS/FS) and cash mergers
    (TC). Anything else is returned under key 'unsupported' for an alert."""
    events, unsupported = [], []
    for x in corp_rows:
        if x.get("LevelOfDetail") != "DETAIL":
            continue
        if x.get("AssetClass") and x["AssetClass"] != "STK":
            continue
        typ, desc = x.get("Type", ""), x.get("Description", "")
        ticker = (x.get("UnderlyingSymbol") or x.get("Symbol") or "").upper()
        tid = x.get("TransactionID")
        base = dict(event_id=f"IBKR-CORP-{tid}", ticker=ticker, source=source,
                    effective_at=x.get("Date/Time") or x.get("Report Date"),
                    qty_change=float(x.get("Quantity") or 0),
                    note=f"ActionID {x.get('ActionID')}: {desc}")
        if typ in ("RS", "FS"):
            m = re.search(r"SPLIT (\d+) FOR (\d+)", desc)
            ratio = (float(m.group(1)) / float(m.group(2))) if m else None
            events.append(dict(base, event_type="split", ratio=ratio))
        elif typ == "TC":
            q = float(x.get("Quantity") or 0)
            proceeds = float(x.get("Proceeds") or 0)
            m = re.search(r"FOR USD ([\d.]+) PER SHARE", desc)
            cps = float(m.group(1)) if m else (proceeds / -q if q else None)
            events.append(dict(base, event_type="merger_cash", cash_per_share=cps,
                               proceeds=proceeds))
        else:
            unsupported.append(x)
    return {"events": events, "unsupported": unsupported}


def fetch_flex_statement(token: Optional[str] = None, query_id: Optional[str] = None,
                         polls: int = 24, wait: float = 5.0) -> str:
    """Pull a Flex query via the Flex Web Service. Returns the CSV text.

    Token/query id come from env (IBKR_FLEX_TOKEN / IBKR_FLEX_QUERY_ID) and are
    never printed or logged — every error message is scrubbed of the token.
    Raises RuntimeError on any HTTP / service error.
    """
    import urllib.parse
    import urllib.request
    token = token or os.environ.get("IBKR_FLEX_TOKEN")
    query_id = query_id or os.environ.get("IBKR_FLEX_QUERY_ID")
    if not token or not query_id:
        raise RuntimeError("IBKR_FLEX_TOKEN / IBKR_FLEX_QUERY_ID not set in the environment")

    def scrub(s: str) -> str:
        return s.replace(token, "***")

    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Kairos/1.0"})
        return urllib.request.urlopen(req, timeout=60).read().decode()

    try:
        r = get(f"{FLEX_BASE}/SendRequest?" + urllib.parse.urlencode({"t": token, "q": query_id, "v": 3}))
    except Exception as exc:
        raise RuntimeError(scrub(f"SendRequest failed: {exc}")) from None
    ref = re.search(r"<ReferenceCode>(.*?)</ReferenceCode>", r)
    if not ref:
        err = re.search(r"<ErrorMessage>(.*?)</ErrorMessage>", r)
        raise RuntimeError(scrub("SendRequest refused: " + (err.group(1) if err else r[:300])))
    body = ""
    for _ in range(polls):
        time.sleep(wait)
        try:
            body = get(f"{FLEX_BASE}/GetStatement?" + urllib.parse.urlencode(
                {"t": token, "q": ref.group(1), "v": 3}))
        except Exception as exc:
            raise RuntimeError(scrub(f"GetStatement failed: {exc}")) from None
        if "<ErrorCode>1019</ErrorCode>" in body or "generation in progress" in body.lower():
            continue
        break
    if body.lstrip().startswith("<"):
        err = re.search(r"<ErrorMessage>(.*?)</ErrorMessage>", body)
        raise RuntimeError(scrub("GetStatement error: " + (err.group(1) if err else body[:300])))
    return body


def import_flex_text(text: str, source: str, conn=None) -> dict:
    """Record a Flex CSV's TRNT fills + CORP events. Returns counts."""
    parsed = parse_flex_csv(text)
    fills = flex_trnt_to_fills(parsed.get("TRNT", []))
    corp = flex_corp_to_events(parsed.get("CORP", []), source=f"ibkr_{source}")
    conn, owned = _own(conn)
    try:
        st = record_fills(fills, None, source=source, conn=conn)
        n_ev = record_position_events(corp["events"], conn=conn)
        return dict(fills_seen=len(fills), fills_inserted=st["inserted"],
                    commissions=st["commissions"], events_seen=len(corp["events"]),
                    events_inserted=n_ev, unsupported=corp["unsupported"])
    finally:
        if owned:
            conn.close()


def nightly_flex_pull(conn=None) -> Optional[dict]:
    """Nightly job step (a): Flex Web Service → fills + position_events.

    Never fails silently: any HTTP/token/service error posts an alert that names
    the token and its expiry, then returns None (the rest of the nightly job
    still runs — rebuild and features do not depend on this pull succeeding).
    """
    try:
        text = fetch_flex_statement()
    except Exception as exc:
        alert(f":rotating_light: *Flex nightly pull FAILED* — {exc}\n"
              f"Check IBKR_FLEX_TOKEN (Flex Web Service token, expires "
              f"{FLEX_TOKEN_EXPIRES}) and IBKR_FLEX_QUERY_ID. Fills missed by the live "
              f"path will not be backfilled until this succeeds.")
        return None
    try:
        res = import_flex_text(text, "flex_nightly", conn=conn)
    except Exception as exc:
        alert(f":rotating_light: *Flex nightly import FAILED* — statement fetched but "
              f"could not be recorded: {exc}")
        return None
    print(f"  Flex nightly: {res['fills_seen']} fills seen, {res['fills_inserted']} new, "
          f"{res['commissions']} late commission(s), {res['events_inserted']} new event(s)")
    if res["unsupported"]:
        alert(":warning: *Flex nightly* — unsupported corporate action(s) need a manual "
              "position_event: " + ", ".join(
                  f"{u.get('Symbol')} {u.get('Type')} {u.get('Date/Time')}" for u in res["unsupported"]))
    return res
