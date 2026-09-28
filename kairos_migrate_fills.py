"""
kairos_migrate_fills.py — migrate kairos.db to the fills ledger (kairos_ledger).

Idempotent: every step is INSERT OR IGNORE / IF NOT EXISTS / skip-if-done, so a
second run changes nothing and re-runs the gate.

    python kairos_migrate_fills.py --db ~/kairos-rehearsal/kairos.db   # rehearsal
    python kairos_migrate_fills.py --db ~/Kairos/kairos.db --live      # Phase 2 only

Steps
  1. Refuse if any kairos_run process exists, or if --db is the live database
     and --live was not passed.
  2. ATTACH the ML db; copy thesis_predictions / thesis_checkpoints (and the
     legacy trade_outcomes, as trade_outcomes_legacy); create the new tables.
  3. Import fills: Flex TRNT CSV (statement_import) + today's reqExecutions
     (broker_check). perm_id is recovered from logged execDetails lines.
  4. Import position_events: CORP DETAIL rows + three manual_verified events
     (SATS→ECHO symbol change, AAPL / CI opening balances).
  5. Link fills → decisions (done at insert); backfill decisions.perm_id /
     order_id from the logs; map every legacy trade_outcomes row to its
     decision so its trade_id is preserved; mint trade_ids for the rest.
  6. Entry/exit annotations from trade_outcomes, position_exits_history,
     decisions and armed_trail_context.
  7. rebuild_trades(); trade_features from legacy rows that map 1:1.
  8. Rename holdings, position_exits_history, position_exits, outcomes to
     *_legacy (read-only triggers; drop on/after 2026-10-28); create views;
     seed position_state.
  9. GATE: positions view == ib.positions() for every STK ticker. Prints a
     table; exits non-zero on any mismatch.
"""

import argparse
import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import kairos_ledger as L  # noqa: E402

LIVE_DIR = "/Users/jelmore/Kairos"
LIVE_DBS = {os.path.realpath(os.path.join(LIVE_DIR, n))
            for n in ("kairos.db", "kairos_ml_outcomes.db")}
IMPORTS = os.path.join(LIVE_DIR, "imports")
DEFAULT_FILLS_CSV = os.path.join(IMPORTS, "ibkr_fills_20260401_20260925.csv")
DEFAULT_STATEMENT_CSV = os.path.join(IMPORTS, "ibkr_statement_20260401_20260925_v2.csv")
LEGACY_TABLES = ("holdings", "position_exits_history", "position_exits", "outcomes")
LEGACY_DROP_AFTER = "2026-10-28"
BROKER_CLIENT_ID = 917

# Manual, evidence-backed events (source='manual_verified'). See the notes.
MANUAL_EVENTS = [
    dict(event_id="MANUAL-SYMCHG-SATS-ECHO-20260625", ticker="SATS", new_ticker="ECHO",
         event_type="symbol_change", qty_change=99.0,
         effective_at="2026-06-25T04:00:00Z", source="manual_verified",
         note=("EchoStar SATS -> ECHO symbol change. Same IBKR conid 47965865 for both: "
               "SATS conId=47965865 in kairos logs (2026-05-25 market-data lines), "
               "reqContractDetails(ECHO) -> 47965865 on 2026-09-28. IBKR produced no CORP "
               "row. Recorded on the first ECHO trade date (TRNT: ECHO -99 @97.15 "
               "2026-06-25 10:56:43 ET) at 00:00 ET. 99 shares = the SATS BUY of "
               "2026-06-15 (TRNT), never sold as SATS.")),
    dict(event_id="MANUAL-OPEN-AAPL-20260401", ticker="AAPL", event_type="opening_balance",
         qty_change=64.0, effective_at="2026-04-01T04:00:00Z", source="manual_verified",
         note=("Pre-statement opening balance: TRNT (2026-04-01..) sells 64 AAPL on "
               "2026-05-18 (-40, -24) with no prior BUY in the window; fills + events "
               "otherwise reproduce IBKR positions. Cost basis UNKNOWN (the pre-reset "
               "kairos DB holds qty-0 placeholder lots) — P&L of these shares stays NULL. "
               "Pre-Kairos era: excluded from learning.")),
    dict(event_id="MANUAL-OPEN-CI-20260401", ticker="CI", event_type="opening_balance",
         qty_change=19.0, effective_at="2026-04-01T04:00:00Z", source="manual_verified",
         note=("Pre-statement opening balance: TRNT sells 19 more CI than it buys before "
               "2026-05-18 (-19 on 2026-05-18). Cost basis UNKNOWN — P&L of these "
               "shares stays NULL. Pre-Kairos era: excluded from learning.")),
]
EXCLUDED_TRADE_TICKERS_AT_ENTRY = {"SATS"}   # the SATS→ECHO chain


def banner(t):
    print(f"\n{'━' * 72}\n  {t}\n{'━' * 72}")


# ── 1. Guards ────────────────────────────────────────────────────────

def guard(db: str, live: bool):
    ps = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    running = [l for l in ps.splitlines() if "kairos_run" in l and "grep" not in l]
    if running:
        sys.exit("REFUSED: kairos_run is running:\n" + "\n".join(running))
    if os.path.realpath(db) in LIVE_DBS and not live:
        sys.exit(f"REFUSED: {db} is the LIVE database. Pass --live (Phase 2 only).")
    if not os.path.exists(db):
        sys.exit(f"REFUSED: {db} does not exist")


# ── Helpers ──────────────────────────────────────────────────────────

_EXEC_RX = re.compile(r"execId='([^']+)'.*?permId=(\d+), clientId=(\d+), orderId=(\d+)")


def perm_ids_from_logs(logs_dir: str) -> dict:
    """{exec_id: (perm_id, client_id, api_order_id)} from logged execDetails lines."""
    out = {}
    pats = ["kairos_monitor.log", "kairos_commander.log", "kairos_ibkr_debug.log*"]
    for pat in pats:
        for path in glob.glob(os.path.join(logs_dir, pat)):
            with open(path, errors="ignore") as f:
                for line in f:
                    if "execId=" not in line:
                        continue
                    for m in _EXEC_RX.finditer(line):
                        out[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return out


def broker_read(snapshot_path: str | None):
    """(executions, positions) — from a JSON snapshot or a live read-only
    connection on a clientId Kairos never uses."""
    if snapshot_path:
        snap = json.load(open(snapshot_path))
        execs = [dict(exec_id=e["exec_id"], perm_id=e["perm_id"], order_id=None,
                      ticker=e["sym"], side=e["side"], quantity=e["qty"], price=e["px"],
                      commission=None, executed_at=e["t"])
                 for e in snap["executions"] if e.get("sec", "STK") == "STK"]
        pos = {p["sym"].upper(): round(float(p["qty"]), 4)
               for p in snap["positions"] if p.get("sec", "STK") == "STK" and abs(p["qty"]) > 1e-9}
        return execs, pos, f"snapshot {snapshot_path}"
    from ib_insync import IB, ExecutionFilter
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=BROKER_CLIENT_ID, readonly=True, timeout=20)
    try:
        fills = ib.reqExecutions(ExecutionFilter())
        ib.reqPositions()
        ib.sleep(1)
        pos = L.broker_positions(ib)
        execs = L.normalize_fills(fills)
    finally:
        ib.disconnect()
    return execs, pos, f"live IBKR clientId={BROKER_CLIENT_ID} readonly"


def legacy_table(conn, name: str) -> str:
    """The pre-ledger table's current name: `name` before step 8, `name_legacy`
    after it (so a re-run reads the same rows)."""
    return name if L._object_type(conn, name) == "table" else f"{name}_legacy"


def ts_close(a: str, b: str, seconds: int) -> bool:
    try:
        return abs((L._parse_iso(L.to_utc_iso(a)) - L._parse_iso(L.to_utc_iso(b)))
                   .total_seconds()) <= seconds
    except Exception:
        return False


# ── 2. Schema + ML copy ──────────────────────────────────────────────

def copy_ml(conn, ml_db: str):
    conn.execute("ATTACH DATABASE ? AS ml", (f"file:{ml_db}?mode=ro",))
    try:
        for name in ("thesis_predictions", "thesis_checkpoints"):
            if L._object_type(conn, name) is None:
                sql = conn.execute("SELECT sql FROM ml.sqlite_master WHERE type='table' AND name=?",
                                   (name,)).fetchone()[0]
                conn.execute(sql)
                for (isql,) in conn.execute(
                        "SELECT sql FROM ml.sqlite_master WHERE type='index' AND tbl_name=? "
                        "AND sql IS NOT NULL", (name,)).fetchall():
                    conn.execute(isql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS"))
            conn.execute(f"INSERT OR IGNORE INTO main.{name} SELECT * FROM ml.{name}")
        if L._object_type(conn, "trade_outcomes_legacy") is None:
            sql = conn.execute("SELECT sql FROM ml.sqlite_master WHERE type='table' "
                               "AND name='trade_outcomes'").fetchone()[0]
            conn.execute(sql.replace("CREATE TABLE trade_outcomes",
                                     "CREATE TABLE trade_outcomes_legacy", 1))
            conn.execute("INSERT INTO trade_outcomes_legacy SELECT * FROM ml.trade_outcomes")
        conn.commit()
        n = {t: conn.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
             for t in ("thesis_predictions", "thesis_checkpoints", "trade_outcomes_legacy")}
        m = {t: conn.execute(f"SELECT COUNT(*) FROM ml.{t}").fetchone()[0]
             for t in ("thesis_predictions", "thesis_checkpoints", "trade_outcomes")}
    finally:
        conn.execute("DETACH DATABASE ml")
    print(f"  copied from ML db: {n}  (source counts {m})")
    assert n["thesis_predictions"] == m["thesis_predictions"]
    assert n["thesis_checkpoints"] == m["thesis_checkpoints"]
    assert n["trade_outcomes_legacy"] == m["trade_outcomes"]


# ── 5. decisions: perm/order ids, legacy trade_id mapping ────────────

def backfill_decision_ids(conn, perm_map: dict):
    rows = conn.execute("SELECT f.decision_id, f.exec_id, f.perm_id FROM fills f "
                        "WHERE f.decision_id IS NOT NULL").fetchall()
    per_dec = defaultdict(set)
    api_ids = defaultdict(set)
    for r in rows:
        if r["perm_id"]:
            per_dec[r["decision_id"]].add(r["perm_id"])
        if r["exec_id"] in perm_map:
            api_ids[r["decision_id"]].add(perm_map[r["exec_id"]][2])
    n_perm = n_oid = 0
    for did, perms in per_dec.items():
        if len(perms) == 1:
            n_perm += conn.execute("UPDATE decisions SET perm_id = ? WHERE id = ? AND perm_id IS NULL",
                                   (next(iter(perms)), did)).rowcount
    for did, oids in api_ids.items():
        if len(oids) == 1:
            n_oid += conn.execute("UPDATE decisions SET order_id = ? WHERE id = ? AND order_id IS NULL",
                                  (next(iter(oids)), did)).rowcount
    conn.commit()
    print(f"  decisions.perm_id backfilled: {n_perm}   decisions.order_id (API) backfilled: {n_oid}")


def map_legacy_trade_ids(conn) -> dict:
    """legacy trade_outcomes row → BUY decision (exact ticker + timestamp, the
    two were written from the same variable), falling back to the nearest BUY
    decision of the same ticker within 5 minutes with the same quantity.
    Sets decisions.trade_id = legacy trade_id. Returns {trade_id: decision_id}."""
    legacy = conn.execute("SELECT trade_id, ticker, timestamp_entry, quantity, price_entry "
                          "FROM trade_outcomes_legacy ORDER BY timestamp_entry").fetchall()
    mapped, how = {}, defaultdict(int)
    for r in legacy:
        existing = conn.execute("SELECT id FROM decisions WHERE trade_id = ?",
                                (r["trade_id"],)).fetchone()
        if existing:
            mapped[r["trade_id"]] = existing[0]
            how["already"] += 1
            continue
        d = conn.execute("SELECT id, trade_id FROM decisions WHERE ticker = ? AND action = 'BUY' "
                         "AND timestamp = ? ORDER BY id", (r["ticker"], r["timestamp_entry"])).fetchall()
        d = [x for x in d if x["trade_id"] is None]
        method = "exact_timestamp"
        if not d:
            cands = conn.execute("SELECT id, timestamp, quantity, trade_id FROM decisions "
                                 "WHERE ticker = ? AND action = 'BUY' AND trade_id IS NULL "
                                 "AND COALESCE(execution_status,'') <> 'Skipped'",
                                 (r["ticker"],)).fetchall()
            d = [c for c in cands if ts_close(c["timestamp"], r["timestamp_entry"], 300)
                 and abs(float(c["quantity"] or 0) - float(r["quantity"] or 0)) < 1e-6]
            method = "nearest_5min_same_qty"
        if not d:
            how["unmapped"] += 1
            continue
        conn.execute("UPDATE decisions SET trade_id = ? WHERE id = ?", (r["trade_id"], d[0]["id"]))
        mapped[r["trade_id"]] = d[0]["id"]
        how[method] += 1
    conn.commit()
    print(f"  legacy trade_outcomes → decision: {dict(how)}")
    return mapped


def mint_trade_ids(conn) -> int:
    """A trade_id for every BUY decision whose order produced fills but that no
    legacy row claimed (e.g. the 'Expired' decisions whose orders filled)."""
    rows = conn.execute(
        "SELECT DISTINCT d.id FROM decisions d JOIN fills f ON f.decision_id = d.id "
        "WHERE d.action = 'BUY' AND d.trade_id IS NULL").fetchall()
    for r in rows:
        conn.execute("UPDATE decisions SET trade_id = ? WHERE id = ?", (str(uuid.uuid4()), r[0]))
    conn.commit()
    print(f"  new trade_ids minted for linked BUY decisions with no legacy row: {len(rows)}")
    return len(rows)


# ── 6. Annotations ───────────────────────────────────────────────────

def build_entry_annotations(conn) -> dict:
    legacy = {r["trade_id"]: r for r in conn.execute("SELECT * FROM trade_outcomes_legacy")}
    rows = conn.execute(
        "SELECT DISTINCT d.id, d.trade_id, d.ticker, d.timestamp, d.data_inputs FROM decisions d "
        "JOIN fills f ON f.decision_id = d.id WHERE d.action = 'BUY' AND d.trade_id IS NOT NULL"
    ).fetchall()
    stats = defaultdict(int)
    for d in rows:
        lg = legacy.get(d["trade_id"])
        if lg is not None:
            src = lg["signal_attribution_source"]
            vals = (lg["signals_fired"], src, lg["confluence_score"], lg["conviction"],
                    lg["market_regime"], lg["sector"], lg["ml_confidence_at_entry"],
                    lg["ml_signal_at_entry"], lg["ml_trained_on_at_entry"])
            stats["from_trade_outcomes"] += 1
        else:
            try:
                di = json.loads(d["data_inputs"] or "{}")
            except (TypeError, json.JSONDecodeError):
                di = {}
            conf = di.get("confluence") or {}
            sigs = [str(s).upper() for s in (conf.get("signals") or []) if s]
            src = "decision_record" if sigs else "none"
            vals = (json.dumps(sigs) if sigs else None, src, conf.get("score"), None, None,
                    di.get("sector") or None, None, None, None)
            stats["from_decision_record"] += 1
        vals = list(vals)
        if d["ticker"] in EXCLUDED_TRADE_TICKERS_AT_ENTRY:
            vals[1] = L.EXCLUDED_ATTRIBUTION
            stats["excluded"] += 1
        conn.execute(
            "INSERT OR IGNORE INTO entry_annotations (trade_id, signals, signal_attribution_source, "
            "confluence_score, conviction, market_regime, sector, ml_confidence_at_entry, "
            "ml_signal_at_entry, ml_trained_on_at_entry, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (d["trade_id"], *vals, d["timestamp"]))
    for e in MANUAL_EVENTS:
        if e["event_type"] == "opening_balance":
            conn.execute(
                "INSERT OR IGNORE INTO entry_annotations (trade_id, signal_attribution_source, "
                "created_at) VALUES (?, ?, ?)", (e["event_id"], L.EXCLUDED_ATTRIBUTION,
                                                 e["effective_at"]))
            stats["opening_balance_excluded"] += 1
    conn.commit()
    print(f"  entry_annotations: {dict(stats)}")
    return stats


def build_exit_annotations(conn) -> dict:
    """One row per SELL decision whose order produced fills."""
    hist = defaultdict(list)
    for h in conn.execute(f"SELECT * FROM {legacy_table(conn, 'position_exits_history')} ORDER BY id"):
        hist[h["ticker"]].append(h)
    legacy_closes = defaultdict(list)
    for r in conn.execute("SELECT ticker, timestamp_exit, exit_reason, exit_params_snapshot "
                          "FROM trade_outcomes_legacy WHERE timestamp_exit IS NOT NULL"):
        legacy_closes[r["ticker"]].append(r)
    arms = defaultdict(list)
    for a in conn.execute("SELECT * FROM armed_trail_context ORDER BY id"):
        arms[a["ticker"]].append(a)
    sells = conn.execute(
        "SELECT DISTINCT d.id, d.ticker, d.timestamp, d.rationale FROM decisions d "
        "JOIN fills f ON f.decision_id = d.id WHERE d.action = 'SELL'").fetchall()
    stats = defaultdict(int)
    for d in sells:
        h = next((x for x in hist.get(d["ticker"], [])
                  if x["exit_date"] and ts_close(x["exit_date"], d["timestamp"], 600)), None)
        lc = [x for x in legacy_closes.get(d["ticker"], [])
              if ts_close(x["timestamp_exit"], d["timestamp"], 600)]
        reason = (h["exit_reason"] if h and h["exit_reason"] else None) \
            or next((x["exit_reason"] for x in lc if x["exit_reason"]), None) \
            or (d["rationale"] or "").strip() or "SELL (unspecified)"
        sigs, note = [], None
        if h and h["exit_signals"]:
            try:
                sigs = [str(s) for s in json.loads(h["exit_signals"]) if s]
            except (TypeError, json.JSONDecodeError):
                sigs = []
        if not sigs:
            note = ("legacy exit (pre-ledger): no exit signals recorded"
                    if h else "legacy exit (pre-ledger): no position_exits_history row")
        snap = next((x["exit_params_snapshot"] for x in lc if x["exit_params_snapshot"]), None)
        if snap:
            stats["snapshot_from_trade_outcomes"] += 1
        else:
            arm = next((a for a in reversed(arms.get(d["ticker"], []))
                        if a["armed_at"] and a["armed_at"][:19] <= d["timestamp"][:19]), None)
            if arm is not None:
                snap = json.dumps({"reconstructed": True, "params": {}, "axis_weights": {},
                                   "armed_trail_context": {k: arm[k] for k in arm.keys()},
                                   "captured_at": L.to_utc_iso(d["timestamp"])})
                stats["snapshot_from_armed_trail_context"] += 1
        stats["with_history_row" if h else "without_history_row"] += 1
        conn.execute(
            "INSERT OR IGNORE INTO exit_annotations (decision_id, ticker, exit_reason, exit_signals, "
            "exit_signals_note, exit_params_snapshot, created_at) VALUES (?,?,?,?,?,?,?)",
            (d["id"], d["ticker"], reason, json.dumps(sigs), note, snap, d["timestamp"]))
    conn.commit()
    print(f"  exit_annotations for {len(sells)} filled SELL decisions: {dict(stats)}")
    return stats


# ── 7. trade_features from 1:1 legacy rows ───────────────────────────

FEATURE_COLS = ("mfe_pct", "give_back_pct", "post_exit_peak_pct", "post_exit_window_days",
                "forgone_gain_5d_pct", "forgone_gain_14d_pct", "forgone_gain_30d_pct",
                "forgone_gain_60d_pct", "features_filled_at", "forgone_filled_at",
                "prediction_accuracy", "thesis_score")


def legacy_maps_1to1(t, lg) -> bool:
    if abs(float(t["quantity_adj"]) - float(lg["quantity"] or 0)) > 1e-6:
        return False
    if t["price_entry_adj"] is None or abs(float(t["price_entry_adj"]) - float(lg["price_entry"])) > 0.01:
        return False
    if (t["timestamp_exit"] is None) != (lg["timestamp_exit"] is None):
        return False
    if t["timestamp_exit"] is not None:
        if lg["price_exit"] is None or t["price_exit"] is None:
            return False
        if abs(float(t["price_exit"]) - float(lg["price_exit"])) > 0.01:
            return False
        if not ts_close(t["timestamp_exit"], lg["timestamp_exit"], 86400):
            return False
    return True


def carry_features(conn) -> dict:
    legacy = {r["trade_id"]: r for r in conn.execute("SELECT * FROM trade_outcomes_legacy")}
    n = defaultdict(int)
    for t in conn.execute("SELECT * FROM trades").fetchall():
        lg = legacy.get(t["trade_id"])
        if lg is None:
            continue
        if not legacy_maps_1to1(t, lg):
            n["recompute"] += 1
            continue
        if all(lg[c] is None for c in FEATURE_COLS):
            n["1to1_no_features"] += 1
            continue
        conn.execute(
            f"INSERT OR IGNORE INTO trade_features (trade_id, {', '.join(FEATURE_COLS)}, source) "
            f"VALUES (?, {', '.join('?' * len(FEATURE_COLS))}, 'legacy_1to1')",
            (t["trade_id"], *[lg[c] for c in FEATURE_COLS]))
        n["carried"] += 1
    conn.commit()
    print(f"  trade_features: {dict(n)}")
    return n


# ── 8. Rename legacy + views + position_state ───────────────────────

def rename_and_view(conn):
    seeded = 0
    if L._object_type(conn, "holdings") == "table":
        now = L._now_iso()
        for r in conn.execute(
                "SELECT ticker, MAX(peak_gain_pct) pk, MAX(protected) pr, MAX(drip_enabled) dr "
                "FROM holdings WHERE sold_date IS NULL OR sold_date = '' GROUP BY ticker"):
            seeded += conn.execute(
                "INSERT OR IGNORE INTO position_state (ticker, peak_gain_pct, protected, "
                "drip_enabled, updated_at) VALUES (?,?,?,?,?)",
                (r["ticker"], r["pk"] or 0.0, r["pr"] or 0, r["dr"] or 0, now)).rowcount
    for t in LEGACY_TABLES:
        if L._object_type(conn, t) == "table" and L._object_type(conn, f"{t}_legacy") is None:
            conn.execute(f"ALTER TABLE {t} RENAME TO {t}_legacy")
    for t in (*LEGACY_TABLES, "trade_outcomes"):
        lt = f"{t}_legacy"
        if L._object_type(conn, lt) != "table":
            continue
        for op in ("INSERT", "UPDATE", "DELETE"):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS trg_{lt}_ro_{op.lower()} BEFORE {op} ON {lt} "
                f"BEGIN SELECT RAISE(ABORT, '{lt} is read-only (replaced by the fills ledger "
                f"2026-09-28; drop on/after {LEGACY_DROP_AFTER})'); END;")
    conn.commit()
    L.ensure_schema(conn, views=True)
    print(f"  position_state seeded: {seeded} ticker(s); legacy tables renamed + read-only; "
          f"views: {[v for v in L.LEDGER_VIEWS if L._object_type(conn, v) == 'view']}")


# ── 9. Gate ──────────────────────────────────────────────────────────

def gate(conn, broker: dict, where: str) -> list:
    ledger = {r[0]: float(r[1]) for r in conn.execute("SELECT ticker, qty FROM positions")}
    tickers = sorted(set(ledger) | set(broker))
    bad = L.compare_positions(conn, broker)
    print(f"  broker source: {where}")
    print(f"  {'ticker':<8}{'ledger':>12}{'broker':>12}  status")
    for tk in tickers:
        a, b = ledger.get(tk, 0.0), broker.get(tk, 0.0)
        ok = abs(a - b) <= L.EPS
        print(f"  {tk:<8}{a:>12g}{b:>12g}  {'ok' if ok else 'MISMATCH'}")
    print(f"  → {len(tickers) - len(bad)}/{len(tickers)} match, {len(bad)} mismatch")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--db", required=True, help="kairos.db to migrate (a COPY unless --live)")
    ap.add_argument("--live", action="store_true", help="allow the live database (Phase 2)")
    ap.add_argument("--ml-db", help="ML outcomes db (default: kairos_ml_outcomes.db next to --db)")
    ap.add_argument("--fills-csv", default=DEFAULT_FILLS_CSV)
    ap.add_argument("--statement-csv", default=DEFAULT_STATEMENT_CSV)
    ap.add_argument("--logs-dir", default=LIVE_DIR, help="where execDetails logs live (read only)")
    ap.add_argument("--broker-snapshot", help="JSON snapshot instead of a live broker read")
    args = ap.parse_args()

    db = os.path.abspath(os.path.expanduser(args.db))
    guard(db, args.live)
    ml_db = os.path.abspath(os.path.expanduser(
        args.ml_db or os.path.join(os.path.dirname(db), "kairos_ml_outcomes.db")))
    if not os.path.exists(ml_db):
        sys.exit(f"REFUSED: ML db {ml_db} not found")
    if os.path.realpath(ml_db) in LIVE_DBS and not args.live:
        sys.exit("REFUSED: --ml-db is the live ML database; pass --live (Phase 2 only).")

    conn = L.get_connection(db)
    print(f"  target: {db}\n  ml db : {ml_db} (attached read-only)")

    banner("2. Schema + ML tables")
    copy_ml(conn, ml_db)
    L.ensure_schema(conn, views=not (L._object_type(conn, "holdings") == "table"))

    banner("3. Fills")
    perm_map = perm_ids_from_logs(args.logs_dir)
    print(f"  execDetails recovered from logs: {len(perm_map)}")
    parsed = L.parse_flex_csv(open(args.fills_csv).read())
    fills = L.flex_trnt_to_fills(parsed["TRNT"], {k: v[0] for k, v in perm_map.items()})
    # One order, one key: carry a log-recovered perm_id to every exec of its IBOrderID.
    perm_by_order = {}
    for f in fills:
        if f["perm_id"]:
            perm_by_order.setdefault(f["order_id"], f["perm_id"])
    for f in fills:
        if not f["perm_id"] and f["order_id"] in perm_by_order:
            f["perm_id"] = perm_by_order[f["order_id"]]
    st = L.record_fills(fills, None, source="statement_import", conn=conn)
    print(f"  TRNT statement: {len(fills)} fills → {st}")
    execs, broker_pos, where = broker_read(args.broker_snapshot)
    st2 = L.record_fills(execs, None, source="broker_check", conn=conn)
    print(f"  today's reqExecutions: {len(execs)} STK fills → {st2}")
    for how, n in conn.execute("SELECT link_method, COUNT(*) FROM fills GROUP BY 1"):
        print(f"    link_method {how:<16} {n}")

    banner("4. Position events")
    stmt = L.parse_flex_csv(open(args.statement_csv).read())
    corp = L.flex_corp_to_events(stmt.get("CORP", []), source="statement_import")
    if corp["unsupported"]:
        sys.exit(f"REFUSED: unsupported CORP rows: {corp['unsupported']}")
    n_ev = L.record_position_events(corp["events"] + MANUAL_EVENTS, conn=conn)
    print(f"  CORP DETAIL events: {len(corp['events'])}  manual: {len(MANUAL_EVENTS)}  new: {n_ev}")
    for e in conn.execute("SELECT event_id, ticker, event_type, qty_change, ratio, new_ticker, "
                          "cash_per_share, proceeds, effective_at FROM position_events ORDER BY effective_at"):
        print(f"    {dict(e)}")

    banner("5. Decisions ⇄ fills ⇄ legacy trade ids")
    backfill_decision_ids(conn, perm_map)
    map_legacy_trade_ids(conn)
    mint_trade_ids(conn)

    banner("6. Annotations")
    build_entry_annotations(conn)
    build_exit_annotations(conn)

    banner("7. rebuild_trades + trade_features")
    print(f"  {L.rebuild_trades(conn)}")
    carry_features(conn)

    banner("8. Legacy rename + views")
    rename_and_view(conn)

    banner("9. GATE — positions view vs IBKR")
    bad = gate(conn, broker_pos, where)
    conn.close()
    if bad:
        print("\n  GATE FAILED")
        return 1
    print("\n  GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
