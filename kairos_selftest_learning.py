"""
Kairos learning-loop selftest — gates the unfreeze (Task 5).

Runs the redesigned two-sided / regime-windowed / freshness-gated compute+propose+
apply pipeline entirely against DISPOSABLE /tmp copies of the config + DBs. It never
touches the live kairos_config.json, kairos.db, or kairos_ml_outcomes.db. Exercises:

  1. regime window excludes pre-change (out-of-regime) trades         (2a)
  2. contributing-row min_sample gates                                (2a)
  3. two-sided score proposes LOOSENING when forgone dominates        (2b)
  4. two-sided score proposes TIGHTENING when give-back dominates     (2b)
  5. freshness gate blocks identical evidence, admits changed evidence(2c)
  6. cumulative 7-day ±40% cap blocks at compute AND at apply         (2e)
  7. hard bounds + 25%/step cap hold                                  (2f)
  8. a full propose -> approve cycle writes config + marks approved   (2f/2g)

Exit code 0 iff every assertion passes.
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import kairos_axis_weights as A
import kairos_log_db

TRAIL = "exits.trailing_stop.target_armed.trail_pct"
FLOOR = "exits.trailing_stop.profit_floor_pp"

_checks = {"pass": 0, "fail": 0}


def check(name, cond):
    if cond:
        _checks["pass"] += 1
        print(f"  PASS  {name}")
    else:
        _checks["fail"] += 1
        print(f"  FAIL  {name}")


def _snapshot(trail=8.0, floor=1.0, exit_timing=0.3626):
    return json.dumps({
        "params": {TRAIL: trail, FLOOR: floor},
        "axis_weights": {"exit_timing": exit_timing},
        "reconstructed": True,
    })


def _make_ml_db(path, rows):
    """rows: list of dicts with keys trail, floor, mfe, forgone, pnl, exit_timing,
    contributing(bool). Builds trade_outcomes with controlled snapshots."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE trade_outcomes (
        trade_id TEXT, timestamp_entry TEXT, timestamp_exit TEXT, ticker TEXT,
        action TEXT, pnl_pct REAL, mfe_pct REAL, give_back_pct REAL,
        forgone_gain_5d_pct REAL, post_exit_peak_pct REAL, exit_reason TEXT,
        exit_params_snapshot TEXT)""")
    for i, r in enumerate(rows):
        mfe = r.get("mfe")
        pnl = r.get("pnl", 0.0)
        forgone = r.get("forgone")
        if not r.get("contributing", True):
            # break contribution by nulling one required field
            forgone = None
        gb = (mfe - pnl) if (mfe is not None and pnl is not None) else None
        conn.execute(
            "INSERT INTO trade_outcomes (trade_id, timestamp_entry, timestamp_exit, "
            "ticker, action, pnl_pct, mfe_pct, give_back_pct, forgone_gain_5d_pct, "
            "post_exit_peak_pct, exit_reason, exit_params_snapshot) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"t{i:03d}", "2026-06-01 00:00:00 UTC", "2026-06-05 00:00:00 UTC",
             f"TK{i}", "BUY", pnl, mfe, gb, forgone, r.get("post_exit_peak"),
             "TRAILING-STOP: retreated x%",
             _snapshot(r.get("trail", 8.0), r.get("floor", 1.0),
                       r.get("exit_timing", 0.3626))))
    conn.commit()
    conn.close()


def _make_kairos_db(path, approved=None):
    """Fresh temp kairos.db with the two tables the pipeline writes/reads."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE axis_weight_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, axis TEXT, run_id TEXT,
        computed_score REAL, sample_size INTEGER, evidence TEXT,
        prior_weight REAL, proposed_delta REAL, new_weight REAL, status TEXT,
        created_at TEXT, decided_at TEXT, decided_by TEXT)""")
    conn.execute("""CREATE TABLE axis_weights (
        axis TEXT PRIMARY KEY, weight REAL, status TEXT, positive_means TEXT,
        updated_at TEXT)""")
    conn.execute("INSERT INTO axis_weights VALUES ('exit_timing', 0.3626, 'active', '', '')")
    for a in (approved or []):
        conn.execute(
            "INSERT INTO axis_weight_history (axis, status, prior_weight, new_weight, "
            "decided_at, evidence) VALUES (?, 'approved', ?, ?, ?, ?)",
            (a["axis"], a["prior"], a["new"], a["decided_at"], a.get("evidence")))
    conn.commit()
    conn.close()


def _make_config(path, trail=8.0, floor=1.0):
    cfg = {"exits": {"trailing_stop": {"target_armed": {"trail_pct": trail},
                                       "profit_floor_pp": floor}}}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _env(tmp, ml_rows, approved=None, trail=8.0, floor=1.0):
    """Point the modules at fresh temp copies; return the config path."""
    cfg_p = os.path.join(tmp, "cfg.json")
    ml_p = os.path.join(tmp, "ml.db")
    k_p = os.path.join(tmp, "kairos.db")
    _make_config(cfg_p, trail, floor)
    _make_ml_db(ml_p, ml_rows)
    _make_kairos_db(k_p, approved)
    A.CONFIG_PATH = cfg_p
    A.ML_DB_PATH = ml_p
    kairos_log_db.DB_PATH = k_p
    kairos_log_db.init_db = lambda *a, **k: None   # tables already exist in temp
    A._post_slack = lambda *a, **k: True           # no real Slack
    return cfg_p, k_p


def _now_utc(days_ago=0):
    # Avoid Date.now-style nondeterminism concerns: derive from the module's _now.
    from datetime import datetime, timezone, timedelta
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


def main():
    tmp = tempfile.mkdtemp(prefix="kairos_selftest_")
    print(f"selftest sandbox: {tmp}\n")

    # ── 1 & 2: regime window + contributing min_sample ──────────────
    print("[regime window + min_sample]")
    rows = ([{"trail": 8.0, "mfe": 18.0, "pnl": 2.0, "forgone": 3.0}] * 12 +
            [{"trail": 4.0, "mfe": 30.0, "pnl": 1.0, "forgone": 2.0}] * 10)  # out-of-regime
    _env(tmp, rows, trail=8.0)
    r = A.compute_param(TRAIL)
    check("regime window excludes pre-change trades (sample_size==12, not 22)",
          r["sample_size"] == 12)
    check("in-regime compute is NOT gated (12 >= min_sample 10)", r["gated"] is False)

    rows_thin = ([{"trail": 8.0, "mfe": 18.0, "pnl": 2.0, "forgone": 3.0}] * 5 +
                 [{"trail": 8.0, "mfe": 18.0, "pnl": 2.0, "forgone": 3.0,
                   "contributing": False}] * 8)  # in-regime but non-contributing
    _env(tmp, rows_thin, trail=8.0)
    rg = A.compute_param(TRAIL)
    check("contributing min_sample gates (5 < 10)", rg["gated"] is True)
    check("gate_reason cites insufficient fresh evidence",
          "insufficient fresh evidence" in (rg["gate_reason"] or ""))
    check("gated proposal leaves value unchanged", rg["proposed_value"] == rg["current_value"])
    check("non-contributing rows excluded from sample (==5)", rg["sample_size"] == 5)

    # ── 3: forgone dominates -> LOOSEN ──────────────────────────────
    print("\n[two-sided: forgone dominates -> loosen]")
    rows_loose = [{"trail": 8.0, "mfe": 3.0, "pnl": 1.0, "forgone": 15.0}] * 12
    _env(tmp, rows_loose, trail=8.0)
    rl = A.compute_param(TRAIL)
    check("trail LOOSENS (proposed > current) when forgone dominates",
          rl["proposed_value"] > rl["current_value"])
    check("direction reported as 'loosen'", rl["evidence"]["direction"] == "loosen")
    rlf = A.compute_param(FLOOR)
    check("floor LOOSENS (proposed < current) when forgone dominates",
          rlf["proposed_value"] < rlf["current_value"])

    # ── 4: give-back dominates -> TIGHTEN ───────────────────────────
    print("\n[two-sided: give-back dominates -> tighten]")
    rows_tight = [{"trail": 8.0, "mfe": 18.0, "pnl": 1.0, "forgone": 2.0}] * 12
    _env(tmp, rows_tight, trail=8.0)
    rt = A.compute_param(TRAIL)
    check("trail TIGHTENS (proposed < current) when give-back dominates",
          rt["proposed_value"] < rt["current_value"])
    check("single step within ±25% of current",
          abs(rt["proposed_value"] - rt["current_value"]) <= 0.25 * rt["current_value"] + 1e-9)
    check("proposed stays within whitelist bounds [4,12]",
          4.0 <= rt["proposed_value"] <= 12.0)
    check("direction reported as 'tighten'", rt["evidence"]["direction"] == "tighten")

    # ── 5: freshness gate ───────────────────────────────────────────
    print("\n[freshness gate]")
    _env(tmp, rows_tight, trail=8.0)
    p1 = A.propose_param_update(TRAIL)
    check("first proposal writes a row", p1.get("history_id") is not None)
    p2 = A.propose_param_update(TRAIL)
    check("second identical proposal is skipped (no new evidence)",
          p2.get("history_id") is None and p2.get("skipped") == "no new evidence")
    # change the evidence (add a distinct contributing trade) -> fresh
    rows_tight2 = rows_tight + [{"trail": 8.0, "mfe": 25.0, "pnl": 1.0, "forgone": 2.0}]
    _make_ml_db(A.ML_DB_PATH, rows_tight2)
    p3 = A.propose_param_update(TRAIL)
    check("changed evidence produces a fresh proposal", p3.get("history_id") is not None)

    # ── 6: cumulative 7-day band (compute + apply) ──────────────────
    print("\n[cumulative 7-day ±40% band]")
    # Approved change 8.0 -> 6.0 within the window makes base_7d = 8.0; config now 6.0.
    approved = [{"axis": "param:" + TRAIL, "prior": 8.0, "new": 6.0,
                 "decided_at": _now_utc(1)}]
    _env(tmp, rows_tight, approved=approved, trail=6.0)
    rc = A.compute_param(TRAIL)
    band = rc["evidence"]["cumulative_band"]
    check("cumulative band anchored at base_7d 8.0", abs(band["base_7d"] - 8.0) < 1e-6)
    check("band low bound is 8.0*0.6 = 4.8", abs(band["lo"] - 4.8) < 1e-6)
    check("compute clamps proposed to >= band low (>=4.8)",
          rc["proposed_value"] >= 4.8 - 1e-9)
    # apply-side: a value that clears the 25% step but violates the band must raise.
    raised = False
    try:
        A._apply_param_to_config(TRAIL, 4.6)  # within 25% of 6.0 but < band 4.8
    except ValueError as e:
        raised = "cumulative" in str(e)
    check("apply REFUSES a value outside the cumulative band", raised)

    # ── 7 & 8: full propose -> approve cycle ────────────────────────
    print("\n[full propose -> approve cycle]")
    _env(tmp, rows_tight, trail=8.0)
    prop = A.propose_param_update(TRAIL)
    hid = prop["history_id"]
    check("cycle: proposal created", hid is not None)
    before = A._current_param_value(TRAIL)
    outcome = A.apply_decision(hid, "approve", "selftest")
    after = A._current_param_value(TRAIL)
    check("cycle: approval marked approved", outcome["status"] == "approved")
    check("cycle: config value actually changed on approve", after != before)
    check("cycle: applied value matches the proposal", abs(after - prop["new_weight"]) < 1e-6)
    # row is now approved in temp history
    kc = sqlite3.connect(kairos_log_db.DB_PATH)
    st = kc.execute("SELECT status FROM axis_weight_history WHERE id=?", (hid,)).fetchone()[0]
    kc.close()
    check("cycle: history row status == approved", st == "approved")

    # ── exit_timing regime window (2g) ──────────────────────────────
    print("\n[exit_timing regime window]")
    et_rows = ([{"trail": 8.0, "mfe": 12.0, "pnl": 2.0, "forgone": 1.0,
                 "post_exit_peak": 1.0, "exit_timing": 0.3626}] * 11 +
               [{"trail": 8.0, "mfe": 12.0, "pnl": 2.0, "forgone": 1.0,
                 "post_exit_peak": 1.0, "exit_timing": 0.99}] * 9)  # wrong regime
    _env(tmp, et_rows, trail=8.0)
    et = A.compute_exit_timing()
    check("exit_timing regime window keeps only current-weight trades (n_in_regime==11)",
          et["evidence"]["n_in_regime"] == 11)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n==== selftest: {_checks['pass']} passed, {_checks['fail']} failed "
          f"({_checks['pass'] + _checks['fail']} assertions) ====")
    return 1 if _checks["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
