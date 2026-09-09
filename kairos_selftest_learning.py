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


ATR_MULT = "exits.trailing_stop.target_armed.atr_mult"
LO_PCT = "exits.trailing_stop.target_armed.trail_lo_pct"
HI_PCT = "exits.trailing_stop.target_armed.trail_hi_pct"
ATR_ON = "exits.trailing_stop.target_armed.atr_enabled"


def _snapshot(trail=8.0, floor=1.0, exit_timing=0.3626, atr_mult=0.75,
              lo=2.0, hi=4.0, bind=None, atr_pct=None):
    """A stored snapshot. `bind` adds the armed_trail attribution block.

    bind=None models a close that predates bind-state stamping: it must route
    to NONE of the three ATR parameters.
    """
    snap = {
        "params": {TRAIL: trail, FLOOR: floor, ATR_ON: False,
                   ATR_MULT: atr_mult, LO_PCT: lo, HI_PCT: hi},
        "axis_weights": {"exit_timing": exit_timing},
        "reconstructed": True,
    }
    if bind is not None:
        snap["armed_trail"] = {
            "bind_state": bind, "atr_pct": atr_pct, "atr_mult": atr_mult,
            "trail_lo_pct": lo, "trail_hi_pct": hi, "trail_pct": trail,
            "atr_enabled": False, "shadow": True,
        }
    return json.dumps(snap)


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
                       r.get("exit_timing", 0.3626),
                       atr_mult=r.get("atr_mult", 0.75),
                       lo=r.get("lo", 2.0), hi=r.get("hi", 4.0),
                       bind=r.get("bind"), atr_pct=r.get("atr_pct"))))
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


def _make_config(path, trail=8.0, floor=1.0, atr_mult=0.75,
                 lo=2.0, hi=4.0, atr_enabled=False):
    """Temp config carrying EVERY stamped path, including the ATR family.

    The ATR keys are here deliberately: without them _get_dotted returns
    not-found, the snapshot stamps None, and _current_param_value also returns
    None — so the dual-stamp assertions would compare None to None and pass
    while testing nothing at all.
    """
    cfg = {"exits": {"trailing_stop": {
        "target_armed": {"trail_pct": trail,
                         "atr_enabled": atr_enabled,
                         "atr_mult": atr_mult,
                         "trail_lo_pct": lo,
                         "trail_hi_pct": hi},
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

    # ── 1 & 2: weighted evidence + contributing min_sample ──────────
    # The window used to be BINARY: 12 exact-match closes counted, 10 closes at
    # trail=4.0 counted for nothing. That is what made the pool resettable by
    # any config change. They are now DOWN-WEIGHTED instead: |4.0-8.0| is half
    # the [4,12] bounds range, so each is worth ~0.35 of an exact match.
    print("[weighted evidence + min_sample]")
    rows = ([{"trail": 8.0, "mfe": 18.0, "pnl": 2.0, "forgone": 3.0}] * 12 +
            [{"trail": 4.0, "mfe": 30.0, "pnl": 1.0, "forgone": 2.0}] * 10)  # distant
    _env(tmp, rows, trail=8.0)
    r = A.compute_param(TRAIL)
    ev1 = r["evidence"]
    check("distant closes are down-weighted, not discarded "
          f"(effective_n {ev1['effective_n']} between 12 and 22)",
          12.0 < ev1["effective_n"] < 22.0)
    check("sample_size reports the effective sample (16, not 12 and not 22)",
          r["sample_size"] == 16)
    check("exact-match count still reported for comparison (==12)",
          ev1["n_exact_match"] == 12)
    check("every snapshot-bearing close contributes (n_contributing==22)",
          ev1["n_contributing"] == 22)
    check("nothing is excluded for being out of regime any more",
          ev1["n_excluded_out_of_regime"] == 0
          and ev1["starvation"]["n_wrong_regime"] == 0)
    check("weight breakdown names both parameter values",
          {b["value"] for b in ev1["weighting"]["by_param_value"]} == {8.0, 4.0})
    check("exact-match bucket carries proximity 1.0",
          all(abs(b["proximity"] - 1.0) < 1e-9
              for b in ev1["weighting"]["by_param_value"] if b["exact_match"]))
    check("distant bucket sits above the proximity floor and below 1.0",
          all(A.WEIGHT_PROXIMITY_FLOOR < b["proximity"] < 1.0
              for b in ev1["weighting"]["by_param_value"]
              if not b["exact_match"]))
    check("weighted compute is NOT gated (effective 15.5 >= min_sample 10)",
          r["gated"] is False)
    check("confidence recorded and strictly below full strength",
          0.0 < ev1["confidence"] < 1.0)
    # Determinism matters as much as the weights: the recency anchor is the
    # newest CONTRIBUTING close, not wall-clock now(), so an unchanged corpus
    # must produce a byte-identical proposal (this is what keeps the evidence
    # hash / freshness gate load-bearing).
    r_again = A.compute_param(TRAIL)
    check("compute is deterministic on an unchanged corpus",
          r_again["proposed_value"] == r["proposed_value"]
          and r_again["evidence"]["evidence_hash"] == ev1["evidence_hash"])

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
    check("non-contributing rows are reported as such (==8)",
          rg["evidence"]["n_excluded_non_contributing"] == 8)

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

    # ── 5b: delta magnitude scales with the EFFECTIVE sample ────────
    # The anti-ratchet half of the weighting trade. Two corpora with IDENTICAL
    # per-trade evidence (same mfe / pnl / forgone, so the same mean_error and
    # the same severity 1.0) and the SAME row count, differing only in how far
    # the parameter sat from today's value. The distant one must move the
    # parameter LESS, because its effective sample — and therefore its
    # confidence factor — is smaller. Without this, loosening the filter would
    # just be a larger step on weaker evidence.
    print("\n[delta magnitude scales with effective_n]")
    same_evidence = {"mfe": 18.0, "pnl": 1.0, "forgone": 2.0}
    matched_rows = [dict(same_evidence, trail=8.0)] * 40      # exact match
    distant_rows = [dict(same_evidence, trail=4.0)] * 40      # far end of bounds

    _env(tmp, matched_rows, trail=8.0)
    r_near = A.compute_param(TRAIL)
    _env(tmp, distant_rows, trail=8.0)
    r_far = A.compute_param(TRAIL)

    d_near = abs(r_near["proposed_value"] - r_near["current_value"])
    d_far = abs(r_far["proposed_value"] - r_far["current_value"])
    check("both cases clear the gate (so the comparison is about size, "
          "not gating)",
          r_near["gated"] is False and r_far["gated"] is False)
    check("both cases measure the SAME bias (identical computed_score)",
          abs(r_near["computed_score"] - r_far["computed_score"]) < 1e-9)
    check(f"well-matched evidence has the larger effective sample "
          f"({r_near['evidence']['effective_n']} > "
          f"{r_far['evidence']['effective_n']})",
          r_near["evidence"]["effective_n"] > r_far["evidence"]["effective_n"])
    check(f"mostly-distant evidence produces the SMALLER delta "
          f"({d_far:.4f} < {d_near:.4f})", d_far < d_near)
    check("confidence is what shrinks it, and is strictly < 1 in both cases",
          r_far["evidence"]["confidence"] < r_near["evidence"]["confidence"] < 1.0)
    check("a step never exceeds the ±25% per-run cap in either case",
          d_near <= 0.25 * r_near["current_value"] + 1e-9
          and d_far <= 0.25 * r_far["current_value"] + 1e-9)
    # Same property on the AXIS path, through compute_delta directly.
    cfg_ax = {"learning_rate": 0.3, "per_run_cap": 0.15, "min_sample": 20,
              "weight_bound": 1.0}
    d_hi = A.compute_delta(0.8, 0.0, 60, cfg_ax, effective_n=60.0)
    d_lo = A.compute_delta(0.8, 0.0, 60, cfg_ax, effective_n=22.0)
    check("axis path: same score, smaller effective_n -> smaller delta",
          abs(d_lo["proposed_delta"]) < abs(d_hi["proposed_delta"]))
    check("axis path: effective_n below min_sample still gates outright",
          A.compute_delta(0.8, 0.0, 60, cfg_ax, effective_n=19.9)["gated"] is True)
    check("axis path: confidence is exactly 0.5 at the gate threshold",
          abs(A.compute_delta(0.8, 0.0, 20, cfg_ax,
                              effective_n=20.0)["confidence"] - 0.5) < 1e-9)
    check("axis path: per_run_cap still binds regardless of confidence",
          abs(A.compute_delta(1.0, -1.0, 999, cfg_ax,
                              effective_n=999.0)["proposed_delta"])
          <= cfg_ax["per_run_cap"] + 1e-12)

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

    # ── propose_all_params is re-runnable ───────────────────────────
    # The freshness skip and the normal path return DIFFERENT dicts unless they
    # share a builder. propose_all_params projects a fixed key set out of both,
    # so a missing key raised KeyError inside the per-param try/except and was
    # reported to Slack as an ERROR — a healthy "nothing new to say" was
    # indistinguishable from a broken param loop. Running twice is the test:
    # the second run is skip-only by construction.
    print("\n[propose_all_params is re-runnable]")
    _env(tmp, rows_tight, trail=8.0)
    r1 = A.propose_all_params(run_id="selftest_1")
    check("first propose_all_params run has no errors", r1["errors"] == [])
    check("first run writes rows", any(p["history_id"] is not None
                                       for p in r1["proposals"]))
    r2 = A.propose_all_params(run_id="selftest_2")
    check("SECOND propose_all_params run produces NO errors", r2["errors"] == [])
    check("second run is all skips (no rows written)",
          all(p["history_id"] is None for p in r2["proposals"]))
    check("skipped proposals report skipped='no new evidence'",
          all(p["skipped"] == "no new evidence" for p in r2["proposals"]))
    check("skipped proposals still carry the full summary key set",
          all(all(k in p for k in A.PARAM_PROPOSAL_SUMMARY_KEYS)
              for p in r2["proposals"]))
    check("skipped proposal reports prior_weight == current value",
          all(abs(p["prior_weight"] - A._current_param_value(p["path"])) < 1e-9
              for p in r2["proposals"]))
    check("skipped proposal reports zero delta and new_weight == prior",
          all(p["proposed_delta"] == 0.0 and p["new_weight"] == p["prior_weight"]
              for p in r2["proposals"]))
    check("Slack text renders skips distinctly (not as an error/gate)",
          "skipped — no new evidence" in r2["slack_text"]
          and "error —" not in r2["slack_text"])

    # ── snapshot capture (regime tagging at close) ───────────────────
    # A close that is not stamped can never be evidence for ANY axis, and the
    # snapshot is unrecoverable after the config moves on — so the write path,
    # not a backfill, has to be the thing that guarantees it.
    print("\n[exit_params_snapshot capture]")
    import kairos_ml_outcomes as M
    # Was set-EQUALITY. Relaxed 2026-09-09 to the invariant that actually
    # carries the failure mode, because dual-stamping deliberately breaks
    # equality: the ATR paths are stamped while atr_enabled is false so that
    # atr_mult has an evidence pool on the day the behaviour flips, and
    # atr_enabled is a BOOLEAN, which _snapshot_value rejects and which must
    # therefore never be whitelisted as learnable.
    #
    # The dangerous direction is one-way: a proposable path that is NOT stamped
    # is invisible to every regime window forever, silently. The reverse —
    # stamped but not proposable — is just recorded context. So: assert the
    # subset, and assert that every stamped-only path is one we MEANT to add,
    # so a typo in a new path still fails here instead of being waved through.
    # Shrank from four to one when Part 4 made the three ATR params learnable.
    # atr_enabled stays stamp-only forever: it is a BOOLEAN, _snapshot_value
    # rejects it, so it can never serve as a regime key and must never be
    # whitelisted — it is stamped purely so a close records whether it ran
    # under the ATR trail or the fallback.
    STAMP_ONLY = {
        "exits.trailing_stop.target_armed.atr_enabled",
    }
    check("every proposable path is stamped (the silent-starvation direction)",
          set(A.PARAM_WHITELIST) <= set(M.SNAPSHOT_PARAM_PATHS))
    check("every stamped-but-not-proposable path is deliberate (catches typos)",
          set(M.SNAPSHOT_PARAM_PATHS) - set(A.PARAM_WHITELIST) == STAMP_ONLY)
    check("atr_enabled is stamped but NOT learnable (boolean, no regime key)",
          "exits.trailing_stop.target_armed.atr_enabled"
          in M.SNAPSHOT_PARAM_PATHS
          and "exits.trailing_stop.target_armed.atr_enabled"
          not in A.PARAM_WHITELIST)
    check("exit_params_snapshot is schema-managed",
          "exit_params_snapshot" in dict(M._TRADE_OUTCOMES_EXTRA_COLUMNS))

    ml_live = os.path.join(tmp, "ml_live.db")
    if os.path.exists(ml_live):
        os.remove(ml_live)
    old_db, old_cfg, old_k = M.DB_PATH, M.CONFIG_PATH, M.KAIROS_DB_PATH
    try:
        M.DB_PATH, M.CONFIG_PATH, M.KAIROS_DB_PATH = ml_live, A.CONFIG_PATH, kairos_log_db.DB_PATH
        M.init_db()
        tid = M.write_trade_open("SELFTEST", "BUY", 10, 100.0)
        M.write_trade_close(tid, 110.0)
        conn = M.get_connection()
        raw = conn.execute("SELECT exit_params_snapshot FROM trade_outcomes "
                           "WHERE trade_id = ?", (tid,)).fetchone()[0]
        conn.close()
        check("write_trade_close stamps a snapshot", raw is not None)
        snap = json.loads(raw or "{}")
        check("live snapshot key order matches the stored corpus",
              list(snap.keys()) == ["params", "axis_weights", "reconstructed",
                                    "trailing_stop", "captured_at"])
        check("live snapshot is flagged reconstructed=False",
              snap.get("reconstructed") is False)
        # Compare against the CONFIG, not _current_param_value: that helper
        # nulls booleans by design, so routing atr_enabled through it would
        # compare None to None and assert nothing.
        with open(A.CONFIG_PATH) as _f:
            _live_cfg = json.load(_f)
        check("snapshot records the live value at every stamped path",
              all(snap["params"].get(p) == A._get_dotted(_live_cfg, p)[0]
                  for p in M.SNAPSHOT_PARAM_PATHS))
        check("DUAL-STAMP: a real close stamps all four ATR paths alongside "
              "the deprecated trail_pct",
              all(p in snap["params"] for p in (
                  "exits.trailing_stop.target_armed.trail_pct",
                  "exits.trailing_stop.target_armed.atr_enabled",
                  "exits.trailing_stop.target_armed.atr_mult",
                  "exits.trailing_stop.target_armed.trail_lo_pct",
                  "exits.trailing_stop.target_armed.trail_hi_pct")))
        check("DUAL-STAMP: the ATR values are stamped while behaviour is OFF",
              snap["params"]["exits.trailing_stop.target_armed.atr_enabled"]
              is False
              and snap["params"]["exits.trailing_stop.target_armed.atr_mult"]
              == 0.75)
        check("atr_mult is usable as a regime key (numeric, _in_regime reads it)",
              A._in_regime(raw, "params",
                           "exits.trailing_stop.target_armed.atr_mult", 0.75))
        check("atr_enabled is NOT usable as a regime key (boolean -> None)",
              A._snapshot_value(
                  raw, "params",
                  "exits.trailing_stop.target_armed.atr_enabled") is None)
        check("_in_regime accepts a freshly written snapshot",
              A._in_regime(raw, "params", TRAIL, A._current_param_value(TRAIL)))
        # A reconstruction must stay comparable to a live capture: same two
        # sections, read by the same _in_regime, no trailing_stop block.
        rec = M.build_exit_params_snapshot(
            reconstructed=True,
            param_overrides={TRAIL: 8.0, FLOOR: 1.0},
            weight_overrides={"exit_timing": 0.3626},
            as_of="2026-05-27 13:37:55 UTC")
        check("reconstructed snapshot omits the trailing_stop block",
              list(rec.keys()) == ["params", "axis_weights", "reconstructed",
                                   "captured_at"])
        check("reconstructed snapshot preserves as_of as captured_at",
              rec["captured_at"] == "2026-05-27 13:37:55 UTC")
        check("_in_regime reads both shapes identically",
              A._in_regime(json.dumps(rec), "params", TRAIL, 8.0)
              and not A._in_regime(json.dumps(rec), "params", TRAIL, 6.0))
    finally:
        M.DB_PATH, M.CONFIG_PATH, M.KAIROS_DB_PATH = old_db, old_cfg, old_k

    # ── role sign is explicit per full path, never by suffix ────────
    # The old test was `path.endswith("trail_pct")`, which gives atr_mult
    # role_sign +1.0. atr_mult TIGHTENS by decreasing, exactly like trail_pct,
    # so +1.0 inverts every proposal it makes: evidence saying "give back too
    # much, tighten" would have proposed a LARGER multiplier. Silent, and in
    # the direction that compounds give-back.
    print("\n[role sign — explicit per path]")
    ATR_MULT = "exits.trailing_stop.target_armed.atr_mult"
    LO = "exits.trailing_stop.target_armed.trail_lo_pct"
    HI = "exits.trailing_stop.target_armed.trail_hi_pct"
    check("atr_mult and trail_pct share role_sign -1.0 (both tighten by "
          "DECREASING)",
          A._role_sign(ATR_MULT) == -1.0 and A._role_sign(TRAIL) == -1.0)
    check("both clamp bounds also tighten by decreasing (role_sign -1.0)",
          A._role_sign(LO) == -1.0 and A._role_sign(HI) == -1.0)
    check("profit_floor_pp tightens by INCREASING (role_sign +1.0)",
          A._role_sign(FLOOR) == +1.0)
    check("the old suffix test would have got atr_mult WRONG "
          "(regression guard)",
          (-1.0 if ATR_MULT.endswith("trail_pct") else +1.0)
          != A._role_sign(ATR_MULT))
    raised = False
    try:
        A._role_sign("exits.trailing_stop.some_new_knob")
    except ValueError:
        raised = True
    check("an unmapped path RAISES rather than defaulting to a guessed sign",
          raised)
    check("every whitelisted param has an explicit role sign",
          all(p in A.PARAM_ROLE_SIGN for p in A.PARAM_WHITELIST))

    # ── starvation diagnostics (sample=0 must say why) ──────────────
    print("\n[starvation diagnostics]")
    # Every row out of regime -> sample 0. The message must name the cause, not
    # just the count, or a starved loop is indistinguishable from a young one.
    starved = [{"trail": 4.0, "mfe": 18.0, "pnl": 2.0, "forgone": 3.0}] * 12
    _env(tmp, starved, trail=8.0)
    rs = A.compute_param(TRAIL)
    st = rs["evidence"]["starvation"]
    # Distant-ONLY evidence no longer reads as an empty sample — but 12 closes
    # at ~0.35 weight is still only ~4.2 effective, so it still GATES. That is
    # the property the redesign has to keep: weighting is not a bypass.
    check("distant-only evidence still gates (effective ~4.2 < 10)",
          rs["gated"] is True and rs["sample_size"] < A.PARAM_MIN_SAMPLE)
    check("distant-only evidence is no longer reported as sample 0",
          rs["sample_size"] > 0)
    check("starvation reports down-weighted rows, not discarded ones",
          st["n_downweighted"] == 12 and st["n_wrong_regime"] == 0)
    check("gate_reason names the weighted basis",
          "weighted by recency" in (rs["gate_reason"] or ""))
    # Unstamped rows are the unrecoverable case and must be counted separately.
    conn = sqlite3.connect(A.ML_DB_PATH)
    conn.execute("UPDATE trade_outcomes SET exit_params_snapshot = NULL")
    conn.commit()
    conn.close()
    rn = A.compute_param(TRAIL)
    check("rows with no snapshot are counted as no_snapshot, not wrong_regime",
          rn["evidence"]["starvation"]["n_no_snapshot"] == 12
          and rn["evidence"]["starvation"]["n_wrong_regime"] == 0)
    check("rows with no snapshot are still EXCLUDED, not weighted "
          "(effective sample 0)",
          rn["sample_size"] == 0 and rn["evidence"]["effective_n"] == 0.0
          and rn["gated"] is True)
    check("summary names the missing-snapshot cause",
          "lack a regime snapshot" in rn["evidence"]["starvation"]["summary"])
    check("Slack note flags an empty sample with a reason",
          "sample is 0 — " in A._format_slack_propose_params(
              "r", [{"path": TRAIL, "sample_size": 0, "gated": True,
                     "skipped": None, "prior_weight": 8.0, "new_weight": 8.0,
                     "proposed_delta": 0.0, "computed_score": 0.0,
                     "history_id": None, "evidence": rn["evidence"]}], []))
    # The loud case: empty across MULTIPLE runs while trades were closing.
    check("repeated empty samples across real closes escalate to :rotating_light:",
          ":rotating_light:" in A._format_slack_propose_params(
              "r", [{"path": TRAIL, "sample_size": 0, "gated": True,
                     "skipped": None, "prior_weight": 8.0, "new_weight": 8.0,
                     "proposed_delta": 0.0, "computed_score": 0.0,
                     "history_id": None,
                     "evidence": {"starvation": {
                         "summary": "x",
                         "zero_sample_streak": {"runs": 4, "since": "2026-07-29",
                                                "closes_since": 55}}}}], []))
    check("a single empty run stays a :warning:, not an alarm",
          ":rotating_light:" not in A._format_slack_propose_params(
              "r", [{"path": TRAIL, "sample_size": 0, "gated": True,
                     "skipped": None, "prior_weight": 8.0, "new_weight": 8.0,
                     "proposed_delta": 0.0, "computed_score": 0.0,
                     "history_id": None,
                     "evidence": {"starvation": {
                         "summary": "x",
                         "zero_sample_streak": {"runs": 1, "since": "2026-07-29",
                                                "closes_since": 55}}}}], []))

    # ── Part 4(c): evidence routing by clamp-bind state ─────────────
    # A close is evidence about the ONE parameter that governed its trail. On
    # the live book 30 of 49 positions are clamp-bound, so pooling clamped
    # closes into atr_mult would mean the multiplier is mostly learned from
    # trades where it had no effect at all.
    print("\n[bind-state evidence routing]")
    E = {"mfe": 18.0, "pnl": 2.0, "forgone": 3.0}
    mixed = ([dict(E, bind="free")] * 14 +
             [dict(E, bind="floor")] * 11 +
             [dict(E, bind="ceiling")] * 7 +
             [dict(E, bind="no_atr")] * 5 +
             [dict(E)] * 9)              # bind=None -> unattributed
    _env(tmp, mixed, trail=8.0)
    rm = A.compute_param(ATR_MULT)
    rlo = A.compute_param(LO_PCT)
    rhi = A.compute_param(HI_PCT)
    rtp = A.compute_param(TRAIL)
    rt = rm["evidence"]["routing"]
    check("routing counts every bind state present in the corpus",
          rt["by_bind_state"] == {"free": 14, "floor": 11, "ceiling": 7,
                                  "no_atr": 5, "unattributed": 9})
    check("atr_mult learns ONLY from free closes (14 of 46)",
          rt["routed"] and rt["governs_bind_state"] == "free"
          and rt["n_after_routing"] == 14)
    check("trail_lo_pct learns ONLY from floor-bound closes (11)",
          rlo["evidence"]["routing"]["n_after_routing"] == 11)
    check("trail_hi_pct learns ONLY from ceiling-bound closes (7)",
          rhi["evidence"]["routing"]["n_after_routing"] == 7)
    check("NO clamp-bound close reaches atr_mult's pool",
          rm["evidence"]["n_contributing"] == 14)
    check("no_atr and unattributed closes reach NONE of the three",
          rm["evidence"]["n_contributing"]
          + rlo["evidence"]["n_contributing"]
          + rhi["evidence"]["n_contributing"] == 32)
    check("the three pools are disjoint and sum to the measured closes only",
          14 + 11 + 7 == 32)
    check("trail_pct is UNROUTED — every close is eligible, as before",
          rtp["evidence"]["routing"]["routed"] is False
          and rtp["evidence"]["routing"]["n_after_routing"] == 46)
    check("profit_floor_pp is UNROUTED (separate mechanism, not the family)",
          A.compute_param(FLOOR)["evidence"]["routing"]["routed"] is False)
    check("a routed parameter says WHY its pool is small, not that it is broken",
          "different clamp state" in
          (rhi["evidence"]["starvation"]["summary"] or ""))
    check("routing is reported for the card (Part 4d)",
          "free" in rt["by_bind_state"] and rt["n_before_routing"] == 46)
    # A corpus with no attribution at all must starve the three, not guess.
    _env(tmp, [dict(E)] * 30, trail=8.0)
    check("a corpus with NO bind attribution gates all three ATR params",
          all(A.compute_param(q)["gated"] for q in (ATR_MULT, LO_PCT, HI_PCT)))
    check("...while trail_pct still computes from the same corpus",
          A.compute_param(TRAIL)["gated"] is False)

    # ── Part 4(b): ordered bounds cannot cross ──────────────────────
    print("\n[ordered bounds cannot cross]")

    def _set_bounds(lo_v, hi_v):
        """Rewrite lo/hi in the temp config (both are read live by the guard)."""
        with open(A.CONFIG_PATH) as f:
            c = json.load(f)
        c["exits"]["trailing_stop"]["target_armed"]["trail_lo_pct"] = lo_v
        c["exits"]["trailing_stop"]["target_armed"]["trail_hi_pct"] = hi_v
        with open(A.CONFIG_PATH, "w") as f:
            json.dump(c, f, indent=2)

    # Pure logic first, independent of any evidence.
    _env(tmp, [dict(mfe=18.0, pnl=2.0, forgone=3.0, bind="floor",
                    lo=3.0, hi=3.2)] * 14, trail=8.0)
    _set_bounds(3.0, 3.2)
    check("_ordering_violation flags a floor above the ceiling",
          A._ordering_violation(LO_PCT, 3.5) is not None)
    check("_ordering_violation flags a ceiling below the floor",
          A._ordering_violation(HI_PCT, 2.9) is not None)
    check("_ordering_violation passes a non-crossing value",
          A._ordering_violation(LO_PCT, 3.1) is None
          and A._ordering_violation(HI_PCT, 3.3) is None)
    check("unpaired params are never subject to the ordering guard",
          A._ordering_violation(TRAIL, 99.0) is None
          and A._ordering_violation(FLOOR, 99.0) is None)

    # Compute side: floor evidence saying LOOSEN pushes trail_lo_pct up past
    # a nearby ceiling. It must GATE to a human, not silently clamp — clamping
    # would apply a value nobody proposed and would hide the real signal,
    # which is that the two pools disagree about where the band belongs.
    # The snapshot bounds must MATCH the config, or proximity weighting
    # discounts the pool below min_sample and the sample gate fires first —
    # which would make this test pass for the wrong reason.
    loose = [dict(mfe=3.0, pnl=1.0, forgone=15.0, bind="floor",
                  lo=3.0, hi=3.2)] * 14
    _env(tmp, loose, trail=8.0)
    _set_bounds(3.0, 3.2)
    rx = A.compute_param(LO_PCT)
    check(f"a crossing proposal is GATED, not clamped "
          f"(lo 3.0 -> would be >3.2)",
          rx["gated"] is True
          and "cross the ordered pair" in (rx["gate_reason"] or ""))
    check("the gated crossing leaves the value untouched",
          rx["proposed_value"] == rx["current_value"] == 3.0)
    check("the gate reason names both bounds and says what to do",
          "trail_hi_pct" in (rx["gate_reason"] or "")
          and "Gated to a human" in (rx["gate_reason"] or ""))

    # Apply side: the guard that actually holds, because a proposal can sit
    # pending while the OTHER bound moves underneath it.
    raised = False
    try:
        A._apply_param_to_config(LO_PCT, 3.5)   # inside [1,4], but > hi 3.2
    except ValueError as e:
        raised = "cross the ordered pair" in str(e)
    check("APPLY refuses a floor above the ceiling (defence in depth)", raised)
    _set_bounds(3.5, 4.0)
    raised2 = False
    try:
        A._apply_param_to_config(HI_PCT, 3.4)   # inside [3,10], but < lo 3.5
    except ValueError as e:
        raised2 = "cross the ordered pair" in str(e)
    check("APPLY refuses a ceiling below the floor", raised2)
    _set_bounds(2.0, 4.0)
    applied_ok = False
    try:
        A._apply_param_to_config(LO_PCT, 2.4)   # non-crossing, in bounds
        applied_ok = abs(A._current_param_value(LO_PCT) - 2.4) < 1e-9
    except ValueError:
        applied_ok = False
    check("a NON-crossing value on the same pair still applies normally",
          applied_ok)

    # ── ATR arm context: a FAILED measurement must not be frozen ────
    # Found live on 2026-09-09: a yfinance rate-limit burst made 14 held
    # positions arm as no_atr, and an unconditional store-reuse would have
    # pinned "ATR unavailable" on all of them for the rest of their lives —
    # sending them to trail_pct's pool forever instead of the ATR family's.
    print("\n[ATR arm context: transient failures must not freeze]")
    import kairos_atr_trail as AT
    _k = os.path.join(tmp, "armctx.db")
    if os.path.exists(_k):
        os.remove(_k)
    kairos_log_db.DB_PATH = _k
    _orig_init = kairos_log_db.init_db
    kairos_log_db.init_db = _orig_init          # real init: we need the table
    import sqlite3 as _sq
    _c = _sq.connect(_k)
    _c.execute("""CREATE TABLE armed_trail_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL,
        armed_at TEXT NOT NULL, atr_pct REAL, raw_trail_pct REAL,
        trail_pct REAL, bind_state TEXT, atr_mult REAL, trail_lo_pct REAL,
        trail_hi_pct REAL, fallback_trail_pct REAL,
        atr_enabled INTEGER NOT NULL DEFAULT 0, peak_gain_pct REAL,
        target_pct REAL, created_at TEXT)""")
    _c.commit(); _c.close()
    _p = {"atr_enabled": False, "atr_mult": 0.75, "trail_lo_pct": 2.0,
          "trail_hi_pct": 4.0, "trail_pct": 6.2255}
    _cfg = {"trailing_stop": {"target_armed": {
        "atr_enabled": False, "atr_mult": 0.75, "trail_lo_pct": 2.0,
        "trail_hi_pct": 4.0, "trail_pct": 6.2255}}}

    _orig_atr = AT.atr14_pct
    try:
        # Cycle 1: the fetch fails.
        AT.atr14_pct = lambda tk, window=14: None
        c1 = AT.arm_context("ZZTOP", cfg=_cfg)
        n1 = _sq.connect(_k).execute(
            "SELECT COUNT(*) FROM armed_trail_context").fetchone()[0]
        check("a failed ATR fetch records no_atr, not a clamped default",
              c1["bind_state"] == "no_atr" and c1["raw_trail_pct"] is None
              and c1["trail_pct"] == 6.2255)
        check("the failure writes exactly ONE audit row", n1 == 1)
        # Cycle 2: still failing — must NOT append another row per cycle.
        AT.arm_context("ZZTOP", cfg=_cfg)
        AT.arm_context("ZZTOP", cfg=_cfg)
        n2 = _sq.connect(_k).execute(
            "SELECT COUNT(*) FROM armed_trail_context").fetchone()[0]
        check("a sustained outage does NOT append a row per cycle", n2 == 1)
        # Cycle 3: yfinance recovers. The failure must be superseded.
        AT.atr14_pct = lambda tk, window=14: 4.0      # -> raw 3.0 -> free
        c3 = AT.arm_context("ZZTOP", cfg=_cfg)
        check("once ATR is measurable the arm is RE-MEASURED, not left no_atr",
              c3["bind_state"] == "free" and c3["atr_pct"] == 4.0
              and c3["from_store"] is False)
        n3 = _sq.connect(_k).execute(
            "SELECT COUNT(*) FROM armed_trail_context").fetchone()[0]
        check("the real measurement is recorded alongside the failure "
              "(append-only audit trail)", n3 == 2)
        # Cycle 4+: a SUCCESSFUL measurement IS frozen — arm-time semantics.
        AT.atr14_pct = lambda tk, window=14: 12.0     # would be ceiling-bound
        c4 = AT.arm_context("ZZTOP", cfg=_cfg)
        check("a successful measurement IS frozen — the trail cannot drift "
              "under a live position as volatility changes",
              c4["from_store"] is True and c4["atr_pct"] == 4.0
              and c4["bind_state"] == "free")
        n4 = _sq.connect(_k).execute(
            "SELECT COUNT(*) FROM armed_trail_context").fetchone()[0]
        check("reuse writes nothing further", n4 == 2)
        check("latest_arm(measured_only=True) skips the no_atr row",
              AT.latest_arm("ZZTOP", measured_only=True)["bind_state"] == "free")
        check("latest_arm() without the filter still sees the newest row",
              AT.latest_arm("ZZTOP") is not None)
        # persist=False must never write (the reporting/dry-run path).
        AT.atr14_pct = lambda tk, window=14: 3.0
        AT.arm_context("NEVERWRITE", cfg=_cfg, persist=False)
        n5 = _sq.connect(_k).execute(
            "SELECT COUNT(*) FROM armed_trail_context WHERE ticker='NEVERWRITE'"
        ).fetchone()[0]
        check("persist=False measures without writing (inspecting the book "
              "must not create arm records)", n5 == 0)
        # `since` must stop a re-entered ticker inheriting an old arm.
        check("`since` prevents inheriting a previous position's arm context",
              AT.latest_arm("ZZTOP", since="2099-01-01T00:00:00Z") is None)
    finally:
        AT.atr14_pct = _orig_atr

    # ── Part 7: one coupled parameter per run ───────────────────────
    # All three qualify simultaneously. Exactly one may carry a live delta;
    # the other two are DEFERRED (not gated — the evidence was sufficient) and
    # recomputed next run against the routing that then exists.
    print("\n[Part 7: coupled family — one change per run]")
    # Different pool SIZES so the winner is unambiguous: free 20 > floor 15 >
    # ceiling 12. All say "tighten" (give-back dominates forgone).
    T = dict(mfe=18.0, pnl=1.0, forgone=2.0)
    allthree = ([dict(T, bind="free", lo=2.0, hi=4.0)] * 20 +
                [dict(T, bind="floor", lo=2.0, hi=4.0)] * 15 +
                [dict(T, bind="ceiling", lo=2.0, hi=4.0)] * 12)
    _env(tmp, allthree, trail=8.0)
    r1 = A.propose_all_params(run_id="part7_run1")
    cd = r1["coupled_decision"]
    by = {p["path"]: p for p in r1["proposals"]}
    fam = [by[q] for q in A.PARAM_COUPLED_FAMILY]
    live = [p for p in fam if abs(p["proposed_delta"]) > 1e-9]
    defd = [p for p in fam if p.get("deferred")]

    check("all three coupled params qualified this run (3 contenders)",
          cd["n_contenders"] == 3)
    check("EXACTLY ONE coupled param carries a live delta",
          len(live) == 1)
    check("the other TWO are deferred", len(defd) == 2)
    check("the winner is the highest-effective_n pool (free, n=20 -> atr_mult)",
          cd["winner"] == ATR_MULT and live[0]["path"] == ATR_MULT)
    check("deferred rows carry ZERO delta, so approving one cannot move it",
          all(p["proposed_delta"] == 0.0
              and p["new_weight"] == p["prior_weight"] for p in defd))
    check("the deferral reason names the coupling, not a gate",
          all("coupled parameter changed this run" in (p["deferred"] or "")
              or "changes which trades this parameter governs"
              in (p["deferred"] or "") for p in defd))
    check("deferred rows are NOT marked gated (the evidence was sufficient)",
          all(p["gated"] is False for p in defd))
    check("deferred rows are NOT marked skipped (there WAS something to say)",
          all(p["skipped"] is None for p in defd))
    check("each deferral records the move it would have made",
          all((p.get("deferral") or {}).get("would_have_been") is not None
              and (p.get("deferral") or {}).get("would_have_moved")
              for p in defd))
    check("each deferral records which sibling went first",
          all((p.get("deferral") or {}).get("applied_instead") == ATR_MULT
              for p in defd))
    check("deferred rows are still WRITTEN as 'proposed' (auditable)",
          all(p["history_id"] is not None for p in defd))
    check("the Slack card states the deferral, not a gate failure",
          "deferred this run" in r1["slack_text"]
          and "not\na gate failure" in r1["slack_text"].replace(" ", "\n")
          or "not a gate failure" in r1["slack_text"])
    check("the card explains the one-change-per-run rule",
          "one change per run" in r1["slack_text"])
    check("profit_floor_pp is NOT in the coupled family and may still move",
          FLOOR not in A.PARAM_COUPLED_FAMILY
          and A.PARAM_COUPLED_FAMILY == (ATR_MULT, LO_PCT, HI_PCT))
    check("trail_pct is NOT in the coupled family either",
          TRAIL not in A.PARAM_COUPLED_FAMILY)

    # Next run, AFTER the winner is applied: routing changes, so the deferred
    # params must be RECOMPUTED rather than carried forward on a stale premise.
    winner_row = live[0]
    A.apply_decision(winner_row["history_id"], "approve", "selftest")
    applied_mult = A._current_param_value(ATR_MULT)
    check("the winner applied and moved the config",
          abs(applied_mult - winner_row["new_weight"]) < 1e-9
          and applied_mult != winner_row["prior_weight"])
    r2 = A.propose_all_params(run_id="part7_run2")
    by2 = {p["path"]: p for p in r2["proposals"]}
    defd_now = [by2[p["path"]] for p in defd]
    check("the previously-deferred params are RECOMPUTED next run "
          "(fresh rows, not the stale ones)",
          all(q["history_id"] is not None
              and q["history_id"] != by[q["path"]]["history_id"]
              for q in defd_now))
    check("a deferred move is PROMOTED next run, not silently discarded "
          "(the freshness gate must not read a zero-delta deferral as an "
          "equivalent proposal)",
          any(abs(q["proposed_delta"]) > 1e-9 for q in defd_now))
    check("their recomputation is routed to their own bind-state subset",
          all((q.get("evidence") or {}).get("routing", {}).get("routed")
              for q in defd_now))
    check("no stale delta was carried forward from run 1",
          all(q.get("deferral") is None
              or q["deferral"]["from_value"] == q["prior_weight"]
              for q in defd_now))
    check("run 2 again admits at most ONE live coupled delta",
          len([q for q in (by2[x] for x in A.PARAM_COUPLED_FAMILY)
               if abs(q["proposed_delta"]) > 1e-9]) <= 1)

    # A single contender must NOT be deferred — the rule is a cap, not a queue.
    _env(tmp, [dict(T, bind="free", lo=2.0, hi=4.0)] * 20, trail=8.0)
    r3 = A.propose_all_params(run_id="part7_run3")
    by3 = {p["path"]: p for p in r3["proposals"]}
    check("with ONE contender nothing is deferred and it carries its delta",
          r3["coupled_decision"]["n_contenders"] == 1
          and by3[ATR_MULT].get("deferred") is None
          and abs(by3[ATR_MULT]["proposed_delta"]) > 1e-9)
    check("the family footer is absent when there is nothing to defer",
          "one change per run" not in r3["slack_text"])

    # ── exit_timing regime window (2g) ──────────────────────────────
    print("\n[exit_timing regime window]")
    et_rows = ([{"trail": 8.0, "mfe": 12.0, "pnl": 2.0, "forgone": 1.0,
                 "post_exit_peak": 1.0, "exit_timing": 0.3626}] * 11 +
               [{"trail": 8.0, "mfe": 12.0, "pnl": 2.0, "forgone": 1.0,
                 "post_exit_peak": 1.0, "exit_timing": 0.99}] * 9)  # wrong regime
    _env(tmp, et_rows, trail=8.0)
    et = A.compute_exit_timing()
    ete = et["evidence"]
    check("exit_timing weights ALL matured closes, not just current-weight "
          "(n_matured==20)", ete["n_matured"] == 20)
    check("exit_timing reports the exact-match count separately (==11)",
          ete["n_exact_match"] == 11 and ete["n_downweighted"] == 9)
    check("exit_timing effective sample sits between the exact-match count "
          f"and the full corpus ({ete['effective_n']})",
          11.0 < ete["effective_n"] < 20.0)
    check("exit_timing discards nothing for being out of regime",
          ete["n_excluded_out_of_regime"] == 0)

    # ── snapshot coverage: every proposable path is recorded ────────
    # A path the loop can propose on but write_trade_close does not stamp has no
    # value in any snapshot, so _in_regime is False for every trade and the param
    # sits gated at n=0 forever — failing closed, but indistinguishably from a
    # regime that simply has not filled in yet.
    print("\n[snapshot coverage]")
    import kairos_ml_outcomes as M
    check("every PARAM_WHITELIST path is recorded in SNAPSHOT_PARAM_PATHS",
          set(A.PARAM_WHITELIST) <= set(M.SNAPSHOT_PARAM_PATHS))
    live = M.build_exit_params_snapshot()
    check("a live snapshot carries every proposable path",
          all(p in live["params"] for p in A.PARAM_WHITELIST))
    check("a live snapshot is flagged as not reconstructed",
          live["reconstructed"] is False)

    # ── propose_all_params survives a freshness SKIP ────────────────
    # The skip return used to omit the keys propose_all_params projects, so every
    # skip raised KeyError and was reported to Slack as a param-loop FAILURE.
    print("\n[skip does not read as an error]")
    _env(tmp, rows_tight, trail=8.0)
    _frozen = A.PROPOSALS_FROZEN
    A.PROPOSALS_FROZEN = False          # in-process only; the file is untouched
    try:
        A.propose_all_params(run_id="selftest_skip_1")        # writes rows
        skip = A.propose_all_params(run_id="selftest_skip_2")  # identical evidence
    finally:
        A.PROPOSALS_FROZEN = _frozen
    check("a skipped run reports no errors", not skip["errors"])
    check("a skipped run still returns one summary per param",
          len(skip["proposals"]) == len(A.PARAM_WHITELIST))
    check("skipped summaries carry every projected key",
          all(k in p for p in skip["proposals"]
              for k in A.PARAM_PROPOSAL_SUMMARY_KEYS))
    check("Slack renders the skip as skipped, not as an error",
          "skipped — no new evidence" in skip["slack_text"]
          and "error —" not in skip["slack_text"])

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n==== selftest: {_checks['pass']} passed, {_checks['fail']} failed "
          f"({_checks['pass'] + _checks['fail']} assertions) ====")
    return 1 if _checks["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
