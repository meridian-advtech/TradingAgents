"""Selftest for kairos_watch — fire every alert condition in a sandbox.

Each condition below is driven from DISPOSABLE /tmp databases. Nothing here touches the live databases,
kairos_config.json, or Slack.

Two things are asserted per condition: that it fires on the state it is meant
to catch, and — at least as important for a channel whose value is being rare
— that it does NOT fire on the neighbouring healthy state. A monitor that
cries wolf gets muted, and a muted monitor is the same as no monitor, which is
the failure mode the whole escalation layer exists to prevent.

Run with --show to print the rendered Slack text for each condition.

Exit code 0 iff every assertion passes.
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import kairos_log_db
import kairos_watch as W

_checks = {"pass": 0, "fail": 0}
_rendered = []


def check(name, cond):
    if cond:
        _checks["pass"] += 1
        print(f"  PASS  {name}")
    else:
        _checks["fail"] += 1
        print(f"  FAIL  {name}")


def _ts(days_ago=0, hours_ago=0):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
            ).strftime("%Y-%m-%d %H:%M:%S UTC")


def _make_db(path):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE autonomy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, history_id INTEGER,
            axis TEXT, applied_at TEXT, prior_weight REAL, new_weight REAL,
            baseline_error REAL, closes_at_apply INTEGER, verdict TEXT,
            verdict_at TEXT, post_error REAL, note TEXT);
        CREATE TABLE axis_weight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, axis TEXT, run_id TEXT,
            computed_score REAL, sample_size INTEGER, evidence TEXT,
            prior_weight REAL, proposed_delta REAL, new_weight REAL,
            status TEXT, created_at TEXT, decided_at TEXT, decided_by TEXT);
        CREATE TABLE axis_weights (
            axis TEXT PRIMARY KEY, weight REAL, status TEXT,
            positive_means TEXT, updated_at TEXT);
        CREATE TABLE decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, ticker TEXT,
            action TEXT, execution_status TEXT, net_liq_after REAL);
        CREATE TABLE nlv_snapshots (
            snapshot_date TEXT, nlv REAL);
        CREATE TABLE armed_trail_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, armed_at TEXT,
            atr_pct REAL, raw_trail_pct REAL, trail_pct REAL, bind_state TEXT,
            atr_mult REAL, trail_lo_pct REAL, trail_hi_pct REAL,
            fallback_trail_pct REAL, atr_enabled INTEGER, peak_gain_pct REAL,
            target_pct REAL, created_at TEXT);
    """)
    conn.commit()
    conn.close()


def _env(tmp):
    """Fresh sandbox DB + neutralised Slack, watch state and closed-trade clock."""
    db = os.path.join(tmp, "kairos.db")
    _make_db(db)
    kairos_log_db.DB_PATH = db
    W.STATE_FILE = os.path.join(tmp, "watch_state.json")
    if os.path.exists(W.STATE_FILE):
        os.remove(W.STATE_FILE)
    return db


def _weekdays_back(n, end=None):
    return W._weekdays_back(end or date.today(), n)


def _alerts_producers() -> set:
    """Modules with a literal 'alerts' Slack destination, via AST.

    AST rather than grep because several call sites put the channel argument
    on its own line, and a line-oriented scan silently misses them — which
    would make this audit pass by not looking.
    """
    import ast, glob
    out = set()
    for f in glob.glob(os.path.join(SCRIPT_DIR, "kairos_*.py")):
        base = os.path.basename(f)
        if "selftest" in base:
            continue
        try:
            tree = ast.parse(open(f).read(), base)
        except SyntaxError:
            continue
        consts = {t.id for n in ast.walk(tree)
                  if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Constant)
                  and n.value.value == "alerts"
                  for t in n.targets if isinstance(t, ast.Name)}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if fn not in ("post_message", "post_to_slack", "alert_pipeline_event",
                          "_slack_alert", "_post_slack"):
                continue
            cand = list(n.args)[:1] + [k.value for k in n.keywords
                                       if k.arg == "channel"]
            for a in cand:
                if isinstance(a, ast.Constant) and a.value == "alerts":
                    out.add(base)
                elif isinstance(a, ast.Name) and a.id in consts:
                    out.add(base)
    return out


def _here_users() -> set:
    """Modules that emit an @here / <!here> mention."""
    import glob
    out = set()
    for f in glob.glob(os.path.join(SCRIPT_DIR, "kairos_*.py")):
        base = os.path.basename(f)
        if "selftest" in base:
            continue
        txt = open(f).read()
        # The docstring in kairos_weekly_summary only NAMES the convention;
        # what counts is emitting the literal Slack mention token.
        if "<!here>" in txt.replace('`<!here>`', ''):
            out.add(base)
    return out


def _record(title, conditions):
    if conditions:
        _rendered.append((title, W.format_alert(conditions)))


def main():
    tmp = tempfile.mkdtemp(prefix="kairos_watch_selftest_")
    print(f"selftest sandbox: {tmp}\n")

    # ── 1: a rollback fired ─────────────────────────────────────────
    print("[condition 1 — a rollback fired]")
    db = _env(tmp)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO autonomy_log (history_id, axis, applied_at, prior_weight, "
        "new_weight, baseline_error, closes_at_apply, verdict, verdict_at, "
        "post_error, note) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (99, "param:exits.trailing_stop.profit_floor_pp", _ts(5), 1.1867,
         1.3135, 2.4, 260, "rolled_back", _ts(hours_ago=2), 3.9,
         "reverted 1.3135 → 1.1867 (backup kairos_config.json.bak_20260912)"))
    conn.commit()
    conn.close()
    c1 = W.check_rollback_fired()
    check("a recent rollback fires", len(c1) == 1)
    check("...names the axis, the reason and what it reverted TO",
          "profit_floor_pp" in c1[0]["body"]
          and "2.4000 → 3.9000pp" in c1[0]["body"]
          and "1.1867" in c1[0]["body"])
    check("...and says the axis is now in cooldown",
          "barred from self-applying" in c1[0]["body"])
    _record("1. Rollback fired", c1)

    # A FAILED rollback is a different, louder thing.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE autonomy_log SET verdict = 'rollback_failed', "
                 "note = 'ROLLBACK FAILED: config write refused'")
    conn.commit()
    conn.close()
    c1b = W.check_rollback_fired()
    check("a FAILED rollback is critical, not high",
          c1b[0]["severity"] == "critical")
    check("...and says the harmful value may still be live",
          "may still be live" in c1b[0]["body"])
    _record("1b. Rollback FAILED", c1b)

    conn = sqlite3.connect(db)
    conn.execute("UPDATE autonomy_log SET verdict = 'kept'")
    conn.commit()
    conn.close()
    check("a change that was KEPT does not alert", W.check_rollback_fired() == [])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE autonomy_log SET verdict = 'rolled_back', "
                 "verdict_at = ?", (_ts(30),))
    conn.commit()
    conn.close()
    check("a rollback from a month ago does not re-alert",
          W.check_rollback_fired() == [])

    # ── 2: a guard bound and escalated ──────────────────────────────
    print("\n[condition 2 — a ratchet guard bound]")
    db = _env(tmp)
    band_ev = {"guards": {"cumulative_band": {
        "bound": True,
        "reason": "outside 7d band [5.9049, 7.9889] anchored at 6.9469",
        "would_have_been": 5.4, "clamped_to": 5.9049}}}
    mat_ev = {"guards": {"materiality": {
        "bound": True, "escalates": False, "reason": "|Δ| below threshold"}}}
    conn = sqlite3.connect(db)
    for ev, axis in ((band_ev, "param:exits.trailing_stop.target_armed.trail_pct"),
                     (mat_ev, "exit_timing")):
        conn.execute(
            "INSERT INTO axis_weight_history (axis, evidence, prior_weight, "
            "proposed_delta, new_weight, status, created_at) "
            "VALUES (?,?,?,?,?,'proposed',?)",
            (axis, json.dumps(ev), 6.2255, -0.3206, 5.9049, _ts(0)))
    conn.commit()
    conn.close()
    c2 = W.check_guard_escalated()
    check("a bound cumulative_band on an autonomous axis fires", len(c2) == 1)
    check("...and it is the ratchet guard, not the materiality one",
          "cumulative_band" in c2[0]["body"]
          and "trail_pct" in c2[0]["body"])
    check("...it names the value the guard suppressed",
          "5.4" in c2[0]["body"])
    check("a sub-threshold (materiality-only) proposal NEVER fires this — "
          "that is the Part 2 regression this guards against",
          not any("exit_timing" in c["body"] for c in c2))
    _record("2. Ratchet guard bound", c2)

    # ── 3: nothing traded ───────────────────────────────────────────
    print("\n[condition 3 — no trades in N market days]")
    db = _env(tmp)
    days = _weekdays_back(W.NO_TRADE_DAYS)
    conn = sqlite3.connect(db)
    for d in days:                       # pipeline ran, chose not to act
        for _ in range(12):
            conn.execute("INSERT INTO decisions (timestamp, ticker, action, "
                         "execution_status) VALUES (?, 'AAPL', 'BUY', 'Skipped')",
                         (f"{d.isoformat()} 14:00:00 UTC",))
    conn.commit()
    conn.close()
    c3 = W.check_no_trades()
    check("zero fills across 3 weekdays fires", len(c3) == 1)
    check("...and correctly reads as 'ran but did not act'",
          "The pipeline DID run" in c3[0]["body"])
    _record("3a. No trades — pipeline ran, no fills", c3)

    db = _env(tmp)                        # nothing recorded at all
    c3b = W.check_no_trades()
    check("an empty decisions table reads as the 2026-08-10 silent-death shape",
          len(c3b) == 1 and "2026-08-10 shape" in c3b[0]["body"]
          and "not running" in c3b[0]["body"])
    _record("3b. No trades — pipeline silent (the 2026-08-10 mode)", c3b)

    conn = sqlite3.connect(db)            # one fill anywhere in the window
    conn.execute("INSERT INTO decisions (timestamp, ticker, action, "
                 "execution_status) VALUES (?, 'AAPL', 'BUY', 'Filled')",
                 (f"{days[0].isoformat()} 14:00:00 UTC",))
    conn.commit()
    conn.close()
    check("a single fill inside the window clears it", W.check_no_trades() == [])
    check("no US market holiday closes 3 consecutive weekdays, so the "
          "weekday proxy cannot false-fire on the calendar",
          W.NO_TRADE_DAYS >= 3)

    # ── 4: NLV drawdown from peak ───────────────────────────────────
    print("\n[condition 4 — NLV drawdown from peak]")
    db = _env(tmp)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO nlv_snapshots VALUES ('2026-08-21', 1097275.69)")
    conn.execute("INSERT INTO decisions (timestamp, ticker, action, "
                 "execution_status, net_liq_after) "
                 "VALUES ('2026-09-12 15:00:00 UTC','AAPL','BUY','Filled', ?)",
                 (1097275.69 * 0.905,))       # -9.5%
    conn.commit()
    conn.close()
    c4 = W.check_drawdown()
    check(f"a 9.5% drawdown fires the {W.NLV_DRAWDOWN_PCT:.0f}% threshold",
          len(c4) == 1)
    check("...naming the peak, the peak date and the current value",
          "1,097,276" in c4[0]["body"] and "2026-08-21" in c4[0]["body"])
    check("...and it says to check the index first (a long-only book "
          "tracks it)", "index" in c4[0]["body"])
    _record("4. NLV drawdown from peak", c4)

    db = _env(tmp)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO nlv_snapshots VALUES ('2026-08-21', 1097275.69)")
    conn.execute("INSERT INTO decisions (timestamp, ticker, action, "
                 "execution_status, net_liq_after) "
                 "VALUES ('2026-09-12 15:00:00 UTC','AAPL','BUY','Filled', ?)",
                 (1097275.69 * 0.94,))        # -6%, ordinary market noise
    conn.commit()
    conn.close()
    check("a 6% drawdown does NOT fire (that is market beta, not a fault)",
          W.check_drawdown() == [])

    # ── 5: an auto-applied change trending worse ────────────────────
    print("\n[condition 5 — early warning, no action]")
    db = _env(tmp)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO autonomy_log (history_id, axis, applied_at, prior_weight, "
        "new_weight, baseline_error, closes_at_apply, verdict) "
        "VALUES (?,?,?,?,?,?,?, 'pending')",
        (101, "param:exits.trailing_stop.target_armed.trail_pct", _ts(1),
         6.2255, 5.9324, 3.0, 995))
    conn.commit()
    conn.close()
    _saved = W.__dict__.get("_test_metric")
    import kairos_autonomy as AU
    _real_metric, _real_closes = AU._baseline_metric, AU._closed_count
    AU._baseline_metric = lambda axis: 3.45      # +15% vs the 3.0 baseline
    AU._closed_count = lambda: 1000              # 5 closes since apply
    try:
        c5 = W.check_early_warning()
    finally:
        AU._baseline_metric, AU._closed_count = _real_metric, _real_closes
    check("a change trending 15% worse fires an early warning", len(c5) == 1)
    check("...at 'warning' severity, below the two action conditions",
          c5[0]["severity"] == "warning")
    check("...and it states plainly that no action was taken",
          "No action taken" in c5[0]["body"])
    check("...and why: the sample is still under ROLLBACK_MIN_CLOSES",
          "5 of 10 closes" in c5[0]["body"])
    _record("5. Auto-applied change trending worse (early warning)", c5)

    AU._baseline_metric = lambda axis: 3.03      # +1%, inside the noise
    AU._closed_count = lambda: 1000
    try:
        check("a change trending 1% worse does not fire",
              W.check_early_warning() == [])
        AU._baseline_metric = lambda axis: 3.45
        AU._closed_count = lambda: 1010          # now judgeable
        check("once ROLLBACK_MIN_CLOSES is reached this goes quiet and "
              "check_rollbacks owns it", W.check_early_warning() == [])
    finally:
        AU._baseline_metric, AU._closed_count = _real_metric, _real_closes

    # ── dedupe: rare means rare ─────────────────────────────────────
    print("\n[dedupe — a still-true condition does not re-page]")
    db = _env(tmp)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO autonomy_log (history_id, axis, applied_at, prior_weight, "
        "new_weight, baseline_error, closes_at_apply, verdict, verdict_at, "
        "post_error) VALUES (1,'exit_timing',?,0.24,0.27,2.0,10,"
        "'rolled_back',?,3.0)", (_ts(1), _ts(hours_ago=1)))
    # Seed fills so the rollback is the ONLY condition true — this section is
    # about the dedupe ledger, not about how many things can fire at once.
    for d in _weekdays_back(5):
        conn.execute("INSERT INTO decisions (timestamp, ticker, action, "
                     "execution_status, net_liq_after) "
                     "VALUES (?, 'AAPL','BUY','Filled', 1090000)",
                     (f"{d.isoformat()} 14:00:00 UTC",))
    conn.commit()
    conn.close()
    W._alerted = []
    import kairos_alerts
    _real_post = kairos_alerts.post_message
    kairos_alerts.post_message = lambda ch, txt, **k: (
        W._alerted.append((ch, txt)), True)[1]
    try:
        r1 = W.run()
        r2 = W.run()
        check("the first run posts", r1["posted"] and len(W._alerted) == 1)
        check("...to #kairos-alerts", W._alerted[0][0] == "alerts")
        check("an immediate second run does not re-post",
              not r2["posted"] and len(W._alerted) == 1)
        check("...but the condition is still reported as true",
              len(r2["found"]) == 1 and not r2["fresh"])
        check("--force overrides the cooldown", W.run(force=True)["posted"])
    finally:
        kairos_alerts.post_message = _real_post

    check("every alert carries an @here so it is visually distinct",
          all("<!here>" in t for _, t in W._alerted))
    check("...and says the channel is exception-only",
          all("only posts when something needs a decision" in t
              for _, t in W._alerted))

    # ── 6: ATR pools ready — ONE-TIME ───────────────────────────────
    print("\n[condition 6 — ATR shadow period complete, fires once]")
    db = _env(tmp)
    conn = sqlite3.connect(db)
    for state, n in (("free", 4), ("floor", 6), ("ceiling", 6), ("no_atr", 14)):
        for i in range(n):
            conn.execute(
                "INSERT INTO armed_trail_context (ticker, armed_at, bind_state, "
                "atr_enabled) VALUES (?,?,?,0)",
                (f"TK{state}{i}", "2026-09-09T14:41:00Z", state))
    for d in _weekdays_back(5):          # keep no_trades quiet
        conn.execute("INSERT INTO decisions (timestamp, ticker, action, "
                     "execution_status, net_liq_after) "
                     "VALUES (?, 'AAPL','BUY','Filled', 1090000)",
                     (f"{d.isoformat()} 14:00:00 UTC",))
    conn.commit()
    conn.close()

    _real_pools = W.atr_pool_state
    def _pools(eff):
        return lambda: {p: {"effective_n": eff, "n_contributing": int(eff) + 4,
                            "by_bind_state": {"free": 9, "floor": 12,
                                              "ceiling": 11, "no_atr": 14}}
                        for p in W.ATR_PARAMS}

    try:
        W.atr_pool_state = _pools(9.9)
        check("nothing fires while any pool is below effective_n 10",
              W.check_atr_pools_ready() == [])
        W.atr_pool_state = lambda: dict(
            _pools(14.2)(), **{W.ATR_PARAMS[0]: {
                "effective_n": 3.0, "n_contributing": 3, "by_bind_state": {}}})
        check("...including when only ONE of the three lags",
              W.check_atr_pools_ready() == [])

        W.atr_pool_state = _pools(14.2)
        c6 = W.check_atr_pools_ready()
        check("all three pools at effective_n >= 10 fires", len(c6) == 1)
        b = c6[0]["body"]
        check("...it is flagged one-time", c6[0].get("once") is True)
        check("...says the shadow period is complete and a decision is due",
              "ready for a decision" in c6[0]["title"])
        check("...reports the observed HELD-BOOK bind split",
              "free 4" in b and "floor 6" in b and "ceiling 6" in b)
        check("...compares it against the 2026-09-09 backtest assumption",
              "19 free / 23 floor / 7 ceiling" in b and "expected" in b)
        check("...reports the closed-evidence split separately (different "
              "population, different question)", "closed evidence" in b)
        check("...states plainly that flipping tightens EVERY armed position",
              "tightens the trail on *every* armed position" in b)
        check("...and that the learning loop can then correct it, which is "
              "why the clamp is tight",
              "learning loop loosen it" in b and "[2.0, 4.0]" in b)
        check("...and that nothing was changed and it will not repeat",
              "Nothing has been changed" in b
              and "`atr_enabled` is still" in b
              and "fires once and will not repeat" in b)
        check("...it does NOT flip atr_enabled",
              "no automatic path will flip it" in b)
        _record("6. ATR pools ready (one-time)", c6)

        # The flag: fire once, never again.
        W._alerted = []
        import kairos_alerts
        _real_post = kairos_alerts.post_message
        kairos_alerts.post_message = lambda ch, txt, **k: (
            W._alerted.append((ch, txt)), True)[1]
        try:
            r1 = W.run()
            check("the one-time alert posts on the first run",
                  r1["posted"] and len(W._alerted) == 1)
            state = W.load_state()
            check("...and persists a flag in the `once` ledger, not `fired`",
                  "atr_pools_ready" in (state.get("once") or {})
                  and "atr_pools_ready" not in (state.get("fired") or {}))
            r2 = W.run()
            check("a second run does NOT re-fire it",
                  not r2["posted"] and len(W._alerted) == 1)
            check("...even though the condition is still true",
                  any(c["key"] == "atr_pools_ready" for c in r2["found"]))
            r3 = W.run(force=True)
            check("--force does NOT override a one-time flag",
                  not r3["posted"] and len(W._alerted) == 1)

            # The flag must survive the 7-day prune that clears `fired`.
            state = W.load_state()
            state["fired"] = {"stale:key": _ts(30)}
            W.save_state(state)
            conn = sqlite3.connect(db)   # add a rollback so run() has work
            conn.execute(
                "INSERT INTO autonomy_log (history_id, axis, applied_at, "
                "prior_weight, new_weight, baseline_error, closes_at_apply, "
                "verdict, verdict_at, post_error) VALUES (1,'exit_timing',?,"
                "0.24,0.27,2.0,10,'rolled_back',?,3.0)",
                (_ts(1), _ts(hours_ago=1)))
            conn.commit()
            conn.close()
            r4 = W.run()
            state = W.load_state()
            check("the prune that clears stale `fired` keys leaves `once` "
                  "intact — a pruned one-time flag is a repeating alert",
                  "atr_pools_ready" in (state.get("once") or {})
                  and "stale:key" not in (state.get("fired") or {}))
            check("...and the ATR notice is not in that new post",
                  all("shadow period" not in t for _, t in W._alerted[1:]))
        finally:
            kairos_alerts.post_message = _real_post
    finally:
        W.atr_pool_state = _real_pools

    # ── quiet state posts nothing ───────────────────────────────────
    print("\n[a healthy system is silent]")
    db = _env(tmp)
    conn = sqlite3.connect(db)
    for d in _weekdays_back(5):
        conn.execute("INSERT INTO decisions (timestamp, ticker, action, "
                     "execution_status, net_liq_after) "
                     "VALUES (?, 'AAPL','BUY','Filled', 1090000)",
                     (f"{d.isoformat()} 14:00:00 UTC",))
    conn.execute("INSERT INTO nlv_snapshots VALUES ('2026-09-01', 1095000)")
    conn.commit()
    conn.close()
    quiet = W.run(dry_run=True)
    check("nothing fires on a healthy system",
          quiet["found"] == [] and quiet["text"] is None)

    # ── channel discipline (static audit) ───────────────────────────
    # #kairos-alerts is only worth having if it is rare. Over a 3-6 month
    # low-touch window the realistic way it degrades is not one bad decision
    # but accretion: someone adds a per-cycle summary to it, the channel
    # becomes routine, and by the time it matters J has stopped reading. So
    # the allowlist is pinned here, and adding a producer is a deliberate edit
    # to this list with a justification in the commit — the same contract as
    # AUTO_APPLY_AXES.
    print("\n[channel discipline — who may post to #kairos-alerts]")
    allowed = {
        "kairos_alerts.py":            "Phase 2 reasoning timeout",
        "kairos_arbiter.py":           "Arbiter run FAILED (not the daily report)",
        "kairos_autonomy.py":          "rollback could not be verified",
        "kairos_dividend.py":          "protected position down >N% — human review",
        "kairos_health.py":            "health check failed",
        "kairos_healthcheck.py":       "infrastructure findings",
        "kairos_ledger.py":            "trade-record write failed / positions ledger "
                                       "≠ broker (come look) / long-only violation / "
                                       "Flex nightly pull failed",
        "kairos_reallocation.py":      "reallocation aborted / buy leg failed",
        "kairos_regime.py":            "regime transition (rare; changes guardrails)",
        "kairos_run.py":               "scheduler crash",
        "kairos_sell_guard.py":        "oversell clamped or aborted",
        "kairos_stoploss.py":          "wash-sale violation",
        "kairos_thesis_review.py":     "wash-sale violation",
        "kairos_universe_liveness.py": "held symbol stopped trading",
        "kairos_watch.py":             "the escalation layer itself",
    }
    found = _alerts_producers()
    unexpected = sorted(found - set(allowed))
    check("no module posts to #kairos-alerts without being on the allowlist"
          + (f" (unexpected: {unexpected})" if unexpected else ""),
          not unexpected)
    missing = sorted(set(allowed) - found)
    check("every allowlisted module still posts there (stale entries removed)"
          + (f" (stale: {missing})" if missing else ""),
          not missing)
    check("only the escalation layer uses @here",
          _here_users() == {"kairos_autonomy.py", "kairos_watch.py"})

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n==== watch selftest: {_checks['pass']} passed, "
          f"{_checks['fail']} failed "
          f"({_checks['pass'] + _checks['fail']} assertions) ====")
    return 1 if _checks["fail"] else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true",
                    help="Print the rendered Slack text for each condition")
    args = ap.parse_args()
    rc = main()
    if args.show:
        print("\n\n" + "=" * 72)
        print("  RENDERED SLACK MESSAGES (simulated)")
        print("=" * 72)
        for title, text in _rendered:
            print(f"\n\n─── {title} " + "─" * max(0, 60 - len(title)))
            print(text)
    sys.exit(rc)
