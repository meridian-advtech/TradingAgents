"""Autonomous application of Arbiter proposals, with automatic rollback.

WHY THIS IS A SEPARATE MODULE
It calls kairos_axis_weights.apply_decision() rather than reaching into the
proposal machinery. That keeps every existing guard on the live path -- the
status check, the param whitelist and bounds, the config backup -- and means
autonomy cannot introduce a failure mode that supervised approval does not
already have. It also means this file can be deleted to fully disable autonomy.

WHAT EARNS AUTONOMY
Per-axis, never global. The 2026-08-20 shadow replay
(kairos_shadow_autonomy.py) measured what full autonomy WOULD have produced
against what supervision actually produced:

    exit_timing        29 runs, drift +0.0513, band never bound  -> eligible
    exit params         2-3 runs post-redesign, mostly gated     -> NOT eligible

That param verdict was correct AT THE TIME and has since been earned out. On
2026-09-12 the two non-ATR params carry real, weighted, two-sided evidence
pools -- profit_floor_pp effective_n 33.4 over 71 contributing closes,
trail_pct effective_n 28.8 over the same corpus -- and have each proposed
through the redesigned objective for weeks without a band bind. They are now
in AUTO_APPLY_AXES.

The three ATR-family params are deliberately NOT, and the reason is not
sample size, it is that the mechanism has never run: atr_enabled is false, so
no close has ever been governed by atr_mult / trail_lo_pct / trail_hi_pct and
all three pools are empty (effective_n 0, gated). Auto-applying changes to a
mechanism with no live history is precisely the thing not to do unattended.

WHY ROLLBACK IS PART OF THE SAME FILE
Autonomy without reversion is not learning, it is committing. Every change was
one-way before this: nothing measured whether an applied change actually helped,
and nothing could undo it. AUTO-APPLY AND ROLLBACK SHIP TOGETHER OR NEITHER
SHIPS -- the July ratchet was not caused by a single bad step but by nothing
noticing the accumulated direction was wrong.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone


# Axes permitted to self-apply. Deliberately a hardcoded allowlist rather than
# a config key: adding an axis here should be a reviewed code change backed by
# a shadow replay, not a value someone can flip in a JSON file at 2am.
#
# Widened 2026-09-12 (see module docstring). The ATR family stays out.
AUTO_APPLY_AXES = [
    "exit_timing",
    "param:exits.trailing_stop.profit_floor_pp",
    "param:exits.trailing_stop.target_armed.trail_pct",
]

# Closed trades that must accumulate after an auto-applied change before its
# effect can be judged. Below this the post-change sample is noise.
ROLLBACK_MIN_CLOSES = 10

# Fractional worsening of the axis's own error metric that triggers reversion.
# 0.20 = the change made things 20% worse than the pre-change baseline.
ROLLBACK_DEGRADE_FRAC = 0.20

# After a rollback, an axis is barred from self-applying again for this long
# and routes to a human instead.
#
# Without it the loop oscillates. A revert restores the prior value, which
# changes the evidence hash, which lets the next run re-propose the very move
# that was just judged harmful -- apply, revert, apply, revert, each step
# individually legal, exactly the shape of the July ratchet with the sign
# flipped. 14 days is chosen against this corpus's cadence: 3-8 closes a week
# means two weeks is roughly the ROLLBACK_MIN_CLOSES horizon again, so the
# axis cannot re-apply until there is genuinely a fresh sample behind it.
ROLLBACK_COOLDOWN_DAYS = 14


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _is_param_axis(axis: str) -> bool:
    return str(axis).startswith("param:")


def _param_path(axis: str) -> str:
    return str(axis)[len("param:"):]


def _baseline_metric(axis: str) -> float | None:
    """Current absolute error for an axis -- the thing a change should reduce.

    Both compute paths report mean_error_pp on the same two-sided definition
    (give-back minus forgone gain), which is exactly 'how wrong is our exit
    timing'. Smaller absolute value is better regardless of sign, since either
    pole is a cost.

    Param axes route to compute_param, NOT compute_axis: compute_axis raises
    ValueError on anything outside AXES, so before this branch existed every
    param baseline and every param post-change reading came back None, and
    _judge could only ever return 'unmeasurable'. A param could be applied and
    then never judged.
    """
    try:
        if _is_param_axis(axis):
            from kairos_axis_weights import compute_param
            ev = (compute_param(_param_path(axis)) or {}).get("evidence") or {}
        else:
            from kairos_axis_weights import compute_axis
            ev = (compute_axis(axis) or {}).get("evidence") or {}
        err = ev.get("mean_error_pp")
        return abs(float(err)) if err is not None else None
    except Exception:
        return None


def _live_value(axis: str) -> float | None:
    """What this axis is worth RIGHT NOW, read from wherever it actually lives.

    The distinction this function exists to enforce: a param's live value is a
    number in kairos_config.json, and a plain axis's live value is a row in
    kairos.db:axis_weights. Reading (and writing) the wrong one is the Part 1
    bug -- 'UPDATE axis_weights SET weight = ?' on a param:* axis updates a
    row nothing reads while the config keeps the harmful number.
    """
    try:
        if _is_param_axis(axis):
            from kairos_axis_weights import _current_param_value
            v = _current_param_value(_param_path(axis))
            return None if v is None else float(v)
        import kairos_log_db as kdb
        conn = kdb.get_connection()
        try:
            row = conn.execute(
                "SELECT weight FROM axis_weights WHERE axis = ?", (axis,)).fetchone()
        finally:
            conn.close()
        return None if row is None or row["weight"] is None else float(row["weight"])
    except Exception:
        return None


def _alert(text: str) -> bool:
    """Post to #kairos-alerts. Never raises -- a Slack outage must not stop a
    revert, and a revert must not be silently un-reported either, so a failure
    here still prints."""
    try:
        from kairos_alerts import post_message
        return post_message("alerts", text)
    except Exception as exc:  # pragma: no cover - transport failure
        print(f"  WARNING: autonomy alert could not be posted: {exc}\n{text}")
        return False


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Ledger of auto-applied changes awaiting an outcome verdict."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS autonomy_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            history_id     INTEGER NOT NULL,
            axis           TEXT NOT NULL,
            applied_at     TEXT NOT NULL,
            prior_weight   REAL,
            new_weight     REAL,
            baseline_error REAL,
            closes_at_apply INTEGER,
            verdict        TEXT DEFAULT 'pending',
            verdict_at     TEXT,
            post_error     REAL,
            note           TEXT
        )
    """)
    conn.commit()


def _closed_count() -> int:
    """Total closed trades -- the clock against which 'settled' is measured."""
    try:
        from kairos_ml_outcomes import DB_PATH as ML_DB
        c = sqlite3.connect(ML_DB)
        try:
            return c.execute(
                "SELECT COUNT(*) FROM trade_outcomes "
                "WHERE outcome_label IS NOT NULL").fetchone()[0]
        finally:
            c.close()
    except Exception:
        return 0


def _parse_ts(raw) -> "datetime | None":
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).replace(" UTC", ""),
                                 "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def cooldown_axes(now: "datetime | None" = None) -> dict:
    """{axis: verdict_at} for axes inside ROLLBACK_COOLDOWN_DAYS of a rollback."""
    import kairos_log_db as kdb

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ROLLBACK_COOLDOWN_DAYS)
    conn = kdb.get_connection()
    try:
        _ensure_table(conn)
        rows = [dict(r) for r in conn.execute(
            "SELECT axis, verdict_at FROM autonomy_log "
            "WHERE verdict IN ('rolled_back', 'rollback_failed')")]
    finally:
        conn.close()
    out = {}
    for r in rows:
        ts = _parse_ts(r["verdict_at"])
        if ts is not None and ts >= cutoff:
            out[r["axis"]] = r["verdict_at"]
    return out


def auto_apply(dry_run: bool = True) -> dict:
    """Apply eligible pending proposals without a human decision.

    Eligibility is intentionally narrow. A proposal must be: for an axis in
    AUTO_APPLY_AXES, still 'proposed', not gated, an actual movement, not
    inside a post-rollback cooldown, computed against the value that is still
    live, and free of any ESCALATING guard bind. A guard that fired means the
    loop tried to move further than the cumulative band allows -- that is
    precisely the ratchet signature, so it stays for a human even on an
    otherwise-eligible axis.

    'Escalating' is the 2026-09-12 correction. Materiality is recorded through
    the same guard channel but gates CARDING, not APPLYING: a move too small
    to interrupt a human for is not a move too dangerous to make. Blocking on
    it left exit_timing unable to either apply or ask -- see
    kairos_axis_weights.NON_ESCALATING_GUARDS.
    """
    from kairos_axis_weights import apply_decision, escalating_guards
    import kairos_log_db as kdb

    conn = kdb.get_connection()
    try:
        _ensure_table(conn)
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM axis_weight_history WHERE status = 'proposed'")]
    finally:
        conn.close()

    cooling = cooldown_axes()
    applied, skipped = [], []
    for r in rows:
        axis = r["axis"]
        why = None
        if axis not in AUTO_APPLY_AXES:
            why = "axis not approved for autonomy"
        elif abs(r["proposed_delta"] or 0) < 1e-9:
            why = "no movement proposed"
        elif axis in cooling:
            why = (f"post-rollback cooldown until "
                   f"{ROLLBACK_COOLDOWN_DAYS}d after {cooling[axis]} — "
                   f"escalating to human")
        else:
            ev = {}
            try:
                ev = json.loads(r["evidence"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            bound = escalating_guards(ev)
            if bound:
                why = (f"a ratchet guard bound ({', '.join(sorted(bound))}) - "
                       f"escalating to human")
            else:
                # Staleness. A proposal carries the value it was computed
                # against; if the live value has moved since (a human applied
                # something, or a revert fired), new_weight is an answer to a
                # question nobody is asking any more. _apply_param_to_config
                # re-checks bounds against the live file but cannot know the
                # DIRECTION was computed from a stale anchor.
                live = _live_value(axis)
                prior = r["prior_weight"]
                if (live is not None and prior is not None
                        and abs(float(live) - float(prior)) > 1e-9):
                    why = (f"stale: computed against {float(prior):.6g}, "
                           f"live value is now {float(live):.6g} — "
                           f"escalating to human")
        if why:
            skipped.append({"id": r["id"], "axis": axis, "reason": why})
            continue
        applied.append(r)

    if dry_run:
        return {"dry_run": True, "would_apply": applied, "skipped": skipped}
    return _commit_applies(applied, skipped, apply_decision)


def _commit_applies(applied: list, skipped: list, apply_decision) -> dict:
    """Record a baseline, then apply. Baseline FIRST so a crash mid-apply
    leaves evidence of what was about to change."""
    import kairos_log_db as kdb

    done = []
    for r in applied:
        baseline = _baseline_metric(r["axis"])
        closes = _closed_count()
        conn = kdb.get_connection()
        try:
            _ensure_table(conn)
            cur = conn.execute(
                "INSERT INTO autonomy_log (history_id, axis, applied_at, "
                " prior_weight, new_weight, baseline_error, closes_at_apply) "
                "VALUES (?,?,?,?,?,?,?)",
                (r["id"], r["axis"], _now(), r["prior_weight"],
                 r["new_weight"], baseline, closes))
            conn.commit()
            log_id = cur.lastrowid
        finally:
            conn.close()
        try:
            apply_decision(r["id"], "approve", "auto")
            done.append({"id": r["id"], "axis": r["axis"],
                         "prior": r["prior_weight"], "new": r["new_weight"],
                         "autonomy_log_id": log_id})
        except Exception as exc:
            # The ledger row was written before the attempt on purpose, but a
            # FAILED apply must not sit 'pending' forever waiting for a
            # verdict on a change that never landed.
            conn = kdb.get_connection()
            try:
                conn.execute(
                    "UPDATE autonomy_log SET verdict = 'apply_failed', "
                    " verdict_at = ?, note = ? WHERE id = ?",
                    (_now(), f"apply raised: {exc}", log_id))
                conn.commit()
            finally:
                conn.close()
            skipped.append({"id": r["id"], "axis": r["axis"],
                            "reason": f"apply failed: {exc}"})
    return {"dry_run": False, "applied": done, "skipped": skipped}


def check_rollbacks(dry_run: bool = True) -> dict:
    """Judge settled auto-applied changes; revert the ones that hurt.

    A change is judged only once ROLLBACK_MIN_CLOSES trades have closed since
    it landed -- before that the post-change sample is noise and a verdict
    would be superstition. Reversion writes the value back directly rather
    than going through apply_decision, because there is no proposal row to
    approve: this is undoing one, and it must be recorded as a distinct event
    so a reverted change never reads as a supervised decision.
    """
    import kairos_log_db as kdb

    now_closes = _closed_count()
    conn = kdb.get_connection()
    try:
        _ensure_table(conn)
        pending = [dict(r) for r in conn.execute(
            "SELECT * FROM autonomy_log WHERE verdict = 'pending'")]
    finally:
        conn.close()

    verdicts = []
    for row in pending:
        elapsed = now_closes - (row["closes_at_apply"] or 0)
        if elapsed < ROLLBACK_MIN_CLOSES:
            verdicts.append({"id": row["id"], "axis": row["axis"],
                             "verdict": "too_early",
                             "closes_since": elapsed})
            continue
        verdicts.append(_judge(row, elapsed, dry_run))
    return {"dry_run": dry_run, "closed_total": now_closes,
            "verdicts": verdicts}


def _revert(axis: str, prior_value) -> dict:
    """Put `axis` back to `prior_value` THROUGH THE PATH THAT OWNS IT.

    This branch is the Part 1 fix. Both arms verify by re-reading the live
    value from its own source of truth afterwards, and raise if it did not
    land -- a revert that cannot be proven is not a revert, and logging
    'rolled_back' over a value that is still live is a false all-clear, which
    is strictly worse than having no rollback at all.
    """
    if prior_value is None:
        raise RuntimeError(f"cannot revert {axis}: no prior value recorded")

    if _is_param_axis(axis):
        # kairos_config.json, via the same writer apply uses (whitelist, hard
        # bounds, ordered-pair check, timestamped backup, atomic write, JSON
        # re-validate), in restore mode -- see revert_param_in_config.
        from kairos_axis_weights import revert_param_in_config
        return revert_param_in_config(_param_path(axis), float(prior_value))

    import kairos_log_db as kdb
    conn = kdb.get_connection()
    try:
        before = _live_value(axis)
        conn.execute("UPDATE axis_weights SET weight = ?, updated_at = ? "
                     "WHERE axis = ?", (prior_value, _now(), axis))
        conn.commit()
    finally:
        conn.close()
    landed = _live_value(axis)
    if landed is None or abs(float(landed) - float(prior_value)) > 1e-9:
        raise RuntimeError(
            f"REVERT NOT VERIFIED: axis_weights.{axis} should be "
            f"{prior_value} after the revert but reads {landed!r}. The "
            f"harmful value may still be live — do not treat this rollback "
            f"as done.")
    return {"axis": axis, "from_value": before, "to_value": float(landed),
            "backup": None, "verified": True}


def _judge(row: dict, elapsed: int, dry_run: bool) -> dict:
    """Compare post-change error against the baseline; revert if worse."""
    import kairos_log_db as kdb

    post = _baseline_metric(row["axis"])
    base = row["baseline_error"]
    out = {"id": row["id"], "axis": row["axis"], "closes_since": elapsed,
           "baseline_error": base, "post_error": post}

    if post is None or base is None:
        out["verdict"] = "unmeasurable"
        out["note"] = "no error metric available on one side of the comparison"
    elif base <= 1e-9:
        # Baseline was already ~perfect; any worsening is real but the ratio
        # is undefined, so fall back to an absolute check.
        out["verdict"] = "rolled_back" if post > 0.5 else "kept"
    else:
        worsened = (post - base) / base
        out["worsened_frac"] = round(worsened, 4)
        out["verdict"] = ("rolled_back" if worsened > ROLLBACK_DEGRADE_FRAC
                          else "kept")

    if dry_run:
        return out

    note = out.get("note")
    if out["verdict"] == "rolled_back":
        try:
            rv = _revert(row["axis"], row["prior_weight"])
            out["reverted"] = rv
            note = (f"reverted {rv['from_value']} → {rv['to_value']}"
                    + (f" (backup {rv['backup']})" if rv.get("backup") else ""))
        except Exception as exc:
            # Do NOT record 'rolled_back'. The whole point of Part 1 is that a
            # verdict must describe what is true on disk.
            out["verdict"] = "rollback_failed"
            out["error"] = str(exc)
            note = f"ROLLBACK FAILED: {exc}"
            _alert(
                f"<!here> :rotating_light: *Kairos — ROLLBACK FAILED*\n"
                f"`{row['axis']}` was judged harmful (error "
                f"{base} → {post} over {elapsed} closes) but could NOT be "
                f"reverted to {row['prior_weight']}.\n"
                f"```{exc}```\n"
                f"*The harmful value is still live.* Revert it by hand.")

    conn = kdb.get_connection()
    try:
        conn.execute(
            "UPDATE autonomy_log SET verdict = ?, verdict_at = ?, "
            " post_error = ?, note = ? WHERE id = ?",
            (out["verdict"], _now(), post, note, row["id"]))
        conn.commit()
    finally:
        conn.close()
    out["note"] = note
    return out


def _cli() -> int:
    """Dry-run report. Compact on purpose: dumping each row's `evidence` blob
    buries the two lines that matter (what would move, and what would not) in
    several KB of JSON, and this is read from a log."""
    aa = auto_apply(dry_run=True)
    print("AUTO-APPLY (dry run) — axes with autonomy: "
          + ", ".join(AUTO_APPLY_AXES))
    if not aa["would_apply"]:
        print("  would apply: nothing")
    for r in aa["would_apply"]:
        ev = {}
        try:
            ev = json.loads(r["evidence"] or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        mat = (ev.get("materiality") or {}).get("material")
        band = ev.get("cumulative_band") or {}
        print(f"  WOULD APPLY  id {r['id']:>4}  {r['axis']}")
        print(f"               {r['prior_weight']:g} → {r['new_weight']:g}  "
              f"(Δ {r['proposed_delta']:+g}, "
              f"{'material' if mat else 'sub-threshold: applies quietly'}, "
              f"effective_n {ev.get('effective_n')})")
        if band.get("lo") is not None:
            print(f"               7d band [{band['lo']:g}, {band['hi']:g}] "
                  f"anchored {band['base_7d']:g} — not bound")
    for s in aa["skipped"]:
        print(f"  skip         id {s['id']:>4}  {s['axis']}\n"
              f"               {s['reason']}")

    rb = check_rollbacks(dry_run=True)
    print(f"\nROLLBACK CHECK (dry run) — {rb['closed_total']} closes total, "
          f"need {ROLLBACK_MIN_CLOSES} since apply to judge")
    if not rb["verdicts"]:
        print("  nothing pending a verdict")
    for v in rb["verdicts"]:
        print(f"  {v['verdict']:<14} {v['axis']}  "
              f"closes_since={v.get('closes_since')}  "
              f"base={v.get('baseline_error')} post={v.get('post_error')}")

    cd = cooldown_axes()
    print(f"\nPOST-ROLLBACK COOLDOWN ({ROLLBACK_COOLDOWN_DAYS}d): "
          + (", ".join(f"{k} since {v}" for k, v in cd.items()) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
