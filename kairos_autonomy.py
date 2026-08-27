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

The params look identical to supervision only because almost nothing happened;
that is inertia, not evidence. Their first ungated proposal would move 25% off
a 2-trade sample while the 30d/60d forgone-gain horizons hold zero rows, so the
too-early pole cannot yet argue back. They stay human-gated until each has
cleared PARAM_MIN_SAMPLE on at least two post-redesign runs.

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
from datetime import datetime, timezone


# Axes permitted to self-apply. Deliberately a hardcoded allowlist rather than
# a config key: adding an axis here should be a reviewed code change backed by
# a shadow replay, not a value someone can flip in a JSON file at 2am.
# Params are absent by design -- see module docstring.
AUTO_APPLY_AXES = ["exit_timing"]

# Closed trades that must accumulate after an auto-applied change before its
# effect can be judged. Below this the post-change sample is noise.
ROLLBACK_MIN_CLOSES = 10

# Fractional worsening of the axis's own error metric that triggers reversion.
# 0.20 = the change made things 20% worse than the pre-change baseline.
ROLLBACK_DEGRADE_FRAC = 0.20


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _baseline_metric(axis: str) -> float | None:
    """Current absolute error for an axis -- the thing a change should reduce.

    exit_timing's compute already reports mean_error_pp (give-back minus
    forgone gain), which is exactly 'how wrong is our exit timing'. Smaller
    absolute value is better regardless of sign, since either pole is a cost.
    """
    try:
        from kairos_axis_weights import compute_axis
        ev = (compute_axis(axis) or {}).get("evidence") or {}
        err = ev.get("mean_error_pp")
        return abs(float(err)) if err is not None else None
    except Exception:
        return None


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


def auto_apply(dry_run: bool = True) -> dict:
    """Apply eligible pending proposals without a human decision.

    Eligibility is intentionally narrow. A proposal must be: for an axis in
    AUTO_APPLY_AXES, still 'proposed', not gated, an actual movement, and free
    of any guard bind. A guard that fired means the loop tried to move further
    than the cumulative band allows -- that is precisely the ratchet signature,
    so it stays for a human even on an otherwise-eligible axis.
    """
    from kairos_axis_weights import apply_decision
    import kairos_log_db as kdb

    conn = kdb.get_connection()
    try:
        _ensure_table(conn)
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM axis_weight_history WHERE status = 'proposed'")]
    finally:
        conn.close()

    applied, skipped = [], []
    for r in rows:
        axis = r["axis"]
        why = None
        if axis not in AUTO_APPLY_AXES:
            why = "axis not approved for autonomy"
        elif abs(r["proposed_delta"] or 0) < 1e-9:
            why = "no movement proposed"
        else:
            ev = {}
            try:
                ev = json.loads(r["evidence"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            if any(g.get("bound") for g in (ev.get("guards") or {}).values()):
                why = "a ratchet guard bound - escalating to human"
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
            conn.execute(
                "INSERT INTO autonomy_log (history_id, axis, applied_at, "
                " prior_weight, new_weight, baseline_error, closes_at_apply) "
                "VALUES (?,?,?,?,?,?,?)",
                (r["id"], r["axis"], _now(), r["prior_weight"],
                 r["new_weight"], baseline, closes))
            conn.commit()
        finally:
            conn.close()
        try:
            apply_decision(r["id"], "approve", "auto")
            done.append({"id": r["id"], "axis": r["axis"],
                         "prior": r["prior_weight"], "new": r["new_weight"]})
        except Exception as exc:
            skipped.append({"id": r["id"], "axis": r["axis"],
                            "reason": f"apply failed: {exc}"})
    return {"dry_run": False, "applied": done, "skipped": skipped}


def check_rollbacks(dry_run: bool = True) -> dict:
    """Judge settled auto-applied changes; revert the ones that hurt.

    A change is judged only once ROLLBACK_MIN_CLOSES trades have closed since
    it landed -- before that the post-change sample is noise and a verdict
    would be superstition. Reversion writes the weight back directly rather
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

    conn = kdb.get_connection()
    try:
        if out["verdict"] == "rolled_back":
            conn.execute("UPDATE axis_weights SET weight = ? WHERE axis = ?",
                         (row["prior_weight"], row["axis"]))
        conn.execute(
            "UPDATE autonomy_log SET verdict = ?, verdict_at = ?, "
            " post_error = ?, note = ? WHERE id = ?",
            (out["verdict"], _now(), post, out.get("note"), row["id"]))
        conn.commit()
    finally:
        conn.close()
    return out


if __name__ == "__main__":
    print("AUTO-APPLY (dry run):")
    print(json.dumps(auto_apply(dry_run=True), indent=2, default=str))
    print("\nROLLBACK CHECK (dry run):")
    print(json.dumps(check_rollbacks(dry_run=True), indent=2, default=str))
