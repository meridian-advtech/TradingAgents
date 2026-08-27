"""Shadow autonomy: what WOULD the loop have done, unattended?

WHY THIS EXISTS
The plan is to let the Arbiter apply its own proposals with no human in the
loop. The honest precondition for that is evidence, not confidence: a record of
what full autonomy would actually have produced, compared against what the
supervised system did produce. Until that comparison exists, "the system has
learned enough to be trusted" is an assertion.

DESIGN: READ-ONLY REPLAY, NOT A HOOK
This module never participates in the proposal loop. It reads
axis_weight_history after the fact and re-derives the counterfactual. Three
consequences, all deliberate:

  * It cannot perturb what it measures. A shadow mode wired INTO
    propose_update would share a code path with live proposals; a bug in the
    shadow would become a bug in production.
  * It can backfill. Every proposal since 2026-06-19 is already on disk, so the
    autonomy question is answerable today rather than after weeks of new
    collection.
  * It stays honest across code changes. compute_delta is a pure function of
    (computed_score, prior_weight, sample_size, cfg), and both score and sample
    size are stored per row — so the replay re-derives each step from the
    SHADOW weight rather than replaying the stored delta, which is what makes
    the counterfactual faithful instead of merely additive.

WHAT IT COMPARES
  actual   — only 'approved' rows applied (what really happened)
  shadow   — every proposal applied, no gate, no guards (naive autonomy)
  guarded  — every proposal applied, but with the cumulative band enforced
             (autonomy as it would run TODAY, post-2026-08-20 guards)

The gap between `shadow` and `guarded` is the value the guards add. The gap
between `guarded` and `actual` is what the human gate is still contributing.
When that second gap stops mattering, autonomy has been earned.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone


def _load_proposals(conn: sqlite3.Connection, axis: str) -> list[dict]:
    """Chronological proposal history for one axis, one row per run.

    Several rows can share a run_id when a proposal was superseded and
    re-issued inside the same run. Replaying all of them would apply the same
    run's decision more than once and inflate the counterfactual, so we keep
    only the last row per run_id -- the one that actually stood.
    """
    rows = [dict(r) for r in conn.execute(
        "SELECT id, axis, run_id, computed_score, sample_size, evidence, "
        "       prior_weight, proposed_delta, new_weight, status, created_at "
        "FROM axis_weight_history WHERE axis = ? "
        "ORDER BY created_at ASC, id ASC", (axis,))]

    by_run: dict = {}
    for r in rows:
        by_run[r["run_id"]] = r          # later row for a run wins
    return sorted(by_run.values(), key=lambda r: (r["created_at"], r["id"]))


def _is_param(axis: str) -> bool:
    return isinstance(axis, str) and axis.startswith("param:")


def _band_bounds(axis: str, anchor: float, history: list[tuple]) -> tuple:
    """Cumulative-band bounds for the shadow trajectory.

    Mirrors the live guards but operates on the SHADOW history rather than
    axis_weight_history, because under autonomy the anchor is whatever the
    shadow had in force at the window's start -- not what really happened.
    Param axes use the multiplicative band, axes the additive one, matching
    PARAM_CUMULATIVE_BAND_FRAC / AXIS_CUMULATIVE_BAND_ABS respectively.
    """
    from kairos_axis_weights import (
        PARAM_CUMULATIVE_WINDOW_DAYS, PARAM_CUMULATIVE_BAND_FRAC,
        AXIS_CUMULATIVE_WINDOW_DAYS, AXIS_CUMULATIVE_BAND_ABS,
    )
    if _is_param(axis):
        window, frac, absolute = PARAM_CUMULATIVE_WINDOW_DAYS, PARAM_CUMULATIVE_BAND_FRAC, None
    else:
        window, frac, absolute = AXIS_CUMULATIVE_WINDOW_DAYS, None, AXIS_CUMULATIVE_BAND_ABS

    base = anchor
    if history:
        cutoff = history[-1][0] - timedelta(days=window)
        older = [h for h in history if h[0] >= cutoff]
        if older:
            base = older[0][1]           # weight in force at window start

    if absolute is not None:
        return base - absolute, base + absolute, base
    return base * (1.0 - frac), base * (1.0 + frac), base


def _param_delta(score: float, current: float, n: int) -> dict:
    """Param-side equivalent of compute_delta.

    CORRECTION (2026-08-20): the first cut of this replay ran param axes
    through compute_delta, which is AXIS machinery — it gates on
    cfg['min_sample']=20 and clamps to cfg['weight_bound']=±1.0. Feeding it
    trail_pct=8.0 clamped the value to 1.0 on the very first step, and the
    resulting "full autonomy drives trail_pct to 0.70" was an artifact of that
    clamp, not a property of the loop. Params gate on PARAM_MIN_SAMPLE=10 and
    move by a proportional PARAM_MAX_CHANGE_FRAC step with no ±1.0 bound, so
    they need their own path.
    """
    from kairos_axis_weights import (PARAM_MIN_SAMPLE, PARAM_MAX_CHANGE_FRAC,
                                     PARAM_WHITELIST)
    if n < PARAM_MIN_SAMPLE:
        return {"new_value": current, "gated": True}
    # Live code is: change = role_sign * error_sign * severity * max_step,
    # and computed_score is that same product without max_step — so the step
    # is simply score * PARAM_MAX_CHANGE_FRAC * |current|. Deriving it from the
    # stored score (rather than re-deriving role_sign here) keeps the replay
    # tied to what was actually recorded.
    frac = max(-PARAM_MAX_CHANGE_FRAC,
               min(PARAM_MAX_CHANGE_FRAC, score * PARAM_MAX_CHANGE_FRAC))
    return {"new_value": current * (1.0 + frac), "gated": False}


# The 2026-08-03 redesign (two-sided objective + role_sign) changed how
# computed_score is signed for params. Rows written before it used the older
# convention, so replaying across the boundary silently mixes two incompatible
# sign systems — a 7/14 trail_pct row has score +0.62 for a TIGHTENING that a
# post-redesign row would record as negative. Param replay therefore starts
# here; axes are unaffected (compute_delta's convention did not change).
PARAM_CONVENTION_CUTOFF = "2026-08-03"


def replay_axis(axis: str, db_path: str = None) -> dict:
    """Re-derive actual / shadow / guarded trajectories for one axis."""
    from kairos_axis_weights import compute_delta, load_config
    import kairos_log_db as kdb

    cfg = load_config()
    conn = sqlite3.connect(db_path or kdb.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = _load_proposals(conn, axis)
    finally:
        conn.close()
    if not rows:
        return {"axis": axis, "steps": [], "n_rows": 0}

    truncated = False
    if _is_param(axis):
        kept = [r for r in rows if (r["created_at"] or "") >= PARAM_CONVENTION_CUTOFF]
        truncated = len(kept) < len(rows)
        rows = kept
        if not rows:
            return {"axis": axis, "steps": [], "n_rows": 0, "truncated": True}

    start = rows[0]["prior_weight"] or 0.0
    actual = shadow = guarded = start
    guarded_hist: list[tuple] = []
    steps, band_binds = [], 0

    for r in rows:
        score, n = r["computed_score"], r["sample_size"] or 0
        created = r["created_at"]

        # actual: only approvals moved the real weight
        if r["status"] == "approved":
            actual = r["new_weight"] if r["new_weight"] is not None else actual

        # shadow: every proposal applied, re-derived from the SHADOW prior
        if _is_param(axis):
            shadow = _param_delta(score, shadow, n)["new_value"]
        else:
            shadow = compute_delta(score, shadow, n, cfg)["new_weight"]

        # guarded: same, then clamped by the cumulative band
        if _is_param(axis):
            want = _param_delta(score, guarded, n)["new_value"]
        else:
            want = compute_delta(score, guarded, n, cfg)["new_weight"]
        lo, hi, base = _band_bounds(axis, guarded, guarded_hist)
        got = min(max(want, lo), hi)
        if abs(got - want) > 1e-9:
            band_binds += 1
        guarded = got
        try:
            ts = datetime.strptime(created[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
            guarded_hist.append((ts.replace(tzinfo=timezone.utc), guarded))
        except ValueError:
            pass

        steps.append({"created_at": created, "status": r["status"], "n": n,
                      "actual": actual, "shadow": shadow, "guarded": guarded,
                      "band_bound": abs(got - want) > 1e-9})

    return {"axis": axis, "n_rows": len(rows), "start": start,
            "actual": actual, "shadow": shadow, "guarded": guarded,
            "band_binds": band_binds, "truncated": truncated, "steps": steps}


def report(db_path: str = None) -> str:
    """Human-readable autonomy readiness summary across all axes."""
    import kairos_log_db as kdb
    conn = sqlite3.connect(db_path or kdb.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        axes = [r[0] for r in conn.execute(
            "SELECT DISTINCT axis FROM axis_weight_history ORDER BY axis")]
    finally:
        conn.close()

    out = ["SHADOW AUTONOMY REPLAY",
           "  actual  = supervised reality (approvals only)",
           "  shadow  = full autonomy, no guards",
           "  guarded = full autonomy WITH the cumulative band",
           ""]
    for axis in axes:
        r = replay_axis(axis, db_path)
        if not r["n_rows"]:
            continue
        drift_s = r["shadow"] - r["actual"]
        drift_g = r["guarded"] - r["actual"]
        out.append(f"{axis}   ({r['n_rows']} runs, start {r['start']:+.4f})")
        if r.get("truncated"):
            out.append(f"   [param history truncated at {PARAM_CONVENTION_CUTOFF} "
                       f"- earlier rows use a different sign convention]")
        out.append(f"   actual  {r['actual']:+.4f}")
        out.append(f"   shadow  {r['shadow']:+.4f}   drift vs actual {drift_s:+.4f}")
        out.append(f"   guarded {r['guarded']:+.4f}   drift vs actual {drift_g:+.4f}"
                   f"   [band bound {r['band_binds']}x]")
        contained = abs(drift_s) - abs(drift_g)
        if contained > 1e-9:
            out.append(f"   -> guards contained {contained:+.4f} of drift")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
