"""
Kairos Axis Weights — Phase B2b: exit_timing (bidirectional) +
reallocation_aggressiveness, alongside the original conviction_calibration.

Computes each learning axis from CLOSED-trade data and runs it through a HUMAN
APPROVAL GATE. This is the second half of the Arbiter→council feedback loop, but
it is deliberately INERT with respect to trading:

  * compute  — measure how well conviction scores predicted realized P&L.
  * propose  — write a 'proposed' weight update to axis_weight_history and post a
               Slack note; the stored axis_weights.weight is NOT touched.
  * approve  — a human applies the proposal: axis_weights.weight is updated and
               OBSERVED. Nothing reads that weight into any reasoning prompt yet.
  * reject   — discard the proposal; weight unchanged.

The sign convention matches axis_weights.positive_means for this axis:
    Positive = conviction has been OVER-confident (high-conviction trades
    underperformed) → correction is to discount conviction.
Spearman rho between conviction and pnl_pct is POSITIVE when higher conviction
tracks higher P&L (well-calibrated). We therefore use computed_score = -rho, so a
positive computed_score means miscalibration in the "over-confident" direction.

CLI:
    python3 kairos_axis_weights.py --compute        # dry-run; writes/posts nothing
    python3 kairos_axis_weights.py --propose        # write proposed row + Slack
    python3 kairos_axis_weights.py --review         # list pending proposals
    python3 kairos_axis_weights.py --approve <ID> [--by name]
    python3 kairos_axis_weights.py --reject  <ID> [--by name]

Phase B2b scope: compute + propose only. No change to kairos_reason.py,
kairos_arbiter.py, the scheduler, or any trading path — weights remain observed,
not injected, until Phase C.
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DB_PATH = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "kairos_config.json")

AXIS = "conviction_calibration"       # original Phase B1 axis (kept as a default)
AXES = ("conviction_calibration", "exit_timing", "reallocation_aggressiveness")
DEFAULT_AXIS = "exit_timing"          # CLI default for --compute / --propose
ARBITER_CHANNEL = "#kairos-arbiter"   # same channel the Arbiter posts to

# Axes the weekly auto-propose loop (propose_all / --propose-all) refreshes each
# run. exit_timing is live; conviction_calibration is shelved; add
# "reallocation_aggressiveness" here once its compute is validated. Every axis
# still passes through the SAME min_sample gate and the SAME human approval gate —
# auto-propose only writes 'proposed' rows; it NEVER approves or changes a weight.
AUTO_PROPOSE_AXES = ["exit_timing"]

# B2b signed exit-timing scale: per-trade net error (points) that maps to a
# full-strength score of ±1.0. Widened from the old one-sided 5.0 so neither
# pole — too-late give-back nor too-early post-exit run-up — saturates the score.
GIVEBACK_SCALE = 10.0
REALLOC_SCALE = 10.0

# Defaults if the config block is missing; real values come from kairos_config.json.
_DEFAULT_CFG = {
    "learning_rate": 0.3,
    "per_run_cap": 0.15,
    "min_sample": 20,
    "weight_bound": 1.0,
}


# ── Helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _today_et() -> str:
    """ET calendar date, to match the Arbiter's run_id convention."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        from datetime import timedelta
        return (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%d")


def _run_id() -> str:
    return f"{_today_et()}_b1"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_config() -> dict:
    """Read the axis_weights block from kairos_config.json (with defaults)."""
    cfg = dict(_DEFAULT_CFG)
    try:
        with open(CONFIG_PATH) as f:
            block = (json.load(f) or {}).get("axis_weights", {})
        for k in cfg:
            if k in block and block[k] is not None:
                cfg[k] = block[k]
    except (json.JSONDecodeError, IOError):
        pass
    return cfg


def _ml_connect_ro() -> sqlite3.Connection:
    """Open the ML outcomes DB strictly read-only."""
    conn = sqlite3.connect(f"file:{ML_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ── Compute: conviction_calibration statistic ────────────────────────

def compute_conviction_calibration() -> dict:
    """Measure conviction→P&L calibration from closed trades.

    Returns {axis, computed_score, sample_size, evidence}. evidence carries the
    Spearman rho, the conviction-bucket table, and the contributing trade_ids.
    """
    import pandas as pd

    sql = """
        SELECT t.trade_id            AS trade_id,
               MAX(p.conviction_score) AS conviction,
               t.pnl_pct             AS pnl_pct
        FROM trade_outcomes t
        JOIN thesis_predictions p ON p.decision_id = t.trade_id
        WHERE t.timestamp_exit IS NOT NULL
          AND p.conviction_score IS NOT NULL
          AND t.pnl_pct IS NOT NULL
        GROUP BY t.trade_id
    """
    conn = _ml_connect_ro()
    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()

    n = len(rows)
    trade_ids = [r["trade_id"] for r in rows]

    if n == 0:
        rho = 0.0
    else:
        df = pd.DataFrame(rows)
        rho = df["conviction"].corr(df["pnl_pct"], method="spearman")
        # NaN when there is insufficient variance (e.g. all-equal conviction).
        if rho is None or pd.isna(rho):
            rho = 0.0
        rho = float(rho)

    computed_score = _clamp(-rho, -1.0, 1.0)

    # Conviction buckets: high >=7, mid 4-6, low <4.
    def _bucket(score):
        if score is None:
            return None
        if score >= 7:
            return "high (>=7)"
        if score >= 4:
            return "mid (4-6)"
        return "low (<4)"

    buckets = []
    for label in ("high (>=7)", "mid (4-6)", "low (<4)"):
        members = [r for r in rows if _bucket(r["conviction"]) == label]
        bn = len(members)
        if bn:
            avg_pnl = sum(r["pnl_pct"] for r in members) / bn
            win_rate = sum(1 for r in members if r["pnl_pct"] > 0) / bn
        else:
            avg_pnl = None
            win_rate = None
        buckets.append({
            "bucket": label,
            "n": bn,
            "avg_pnl_pct": round(avg_pnl, 4) if avg_pnl is not None else None,
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
        })

    evidence = {
        "rho": round(rho, 6),
        "computed_score": round(computed_score, 6),
        "n": n,
        "buckets": buckets,
        "trade_ids": trade_ids,
    }
    return {
        "axis": AXIS,
        "computed_score": computed_score,
        "sample_size": n,
        "evidence": evidence,
    }


# ── Compute: exit_timing statistic ───────────────────────────────────

def compute_exit_timing() -> dict:
    """Signed, bidirectional exit-timing score from the persisted B2a features.

    Reads the post-hoc features written by kairos_outcome_features.py onto
    trade_outcomes. Each MATURED closed trade carries two real, opposite
    quantities measured against its own exit:

      * give_back_pct      — edge surrendered by holding PAST the in-hold peak
                             (the too-LATE pole; + ⇒ we exited late).
      * post_exit_peak_pct — favorable move still available AFTER the exit, within
                             the matured post-exit window (the too-EARLY pole;
                             + ⇒ we exited early). NULL until the window elapses.

    Per trade we net the two poles directly (no flooring):
        per_trade_error = give_back_pct − post_exit_peak_pct
        (+ held too long / gave back ; − left post-exit upside on the table)
    and average over the matured set (trades carrying BOTH features):
        mean_error     = mean(per_trade_error)
        computed_score = clamp(mean_error / GIVEBACK_SCALE, −1.0, +1.0)
    Positive ⇒ exiting too LATE (correction: exit earlier; matches
    exit_timing.positive_means); negative ⇒ too EARLY (correction: hold longer).
    """
    sql = """
        SELECT trade_id, give_back_pct, post_exit_peak_pct
        FROM trade_outcomes
        WHERE timestamp_exit IS NOT NULL
    """
    conn = _ml_connect_ro()
    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()

    # Matured set: both poles present. post_exit_peak_pct is NULL until the
    # post-exit window matures — a feature-filled trade still missing it counts
    # as pending, not as evidence of either direction. (Closed trades that never
    # had outcome features computed at all — give_back_pct NULL — are out of
    # scope here, neither matured nor pending.)
    matured = [r for r in rows
               if r["give_back_pct"] is not None
               and r["post_exit_peak_pct"] is not None]
    n_pending = sum(1 for r in rows
                    if r["give_back_pct"] is not None
                    and r["post_exit_peak_pct"] is None)

    n = len(matured)
    if n:
        errors = [r["give_back_pct"] - r["post_exit_peak_pct"] for r in matured]
        mean_error = sum(errors) / n
        mean_giveback = sum(r["give_back_pct"] for r in matured) / n
        mean_post_exit_peak = sum(r["post_exit_peak_pct"] for r in matured) / n
    else:
        mean_error = mean_giveback = mean_post_exit_peak = 0.0

    computed_score = _clamp(mean_error / GIVEBACK_SCALE, -1.0, 1.0)
    trade_ids = sorted(r["trade_id"] for r in matured)

    evidence = {
        "mean_error_pp": round(mean_error, 6),
        "mean_giveback_pp": round(mean_giveback, 6),
        "mean_post_exit_peak_pp": round(mean_post_exit_peak, 6),
        "n_matured": n,
        "n_pending": n_pending,
        "trade_ids": trade_ids,
    }
    return {
        "axis": "exit_timing",
        "computed_score": computed_score,
        "sample_size": n,
        "evidence": evidence,
    }


# ── Compute: reallocation_aggressiveness statistic ───────────────────

def compute_reallocation_aggressiveness() -> dict:
    """Bidirectional rotation-discipline score from thesis decay + B2a features.

    exit_reason-based reallocation tagging is unusable today (position_exits is
    almost entirely NULL — a downstream symptom of the write_trade_close
    decision_id bug), so this keys off thesis_conditions_intact at the LAST
    checkpoint of each closed trade instead:

      * too-EAGER (+): thesis was still INTACT at exit yet the price ran up after
        we sold (post_exit_peak_pct) — we rotated out of a live thesis early.
      * too-STICKY (−): thesis had DECAYED (intact = 0) yet we kept holding and
        surrendered the peak (give_back_pct) — we overstayed a dead thesis.

    Both poles floored at 0. net = mean(eager) − mean(sticky);
    score = clamp(net / SCALE, −1, +1). Positive ⇒ rotating too EAGERLY
    (correction: hold longer before rotating); negative ⇒ too STICKY (correction:
    rotate sooner). Matches axis_weights.positive_means.

    NOTE (data caveat): the sticky pole is thin and reallocation exits are
    under-logged until the write_trade_close decision_id fix lands; treat a
    proposal here as provisional and lean on the human gate.
    """
    sql = """
        WITH last_ck AS (
            SELECT c.decision_id,
                   c.thesis_conditions_intact AS intact_at_exit
            FROM thesis_checkpoints c
            JOIN (
                SELECT decision_id, MAX(checkpoint_day) AS md
                FROM thesis_checkpoints GROUP BY decision_id
            ) m ON m.decision_id = c.decision_id AND m.md = c.checkpoint_day
        )
        SELECT t.trade_id, t.give_back_pct, t.post_exit_peak_pct,
               l.intact_at_exit
        FROM trade_outcomes t
        JOIN last_ck l ON l.decision_id = t.trade_id
        WHERE t.timestamp_exit IS NOT NULL
          AND t.features_filled_at IS NOT NULL
          AND l.intact_at_exit IS NOT NULL
    """
    conn = _ml_connect_ro()
    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()

    # too-EAGER: intact thesis at exit, missed post-exit upside (matured only).
    eager = [(r["trade_id"], max(r["post_exit_peak_pct"], 0.0)) for r in rows
             if r["intact_at_exit"] == 1 and r["post_exit_peak_pct"] is not None]
    # too-STICKY: decayed thesis at exit, peak surrendered (give-back).
    sticky = [(r["trade_id"], max(r["give_back_pct"], 0.0)) for r in rows
              if r["intact_at_exit"] == 0 and r["give_back_pct"] is not None]

    mean_eager = (sum(v for _, v in eager) / len(eager)) if eager else 0.0
    mean_sticky = (sum(v for _, v in sticky) / len(sticky)) if sticky else 0.0
    net = mean_eager - mean_sticky

    computed_score = _clamp(net / REALLOC_SCALE, -1.0, 1.0)
    trade_ids = sorted({tid for tid, _ in eager} | {tid for tid, _ in sticky})
    n = len(trade_ids)

    evidence = {
        "mean_eager_pp": round(mean_eager, 6),
        "mean_sticky_pp": round(mean_sticky, 6),
        "net_pp": round(net, 6),
        "n_eager": len(eager),
        "n_sticky": len(sticky),
        "scale": REALLOC_SCALE,
        "trade_ids": trade_ids,
    }
    return {
        "axis": "reallocation_aggressiveness",
        "computed_score": computed_score,
        "sample_size": n,
        "evidence": evidence,
    }


# ── Axis dispatch ────────────────────────────────────────────────────

def compute_axis(axis: str) -> dict:
    """Route to the right compute fn for the requested axis."""
    if axis == "conviction_calibration":
        return compute_conviction_calibration()
    if axis == "exit_timing":
        return compute_exit_timing()
    if axis == "reallocation_aggressiveness":
        return compute_reallocation_aggressiveness()
    raise ValueError(f"unknown axis {axis!r} (expected one of {AXES})")


# ── Delta math (shared by --compute preview and --propose) ───────────

def _get_prior_weight(conn: sqlite3.Connection, axis: str) -> float:
    row = conn.execute(
        "SELECT weight FROM axis_weights WHERE axis = ?", (axis,)
    ).fetchone()
    return float(row["weight"]) if row and row["weight"] is not None else 0.0


def compute_delta(computed_score: float, prior_weight: float,
                  sample_size: int, cfg: dict) -> dict:
    """Smoothed tracker toward the current statistic, with sample gate + caps.

    Returns {proposed_delta, new_weight, gated}.
    """
    if sample_size < cfg["min_sample"]:
        # Gated: not enough data to move the weight, but we still record a row.
        return {"proposed_delta": 0.0, "new_weight": prior_weight, "gated": True}

    raw = cfg["learning_rate"] * (computed_score - prior_weight)
    proposed_delta = _clamp(raw, -cfg["per_run_cap"], cfg["per_run_cap"])
    new_weight = _clamp(prior_weight + proposed_delta,
                        -cfg["weight_bound"], cfg["weight_bound"])
    return {"proposed_delta": proposed_delta, "new_weight": new_weight,
            "gated": False}


# ── Propose: write a 'proposed' row (does NOT touch axis_weights) ─────

def propose_update(run_id: str | None = None, axis: str = DEFAULT_AXIS) -> dict:
    """Compute the statistic and write a 'proposed' weight update.

    Supersedes any existing 'proposed' row for this axis, then inserts the new
    one. Does NOT change axis_weights.weight. Returns the proposal dict including
    history_id. The EMA tracker / gate / caps are axis-agnostic — only the
    compute step differs (dispatched by axis).
    """
    from kairos_log_db import get_connection, init_db
    init_db()  # ensure axis_weight_history + axis_weights exist (idempotent)

    cfg = load_config()
    run_id = run_id or _run_id()
    result = compute_axis(axis)
    computed_score = result["computed_score"]
    sample_size = result["sample_size"]

    conn = get_connection()
    try:
        prior_weight = _get_prior_weight(conn, axis)
        delta = compute_delta(computed_score, prior_weight, sample_size, cfg)

        # Supersede any still-pending proposal for this axis.
        conn.execute(
            "UPDATE axis_weight_history SET status = 'superseded' "
            "WHERE axis = ? AND status = 'proposed'",
            (axis,),
        )
        cur = conn.execute(
            "INSERT INTO axis_weight_history "
            "(axis, run_id, computed_score, sample_size, evidence, prior_weight, "
            " proposed_delta, new_weight, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?)",
            (
                axis, run_id, computed_score, sample_size,
                json.dumps(result["evidence"]), prior_weight,
                delta["proposed_delta"], delta["new_weight"], _now(),
            ),
        )
        conn.commit()
        history_id = cur.lastrowid
    finally:
        conn.close()

    return {
        "history_id": history_id,
        "axis": axis,
        "run_id": run_id,
        "computed_score": computed_score,
        "sample_size": sample_size,
        "prior_weight": prior_weight,
        "proposed_delta": delta["proposed_delta"],
        "new_weight": delta["new_weight"],
        "gated": delta["gated"],
        "evidence": result["evidence"],
    }


# ── Auto-propose: refresh proposals for every auto axis (weekly loop) ─

def _has_compute(axis: str) -> bool:
    """True if a compute function is wired for this axis (else skip it)."""
    try:
        compute_axis(axis)
        return True
    except ValueError:
        return False
    except Exception:
        # A real compute error (e.g. DB read) still counts as 'computable'; the
        # per-axis try/except in propose_all surfaces it without skipping silently.
        return True


def propose_all(run_id: str | None = None) -> dict:
    """Write a fresh 'proposed' row for each AUTO_PROPOSE_AXES axis, one Slack note.

    For every axis in AUTO_PROPOSE_AXES that has a compute function, call the
    existing propose_update (supersede-then-insert, min_sample-gated). Collect the
    per-axis proposal summaries and post ONE combined note to ARBITER_CHANNEL.

    This NEVER approves anything and NEVER changes a live axis_weights.weight — it
    only writes 'proposed' rows for the human gate. Each axis is isolated in its
    own try/except so one failure cannot block the others. Returns
    {run_id, proposals: [...], errors: [...], slack_posted: bool}.
    """
    run_id = run_id or f"{_today_et()}_weekly"
    proposals: list[dict] = []
    errors: list[dict] = []

    for axis in AUTO_PROPOSE_AXES:
        try:
            if not _has_compute(axis):
                errors.append({"axis": axis, "error": "no compute function — skipped"})
                continue
            p = propose_update(run_id=run_id, axis=axis)
            proposals.append({
                "axis": p["axis"],
                "history_id": p["history_id"],
                "computed_score": p["computed_score"],
                "sample_size": p["sample_size"],
                "prior_weight": p["prior_weight"],
                "proposed_delta": p["proposed_delta"],
                "new_weight": p["new_weight"],
                "gated": p["gated"],
            })
        except Exception as exc:
            errors.append({"axis": axis, "error": str(exc)})
            print(f"  propose_all: axis {axis} failed: {exc}", file=sys.stderr)

    slack_text = _format_slack_propose_all(run_id, proposals, errors)
    slack_posted = _post_slack(slack_text)
    return {
        "run_id": run_id,
        "proposals": proposals,
        "errors": errors,
        "slack_text": slack_text,
        "slack_posted": slack_posted,
    }


def _format_slack_propose_all(run_id: str, proposals: list[dict],
                              errors: list[dict]) -> str:
    """One combined Slack summary for a propose_all run (human gate unchanged)."""
    lines = [
        f":robot_face: *Weekly axis-weight proposals* — run {run_id}",
        "_Auto-generated from closed-trade outcomes. Nothing approved — these are "
        "'proposed' rows only._",
    ]
    for p in proposals:
        if p["gated"]:
            detail = (f"gated — insufficient data "
                      f"(sample {p['sample_size']} < min_sample); no change")
        elif abs(p["proposed_delta"]) < 1e-9:
            detail = (f"no change (weight stays {p['new_weight']:+.4f}; "
                      f"score {p['computed_score']:+.4f}, n={p['sample_size']})")
        else:
            detail = (f"{p['prior_weight']:+.4f} → *{p['new_weight']:+.4f}* "
                      f"(Δ {p['proposed_delta']:+.4f}; score {p['computed_score']:+.4f}, "
                      f"n={p['sample_size']})  [id {p['history_id']}]")
        lines.append(f"  • `{p['axis']}`: {detail}")
    for e in errors:
        lines.append(f"  • `{e['axis']}`: error — {e['error']}")
    if not proposals and not errors:
        lines.append("  • (no axes configured for auto-propose)")
    lines.append("Review with `--review`; approve at the terminal.")
    return "\n".join(lines)


# ── Apply a human decision (the approval gate) ───────────────────────

def apply_decision(history_id: int, decision: str, decided_by: str) -> dict:
    """Approve or reject a pending proposal.

    approve → set axis_weights.weight = new_weight (OBSERVED only, not injected
              into any reasoning prompt), mark the row approved, post Slack.
    reject  → mark the row rejected; axis_weights is untouched.
    Raises ValueError if the row is missing or not in 'proposed' status, or if
    the decision string is unknown.
    """
    decision = decision.lower().strip()
    if decision not in ("approve", "reject"):
        raise ValueError(f"unknown decision {decision!r} (expected approve/reject)")

    from kairos_log_db import get_connection, init_db
    init_db()

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM axis_weight_history WHERE id = ?", (history_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"axis_weight_history id {history_id} not found")
        if row["status"] != "proposed":
            raise ValueError(
                f"id {history_id} is '{row['status']}', not 'proposed' — "
                f"cannot {decision}")

        now = _now()
        if decision == "approve":
            conn.execute(
                "UPDATE axis_weights SET weight = ?, updated_at = ? WHERE axis = ?",
                (row["new_weight"], now, row["axis"]),
            )
            conn.execute(
                "UPDATE axis_weight_history "
                "SET status = 'approved', decided_at = ?, decided_by = ? WHERE id = ?",
                (now, decided_by, history_id),
            )
            conn.commit()
            outcome = {
                "axis": row["axis"], "status": "approved",
                "new_weight": row["new_weight"], "decided_by": decided_by,
            }
            _post_slack(
                f":white_check_mark: *Axis weight approved* — `{row['axis']}`\n"
                f"weight {row['prior_weight']:+.4f} → *{row['new_weight']:+.4f}* "
                f"(Δ {row['proposed_delta']:+.4f}) by {decided_by}\n"
                f"Observed only — not yet injected into any reasoning prompt."
            )
        else:  # reject
            conn.execute(
                "UPDATE axis_weight_history "
                "SET status = 'rejected', decided_at = ?, decided_by = ? WHERE id = ?",
                (now, decided_by, history_id),
            )
            conn.commit()
            outcome = {
                "axis": row["axis"], "status": "rejected",
                "decided_by": decided_by,
            }
    finally:
        conn.close()
    return outcome


# ── Slack + formatting ───────────────────────────────────────────────

def _post_slack(text: str) -> bool:
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from kairos_alerts import post_message
        return post_message(ARBITER_CHANNEL, text)
    except Exception as exc:
        print(f"  Slack post failed: {exc}", file=sys.stderr)
        return False


def _bucket_table_lines(buckets: list[dict]) -> list[str]:
    lines = [f"  {'bucket':<12} {'n':>4} {'avg_pnl_pct':>12} {'win_rate':>9}"]
    for b in buckets:
        avg = "—" if b["avg_pnl_pct"] is None else f"{b['avg_pnl_pct']:+.2f}"
        win = "—" if b["win_rate"] is None else f"{b['win_rate']:.0%}"
        lines.append(f"  {b['bucket']:<12} {b['n']:>4} {avg:>12} {win:>9}")
    return lines


def _format_slack_proposal(p: dict) -> str:
    ev = p["evidence"]
    gated = " _(GATED: sample below min_sample → Δ forced to 0)_" if p["gated"] else ""
    lines = [
        f":balance_scale: *Axis weight proposal — `{p['axis']}`*  |  run {p['run_id']}"
    ]
    # Axis-specific score explanation.
    if p["axis"] == "conviction_calibration":
        lines += [
            f"Spearman rho (conviction vs realized P&L): *{ev['rho']:+.4f}*",
            f"computed_score (= -rho, + = over-confident): *{p['computed_score']:+.4f}*",
        ]
    elif p["axis"] == "exit_timing":
        lines += [
            f"mean per-trade error: *{ev['mean_error_pp']:+.2f}pp*  "
            f"(too-late give-back {ev['mean_giveback_pp']:.2f}pp − too-early "
            f"post-exit run-up {ev['mean_post_exit_peak_pp']:.2f}pp; scale "
            f"{GIVEBACK_SCALE:.0f}pp → ±1.0)",
            f"computed_score (+ = too LATE / exit earlier, − = too EARLY / hold "
            f"longer): *{p['computed_score']:+.4f}*",
        ]
    elif p["axis"] == "reallocation_aggressiveness":
        lines += [
            f"net regret: *{ev['net_pp']:+.2f}pp*  "
            f"(too-eager missed run-up {ev['mean_eager_pp']:.2f}pp − too-sticky "
            f"give-back {ev['mean_sticky_pp']:.2f}pp; scale {ev['scale']:.0f}pp → "
            f"±1.0)",
            f"computed_score (+ = too EAGER / hold longer before rotating, − = "
            f"too STICKY / rotate sooner): *{p['computed_score']:+.4f}*",
        ]
    lines.append(f"sample_size: *{p['sample_size']}*{gated}")
    lines.append(
        f"prior weight: {p['prior_weight']:+.4f}  →  proposed Δ "
        f"*{p['proposed_delta']:+.4f}*  →  resulting weight *{p['new_weight']:+.4f}*")
    # Axis-specific evidence block.
    if p["axis"] == "conviction_calibration":
        lines += ["Conviction buckets:", "```", *_bucket_table_lines(ev["buckets"]), "```"]
    elif p["axis"] == "exit_timing":
        lines += [
            "Exit-timing evidence (matured set):",
            "```",
            f"  too-late  give-back:    {ev['mean_giveback_pp']:.2f}pp",
            f"  too-early run-up:       {ev['mean_post_exit_peak_pp']:.2f}pp",
            f"  mean per-trade error:   {ev['mean_error_pp']:+.2f}pp  (n_matured={ev['n_matured']})",
            f"  pending (no post-peak): {ev['n_pending']}",
            "```",
        ]
    elif p["axis"] == "reallocation_aggressiveness":
        lines += [
            "Reallocation evidence:",
            "```",
            f"  too-eager  run-up:     {ev['mean_eager_pp']:.2f}pp  (n={ev['n_eager']}, intact@exit)",
            f"  too-sticky give-back:  {ev['mean_sticky_pp']:.2f}pp  (n={ev['n_sticky']}, decayed@exit)",
            f"  net regret:            {ev['net_pp']:+.2f}pp",
            "```",
        ]
    lines.append(
        f"Pending approval (id {p['history_id']}). Review with "
        f"`--review`; apply with `--approve {p['history_id']}` / "
        f"`--reject {p['history_id']}`.")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

def _cli_compute(axis: str) -> int:
    cfg = load_config()
    result = compute_axis(axis)
    ev = result["evidence"]
    # Read prior weight (read-only intent; no writes on --compute).
    from kairos_log_db import get_connection, init_db
    init_db()
    conn = get_connection()
    try:
        prior = _get_prior_weight(conn, axis)
    finally:
        conn.close()
    delta = compute_delta(result["computed_score"], prior,
                          result["sample_size"], cfg)

    print(f"  Axis: {axis}  (DRY RUN — nothing written, nothing posted)")
    if axis == "conviction_calibration":
        print(f"  Spearman rho (conviction vs pnl_pct): {ev['rho']:+.6f}")
        print(f"  computed_score (= -rho):              {result['computed_score']:+.6f}")
        print(f"  sample_size:                          {result['sample_size']}"
              f"  (min_sample={cfg['min_sample']})")
        print("  Conviction buckets:")
        for line in _bucket_table_lines(ev["buckets"]):
            print("  " + line)
    elif axis == "exit_timing":
        print(f"  mean give-back (too-late pole):       {ev['mean_giveback_pp']:+.6f}pp")
        print(f"  mean post-exit peak (too-early pole): {ev['mean_post_exit_peak_pp']:+.6f}pp")
        print(f"  mean per-trade error (late − early):  {ev['mean_error_pp']:+.6f}pp")
        print(f"  computed_score (= error/{GIVEBACK_SCALE:.0f}, + late / − early): "
              f"{result['computed_score']:+.6f}")
        print(f"  sample_size (n_matured):              {result['sample_size']}"
              f"  (min_sample={cfg['min_sample']})")
        print(f"  n_pending (post-exit window not matured): {ev['n_pending']}")
    elif axis == "reallocation_aggressiveness":
        print(f"  too-eager missed run-up (intact@exit):{ev['mean_eager_pp']:+.6f}pp (n={ev['n_eager']})")
        print(f"  too-sticky give-back (decayed@exit):  {ev['mean_sticky_pp']:+.6f}pp (n={ev['n_sticky']})")
        print(f"  net regret (eager − sticky):          {ev['net_pp']:+.6f}pp")
        print(f"  computed_score (= net/{ev['scale']:.0f}, + eager / − sticky): "
              f"{result['computed_score']:+.6f}")
        print(f"  sample_size (either pole):            {result['sample_size']}"
              f"  (min_sample={cfg['min_sample']})")
    if delta["gated"]:
        print(f"  would-be Δ: 0.000000 (GATED — sample < min_sample)")
    print(f"  prior weight:   {prior:+.6f}")
    print(f"  would-be Δ:     {delta['proposed_delta']:+.6f}  "
          f"(lr={cfg['learning_rate']}, cap=±{cfg['per_run_cap']})")
    print(f"  would-be weight:{delta['new_weight']:+.6f}  "
          f"(bound=±{cfg['weight_bound']})")
    return 0


def _cli_propose(axis: str) -> int:
    p = propose_update(axis=axis)
    slack_text = _format_slack_proposal(p)
    posted = _post_slack(slack_text)
    print("  Proposed row written to axis_weight_history "
          f"(id {p['history_id']}, status 'proposed').")
    print(f"  axis_weights.weight is UNCHANGED (proposal does not apply).")
    print(f"  Slack post to {ARBITER_CHANNEL}: {'OK' if posted else 'FAILED'}")
    print("\n  ----- Slack message text -----")
    print(slack_text)
    return 0


def _cli_propose_all() -> int:
    result = propose_all()
    n_prop = len(result["proposals"])
    n_err = len(result["errors"])
    print(f"  propose_all run {result['run_id']}: "
          f"{n_prop} proposal(s), {n_err} error(s).")
    for p in result["proposals"]:
        print(f"    {p['axis']}: id {p['history_id']}, score {p['computed_score']:+.4f}, "
              f"n={p['sample_size']}, prior {p['prior_weight']:+.4f} → "
              f"new {p['new_weight']:+.4f} (Δ {p['proposed_delta']:+.4f})"
              f"{' [GATED]' if p['gated'] else ''}")
    for e in result["errors"]:
        print(f"    {e['axis']}: ERROR — {e['error']}")
    print("  axis_weights.weight is UNCHANGED for every axis (nothing approved).")
    print(f"  Slack post to {ARBITER_CHANNEL}: "
          f"{'OK' if result['slack_posted'] else 'FAILED'}")
    print("\n  ----- Slack message text -----")
    print(result["slack_text"])
    return 0


def _cli_review() -> int:
    from kairos_log_db import get_connection, init_db
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM axis_weight_history WHERE status = 'proposed' "
            "ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        print("  No pending ('proposed') axis weight proposals.")
        return 0
    print(f"  {len(rows)} pending proposal(s):\n")
    for r in rows:
        ev = json.loads(r["evidence"]) if r["evidence"] else {}
        print(f"  ── id {r['id']}  axis={r['axis']}  run={r['run_id']}")
        print(f"     computed_score={r['computed_score']:+.6f}  "
              f"sample_size={r['sample_size']}")
        print(f"     prior_weight={r['prior_weight']:+.6f}  "
              f"proposed_delta={r['proposed_delta']:+.6f}  "
              f"new_weight={r['new_weight']:+.6f}")
        print(f"     created_at={r['created_at']}")
        if ev:
            if r["axis"] == "conviction_calibration":
                print(f"     rho={ev.get('rho')}  buckets:")
                for line in _bucket_table_lines(ev.get("buckets", [])):
                    print("     " + line)
            elif r["axis"] == "exit_timing":
                print(f"     mean_error={ev.get('mean_error_pp')}pp  "
                      f"too_late_giveback={ev.get('mean_giveback_pp')}pp  "
                      f"too_early_runup={ev.get('mean_post_exit_peak_pp')}pp")
                print(f"     n_matured={ev.get('n_matured')}  "
                      f"n_pending={ev.get('n_pending')}")
            elif r["axis"] == "reallocation_aggressiveness":
                print(f"     net={ev.get('net_pp')}pp  "
                      f"too_eager_runup={ev.get('mean_eager_pp')}pp (n={ev.get('n_eager')})  "
                      f"too_sticky_giveback={ev.get('mean_sticky_pp')}pp (n={ev.get('n_sticky')})")
                print(f"     scale={ev.get('scale')}")
            tids = ev.get("trade_ids", [])
            print(f"     contributing trade_ids ({len(tids)}): "
                  f"{', '.join(str(t)[:8] for t in tids[:12])}"
                  f"{' …' if len(tids) > 12 else ''}")
        print()
    print(f"  Apply with --approve <ID> / --reject <ID>.")
    return 0


def _cli_decide(history_id: int, decision: str, decided_by: str) -> int:
    try:
        outcome = apply_decision(history_id, decision, decided_by)
    except ValueError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  {decision.upper()} applied to id {history_id}: {outcome}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kairos axis weights — conviction_calibration, exit_timing, "
                    "reallocation_aggressiveness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compute", action="store_true",
                       help="Dry-run: print the statistic + would-be delta; write/post nothing")
    group.add_argument("--propose", action="store_true",
                       help="Compute, write a proposed row, post Slack note")
    group.add_argument("--propose-all", action="store_true",
                       help="Auto-propose every AUTO_PROPOSE_AXES axis (weekly loop); "
                            "one combined Slack note. Approves nothing.")
    group.add_argument("--review", action="store_true",
                       help="List all pending ('proposed') proposals with evidence")
    group.add_argument("--approve", type=int, metavar="ID",
                       help="Approve a pending proposal: apply its new_weight")
    group.add_argument("--reject", type=int, metavar="ID",
                       help="Reject a pending proposal: leave the weight unchanged")
    parser.add_argument("--axis", choices=AXES, default=DEFAULT_AXIS,
                        help="Axis for --compute / --propose (default: exit_timing). "
                             "Ignored by --review/--approve/--reject (they act by id).")
    parser.add_argument("--by", default=os.environ.get("USER", "operator"),
                        help="Name recorded as decided_by for approve/reject")
    args = parser.parse_args()

    if args.compute:
        return _cli_compute(args.axis)
    if args.propose:
        return _cli_propose(args.axis)
    if args.propose_all:
        return _cli_propose_all()
    if args.review:
        return _cli_review()
    if args.approve is not None:
        return _cli_decide(args.approve, "approve", args.by)
    if args.reject is not None:
        return _cli_decide(args.reject, "reject", args.by)
    return 1


if __name__ == "__main__":
    sys.exit(main())
