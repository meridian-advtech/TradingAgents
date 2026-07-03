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

# ── Exit-parameter proposals (axis = 'param:<dotted.config.path>') ────
# The Arbiter may also propose changes to a SMALL, HARD-CODED whitelist of
# exit-engine parameters, reusing the axis_weight_history table and the
# proposed/approved/rejected/superseded lifecycle unchanged. A param proposal
# stores axis='param:<dotted.path>', prior_weight=the current config value, and
# new_weight=the proposed value. Approval writes the value into kairos_config.json
# at that dotted path (timestamped backup + json re-validate) and NEVER touches
# axis_weights. The human gate is mandatory: params are deliberately excluded from
# AUTO_PROPOSE_AXES and nothing here ever auto-applies.
PARAM_PREFIX = "param:"

# Whitelist: dotted path → (lo_bound, hi_bound). ANY path not listed here is
# refused at both propose and apply time.
PARAM_WHITELIST = {
    "exits.trailing_stop.profit_floor_pp":        (0.5, 3.0),
    "exits.trailing_stop.target_armed.trail_pct": (4.0, 12.0),
}
PARAM_MIN_SAMPLE = 10          # min TRAILING-STOP closes before a param is proposed
PARAM_MAX_CHANGE_FRAC = 0.25   # an approved proposal may move a value at most ±25%
# Give-back (pp) that maps to a full-strength (100% of the allowed ±25%) nudge.
PARAM_GIVEBACK_REF = 15.0


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


# ── Exit-parameter proposals: helpers ────────────────────────────────

def _is_param_axis(axis: str) -> bool:
    return isinstance(axis, str) and axis.startswith(PARAM_PREFIX)


def _param_path(axis: str) -> str | None:
    """Strip the 'param:' prefix; None if axis is not a param proposal."""
    return axis[len(PARAM_PREFIX):] if _is_param_axis(axis) else None


def _get_dotted(cfg: dict, path: str):
    """Return (value, found) for a dotted path into a nested dict."""
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None, False
        cur = cur[key]
    return cur, True


def _set_dotted(cfg: dict, path: str, value) -> None:
    """Set a dotted path into a nested dict, creating intermediate dicts."""
    keys = path.split(".")
    cur = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _current_param_value(path: str) -> float | None:
    """Current numeric value at a dotted config path, or None if absent/non-numeric."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    val, found = _get_dotted(cfg, path)
    if not found or isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


# ── Compute: exit-parameter statistic ────────────────────────────────

def compute_param(path: str) -> dict:
    """Compute exit-parameter evidence from TRAILING-STOP closed trades.

    Whitelisted paths only (raises ValueError otherwise). Reads trade_outcomes
    rows whose exit_reason was tagged 'TRAILING-STOP…' — the trailing-stop
    profit-capture exits now recorded by the exit-metadata plumbing fix. Evidence:
        n, avg_mfe_pct, avg_pnl_pct, avg_give_back_pct,
        round_trips (winners that round-tripped to a loss: mfe>2 and pnl<=0).

    The proposed new value is a bounded, interpretable nudge: give-back severity
    (blended with the round-trip rate) sets the MAGNITUDE, the parameter's role
    sets the DIRECTION (when we are surrendering gains, tighten the trail /
    raise the floor), and the change is clamped to ±PARAM_MAX_CHANGE_FRAC of the
    current value AND to the whitelist bounds. Below PARAM_MIN_SAMPLE the compute
    is gated (no change proposed). Returns
    {axis, path, computed_score, sample_size, evidence, current_value,
     proposed_value, gated}.
    """
    if path not in PARAM_WHITELIST:
        raise ValueError(f"param path {path!r} is not whitelisted")

    sql = """
        SELECT pnl_pct, mfe_pct, give_back_pct
        FROM trade_outcomes
        WHERE exit_reason LIKE 'TRAILING-STOP%'
          AND timestamp_exit IS NOT NULL
    """
    conn = _ml_connect_ro()
    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()

    n = len(rows)

    def _avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else 0.0

    avg_mfe = _avg("mfe_pct")
    avg_pnl = _avg("pnl_pct")
    avg_gb = _avg("give_back_pct")
    round_trips = sum(
        1 for r in rows
        if r["mfe_pct"] is not None and r["mfe_pct"] > 2.0
        and r["pnl_pct"] is not None and r["pnl_pct"] <= 0.0
    )

    lo, hi = PARAM_WHITELIST[path]
    current = _current_param_value(path)
    gated = n < PARAM_MIN_SAMPLE or current is None

    # Give-back severity in [0,1], blended with the round-trip rate. High severity
    # ⇒ we are surrendering gains ⇒ tighten profit capture.
    roundtrip_rate = (round_trips / n) if n else 0.0
    severity = _clamp(avg_gb / PARAM_GIVEBACK_REF, 0.0, 1.0)
    severity = _clamp(0.7 * severity + 0.3 * roundtrip_rate, 0.0, 1.0)

    # Direction: for both whitelisted params, surrendering gains ⇒ tighter capture.
    #   trail_pct       ↓ (narrow the trail)   → direction −1
    #   profit_floor_pp ↑ (lock in more)       → direction +1
    direction = -1.0 if path.endswith("trail_pct") else +1.0
    computed_score = round(direction * severity, 6)

    if gated or current is None:
        proposed_value = current
    else:
        max_step = PARAM_MAX_CHANGE_FRAC * abs(current)
        change = _clamp(direction * severity * max_step, -max_step, max_step)
        proposed_value = round(_clamp(current + change, lo, hi), 4)

    evidence = {
        "n": n,
        "avg_mfe_pct": round(avg_mfe, 4),
        "avg_pnl_pct": round(avg_pnl, 4),
        "avg_give_back_pct": round(avg_gb, 4),
        "round_trips": round_trips,
        "roundtrip_rate": round(roundtrip_rate, 4),
        "severity": round(severity, 4),
        "bounds": [lo, hi],
        "current_value": current,
        "max_change_frac": PARAM_MAX_CHANGE_FRAC,
    }
    return {
        "axis": PARAM_PREFIX + path,
        "path": path,
        "computed_score": computed_score,
        "sample_size": n,
        "evidence": evidence,
        "current_value": current,
        "proposed_value": proposed_value,
        "gated": gated,
    }


# ── Propose: write a 'proposed' param row (does NOT touch config) ─────

def propose_param_update(path: str, run_id: str | None = None) -> dict:
    """Compute a param statistic and write a 'proposed' row (whitelisted only).

    Supersede-then-insert into axis_weight_history under axis='param:<path>', with
    prior_weight=current config value and new_weight=proposed value. Does NOT touch
    kairos_config.json or axis_weights. min_sample-gated: a gated compute still
    records a row (proposed_delta 0) so the evidence is auditable, mirroring the
    weight-axis behavior. Returns the proposal dict (incl. history_id).
    """
    if path not in PARAM_WHITELIST:
        raise ValueError(f"param path {path!r} is not whitelisted — refusing to propose")

    from kairos_log_db import get_connection, init_db
    init_db()  # ensure axis_weight_history exists (idempotent)

    run_id = run_id or f"{_today_et()}_weekly"
    result = compute_param(path)
    axis = result["axis"]
    current = result["current_value"]
    proposed = result["proposed_value"]
    gated = result["gated"]

    if gated or current is None or proposed is None:
        prior_weight = current if current is not None else 0.0
        new_weight = prior_weight
        proposed_delta = 0.0
    else:
        prior_weight = current
        new_weight = proposed
        proposed_delta = round(new_weight - prior_weight, 6)

    conn = get_connection()
    try:
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
                axis, run_id, result["computed_score"], result["sample_size"],
                json.dumps(result["evidence"]), prior_weight,
                proposed_delta, new_weight, _now(),
            ),
        )
        conn.commit()
        history_id = cur.lastrowid
    finally:
        conn.close()

    return {
        "history_id": history_id,
        "axis": axis,
        "path": path,
        "run_id": run_id,
        "computed_score": result["computed_score"],
        "sample_size": result["sample_size"],
        "prior_weight": prior_weight,
        "proposed_delta": proposed_delta,
        "new_weight": new_weight,
        "gated": gated,
        "evidence": result["evidence"],
    }


def propose_all_params(run_id: str | None = None) -> dict:
    """Write a fresh 'proposed' row for every whitelisted exit param, one Slack note.

    The param counterpart to propose_all: iterate PARAM_WHITELIST, propose each
    (min_sample-gated, supersede-then-insert), collect the summaries, and post ONE
    combined note to ARBITER_CHANNEL via the env-first Slack path. Approves nothing
    and writes no config. Each param is isolated in its own try/except so one
    failure cannot block the others. Returns
    {run_id, proposals, errors, slack_text, slack_posted}.
    """
    run_id = run_id or f"{_today_et()}_weekly"
    proposals: list[dict] = []
    errors: list[dict] = []
    for path in PARAM_WHITELIST:
        try:
            p = propose_param_update(path, run_id=run_id)
            proposals.append({k: p[k] for k in (
                "axis", "path", "history_id", "computed_score", "sample_size",
                "prior_weight", "proposed_delta", "new_weight", "gated")})
        except Exception as exc:
            errors.append({"axis": PARAM_PREFIX + path, "error": str(exc)})
            print(f"  propose_all_params: {path} failed: {exc}", file=sys.stderr)

    slack_text = _format_slack_propose_params(run_id, proposals, errors)
    slack_posted = _post_slack(slack_text)
    return {
        "run_id": run_id,
        "proposals": proposals,
        "errors": errors,
        "slack_text": slack_text,
        "slack_posted": slack_posted,
    }


def _format_slack_propose_params(run_id: str, proposals: list[dict],
                                 errors: list[dict]) -> str:
    """One combined Slack summary for a param propose run (human gate unchanged)."""
    lines = [
        f":wrench: *Weekly exit-parameter proposals* — run {run_id}",
        "_From TRAILING-STOP closed-trade outcomes. Nothing applied — 'proposed' "
        "rows only; approval writes kairos_config.json behind the human gate._",
    ]
    for p in proposals:
        n = p["sample_size"]
        if p["gated"]:
            detail = (f"gated — insufficient data (n {n} < {PARAM_MIN_SAMPLE}); "
                      f"no change (stays {p['prior_weight']:g})")
        elif abs(p["proposed_delta"]) < 1e-9:
            detail = (f"no change (stays {p['new_weight']:g}; "
                      f"score {p['computed_score']:+.4f}, n={n})")
        else:
            detail = (f"{p['prior_weight']:g} → *{p['new_weight']:g}* "
                      f"(Δ {p['proposed_delta']:+g}; score {p['computed_score']:+.4f}, "
                      f"n={n})  [id {p['history_id']}]")
        lines.append(f"  • `{p['path']}`: {detail}")
    for e in errors:
        lines.append(f"  • `{e['axis']}`: error — {e['error']}")
    if not proposals and not errors:
        lines.append("  • (no whitelisted params configured)")
    lines.append("Review with `--review`; approve at the terminal.")
    return "\n".join(lines)


def _apply_param_to_config(path: str, value: float) -> str:
    """Backup kairos_config.json, set the dotted path, re-load and json-validate.

    Whitelisted paths only, with a final bounds + 25%-max-change guard against the
    CURRENT on-disk value (defense in depth — the config may have changed since the
    proposal was written). Raises ValueError on any violation BEFORE writing so the
    caller can abort the approval without side effects. Returns the backup path.
    """
    if path not in PARAM_WHITELIST:
        raise ValueError(f"param path {path!r} is not whitelisted — refusing to apply")
    lo, hi = PARAM_WHITELIST[path]
    value = float(value)
    if not (lo <= value <= hi):
        raise ValueError(f"{path}={value} out of bounds [{lo}, {hi}] — refusing to apply")

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    current, found = _get_dotted(cfg, path)
    if found and isinstance(current, (int, float)) and not isinstance(current, bool) \
            and current != 0:
        if abs(value - current) > PARAM_MAX_CHANGE_FRAC * abs(current) + 1e-9:
            raise ValueError(
                f"{path}: change {current} → {value} exceeds "
                f"{PARAM_MAX_CHANGE_FRAC:.0%} of the current value — refusing to apply")

    # Timestamped backup of the exact current file BEFORE any write.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = f"{CONFIG_PATH}.bak_{ts}"
    with open(CONFIG_PATH) as f:
        original = f.read()
    with open(backup, "w") as f:
        f.write(original)

    # Write to a temp file, re-load to prove it parses, then atomically swap.
    _set_dotted(cfg, path, value)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(tmp) as f:
        json.load(f)  # raises if the written JSON is somehow invalid
    os.replace(tmp, CONFIG_PATH)
    return backup


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
        if decision == "approve" and _is_param_axis(row["axis"]):
            # Exit-parameter proposal: apply to kairos_config.json (whitelist +
            # bounds + 25% guard, timestamped backup, json re-validate). If the
            # apply raises, nothing below runs and the row stays 'proposed'.
            # axis_weights is NEVER touched for a param proposal.
            path = _param_path(row["axis"])
            backup = _apply_param_to_config(path, row["new_weight"])
            conn.execute(
                "UPDATE axis_weight_history "
                "SET status = 'approved', decided_at = ?, decided_by = ? WHERE id = ?",
                (now, decided_by, history_id),
            )
            conn.commit()
            outcome = {
                "axis": row["axis"], "path": path, "status": "approved",
                "new_value": row["new_weight"], "backup": backup,
                "decided_by": decided_by,
            }
            _post_slack(
                f":wrench: *Exit parameter approved* — `{path}`\n"
                f"value {row['prior_weight']:g} → *{row['new_weight']:g}* "
                f"(Δ {row['proposed_delta']:+g}) by {decided_by}\n"
                f"Written to kairos_config.json (backup saved). Live on the next "
                f"exit-engine run."
            )
        elif decision == "approve":
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
        from kairos_alerts import post_message, _load_slack_config
        # Env-first token resolution: SLACK_BOT_TOKEN over kairos_config.json.
        # _load_slack_config already applies that override; resolving it here and
        # passing it explicitly keeps this notify path env-first (so a blank
        # config bot_token no longer forces a 'Slack skipped') independent of
        # post_message's default config loading.
        cfg = _load_slack_config()
        return post_message(ARBITER_CHANNEL, text, cfg=cfg)
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
            if _is_param_axis(r["axis"]):
                print(f"     n={ev.get('n')}  avg_mfe={ev.get('avg_mfe_pct')}pp  "
                      f"avg_pnl={ev.get('avg_pnl_pct')}pp  "
                      f"avg_give_back={ev.get('avg_give_back_pct')}pp")
                print(f"     round_trips={ev.get('round_trips')} "
                      f"(rate {ev.get('roundtrip_rate')})  severity={ev.get('severity')}  "
                      f"bounds={ev.get('bounds')}  max_change={ev.get('max_change_frac')}")
                print()
                continue
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


def _print_param_compute(r: dict) -> None:
    ev = r["evidence"]
    lo, hi = ev["bounds"]
    print(f"  ── param: {r['path']}")
    print(f"     current_value={r['current_value']}  bounds=[{lo}, {hi}]  "
          f"max_change=±{ev['max_change_frac']:.0%}")
    print(f"     n={ev['n']} (min_sample={PARAM_MIN_SAMPLE})  "
          f"avg_mfe={ev['avg_mfe_pct']}pp  avg_pnl={ev['avg_pnl_pct']}pp  "
          f"avg_give_back={ev['avg_give_back_pct']}pp")
    print(f"     round_trips={ev['round_trips']} (rate {ev['roundtrip_rate']})  "
          f"severity={ev['severity']}  computed_score={r['computed_score']:+.4f}")
    if r["gated"]:
        print(f"     GATED (n < {PARAM_MIN_SAMPLE} or value absent) — no change; "
              f"proposed stays {r['proposed_value']}")
    else:
        print(f"     proposed_value={r['proposed_value']} "
              f"(Δ {r['proposed_value'] - r['current_value']:+g})")


def _cli_compute_params(dry_run: bool) -> int:
    if dry_run:
        print("  Exit-parameter compute (DRY RUN — nothing written, nothing posted)\n")
        for path in PARAM_WHITELIST:
            _print_param_compute(compute_param(path))
        return 0
    result = propose_all_params()
    print(f"  propose_all_params run {result['run_id']}: "
          f"{len(result['proposals'])} proposal(s), {len(result['errors'])} error(s).")
    for p in result["proposals"]:
        print(f"    {p['path']}: id {p['history_id']}, score {p['computed_score']:+.4f}, "
              f"n={p['sample_size']}, {p['prior_weight']:g} → {p['new_weight']:g} "
              f"(Δ {p['proposed_delta']:+g}){' [GATED]' if p['gated'] else ''}")
    for e in result["errors"]:
        print(f"    {e['axis']}: ERROR — {e['error']}")
    print("  kairos_config.json is UNCHANGED (nothing approved).")
    print(f"  Slack post to {ARBITER_CHANNEL}: "
          f"{'OK' if result['slack_posted'] else 'FAILED'}")
    print("\n  ----- Slack message text -----")
    print(result["slack_text"])
    return 0


# ── Self-test (isolated: temp DBs + a /tmp config copy) ──────────────

def _selftest() -> int:
    """Exercise whitelist rejection, bounds clamping, and a full propose→approve
    cycle against COPIES in /tmp — never the live config or the live DBs."""
    import shutil
    import tempfile
    import kairos_log_db

    global CONFIG_PATH, ML_DB_PATH, _post_slack
    orig_config, orig_ml, orig_db = CONFIG_PATH, ML_DB_PATH, kairos_log_db.DB_PATH
    orig_post = _post_slack
    _post_slack = lambda *a, **k: True  # no live Slack posts during the self-test
    tmpdir = tempfile.mkdtemp(prefix="axis_param_selftest_")
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    try:
        # 1. Isolated config copy in /tmp.
        CONFIG_PATH = os.path.join(tmpdir, "kairos_config.json")
        shutil.copy(orig_config, CONFIG_PATH)

        # 2. Synthetic ML DB with >= min_sample TRAILING-STOP rows (high give-back
        #    so severity is non-trivial and drives a real, bounded proposal).
        ML_DB_PATH = os.path.join(tmpdir, "ml.db")
        mc = sqlite3.connect(ML_DB_PATH)
        mc.execute("CREATE TABLE trade_outcomes (pnl_pct REAL, mfe_pct REAL, "
                   "give_back_pct REAL, exit_reason TEXT, timestamp_exit TEXT)")
        for i in range(15):
            # winners that round-tripped: big mfe, small/negative pnl → big give-back
            mc.execute("INSERT INTO trade_outcomes VALUES (?,?,?,?,?)",
                       (-1.0 if i % 2 else 2.0, 22.0, 21.0,
                        "TRAILING-STOP: retreated 18% from peak 22%", "2026-06-01 00:00:00 UTC"))
        mc.commit(); mc.close()

        # 3. Fresh temp kairos.db for axis_weight_history.
        kairos_log_db.DB_PATH = os.path.join(tmpdir, "kairos.db")
        kairos_log_db.init_db()

        # ── Whitelist rejection ──────────────────────────────────────
        try:
            compute_param("exits.trailing_stop.enabled")
            check("whitelist rejection: compute_param(non-whitelisted) raises", False)
        except ValueError:
            check("whitelist rejection: compute_param(non-whitelisted) raises", True)
        try:
            propose_param_update("exits.some.other.path")
            check("whitelist rejection: propose_param_update(non-whitelisted) raises", False)
        except ValueError:
            check("whitelist rejection: propose_param_update(non-whitelisted) raises", True)
        try:
            _apply_param_to_config("exits.not.whitelisted", 1.0)
            check("whitelist rejection: _apply_param_to_config(non-whitelisted) raises", False)
        except ValueError:
            check("whitelist rejection: _apply_param_to_config(non-whitelisted) raises", True)

        # ── Bounds clamping (compute) ────────────────────────────────
        trail = compute_param("exits.trailing_stop.target_armed.trail_pct")
        lo, hi = PARAM_WHITELIST["exits.trailing_stop.target_armed.trail_pct"]
        cur = trail["current_value"]
        within_bounds = lo <= trail["proposed_value"] <= hi
        within_25 = abs(trail["proposed_value"] - cur) <= PARAM_MAX_CHANGE_FRAC * abs(cur) + 1e-9
        check(f"bounds: trail_pct proposal {trail['proposed_value']} in [{lo},{hi}]", within_bounds)
        check(f"25% cap: |{trail['proposed_value']}-{cur}| <= 25% of {cur}", within_25)
        check("direction: high give-back TIGHTENS trail_pct (proposed <= current)",
              trail["proposed_value"] <= cur)

        # ── Bounds clamping (apply guard) ────────────────────────────
        try:
            _apply_param_to_config("exits.trailing_stop.target_armed.trail_pct", 99.0)
            check("bounds: _apply out-of-bounds (99.0) raises", False)
        except ValueError:
            check("bounds: _apply out-of-bounds (99.0) raises", True)
        try:
            # within bounds but > 25% jump from current (8.0 → 11.0 is +37.5%)
            _apply_param_to_config("exits.trailing_stop.target_armed.trail_pct", 11.0)
            check("25% cap: _apply >25% jump (8.0→11.0) raises", False)
        except ValueError:
            check("25% cap: _apply >25% jump (8.0→11.0) raises", True)

        # ── Full propose → approve cycle ─────────────────────────────
        p = propose_param_update("exits.trailing_stop.target_armed.trail_pct")
        check("propose: non-gated proposal written (n>=min_sample)", not p["gated"])
        check("propose: history_id assigned", isinstance(p["history_id"], int))
        outcome = apply_decision(p["history_id"], "approve", "selftest")
        check("approve: outcome status approved", outcome.get("status") == "approved")
        # Config now carries the approved value.
        with open(CONFIG_PATH) as f:
            applied = json.load(f)
        applied_val, _ = _get_dotted(applied, "exits.trailing_stop.target_armed.trail_pct")
        check(f"approve: config updated to proposed value ({p['new_weight']})",
              abs(applied_val - p["new_weight"]) < 1e-9)
        check("approve: a timestamped backup was created",
              os.path.exists(outcome.get("backup", "")))
        # Row is now approved, not re-approvable.
        try:
            apply_decision(p["history_id"], "approve", "selftest")
            check("lifecycle: re-approving an approved row raises", False)
        except ValueError:
            check("lifecycle: re-approving an approved row raises", True)

        print()
        if failures:
            print(f"  SELFTEST FAILED — {len(failures)} check(s): {failures}")
            return 1
        print("  SELFTEST PASSED — all checks green.")
        return 0
    finally:
        CONFIG_PATH, ML_DB_PATH, kairos_log_db.DB_PATH = orig_config, orig_ml, orig_db
        _post_slack = orig_post
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    group.add_argument("--compute-params", action="store_true",
                       help="Compute + propose exit-parameter updates (param:<dotted.path>) "
                            "for every whitelisted param; one combined Slack note. Approves "
                            "nothing. With --dry-run, preview only (writes/posts nothing).")
    group.add_argument("--selftest", action="store_true",
                       help="Run the param propose→approve self-test against /tmp copies.")
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
    parser.add_argument("--dry-run", action="store_true",
                        help="With --compute-params: compute and print, but write/post nothing.")
    args = parser.parse_args()

    if args.compute:
        return _cli_compute(args.axis)
    if args.propose:
        return _cli_propose(args.axis)
    if args.propose_all:
        return _cli_propose_all()
    if args.compute_params:
        return _cli_compute_params(args.dry_run)
    if args.selftest:
        return _selftest()
    if args.review:
        return _cli_review()
    if args.approve is not None:
        return _cli_decide(args.approve, "approve", args.by)
    if args.reject is not None:
        return _cli_decide(args.reject, "reject", args.by)
    return 1


if __name__ == "__main__":
    sys.exit(main())
