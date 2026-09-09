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
import hashlib
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

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
    "exits.trailing_stop.profit_floor_pp":           (0.5, 3.0),
    # Deprecated as the primary trail, still the fallback — still learnable,
    # because it still governs every close where ATR was unavailable.
    "exits.trailing_stop.target_armed.trail_pct":    (4.0, 12.0),
    # ATR-scaled trail. Each has its OWN evidence pool, routed by bind state
    # (see PARAM_BIND_ROUTE) — a close is evidence about exactly one of them.
    "exits.trailing_stop.target_armed.atr_mult":     (0.4, 1.5),
    "exits.trailing_stop.target_armed.trail_lo_pct": (1.0, 4.0),
    "exits.trailing_stop.target_armed.trail_hi_pct": (3.0, 10.0),
}

# ── Evidence routing by clamp-bind state ─────────────────────────────
# The trail applied to a position is decided by exactly ONE parameter: the
# multiplier if the raw value landed inside the clamp, otherwise whichever
# bound truncated it. So a close is evidence about that one parameter and no
# other. On the live book of 2026-09-09, 30 of 49 held positions are
# clamp-bound — if clamped closes were pooled into atr_mult, the multiplier
# would be "learned" mostly from trades where it had literally no effect.
#
# A path absent from this map is UNROUTED: every close is eligible, which is
# the pre-existing behaviour for trail_pct and profit_floor_pp. profit_floor_pp
# is deliberately unrouted — it is a different mechanism (the floor on allowed
# retreat), it applies whatever the trail was, and it is NOT part of the
# coupled family.
PARAM_BIND_ROUTE = {
    "exits.trailing_stop.target_armed.atr_mult":     "free",
    "exits.trailing_stop.target_armed.trail_lo_pct": "floor",
    "exits.trailing_stop.target_armed.trail_hi_pct": "ceiling",
}

# ── Coupled family (see propose_all_params) ──────────────────────────
# These three are coupled THROUGH the routing above: raising trail_lo_pct moves
# trades from the free bucket into the floor bucket, shrinking atr_mult's pool
# and growing trail_lo_pct's. At most one may carry a live delta per run.
# trail_pct and profit_floor_pp are not members — they are separate mechanisms.
PARAM_COUPLED_FAMILY = (
    "exits.trailing_stop.target_armed.atr_mult",
    "exits.trailing_stop.target_armed.trail_lo_pct",
    "exits.trailing_stop.target_armed.trail_hi_pct",
)

# Bound pairs that must never cross. A proposal that would push lo above hi is
# GATED to a human, not silently clamped: silently ordering them would apply a
# value nobody proposed, and the crossing itself is the signal that the two
# pools are disagreeing about where the band belongs.
PARAM_ORDERED_PAIRS = (
    ("exits.trailing_stop.target_armed.trail_lo_pct",
     "exits.trailing_stop.target_armed.trail_hi_pct"),
)
# ── Role sign: which DIRECTION tightens profit capture, per parameter ──
# Was `-1.0 if path.endswith("trail_pct") else +1.0`. That suffix test is a
# landmine the moment a parameter is added whose name does not end in
# "trail_pct" but which still TIGHTENS by decreasing. The ATR family is exactly
# that case: atr_mult, trail_lo_pct and trail_hi_pct all tighten when they go
# DOWN (a smaller multiplier or a smaller clamp bound is a narrower trail), yet
# only trail_lo_pct/trail_hi_pct would have matched the suffix, and atr_mult
# would have fallen through to +1.0 — proposing a LOOSENING every time the
# evidence said tighten, and vice versa. Silently, and in the direction that
# compounds give-back.
#
# So the mapping is explicit and keyed on the FULL path. A path with no entry
# raises rather than defaulting: a wrong sign here is worse than a crash,
# because the crash is caught by the per-param try/except and reported, while
# a wrong sign quietly drives a live exit parameter the wrong way.
PARAM_ROLE_SIGN = {
    # Narrower trail / smaller multiplier / smaller clamp bound = tighter.
    "exits.trailing_stop.target_armed.trail_pct":     -1.0,
    "exits.trailing_stop.target_armed.atr_mult":      -1.0,
    "exits.trailing_stop.target_armed.trail_lo_pct":  -1.0,
    "exits.trailing_stop.target_armed.trail_hi_pct":  -1.0,
    # A HIGHER floor protects more profit = tighter.
    "exits.trailing_stop.profit_floor_pp":            +1.0,
}


def _role_sign(path: str) -> float:
    """Direction that TIGHTENS profit capture for this exact config path.

    Raises on an unmapped path. See PARAM_ROLE_SIGN for why this refuses to
    guess: defaulting a sign is how a parameter gets driven backwards.
    """
    if path not in PARAM_ROLE_SIGN:
        raise ValueError(
            f"no role sign mapped for {path!r} — add it to PARAM_ROLE_SIGN "
            f"(a guessed sign can drive a live exit parameter backwards)")
    return PARAM_ROLE_SIGN[path]


PARAM_MIN_SAMPLE = 10          # min TRAILING-STOP closes before a param is proposed
PARAM_MAX_CHANGE_FRAC = 0.25   # an approved proposal may move a value at most ±25%
# Net error (pp) that maps to a full-strength (100% of the allowed ±25%) nudge.
PARAM_GIVEBACK_REF = 15.0

# ── Materiality: computing is not the same as surfacing ──────────────
# Removing the min_sample gate (2026-09-09) means a proposal is now emitted on
# essentially every run, most of them tiny — an effective_n of 1 buys ~2.3% of
# the current value. Every proposal used to raise an interactive Approve/Reject
# card, and 4-6 sub-1%-confidence cards a day would bury the one that matters.
# Alert fatigue is not a cosmetic problem here: the human gate is the last
# thing standing between the loop and live capital, and a gate nobody reads is
# not a gate.
#
# So: ALWAYS compute, ALWAYS record the row, ALWAYS route the evidence. Raise a
# card only when the move is big enough for a human decision to be worth
# making. Everything else is recorded, digested in one line, and reviewable on
# demand via --pending-minor.
#
# PARAM threshold — 5% of the current value, chosen for two reasons:
#   * it is exactly one fifth of PARAM_MAX_CHANGE_FRAC (0.25), so a proposal
#     must be worth at least 20% of a maximum step to interrupt someone;
#   * at trail_pct=6.2255 it is ±0.31pp of trail, which moves the intraday
#     backstop by ~0.47pp. Below that the behavioural difference is smaller
#     than the spread between where a stop triggers and where it fills, so
#     there is nothing a human could meaningfully judge.
# Working backwards through delta = severity x confidence x 0.25 x current, a
# card needs severity x confidence >= 0.2 — i.e. effective_n >= ~2.5 even at
# maximum severity, and more when the measured bias is milder.
PARAM_MATERIALITY_FRAC = 0.05
#
# AXIS threshold — 0.02 absolute. Axis weights live in [-1, 1], so this is 1%
# of the range and ~13% of per_run_cap (0.15). Absolute rather than relative
# because an axis weight is legitimately 0.0, where a relative threshold is
# either meaningless or infinite. The bar sits a little higher relative to the
# cap than the param one because axis weights are still OBSERVED ONLY — not
# injected into any reasoning prompt — so a small move has no live effect at
# all and certainly does not warrant an interrupt.
AXIS_MATERIALITY_ABS = 0.02


def materiality_threshold(axis: str, prior_weight) -> float:
    """Smallest |delta| worth raising an interactive card for, on this axis."""
    if axis.startswith(PARAM_PREFIX):
        base = abs(float(prior_weight)) if prior_weight else 0.0
        # A param sitting at 0.0 has no meaningful relative scale; fall back to
        # the axis threshold rather than surfacing every microscopic move.
        return (base * PARAM_MATERIALITY_FRAC) if base > 0 else AXIS_MATERIALITY_ABS
    return AXIS_MATERIALITY_ABS


def is_material(axis: str, prior_weight, proposed_delta) -> bool:
    """True iff this proposal is big enough to interrupt a human for.

    Immaterial is NOT the same as gated, skipped or deferred. The evidence was
    usable, the move is real and the row is written; it is simply too small to
    be worth a decision. It stays reviewable via --pending-minor.
    """
    if proposed_delta is None:
        return False
    d = abs(float(proposed_delta))
    if d < 1e-9:
        return False
    return d >= materiality_threshold(axis, prior_weight)


# ── Ratchet guards (2026-07-11 postmortem) ───────────────────────────
# The ratchet compounded trail_pct 8.0 → 4.0 across three days because each
# day's proposal was legal in isolation: every step was within ±25% of the
# value the PREVIOUS step had just written. A cumulative band anchored at the
# value in force at the start of a rolling window bounds the drift no matter
# how many individually-legal steps are taken inside it. Enforced at BOTH
# compute (clamp) and apply (refuse) — the apply side is what actually holds,
# since a proposal can sit pending while other changes land.
PARAM_CUMULATIVE_WINDOW_DAYS = 7
# TIGHTENED 0.40 -> 0.15 on 2026-09-09, when the min_sample gate was removed.
#
# The band used to be a backstop BEHIND a gate that refused to propose at all
# below min_sample. With the gate gone it is the primary defence against drift,
# and 0.40 never binds at the step sizes thin evidence now produces:
#
#   effective_n=1 -> confidence 0.091 x PARAM_MAX_CHANGE_FRAC 0.25
#                 =  ~2.3% of the current value, per run
#   run daily     -> ~16%/week compounding
#
# 16% fits inside a 40% band without ever touching it. On unbiased noise that
# random-walks harmlessly; on BIASED thin evidence — say a week of closes all
# from one regime, all saying "tighten" — it drifts steadily. That is the July
# ratchet in slow motion: every individual step legal, the cumulative move
# unbounded. 0.15 is chosen so that a week of maximally-biased 2.3% steps
# binds partway through rather than never.
PARAM_CUMULATIVE_BAND_FRAC = 0.15

# Axis-side equivalents of the two post-ratchet guards (added 2026-08-20).
# The 2026-08-03 unfreeze shipped the cumulative band + freshness gate to the
# PARAM path only. Axes got neither, and exit_timing is the sole auto-proposing
# axis — so the one surface that generates proposals unattended was the one
# running unguarded. Replaying 2026-07-08..11 shows why: exit_timing moved
# +0.3626 -> +0.5179 across four consecutive days on an UNCHANGED n=37 sample
# (five writes, same 37 trade_ids), a +43% drift that nothing bounded.
#
# The band is ADDITIVE here, not multiplicative like PARAM_CUMULATIVE_BAND_FRAC.
# Axis weights are additive offsets in [-weight_bound, +weight_bound] and are
# legitimately 0.0 (exit_timing was exactly 0.0000 on 2026-06-30). A
# multiplicative band anchored at 0.0 collapses to lo == hi == 0.0 and would
# freeze the axis permanently.
AXIS_CUMULATIVE_WINDOW_DAYS = 7
# TIGHTENED 0.20 -> 0.08 on 2026-09-09, same reasoning as
# PARAM_CUMULATIVE_BAND_FRAC: this is now the primary drift bound, not a
# backstop behind the min_sample gate. Additive rather than multiplicative
# because an axis weight is legitimately 0.0 and a multiplicative band
# anchored at zero collapses to lo == hi == 0 and freezes the axis.
AXIS_CUMULATIVE_BAND_ABS = 0.08

# Horizon over which forgone gain (the too-EARLY pole) is measured, in calendar
# days. Must be one of kairos_ml_outcomes.FORGONE_HORIZONS.
#
# This choice is load-bearing, not cosmetic. Give-back is realised the moment a
# position is sold, but forgone gain accrues for as long as the stock keeps
# running — so a SHORT horizon systematically favours tightening. At 5 days a
# winner sold into a multi-week advance looks nearly costless; at 60 days it
# does not. That asymmetry is the measurement analogue of the one-sided
# objective the ratchet ran on, which is why the horizon is explicit and
# recorded in the evidence rather than hard-coded at the shortest option.
#
# Longer horizons need time to mature, so raise this as the corpus fills in;
# compute_param reports coverage at every horizon to make that call concrete.
#
# Set to 14d on 2026-08-03. It covers the SAME 19 in-regime trailing-stop
# trades as 5d — so the longer window costs no sample — while measuring nearly
# 50% more forgone gain (6.01 vs 4.14) and correspondingly tempering the
# proposal (8.0 -> 6.95 rather than 6.70). 30d is the next step at n=13 and
# should be adopted once coverage grows; 60d is n=2 and gated.
PARAM_FORGONE_HORIZON_DAYS = 14


# ── Weighted evidence (2026-09-09) ───────────────────────────────────
# The regime window was a BINARY filter: a close counted only if the snapshot
# recorded the exact value now in force. That stopped the July ratchet, but it
# also meant every config change reset the evidence pool to empty. With 3-8
# closes a week and a parameter that moves on approval, the pool could never
# reach PARAM_MIN_SAMPLE — on 2026-09-08, trail_pct had 75 trailing-stop closes
# and a sample of 0, with 72 of them discarded for no reason other than having
# closed under a different number. That is not conservatism; it is a loop that
# is structurally unable to learn.
#
# Evidence is now WEIGHTED rather than discarded:
#
#   weight(trade) = recency_weight × parameter_proximity_weight
#
# and the gate reads sum(weights) — the EFFECTIVE sample — instead of a count
# of exact matches. What still gets discarded outright is a close with no
# regime snapshot at all: that is genuinely missing data, not a mismatch, and
# there is no distance to measure.
#
# Half-life for the recency term. 45 days is chosen against this corpus, not in
# the abstract: the trailing-stop close history spans ~100 days, so 45d leaves
# the oldest evidence at ~0.2 weight — present, clearly outranked, not erased.
# A much shorter half-life would reproduce the starvation problem through a
# different door (only the last three weeks would count); a much longer one
# would let pre-redesign exits keep voting at nearly full strength.
WEIGHT_RECENCY_HALF_LIFE_DAYS = 45.0

# Proximity floor: how much a trade closed at the far end of the parameter's
# configured bounds is still worth. NOT zero — a trail_pct=4.0 close still says
# something about give-back dynamics at 6.95, just much less than a 6.6 close
# does. Zero here is what made the pool resettable.
WEIGHT_PROXIMITY_FLOOR = 0.25

# e-folding distance for the proximity term, as a FRACTION of the parameter's
# configured bounds range. 0.25 means a trade one quarter of the full bounds
# range away from the current value retains ~1/e of the above-floor weight
# (trail_pct bounds are [4,12], so 0.25 ⇒ 2.0pp). Normalising by the bounds
# range is what lets one constant serve parameters on different scales.
WEIGHT_PROXIMITY_SCALE = 0.25

# ── Confidence: the anti-ratchet half of the trade ───────────────────
# Loosening the filter without damping the step would be a straight increase in
# how far a single run can move a live parameter. It is not: the step is now
# scaled by a confidence factor that rises monotonically with the EFFECTIVE
# sample,
#
#   confidence = effective_n / (effective_n + min_sample)
#
# so weak evidence — few trades, or many trades all measured far from the
# current value — produces a proportionally smaller move. Properties that
# matter here:
#   * monotone in effective_n at EVERY sample size, with no saturation cliff,
#     so "more/closer evidence ⇒ larger step" holds everywhere rather than only
#     below some threshold;
#   * strictly < 1.0, so this can only ever SHRINK a step relative to the
#     pre-2026-09-09 formula (which took a full-strength step the instant the
#     exact-match count crossed min_sample), never grow one;
#   * exactly 0.5 at effective_n == min_sample — barely enough evidence buys
#     half a step, which is the reading the gate boundary should have.
# The per-run cap, the ±25% step cap, the cumulative 7-day band, the freshness
# gate and the human approval gate are all untouched and still sit downstream.


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

def _current_axis_weight(axis: str):
    """Live weight for an axis from kairos.db, or None if unreadable.

    None means "regime cannot be established", which callers treat as
    excluding every row rather than admitting all of them.
    """
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT weight FROM axis_weights WHERE axis = ?", (axis,)
            ).fetchone()
        finally:
            conn.close()
        if row is not None and row["weight"] is not None:
            return float(row["weight"])
    except Exception:
        pass
    return None


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
        SELECT trade_id, timestamp_exit, give_back_pct, post_exit_peak_pct,
               exit_params_snapshot
        FROM trade_outcomes
        WHERE timestamp_exit IS NOT NULL
    """
    conn = _ml_connect_ro()
    try:
        all_rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()

    # ── Weighted evidence (replaces the binary regime window) ────────
    # The old window kept ONLY trades that closed while exit_timing carried the
    # exact weight in force today, which made every approved weight change
    # reset the evidence pool to empty — the same starvation the param loop hit
    # (75 trailing-stop closes, sample 0). A trade that closed under a nearby
    # weight was not produced by an unrelated system; it was produced by a
    # slightly different one, and it is correspondingly less informative rather
    # than uninformative. So it is DOWN-WEIGHTED, not discarded. Only trades
    # with no regime snapshot at all are excluded — that is missing data.
    #
    # An axis weight lives in [-weight_bound, +weight_bound], so the bounds
    # range the proximity term normalises by is 2·weight_bound.
    current_weight = _current_axis_weight("exit_timing")
    bound = float(load_config()["weight_bound"])

    def _matures(r) -> bool:
        return (r["give_back_pct"] is not None
                and r["post_exit_peak_pct"] is not None)

    weighted_all, n_no_snapshot, anchor = _weigh_rows(
        all_rows, "axis_weights", "exit_timing", current_weight,
        -bound, bound, contributes=_matures)
    snapshot_rows = [r for r, _, _ in weighted_all]

    # Matured set: both poles present. post_exit_peak_pct is NULL until the
    # post-exit window matures — a feature-filled trade still missing it counts
    # as pending, not as evidence of either direction. (Closed trades that never
    # had outcome features computed at all — give_back_pct NULL — are out of
    # scope here, neither matured nor pending.) UNCHANGED by the weighting
    # redesign: what counts as matured is exactly what it was.
    weighted = [(r, w, v) for r, w, v in weighted_all if _matures(r)]
    matured = [r for r, _, _ in weighted]
    effective_n = sum(w for _, w, _ in weighted)
    n_matured = len(matured)
    n_exact_match = sum(
        1 for _, _, v in weighted
        if current_weight is not None
        and abs(float(v) - float(current_weight)) <= 1e-6)
    n_pending = sum(1 for r in snapshot_rows
                    if r["give_back_pct"] is not None
                    and r["post_exit_peak_pct"] is None)

    # Weighted means, over the same weights the gate sums.
    mean_error = _weighted_mean(
        [(r["give_back_pct"] - r["post_exit_peak_pct"], w)
         for r, w, _ in weighted])
    mean_giveback = _weighted_mean([(r["give_back_pct"], w)
                                    for r, w, _ in weighted])
    mean_post_exit_peak = _weighted_mean([(r["post_exit_peak_pct"], w)
                                          for r, w, _ in weighted])

    computed_score = _clamp(mean_error / GIVEBACK_SCALE, -1.0, 1.0)
    trade_ids = sorted(r["trade_id"] for r in matured)

    # sample_size reports the EFFECTIVE sample, because that is what the gate
    # compares and what _zero_sample_streak reads to detect a starved loop.
    n = int(round(effective_n))

    # Why the sample is what it is — see _describe_starvation. n_pending here is
    # "enriched but the post-exit window has not elapsed"; a row with no
    # give_back_pct at all was never enriched, which is a different problem with
    # a different fix. "Closed under a different weight" is no longer a cause
    # of exclusion, so it is reported as n_downweighted.
    starvation = {
        "n_no_snapshot": n_no_snapshot,
        "n_wrong_regime": 0,
        "n_downweighted": n_matured - n_exact_match,
        "n_missing_features": sum(1 for r in snapshot_rows
                                  if r["give_back_pct"] is None),
        "n_pending_maturation": n_pending,
        "zero_sample_streak": (_zero_sample_streak("exit_timing") if n == 0 else
                               {"runs": 0, "since": None, "closes_since": 0}),
    }
    starvation["summary"] = _describe_starvation(starvation)

    evidence = {
        "mean_error_pp": round(mean_error, 6),
        "mean_giveback_pp": round(mean_giveback, 6),
        "mean_post_exit_peak_pp": round(mean_post_exit_peak, 6),
        "n_matured": n_matured,
        "effective_n": round(effective_n, 4),
        "n_exact_match": n_exact_match,
        "n_downweighted": n_matured - n_exact_match,
        "n_pending": n_pending,
        # Back-compat key: "in regime" in the weighted sense — every matured
        # close that carries a snapshot.
        "n_in_regime": n_matured,
        "n_total_closed": len(all_rows),
        "n_excluded_out_of_regime": 0,
        "n_excluded_no_snapshot": n_no_snapshot,
        "weighting": {
            "recency_half_life_days": WEIGHT_RECENCY_HALF_LIFE_DAYS,
            "proximity_floor": WEIGHT_PROXIMITY_FLOOR,
            "proximity_scale_frac_of_bounds": WEIGHT_PROXIMITY_SCALE,
            "bounds": [-bound, bound],
            "recency_anchor": (anchor.strftime("%Y-%m-%d %H:%M:%S UTC")
                               if anchor is not None else None),
            "by_param_value": _weight_breakdown(
                weighted, current_weight, -bound, bound),
        },
        "current_weight": current_weight,
        "trade_ids": trade_ids,
        "starvation": starvation,
    }
    return {
        "axis": "exit_timing",
        "computed_score": computed_score,
        "sample_size": n,
        "effective_n": effective_n,
        "evidence": evidence,
    }


# ── Compute: reallocation_aggressiveness statistic ───────────────────

def compute_reallocation_aggressiveness() -> dict:
    """Bidirectional rotation-discipline score from thesis decay + B2a features.

    exit_reason-based reallocation tagging keys off trade_outcomes.exit_reason,
    which was almost entirely NULL when this was written (a downstream symptom of
    the write_trade_close decision_id bug). Exit reasons now persist per-close in
    position_exits_history and are backfilled into trade_outcomes, but this stat
    still keys off thesis_conditions_intact at the LAST checkpoint of each closed
    trade — reallocation exits remain too thin to tag reliably by reason:

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
                  sample_size: int, cfg: dict,
                  effective_n: float | None = None) -> dict:
    """Smoothed tracker toward the current statistic, scaled by confidence.

    `effective_n` is the weighted sample (see the WEIGHT_* / CONFIDENCE block).
    It scales the step through _confidence — so the same measured bias moves
    the weight LESS when the evidence behind it is thin or was gathered far
    from the current value. Defaults to sample_size for axes that carry no
    regime weighting.

    ── The min_sample CLIFF was removed 2026-09-09 ──
    This used to early-return gated=True whenever eff < min_sample, which
    threw away the very property _confidence was designed to have. Two
    mechanisms were doing the same job and the cruder one won:

        confidence = eff/(eff+min_sample)  ->  9% of a step at eff=1,
        33% at 5, 50% at 10, 80% at 40.

    That is already the correct treatment of thin evidence — a proportional
    move, sized to what the evidence is worth. The gate replaced it with
    "learn nothing", which on a loop seeing 3-8 closes a week means most weeks
    teach the system nothing at all.

    min_sample is RETAINED, unchanged, but now serves exactly one purpose: it
    is the scaling constant in _confidence (the effective_n at which a step is
    half strength). It is no longer a threshold.

    What still gates is MISSING evidence, which is a different thing from thin
    evidence: effective_n of exactly 0 means no contributing close exists, and
    there is nothing to be proportional to. That returns gated=True as before.

    Returns {proposed_delta, new_weight, gated, confidence}.
    """
    eff = float(sample_size if effective_n is None else effective_n)
    if eff <= 0.0:
        # MISSING evidence, not thin evidence. No close contributed, so there
        # is no measurement to scale down — a zero-evidence "proportional
        # step" would be a step taken on nothing.
        return {"proposed_delta": 0.0, "new_weight": prior_weight,
                "gated": True, "confidence": 0.0}

    confidence = _confidence(eff, cfg["min_sample"])
    raw = cfg["learning_rate"] * confidence * (computed_score - prior_weight)
    proposed_delta = _clamp(raw, -cfg["per_run_cap"], cfg["per_run_cap"])
    new_weight = _clamp(prior_weight + proposed_delta,
                        -cfg["weight_bound"], cfg["weight_bound"])
    return {"proposed_delta": proposed_delta, "new_weight": new_weight,
            "gated": False, "confidence": confidence}


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
        delta = compute_delta(computed_score, prior_weight, sample_size, cfg,
                              effective_n=result.get("effective_n"))

        # ── Ratchet guards (ported to the axis path 2026-08-20) ──────
        # Both were previously PARAM-only. See AXIS_CUMULATIVE_BAND_ABS.
        # Every bind is recorded in evidence["guards"] with the value that
        # WOULD have been written, so the guards are auditable and their cost
        # measurable — a guard that quietly suppresses good moves has to be
        # visible before it can be tuned.
        evidence = dict(result["evidence"])
        guards: dict = {}

        new_weight = delta["new_weight"]
        proposed_delta = delta["proposed_delta"]
        gated = delta["gated"]
        # The confidence factor that scaled this step, recorded so a reviewer
        # can see WHY a real measured bias produced a small move.
        evidence["confidence"] = round(delta.get("confidence", 1.0), 6)
        if result.get("effective_n") is not None:
            evidence["effective_n"] = round(float(result["effective_n"]), 4)

        ev_hash = _evidence_hash(
            evidence.get("trade_ids") or [], prior_weight, new_weight)
        evidence["evidence_hash"] = ev_hash

        band = _axis_cumulative_band(axis, prior_weight)
        evidence["cumulative_band"] = band

        # Freshness: an unchanged corpus landing on the same value is not new
        # information. This is the guard that would have stopped four of the
        # five 2026-07-08..11 exit_timing writes outright (identical n=37).
        prev = conn.execute(
            "SELECT evidence FROM axis_weight_history "
            "WHERE axis = ? AND status IN ('proposed','approved','superseded') "
            "ORDER BY id DESC LIMIT 1",
            (axis,),
        ).fetchone()
        if prev is not None and prev["evidence"]:
            try:
                prev_hash = (json.loads(prev["evidence"]) or {}).get("evidence_hash")
            except (json.JSONDecodeError, TypeError):
                prev_hash = None
            if prev_hash and prev_hash == ev_hash and abs(proposed_delta) > 1e-9:
                guards["freshness"] = {
                    "bound": True,
                    "reason": "evidence unchanged since last proposal",
                    "would_have_been": round(new_weight, 6),
                    "evidence_hash": ev_hash,
                }
                new_weight = prior_weight
                proposed_delta = 0.0
                gated = True

        # ── Materiality (2026-09-09) ─────────────────────────────────
        # Recorded as a BOUND GUARD when the move is too small to surface.
        # That is not cosmetic: kairos_autonomy.auto_apply skips any proposal
        # whose evidence carries a bound guard ("escalating to human"), and
        # exit_timing IS in AUTO_APPLY_AXES. Without this, a sub-threshold
        # proposal that never raised a card could still auto-apply — a change
        # to live exit behaviour that no human ever saw. Using the existing
        # guard-bind contract achieves the block without touching
        # kairos_autonomy.py, and it is honest: a guard did bind, and the
        # documented consequence of a bind is that a human decides.
        _material = is_material(axis, prior_weight, proposed_delta)
        evidence["materiality"] = {
            "threshold": round(materiality_threshold(axis, prior_weight), 6),
            "delta": round(abs(proposed_delta), 6),
            "material": bool(_material),
            "basis": "absolute weight units",
        }
        if not _material and abs(proposed_delta) > 1e-9:
            guards["materiality"] = {
                "bound": True,
                "reason": (f"|Δ| {abs(proposed_delta):.6f} < materiality "
                           f"threshold {materiality_threshold(axis, prior_weight):.6f}"
                           f" — recorded, not surfaced, and NOT auto-appliable"),
                "would_have_been": round(new_weight, 6),
            }

        # Cumulative band: bounds drift across many individually-legal steps.
        if (band.get("lo") is not None and abs(proposed_delta) > 1e-9):
            clamped = min(max(new_weight, band["lo"]), band["hi"])
            if abs(clamped - new_weight) > 1e-9:
                guards["cumulative_band"] = {
                    "bound": True,
                    "reason": (f"outside 7d band "
                               f"[{band['lo']:+.4f}, {band['hi']:+.4f}] "
                               f"anchored at {band['base_7d']:+.4f}"),
                    "would_have_been": round(new_weight, 6),
                    "clamped_to": round(clamped, 6),
                }
                new_weight = clamped
                proposed_delta = round(clamped - prior_weight, 6)

        evidence["guards"] = guards

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
                json.dumps(evidence), prior_weight,
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
        "run_id": run_id,
        "computed_score": computed_score,
        "sample_size": sample_size,
        "prior_weight": prior_weight,
        "proposed_delta": proposed_delta,
        "new_weight": new_weight,
        "gated": gated,
        "material": bool(evidence.get("materiality", {}).get("material")),
        "materiality_threshold": evidence.get(
            "materiality", {}).get("threshold"),
        "guards": guards,
        "evidence": evidence,
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



# 2026-07-11: proposal generation frozen after the approve-ratchet postmortem.
# Evidence was stale (mfe NULL on 28/30 rows; averages driven by 2 trades) and the
# objective one-sided (no forgone-gain term), so daily proposals compounded 25%/night
# on unchanged information.
#
# 2026-08-03: UNFROZEN. The redesign the freeze was waiting on has landed and is
# verified — two-sided objective, regime window, contributing-row filter,
# cumulative 7-day ±40% band, freshness gate; 26/26 selftests green; replaying
# the original ratchet now halts at 4.8 (−40%) instead of 4.0 (−50%). Forgone
# gain is measured at a 14-day horizon rather than 5 (see
# PARAM_FORGONE_HORIZON_DAYS) so the too-early pole is not systematically
# understated. The human approval gate is unchanged and remains mandatory:
# unfreezing lets the loop PROPOSE, never apply.
PROPOSALS_FROZEN = False

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
    if PROPOSALS_FROZEN:
        _p = f"proposals FROZEN pending evidence redesign ({PROPOSALS_FROZEN}); compute skipped."
        print(_p)
        return {"frozen": True, "reason": PROPOSALS_FROZEN}
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
                # Carried so the Slack note can explain an empty sample.
                "evidence": p.get("evidence") or {},
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
        # Plain-English block first: the raw-number line below is retained as a
        # precise audit trail, but the reviewer needs to know what the change
        # MEANS before they see the internals. Failure here must never break
        # the proposal loop, so the whole thing degrades to raw numbers only.
        try:
            from kairos_arbiter_explain import explain_axis_proposal
            lines.extend(explain_axis_proposal(p))
        except Exception as _exc:
            lines.append(f"  _(plain-English render unavailable: {_exc})_")

        if p["gated"]:
            detail = (f"gated — no usable evidence "
                      f"(effective sample {p['sample_size']}); no change")
        elif abs(p["proposed_delta"]) < 1e-9:
            detail = (f"no change (weight stays {p['new_weight']:+.4f}; "
                      f"score {p['computed_score']:+.4f}, n={p['sample_size']})")
        else:
            detail = (f"{p['prior_weight']:+.4f} → *{p['new_weight']:+.4f}* "
                      f"(Δ {p['proposed_delta']:+.4f}; score {p['computed_score']:+.4f}, "
                      f"n={p['sample_size']})  [id {p['history_id']}]")
        lines.append(f"  • `{p['axis']}`: {detail}")

        # Surface any ratchet guard that bound, with the value it suppressed.
        # A guard that silently blocks good moves is as costly as one that is
        # too loose, so the cost is always stated and never inferred.
        for name, g in (p.get("guards") or {}).items():
            if not g.get("bound"):
                continue
            if name == "freshness":
                lines.append(
                    f"      :lock: freshness — {g['reason']}; "
                    f"would have written {g['would_have_been']:+.4f}")
            elif name == "cumulative_band":
                lines.append(
                    f"      :lock: 7d band — {g['reason']}; "
                    f"{g['would_have_been']:+.4f} → clamped "
                    f"{g['clamped_to']:+.4f}")

        # Same rule as the param loop: an empty sample must say why it is empty.
        if not p["sample_size"]:
            starve = (p.get("evidence") or {}).get("starvation") or {}
            why = starve.get("summary")
            streak = (starve.get("zero_sample_streak") or {})
            # Red only when it is unambiguous: repeatedly empty AND trades
            # closed in between. One empty run on a young regime is a warning.
            marker = (":rotating_light:"
                      if streak.get("runs", 0) >= 2 and streak.get("closes_since")
                      else ":warning:")
            lines.append(f"      {marker} sample is 0"
                         + (f" — {why}" if why else ""))
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


# ── Regime windowing + evidence freshness ────────────────────────────
# The ratchet ran on FROZEN evidence: 28 of 30 mfe_pct values were NULL, so
# every "average" was computed from 2 trades while n reported 30, and the same
# stale numbers re-proposed a fresh tightening every day. Three fixes live
# here, and every one of them is a REDUCTION in what counts as evidence:
#
#   regime window — a trade closed under trail_pct=8 tells you nothing about
#                   whether trail_pct=4 is right. Only trades whose recorded
#                   exit_params_snapshot matches the CURRENT value count.
#   contributing  — a row counts only if it carries every field the objective
#                   consumes. A NULL is not a zero.
#   freshness     — if the contributing set is byte-identical to the one that
#                   produced the standing proposal, there is nothing new to
#                   say, and re-proposing is how a ratchet compounds.


def _snapshot_value(snapshot_json, kind: str, key: str):
    """Read params[key] / axis_weights[key] out of an exit_params_snapshot.

    Returns None when the snapshot is absent, unparseable, or does not carry
    the key — all of which mean "cannot attribute this trade to a regime", and
    the caller must therefore EXCLUDE the row rather than assume it matches.
    """
    if not snapshot_json:
        return None
    try:
        snap = json.loads(snapshot_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(snap, dict):
        return None
    section = snap.get(kind)
    if not isinstance(section, dict):
        return None
    val = section.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _snapshot_bind_state(snapshot_json):
    """exit_params_snapshot["armed_trail"]["bind_state"], or None.

    None means the close carries no clamp-bind attribution at all — it closed
    before the mechanism existed, or never armed. Such a close is evidence
    about NONE of the three ATR parameters, which is the correct reading: there
    is no fact about which bound governed it, and inventing one would be worse
    than a small sample.
    """
    if not snapshot_json:
        return None
    try:
        snap = json.loads(snapshot_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(snap, dict):
        return None
    block = snap.get("armed_trail")
    if not isinstance(block, dict):
        return None
    bs = block.get("bind_state")
    return bs if isinstance(bs, str) else None


def _route_rows(rows, path: str):
    """(eligible, routing) — keep only closes this parameter actually governed.

    Unrouted paths (trail_pct, profit_floor_pp) pass through untouched, so
    their behaviour is exactly what it was before routing existed.
    """
    want = PARAM_BIND_ROUTE.get(path)
    counts = {"free": 0, "floor": 0, "ceiling": 0, "no_atr": 0, "unattributed": 0}
    for r in rows:
        bs = _snapshot_bind_state(r.get("exit_params_snapshot"))
        counts[bs if bs in counts else "unattributed"] += 1
    routing = {
        "routed": want is not None,
        "governs_bind_state": want,
        "by_bind_state": counts,
        "n_before_routing": len(rows),
    }
    if want is None:
        routing["n_after_routing"] = len(rows)
        return list(rows), routing
    keep = [r for r in rows
            if _snapshot_bind_state(r.get("exit_params_snapshot")) == want]
    routing["n_after_routing"] = len(keep)
    return keep, routing


def _in_regime(snapshot_json, kind: str, key: str, current, tol: float = 1e-6) -> bool:
    """True iff this trade closed under the value currently in force."""
    if current is None:
        return False
    val = _snapshot_value(snapshot_json, kind, key)
    return val is not None and abs(val - float(current)) <= tol


# ── Evidence weighting (replaces the binary regime window) ───────────
# See the WEIGHT_* constants for why this exists. Three functions:
#   _recency_weight    — exponential decay on how old a close is
#   _proximity_weight  — decay on how far its parameter value sat from today's
#   _weigh_rows        — applies both, and reports the breakdown
# plus _weighted_mean / _confidence, which every downstream statistic uses.


def _parse_exit_ts(raw):
    """Parse a trade_outcomes.timestamp_exit into an aware UTC datetime, or None.

    The corpus stores three shapes ('… UTC', '…T…Z', '…T…' bare), which is why
    this normalises before parsing rather than trusting one format.
    """
    if not raw:
        return None
    txt = str(raw).strip().replace(" UTC", "").replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M",
                "%Y-%m-%d"):
        try:
            return datetime.strptime(txt, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _corpus_anchor(rows) -> "datetime | None":
    """Newest parseable close in the weighted set — the recency reference point.

    Deliberately the newest CLOSE, not wall-clock now(). Anchoring on now()
    would make every weight drift a little every day, which changes the
    proposed value at the 4th decimal on an otherwise identical corpus — and
    the evidence hash covers the proposed value, so the freshness gate would
    stop binding and the loop would re-propose micro-moves forever. That is a
    ratchet with a smaller step size. Anchoring on the corpus makes the compute
    a pure function of the evidence: same trades in, same answer out, freshness
    gate holds exactly as before.
    """
    stamps = [t for t in (_parse_exit_ts(r.get("timestamp_exit")) for r in rows)
              if t is not None]
    return max(stamps) if stamps else None


def _recency_weight(raw_ts, anchor) -> float:
    """0 < w <= 1: exponential half-life decay in days before the anchor.

    An undateable close keeps full recency weight — its age is unknown, and its
    snapshot (the thing this loop actually reasons about) is intact. All 75
    live trailing-stop closes parse; this is a defensive branch, not a path.
    """
    if anchor is None:
        return 1.0
    ts = _parse_exit_ts(raw_ts)
    if ts is None:
        return 1.0
    days = max(0.0, (anchor - ts).total_seconds() / 86400.0)
    return 0.5 ** (days / WEIGHT_RECENCY_HALF_LIFE_DAYS)


def _proximity_weight(val, current, lo, hi) -> float:
    """1.0 at the exact current value, decaying to WEIGHT_PROXIMITY_FLOOR.

    Distance is normalised by the parameter's configured bounds range so a
    single scale constant works for trail_pct (span 8.0) and profit_floor_pp
    (span 2.5) alike.
    """
    if val is None or current is None:
        return WEIGHT_PROXIMITY_FLOOR
    span = abs(float(hi) - float(lo))
    dist = abs(float(val) - float(current))
    if span <= 0:
        return 1.0 if dist <= 1e-9 else WEIGHT_PROXIMITY_FLOOR
    rel = dist / span
    return WEIGHT_PROXIMITY_FLOOR + (1.0 - WEIGHT_PROXIMITY_FLOOR) * math.exp(
        -rel / WEIGHT_PROXIMITY_SCALE)


def _weigh_rows(rows, kind: str, key: str, current, lo, hi, contributes=None):
    """Attach an evidence weight to every row that carries a regime snapshot.

    Returns (weighted, n_no_snapshot, anchor):
      weighted        — [(row, weight, snapshot_value), …], snapshot-bearing only
      n_no_snapshot   — rows dropped because they cannot be dated to a regime
      anchor          — the recency reference point actually used

    Rows with no snapshot are EXCLUDED, not floored: there is no parameter
    distance to measure, so there is nothing to down-weight. That is the one
    exclusion the old binary filter got right.

    `contributes` selects the rows the recency anchor is drawn from — pass the
    caller's own contributing-row predicate. The anchor must be a function of
    exactly the set the evidence hash covers, or a newly-closed-but-unmatured
    trade would shift every weight (and so the proposed value) without adding
    any evidence, and the freshness gate would read that as something new.
    Non-contributing rows are still weighted and returned, because the
    starvation diagnostics need them.
    """
    have_snap = []
    n_no_snapshot = 0
    for r in rows:
        val = _snapshot_value(r.get("exit_params_snapshot"), kind, key)
        if val is None:
            n_no_snapshot += 1
            continue
        have_snap.append((r, val))

    anchor_rows = [r for r, _ in have_snap
                   if contributes is None or contributes(r)]
    anchor = _corpus_anchor(anchor_rows)
    weighted = []
    for r, val in have_snap:
        prox = _proximity_weight(val, current, lo, hi)
        rec = _recency_weight(r.get("timestamp_exit"), anchor)
        weighted.append((r, prox * rec, val))

    return weighted, n_no_snapshot, anchor


def _weight_breakdown(weighted, current, lo, hi) -> list:
    """Per-distinct-parameter-value view of where the effective sample came from.

    This is the auditability half of the change: a reviewer has to be able to
    see that a proposal rests on (say) 15 closes at 6.76 rather than 19 at 8.0,
    without re-deriving the weights by hand.
    """
    buckets: dict = {}
    for r, w, val in weighted:
        b = buckets.setdefault(round(float(val), 6),
                               {"value": round(float(val), 6), "n": 0,
                                "proximity": None, "recency_sum": 0.0,
                                "effective_n": 0.0})
        b["n"] += 1
        b["effective_n"] += w
        if b["proximity"] is None:
            b["proximity"] = round(_proximity_weight(val, current, lo, hi), 4)
        b["recency_sum"] += (w / b["proximity"]) if b["proximity"] else 0.0
    out = []
    for b in sorted(buckets.values(), key=lambda x: x["value"]):
        out.append({
            "value": b["value"],
            "n": b["n"],
            "proximity": b["proximity"],
            "mean_recency": round(b["recency_sum"] / b["n"], 4) if b["n"] else 0.0,
            "effective_n": round(b["effective_n"], 4),
            "exact_match": abs(b["value"] - float(current)) <= 1e-6
                           if current is not None else False,
        })
    return out


def _weighted_mean(pairs) -> float:
    """Sum(w·x)/Sum(w) over (value, weight) pairs, skipping None values."""
    num = den = 0.0
    for x, w in pairs:
        if x is None:
            continue
        num += float(x) * float(w)
        den += float(w)
    return (num / den) if den > 0 else 0.0


def _confidence(effective_n: float, min_sample: float) -> float:
    """effective_n / (effective_n + min_sample) — see the CONFIDENCE comment.

    Monotone, strictly < 1.0, exactly 0.5 at the gate threshold. Scales the
    step; does not touch the caps, the band, or the human gate.
    """
    eff = max(0.0, float(effective_n))
    ms = max(0.0, float(min_sample))
    if eff <= 0.0:
        return 0.0
    if ms <= 0.0:
        return 1.0
    return eff / (eff + ms)


# ── Starvation diagnostics ───────────────────────────────────────────
# A gated proposal reporting "sample 0" is ambiguous in the worst possible way:
# it reads identically whether the regime is simply young (correct, expect it to
# fill in) or the evidence pipeline has stopped feeding it (a silent, permanent
# outage). The two are distinguishable — a starved loop shows trades CLOSING
# while the sample stays empty — but only if the message carries the reason
# alongside the number. These helpers exist to make that difference loud.


def _zero_sample_streak(axis: str) -> dict:
    """How long this axis has reported an EMPTY sample, and what closed meanwhile.

    Walks axis_weight_history newest→oldest while sample_size is 0/NULL. Returns
    {runs, since, closes_since}: consecutive empty-sample proposal rows, the
    timestamp of the earliest one, and how many trades have closed since then.
    A single empty run is ordinary. Empty runs spanning real closes are not.
    Degrades to zeros if either DB is unreadable — a diagnostic must never be
    able to break the compute it is describing.
    """
    out = {"runs": 0, "since": None, "closes_since": 0}
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT sample_size, created_at FROM axis_weight_history "
                "WHERE axis = ? ORDER BY id DESC LIMIT 50", (axis,)).fetchall()
        finally:
            conn.close()
    except Exception:
        return out

    for r in rows:
        if r["sample_size"]:      # non-zero, non-NULL → streak ends here
            break
        out["runs"] += 1
        out["since"] = r["created_at"]

    if out["since"]:
        try:
            conn = _ml_connect_ro()
            try:
                out["closes_since"] = conn.execute(
                    # Normalise BOTH sides before comparing. timestamp_exit is
                    # stored in three formats across the corpus (173 '…T…Z',
                    # 39 'space, no suffix', 29 '…T…' no Z) while `since` comes
                    # from axis_weight_history.created_at as '… UTC'. A raw
                    # string >= is lexicographic, and 'T' (ASCII 84) sorts above
                    # ' ' (32) — so '2026-08-19T09:00:00Z' compares GREATER than
                    # a 10:26 cutoff despite being an hour earlier, silently
                    # counting closes that precede the window. Latent today only
                    # because no T-format exit has yet landed earlier in the day
                    # than a cutoff; that is luck, not correctness.
                    "SELECT COUNT(*) FROM trade_outcomes "
                    "WHERE timestamp_exit IS NOT NULL "
                    "  AND REPLACE(REPLACE(REPLACE(timestamp_exit,'T',' '),"
                    "'Z',''),' UTC','') >= "
                    "      REPLACE(REPLACE(REPLACE(?,'T',' '),'Z',''),' UTC','')",
                    (out["since"],)).fetchone()[0]
            finally:
                conn.close()
        except Exception:
            pass
    return out


def _describe_starvation(starve: dict) -> str:
    """One human sentence naming why a sample is thin, or '' when it is not.

    Ordered most-actionable first: a missing regime snapshot is unrecoverable
    for that row and needs a code fix; unmatured rows just need time.
    """
    parts = []
    if starve.get("n_no_snapshot"):
        parts.append(f"{starve['n_no_snapshot']} closed trade(s) lack a regime "
                     f"snapshot (permanently unusable as evidence)")
    if starve.get("n_wrong_regime"):
        parts.append(f"{starve['n_wrong_regime']} closed under different "
                     f"parameters")
    elif starve.get("n_downweighted"):
        # NOT an exclusion. Named anyway, because a reviewer looking at a thin
        # effective sample needs to know it is thin because the evidence is
        # distant/stale, not because rows were thrown away.
        parts.append(f"{starve['n_downweighted']} closed under different "
                     f"parameters (down-weighted, not discarded)")
    if starve.get("n_missing_features"):
        parts.append(f"{starve['n_missing_features']} in-regime row(s) never "
                     f"had outcome features computed "
                     f"(kairos_outcome_features.fill_features)")
    if starve.get("n_pending_maturation"):
        parts.append(f"{starve['n_pending_maturation']} in-regime row(s) pending "
                     f"maturation")
    if starve.get("governs_bind_state"):
        # Say this plainly, and split the two causes — otherwise a routed
        # parameter's small pool reads as a bug, or worse, an unrecoverable
        # backlog reads as a self-healing one.
        if starve.get("n_other_bind_state"):
            parts.append(f"{starve['n_other_bind_state']} close(s) governed by "
                         f"a different clamp state (this parameter learns only "
                         f"from {starve['governs_bind_state']}-bound closes)")
        if starve.get("n_unattributed"):
            parts.append(f"{starve['n_unattributed']} close(s) carry no "
                         f"clamp-bind attribution (closed before bind-state "
                         f"stamping; permanently unusable for this parameter, "
                         f"the pool refills from new closes)")
    streak = starve.get("zero_sample_streak") or {}
    if streak.get("runs", 0) >= 2:
        detail = f"sample has read 0 for {streak['runs']} consecutive run(s)"
        if streak.get("since"):
            detail += f" since {streak['since']}"
        if streak.get("closes_since"):
            # This is the line that separates "young regime" from "starved loop":
            # trades closed and the sample still did not move.
            detail += f" while {streak['closes_since']} trade(s) closed"
        parts.append(detail)
    return "; ".join(parts)


def _evidence_hash(trade_ids, current, proposed) -> str:
    """Stable fingerprint of what a proposal is based on.

    Covers the identity of every contributing trade plus both endpoints of the
    proposed move, so adding a trade, losing one, or landing on a different
    value all read as fresh evidence — and re-running on an unchanged corpus
    does not.
    """
    payload = json.dumps({
        "trade_ids": sorted(str(t) for t in trade_ids),
        "current": None if current is None else round(float(current), 6),
        "proposed": None if proposed is None else round(float(proposed), 6),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _ordering_violation(path: str, proposed) -> str | None:
    """Reason string if `proposed` would cross an ordered bound pair, else None.

    Checked against the OTHER bound's live on-disk value, because that is what
    the engine will actually read. Deliberately returns a gate reason rather
    than a clamped value: silently ordering the pair would apply a number
    nobody proposed, and would hide the thing worth seeing — the two evidence
    pools disagreeing about where the band belongs. That is a decision for J,
    not an arithmetic fix.
    """
    if proposed is None:
        return None
    for lo_path, hi_path in PARAM_ORDERED_PAIRS:
        if path == lo_path:
            other = _current_param_value(hi_path)
            if other is not None and float(proposed) > float(other):
                return (f"would cross the ordered pair: {lo_path.rsplit('.', 1)[-1]}"
                        f"={float(proposed):.4f} > {hi_path.rsplit('.', 1)[-1]}"
                        f"={float(other):.4f} — a floor above the ceiling is not a "
                        f"band. Gated to a human rather than clamped; raise "
                        f"{hi_path.rsplit('.', 1)[-1]} first if this is intended")
        elif path == hi_path:
            other = _current_param_value(lo_path)
            if other is not None and float(proposed) < float(other):
                return (f"would cross the ordered pair: {hi_path.rsplit('.', 1)[-1]}"
                        f"={float(proposed):.4f} < {lo_path.rsplit('.', 1)[-1]}"
                        f"={float(other):.4f} — a ceiling below the floor is not a "
                        f"band. Gated to a human rather than clamped; lower "
                        f"{lo_path.rsplit('.', 1)[-1]} first if this is intended")
    return None


def _cumulative_band(axis: str, current) -> dict:
    """Rolling-window bounds for a param, anchored at the window's opening value.

    base_7d is the prior_weight of the EARLIEST approved change inside the
    window — i.e. what the parameter was before the window's first step. With
    no approved change in the window, the current value is itself the anchor.
    Degrades to an unbounded-but-reported band if the history table cannot be
    read, so a DB problem never silently removes the guard's visibility.
    """
    base = None if current is None else float(current)
    try:
        from kairos_log_db import get_connection
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=PARAM_CUMULATIVE_WINDOW_DAYS)
                  ).strftime("%Y-%m-%d %H:%M:%S UTC")
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT prior_weight FROM axis_weight_history "
                "WHERE axis = ? AND status = 'approved' AND decided_at IS NOT NULL "
                "  AND decided_at >= ? "
                "ORDER BY decided_at ASC LIMIT 1",
                (axis, cutoff),
            ).fetchone()
        finally:
            conn.close()
        if row is not None and row["prior_weight"] is not None:
            base = float(row["prior_weight"])
    except Exception:
        pass

    if base is None:
        return {"base_7d": None, "lo": None, "hi": None,
                "window_days": PARAM_CUMULATIVE_WINDOW_DAYS,
                "band_frac": PARAM_CUMULATIVE_BAND_FRAC}
    return {
        "base_7d": round(base, 6),
        "lo": round(base * (1.0 - PARAM_CUMULATIVE_BAND_FRAC), 6),
        "hi": round(base * (1.0 + PARAM_CUMULATIVE_BAND_FRAC), 6),
        "window_days": PARAM_CUMULATIVE_WINDOW_DAYS,
        "band_frac": PARAM_CUMULATIVE_BAND_FRAC,
    }


def _axis_cumulative_band(axis: str, current) -> dict:
    """Additive rolling-window bounds for an AXIS weight.

    Same anchoring rule as _cumulative_band (the prior_weight of the earliest
    approved change inside the window, else the current value), but the bounds
    are base ± AXIS_CUMULATIVE_BAND_ABS rather than a percentage of base — see
    the AXIS_CUMULATIVE_BAND_ABS comment for why a multiplicative band is
    unusable on a quantity that is legitimately 0.0.

    Degrades to an unbounded-but-reported band if history cannot be read, so a
    DB problem never silently removes the guard's visibility.
    """
    base = None if current is None else float(current)
    try:
        from kairos_log_db import get_connection
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=AXIS_CUMULATIVE_WINDOW_DAYS)
                  ).strftime("%Y-%m-%d %H:%M:%S UTC")
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT prior_weight FROM axis_weight_history "
                "WHERE axis = ? AND status = 'approved' AND decided_at IS NOT NULL "
                "  AND decided_at >= ? "
                "ORDER BY decided_at ASC LIMIT 1",
                (axis, cutoff),
            ).fetchone()
        finally:
            conn.close()
        if row is not None and row["prior_weight"] is not None:
            base = float(row["prior_weight"])
    except Exception:
        pass

    if base is None:
        return {"base_7d": None, "lo": None, "hi": None,
                "window_days": AXIS_CUMULATIVE_WINDOW_DAYS,
                "band_abs": AXIS_CUMULATIVE_BAND_ABS}
    return {
        "base_7d": round(base, 6),
        "lo": round(base - AXIS_CUMULATIVE_BAND_ABS, 6),
        "hi": round(base + AXIS_CUMULATIVE_BAND_ABS, 6),
        "window_days": AXIS_CUMULATIVE_WINDOW_DAYS,
        "band_abs": AXIS_CUMULATIVE_BAND_ABS,
    }


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

    axis = PARAM_PREFIX + path
    lo, hi = PARAM_WHITELIST[path]
    current = _current_param_value(path)

    # Forgone-gain horizon is configurable (see PARAM_FORGONE_HORIZON_DAYS).
    try:
        from kairos_ml_outcomes import FORGONE_HORIZONS, forgone_column
        horizon = (PARAM_FORGONE_HORIZON_DAYS
                   if PARAM_FORGONE_HORIZON_DAYS in FORGONE_HORIZONS
                   else FORGONE_HORIZONS[0])
        all_horizons = list(FORGONE_HORIZONS)
    except Exception:
        horizon, all_horizons = 5, [5]
        def forgone_column(d):  # noqa: E306 — local fallback
            return f"forgone_gain_{int(d)}d_pct"
    conn = _ml_connect_ro()
    try:
        # Select only horizon columns that actually exist: a database that has
        # not yet run the migration must degrade to the horizons it has, not
        # crash the learning loop.
        present = {r["name"] for r in conn.execute("PRAGMA table_info(trade_outcomes)")}
        all_horizons = [h for h in all_horizons if forgone_column(h) in present]
        if not all_horizons:
            all_horizons = [horizon]
        if forgone_column(horizon) not in present and all_horizons:
            horizon = all_horizons[0]
        fg_col = forgone_column(horizon)

        sql = f"""
            SELECT trade_id, timestamp_exit, pnl_pct, mfe_pct, give_back_pct,
                   {', '.join(forgone_column(h) for h in all_horizons)},
                   exit_params_snapshot
            FROM trade_outcomes
            WHERE exit_reason LIKE 'TRAILING-STOP%'
              AND timestamp_exit IS NOT NULL
        """
        all_rows = [dict(r) for r in conn.execute(sql).fetchall()]
        # Table-wide, not just trailing-stop: an unstamped close is invisible to
        # EVERY regime window, so this counts the systemic hole rather than this
        # one param's slice of it.
        n_closed_no_snapshot = conn.execute(
            "SELECT COUNT(*) FROM trade_outcomes WHERE timestamp_exit IS NOT NULL "
            "AND exit_params_snapshot IS NULL").fetchone()[0]
    finally:
        conn.close()

    # ── Bind-state routing (2026-09-09) ─────────────────────────────
    # BEFORE weighting, not after: a close the clamp handed to a different
    # parameter is not weak evidence about this one, it is evidence about
    # something else, and down-weighting it would still let it vote.
    routed_rows, routing = _route_rows(all_rows, path)

    # ── Contributing rows ────────────────────────────────────────────
    # Both poles of the objective must be present. A trade missing either one
    # cannot say which way we were wrong, and averaging over its absence is
    # exactly what let 2 trades masquerade as 30. This filter is UNCHANGED by
    # the weighting redesign — a NULL is still not a zero.
    def _contributes(r) -> bool:
        return (r.get("mfe_pct") is not None
                and r.get("pnl_pct") is not None
                and r.get(fg_col) is not None)

    # ── Weighted evidence (replaces the binary regime window) ────────
    # Every snapshot-bearing close counts, weighted by recency × parameter
    # proximity. Only closes with NO snapshot are excluded — see _weigh_rows.
    weighted_all, n_no_snapshot_ts, anchor = _weigh_rows(
        routed_rows, "params", path, current, lo, hi, contributes=_contributes)

    weighted = [(r, w, v) for r, w, v in weighted_all if _contributes(r)]
    rows = [r for r, _, _ in weighted]
    weights = [w for _, w, _ in weighted]
    effective_n = sum(weights)
    # The number the OLD gate used: contributing closes at the exact current
    # value. Retained purely so the proposal can show what it would have had.
    n_exact_match = sum(
        1 for _, _, v in weighted
        if current is not None and abs(float(v) - float(current)) <= 1e-6)
    n_contributing = len(weighted)
    # sample_size is the quantity the gate acts on, so it reports the EFFECTIVE
    # sample (the column is INTEGER, and _zero_sample_streak reads it to detect
    # a starved loop — both want the gated number, not the raw count).
    n = int(round(effective_n))

    # ── Why the sample is what it is ─────────────────────────────────
    # Split the non-contributing rows by CAUSE, because the causes need
    # different responses: no snapshot is a code fix, missing features means
    # the enricher is not running, and pending maturation just needs time.
    # "Closed under a different parameter" is NO LONGER a cause of exclusion —
    # it is reported as n_downweighted so the discount stays visible.
    snapshot_rows = [r for r, _, _ in weighted_all]
    n_no_snapshot = n_no_snapshot_ts
    starvation = {
        "n_no_snapshot": n_no_snapshot,
        # Kept at 0 so any reader still keyed on it reports the truth: nothing
        # is discarded for being out of regime any more.
        "n_wrong_regime": 0,
        "n_downweighted": n_contributing - n_exact_match,
        # Never enriched at all — mfe is written by the same pass as forgone.
        "n_missing_features": sum(1 for r in snapshot_rows
                                  if r.get("mfe_pct") is None),
        # Enriched, but this horizon has not elapsed yet.
        "n_pending_maturation": sum(1 for r in snapshot_rows
                                    if r.get("mfe_pct") is not None
                                    and r.get(fg_col) is None),
        "n_closed_no_snapshot_all": n_closed_no_snapshot,
        # Closes this parameter did not govern. NOT a fault and NOT a data
        # gap — they are in another parameter's pool by design.
        "n_routed_elsewhere": (routing["n_before_routing"]
                               - routing["n_after_routing"]),
        "governs_bind_state": routing["governs_bind_state"],
        # Split the routed-away rows: a close with a DIFFERENT bind state is in
        # another parameter's pool (expected, self-healing), whereas a close
        # with NO attribution at all predates the mechanism and will never be
        # usable by any of the three. Those need different responses, so they
        # must not read as the same sentence.
        "n_unattributed": (routing.get("by_bind_state") or {}).get(
            "unattributed", 0),
        "n_other_bind_state": (routing["n_before_routing"]
                               - routing["n_after_routing"]
                               - (routing.get("by_bind_state") or {}).get(
                                   "unattributed", 0)),
        "zero_sample_streak": _zero_sample_streak(axis) if n == 0 else
                              {"runs": 0, "since": None, "closes_since": 0},
    }
    starvation["summary"] = _describe_starvation(starvation)

    def _avg(key):
        """Weighted mean over the contributing set (same weights as the gate)."""
        return _weighted_mean([(r.get(key), w) for r, w, _ in weighted])

    # ── Two-sided objective ──────────────────────────────────────────
    # give_back  — edge surrendered by exiting too LATE  (+ ⇒ tighten)
    # forgone    — edge left behind by exiting too EARLY (+ ⇒ loosen)
    # The old objective measured give-back ONLY, so its error term could never
    # be negative and every proposal it could physically emit was a tightening.
    # That is the ratchet, in one line of arithmetic. Netting the two poles is
    # what makes loosening representable at all.
    # Every mean below is WEIGHTED by the same recency × proximity weight the
    # gate sums, so the statistic and the sample it is gated on describe the
    # same object. Unweighted means over a mixed-parameter corpus would let a
    # cluster of distant, stale closes set the direction while contributing
    # almost nothing to effective_n.
    gb_pairs, fg_pairs, err_pairs = [], [], []
    for r, w, _v in weighted:
        gb = r["give_back_pct"]
        if gb is None:
            gb = r["mfe_pct"] - r["pnl_pct"]
        fg = r[fg_col]
        gb_pairs.append((gb, w))
        fg_pairs.append((fg, w))
        err_pairs.append((gb - fg, w))

    mean_error = _weighted_mean(err_pairs)
    avg_gb = _weighted_mean(gb_pairs)
    avg_fg = _weighted_mean(fg_pairs)

    def _is_round_trip(r) -> bool:
        return (r["mfe_pct"] is not None and r["mfe_pct"] > 2.0
                and r["pnl_pct"] is not None and r["pnl_pct"] <= 0.0)

    # Raw count stays a count (it is a tally of real trades); the RATE is
    # weighted, because it feeds the reviewer's read of the same weighted set.
    round_trips = sum(1 for r in rows if _is_round_trip(r))
    roundtrip_rate = _weighted_mean(
        [(1.0 if _is_round_trip(r) else 0.0, w) for r, w, _ in weighted])

    # ── Gate: MISSING evidence only (the min_sample cliff is gone) ───
    # Removed 2026-09-09. `effective_n < PARAM_MIN_SAMPLE` used to gate the
    # proposal outright, discarding the proportional scaling _confidence
    # already applies. See compute_delta for the full reasoning; the short
    # version is that thin evidence deserves a small step, not silence, and
    # PARAM_MIN_SAMPLE is now purely the _confidence scaling constant.
    #
    # These three are genuinely-absent evidence, not thin evidence, and there
    # is nothing for a proportional step to be proportional TO:
    #   * no current value on disk        -> nothing to move
    #   * effective_n exactly 0           -> no contributing close exists
    #   * no contributing close carries a snapshot -> no measurable regime
    gate_reason = None
    if current is None:
        gate_reason = f"current value unavailable at config path {path!r}"
    elif effective_n <= 0.0 or n_contributing == 0:
        _route_note = ("" if not routing["routed"] else
                       f", routed to the {routing['governs_bind_state']}-bound "
                       f"subset ({routing['n_after_routing']} of "
                       f"{routing['n_before_routing']})")
        gate_reason = (
            f"no usable evidence: effective sample {effective_n:.2f} from "
            f"{n_contributing} contributing close(s) of {len(all_rows)} "
            f"trailing-stop closes{_route_note}. This is MISSING evidence, "
            f"not thin evidence — a proportional step needs something to be "
            f"proportional to")
        # A bare count says the loop is quiet; the cause says whether that is
        # expected. Always carried when the sample is EMPTY, where the ambiguity
        # between "young regime" and "broken pipeline" is total.
        if starvation["summary"]:
            gate_reason += f" — {starvation['summary']}"
    gated = gate_reason is not None

    # Confidence from the effective sample — the anti-ratchet half of the
    # weighting trade. Weak or mostly-distant evidence buys a smaller step.
    # This is now the ONLY thing standing between thin evidence and a full
    # step, which is why Part 2 tightened the cumulative band underneath it.
    confidence = _confidence(effective_n, PARAM_MIN_SAMPLE)

    # Magnitude from the size of the net error; sign from which pole dominates.
    severity = _clamp(abs(mean_error) / PARAM_GIVEBACK_REF, 0.0, 1.0)
    if mean_error > 0:
        direction_label = "tighten"
    elif mean_error < 0:
        direction_label = "loosen"
    else:
        direction_label = "hold"

    # Role sign: which way must THIS parameter move in order to tighten profit
    # capture?  Explicit per full path — never inferred from the name.
    role_sign = _role_sign(path)
    error_sign = 1.0 if mean_error > 0 else (-1.0 if mean_error < 0 else 0.0)
    computed_score = round(role_sign * error_sign * severity, 6)

    band = _cumulative_band(axis, current)

    if gated or current is None:
        proposed_value = current
    else:
        max_step = PARAM_MAX_CHANGE_FRAC * abs(current)
        change = _clamp(role_sign * error_sign * severity * confidence * max_step,
                        -max_step, max_step)
        proposed = current + change
        # Hard whitelist bounds, then the cumulative window band.
        proposed = _clamp(proposed, lo, hi)
        if band["lo"] is not None and band["hi"] is not None:
            proposed = _clamp(proposed, band["lo"], band["hi"])
        proposed_value = round(proposed, 4)
        # Ordered-pair guard, LAST — after every clamp, because the clamps are
        # what determine the value that would actually be written.
        _cross = _ordering_violation(path, proposed_value)
        if _cross is not None:
            gate_reason = _cross
            gated = True
            proposed_value = current

    evidence = {
        # n IS the effective sample, rounded — it is what the gate compares.
        "n": n,
        "effective_n": round(effective_n, 4),
        "confidence": round(confidence, 4),
        # What the OLD binary regime window would have counted. Kept so a
        # reviewer can see the difference the weighting made on this very run.
        "n_exact_match": n_exact_match,
        "n_contributing": n_contributing,
        "n_downweighted": n_contributing - n_exact_match,
        # Back-compat key: it now means "in regime" in the weighted sense —
        # every contributing close that carries a snapshot.
        "n_in_regime": n_contributing,
        "n_total_trailing_stop": len(all_rows),
        "routing": routing,
        # Only genuinely missing data is excluded now, not parameter mismatch.
        "n_excluded_out_of_regime": 0,
        "n_excluded_no_snapshot": n_no_snapshot,
        "n_excluded_non_contributing": len(weighted_all) - n_contributing,
        "weighting": {
            "recency_half_life_days": WEIGHT_RECENCY_HALF_LIFE_DAYS,
            "proximity_floor": WEIGHT_PROXIMITY_FLOOR,
            "proximity_scale_frac_of_bounds": WEIGHT_PROXIMITY_SCALE,
            "recency_anchor": (anchor.strftime("%Y-%m-%d %H:%M:%S UTC")
                               if anchor is not None else None),
            # Where the effective sample came from, per distinct parameter
            # value: the audit trail for a weighted proposal.
            "by_param_value": _weight_breakdown(weighted, current, lo, hi),
        },
        "avg_mfe_pct": round(_avg("mfe_pct"), 4),
        "avg_pnl_pct": round(_avg("pnl_pct"), 4),
        "avg_give_back_pct": round(avg_gb, 4),
        "avg_forgone_gain_pct": round(avg_fg, 4),
        "forgone_horizon_days": horizon,
        # Coverage and mean forgone gain at EVERY horizon, over the weighted
        # (snapshot-bearing) set. Makes the horizon decision inspectable: if
        # forgone gain climbs steeply with horizon, the short window was hiding
        # winner-harvesting.
        "forgone_by_horizon": {
            f"{h}d": {
                "n": sum(1 for r in snapshot_rows
                         if r.get(forgone_column(h)) is not None),
                "avg": (round(sum(r[forgone_column(h)] for r in snapshot_rows
                                  if r.get(forgone_column(h)) is not None)
                              / max(1, sum(1 for r in snapshot_rows
                                           if r.get(forgone_column(h)) is not None)), 4)
                        if any(r.get(forgone_column(h)) is not None
                               for r in snapshot_rows)
                        else None),
            } for h in all_horizons
        },
        "mean_error_pp": round(mean_error, 4),
        "direction": direction_label,
        "round_trips": round_trips,
        "roundtrip_rate": round(roundtrip_rate, 4),
        "severity": round(severity, 4),
        "bounds": [lo, hi],
        "cumulative_band": band,
        "current_value": current,
        "max_change_frac": PARAM_MAX_CHANGE_FRAC,
        "gate_reason": gate_reason,
        "starvation": starvation,
        "evidence_hash": _evidence_hash(
            [r["trade_id"] for r in rows], current,
            current if gated else proposed_value),
    }
    return {
        "axis": axis,
        "path": path,
        "computed_score": computed_score,
        "sample_size": n,
        "evidence": evidence,
        "current_value": current,
        "proposed_value": proposed_value,
        "gated": gated,
        "gate_reason": gate_reason,
    }


# ── Propose: write a 'proposed' param row (does NOT touch config) ─────

# Keys propose_all_params projects out of every propose_param_update return.
# Kept as one list so the projection and the builder cannot disagree.
PARAM_PROPOSAL_SUMMARY_KEYS = (
    "axis", "path", "history_id", "computed_score", "sample_size",
    "prior_weight", "proposed_delta", "new_weight", "gated", "skipped",
    "deferred", "deferral", "material", "materiality_threshold",
)


def _param_proposal_result(result: dict, run_id: str, history_id: int | None,
                           prior_weight: float, new_weight: float,
                           proposed_delta: float,
                           skipped: str | None = None,
                           deferred: str | None = None) -> dict:
    """The ONE shape every propose_param_update return path emits.

    propose_all_params projects a FIXED key set out of this dict, so a path that
    omits a key does not degrade — it raises KeyError, which the per-param
    try/except swallows and reports to Slack as an *error*. That is how the
    freshness skip (a healthy "nothing new to say") came to be indistinguishable
    from a broken param loop. Both paths now build their result here.
    """
    return {
        "history_id": history_id,
        # None when a row was actually written; else the reason it was not.
        "skipped": skipped,
        # Set when the evidence cleared every gate but the move is withheld for
        # one run because a COUPLED sibling is moving. Distinct from `gated`
        # (insufficient/contradictory evidence) and from `skipped` (nothing new
        # to say) — all three read differently on the card, by design.
        "deferred": deferred,
        "deferral": (result.get("evidence") or {}).get("deferred"),
        "axis": result["axis"],
        "path": result["path"],
        "run_id": run_id,
        "computed_score": result["computed_score"],
        "sample_size": result["sample_size"],
        "prior_weight": prior_weight,
        "proposed_delta": proposed_delta,
        "new_weight": new_weight,
        "gated": result["gated"],
        "gate_reason": result.get("gate_reason"),
        # Big enough to raise an interactive card. Immaterial != gated: the
        # row is written and the evidence routed either way.
        "material": bool((result.get("evidence") or {}).get(
            "materiality", {}).get("material")),
        "materiality_threshold": (result.get("evidence") or {}).get(
            "materiality", {}).get("threshold"),
        "evidence": result["evidence"],
        "evidence_hash": result["evidence"].get("evidence_hash"),
    }


def propose_param_update(path: str, run_id: str | None = None,
                         defer_reason: str | None = None,
                         defer_detail: dict | None = None) -> dict:
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

    # Endpoints are decided BEFORE the freshness gate so both return paths can
    # report the same fields (see _param_proposal_result).
    if gated or current is None or proposed is None:
        prior_weight = current if current is not None else 0.0
        new_weight = prior_weight
        proposed_delta = 0.0
    else:
        prior_weight = current
        new_weight = proposed
        proposed_delta = round(new_weight - prior_weight, 6)

    # ── Coupled-family deferral (Part 7) ────────────────────────────
    # NOT a gate: the evidence cleared every gate and the computed move stands.
    # It is withheld for ONE run because a sibling parameter is moving, and
    # applying both would apply two changes computed from routing that was
    # valid only before either landed. The row is still written — with a ZERO
    # delta, so approving it cannot move the value — and the move it would have
    # made is recorded so next run can be compared against it.
    if defer_reason and not gated:
        result["evidence"]["deferred"] = {
            "reason": defer_reason,
            "would_have_been": proposed,
            "from_value": current,
            "would_have_moved": round((proposed - current), 6)
                                if (proposed is not None and current is not None)
                                else None,
            **(defer_detail or {}),
        }
        new_weight = prior_weight
        proposed_delta = 0.0
        proposed = prior_weight

    # ── Materiality (2026-09-09) ────────────────────────────────────
    # Computed on the FINAL delta, i.e. after the coupled-family deferral has
    # had its say — a deferred row carries a zero delta and is therefore
    # neither material nor guard-bound here, because it is already withheld
    # for its own reason and reads as "deferred", not "too small".
    #
    # Same contract as the axis path: an immaterial move is recorded but not
    # surfaced, and the bound guard is what makes "not surfaced" also mean
    # "not auto-appliable". Param axes are not in AUTO_APPLY_AXES today, but
    # recording it keeps the two paths honest about the same thing, so adding
    # a param to that list later cannot silently bypass the human gate.
    _material = is_material(axis, prior_weight, proposed_delta)
    result["evidence"]["materiality"] = {
        "threshold": round(materiality_threshold(axis, prior_weight), 6),
        "delta": round(abs(proposed_delta), 6),
        "material": bool(_material),
        "basis": f"{PARAM_MATERIALITY_FRAC:.0%} of the current value",
    }
    if not _material and abs(proposed_delta) > 1e-9 and not gated:
        result["evidence"].setdefault("guards", {})["materiality"] = {
            "bound": True,
            "reason": (f"|Δ| {abs(proposed_delta):.6f} < materiality threshold "
                       f"{materiality_threshold(axis, prior_weight):.6f} — "
                       f"recorded, not surfaced, and NOT auto-appliable"),
            "would_have_been": proposed,
        }

    # ── Freshness gate ───────────────────────────────────────────────
    # If the contributing corpus and both endpoints are identical to the most
    # recent recorded proposal, there is no new information — writing another
    # row would just re-arm the same change against a human who has already
    # seen it. Checked BEFORE the supersede below so a standing proposal is
    # left intact rather than replaced by its own twin.
    new_hash = result["evidence"].get("evidence_hash")
    try:
        from kairos_log_db import get_connection as _gc
        _c = _gc()
        try:
            prev = _c.execute(
                "SELECT evidence FROM axis_weight_history WHERE axis = ? "
                "ORDER BY id DESC LIMIT 1", (axis,)).fetchone()
        finally:
            _c.close()
        if prev is not None and prev["evidence"]:
            _prev_ev = json.loads(prev["evidence"]) or {}
            prev_hash = _prev_ev.get("evidence_hash")
            # A DEFERRED standing row is not an equivalent proposal — it is an
            # explicit NON-change (zero delta), written because a coupled
            # sibling went first. Applying the winner does not re-route
            # already-stamped closes, so this parameter's pool is byte-identical
            # next run and the hash matches. Letting the freshness gate fire on
            # that would lock the deferred move out permanently: it would never
            # be proposed again until unrelated new closes arrived, which turns
            # "deferred for one run" into "silently discarded".
            #
            # Re-proposing the real move IS new information relative to what
            # the human last saw, so the gate is bypassed for exactly this
            # case. Nothing is loosened: the ±25% step cap, the cumulative
            # 7-day band and the human gate all still apply, and the band is
            # precisely the guard that bounds a SEQUENCE of coupled steps.
            _prev_deferred = bool(_prev_ev.get("deferred"))
            if (prev_hash and new_hash and prev_hash == new_hash
                    and not _prev_deferred):
                # Nothing proposed this run: the standing row stands untouched,
                # so this run's own delta is zero against the current value.
                return _param_proposal_result(
                    result, run_id, history_id=None,
                    prior_weight=prior_weight, new_weight=prior_weight,
                    proposed_delta=0.0, skipped="no new evidence")
    except Exception:
        # A freshness check that cannot run must not block proposing; the
        # human gate and the cumulative band are the load-bearing guards.
        pass

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

    return _param_proposal_result(
        result, run_id, history_id=history_id, prior_weight=prior_weight,
        new_weight=new_weight, proposed_delta=proposed_delta,
        deferred=defer_reason if not result["gated"] else None)


def propose_all_params(run_id: str | None = None) -> dict:
    """Write a fresh 'proposed' row for every whitelisted exit param, one Slack note.

    The param counterpart to propose_all: iterate PARAM_WHITELIST, propose each
    (min_sample-gated, supersede-then-insert), collect the summaries, and post ONE
    combined note to ARBITER_CHANNEL via the env-first Slack path. Approves nothing
    and writes no config. Each param is isolated in its own try/except so one
    failure cannot block the others. Returns
    {run_id, proposals, errors, slack_text, slack_posted}.
    """
    if PROPOSALS_FROZEN:
        _p = f"proposals FROZEN pending evidence redesign ({PROPOSALS_FROZEN}); compute skipped."
        print(_p)
        return {"frozen": True, "reason": PROPOSALS_FROZEN}
    run_id = run_id or f"{_today_et()}_weekly"
    proposals: list[dict] = []
    errors: list[dict] = []

    # ── Coupled-family mutual exclusion (Part 7) ────────────────────
    # atr_mult / trail_lo_pct / trail_hi_pct are coupled THROUGH the bind-state
    # routing: raising trail_lo_pct moves closes out of the free bucket and
    # into the floor bucket, which shrinks atr_mult's pool and grows
    # trail_lo_pct's. The Arbiter models each parameter independently and has
    # no representation of that.
    #
    # The coupling itself is not the danger. The danger is APPLYING two
    # proposals that were each computed from routing valid only BEFORE either
    # landed — after the first apply, the second one's premise is stale, and
    # nothing in the pipeline would notice. So at most one of the three carries
    # a live delta per run; the rest are deferred and recomputed next run
    # against the routing that then exists.
    #
    # Deliberately NOT a joint optimisation over the three. At 3-8 closes a
    # week there is not enough evidence to fit a three-parameter surface, and
    # attempting it would produce confident noise — which is worse than a slow
    # sequence of individually-evidenced steps.
    #
    # Decided on a DRY compute pass first: compute_param is deterministic on an
    # unchanged corpus (the recency anchor is the newest contributing close,
    # not wall-clock), so the recompute inside propose_param_update reproduces
    # these numbers exactly.
    deferrals: dict = {}
    coupled_eval: list[dict] = []
    for path in PARAM_COUPLED_FAMILY:
        if path not in PARAM_WHITELIST:
            continue
        try:
            r = compute_param(path)
        except Exception as exc:
            print(f"  propose_all_params: coupled pre-check {path} failed: {exc}",
                  file=sys.stderr)
            continue
        moves = (not r["gated"] and r["current_value"] is not None
                 and r["proposed_value"] is not None
                 and abs(r["proposed_value"] - r["current_value"]) > 1e-9)
        coupled_eval.append({
            "path": path, "moves": moves,
            "effective_n": (r["evidence"] or {}).get("effective_n") or 0.0,
            "proposed": r["proposed_value"], "current": r["current_value"],
        })
    contenders = [c for c in coupled_eval if c["moves"]]
    coupled_decision = {
        "family": list(PARAM_COUPLED_FAMILY),
        "evaluated": coupled_eval,
        "n_contenders": len(contenders),
        "winner": None,
        "deferred": [],
    }
    if len(contenders) > 1:
        # Highest effective_n wins: the best-supported change goes first.
        winner = max(contenders, key=lambda c: c["effective_n"])
        coupled_decision["winner"] = winner["path"]
        short = lambda q: q.rsplit(".", 1)[-1]  # noqa: E731
        for c in contenders:
            if c["path"] == winner["path"]:
                continue
            deferrals[c["path"]] = (
                f"deferred this run — {short(winner['path'])} moved and "
                f"changes which trades this parameter governs")
            coupled_decision["deferred"].append({
                "path": c["path"],
                "effective_n": round(c["effective_n"], 4),
                "would_have_been": c["proposed"],
            })
    elif len(contenders) == 1:
        coupled_decision["winner"] = contenders[0]["path"]

    for path in PARAM_WHITELIST:
        try:
            _dr = deferrals.get(path)
            p = propose_param_update(
                path, run_id=run_id, defer_reason=_dr,
                defer_detail={
                    "coupled_family": list(PARAM_COUPLED_FAMILY),
                    "applied_instead": coupled_decision["winner"],
                    "winner_effective_n": next(
                        (round(c["effective_n"], 4) for c in contenders
                         if c["path"] == coupled_decision["winner"]), None),
                    "recompute": "next run, against the post-apply routing",
                } if _dr else None)
            summary = {k: p[k] for k in PARAM_PROPOSAL_SUMMARY_KEYS}
            # Starvation diagnostics ride along so the Slack note can say WHY a
            # sample is 0 rather than just that it is.
            summary["evidence"] = p.get("evidence") or {}
            proposals.append(summary)
        except Exception as exc:
            errors.append({"axis": PARAM_PREFIX + path, "error": str(exc)})
            print(f"  propose_all_params: {path} failed: {exc}", file=sys.stderr)

    slack_text = _format_slack_propose_params(run_id, proposals, errors,
                                              coupled_decision)
    slack_posted = _post_slack(slack_text)
    return {
        "run_id": run_id,
        "proposals": proposals,
        "errors": errors,
        "coupled_decision": coupled_decision,
        "slack_text": slack_text,
        "slack_posted": slack_posted,
    }


def _format_slack_propose_params(run_id: str, proposals: list[dict],
                                 errors: list[dict],
                                 coupled_decision: dict | None = None) -> str:
    """One combined Slack summary for a param propose run (human gate unchanged)."""
    lines = [
        f":wrench: *Weekly exit-parameter proposals* — run {run_id}",
        "_From TRAILING-STOP closed-trade outcomes. Nothing applied — 'proposed' "
        "rows only; approval writes kairos_config.json behind the human gate._",
    ]
    for p in proposals:
        n = p["sample_size"]
        # Plain-English block first — same rationale as the axis card: the
        # reviewer needs the meaning before the internals. Degrades to raw
        # numbers only if the renderer fails; never breaks the proposal loop.
        try:
            from kairos_arbiter_explain import explain_param_proposal
            lines.extend(explain_param_proposal(p))
        except Exception as _exc:
            lines.append(f"  _(plain-English render unavailable: {_exc})_")

        # Skipped is its OWN bucket. It used to surface as an error (KeyError on
        # the summary projection) or get read as a gate; it is neither — it means
        # the standing proposal still stands and nothing changed underneath it.
        if p.get("skipped"):
            detail = f"skipped — {p['skipped']} (n={n}); standing proposal unchanged"
        elif p.get("deferred"):
            # Worded so it cannot be misread as a gate failure. The evidence
            # WAS sufficient and the move is real; it is queued behind a
            # coupled sibling, not rejected.
            _d = p.get("deferral") or {}
            _wb = _d.get("would_have_been")
            detail = (f"*{p['deferred']}*"
                      + (f" (evidence supports {p['prior_weight']:g} → "
                         f"{_wb:g}, effective n={n} — held for one run, "
                         f"recomputed next run against the new routing)"
                         if _wb is not None else ""))
        elif p["gated"]:
            detail = (f"gated — no usable evidence (effective n {n}); "
                      f"no change (stays {p['prior_weight']:g})")
        elif not p.get("material", True):
            detail = (f"recorded, no card — |Δ| "
                      f"{abs(p['proposed_delta']):g} below the "
                      f"{PARAM_MATERIALITY_FRAC:.0%} materiality threshold "
                      f"({p['prior_weight']:g} → {p['new_weight']:g}); "
                      f"review with `--pending-minor`, cannot auto-apply")
        elif abs(p["proposed_delta"]) < 1e-9:
            detail = (f"no change (stays {p['new_weight']:g}; "
                      f"score {p['computed_score']:+.4f}, n={n})")
        else:
            detail = (f"{p['prior_weight']:g} → *{p['new_weight']:g}* "
                      f"(Δ {p['proposed_delta']:+g}; score {p['computed_score']:+.4f}, "
                      f"n={n})  [id {p['history_id']}]")
        lines.append(f"  • `{p['path']}`: {detail}")

        # ── Part 4(d): evidence routing, on every card ──────────────
        _rt = ((p.get("evidence") or {}).get("routing") or {})
        if _rt:
            _bs = _rt.get("by_bind_state") or {}
            _bits = (f"{_bs.get('free', 0)} free, {_bs.get('floor', 0)} "
                     f"floor-bound, {_bs.get('ceiling', 0)} ceiling-bound")
            _extra = []
            if _bs.get("no_atr"):
                _extra.append(f"{_bs['no_atr']} no-ATR (fallback trail)")
            if _bs.get("unattributed"):
                _extra.append(f"{_bs['unattributed']} unattributed "
                              f"(closed before bind-state stamping)")
            if _extra:
                _bits += ", " + ", ".join(_extra)
            if _rt.get("routed"):
                lines.append(
                    f"      _{_rt['n_before_routing']} contributing closes "
                    f"({_bits}); computed from the "
                    f"{_rt['governs_bind_state']}-bound subset "
                    f"({_rt['n_after_routing']} closes)_")
            else:
                lines.append(
                    f"      _{_rt['n_before_routing']} contributing closes "
                    f"({_bits}); not clamp-routed — this parameter governs "
                    f"every close_")

        # An empty sample is never self-explanatory — say why, every time, and
        # escalate the marker when it has been empty across real trading.
        if n == 0:
            starve = (p.get("evidence") or {}).get("starvation") or {}
            why = starve.get("summary")
            streak = (starve.get("zero_sample_streak") or {})
            # Red only when it is unambiguous: repeatedly empty AND trades
            # closed in between. One empty run on a young regime is a warning.
            marker = (":rotating_light:"
                      if streak.get("runs", 0) >= 2 and streak.get("closes_since")
                      else ":warning:")
            lines.append(f"      {marker} sample is 0"
                         + (f" — {why}" if why else ""))

    if coupled_decision and coupled_decision.get("n_contenders", 0) > 1:
        _w = (coupled_decision["winner"] or "?").rsplit(".", 1)[-1]
        _defs = ", ".join("`" + d["path"].rsplit(".", 1)[-1] + "`"
                          for d in coupled_decision["deferred"])
        lines.append("")
        lines.append(
            f":link: *Coupled family — one change per run.* "
            f"{coupled_decision['n_contenders']} of "
            f"{len(coupled_decision['family'])} qualified; carried `{_w}` "
            f"(highest effective n) and deferred {_defs}.")
        lines.append(
            "_Moving any one of atr_mult / trail_lo_pct / trail_hi_pct "
            "re-routes which closes the other two learn from, so applying two "
            "proposals computed before either landed would act on a stale "
            "premise. The deferred ones are recomputed next run — this is not "
            "a gate failure._")
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

    # Ordered-pair refusal, checked BEFORE the cumulative band. Both refuse a
    # crossing value, but this message is the more specific diagnosis — "a
    # floor above the ceiling is not a band" tells you the structure is wrong,
    # where the band message only says the move was too big. Since the band
    # tightened to 0.15 it now pre-empts most crossings, so without this
    # reordering the structural error would always be reported as a drift
    # error. Nothing changes about WHAT is refused, only which reason is given.
    cross = _ordering_violation(path, value)
    if cross is not None:
        raise ValueError(f"{path}: {cross} — refusing to apply")

    # Cumulative window band. This is the guard the ratchet defeated: each of
    # its three steps was inside ±25% of the value the step before had written,
    # so the per-step check passed every time while the parameter halved. The
    # band is anchored at the value in force when the window opened, so no
    # sequence of individually-legal steps can drift past it.
    band = _cumulative_band(PARAM_PREFIX + path, current)
    if band["lo"] is not None and not (band["lo"] - 1e-9 <= value <= band["hi"] + 1e-9):
        raise ValueError(
            f"{path}={value} violates the cumulative {PARAM_CUMULATIVE_WINDOW_DAYS}-day "
            f"band [{band['lo']}, {band['hi']}] anchored at {band['base_7d']} "
            f"(±{PARAM_CUMULATIVE_BAND_FRAC:.0%}) — refusing to apply")

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

def format_minor_digest(proposals: list) -> str | None:
    """One line summarising the sub-threshold proposals, or None if there are none.

    Goes into the daily Arbiter post so that recorded-but-unsurfaced proposals
    are visible without each one interrupting. The largest is named because
    that is the only one a reader might want to go look at.
    """
    minor = [p for p in (proposals or [])
             if not p.get("gated") and not p.get("skipped")
             and not p.get("deferred")
             and abs(p.get("proposed_delta") or 0.0) > 1e-9
             and not p.get("material")]
    if not minor:
        return None

    def _rel(p):
        prior = abs(float(p.get("prior_weight") or 0.0))
        d = abs(float(p["proposed_delta"]))
        return (d / prior) if prior > 0 else d

    biggest = max(minor, key=_rel)
    name = (biggest.get("path") or biggest.get("axis", "?")).rsplit(".", 1)[-1]
    prior = abs(float(biggest.get("prior_weight") or 0.0))
    d = float(biggest["proposed_delta"])
    size = (f"{d / prior * 100:+.1f}%" if prior > 0 else f"{d:+.4f}")
    return (f":memo: {len(minor)} sub-threshold proposal(s) recorded, no card "
            f"raised (largest: `{name}` {size}). Review with "
            f"`--pending-minor`; none can auto-apply.")


def pending_minor(limit: int = 50) -> list:
    """Standing 'proposed' rows that were recorded but never surfaced.

    Reads the materiality block written at propose time rather than
    re-deriving the threshold, so this reports what was ACTUALLY decided at
    the time — the threshold or the current value may have moved since.
    """
    from kairos_log_db import get_connection, init_db
    init_db()
    conn = get_connection()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM axis_weight_history WHERE status = 'proposed' "
            "ORDER BY id DESC LIMIT ?", (limit,))]
    finally:
        conn.close()
    out = []
    for r in rows:
        try:
            ev = json.loads(r["evidence"] or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            ev = {}
        m = ev.get("materiality") or {}
        if m.get("material") is False and abs(r["proposed_delta"] or 0.0) > 1e-9:
            out.append({
                "id": r["id"], "axis": r["axis"], "run_id": r["run_id"],
                "prior_weight": r["prior_weight"],
                "new_weight": r["new_weight"],
                "proposed_delta": r["proposed_delta"],
                "threshold": m.get("threshold"),
                "effective_n": ev.get("effective_n"),
                "confidence": ev.get("confidence"),
                "created_at": r["created_at"],
                "auto_apply_blocked": bool(
                    (ev.get("guards") or {}).get("materiality", {}).get("bound")),
            })
    return out


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
    gated = (" _(GATED: no usable evidence → Δ forced to 0)_"
             if p["gated"] else
             ("" if p.get("material", True) else
              " _(recorded only — below the materiality threshold, no card)_"))
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
    _eff = ev.get("effective_n")
    _conf = ev.get("confidence")
    lines.append(f"sample_size: *{p['sample_size']}*{gated}"
                 + (f"  (effective {_eff:.2f} of {ev.get('n_matured', '?')} "
                    f"matured closes; confidence ×{_conf:.2f})"
                    if _eff is not None and _conf is not None else ""))
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
            f"  mean per-trade error:   {ev['mean_error_pp']:+.2f}pp  (n_matured={ev['n_matured']}, weighted)",
            f"  effective sample:       {ev.get('effective_n')}  "
            f"(exact-match {ev.get('n_exact_match')}, "
            f"down-weighted {ev.get('n_downweighted')})",
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
                          result["sample_size"], cfg,
                          effective_n=result.get("effective_n"))

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
        print(f"  sample_size (effective):              {result['sample_size']}"
              f"  (min_sample={cfg['min_sample']})")
        print(f"  effective_n / n_matured:              "
              f"{ev.get('effective_n')} / {ev.get('n_matured')}"
              f"  (exact-match {ev.get('n_exact_match')}, "
              f"down-weighted {ev.get('n_downweighted')})")
        print(f"  confidence (× on the step):           "
              f"{delta.get('confidence', 0.0):.4f}")
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
        print(f"  would-be Δ: 0.000000 (GATED — no usable evidence; "
              f"thin evidence would propose proportionally)")
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


def _cli_pending_minor() -> int:
    rows = pending_minor()
    print("  Sub-threshold proposals — recorded, NOT surfaced, NOT auto-appliable.")
    print(f"  Card thresholds: params {PARAM_MATERIALITY_FRAC:.0%} of the current "
          f"value; axes {AXIS_MATERIALITY_ABS} absolute.\n")
    if not rows:
        print("  None standing.")
        return 0
    print(f"  {'id':>5} {'axis':<48}{'prior':>10}{'->new':>10}{'delta':>10}"
          f"{'thresh':>9}{'eff_n':>8}{'conf':>7}  blocked")
    print("  " + "-" * 112)
    for r in rows:
        eff = r["effective_n"]
        conf = r["confidence"]
        print(f"  {r['id']:>5} {r['axis']:<48}{(r['prior_weight'] or 0):>10.4f}"
              f"{(r['new_weight'] or 0):>10.4f}{(r['proposed_delta'] or 0):>+10.4f}"
              f"{(r['threshold'] or 0):>9.4f}"
              f"{(eff if eff is not None else float('nan')):>8.2f}"
              f"{(conf if conf is not None else float('nan')):>7.3f}"
              f"  {'yes' if r['auto_apply_blocked'] else 'NO — CHECK'}")
    print("  " + "-" * 112)
    print(f"  {len(rows)} standing. Approve one deliberately with "
          f"`--approve <ID>` if you want it applied.")
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
                      f"avg_give_back={ev.get('avg_give_back_pct')}pp "
                      f"[weighted]")
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
    print(f"     effective_n={ev.get('effective_n')} "
          f"(min_sample={PARAM_MIN_SAMPLE})  "
          f"exact_match_n={ev.get('n_exact_match')}  "
          f"contributing={ev.get('n_contributing')}  "
          f"confidence={ev.get('confidence')}")
    print(f"     avg_mfe={ev['avg_mfe_pct']}pp  avg_pnl={ev['avg_pnl_pct']}pp  "
          f"avg_give_back={ev['avg_give_back_pct']}pp  "
          f"avg_forgone={ev['avg_forgone_gain_pct']}pp "
          f"({ev.get('forgone_horizon_days')}d)   [weighted means]")
    print(f"     round_trips={ev['round_trips']} (weighted rate "
          f"{ev['roundtrip_rate']})  severity={ev['severity']}  "
          f"computed_score={r['computed_score']:+.4f}")
    w = ev.get("weighting") or {}
    if w.get("by_param_value"):
        print(f"     evidence weights (half-life "
              f"{w.get('recency_half_life_days')}d, floor "
              f"{w.get('proximity_floor')}, anchor {w.get('recency_anchor')}):")
        for b in w["by_param_value"]:
            print(f"        value={b['value']:<9} n={b['n']:<3} "
                  f"prox={b['proximity']:.3f}  mean_recency={b['mean_recency']:.3f}"
                  f"  eff_n={b['effective_n']:.3f}"
                  f"{'  <- exact match' if b['exact_match'] else ''}")
    if r["gated"]:
        print(f"     GATED — no usable evidence (not thin evidence); no change, "
              f"proposed stays {r['proposed_value']}")
        print(f"       reason: {r.get('gate_reason')}")
    else:
        d = r["proposed_value"] - r["current_value"]
        th = materiality_threshold(PARAM_PREFIX + r["path"], r["current_value"])
        mat = is_material(PARAM_PREFIX + r["path"], r["current_value"], d)
        print(f"     proposed_value={r['proposed_value']} (Δ {d:+g})")
        print(f"       materiality: |Δ| {abs(d):.4f} vs threshold {th:.4f} "
              f"({PARAM_MATERIALITY_FRAC:.0%} of current) -> "
              f"{'CARD RAISED' if mat else 'recorded only, no card'}")


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
        #    Schema must carry trade_id, a forgone-gain column and an
        #    exit_params_snapshot: compute_param needs both poles of the
        #    objective and a regime stamp to weigh a row at all. Snapshots are
        #    stamped at the LIVE config values so these rows sit at proximity
        #    1.0 and the proposal is driven by the evidence, not the distance.
        ML_DB_PATH = os.path.join(tmpdir, "ml.db")
        with open(CONFIG_PATH) as f:
            _live_cfg = json.load(f)
        _snap = json.dumps({
            "params": {pp: _get_dotted(_live_cfg, pp)[0]
                       for pp in PARAM_WHITELIST},
            "axis_weights": {"exit_timing": 0.0},
            "reconstructed": True,
        })
        mc = sqlite3.connect(ML_DB_PATH)
        mc.execute("CREATE TABLE trade_outcomes (trade_id TEXT, pnl_pct REAL, "
                   "mfe_pct REAL, give_back_pct REAL, forgone_gain_5d_pct REAL, "
                   "forgone_gain_14d_pct REAL, post_exit_peak_pct REAL, "
                   "exit_reason TEXT, timestamp_exit TEXT, "
                   "exit_params_snapshot TEXT)")
        for i in range(15):
            # winners that round-tripped: big mfe, small/negative pnl → big give-back
            mc.execute("INSERT INTO trade_outcomes VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (f"st{i:03d}", -1.0 if i % 2 else 2.0, 22.0, 21.0,
                        2.0, 2.0, 2.0,
                        "TRAILING-STOP: retreated 18% from peak 22%",
                        "2026-06-01 00:00:00 UTC", _snap))
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
            # within bounds [4,12] but far more than a 25% jump from current
            _apply_param_to_config("exits.trailing_stop.target_armed.trail_pct", 11.9)
            check(f"25% cap: _apply >25% jump ({cur}→11.9) raises", False)
        except ValueError:
            check(f"25% cap: _apply >25% jump ({cur}→11.9) raises", True)

        # ── Full propose → approve cycle ─────────────────────────────
        p = propose_param_update("exits.trailing_stop.target_armed.trail_pct")
        check("weighting: evidence is weighted, and the effective sample is "
              "reported alongside the exact-match count",
              trail["evidence"]["effective_n"] > 0
              and trail["evidence"]["n_exact_match"] == 15
              and trail["evidence"]["n_excluded_out_of_regime"] == 0)
        check("weighting: step is scaled by a confidence factor < 1",
              0.0 < trail["evidence"]["confidence"] < 1.0)
        check("propose: non-gated proposal written (effective n>=min_sample)",
              not p["gated"])
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
    group.add_argument("--pending-minor", action="store_true",
                       help="List recorded-but-unsurfaced (sub-materiality) "
                            "proposals. These raised no card and cannot "
                            "auto-apply; this is how you review them.")
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
    if args.pending_minor:
        return _cli_pending_minor()
    if args.approve is not None:
        return _cli_decide(args.approve, "approve", args.by)
    if args.reject is not None:
        return _cli_decide(args.reject, "reject", args.by)
    return 1


if __name__ == "__main__":
    sys.exit(main())
