"""Plain-English rendering of Arbiter proposals for the Slack human gate.

WHY THIS EXISTS
The gate is only a safety mechanism if the person operating it can tell a good
proposal from a bad one. The prior card rendered raw internals --

    - `exit_timing`: +0.2794 -> *+0.2423* (d -0.0371; score +0.1558, n=42)

-- which states what moved but not what it MEANS, what evidence drove it, how
big it is relative to the allowed range, or what breaks if it is wrong. A
reviewer who cannot answer those cannot meaningfully withhold approval, so the
gate degrades into a rubber stamp and the ratchet postmortem's "human approval
gate is unchanged and remains mandatory" stops being a real control.

DESIGN RULES
  * Reuse kairos_reason._AXIS_LEAN_PHRASES for the behavioural translation.
    That mapping is what actually renders the weight into the Claude prompt, so
    sourcing the Slack wording from the same table means the card cannot drift
    from what the system really does. Fall back to neutral phrasing if the
    import fails -- a formatting helper must never break the proposal loop.
  * State evidence in the units the reviewer sees on the dashboard
    (percentage points of give-back / forgone gain), never model-internal
    scores.
  * Always state the counter-risk. A proposal that only lists reasons to say
    yes is an argument, not a briefing.
  * Always state what the evidence EXCLUDES. "42 trades" reads very
    differently once you know 177 were excluded for closing under different
    settings.
"""

from __future__ import annotations

# Neutral fallback used when an axis has no entry in _AXIS_LEAN_PHRASES.
_GENERIC_LEAN = ("increase this axis's influence", "decrease this axis's influence")


def _lean_phrases(axis: str) -> tuple[str, str]:
    """(positive_lean, negative_lean) for an axis, sourced from the prompt table."""
    try:
        from kairos_reason import _AXIS_LEAN_PHRASES
        phrases = _AXIS_LEAN_PHRASES.get(axis)
        if phrases and len(phrases) == 2:
            return phrases[0], phrases[1]
    except Exception:
        pass
    return _GENERIC_LEAN


def _magnitude(delta: float, bound: float = 1.0) -> str:
    """Size of a move as a share of the full allowed range, in words."""
    if not bound:
        return "unknown size"
    frac = abs(delta) / (2.0 * bound)          # full range is -bound..+bound
    pct = frac * 100.0
    if pct < 2:
        word = "very small"
    elif pct < 5:
        word = "small"
    elif pct < 12:
        word = "moderate"
    else:
        word = "large"
    return f"{word} - {pct:.1f}% of the full range"


# Counter-risk of moving each axis further in the direction it is being moved.
_AXIS_RISK = {
    "exit_timing": (
        "exits trigger too early and cut winners short",
        "positions are held past their peak and give back more",
    ),
    "conviction_calibration": (
        "genuinely strong setups get discounted and are under-sized",
        "over-confident conviction scores stop being corrected",
    ),
    "reallocation_aggressiveness": (
        "capital sits in decaying positions too long",
        "positions are rotated out before the thesis has played out",
    ),
}


def explain_axis_proposal(p: dict) -> list[str]:
    """Slack lines explaining ONE axis proposal in reviewable terms."""
    axis = p.get("axis", "?")
    ev = p.get("evidence") or {}
    prior = p.get("prior_weight") or 0.0
    new = p.get("new_weight") or 0.0
    delta = p.get("proposed_delta") or 0.0
    n = p.get("sample_size") or 0

    pos_lean, neg_lean = _lean_phrases(axis)
    risk_pos, risk_neg = _AXIS_RISK.get(
        axis, ("this axis is over-applied", "this axis is under-applied"))

    lines: list[str] = [f":robot_face: *{axis}* - proposed change"]

    # -- What actually changes, in behavioural terms --
    if abs(delta) < 1e-9:
        lines.append(f"   *What changes:* nothing - weight stays {new:+.4f}")
    else:
        # The weight is an offset: its SIGN says which lean is active, and the
        # direction of travel says whether that lean strengthens or softens.
        active = pos_lean if new >= 0 else neg_lean
        verb = "more" if abs(new) > abs(prior) else "less"
        lines.append(f"   *What changes:* {verb} inclined to {active}")
        lines.append(f"      weight {prior:+.4f} -> *{new:+.4f}*   "
                     f"(d {delta:+.4f}, {_magnitude(delta)})")
    return lines + _axis_evidence_lines(p, ev, n, delta, new,
                                        risk_pos, risk_neg)


def _axis_evidence_lines(p: dict, ev: dict, n: int, delta: float,
                         new: float, risk_pos: str, risk_neg: str) -> list[str]:
    """Evidence, counter-risk, and sample provenance for an axis proposal."""
    lines: list[str] = []

    give = ev.get("mean_giveback_pp")
    peak = ev.get("mean_post_exit_peak_pp")
    err = ev.get("mean_error_pp")
    if n and give is not None and peak is not None:
        lines.append(f"   *Why:* across {n} matured trade(s) since the last change")
        lines.append(f"      - gave back *{give:.1f}pp* from peak before exiting")
        lines.append(f"      - price ran a further *{peak:.1f}pp* after exiting")
        if err is not None:
            side = "too late" if err > 0 else "too early"
            lines.append(f"      - net: exiting *{side}* by *{abs(err):.1f}pp* "
                         f"per trade on average")

        # Reconcile a reading that looks self-contradictory but is not.
        # The weight is an EMA that TRACKS the measured bias, so when the bias
        # is real but MILDER than the weight currently encodes, the correct
        # move is to soften the lean while keeping its sign. Without this line
        # the card reads "still exiting too late" next to "lean less toward
        # earlier exits" and looks like it is correcting backwards -- which
        # would earn a wrong rejection, or worse, quiet distrust of the loop.
        score = p.get("computed_score")
        prior = p.get("prior_weight")
        if (score is not None and prior is not None
                and abs(delta) > 1e-9 and abs(score) < abs(prior)):
            lines.append(f"      _the bias is real but milder than the current "
                         f"weight assumes (measured {score:+.2f} vs weight "
                         f"{prior:+.2f}), so the lean softens without flipping_")

    if abs(delta) > 1e-9:
        lines.append(f"   *If this is wrong:* {risk_pos if new >= 0 else risk_neg}")

    total = ev.get("n_total_closed")
    excl = ev.get("n_excluded_out_of_regime")
    pend = ev.get("n_pending")
    if total:
        bits = [f"{n} of {total} closed trades"]
        if excl:
            bits.append(f"{excl} excluded (closed under different settings)")
        if pend:
            bits.append(f"{pend} still maturing")
        lines.append(f"   _Evidence base: {'; '.join(bits)}_")
    return lines


def explain_param_proposal(p: dict) -> list[str]:
    """Slack lines explaining ONE exit-parameter proposal."""
    path = p.get("path") or p.get("axis", "?")
    ev = p.get("evidence") or {}
    cur = ev.get("current_value")
    new = p.get("proposed_value", p.get("new_weight"))
    n = ev.get("n") or p.get("sample_size") or 0

    lines: list[str] = [f":robot_face: *{path}* - proposed change"]

    direction = ev.get("direction")
    if cur is not None and new is not None and abs((new or 0) - (cur or 0)) > 1e-9:
        human = {"tighten": "exit sooner / protect gains earlier",
                 "loosen": "give positions more room to run"}.get(
                     direction, direction or "adjust")
        lines.append(f"   *What changes:* {human}")
        lines.append(f"      value {cur:.4f} -> *{new:.4f}*")
    else:
        lines.append(f"   *What changes:* nothing - value stays {cur}")

    mfe = ev.get("avg_mfe_pct")
    pnl = ev.get("avg_pnl_pct")
    give = ev.get("avg_give_back_pct")
    forgone = ev.get("avg_forgone_gain_pct")
    horizon = ev.get("forgone_horizon_days")
    rt = ev.get("roundtrip_rate")
    if n and mfe is not None and pnl is not None:
        lines.append(f"   *Why:* across {n} qualifying trailing-stop exit(s)")
        lines.append(f"      - peaked at *+{mfe:.1f}%* but closed at *{pnl:+.1f}%*")
        if give is not None:
            lines.append(f"      - gave back *{give:.1f}pp* from peak")
        if forgone is not None:
            lines.append(f"      - only *{forgone:.1f}pp* more was available in "
                         f"the {horizon or '?'}d after exit")
        if rt is not None:
            lines.append(f"      - *{rt * 100:.0f}%* round-tripped from winner to loss")
    return lines + _param_tail_lines(ev, n, direction)


def _param_tail_lines(ev: dict, n: int, direction) -> list[str]:
    """Counter-risk, sample provenance, and gate reason for a param proposal."""
    lines: list[str] = []

    if direction == "tighten":
        lines.append("   *If this is wrong:* stops trigger on normal noise "
                     "and cut runners short")
    elif direction == "loosen":
        lines.append("   *If this is wrong:* more of each winner is given back "
                     "before the stop fires")

    tot = ev.get("n_total_trailing_stop")
    excl = ev.get("n_excluded_out_of_regime")
    nonc = ev.get("n_excluded_non_contributing")
    if tot:
        bits = [f"{n} of {tot} trailing-stop closes"]
        if excl:
            bits.append(f"{excl} closed under different settings")
        if nonc:
            bits.append(f"{nonc} non-contributing")
        lines.append(f"   _Evidence base: {'; '.join(bits)}_")

    gate = ev.get("gate_reason")
    if gate:
        lines.append(f"   :no_entry: _gated - {gate}_")
    return lines
