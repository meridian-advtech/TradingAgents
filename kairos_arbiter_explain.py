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
    no_snap = ev.get("n_excluded_no_snapshot")
    pend = ev.get("n_pending")
    eff = ev.get("effective_n")
    matured = ev.get("n_matured")
    down = ev.get("n_downweighted")
    if total:
        # Evidence is weighted, not filtered (2026-09-09), so the provenance
        # line has to say "counted less" rather than "excluded" or a reviewer
        # will read a small effective sample as a small corpus.
        if eff is not None and matured:
            bits = [f"{matured} of {total} closed trades, weighted to an "
                    f"effective {eff:.1f}"]
        else:
            bits = [f"{n} of {total} closed trades"]
        if down:
            bits.append(f"{down} closed under different settings "
                        f"(counted less, not dropped)")
        if excl:
            bits.append(f"{excl} excluded (closed under different settings)")
        if no_snap:
            bits.append(f"{no_snap} unusable (no settings snapshot)")
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
    human = {"tighten": "exit sooner / protect gains earlier",
             "loosen": "give positions more room to run"}.get(
                 direction, direction or "adjust")
    # A DEFERRED proposal is not a no-op. "nothing changes" would under-read it:
    # the evidence cleared every gate and the move is real, it is just queued
    # behind a coupled sibling for one run. Say that in the plain-English
    # header, not only in the raw bullet underneath.
    deferral = ev.get("deferred") or {}
    if deferral:
        would = deferral.get("would_have_been")
        lines.append(f"   *What changes:* nothing THIS RUN - deferred, not rejected")
        if cur is not None and would is not None:
            lines.append(f"      the evidence supports {human}: "
                         f"{cur:.4f} -> {would:.4f}")
        sib = (deferral.get("applied_instead") or "").rsplit(".", 1)[-1]
        lines.append(f"      held because `{sib}` moved this run and changes "
                     f"which trades this parameter governs; recomputed next run")
    elif cur is not None and new is not None and abs((new or 0) - (cur or 0)) > 1e-9:
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
        qual = ev.get("n_contributing") or n
        lines.append(f"   *Why:* across {qual} qualifying trailing-stop exit(s), "
                     f"weighted by recency and by how close each one's settings "
                     f"were to today's")
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
    no_snap = ev.get("n_excluded_no_snapshot")
    nonc = ev.get("n_excluded_non_contributing")
    contrib = ev.get("n_contributing")
    eff = ev.get("effective_n")
    down = ev.get("n_downweighted")
    conf = ev.get("confidence")
    if tot:
        if eff is not None and contrib:
            bits = [f"{contrib} of {tot} trailing-stop closes, weighted to an "
                    f"effective {eff:.1f}"]
        else:
            bits = [f"{n} of {tot} trailing-stop closes"]
        if down:
            bits.append(f"{down} closed under different settings "
                        f"(counted less, not dropped)")
        if excl:
            bits.append(f"{excl} closed under different settings")
        if no_snap:
            bits.append(f"{no_snap} unusable (no settings snapshot)")
        if nonc:
            bits.append(f"{nonc} non-contributing")
        lines.append(f"   _Evidence base: {'; '.join(bits)}_")
    if conf is not None and conf < 0.999:
        lines.append(f"   _Step scaled to {conf * 100:.0f}% of full strength "
                     f"by the weight of the evidence behind it_")

    gate = ev.get("gate_reason")
    if gate:
        lines.append(f"   :no_entry: _gated - {gate}_")
    return lines
