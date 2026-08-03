"""
kairos_slack_cards.py — Block Kit approval cards for axis-weight / exit-param
proposals, plus the curl-based post/update helpers.

Shared by kairos_commander (Socket Mode button handlers + !pending) and
kairos_arbiter (daily/weekly card posting). Card *building* (pure, no I/O) is
kept separate from *posting* so the Block Kit JSON can be dry-tested without
touching Slack.

Transport: every Slack write routes through kairos_alerts._slack_api_call
(curl), because this host's endpoint filter blocks the Python socket layer for
Slack's IP range — slack_sdk / Bolt's own HTTP client cannot reach Slack here.
Bolt's Socket Mode websocket, when it connects, is used ONLY to *receive* button
clicks and ack() them; the message post/update always goes over curl.

No circular imports: this module imports kairos_axis_weights and kairos_alerts
(neither imports this module). commander and arbiter import this module.
"""

import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Action IDs — must match the Bolt handlers registered in kairos_commander.
ACTION_APPROVE = "kairos_approve"
ACTION_REJECT = "kairos_reject"

# Weight axes (non-param). param axes carry the 'param:' prefix.
_PARAM_PREFIX = "param:"


# ── Time / user display ──────────────────────────────────────────────

def _now_et_str() -> str:
    """Human 'YYYY-MM-DD HH:MM ET' timestamp for outcome lines."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        return now.strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def slack_user_display(token: str, user_id: str, fallback: str = "") -> str:
    """Resolve a Slack user's real name via users.info (curl). Falls back to the
    supplied fallback (username/id) on any error — never raises."""
    fb = fallback or user_id or "unknown"
    if not token or not user_id:
        return fb
    try:
        from kairos_alerts import _slack_api_call
        resp = _slack_api_call("users.info", token, {"user": user_id})
        if resp.get("ok"):
            u = resp.get("user", {}) or {}
            prof = u.get("profile", {}) or {}
            return (prof.get("real_name") or u.get("real_name")
                    or u.get("name") or fb)
    except Exception:
        pass
    return fb


# ── Normalized proposal loading (reuses the --review data path) ──────

def _is_param(axis: str) -> bool:
    return isinstance(axis, str) and axis.startswith(_PARAM_PREFIX)


def _param_path(axis: str) -> str:
    return axis[len(_PARAM_PREFIX):] if _is_param(axis) else ""


def _row_to_proposal(row) -> dict:
    """sqlite Row → normalized proposal dict (evidence parsed to a dict)."""
    try:
        ev = json.loads(row["evidence"]) if row["evidence"] else {}
    except (TypeError, ValueError):
        ev = {}
    axis = row["axis"]
    return {
        "id": row["id"],
        "history_id": row["id"],
        "axis": axis,
        "is_param": _is_param(axis),
        "path": _param_path(axis),
        "run_id": row["run_id"],
        "computed_score": row["computed_score"],
        "sample_size": row["sample_size"],
        "prior_weight": row["prior_weight"],
        "proposed_delta": row["proposed_delta"],
        "new_weight": row["new_weight"],
        "status": row["status"],
        "created_at": row["created_at"],
        "evidence": ev,
    }


def load_proposal(history_id: int):
    """Load one proposal (any status) as a normalized dict, or None if missing."""
    from kairos_log_db import get_connection, init_db
    init_db()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM axis_weight_history WHERE id = ?", (history_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_proposal(row) if row is not None else None


def list_pending_proposals() -> list:
    """All 'proposed' rows as normalized dicts, oldest first — the same data
    path used by --review / _cli_review."""
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
    return [_row_to_proposal(r) for r in rows]


# ── Value / guardrail formatting ─────────────────────────────────────

def _fmt_value(is_param: bool, value) -> str:
    if value is None:
        return "—"
    return f"{value:g}" if is_param else f"{value:+.4f}"


def _guardrail_note(prop: dict) -> str:
    """Human note on the bounds + cap that kairos_axis_weights enforces at APPLY
    time (this module never re-checks; the apply fn is the single source)."""
    if prop["is_param"]:
        bounds = (prop.get("evidence") or {}).get("bounds")
        if not bounds:
            try:
                from kairos_axis_weights import PARAM_WHITELIST
                bounds = PARAM_WHITELIST.get(prop["path"])
            except Exception:
                bounds = None
        b = f"[{bounds[0]:g}, {bounds[1]:g}]" if bounds else "whitelisted range"
        note = f"stays within {b} · at most ±25% per step · at most ±40% over any 7 days"
        return note
    try:
        from kairos_axis_weights import _DEFAULT_CFG, load_config
        cfg = {**_DEFAULT_CFG, **(load_config() or {})}
    except Exception:
        cfg = {"weight_bound": 1.0, "per_run_cap": 0.15}
    return (f"weight bound ±{cfg.get('weight_bound', 1.0):g} · "
            f"per-run cap ±{cfg.get('per_run_cap', 0.15):g}")


# ── Plain-language translation (phone-approvable copy) ───────────────
# Jargon is translated ONCE here so the card never shows a raw feature name:
#   mfe_pct       → "peak gain while held"
#   give_back     → "gain given back from the peak"
#   forgone_gain  → "gains missed after selling"

def _pct(v, sign=False) -> str:
    if v is None:
        return "—"
    try:
        return (f"{v:+.1f}%" if sign else f"{v:.1f}%")
    except (TypeError, ValueError):
        return str(v)


def _param_label(path: str) -> str:
    if path.endswith("trail_pct"):
        return "trailing stop"
    if path.endswith("profit_floor_pp"):
        return "profit floor"
    return path.rsplit(".", 1)[-1]


def _param_headline(path: str, cur, new, direction, changed: bool) -> str:
    label = _param_label(path)
    cur_s, new_s = _pct(cur), _pct(new)
    if not changed:
        return f"🔧 No change to the {label} (stays {cur_s}) — not enough fresh evidence yet"
    if label == "trailing stop":
        if direction == "tighten":
            return f"🔧 Tighten the {label}: {cur_s} → {new_s} (sell sooner to keep more of the peak)"
        return f"🔧 Loosen the {label}: {cur_s} → {new_s} (let winners run further before selling)"
    # profit floor
    if direction == "tighten":
        return f"🔧 Raise the {label}: {cur_s} → {new_s} (lock in a bit more before an armed winner can exit)"
    return f"🔧 Lower the {label}: {cur_s} → {new_s} (give armed winners more room)"


def _param_why(ev: dict, cur, changed: bool) -> str:
    eff = ev.get("effect") or {}
    curr = eff.get("current") or {}
    n = curr.get("n", ev.get("n_contributing", 0))
    gb = curr.get("avg_give_back_pp")
    fg = curr.get("avg_forgone_5d_pp")
    since = eff.get("since_date")
    since_s = (str(since)[:10] if since else "the last change")
    lines = []
    if n and gb is not None and fg is not None:
        lines.append(
            f"Since {since_s}, *{n}* trades closed under the current {_pct(cur)} setting: "
            f"they gave back *{_pct(gb)}* from their peaks (gain given back from the peak) "
            f"but missed only *{_pct(fg)}* of gains after selling (gains missed after selling).")
    prior = eff.get("prior")
    pv = eff.get("prior_value")
    if prior and prior.get("n"):
        lines.append(
            f"Under the previous {_pct(pv)} setting it was {_pct(prior.get('avg_give_back_pp'))} "
            f"given back / {_pct(prior.get('avg_forgone_5d_pp'))} missed across {prior.get('n')} trades.")
    elif pv is not None:
        lines.append(
            f"There isn't matured data yet under the previous {_pct(pv)} setting to compare against.")
    if not lines:
        need = ""
        gr = ev.get("gate_reason") or ""
        return ("Not enough fresh trades have closed under the current setting with complete "
                "peak/after-sale data to justify a change yet." + (f" ({gr})" if gr else ""))
    # Direction rationale
    if changed:
        if (gb or 0) >= (fg or 0):
            lines.append("Giving back more than we miss means winners are round-tripping — "
                         "tightening should keep more of each peak.")
        else:
            lines.append("Missing more than we give back means we're selling too early — "
                         "loosening should let winners run.")
    return "\n".join(lines)


def _param_effect_lines(path: str, cur, new, direction, changed: bool) -> tuple:
    label = _param_label(path)
    if not changed:
        return (None,
                f"*If you do nothing:* it stays {_pct(cur)}. I'll only re-propose if new "
                f"closed trades change the picture.")
    if label == "trailing stop":
        approve = (f"*If you approve:* armed winners sell after a *{_pct(new)}* pullback from "
                   f"their peak instead of {_pct(cur)}. Live on the next exit-engine run.")
    else:
        approve = (f"*If you approve:* the profit floor moves to *{_pct(new)}*, changing how much "
                   f"an armed winner locks in before it can exit. Live on the next exit-engine run.")
    do_nothing = (f"*If you do nothing:* it stays {_pct(cur)}. I'll only re-propose if new closed "
                  f"trades change the picture.")
    return (approve, do_nothing)


def _weight_headline(axis: str, prior, new, changed: bool) -> str:
    if axis == "exit_timing":
        if not changed:
            return "⚖️ No change to the Council's exit-timing lean — not enough fresh evidence yet"
        earlier = (new is not None and prior is not None and new > prior)
        which = "earlier" if earlier else "later"
        return f"⚖️ Lean the Council slightly more toward {which} exits ({prior:+.2f} → {new:+.2f})"
    verb = "No change to" if not changed else "Adjust"
    return f"⚖️ {verb} `{axis}` ({_fmt_value(False, prior)} → {_fmt_value(False, new)})"


def _weight_why(axis: str, ev: dict, changed: bool) -> str:
    n = ev.get("n_matured", 0)
    gb = ev.get("mean_giveback_pp")
    pk = ev.get("mean_post_exit_peak_pp")
    if not n:
        return ("Not enough trades have closed under the current setting with complete "
                "peak/after-sale data to move it yet.")
    late = (gb or 0) >= (pk or 0)
    tail = ("Net, we've been exiting a touch too late (giving back more than we miss)." if late
            else "Net, we've been exiting a touch too early (missing more than we give back).")
    return (f"Across *{n}* trades under the current setting, holdings gave back *{_pct(gb)}* from "
            f"their peaks (gain given back from the peak) while leaving *{_pct(pk)}* on the table "
            f"after selling (gains missed after selling). {tail}")


# ── Block Kit builders (pure — no Slack I/O) ─────────────────────────

def build_proposal_blocks(proposal: dict) -> list:
    """Return Block Kit blocks for an interactive approval card.

    Accepts either a normalized proposal (from load_proposal /
    list_pending_proposals) or a summary dict carrying at least
    history_id/id + axis + prior_weight + new_weight; missing fields degrade
    gracefully. The proposal id is included both as a visible field and in the
    button values.
    """
    hid = proposal.get("id") or proposal.get("history_id")
    axis = proposal.get("axis", "?")
    is_param = proposal.get("is_param", _is_param(axis))
    path = proposal.get("path") or (_param_path(axis) if is_param else axis)
    ev = proposal.get("evidence") or {}

    prior_v = proposal.get("prior_weight")
    new_v = proposal.get("new_weight")
    delta = proposal.get("proposed_delta")
    changed = delta is not None and abs(delta) > 1e-9

    # Plain-English headline: the ACTION, in words a phone-tapper can act on.
    if is_param:
        direction = ev.get("direction") or ("tighten" if (delta or 0) and (
            (path.endswith("trail_pct") and delta < 0) or
            (path.endswith("profit_floor_pp") and delta > 0)) else "loosen")
        headline = _param_headline(path, prior_v, new_v, direction, changed)
        why = _param_why(ev, prior_v, changed)
        approve_line, do_nothing_line = _param_effect_lines(path, prior_v, new_v, direction, changed)
    else:
        headline = _weight_headline(axis, prior_v, new_v, changed)
        why = _weight_why(axis, ev, changed)
        if changed:
            approve_line = ("*If you approve:* the Council's exit-timing lean shifts "
                            f"to {_fmt_value(False, new_v)} (observed only — not yet injected "
                            "into any live decision).")
        else:
            approve_line = None
        do_nothing_line = ("*If you do nothing:* the setting stays put. I'll only re-propose "
                           "if new closed trades change the picture.")

    blocks = [
        {"type": "header",
         "text": {"type": "plain_text", "text": headline[:150], "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Why:* {why}"}},
    ]

    effect_bits = [ln for ln in (approve_line, do_nothing_line) if ln]
    if effect_bits:
        blocks.append({"type": "section",
                       "text": {"type": "mrkdwn", "text": "\n".join(effect_bits)}})

    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": (
        f"Proposal `{hid}` · guardrails: {_guardrail_note({**proposal, 'is_param': is_param, 'path': path})} "
        f"· enforced at apply time. Nothing changes until you tap a button."
    )}]})

    blocks.append({
        "type": "actions",
        "block_id": f"kairos_proposal_{hid}",
        "elements": [
            {"type": "button", "action_id": ACTION_APPROVE, "style": "primary",
             "text": {"type": "plain_text", "text": "Approve", "emoji": True},
             "value": str(hid)},
            {"type": "button", "action_id": ACTION_REJECT, "style": "danger",
             "text": {"type": "plain_text", "text": "Reject", "emoji": True},
             "value": str(hid)},
        ],
    })
    return blocks


def _fallback_text(proposal: dict) -> str:
    hid = proposal.get("id") or proposal.get("history_id")
    axis = proposal.get("axis", "?")
    is_param = proposal.get("is_param", _is_param(axis))
    path = proposal.get("path") or (_param_path(axis) if is_param else axis)
    delta = proposal.get("proposed_delta")
    changed = delta is not None and abs(delta) > 1e-9
    if is_param:
        ev = proposal.get("evidence") or {}
        direction = ev.get("direction") or "adjust"
        head = _param_headline(path, proposal.get("prior_weight"),
                               proposal.get("new_weight"), direction, changed)
    else:
        head = _weight_headline(axis, proposal.get("prior_weight"),
                                proposal.get("new_weight"), changed)
    # Strip any leading non-letter/emoji prefix for the plain-text fallback.
    head = head.lstrip("🔧⚖️️ ").strip()
    return f"{head} (proposal {hid}) — Approve/Reject."


def build_outcome_blocks(proposal: dict, headline: str, subtext: str = "") -> list:
    """Blocks for the in-place message update after a decision. Preserves the
    FULL original proposal detail (evidence, sample size, guardrails, score) so
    the audit trail stays visible in channel history — only the Approve/Reject
    action block is removed, and the outcome banner is prepended on top."""
    # Outcome banner first
    blocks = [{"type": "section",
               "text": {"type": "mrkdwn", "text": headline}}]
    if subtext:
        blocks.append({"type": "context",
                       "elements": [{"type": "mrkdwn", "text": subtext}]})
    blocks.append({"type": "divider"})
    # Re-render the original detail, then drop the interactive buttons so the
    # evidence stays readable but the card is no longer actionable.
    try:
        detail = build_proposal_blocks(proposal or {})
        detail = [b for b in detail if b.get("type") != "actions"]
        blocks.extend(detail)
    except Exception:
        # If anything about the original dict is missing, degrade to the banner
        # only rather than raise — the outcome line is the essential part.
        pass
    return blocks


# ── Posting / updating (curl transport) ──────────────────────────────

def post_proposal_card(proposal_dict: dict, channel: str, cfg: dict = None) -> bool:
    """Post an interactive approval card to `channel`. Enriches from the DB when
    the passed dict lacks evidence/status (so callers can pass a lightweight
    summary). Returns True on success, never raises."""
    hid = proposal_dict.get("id") or proposal_dict.get("history_id")
    prop = proposal_dict
    if hid is not None and "evidence" not in proposal_dict:
        full = load_proposal(hid)
        if full is not None:
            prop = full
    blocks = build_proposal_blocks(prop)
    text = _fallback_text(prop)
    try:
        from kairos_alerts import post_message, _load_slack_config
        if cfg is None:
            cfg = _load_slack_config()
        return post_message(channel, text, blocks=blocks, cfg=cfg)
    except Exception as exc:
        print(f"  post_proposal_card failed: {exc}", file=sys.stderr)
        return False


def update_proposal_message(token: str, channel: str, ts: str, proposal: dict,
                            decision: str = None, decided_by: str = "",
                            outcome: dict = None, warn: str = "") -> bool:
    """Update the original card in place (chat.update via curl) to show the
    outcome. On warn, the value is left unchanged and a ⚠️ line is shown.

    Returns True if the update posted, False otherwise. Never raises.
    """
    hid = proposal.get("id") or proposal.get("history_id") if proposal else None
    is_param = proposal.get("is_param") if proposal else False
    path = (proposal.get("path") or proposal.get("axis")) if proposal else "?"

    if warn:
        headline = f"⚠️ {warn}"
        subtext = f"Proposal `{hid}` — value unchanged."
    elif decision == "approve":
        val = None
        if outcome is not None:
            val = outcome.get("new_value", outcome.get("new_weight"))
        val_s = _fmt_value(is_param, val) if val is not None else "updated"
        headline = (f"✅ *Approved* by {decided_by} at {_now_et_str()} — "
                    f"`{path}` is now *{val_s}*")
        subtext = (f"Proposal `{hid}` applied by kairos_axis_weights "
                   f"(guardrails enforced).")
    elif decision == "reject":
        headline = (f"🚫 *Rejected* by {decided_by} at {_now_et_str()} — "
                    f"`{path}` unchanged")
        subtext = f"Proposal `{hid}` marked rejected. Nothing applied."
    else:
        headline = f"Proposal `{hid}` handled."
        subtext = ""

    blocks = build_outcome_blocks(proposal or {}, headline, subtext)
    text = headline
    try:
        from kairos_alerts import _slack_api_call
        payload = {"channel": channel, "ts": ts, "text": text, "blocks": blocks}
        resp = _slack_api_call("chat.update", token, payload)
        if not resp.get("ok"):
            print(f"  chat.update failed: {resp.get('error')}", file=sys.stderr)
            return False
        return True
    except Exception as exc:
        print(f"  update_proposal_message failed: {exc}", file=sys.stderr)
        return False
