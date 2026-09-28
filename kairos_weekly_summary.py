"""Kairos Weekly Summary — the 60-second Sunday check-in.

WHY THIS EXISTS
With J away, there are exactly two things he should have to read: a rare
`<!here>` from kairos_watch.py when something needs a decision, and this. The
design target is literal — sixty seconds, on a phone, and at the end of it he
knows whether to open a laptop. That constrains the content more than it
sounds: everything here is either a number that changed or a thing waiting on
him. Anything that is merely interesting belongs in the daily Arbiter post.

Seven sections, in the order a reader actually triages them:
  1. what needs you        — the only section that can require action
  2. what the loop did     — auto-applied and rolled back, unattended
  3. parameters now vs 7d  — the drift the ratchet postmortem is about
  4. realised P&L          — did the week make money (total AND median)
  5. exits by type         — WHICH mechanism produced that P&L, n/total/median
  6. open book             — position count, so an emptying book is visible
  7. activity              — fills, so a silent week is visible as a number

It must stand alone without the dashboard. J is at low touch for 3-6 months
and will not be opening it, so anything that only exists there is, for this
period, information that does not exist.

Read-only. It reads kairos.db and kairos_ml_outcomes.db and posts to Slack.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import statistics
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:                                    # pragma: no cover
    ET = timezone(timedelta(hours=-4))

# The learning-loop channel, deliberately not #kairos-alerts. Alerts is now
# reserved for "a human must decide something", and putting a routine weekly
# post there would dilute exactly the signal kairos_watch.py is trying to
# protect. #kairos-arbiter is where auto-apply, rollback and proposal traffic
# already live, so the weekly lands in context.
SUMMARY_CHANNEL = "#kairos-arbiter"
WINDOW_DAYS = 7


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw):
    """One parser for both databases.

    kairos.db writes '2026-09-10 00:00:51 UTC'; kairos_ml_outcomes.db writes
    '2026-09-11T14:01:53Z'. Reusing kairos_axis_weights._parse_exit_ts, which
    already normalises all three shapes the corpus contains, rather than
    keeping a second half-complete parser here — the first draft of this file
    handled only the kairos.db shape and silently reported "no positions
    closed this week" against a week with 36 fills.
    """
    from kairos_axis_weights import _parse_exit_ts
    return _parse_exit_ts(raw)


def _kairos_db() -> sqlite3.Connection:
    import kairos_log_db as kdb
    return kdb.get_connection()


def _ml_db() -> sqlite3.Connection:
    from kairos_ml_outcomes import DB_PATH
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _short(path: str) -> str:
    return path.rsplit(".", 1)[-1]


# ── Sections ─────────────────────────────────────────────────────────

def autonomy_activity(since: datetime) -> dict:
    """What the loop did to itself this week, without asking."""
    conn = _kairos_db()
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM autonomy_log")]
    finally:
        conn.close()
    applied, reverted, watching = [], [], []
    for r in rows:
        at = _parse_ts(r["applied_at"])
        vt = _parse_ts(r["verdict_at"])
        if at is not None and at >= since:
            applied.append(r)
        if vt is not None and vt >= since and r["verdict"] in (
                "rolled_back", "rollback_failed", "apply_failed"):
            reverted.append(r)
        if r["verdict"] == "pending":
            watching.append(r)
    return {"applied": applied, "reverted": reverted, "watching": watching}


def param_drift(since: datetime) -> list:
    """Every learnable param: value now vs the value in force 7 days ago.

    Derived from the approved-history trail rather than from config backups.
    The earliest approval inside the window carries the value that was live
    when the window opened, which is precisely the cumulative band's own
    anchor — so this table and the guard are describing the same quantity.
    """
    from kairos_axis_weights import PARAM_WHITELIST, _current_param_value

    conn = _kairos_db()
    try:
        hist = [dict(r) for r in conn.execute(
            "SELECT axis, prior_weight, new_weight, decided_at, decided_by "
            "FROM axis_weight_history WHERE status = 'approved' "
            "ORDER BY id ASC")]
    finally:
        conn.close()

    out = []
    for path in PARAM_WHITELIST:
        axis = "param:" + path
        now = _current_param_value(path)
        in_window = [h for h in hist if h["axis"] == axis
                     and (_parse_ts(h["decided_at"]) or _now_utc()) >= since]
        was = in_window[0]["prior_weight"] if in_window else now
        out.append({
            "path": path, "was": was, "now": now,
            "changes": len(in_window),
            "by": sorted({h["decided_by"] or "?" for h in in_window}),
        })
    return out


def axis_drift(since: datetime) -> list:
    conn = _kairos_db()
    try:
        weights = {r["axis"]: r["weight"] for r in conn.execute(
            "SELECT axis, weight FROM axis_weights")}
        hist = [dict(r) for r in conn.execute(
            "SELECT axis, prior_weight, decided_at, decided_by "
            "FROM axis_weight_history WHERE status = 'approved' "
            "AND axis NOT LIKE 'param:%' ORDER BY id ASC")]
    finally:
        conn.close()
    out = []
    for axis, now in sorted(weights.items()):
        in_window = [h for h in hist if h["axis"] == axis
                     and (_parse_ts(h["decided_at"]) or _now_utc()) >= since]
        was = in_window[0]["prior_weight"] if in_window else now
        out.append({"axis": axis, "was": was, "now": now,
                    "changes": len(in_window),
                    "by": sorted({h["decided_by"] or "?" for h in in_window})})
    return out


def realized_pnl(since: datetime) -> dict:
    """Closed-trade P&L and the exit-type breakdown that produced it.

    Bucketed on the exit_reason PREFIX rather than the full string, because
    the reason carries per-trade detail ("TRAILING-STOP: retreated 4.1%") that
    would make every trade its own bucket. The prefix is the mechanism, which
    is the thing worth attributing a week's P&L to.
    """
    conn = _ml_db()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT pnl_dollar, pnl_pct, exit_reason, ticker, timestamp_exit "
            "FROM trade_outcomes WHERE timestamp_exit IS NOT NULL")]
    finally:
        conn.close()
    rows = [r for r in rows
            if (_parse_ts(r["timestamp_exit"]) or datetime.min.replace(
                tzinfo=timezone.utc)) >= since]

    total = sum(r["pnl_dollar"] or 0.0 for r in rows)
    wins = [r for r in rows if (r["pnl_dollar"] or 0.0) > 0]
    buckets: dict = {}
    for r in rows:
        key = re.split(r"[:(]", r["exit_reason"] or "UNKNOWN",
                       maxsplit=1)[0].strip()
        b = buckets.setdefault(key, {"n": 0, "pnl": 0.0, "_d": [], "_p": []})
        b["n"] += 1
        b["pnl"] += r["pnl_dollar"] or 0.0
        b["_d"].append(r["pnl_dollar"] or 0.0)
        b["_p"].append(r["pnl_pct"] or 0.0)
    # A median alongside the total is the whole point of this section at low
    # touch. A bucket's total is dominated by its largest trade — PRICE-
    # INVALIDATION can read as catastrophic on one -$3k outlier while the
    # typical exit is flat — and "is this mechanism systematically bad or did
    # it have one bad day" is exactly the question the total cannot answer and
    # the median can.
    for b in buckets.values():
        b["median_pnl"] = statistics.median(b.pop("_d"))
        b["median_pct"] = statistics.median(b.pop("_p"))
    return {
        "n": len(rows), "total": total,
        "median_pnl": statistics.median([r["pnl_dollar"] or 0.0 for r in rows])
                      if rows else None,
        "median_pct": statistics.median([r["pnl_pct"] or 0.0 for r in rows])
                      if rows else None,
        "win_rate": (len(wins) / len(rows)) if rows else None,
        "best": max(rows, key=lambda r: r["pnl_dollar"] or 0.0) if rows else None,
        "worst": min(rows, key=lambda r: r["pnl_dollar"] or 0.0) if rows else None,
        "buckets": sorted(buckets.items(), key=lambda kv: kv[1]["pnl"], reverse=True),
    }


def open_book() -> dict:
    """Open position count and unrealised P&L — the state the week left behind.

    Included because at low touch this post is the only routine picture of the
    book. A week can show flat realised P&L while the open side quietly grows
    or empties out, and "56 positions" versus "12 positions" is the difference
    between a system that is running and one that has stopped opening.
    """
    conn = _kairos_db()
    try:
        n = conn.execute(
            "SELECT COUNT(*) c FROM holdings WHERE sold_date IS NULL").fetchone()["c"]
        tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) c FROM holdings "
            "WHERE sold_date IS NULL").fetchone()["c"]
        snap = conn.execute(
            "SELECT snapshot_date, num_positions, unrealized_pnl "
            "FROM nlv_snapshots ORDER BY snapshot_date DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    return {"lots": n, "tickers": tickers,
            "snapshot_date": (snap["snapshot_date"] if snap else None),
            "snapshot_positions": (snap["num_positions"] if snap else None),
            "unrealized": (snap["unrealized_pnl"] if snap else None)}


def activity(since: datetime) -> dict:
    conn = _kairos_db()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT substr(timestamp,1,10) d, execution_status s, COUNT(*) n "
            "FROM decisions WHERE substr(timestamp,1,10) >= ? GROUP BY d, s",
            (since.date().isoformat(),))]
        nlv = conn.execute(
            "SELECT net_liq_after v, timestamp t FROM decisions "
            "WHERE net_liq_after IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    fills = sum(r["n"] for r in rows if r["s"] == "Filled")
    fill_days = len({r["d"] for r in rows if r["s"] == "Filled" and r["n"]})
    return {"fills": fills, "fill_days": fill_days,
            "decisions": sum(r["n"] for r in rows),
            "nlv": (nlv["v"] if nlv else None),
            "nlv_at": (nlv["t"] if nlv else None)}


def pending_review() -> dict:
    """Everything standing that a human — and only a human — can clear."""
    from kairos_axis_weights import escalating_guards, pending_minor
    try:
        from kairos_autonomy import AUTO_APPLY_AXES
    except Exception:
        AUTO_APPLY_AXES = []

    conn = _kairos_db()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM axis_weight_history WHERE status = 'proposed'")]
    finally:
        conn.close()
    cards, guarded, self_applying = [], [], []
    for r in rows:
        if abs(r["proposed_delta"] or 0.0) < 1e-9:
            continue
        try:
            ev = json.loads(r["evidence"] or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            ev = {}
        bound = escalating_guards(ev)
        if bound:
            # A guard bound: a human decides regardless of who owns the axis.
            guarded.append((r, bound))
        elif r["axis"] in AUTO_APPLY_AXES:
            # Standing but not waiting on anyone — the next Arbiter run applies
            # it. Listing it under "needs you" would be the reverse of the
            # mistake Part 2 fixed: asking for a decision nobody has to make.
            self_applying.append(r)
        elif (ev.get("materiality") or {}).get("material"):
            cards.append(r)
    return {"cards": cards, "guarded": guarded,
            "self_applying": self_applying, "minor": pending_minor()}


# ── Rendering ────────────────────────────────────────────────────────

def _money(v) -> str:
    return "—" if v is None else f"${v:,.0f}"


def _num(v, places: int = 4) -> str:
    return "—" if v is None else f"{float(v):.{places}g}"


def build(now_et=None) -> str:
    now_et = now_et or datetime.now(ET)
    since = _now_utc() - timedelta(days=WINDOW_DAYS)
    au = autonomy_activity(since)
    pend = pending_review()
    pnl = realized_pnl(since)
    act = activity(since)

    L = [f":calendar: *Kairos — week ending {now_et.strftime('%Y-%m-%d')}*",
         f"_{(now_et - timedelta(days=WINDOW_DAYS)).strftime('%b %d')} – "
         f"{now_et.strftime('%b %d')}  |  NLV {_money(act['nlv'])}_",
         "─" * 40]

    # 1 — what needs you (first, because it is the only actionable part)
    n_need = len(pend["cards"]) + len(pend["guarded"])
    L.append("*1. Needs you*")
    if not n_need:
        L.append("  Nothing. No proposal is waiting on a decision.")
    else:
        for r, g in pend["guarded"]:
            L.append(f"  :warning: `{r['axis']}` id {r['id']} — held by "
                     f"{', '.join(sorted(g))} (ratchet guard). "
                     f"{_num(r['prior_weight'])} → {_num(r['new_weight'])}")
        for r in pend["cards"]:
            L.append(f"  :envelope: `{r['axis']}` id {r['id']} — card raised, "
                     f"{_num(r['prior_weight'])} → {_num(r['new_weight'])} "
                     f"(Δ {r['proposed_delta']:+.4g})")
    if pend["self_applying"]:
        L.append(f"  _{len(pend['self_applying'])} proposal(s) standing on an "
                 f"autonomous axis — they apply on the next Arbiter run, no "
                 f"decision needed:_ "
                 + ", ".join(f"`{_short(r['axis'])}` "
                             f"{_num(r['prior_weight'])} → {_num(r['new_weight'])}"
                             for r in pend["self_applying"]))
    if pend["minor"]:
        parked = [m for m in pend["minor"] if m["auto_apply_blocked"]]
        L.append(f"  _{len(pend['minor'])} sub-threshold proposal(s) recorded; "
                 f"{len(parked)} parked, the rest applied themselves. "
                 f"`--pending-minor` to see them._")

    # 2 — what the loop did unattended
    L.append("\n*2. Applied without asking*")
    if not au["applied"] and not au["reverted"]:
        L.append("  Nothing auto-applied and nothing rolled back this week.")
    for r in au["applied"]:
        L.append(f"  :robot_face: `{r['axis']}` {_num(r['prior_weight'])} → "
                 f"{_num(r['new_weight'])}  ({r['applied_at']}) — "
                 f"verdict *{r['verdict']}*")
    for r in au["reverted"]:
        icon = ":rewind:" if r["verdict"] == "rolled_back" else ":rotating_light:"
        L.append(f"  {icon} *{r['verdict']}* `{r['axis']}` back to "
                 f"{_num(r['prior_weight'])} — {r['note'] or ''}")
    if au["watching"]:
        L.append(f"  _{len(au['watching'])} change(s) still accumulating closes "
                 f"before they can be judged._")

    # 3 — parameter drift, the ratchet's own metric
    L.append("\n*3. Parameters — now vs 7 days ago*")
    for p in param_drift(since):
        arrow = ("no change" if p["was"] is None or p["now"] is None
                 or abs(float(p["was"]) - float(p["now"])) < 1e-9
                 else f"{_num(p['was'])} → *{_num(p['now'])}*"
                      f"  ({len(p['by']) and ', '.join(p['by']) or ''})")
        L.append(f"  `{_short(p['path']):<14}` {arrow}")
    for a in axis_drift(since):
        arrow = ("no change" if a["was"] is None or a["now"] is None
                 or abs(float(a["was"]) - float(a["now"])) < 1e-9
                 else f"{_num(a['was'])} → *{_num(a['now'])}*")
        L.append(f"  `{a['axis']:<14}` {arrow}")

    # 4 — did the week make money
    L.append("\n*4. Realised P&L*")
    if not pnl["n"]:
        L.append("  No positions closed this week.")
    else:
        L.append(f"  *{_money(pnl['total'])}* over {pnl['n']} closes, "
                 f"win rate {pnl['win_rate']:.0%}")
        L.append(f"  median close {_money(pnl['median_pnl'])} "
                 f"({pnl['median_pct']:+.2f}%) — read this against the total: "
                 f"a healthy median under a bad total means one outlier, not a "
                 f"broken mechanism")
        if pnl["best"] is not None:
            L.append(f"  best `{pnl['best']['ticker']}` "
                     f"{_money(pnl['best']['pnl_dollar'])}  |  "
                     f"worst `{pnl['worst']['ticker']}` "
                     f"{_money(pnl['worst']['pnl_dollar'])}")

    # 5 — which mechanism produced it
    L.append("\n*5. Exits by type*  _(n · total · median)_")
    if not pnl["buckets"]:
        L.append("  —")
    for name, b in pnl["buckets"]:
        L.append(f"  `{name[:24]:<24}` {b['n']:>3}  "
                 f"{_money(b['pnl']):>11}   med {_money(b['median_pnl']):>8} "
                 f"({b['median_pct']:+.1f}%)")

    # 6 — the book the week left behind
    ob = open_book()
    L.append("\n*6. Open book*")
    L.append(f"  *{ob['tickers']}* open positions ({ob['lots']} lots)"
             + (f"; unrealised {_money(ob['unrealized'])} as of "
                f"{ob['snapshot_date']}" if ob["unrealized"] is not None else ""))

    # 7 — activity, so a silent week reads as a number not an absence
    L.append("\n*7. Activity*")
    L.append(f"  {act['fills']} fills across {act['fill_days']} day(s); "
             f"{act['decisions']} decisions evaluated.")
    if act["fills"] == 0:
        L.append("  :rotating_light: *Zero fills this week* — see "
                 "#kairos-alerts.")

    L.append("\n_Weekly digest. Anything needing a decision also fires an "
             "`@here` in #kairos-alerts as it happens._")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Kairos weekly Sunday summary")
    ap.add_argument("--dry-run", action="store_true",
                    help="Render to stdout; post nothing")
    ap.add_argument("--channel", default=SUMMARY_CHANNEL)
    args = ap.parse_args()

    if not os.environ.get("SLACK_BOT_TOKEN"):
        try:
            with open(os.path.expanduser("~/.zshrc")) as fh:
                for line in fh:
                    m = re.match(r"^\s*export\s+SLACK_BOT_TOKEN=(.*)$", line)
                    if m:
                        v = m.group(1).strip().strip('"').strip("'")
                        if v:
                            os.environ["SLACK_BOT_TOKEN"] = v
                        break
        except Exception:
            pass

    text = build()
    print(text)
    if args.dry_run:
        return 0
    from kairos_alerts import post_message
    ok = post_message(args.channel, text)
    print(f"\n  Posted to {args.channel}: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
