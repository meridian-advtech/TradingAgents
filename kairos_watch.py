"""Kairos Watch — the "something needs you" channel.

WHY THIS IS NOT ANOTHER DAILY REPORT
J is stepping back from daily review, so the daily Arbiter post stops being a
control and becomes noise: a report that arrives whether or not anything is
wrong trains its reader to skip it, and a gate nobody reads is not a gate.
This module posts ONLY when a human decision is actually required, prefixed
`<!here>` so it is visually distinct from every routine post in the channel.
Everything else stays in the daily digest and the Sunday summary
(kairos_weekly_summary.py).

The bar for adding a condition here is deliberately high: it must be something
that (a) will not fix itself, (b) is invisible in the routine posts, and (c)
has actually happened or has a specific precedent. All five below clear it.

WHAT IT DOES NOT DO
It does not check infrastructure — LaunchAgents, Ollama, the IB Gateway
socket, scheduler log gaps. kairos_healthcheck.py owns that layer and posts to
the same channel. The split is deliberate and is exactly the 2026-08-10 lesson:
healthcheck asks "is the machinery running", this asks "is the machinery
producing outcomes". Between 2026-08-11 and 2026-08-17 the answer to the first
was yes and the answer to the second was no, and nothing was watching the
second, so a full week passed with zero fills and no alert.

Read-only with respect to trading. It reads kairos.db (incl. the
trade_outcomes view), writes only its own dedupe state file, and posts Slack.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:                                    # pragma: no cover
    ET = timezone(timedelta(hours=-4))

ALERTS_CHANNEL = "alerts"
STATE_FILE = os.path.join(SCRIPT_DIR, ".kairos_watch_state.json")

# ── Thresholds ───────────────────────────────────────────────────────

# Consecutive weekdays with zero fills before this is treated as a failure
# rather than a quiet market.
#
# 3, and the unit is WEEKDAYS rather than exchange trading days on purpose.
# There is no market calendar in this codebase, and building one to be exact
# about a threshold this coarse is the wrong trade. The approximation is safe
# at N=3 specifically: no US market holiday closes three consecutive weekdays,
# so a run of three can never be explained away by the calendar. At N=2 it
# could not make that claim (Thanksgiving Thursday plus a thin Friday), which
# is the real reason for 3 — not the number of quiet days a human would
# tolerate.
NO_TRADE_DAYS = 3

# NLV drawdown from peak that warrants an interrupt.
#
# 8%. The book is ~55-65 long-only equity positions, so its beta to the index
# is close to 1 and ordinary market drawdowns pass straight through: 5% is
# noise J should not be paged for, and paging on noise is how this channel
# stops being read. Ordinary S&P corrections run 5-10%, so 8% is the point
# where "the market did this" stops being the most likely explanation and
# "something in the system did this" becomes worth a look. It is also material
# against the mandate — roughly a third of the 29% annualised floor, i.e. a
# hole that takes a real run to climb out of, while still leaving room to act
# before it becomes one that does not.
NLV_DRAWDOWN_PCT = 8.0

# How long a fired condition stays quiet before it re-fires. A condition that
# is still true tomorrow is not new information; a condition still true next
# week is.
REALERT_AFTER_HOURS = 24

# Trend margin for the early-warning condition. A post-change error 10% above
# baseline is not yet a rollback (that needs ROLLBACK_DEGRADE_FRAC = 20% AND
# ROLLBACK_MIN_CLOSES trades) but it is the shape of one forming.
EARLY_WARNING_FRAC = 0.10


# ── Helpers ──────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _kairos_db() -> sqlite3.Connection:
    import kairos_log_db as kdb
    return kdb.get_connection()


def _ml_db() -> sqlite3.Connection:
    from kairos_ml_outcomes import DB_PATH
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(raw):
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).replace(" UTC", ""),
                                 "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _weekdays_back(end_date, n: int) -> list:
    """The n most recent weekdays ending at (and including) end_date."""
    out, cur = [], end_date
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur -= timedelta(days=1)
    return out


def _short(axis: str) -> str:
    return axis.rsplit(".", 1)[-1] if "." in axis else axis


# ── Condition 1: a rollback fired ────────────────────────────────────

def check_rollback_fired(since_hours: int = 36) -> list:
    """Auto-applied changes reverted (or that FAILED to revert) recently.

    A rollback is the loop admitting a change it made was wrong. That is the
    system working, but it is also the single most decision-relevant thing
    that can happen unattended, and a failed one means a value judged harmful
    is still live.
    """
    cutoff = _now_utc() - timedelta(hours=since_hours)
    conn = _kairos_db()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM autonomy_log "
            "WHERE verdict IN ('rolled_back', 'rollback_failed', 'apply_failed')")]
    finally:
        conn.close()

    out = []
    for r in rows:
        ts = _parse_ts(r["verdict_at"])
        if ts is None or ts < cutoff:
            continue
        failed = r["verdict"] != "rolled_back"
        base, post = r["baseline_error"], r["post_error"]
        moved = (f"error {base:.4f} → {post:.4f}pp"
                 if base is not None and post is not None else "error unmeasurable")
        if failed:
            body = (f"*{r['verdict'].replace('_', ' ').upper()}* on "
                    f"`{r['axis']}` — {moved}.\n"
                    f"It should have gone back to *{r['prior_weight']}* and did "
                    f"not. {r['note'] or ''}\n"
                    f":warning: The value that was judged harmful may still be "
                    f"live. Check kairos_config.json and revert by hand.")
        else:
            # NOT closes_at_apply — that is the corpus size when the change
            # landed, not the sample it was judged on. The judged sample is
            # only guaranteed to be >= ROLLBACK_MIN_CLOSES, so say that.
            body = (f"*Rolled back* `{r['axis']}` — {moved} over at least "
                    f"{_min_closes()} subsequent closes, which is worse "
                    f"than the {_degrade_pct()}% degradation trigger.\n"
                    f"Reverted *{r['new_weight']}* → *{r['prior_weight']}*. "
                    f"{r['note'] or ''}\n"
                    f"The axis is now barred from self-applying for "
                    f"{_cooldown_days()} days.")
        out.append({
            "key": f"rollback:{r['id']}",
            "title": "ROLLBACK FAILED" if failed else "Auto-applied change rolled back",
            "severity": "critical" if failed else "high",
            "body": body,
        })
    return out


def _degrade_pct() -> int:
    try:
        from kairos_autonomy import ROLLBACK_DEGRADE_FRAC
        return int(ROLLBACK_DEGRADE_FRAC * 100)
    except Exception:
        return 20


def _min_closes() -> int:
    try:
        from kairos_autonomy import ROLLBACK_MIN_CLOSES
        return ROLLBACK_MIN_CLOSES
    except Exception:
        return 10


def _cooldown_days() -> int:
    try:
        from kairos_autonomy import ROLLBACK_COOLDOWN_DAYS
        return ROLLBACK_COOLDOWN_DAYS
    except Exception:
        return 14


# ── Condition 2: a ratchet guard bound and escalated ─────────────────

def check_guard_escalated() -> list:
    """Standing proposals on an AUTONOMOUS axis held back by a ratchet guard.

    This is the ratchet signature. On an axis that would otherwise have
    applied itself, a bound cumulative_band / ordering / freshness guard means
    the loop tried to move further than the drift bound allows — the exact
    situation the July postmortem says a human must adjudicate. Materiality is
    excluded by construction (escalating_guards), because "too small to card"
    is not "too dangerous to apply".
    """
    from kairos_axis_weights import escalating_guards
    try:
        from kairos_autonomy import AUTO_APPLY_AXES
    except Exception:
        return []

    conn = _kairos_db()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM axis_weight_history WHERE status = 'proposed'")]
    finally:
        conn.close()

    out = []
    for r in rows:
        if r["axis"] not in AUTO_APPLY_AXES:
            continue
        if abs(r["proposed_delta"] or 0.0) < 1e-9:
            continue
        try:
            ev = json.loads(r["evidence"] or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            continue
        bound = escalating_guards(ev)
        if not bound:
            continue
        detail = "\n".join(
            f"    • `{name}` — {g.get('reason', 'bound')}"
            + (f" (would have written {g['would_have_been']})"
               if g.get("would_have_been") is not None else "")
            for name, g in sorted(bound.items()))
        out.append({
            "key": f"guard:{r['id']}:{','.join(sorted(bound))}",
            "title": "Ratchet guard bound on an autonomous axis",
            "severity": "high",
            "body": (f"`{r['axis']}` wanted {r['prior_weight']} → "
                     f"{r['new_weight']} (Δ {r['proposed_delta']:+g}) and was "
                     f"stopped:\n{detail}\n"
                     f"This axis normally applies itself, so it is waiting on "
                     f"you. Approve or reject id *{r['id']}*."),
        })
    return out


# ── Condition 3: nothing traded ──────────────────────────────────────

def check_no_trades(days: int = NO_TRADE_DAYS, now_et=None) -> list:
    """No fills across `days` consecutive weekdays.

    The 2026-08-10 silent-death mode. Between 08-11 and 08-17 the decisions
    table recorded NOTHING — not a fill, not a skip — while the scheduler log
    kept ticking and every health probe stayed green. So the message
    distinguishes the two shapes, because they have completely different
    causes and completely different fixes:
      * decisions present, no fills  -> the pipeline ran and chose not to act
        (or every order failed) — look at guardrails, cash, execution
      * no decisions at all          -> the pipeline is not running the trading
        path at all — look at the scheduler and the cycle
    """
    now_et = now_et or datetime.now(ET)
    days_checked = _weekdays_back(now_et.date(), days)
    lo = min(days_checked).isoformat()

    conn = _kairos_db()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT substr(timestamp,1,10) d, COUNT(*) total, "
            "       SUM(CASE WHEN execution_status = 'Filled' THEN 1 ELSE 0 END) filled "
            "FROM decisions WHERE substr(timestamp,1,10) >= ? GROUP BY d", (lo,))]
    finally:
        conn.close()
    by_day = {r["d"]: r for r in rows}

    fills = sum((by_day.get(d.isoformat()) or {}).get("filled") or 0
                for d in days_checked)
    if fills:
        return []

    decisions = sum((by_day.get(d.isoformat()) or {}).get("total") or 0
                    for d in days_checked)
    span = f"{min(days_checked)} .. {max(days_checked)}"
    if decisions == 0:
        shape = (f"*and no decisions were recorded at all* over that span — "
                 f"the trading path is not running, not merely declining to "
                 f"act. This is the 2026-08-10 shape (a full week, everything "
                 f"green, nothing traded). Check the scheduler and the cycle "
                 f"log first, not the guardrails.")
    else:
        shape = (f"The pipeline DID run — {decisions} decisions recorded, all "
                 f"non-fills. So this is a decision or execution problem, not "
                 f"a dead scheduler: check entry guardrails (cash reserve, "
                 f"sector concentration), the shortlist, and order rejections.")
    return [{
        "key": f"no_trades:{max(days_checked).isoformat()}",
        "title": f"No trades executed in {days} consecutive market days",
        "severity": "critical",
        "body": f"Zero fills across {span} ({days} weekdays).\n{shape}",
    }]


# ── Condition 4: NLV drawdown from peak ──────────────────────────────

def _nlv_series() -> list:
    """(timestamp, nlv) from both places NLV is recorded, newest last.

    nlv_snapshots is authoritative but sparse (13 rows over three months);
    decisions.net_liq_after is a dense by-product of every executed decision.
    Using both means the peak is not an artifact of which days happened to get
    a snapshot.
    """
    out = []
    conn = _kairos_db()
    try:
        for r in conn.execute(
                "SELECT snapshot_date d, nlv FROM nlv_snapshots WHERE nlv IS NOT NULL"):
            out.append((str(r["d"]), float(r["nlv"])))
        for r in conn.execute(
                "SELECT substr(timestamp,1,10) d, net_liq_after v FROM decisions "
                "WHERE net_liq_after IS NOT NULL AND net_liq_after > 0"):
            out.append((str(r["d"]), float(r["v"])))
    finally:
        conn.close()
    return sorted(out)


def check_drawdown(threshold_pct: float = NLV_DRAWDOWN_PCT) -> list:
    series = _nlv_series()
    if len(series) < 2:
        return []
    peak_date, peak = max(series, key=lambda t: t[1])
    cur_date, cur = series[-1]
    dd = (peak - cur) / peak * 100.0
    if dd < threshold_pct:
        return []
    return [{
        "key": f"drawdown:{cur_date}:{int(dd)}",
        "title": f"NLV drawdown {dd:.1f}% from peak",
        "severity": "critical",
        "body": (f"Peak *${peak:,.0f}* on {peak_date} → *${cur:,.0f}* on "
                 f"{cur_date} = *-{dd:.1f}%*, past the {threshold_pct:.0f}% "
                 f"alert threshold.\n"
                 f"A long-only book of this size tracks the index closely, so "
                 f"the first question is whether the index did this too. If it "
                 f"did not, look at position sizing, concentration, and "
                 f"whether the exit stack is firing."),
    }]


# ── Condition 5: an auto-applied change trending worse ───────────────

def check_early_warning() -> list:
    """Auto-applied changes whose error is rising but that are not yet judgeable.

    Explicitly an early warning and NOT an action: it fires below
    ROLLBACK_MIN_CLOSES, where the post-change sample is still too thin for a
    verdict, and says so. Its whole value is lead time — a rollback that
    arrives ten closes from now is worth knowing about today if it is the
    difference between watching and intervening.
    """
    try:
        from kairos_autonomy import (ROLLBACK_MIN_CLOSES, _baseline_metric,
                                     _closed_count)
    except Exception:
        return []

    now_closes = _closed_count()
    conn = _kairos_db()
    try:
        pending = [dict(r) for r in conn.execute(
            "SELECT * FROM autonomy_log WHERE verdict = 'pending'")]
    finally:
        conn.close()

    out = []
    for r in pending:
        elapsed = now_closes - (r["closes_at_apply"] or 0)
        if elapsed >= ROLLBACK_MIN_CLOSES:
            continue                      # judgeable — check_rollbacks owns it
        base = r["baseline_error"]
        post = _baseline_metric(r["axis"])
        if base is None or post is None or base <= 1e-9:
            continue
        worse = (post - base) / base
        if worse < EARLY_WARNING_FRAC:
            continue
        out.append({
            "key": f"early:{r['id']}:{int(worse * 100)}",
            "title": "Auto-applied change trending worse (early warning)",
            "severity": "warning",
            "body": (f"`{r['axis']}` {r['prior_weight']} → {r['new_weight']}, "
                     f"applied {r['applied_at']}.\n"
                     f"Error {base:.4f} → {post:.4f}pp = *{worse:+.0%}* — past "
                     f"the {EARLY_WARNING_FRAC:.0%} watch line but not the "
                     f"{_degrade_pct()}% rollback trigger.\n"
                     f"*No action taken and none needed yet*: only {elapsed} of "
                     f"{ROLLBACK_MIN_CLOSES} closes have accumulated, so the "
                     f"sample is still too thin to judge. If it holds, this "
                     f"reverts on its own."),
        })
    return out


# ── Condition 6: the ATR shadow period has produced decidable pools ──
#
# ONE-TIME. This is not a fault, it is a deadline arriving: atr_enabled has
# been false since the ATR-scaled trail shipped, and the three parameters that
# govern it cannot be evaluated until closes have actually been routed to
# them. The moment they can, someone has to decide — and with J at low touch
# for 3-6 months, nothing else in the system would ever raise it. The work
# would simply sit inert past the point where it became answerable, which is
# the quiet failure this whole layer exists to catch.

ATR_PARAMS = ("exits.trailing_stop.target_armed.atr_mult",
              "exits.trailing_stop.target_armed.trail_lo_pct",
              "exits.trailing_stop.target_armed.trail_hi_pct")

# Effective sample each ATR pool must reach before the question is decidable.
# 10 is PARAM_MIN_SAMPLE, which is not an arbitrary echo: it is the effective_n
# at which _confidence returns exactly 0.5, i.e. the point the learning loop
# itself treats as "enough evidence to take half a step". Below it the loop
# would be correcting atr_enabled's parameters on evidence it does not trust,
# so a decision made below it is a decision made without the correction
# mechanism that justifies making it.
ATR_POOL_READY_N = 10

# The bind split the backtest assumed, from the held book on 2026-09-09.
# Carried as data so the alert can report the DIFFERENCE rather than making J
# hold two sets of numbers in his head three months from now.
ATR_EXPECTED_BOOK_SPLIT = {"free": 19, "floor": 23, "ceiling": 7}


def atr_pool_state() -> dict:
    """effective_n and routed-close counts for each ATR parameter."""
    from kairos_axis_weights import compute_param
    out = {}
    for path in ATR_PARAMS:
        try:
            ev = (compute_param(path) or {}).get("evidence") or {}
            out[path] = {
                "effective_n": float(ev.get("effective_n") or 0.0),
                "n_contributing": int(ev.get("n_contributing") or 0),
                "by_bind_state": (ev.get("routing") or {}).get("by_bind_state") or {},
            }
        except Exception as exc:
            out[path] = {"effective_n": 0.0, "n_contributing": 0,
                         "by_bind_state": {}, "error": str(exc)}
    return out


def _live_book_bind_split() -> dict:
    """Arm-time bind states for the positions the shadow period has armed.

    Read from armed_trail_context rather than from closed trades, because that
    is the same population the 2026-09-09 book figure counted — comparing a
    closed-trade split against a held-book baseline would be comparing two
    different things and calling the difference a finding.
    """
    conn = _kairos_db()
    try:
        return {r["bind_state"]: r["n"] for r in conn.execute(
            "SELECT bind_state, COUNT(*) n FROM armed_trail_context "
            "WHERE bind_state IS NOT NULL GROUP BY bind_state")}
    except Exception:
        return {}
    finally:
        conn.close()


def _split_line(split: dict, expected: dict | None = None) -> str:
    total = sum(v for k, v in split.items() if k in ("free", "floor", "ceiling"))
    bits = []
    for k in ("free", "floor", "ceiling"):
        n = split.get(k, 0)
        pct = (n / total * 100) if total else 0.0
        if expected:
            e_tot = sum(expected.values())
            e_pct = (expected.get(k, 0) / e_tot * 100) if e_tot else 0.0
            bits.append(f"{k} {n} ({pct:.0f}% vs {e_pct:.0f}% expected)")
        else:
            bits.append(f"{k} {n} ({pct:.0f}%)")
    extra = {k: v for k, v in split.items() if k not in ("free", "floor", "ceiling")}
    tail = ("  |  " + ", ".join(f"{k} {v}" for k, v in sorted(extra.items()))
            if extra else "")
    return ", ".join(bits) + tail


def check_atr_pools_ready(min_n: float = ATR_POOL_READY_N) -> list:
    """Fire once when all three ATR pools are decidable. Never auto-flips."""
    pools = atr_pool_state()
    if not all(p["effective_n"] >= min_n for p in pools.values()):
        return []

    book = _live_book_bind_split()
    evidence_split = {}
    for p in pools.values():
        for k, v in (p["by_bind_state"] or {}).items():
            evidence_split[k] = max(evidence_split.get(k, 0), v)

    pool_lines = "\n".join(
        f"    • `{path.rsplit('.', 1)[-1]}` effective_n "
        f"*{pools[path]['effective_n']:.1f}*  "
        f"({pools[path]['n_contributing']} contributing closes)"
        for path in ATR_PARAMS)

    return [{
        "key": "atr_pools_ready",
        "once": True,
        "title": "ATR shadow period complete — atr_enabled is ready for a decision",
        "severity": "high",
        "body": (
            f"All three ATR parameters have reached effective_n ≥ {min_n:g}, "
            f"so the ATR-scaled armed trail is now measurable rather than "
            f"theoretical:\n{pool_lines}\n\n"
            f"*Bind split, held book* (armed_trail_context, comparable to the "
            f"2026-09-09 figure the backtest assumed):\n"
            f"    {_split_line(book, ATR_EXPECTED_BOOK_SPLIT)}\n"
            f"*Bind split, closed evidence* (what the pools above are made of):\n"
            f"    {_split_line(evidence_split)}\n"
            f"Sanity-check the first line against 19 free / 23 floor / "
            f"7 ceiling. A heavier `ceiling` share than expected means "
            f"`trail_hi_pct` (4.0) is truncating more positions than the "
            f"backtest modelled, so worst-case give-back is capped at 6.0pp "
            f"for more of the book — tighter than assumed, not looser.\n\n"
            f"*What flipping `atr_enabled` to true does:* it tightens the "
            f"trail on *every* armed position at once, because the clamp "
            f"[2.0, 4.0] sits below the current flat trail. That was the "
            f"deliberate choice — start tight and let the learning loop loosen "
            f"it if exits prove premature — and the loop can now actually do "
            f"that, which is what these three pools filling in means.\n\n"
            f":no_entry: *Nothing has been changed.* `atr_enabled` is still "
            f"false and no automatic path will flip it: enabling exit "
            f"behaviour that has never run live is not something to do "
            f"unsupervised. This alert fires once and will not repeat."),
    }]


# ── Dedupe + post ────────────────────────────────────────────────────

def load_state(path: str | None = None) -> dict:
    """Read the dedupe ledger.

    `path` resolves to the module-level STATE_FILE at CALL time, not as a
    default argument. A default argument would bind the value at import, which
    makes STATE_FILE unpatchable — and the first version of
    kairos_selftest_watch.py duly wrote its sandbox state straight into the
    live .kairos_watch_state.json, silencing two real conditions on the
    production install. A test that can mute production monitoring is a worse
    bug than anything it is testing for.
    """
    path = path or STATE_FILE
    try:
        with open(path) as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def save_state(state: dict, path: str | None = None) -> None:
    path = path or STATE_FILE
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, path)
    except Exception as exc:               # pragma: no cover
        print(f"  WARNING: could not persist watch state: {exc}")


_SEVERITY_ICON = {"critical": ":rotating_light:", "high": ":warning:",
                  "warning": ":eyes:"}


def format_alert(conditions: list, now_et=None) -> str:
    """One @here post covering everything that fired. One post, not five —
    five separate pages at 8pm is how a rare channel becomes a muted one."""
    now_et = now_et or datetime.now(ET)
    order = {"critical": 0, "high": 1, "warning": 2}
    conditions = sorted(conditions, key=lambda c: order.get(c["severity"], 9))
    head = (f"<!here> :rotating_light: *KAIROS NEEDS YOU* — "
            f"{len(conditions)} condition(s)  |  "
            f"{now_et.strftime('%Y-%m-%d %H:%M ET')}")
    parts = [head, "─" * 40]
    for c in conditions:
        parts.append(f"{_SEVERITY_ICON.get(c['severity'], ':grey_question:')} "
                     f"*{c['title']}*\n{c['body']}")
    parts.append("_This channel only posts when something needs a decision. "
                 "Routine activity is in the daily digest and the Sunday "
                 "summary._")
    return "\n\n".join(parts)


def collect(now_et=None) -> list:
    """Every condition currently true. Each check is isolated: a broken check
    must not be able to suppress the others."""
    checks = (
        ("rollback_fired", check_rollback_fired),
        ("guard_escalated", check_guard_escalated),
        ("no_trades", lambda: check_no_trades(now_et=now_et)),
        ("drawdown", check_drawdown),
        ("early_warning", check_early_warning),
        ("atr_pools_ready", check_atr_pools_ready),
    )
    found = []
    for name, fn in checks:
        try:
            found += fn()
        except Exception as exc:
            print(f"  WARNING: watch check {name} failed: {exc}")
    return found


def run(dry_run: bool = False, force: bool = False, now_et=None) -> dict:
    now_et = now_et or datetime.now(ET)
    found = collect(now_et=now_et)
    state = load_state()
    seen = state.get("fired", {})
    # Separate ledger for conditions that fire exactly once in the system's
    # lifetime. It must NOT live in `fired`: that dict is pruned to 7 days so
    # it cannot grow without bound, and a pruned one-time flag is a one-time
    # alert that fires again next week. Two ledgers with two retention rules
    # is simpler to reason about than one with an exception in it.
    once_fired = state.get("once", {})
    cutoff = _now_utc() - timedelta(hours=REALERT_AFTER_HOURS)

    fresh = []
    for c in found:
        if c.get("once"):
            # `force` deliberately does NOT override a one-time flag. --force
            # exists to re-send something still true today; re-sending a
            # once-in-a-lifetime notice is never what it means.
            if c["key"] not in once_fired:
                fresh.append(c)
            continue
        last = _parse_ts(seen.get(c["key"]))
        if force or last is None or last < cutoff:
            fresh.append(c)

    result = {"found": found, "fresh": fresh, "posted": False, "text": None}
    if not fresh:
        return result

    text = format_alert(fresh, now_et=now_et)
    result["text"] = text
    if dry_run:
        return result

    from kairos_alerts import post_message
    result["posted"] = post_message(ALERTS_CHANNEL, text)
    stamp = _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    # Stamp one-time flags even if the Slack post failed. The alternative —
    # only stamping on success — retries forever against a channel that may be
    # misconfigured, and the ATR notice is also printed to the log and visible
    # in `--dry-run`, so a lost post is recoverable while a loop is not.
    for c in fresh:
        if c.get("once"):
            once_fired[c["key"]] = stamp
        else:
            seen[c["key"]] = stamp
    # Forget non-once keys that have not fired for a week so the file cannot
    # grow without bound; a condition quiet that long is new information anyway.
    keep_after = _now_utc() - timedelta(days=7)
    seen = {k: v for k, v in seen.items()
            if (_parse_ts(v) or keep_after) >= keep_after}
    state["fired"] = seen
    state["once"] = once_fired          # never pruned
    save_state(state)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kairos Watch — alert only when something needs a human")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be posted; post nothing, write no state")
    ap.add_argument("--force", action="store_true",
                    help="Ignore the re-alert cooldown")
    args = ap.parse_args()

    # launchd hands agents a minimal environment; the token lives in ~/.zshrc,
    # never in a plist. Same parser kairos_healthcheck uses.
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

    res = run(dry_run=args.dry_run, force=args.force)
    if not res["found"]:
        print("  Nothing needs attention.")
        return 0
    print(f"  {len(res['found'])} condition(s) true, "
          f"{len(res['fresh'])} past the re-alert cooldown.")
    if res["text"]:
        print("\n" + res["text"] + "\n")
    if not args.dry_run:
        print(f"  Posted: {res['posted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
