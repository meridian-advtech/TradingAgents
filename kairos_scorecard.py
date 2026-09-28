#!/usr/bin/env python3
"""Mechanism scorecard — does each exit mechanism and entry gate actually work?

Why this exists (2026-09-28): the Arbiter learns the PARAMETERS on its
whitelist (trail widths, profit floor, exit timing) but had no view of whether
whole MECHANISMS earn their keep. Every mechanism verdict to date came from
ad-hoc queries in a working session:
  - PRICE-CONTRADICTION: 1 win in 27, disabled 2026-08-27 — found by hand.
  - Armed stop-loss preemption: 15 armed stop-outs — found by hand.
  - HOT-INSIDER: lifetime +$19K hid -$17.6K over the last 90 days — by hand.
Nothing re-checked any of them afterward. This does, weekly, unasked.

Read-only. Never raises into its caller.

  1. Exits by type — last 7d vs prior 28d: fires, win rate, median realized,
     and median 14-day forgone gain ADJUSTED for SPY over the same 14 days.
     Raw forgone gain overstates mistimed exits in a rising market; an exit
     type only 'sells too early' if the stock beat the index afterward.
  2. Entry gates — conviction floor blocks, re-entry guard (shadow)
     observations, paused signals in force.
  3. Flags — plain-language, only where the evidence clears a minimum n.

    python3 kairos_scorecard.py
    from kairos_scorecard import build_scorecard_text
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics as st
from collections import defaultdict
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DB = os.path.join(HERE, "kairos.db")
K_DB = os.path.join(HERE, "kairos.db")
CFG = os.path.join(HERE, "kairos_config.json")

RECENT_DAYS = 7
BASELINE_DAYS = 28
FORGONE_DAYS = 14          # matches the Arbiter's forgone horizon
MIN_N_FOR_VERDICT = 5      # below this, counts only — no verdict

# Exits whose JOB is to cut losers. A 0% win rate is them working as designed,
# so the win-rate flag is noise for these; only 'did the stock beat SPY after we
# sold' can show they are cutting the wrong things.
LOSS_CUTTING_EXITS = {"STOP-LOSS", "PRICE-INVALIDATION", "PRICE-CONTRADICTION",
                      "TAX-HARVEST"}


def _ro(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout = 2000")
    return c


def _d(ts):
    try:
        return date.fromisoformat(str(ts)[:10])
    except Exception:
        return None


def _exit_type(reason):
    return (reason or "none").split(":")[0].strip() or "none"


def _spy_closes(start):
    """{date: close}. Empty on failure — scorecard then says so."""
    try:
        import yfinance as yf
        s = yf.download("SPY", start=start.isoformat(), progress=False,
                        auto_adjust=True)["Close"]
        if hasattr(s, "columns"):
            s = s.iloc[:, 0]
        return {i.date(): float(v) for i, v in s.dropna().items()}
    except Exception:
        return {}


def _spy_return(spy, d0, days):
    if not spy:
        return None
    ds = sorted(spy)
    a = next((x for x in ds if x >= d0), None)
    b = next((x for x in ds if x >= d0 + timedelta(days=days)), None)
    if a is None or b is None or b <= a:
        return None
    return (spy[b] / spy[a] - 1.0) * 100.0


def _fmt(x):
    return "—" if x is None else f"{x:+.2f}%"


def exits_section(today=None):
    today = today or date.today()
    recent_from = today - timedelta(days=RECENT_DAYS)
    base_from = recent_from - timedelta(days=BASELINE_DAYS)
    try:
        c = _ro(ML_DB)
        rows = c.execute(
            "SELECT exit_reason, pnl_pct, timestamp_exit, forgone_gain_14d_pct AS f14 "
            "FROM trade_outcomes WHERE timestamp_exit IS NOT NULL AND pnl_pct IS NOT NULL"
        ).fetchall()
        c.close()
    except Exception as exc:
        return [f"  (exit data unavailable: {exc})"], {}

    spy = _spy_closes(base_from - timedelta(days=3))
    b = {"recent": defaultdict(list), "base": defaultdict(list)}
    for r in rows:
        d = _d(r["timestamp_exit"])
        if d is None or d < base_from:
            continue
        excess = None
        if r["f14"] is not None:
            sr = _spy_return(spy, d, FORGONE_DAYS)
            if sr is not None:
                excess = r["f14"] - sr
        b["recent" if d >= recent_from else "base"][_exit_type(r["exit_reason"])].append(
            (r["pnl_pct"], excess))

    lines = [
        f"1. EXITS BY TYPE — last {RECENT_DAYS}d vs prior {BASELINE_DAYS}d",
        f"   'fwd{FORGONE_DAYS} vs SPY' = how much the stock beat the index in the {FORGONE_DAYS}",
        "   days after we sold. Consistently positive = exiting too early.",
        f"   Blank until {FORGONE_DAYS} days have passed since the exit.",
        "",
        f"   {'exit type':22}{'7d n':>5}{'win':>6}{'median':>9}   "
        f"{'28d n':>6}{'win':>6}{'median':>9}{'fwd14 vs SPY':>16}",
    ]

    def stats(v):
        if not v:
            return 0, None, None
        p = [x[0] for x in v]
        return len(v), sum(1 for x in p if x > 0) / len(v) * 100, st.median(p)

    summary = {}
    for t in sorted(set(b["recent"]) | set(b["base"]),
                    key=lambda t: -(len(b["recent"][t]) + len(b["base"][t]))):
        rn, rw, rm = stats(b["recent"][t])
        bn, bw, bm = stats(b["base"][t])
        exs = [x[1] for x in b["recent"][t] + b["base"][t] if x[1] is not None]
        ex = st.median(exs) if exs else None
        summary[t] = {"recent_n": rn, "base_n": bn, "base_win": bw,
                      "excess14": ex, "n_excess": len(exs)}
        exc_cell = f"{_fmt(ex)} ({len(exs)})" if exs else "—"
        lines.append(
            f"   {t[:22]:22}{rn:>5}{(f'{rw:.0f}%' if rn else '—'):>6}{_fmt(rm):>9}   "
            f"{bn:>6}{(f'{bw:.0f}%' if bn else '—'):>6}{_fmt(bm):>9}{exc_cell:>16}")
    if not spy:
        lines.append("   (SPY data unavailable this run — index adjustment skipped)")
    return lines, summary


def gates_section(today=None):
    today = today or date.today()
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    lines = ["", f"2. ENTRY GATES — last {RECENT_DAYS}d"]
    try:
        cfg = json.load(open(CFG))
        floor = cfg.get("execution", {}).get("min_entry_conviction")
        paused = cfg.get("paused_signals")
    except Exception:
        floor, paused = None, None
    try:
        k = _ro(K_DB)
        rows = k.execute(
            "SELECT ticker FROM decisions WHERE substr(timestamp,1,10) >= ? AND "
            "(rationale LIKE '%CONVICTION FLOOR%' OR data_inputs LIKE '%CONVICTION FLOOR%')",
            (since,)).fetchall()
        k.close()
        tk = sorted({r["ticker"] for r in rows})
        lines.append(f"   Conviction floor ({floor}): blocked {len(rows)} buy(s)"
                     + (f" — {', '.join(tk[:12])}" if tk else ""))
    except Exception as exc:
        lines.append(f"   Conviction floor: unavailable ({exc})")
    try:
        k = _ro(K_DB)
        obs = k.execute("SELECT side, would_block FROM reentry_observations "
                        "WHERE substr(observed_at,1,10) >= ?", (since,)).fetchall()
        k.close()
        by = defaultdict(int)
        for o in obs:
            by[o["side"]] += 1
        wb = sum(int(o["would_block"] or 0) for o in obs)
        lines.append(f"   Re-entry guard (shadow): {len(obs)} attempt(s) "
                     f"[above {by['above']} / below {by['below']} / at {by['at']}], "
                     f"{wb} would have been blocked")
    except sqlite3.OperationalError:
        lines.append("   Re-entry guard (shadow): no re-entry attempts observed yet")
    except Exception as exc:
        lines.append(f"   Re-entry guard: unavailable ({exc})")
    lines.append(f"   Paused signals in force: {json.dumps(paused) if paused else 'none'}")
    return lines


def verdicts(summary):
    out = []
    for t, s in summary.items():
        if (t not in LOSS_CUTTING_EXITS and s["base_n"] >= MIN_N_FOR_VERDICT
                and s["base_win"] is not None and s["base_win"] < 40):
            out.append(f"   • {t}: {s['base_win']:.0f}% win over {s['base_n']} exits in the "
                       f"prior {BASELINE_DAYS}d — this exit is not meant to be a loss-cutter.")
        if s["n_excess"] >= MIN_N_FOR_VERDICT and s["excess14"] is not None and s["excess14"] > 5:
            out.append(f"   • {t}: stocks beat SPY by a median {s['excess14']:+.1f}% in the "
                       f"{FORGONE_DAYS}d after this exit (n={s['n_excess']}) — likely exiting too early.")
    return ["", "3. FLAGS"] + (out or ["   none this week"])


def build_scorecard_text(today=None):
    try:
        ex, summary = exits_section(today)
        return "\n".join(["MECHANISM SCORECARD"] + ex + gates_section(today) + verdicts(summary))
    except Exception as exc:
        return f"MECHANISM SCORECARD\n  (failed to build: {exc})"


if __name__ == "__main__":
    print(build_scorecard_text())
