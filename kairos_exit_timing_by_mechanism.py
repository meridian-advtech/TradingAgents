"""Is any exit mechanism systematically firing too early? Market-controlled.

Read-only diagnostic. Answers a question raised on 2026-08-03: forgone gain
exceeded give-back at 30d/60d on NON-trailing-stop exits, hinting that
reallocation and thesis exits fire too early. That hint is NOT actionable as
it stands, for one reason:

    OVER 30-60 DAYS A RISING MARKET MAKES EVERY EXIT LOOK PREMATURE.

Raw forgone gain measures the best price available after an exit. In an
advancing tape that number is positive for nearly any exit, however correct.
Give-back, by contrast, is measured INSIDE the hold, over a typically shorter
span. Comparing the two raw is therefore biased toward "you exited too early"
by exactly the market's drift over the horizon.

This tool removes that bias by subtracting SPY's return over the SAME window:

    excess_forgone = forgone_gain_Nd - spy_return_over_same_N_days
    net_error      = give_back - excess_forgone

net_error > 0 means the mechanism gives back more than it forgoes -> exits
LATE, tighten. net_error < 0 means it leaves more on the table than it
protects -> exits EARLY, loosen. Both are reported per mechanism, because
"the exit engine" is really six different mechanisms with different jobs.

Usage:  python kairos_exit_timing_by_mechanism.py [--horizon 30]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DB = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")

HORIZONS = (5, 14, 30, 60)

# exit_reason prefix -> mechanism label. Anything unmatched is grouped as
# 'other' rather than silently folded into a named bucket.
MECHANISMS = {
    "TRAILING-STOP": "TRAILING-STOP",
    "REVERSION-COMPLETE": "REVERSION-COMPLETE",
    "PRICE-CONTRADICTION": "PRICE-CONTRADICTION",
    "REALLOCATION": "REALLOCATION",
    "STOP-LOSS": "STOP-LOSS",
    "THESIS-INVALID": "THESIS-INVALID",
    "PRICE-INVALIDATION": "PRICE-INVALIDATION",
}


def _mechanism(reason: str | None) -> str:
    if not reason:
        return "unlabelled"
    head = reason.split(":")[0].strip().upper()
    for k, v in MECHANISMS.items():
        if head.startswith(k):
            return v
    return "other"


def _spy_series():
    """{date -> close} for SPY across the study window."""
    import yfinance as yf
    hist = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
    out = {}
    for idx, val in hist["Close"].items():
        if val == val and val > 0:
            out[idx.date()] = float(val)
    return out


def _spy_return(spy: dict, exit_date, days: int):
    """SPY % return from the first close on/after exit to the close ~N days later."""
    if not spy:
        return None
    keys = sorted(spy)
    start = next((k for k in keys if k >= exit_date), None)
    if start is None:
        return None
    target = exit_date + timedelta(days=days)
    ends = [k for k in keys if k <= target]
    if not ends:
        return None
    end = ends[-1]
    if end <= start:
        return None
    return (spy[end] - spy[start]) / spy[start] * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    args = ap.parse_args()
    H = args.horizon
    col = f"forgone_gain_{H}d_pct"

    m = sqlite3.connect(f"file:{ML_DB}?mode=ro", uri=True)
    m.row_factory = sqlite3.Row
    rows = [dict(r) for r in m.execute(
        f"SELECT ticker, timestamp_exit, exit_reason, give_back_pct, "
        f"       {col} AS forgone, pnl_pct, mfe_pct "
        f"FROM trade_outcomes "
        f"WHERE timestamp_exit IS NOT NULL AND give_back_pct IS NOT NULL "
        f"  AND {col} IS NOT NULL")]
    m.close()

    print(f"horizon: {H} calendar days")
    print(f"closed trades with give-back AND {H}d forgone: {len(rows)}")
    if len(rows) < 10:
        print("\nToo few matured trades at this horizon to say anything. "
              "Try --horizon 14.")
        return 0

    print("fetching SPY for the market control…")
    spy = _spy_series()
    if not spy:
        print("  SPY unavailable — cannot control for market drift. Aborting: "
              "raw numbers here would be misleading.")
        return 1

    groups = defaultdict(list)
    skipped = 0
    for r in rows:
        try:
            ed = datetime.strptime(str(r["timestamp_exit"])[:10], "%Y-%m-%d").date()
        except Exception:
            skipped += 1
            continue
        mkt = _spy_return(spy, ed, H)
        if mkt is None:
            skipped += 1
            continue
        excess = r["forgone"] - mkt
        groups[_mechanism(r["exit_reason"])].append({
            **r, "spy": mkt, "excess_forgone": excess,
            "net_error": r["give_back_pct"] - excess})
    if skipped:
        print(f"  skipped {skipped} trade(s) with no usable SPY window")

    print()
    print("=" * 84)
    print(f"PER MECHANISM — market-controlled ({H}d)")
    print("=" * 84)
    print(f"  {'mechanism':<22}{'n':>4}{'giveback':>10}{'forgone':>9}"
          f"{'SPY':>8}{'excess':>9}{'net err':>9}  verdict")
    print("  " + "-" * 80)
    order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for name, g in order:
        n = len(g)
        gb = sum(x["give_back_pct"] for x in g) / n
        fg = sum(x["forgone"] for x in g) / n
        sp = sum(x["spy"] for x in g) / n
        ex = sum(x["excess_forgone"] for x in g) / n
        ne = sum(x["net_error"] for x in g) / n
        if n < 5:
            verdict = "n too small"
        elif ne > 2:
            verdict = "exits LATE — tighten"
        elif ne < -2:
            verdict = "exits EARLY — loosen"
        else:
            verdict = "balanced"
        print(f"  {name:<22}{n:>4}{gb:>10.2f}{fg:>9.2f}{sp:>8.2f}"
              f"{ex:>9.2f}{ne:>9.2f}  {verdict}")

    allt = [x for g in groups.values() for x in g]
    n = len(allt)
    print("  " + "-" * 80)
    print(f"  {'ALL':<22}{n:>4}"
          f"{sum(x['give_back_pct'] for x in allt)/n:>10.2f}"
          f"{sum(x['forgone'] for x in allt)/n:>9.2f}"
          f"{sum(x['spy'] for x in allt)/n:>8.2f}"
          f"{sum(x['excess_forgone'] for x in allt)/n:>9.2f}"
          f"{sum(x['net_error'] for x in allt)/n:>9.2f}")

    print(f"\n  medians (outlier-resistant; means above are pulled by tails):")
    print(f"  {'mechanism':<22}{'n':>4}{'med net err':>13}")
    for name, g in order:
        if len(g) < 5:
            continue
        print(f"  {name:<22}{len(g):>4}"
              f"{st.median([x['net_error'] for x in g]):>13.2f}")

    print("\n" + "=" * 84)
    print("HOW TO READ THIS")
    print("=" * 84)
    print(f"""  net error = give_back - (forgone - SPY)
    > 0  the mechanism surrenders more than it protects -> exits LATE
    < 0  it protects less than it leaves behind        -> exits EARLY

  The SPY column is the size of the bias this control removes. Where SPY is
  large and positive, the RAW forgone figure was mostly market drift, and any
  conclusion drawn from raw forgone at this horizon was measuring the tape.

  Sample sizes per mechanism are small and all trades sit in one regime
  (2026-05 to 2026-07). Treat a single mechanism's verdict as a hypothesis to
  re-test as data accumulates, not as a mandate to retune it.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
