"""
ATR-scaled armed trail — BACKTEST ONLY. Changes no behaviour, writes no config.

QUESTION
--------
The target-armed trail is a flat percentage (exits.trailing_stop.target_armed.
trail_pct) with a 1.5x intraday backstop. At the live 6.2255 that backstop is
9.34 points; at the 6.9469 in force before the last approval it was 10.42. Either
way, for any position that peaks above roughly that number, give-back is pinned
at the backstop BY CONSTRUCTION — and the same number is applied to a stable
industrial and to RGTI. Does scaling the trail to each name's own volatility
(ATR14) cut median give-back without buying whipsaw?

    ATR-scaled trail:  trail_pct = clamp(ATR_MULT x ATR14/price x 100, LO, HI)

Everything else is held identical to the live engine: the arming condition
(running peak >= the position's OWN logged predicted_return_pct), the profit
floor, the 1.5x intraday multiplier, and the close-eval / intraday split.

METHOD
------
Counterfactual replay over closed LONG trades, on real daily OHLC paths. Price
paths come from the SAME helpers the outcome-feature enrichment and the
target-armed / price-invalidation backtests used — kairos_arbiter._download_daily
and _parse_utc, imported, not re-implemented — and MFE is measured over
[entry, exit] inclusive, exactly as kairos_outcome_features.compute_features_
for_trade defines it, so give-back here is the same quantity the corpus carries.

A variant only changes a trade when its trail fires BEFORE the trade's actual
close. Otherwise the trade keeps its real exit, because in reality some other
mechanism (reallocation, thesis review, stop-loss) closed it, and this study is
not entitled to claim those away. Total return is therefore a genuine
counterfactual on one rule, not a whole-strategy re-run.

Assumptions, stated because they are load-bearing:
  * A stop fills at its TRIGGER level, not at the day's low. The engine sells
    market on the cycle that observes the retreat, so the fill is near the
    trigger; a daily bar cannot resolve intra-day path. This is optimistic by
    roughly one cycle of slippage and is applied IDENTICALLY to every variant,
    including the fixed baseline, so it cannot favour one.
  * ATR14 is the mean True Range over the 14 trading days STRICTLY BEFORE the
    arming day, divided by the prior close. No lookahead: it uses only what the
    engine would have known when the position armed.
  * Give-back is reported two ways. vs_peak_at_exit is the corpus definition
    (mfe over the realised hold). vs_full_hold_peak re-measures every variant
    against the peak of the FULL actual hold, which is the same denominator for
    all variants — because an earlier exit truncates the window in which a peak
    can form, and comparing the first number alone across variants that exit on
    different dates would flatter whichever exits earliest.

CLI
---
    python3 kairos_backtest_atr_trail.py                  # full report
    python3 kairos_backtest_atr_trail.py --no-cache       # refetch prices
    python3 kairos_backtest_atr_trail.py --bounds-sweep   # LO/HI sensitivity only
"""

from __future__ import annotations

import argparse
import os
import pickle
import sqlite3
import statistics as st
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

ML_DB = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
CACHE = os.path.join(SCRIPT_DIR, ".atr_trail_backtest_prices.pkl")

# Reused verbatim from the enrichment / prior-backtest harness.
from kairos_arbiter import _download_daily, _parse_utc  # noqa: E402

# ── Live engine constants, read from config so the baseline is the REAL one ──
INTRADAY_MULT = 1.5
ATR_WINDOW = 14
PRE_ENTRY_DAYS = 45      # calendar days of lead-in so ATR14 exists at arming
POST_EXIT_DAYS = 35      # calendar days after the last exit, for forgone gain
FORGONE_HORIZONS = (5, 14, 30)

# Data-integrity gate. A DB entry price this far outside the entry day's own
# [Low, High] band means the DB price and the price series are not describing
# the same shares — the CRWD pre-split/post-split pattern. Such a trade cannot
# be replayed against the series at all, in either direction, so it is dropped
# rather than silently producing a fabricated 300% "peak".
INTEGRITY_TOL = 0.10


def _cfg() -> dict:
    import json
    with open(os.path.join(SCRIPT_DIR, "kairos_config.json")) as f:
        return (json.load(f) or {}).get("exits", {})


def _live() -> tuple:
    """(live trail_pct, live profit_floor_pp) from kairos_config.json."""
    ts = _cfg().get("trailing_stop", {})
    return (float(ts.get("target_armed", {}).get("trail_pct", 8.0)),
            float(ts.get("profit_floor_pp", 1.0)))


# ── Candidates ───────────────────────────────────────────────────────

def load_candidates() -> list:
    """Closed LONG trades that carry their OWN logged thesis target.

    The target is joined on thesis_predictions.decision_id = trade_id, NOT via
    kairos_ml_outcomes.get_thesis_target — that helper returns the ticker's
    LATEST thesis, which for a re-traded ticker is a target the closed trade
    never had. Using it here would be lookahead.
    """
    conn = sqlite3.connect(f"file:{ML_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("""
            SELECT t.trade_id, t.ticker, t.timestamp_entry, t.timestamp_exit,
                   t.price_entry, t.price_exit, t.pnl_pct, t.mfe_pct,
                   t.give_back_pct, t.exit_reason,
                   t.forgone_gain_5d_pct, t.forgone_gain_14d_pct,
                   t.forgone_gain_30d_pct,
                   (SELECT p.predicted_return_pct FROM thesis_predictions p
                     WHERE p.decision_id = t.trade_id
                       AND p.predicted_return_pct IS NOT NULL
                     ORDER BY p.timestamp_entry DESC LIMIT 1) AS target
            FROM trade_outcomes t
            WHERE t.timestamp_exit IS NOT NULL
              AND UPPER(COALESCE(t.action, 'BUY')) = 'BUY'
              AND t.price_entry > 0 AND t.price_exit > 0
        """).fetchall()]
    finally:
        conn.close()

    out = []
    for r in rows:
        r["entry_dt"] = _parse_utc(r["timestamp_entry"])
        r["exit_dt"] = _parse_utc(r["timestamp_exit"])
        if r["entry_dt"] is None or r["exit_dt"] is None:
            continue
        if r["target"] is None:
            continue
        out.append(r)
    return out


def fetch_paths(trades: list, use_cache: bool = True) -> dict:
    """{ticker: OHLC DataFrame} covering every trade's hold + lead-in + tail."""
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    tickers = sorted({t["ticker"] for t in trades})
    start = (min(t["entry_dt"] for t in trades)
             - timedelta(days=PRE_ENTRY_DAYS)).strftime("%Y-%m-%d")
    end = (max(t["exit_dt"] for t in trades)
           + timedelta(days=POST_EXIT_DAYS)).strftime("%Y-%m-%d")
    print(f"  fetching {len(tickers)} tickers, {start} -> {end} ...")
    hist = _download_daily(tickers, start, end)
    with open(CACHE, "wb") as f:
        pickle.dump(hist, f)
    return hist


# ── Data integrity ───────────────────────────────────────────────────

def integrity_verdict(trade: dict, df) -> tuple:
    """(ok, reason, implied_ratio) — is the DB entry price the same shares?

    Compares price_entry to the entry day's own [Low, High]. A DB price outside
    that band by more than INTEGRITY_TOL means one side has been split-adjusted
    and the other has not; implied_ratio makes the split visible (a ~10.0 or
    ~0.1 is a clean 10:1).
    """
    if df is None or len(df) == 0:
        return False, "no price history", None
    day = _bar_on_or_after(df, trade["entry_dt"])
    if day is None:
        return False, "no bar on/after entry", None
    lo, hi = float(day["Low"]), float(day["High"])
    pe = float(trade["price_entry"])
    if lo <= 0 or hi <= 0:
        return False, "non-positive bar", None
    if lo * (1 - INTEGRITY_TOL) <= pe <= hi * (1 + INTEGRITY_TOL):
        return True, None, None
    mid = (lo + hi) / 2.0
    return False, "entry price inconsistent with series", round(pe / mid, 4)


def _bar_on_or_after(df, dt):
    import pandas as pd
    key = pd.Timestamp(dt.date())
    sub = df.loc[df.index >= key]
    return sub.iloc[0] if len(sub) else None


# ── ATR ──────────────────────────────────────────────────────────────

def atr_pct_series(df):
    """{Timestamp -> ATR14/prior_close x 100} using only STRICTLY PRIOR bars.

    True Range = max(H-L, |H-prev_close|, |L-prev_close|). The value stamped on
    day d is the mean TR over the 14 trading days ending at d-1, over the close
    at d-1 — what the engine could have known at the open of d.
    """
    highs, lows, closes = df["High"], df["Low"], df["Close"]
    idx = list(df.index)
    trs = []
    out = {}
    for i, d in enumerate(idx):
        # STAMP FIRST, using only true ranges from bars strictly before d. Doing
        # this after appending today's TR would fold today's own high/low into
        # the trail that today's arming decision uses — lookahead, and exactly
        # the kind that flatters a volatility rule, because a bar big enough to
        # arm a position is also a bar that widens its own ATR.
        if len(trs) >= ATR_WINDOW and i > 0:
            atr = sum(trs[-ATR_WINDOW:]) / ATR_WINDOW
            pc = float(closes.iloc[i - 1])
            if pc > 0:
                out[d] = atr / pc * 100.0
        if i > 0:
            pc = float(closes.iloc[i - 1])
            tr = max(float(highs.iloc[i]) - float(lows.iloc[i]),
                     abs(float(highs.iloc[i]) - pc),
                     abs(float(lows.iloc[i]) - pc))
            trs.append(tr)
    return out


# ── Variants ─────────────────────────────────────────────────────────

class Variant:
    def __init__(self, label, kind, trail_pct=None, mult=None, lo=None, hi=None):
        self.label, self.kind = label, kind
        self.trail_pct, self.mult, self.lo, self.hi = trail_pct, mult, lo, hi

    def trail_at_arm(self, atr_pct):
        """The trail this variant arms with, or None if it cannot arm."""
        if self.kind == "fixed":
            return self.trail_pct
        if atr_pct is None:
            return None          # no ATR -> variant is undefined for this trade
        return min(max(self.mult * atr_pct, self.lo), self.hi)


# ── Replay ───────────────────────────────────────────────────────────

def replay(trade: dict, df, atr_map, variant: Variant, floor_pp: float) -> dict:
    """Walk the hold day by day and apply ONE variant's armed trail.

    Mirrors kairos_exits.evaluate_position's armed path exactly: arm when the
    running peak reaches the position's own target; then
        backstop    = trail x 1.5
        close_trail = trail
        cap         = max(peak - floor_pp, 0.25)
        if cap < backstop: backstop = cap; close_trail = min(close_trail, cap)
    intraday retreat (peak - low_gain) against backstop on any bar; closing
    retreat (peak - close_gain) against close_trail on the bar's close.

    Returns the realised outcome, which is the ACTUAL exit whenever the trail
    never fired inside the hold.
    """
    import pandas as pd
    entry = float(trade["price_entry"])
    target = float(trade["target"])
    e_key = pd.Timestamp(trade["entry_dt"].date())
    x_key = pd.Timestamp(trade["exit_dt"].date())
    hold = df.loc[(df.index >= e_key) & (df.index <= x_key)]

    peak = 0.0
    armed = False
    trail = None
    arm_atr = None
    arm_date = None

    for d, bar in hold.iterrows():
        hg = (float(bar["High"]) - entry) / entry * 100.0
        lg = (float(bar["Low"]) - entry) / entry * 100.0
        cg = (float(bar["Close"]) - entry) / entry * 100.0
        peak = max(peak, hg)

        if not armed and peak >= target:
            arm_atr = atr_map.get(d)
            trail = variant.trail_at_arm(arm_atr)
            if trail is None:
                # Undefined for this variant (no ATR at arming) — report it as
                # such rather than falling back to the fixed trail, which would
                # quietly blend the two rules.
                return {"outcome": "no_atr_at_arm", "armed": False}
            armed = True
            arm_date = d

        if armed:
            backstop = trail * INTRADAY_MULT
            close_trail = trail
            floored = False
            if floor_pp > 0:
                cap = max(peak - floor_pp, 0.25)
                if cap < backstop:
                    backstop = cap
                    close_trail = min(close_trail, cap)
                    floored = True
            if peak - lg >= backstop:
                g = peak - backstop
                return _exit(trade, df, d, g, peak, "intraday", trail,
                             arm_atr, arm_date, floored)
            if peak - cg >= close_trail:
                # Fill at the close: the closing stop is evaluated ON the close,
                # so the realised price IS the close, not the trigger level.
                return _exit(trade, df, d, cg, peak, "close", trail,
                             arm_atr, arm_date, floored)

    return {
        "outcome": "actual", "armed": armed, "trail": trail,
        "arm_atr_pct": arm_atr, "arm_date": arm_date,
        "exit_gain": float(trade["pnl_pct"]),
        "exit_date": x_key, "exit_price": float(trade["price_exit"]),
        "peak_at_exit": max(peak, 0.0),
    }


def _exit(trade, df, d, gain, peak, how, trail, arm_atr, arm_date, floored):
    entry = float(trade["price_entry"])
    return {
        "outcome": "trail_fired", "armed": True, "how": how, "trail": trail,
        "arm_atr_pct": arm_atr, "arm_date": arm_date, "floored": floored,
        "exit_gain": gain, "exit_date": d,
        "exit_price": entry * (1 + gain / 100.0),
        "peak_at_exit": peak,
    }


def forgone(df, exit_date, exit_price, days, now):
    """Best exit still available within `days` of this one, or None if unmatured.

    Same measure as trade_outcomes.forgone_gain_Nd_pct: (max High in
    (exit, exit+N] - exit_price)/exit_price x 100.
    """
    import pandas as pd
    end = exit_date + timedelta(days=days)
    if now < (end.to_pydatetime().replace(tzinfo=timezone.utc)):
        return None
    win = df.loc[(df.index > exit_date) & (df.index <= pd.Timestamp(end.date()))]
    if len(win) == 0:
        return None
    return (float(win["High"].max()) - exit_price) / exit_price * 100.0


# ── Aggregation ──────────────────────────────────────────────────────

def summarise(results: list, label: str) -> dict:
    """Roll one variant's per-trade replays into the reported statistics."""
    gains = [r["exit_gain"] for r in results]
    gb = [r["give_back"] for r in results]
    gb_full = [r["give_back_full"] for r in results]
    fired = [r for r in results if r["outcome"] == "trail_fired"]

    def _band(lo, hi):
        return sum(1 for x in gb if (lo is None or x >= lo) and (hi is None or x < hi))

    whip = {}
    for h in FORGONE_HORIZONS:
        vals = [r[f"fg{h}"] for r in results if r.get(f"fg{h}") is not None]
        ran = [v for v in vals if v > 0]
        whip[h] = {
            "measurable": len(vals),
            "ran_higher": len(ran),
            "ran_gt_5pp": sum(1 for v in vals if v > 5.0),
            "median_runup": round(st.median(ran), 2) if ran else None,
            "mean_runup": round(sum(ran) / len(ran), 2) if ran else None,
        }
    return {
        "label": label,
        "n": len(results),
        "n_fired": len(fired),
        "n_intraday": sum(1 for r in fired if r.get("how") == "intraday"),
        "n_floored": sum(1 for r in fired if r.get("floored")),
        "total_return_pts": round(sum(gains), 2),
        "mean_return_pct": round(sum(gains) / len(gains), 3) if gains else None,
        "median_return_pct": round(st.median(gains), 3) if gains else None,
        "median_give_back": round(st.median(gb), 3) if gb else None,
        "mean_give_back": round(sum(gb) / len(gb), 3) if gb else None,
        "median_give_back_full": round(st.median(gb_full), 3) if gb_full else None,
        "gb_gt10": _band(10.0, None),
        "gb_5_10": _band(5.0, 10.0),
        "gb_lt5": _band(None, 5.0),
        "mean_trail": (round(sum(r["trail"] for r in fired) / len(fired), 3)
                       if fired else None),
        "whipsaw": whip,
    }


# ── Driver ───────────────────────────────────────────────────────────

def build_variants(bounds=(2.5, 12.0), mults=(1.5, 2.0, 2.5, 3.0)) -> list:
    live_trail, _ = _live()
    lo, hi = bounds
    v = [
        # Two fixed baselines. 6.9469 is the value the brief describes; 6.2255
        # is what is actually in kairos_config.json today (an approved proposal
        # moved it). Both are reported so the comparison is not resting on a
        # stale number.
        Variant(f"FIXED {live_trail:.4f} (live config)", "fixed", trail_pct=live_trail),
        Variant("FIXED 6.9469 (pre-approval)", "fixed", trail_pct=6.9469),
    ]
    for m in mults:
        v.append(Variant(f"ATR x{m} [{lo},{hi}]", "atr", mult=m, lo=lo, hi=hi))
    return v


def run(use_cache=True, bounds=(2.5, 12.0), mults=(1.5, 2.0, 2.5, 3.0),
        verbose=True):
    """Replay every variant over the clean set. Returns (summaries, per_trade, meta)."""
    now = datetime.now(timezone.utc)
    _, floor_pp = _live()
    trades = load_candidates()
    hist = fetch_paths(trades, use_cache=use_cache)

    # ── Data-integrity filter, FIRST ─────────────────────────────────
    clean, excluded = [], []
    for t in trades:
        df = hist.get(t["ticker"])
        ok, why, ratio = integrity_verdict(t, df)
        if ok:
            clean.append(t)
        else:
            excluded.append({"ticker": t["ticker"], "trade_id": t["trade_id"],
                             "reason": why, "implied_ratio": ratio,
                             "db_entry": t["price_entry"]})

    atr_cache = {tk: atr_pct_series(df) for tk, df in hist.items()}

    # Only trades that ARM under the shared arming condition are in scope; a
    # trade whose peak never reached its own target is untouched by every
    # variant and would only dilute every statistic identically.
    variants = build_variants(bounds, mults)
    per_variant = {v.label: [] for v in variants}
    in_scope = []
    no_atr = []

    for t in clean:
        df = hist[t["ticker"]]
        atr_map = atr_cache.get(t["ticker"], {})
        # Arming is variant-independent; test it once with the live fixed trail.
        probe = replay(t, df, atr_map, variants[0], floor_pp)
        if not probe.get("armed"):
            continue
        # Does an ATR exist at this trade's arming bar? If not, the ATR variants
        # are undefined and the trade must be dropped from ALL variants, or the
        # baseline and the ATR arms would run on different sets.
        atr_probe = replay(t, df, atr_map,
                           Variant("probe", "atr", mult=2.0,
                                   lo=bounds[0], hi=bounds[1]), floor_pp)
        if atr_probe.get("outcome") == "no_atr_at_arm":
            no_atr.append(t["ticker"])
            continue
        in_scope.append(t)

        for v in variants:
            r = replay(t, df, atr_map, v, floor_pp)
            r["ticker"] = t["ticker"]
            r["trade_id"] = t["trade_id"]
            r["target"] = t["target"]
            r["actual_gain"] = float(t["pnl_pct"])
            r["actual_mfe"] = t["mfe_pct"]
            r["full_hold_peak"] = t["mfe_pct"] if t["mfe_pct"] is not None else r["peak_at_exit"]
            r["give_back"] = r["peak_at_exit"] - r["exit_gain"]
            r["give_back_full"] = r["full_hold_peak"] - r["exit_gain"]
            for h in FORGONE_HORIZONS:
                r[f"fg{h}"] = forgone(df, r["exit_date"], r["exit_price"], h, now)
            per_variant[v.label].append(r)

    summaries = [summarise(per_variant[v.label], v.label) for v in variants]
    meta = {
        "n_candidates": len(trades), "n_clean": len(clean),
        "n_excluded_integrity": len(excluded), "excluded": excluded,
        "n_no_atr_at_arm": len(no_atr), "n_in_scope": len(in_scope),
        "floor_pp": floor_pp, "bounds": bounds, "mults": mults,
        "now": now.strftime("%Y-%m-%d %H:%M UTC"),
    }
    return summaries, per_variant, meta, in_scope, atr_cache, hist


# ── Reporting ────────────────────────────────────────────────────────

def _fmt(x, w=8, p=2):
    return f"{'—':>{w}}" if x is None else f"{x:>{w}.{p}f}"


def print_report(summaries, per_variant, meta):
    print()
    print("=" * 118)
    print("ATR-SCALED ARMED TRAIL — COUNTERFACTUAL BACKTEST (analysis only; "
          "no config or engine change)")
    print("=" * 118)
    print(f"  as of {meta['now']}   profit_floor_pp={meta['floor_pp']}   "
          f"intraday_mult={INTRADAY_MULT}   ATR window={ATR_WINDOW}d   "
          f"clamp=[{meta['bounds'][0]}, {meta['bounds'][1]}]")
    print(f"  closed LONG trades with their own logged thesis target: "
          f"{meta['n_candidates']}")
    print(f"  excluded by the data-integrity filter:                  "
          f"{meta['n_excluded_integrity']}")
    print(f"  dropped: no ATR14 available at the arming bar:          "
          f"{meta['n_no_atr_at_arm']}")
    print(f"  IN SCOPE (armed under the shared arming condition):     "
          f"{meta['n_in_scope']}")

    if meta["excluded"]:
        print("\n  ── excluded trades (DB entry price vs. the price series) ──")
        for e in meta["excluded"][:25]:
            ratio = "" if e["implied_ratio"] is None else \
                f"  implied ratio {e['implied_ratio']}x"
            print(f"     {e['ticker']:<7} db_entry={e['db_entry']}"
                  f"  {e['reason']}{ratio}")
        if len(meta["excluded"]) > 25:
            print(f"     ... and {len(meta['excluded']) - 25} more")

    print("\n" + "=" * 118)
    print("HEADLINE")
    print("=" * 118)
    hdr = (f"{'variant':<30}{'total pts':>11}{'mean %':>9}{'MEDIAN':>9}"
           f"{'mean':>8}{'>10pp':>7}{'5-10':>6}{'<5pp':>6}{'fired':>7}"
           f"{'intra':>7}{'floor':>7}{'mean':>8}")
    print(hdr)
    print(f"{'':<30}{'':>11}{'':>9}{'give-bk':>9}{'gv-bk':>8}{'':>7}{'':>6}"
          f"{'':>6}{'':>7}{'day':>7}{'hit':>7}{'trail':>8}")
    print("-" * 118)
    base = summaries[0]
    for s in summaries:
        print(f"{s['label']:<30}{s['total_return_pts']:>11.2f}"
              f"{_fmt(s['mean_return_pct'],9,3)}{_fmt(s['median_give_back'],9,3)}"
              f"{_fmt(s['mean_give_back'],8,2)}"
              f"{s['gb_gt10']:>7}{s['gb_5_10']:>6}{s['gb_lt5']:>6}"
              f"{s['n_fired']:>7}{s['n_intraday']:>7}{s['n_floored']:>7}"
              f"{_fmt(s['mean_trail'],8,3)}")
    print("-" * 118)
    print(f"  n = {base['n']} trades in every row (identical set, one rule "
          f"changed).  'total pts' = sum of per-trade % returns, equal weight.")

    print("\n" + "=" * 118)
    print("GIVE-BACK, RE-MEASURED AGAINST THE FULL ACTUAL-HOLD PEAK")
    print("=" * 118)
    print("  The MEDIAN column above uses each variant's OWN realised hold, which")
    print("  is the corpus definition but a moving denominator: exiting earlier")
    print("  truncates the window a peak can form in. This column holds the")
    print("  denominator fixed at the full actual hold's peak for every variant.")
    print(f"\n  {'variant':<30}{'median gb (own hold)':>22}"
          f"{'median gb (full-hold peak)':>28}")
    print("  " + "-" * 80)
    for s in summaries:
        print(f"  {s['label']:<30}{_fmt(s['median_give_back'],22,3)}"
              f"{_fmt(s['median_give_back_full'],28,3)}")

    print("\n" + "=" * 118)
    print("WHIPSAW — did the exit then run higher, and by how much")
    print("=" * 118)
    for h in FORGONE_HORIZONS:
        print(f"\n  ── {h}-day horizon " + "─" * 60)
        print(f"  {'variant':<30}{'measurable':>11}{'ran higher':>12}"
              f"{'>5pp higher':>13}{'median run-up':>15}{'mean run-up':>13}")
        print("  " + "-" * 94)
        for s in summaries:
            w = s["whipsaw"][h]
            print(f"  {s['label']:<30}{w['measurable']:>11}{w['ran_higher']:>12}"
                  f"{w['ran_gt_5pp']:>13}{_fmt(w['median_runup'],15,2)}"
                  f"{_fmt(w['mean_runup'],13,2)}")


def print_helped_hurt(per_variant, summaries):
    print("\n" + "=" * 118)
    print("PER-TRADE HELPED / HURT vs THE LIVE FIXED TRAIL")
    print("=" * 118)
    base_label = summaries[0]["label"]
    base = {r["trade_id"]: r for r in per_variant[base_label]}
    print(f"  {'variant':<30}{'helped':>8}{'hurt':>7}{'same':>7}"
          f"{'net pts':>10}{'avg help':>10}{'avg hurt':>10}")
    print("  " + "-" * 82)
    for s in summaries[1:]:
        rows = per_variant[s["label"]]
        helped = hurt = same = 0
        hsum = xsum = 0.0
        for r in rows:
            d = r["exit_gain"] - base[r["trade_id"]]["exit_gain"]
            if d > 0.01:
                helped += 1; hsum += d
            elif d < -0.01:
                hurt += 1; xsum += d
            else:
                same += 1
        net = hsum + xsum
        print(f"  {s['label']:<30}{helped:>8}{hurt:>7}{same:>7}{net:>10.2f}"
              f"{(hsum/helped if helped else 0):>10.2f}"
              f"{(xsum/hurt if hurt else 0):>10.2f}")


def print_vol_split(per_variant, summaries, split_at=None):
    """The premise of the whole exercise: one number cannot serve both ends."""
    base_label = summaries[0]["label"]
    atrs = [r["arm_atr_pct"] for r in per_variant[base_label]
            if r.get("arm_atr_pct") is not None]
    if not atrs:
        return
    med = split_at if split_at is not None else st.median(atrs)
    print("\n" + "=" * 118)
    print(f"HIGH-VOL vs LOW-VOL  (split at the median ATR14/price at arming = "
          f"{med:.3f}% daily)")
    print("=" * 118)
    for name, keep in (("LOW-VOL  (ATR% <= median)", lambda a: a <= med),
                       ("HIGH-VOL (ATR% >  median)", lambda a: a > med)):
        print(f"\n  ── {name} " + "─" * 55)
        print(f"  {'variant':<30}{'n':>5}{'total pts':>11}{'MEDIAN gb':>11}"
              f"{'>10pp':>7}{'<5pp':>6}{'fired':>7}{'mean trail':>12}")
        print("  " + "-" * 89)
        for s in summaries:
            rows = [r for r in per_variant[s["label"]]
                    if r.get("arm_atr_pct") is not None and keep(r["arm_atr_pct"])]
            if not rows:
                continue
            sub = summarise(rows, s["label"])
            print(f"  {s['label']:<30}{sub['n']:>5}{sub['total_return_pts']:>11.2f}"
                  f"{_fmt(sub['median_give_back'],11,3)}{sub['gb_gt10']:>7}"
                  f"{sub['gb_lt5']:>6}{sub['n_fired']:>7}"
                  f"{_fmt(sub['mean_trail'],12,3)}")


def print_bounds_sweep(use_cache=True):
    print("\n" + "=" * 118)
    print("LO/HI BOUNDS SENSITIVITY  (ATR_MULT held at each value; only the "
          "clamp moves)")
    print("=" * 118)
    print(f"  {'bounds':<14}{'mult':>6}{'total pts':>11}{'MEDIAN gb':>11}"
          f"{'>10pp':>7}{'<5pp':>6}{'fired':>7}{'clamped lo':>12}"
          f"{'clamped hi':>12}{'mean trail':>12}")
    print("  " + "-" * 108)
    for bounds in ((2.5, 12.0), (2.5, 10.0), (3.0, 10.0), (2.0, 15.0),
                   (4.0, 10.0), (2.5, 8.0)):
        for m in (2.0, 2.5):
            sums, pv, meta, *_ = run(use_cache=use_cache, bounds=bounds,
                                     mults=(m,), verbose=False)
            s = sums[-1]
            rows = [r for r in pv[s["label"]] if r["outcome"] == "trail_fired"]
            clo = sum(1 for r in rows
                      if abs(r["trail"] - bounds[0]) < 1e-9)
            chi = sum(1 for r in rows
                      if abs(r["trail"] - bounds[1]) < 1e-9)
            print(f"  [{bounds[0]},{bounds[1]}]".ljust(16)
                  + f"{m:>4.1f}{s['total_return_pts']:>13.2f}"
                  f"{_fmt(s['median_give_back'],11,3)}{s['gb_gt10']:>7}"
                  f"{s['gb_lt5']:>6}{s['n_fired']:>7}{clo:>12}{chi:>12}"
                  f"{_fmt(s['mean_trail'],12,3)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true",
                    help="refetch price paths instead of using the local cache")
    ap.add_argument("--bounds-sweep", action="store_true",
                    help="only run the LO/HI sensitivity sweep")
    args = ap.parse_args()
    use_cache = not args.no_cache

    if args.bounds_sweep:
        print_bounds_sweep(use_cache=use_cache)
        return 0

    sums, pv, meta, in_scope, atr_cache, hist = run(use_cache=use_cache)
    print_report(sums, pv, meta)
    print_helped_hurt(pv, sums)
    print_vol_split(pv, sums)
    print_bounds_sweep(use_cache=use_cache)
    print("\n  NOTHING WAS CHANGED: no config write, no edit to kairos_exits.py, "
          "no DB write.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
