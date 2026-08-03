"""Does entry valuation relate to trade outcome?

This is step 2 of the fundamentals plan and it is a GO/NO-GO, not a feature. If
valuation at entry shows no relationship to how a trade turned out, the
"valuation blind spot" is not costing anything measurable and the work stops
here rather than proceeding to touch the Council prompt.

Read-only. Reads closed trades from kairos_ml_outcomes.db, point-in-time TTM
fundamentals from kairos.db, and sectors from security_master. Writes nothing
and changes no behaviour.

METHOD
  For each closed trade, value the position as it was PRICED AT ENTRY using
  only fundamentals filed on or before the entry date:
      P/E = price_entry / eps_ttm
      P/S = price_entry / (revenue_ttm / shares_diluted)
  Then relate that to the realised pnl_pct.

WHY SECTOR-RELATIVE MATTERS
  Raw multiples are not comparable across sectors — software trades at
  multiples utilities never will — so a raw P/E-vs-outcome correlation mostly
  measures sector mix, not valuation. Every headline number here is therefore
  reported sector-relative: a trade's valuation is expressed as its percentile
  among the universe tickers in the SAME sector, valued point-in-time on the
  same date. Raw figures are shown alongside only to make the confound visible.

READ THE CAVEATS AT THE BOTTOM OF THE OUTPUT BEFORE ACTING ON ANY OF IT.
"""

from __future__ import annotations

import os
import sqlite3
import statistics as st
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
ML_DB = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
DB = os.path.join(SCRIPT_DIR, "kairos.db")

from kairos_fundamentals import get_fundamentals_ttm, load_universe  # noqa: E402


def _spearman(xs, ys):
    """Rank correlation, ties averaged. None when n < 4."""
    n = len(xs)
    if n < 4:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return (num / den) if den else None


def valuation_at(ticker: str, price: float, as_of: str):
    """(pe, ps) from point-in-time TTM fundamentals, either may be None."""
    f = get_fundamentals_ttm(ticker, as_of)
    if not f:
        return None, None
    pe = ps = None
    eps = f.get("eps_diluted", {}).get("value")
    if eps and eps > 0:
        pe = price / eps
    rev = f.get("revenue", {}).get("value")
    sh = f.get("shares_diluted", {}).get("value")
    if rev and sh and rev > 0 and sh > 0:
        sps = rev / sh
        if sps > 0:
            ps = price / sps
    return pe, ps


def main() -> int:
    sectors = {}
    try:
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        for t, s in c.execute("SELECT ticker, sector FROM security_master"):
            sectors[t] = s
        c.close()
    except Exception as exc:
        print(f"security_master unavailable ({exc}) — sector-relative analysis "
              f"cannot run.")
        return 1

    m = sqlite3.connect(f"file:{ML_DB}?mode=ro", uri=True)
    m.row_factory = sqlite3.Row
    trades = [dict(r) for r in m.execute(
        "SELECT ticker, timestamp_entry, price_entry, pnl_pct, signals_fired, "
        "       signal_attribution_source "
        "FROM trade_outcomes "
        "WHERE timestamp_exit IS NOT NULL AND price_entry IS NOT NULL "
        "  AND pnl_pct IS NOT NULL AND price_entry > 0 "
        "ORDER BY timestamp_entry")]
    m.close()
    print(f"closed trades with entry price and P&L : {len(trades)}")

    # Value every trade at its own entry date.
    rows = []
    for t in trades:
        as_of = str(t["timestamp_entry"])[:10]
        pe, ps = valuation_at(t["ticker"], float(t["price_entry"]), as_of)
        if pe is None and ps is None:
            continue
        rows.append({**t, "as_of": as_of, "pe": pe, "ps": ps,
                     "sector": sectors.get(t["ticker"], "Unclassified")})
    print(f"  with a usable valuation at entry     : {len(rows)}")
    print(f"  with P/E                             : "
          f"{sum(1 for r in rows if r['pe'])}")
    print(f"  with P/S                             : "
          f"{sum(1 for r in rows if r['ps'])}")

    # Sector percentile: rank each trade's P/S against the universe tickers in
    # the SAME sector, valued point-in-time on the SAME date. Cached per
    # (sector, date) so each distinct date costs one pass.
    universe = load_universe()
    by_sector = defaultdict(list)
    for u in universe:
        by_sector[sectors.get(u, "Unclassified")].append(u)

    cache: dict = {}

    def sector_ps_distribution(sector: str, as_of: str):
        key = (sector, as_of)
        if key in cache:
            return cache[key]
        vals = []
        for u in by_sector.get(sector, []):
            f = get_fundamentals_ttm(u, as_of)
            if not f:
                continue
            rev = f.get("revenue", {}).get("value")
            sh = f.get("shares_diluted", {}).get("value")
            px = f.get("_price")  # not available historically; see caveats
            if rev and sh and rev > 0 and sh > 0:
                vals.append(rev / sh)      # sales per share
        cache[key] = vals
        return vals

    # Sales-per-share is knowable point-in-time; historical PRICE for every
    # universe peer is not stored, so the peer distribution is built on
    # sales-per-share and the trade is ranked by its own P/S against the
    # sector's P/S implied by peers' SPS at the trade's own price. That is a
    # weaker control than a true peer-price panel — stated plainly in caveats.
    scored = []
    for r in rows:
        if not r["ps"]:
            continue
        peers = sector_ps_distribution(r["sector"], r["as_of"])
        if len(peers) < 5:
            continue
        own_sps = float(r["price_entry"]) / r["ps"]
        below = sum(1 for v in peers if v <= own_sps)
        # High percentile == high sales-per-share relative to sector, i.e.
        # CHEAPER on price/sales for a given price. Invert so that a high
        # percentile means EXPENSIVE, which is the intuitive direction.
        pct = 100.0 * (1.0 - below / len(peers))
        scored.append({**r, "sector_pct": pct})

    print(f"  with a sector-relative percentile     : {len(scored)}\n")

    def bucket_report(data, key, label):
        if len(data) < 20:
            print(f"  {label}: only {len(data)} trades — too few to read.\n")
            return
        s = sorted(data, key=lambda x: x[key])
        n = len(s)
        q = max(1, n // 5)
        print(f"  {label}  (n={n})")
        print(f"    {'quintile':<12}{'n':>4}{'mean P&L%':>11}{'median':>9}{'win%':>7}")
        for i in range(5):
            lo = i * q
            hi = (i + 1) * q if i < 4 else n
            chunk = s[lo:hi]
            if not chunk:
                continue
            p = [c["pnl_pct"] for c in chunk]
            wins = sum(1 for x in p if x > 0)
            name = ["cheapest", "2", "3", "4", "priciest"][i]
            print(f"    {name:<12}{len(chunk):>4}{sum(p)/len(p):>11.2f}"
                  f"{st.median(p):>9.2f}{100.0*wins/len(chunk):>7.1f}")
        rho = _spearman([c[key] for c in s], [c["pnl_pct"] for c in s])
        print(f"    Spearman rho (valuation vs P&L): "
              f"{rho:+.3f}" if rho is not None else "    rho: n/a")
        print()

    print("=" * 72)
    print("SECTOR-RELATIVE  (the headline — controls for sector mix)")
    print("=" * 72)
    bucket_report(scored, "sector_pct", "by sector percentile (100 = priciest in sector)")

    print("=" * 72)
    print("RAW MULTIPLES  (shown to expose the sector confound, not to act on)")
    print("=" * 72)
    bucket_report([r for r in rows if r["ps"]], "ps", "by raw P/S")
    bucket_report([r for r in rows if r["pe"]], "pe", "by raw P/E")

    print("=" * 72)
    print("CAVEATS")
    print("=" * 72)
    print("""  1. The peer control is WEAK. A true sector-relative valuation needs each
     peer's PRICE on the trade date; Kairos stores no historical price panel
     for the universe, so peers are compared on point-in-time sales-per-share
     and the traded name on its own entry price. Directional only.
  2. Sample is one regime. All trades fall in 2026-05-18 to 2026-07-31 — a
     single market environment. Valuation effects are regime-dependent and
     nothing here generalises past that window.
  3. Overlapping holds. Trades are not independent draws; several were open
     simultaneously in correlated names, so effective n is below nominal n.
  4. Survivorship in fundamentals coverage. Names missing from EDGAR (foreign
     filers, acquired shells) drop out, and they are not a random subset.
  5. P/E discards loss-makers by construction (eps <= 0 yields no P/E), which
     removes exactly the speculative names most likely to be valuation-
     sensitive. P/S is the more honest of the two here.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
