#!/usr/bin/env python3
"""
kairos_backtest_congress_lag.py

Question: Is the alpha in HOT-CONGRESS (and the insider+congress confluence
cluster it mostly co-fires with) already realized BEFORE the STOCK Act
disclosure goes public -- i.e. before Kairos can legally see and act on it?

If most of the post-transaction excess return lands in the TransactionDate ->
ReportDate window (the window we CANNOT trade), the signal is structurally
dead. If meaningful excess return persists AFTER ReportDate, the signal is
salvageable and our problem is entry/sizing, not the premise.

Method (per congressional purchase in our executed population):
  - lost window:      TransactionDate -> ReportDate    (invisible to us)
  - tradeable window: ReportDate      -> ReportDate+Nd (N in HOLD_HORIZONS)
  - both measured as EXCESS over SPY.
  - kill/keep metric: fraction of total post-transaction excess return already
    gone before ReportDate. High fraction => dead.

Population: tickers/dates Kairos actually executed on, split into
  (1) Congress-only trades   (2) Insider+Congress (and larger) confluence.

READ-ONLY on kairos.db, Quiver historical endpoint, yfinance. Writes nothing
back to the live system.
"""
import os, sys, json, time, sqlite3, argparse
from datetime import datetime, timedelta, timezone
import requests

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kairos.db")
QUIVER_BASE = "https://api.quiverquant.com/beta/historical/congresstrading"
HOLD_HORIZONS = [5, 10, 21]
FETCH_TIMEOUT = 45


def load_congress_population():
    """Pull executed BUY rows tagged HOT-CONGRESS from kairos.db.

    Returns list of dicts: ticker, entry_ts, signals(list), bucket
    ('congress_only' | 'confluence'). Read-only.
    """
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute(
        "SELECT id, timestamp, ticker, action, data_inputs, rationale "
        "FROM decisions WHERE action='BUY' AND "
        "(lower(data_inputs) LIKE '%congress%' OR lower(rationale) LIKE '%congress%')"
    ).fetchall()
    con.close()

    pop = []
    for r in rows:
        sigs = []
        try:
            di = json.loads(r["data_inputs"] or "{}")
            sigs = di.get("confluence", {}).get("signals", []) or []
        except Exception:
            pass
        # normalize
        sigs = sorted(set(sigs))
        if not any("CONGRESS" in s for s in sigs):
            # tagged via rationale only, no structured signal array; keep but mark
            bucket = "unstructured"
        elif sigs == ["HOT-CONGRESS"]:
            bucket = "congress_only"
        else:
            bucket = "confluence"
        pop.append({
            "id": r["id"],
            "ticker": (r["ticker"] or "").upper(),
            "entry_ts": r["timestamp"],
            "signals": sigs,
            "bucket": bucket,
        })
    return pop


def fetch_quiver_history(ticker, api_key, retries=3, backoff=20):
    """GET Quiver historical congresstrading for one ticker. Returns list or None.
    Retries on 5xx/timeout (the API was mid-outage when this was written)."""
    h = {"Authorization": f"Token {api_key}", "Accept": "application/json",
         "User-Agent": "Kairos/1.0"}
    url = f"{QUIVER_BASE}/{ticker}"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=h, timeout=FETCH_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"    {ticker}: {r.status_code}, retry {attempt}/{retries} in {backoff}s")
                time.sleep(backoff)
                continue
            print(f"    {ticker}: HTTP {r.status_code} (non-retryable)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"    {ticker}: {type(e).__name__}, retry {attempt}/{retries} in {backoff}s")
            time.sleep(backoff)
    return None


def match_disclosure(records, entry_ts):
    """From a ticker's Quiver history, pick the congressional PURCHASE whose
    ReportDate best precedes/matches our entry (the disclosure Kairos acted on).
    Returns dict with TransactionDate, ReportDate (datetime.date) or None."""
    if not records:
        return None
    entry_d = _parse_date(entry_ts)
    best = None
    for rec in records:
        if "urchase" not in (rec.get("Transaction") or ""):
            continue
        rep = _parse_date(rec.get("ReportDate"))
        txn = _parse_date(rec.get("TransactionDate"))
        if not rep or not txn:
            continue
        # disclosure must be on/before our entry (we act after we can see it)
        if entry_d and rep > entry_d + timedelta(days=3):
            continue
        # prefer the disclosure closest to (just before) entry
        if best is None or rep > best["report"]:
            best = {"txn": txn, "report": rep,
                    "range": rec.get("Range"),
                    "rep_name": rec.get("Representative")}
    return best


def _parse_date(s):
    if not s:
        return None
    s = str(s)[:10]
    for fmt in ("%Y-%m-%d",):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def price_series(tickers, start, end):
    """Bulk yfinance download of adjusted closes for tickers + SPY.
    Returns a pandas DataFrame (index=date, cols=tickers). ffill'd."""
    import yfinance as yf
    syms = sorted(set(tickers) | {"SPY"})
    df = yf.download(syms, start=start, end=end, auto_adjust=True,
                     progress=False, threads=True)
    if df is None or df.empty:
        return None
    close = df["Close"] if hasattr(df.columns, "levels") else df
    return close.ffill()


def _ret_between(series, d0, d1):
    """Return pct change of `series` between the first available price on/after
    d0 and on/after d1. None if data missing."""
    import pandas as pd
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    idx = s.index
    def _at(d):
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos >= len(idx):
            return None
        return float(s.iloc[pos])
    p0, p1 = _at(d0), _at(d1)
    if p0 is None or p1 is None or p0 == 0:
        return None
    return (p1 / p0) - 1.0


def excess_return(prices, ticker, d0, d1):
    """Ticker return minus SPY return over [d0, d1]. None if either missing."""
    if prices is None or ticker not in prices.columns:
        return None
    rt = _ret_between(prices[ticker], d0, d1)
    rs = _ret_between(prices["SPY"], d0, d1)
    if rt is None or rs is None:
        return None
    return rt - rs


def analyze(pop, api_key, verbose=False):
    """For each trade with a matched disclosure, compute lost vs tradeable
    excess returns. Returns list of per-trade result dicts."""
    # cache Quiver history per ticker (many trades share tickers)
    hist_cache = {}
    matched = []
    tickers_needed = sorted({t["ticker"] for t in pop})
    print(f"\nFetching Quiver history for {len(tickers_needed)} unique tickers...")
    for tk in tickers_needed:
        hist_cache[tk] = fetch_quiver_history(tk, api_key)
        time.sleep(1)  # be polite to the API

    # match each trade to its disclosure
    for tr in pop:
        recs = hist_cache.get(tr["ticker"])
        disc = match_disclosure(recs, tr["entry_ts"])
        if disc:
            tr["txn_date"] = disc["txn"]
            tr["report_date"] = disc["report"]
            tr["lag_days"] = (disc["report"] - disc["txn"]).days
            matched.append(tr)
        elif verbose:
            print(f"  no disclosure match: {tr['ticker']} (entry {tr['entry_ts'][:10]})")

    if not matched:
        return []

    # price data spanning all needed windows
    all_dates = [t["txn_date"] for t in matched] + [t["report_date"] for t in matched]
    start = (min(all_dates) - timedelta(days=5)).isoformat()
    end = (max(all_dates) + timedelta(days=max(HOLD_HORIZONS) + 10)).isoformat()
    print(f"\nDownloading price data {start} -> {end} ...")
    prices = price_series([t["ticker"] for t in matched], start, end)

    for tr in matched:
        txn, rep = tr["txn_date"], tr["report_date"]
        tr["lost_excess"] = excess_return(prices, tr["ticker"], txn, rep)
        tr["tradeable_excess"] = {}
        for n in HOLD_HORIZONS:
            tr["tradeable_excess"][n] = excess_return(
                prices, tr["ticker"], rep, rep + timedelta(days=n))
    return matched


def _fmt_pct(x):
    return "  n/a" if x is None else f"{x*100:+6.2f}%"


def report(matched):
    import statistics as st
    if not matched:
        print("\nNo trades with usable disclosure + price data. "
              "Cannot draw conclusions.")
        return

    for bucket in ("congress_only", "confluence", "unstructured"):
        rows = [t for t in matched if t["bucket"] == bucket
                and t.get("lost_excess") is not None]
        if not rows:
            continue
        print("\n" + "=" * 68)
        print(f"  BUCKET: {bucket}   (n={len(rows)} with usable data)")
        print("=" * 68)
        lost = [t["lost_excess"] for t in rows]
        med_lag = st.median([t["lag_days"] for t in rows])
        print(f"  median disclosure lag: {med_lag:.0f} days "
              f"(transaction -> public)")
        print(f"  mean LOST excess (txn->disclosure, we can't trade): "
              f"{_fmt_pct(st.mean(lost))}")
        for n in HOLD_HORIZONS:
            tr_vals = [t["tradeable_excess"][n] for t in rows
                       if t["tradeable_excess"].get(n) is not None]
            if not tr_vals:
                continue
            mean_tr = st.mean(tr_vals)
            # kill/keep: share of total post-txn excess already gone by disclosure
            totals = []
            for t in rows:
                te = t["tradeable_excess"].get(n)
                if te is None:
                    continue
                total = t["lost_excess"] + te
                if abs(total) > 1e-9:
                    totals.append(t["lost_excess"] / total)
            share = st.mean(totals) if totals else None
            print(f"  hold {n:>2}d: tradeable excess (disclosure->+{n}d) = "
                  f"{_fmt_pct(mean_tr)}   "
                  f"| share of move already gone by disclosure: "
                  f"{'n/a' if share is None else f'{share*100:.0f}%'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--csv", help="optional path to dump per-trade rows")
    args = ap.parse_args()

    api_key = os.environ.get("CONGRESS_API_KEY") or os.environ.get("QUIVER_API_KEY")
    if not api_key:
        print("ERROR: CONGRESS_API_KEY / QUIVER_API_KEY not set in env.")
        sys.exit(1)

    pop = load_congress_population()
    from collections import Counter
    bc = Counter(t["bucket"] for t in pop)
    print(f"Loaded {len(pop)} executed Congress-tagged BUY rows: {dict(bc)}")

    matched = analyze(pop, api_key, verbose=args.verbose)
    print(f"\nMatched {len(matched)}/{len(pop)} trades to a disclosure with price data.")
    report(matched)

    if args.csv and matched:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "bucket", "entry_ts", "txn_date",
                        "report_date", "lag_days", "lost_excess",
                        *[f"tradeable_{n}d" for n in HOLD_HORIZONS]])
            for t in matched:
                w.writerow([t["ticker"], t["bucket"], t["entry_ts"],
                            t.get("txn_date"), t.get("report_date"),
                            t.get("lag_days"), t.get("lost_excess"),
                            *[t["tradeable_excess"].get(n) for n in HOLD_HORIZONS]])
        print(f"\nWrote per-trade rows -> {args.csv}")


if __name__ == "__main__":
    main()
