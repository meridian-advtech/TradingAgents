"""Catch universe tickers that have stopped trading under that symbol.

Three separate sessions on 2026-08-03 independently tripped over the same
dead symbols — an EDGAR fundamentals sync, a sector-classification backfill,
and a valuation study — because nothing in Kairos checks whether a universe
entry still corresponds to a tradeable security. Renames (BK->BNY,
MMC->MRSH, SQ->XYZ), delistings, and outright wrong mappings (Tier B lists
TTM for "TTM Technologies", which trades as TTMI) all sit in the universe
indefinitely and are screened every cycle.

The cost is not just wasted screening. A renamed symbol that the Council
selects will fail at order placement, and it fails at the worst moment — when
the system has decided it wants the position.

DELIBERATELY DOES NOT AUTO-REMAP. Guessing a replacement ticker from a fuzzy
name match is the wrong risk in a system that places orders: the failure mode
is silently trading the wrong company. This flags and stops. A human confirms
each one.

Signals used, cheapest first, and a ticker must fail BOTH to be called dead:
  1. recent price history (yfinance) — the market's own answer to "does this
     symbol trade?"
  2. presence in SEC's ticker->CIK registry — the regulator's answer to
     "does this issuer file under that symbol?"
Requiring both avoids condemning a live name over one vendor's outage, and
the two disagreeing is itself informative (see MISMATCH below).

Usage:
    python kairos_universe_liveness.py                 # check all, report
    python kairos_universe_liveness.py --held-only     # only open positions
    python kairos_universe_liveness.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")

STALE_DAYS = 10          # no trade in this many days -> price signal says dead
_SEC_UA = {"User-Agent": "Kairos Trading Research jason@meridiangroup.llc"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_liveness (
    ticker       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,   -- live | dead | mismatch | unknown
    has_price    INTEGER,
    last_price_date TEXT,
    in_sec       INTEGER,
    cik          TEXT,
    held         INTEGER,
    detail       TEXT,
    checked_at   TEXT NOT NULL
);
"""


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _sec_registry() -> dict:
    """{TICKER: cik} from SEC. Empty dict on failure — callers must handle."""
    out = {}
    for url in ("https://www.sec.gov/files/company_tickers.json",
                "https://www.sec.gov/files/company_tickers_exchange.json"):
        try:
            import requests
            d = requests.get(url, headers=_SEC_UA, timeout=25).json()
            if isinstance(d, dict) and "data" in d and "fields" in d:
                ti = d["fields"].index("ticker")
                ci = d["fields"].index("cik")
                for row in d["data"]:
                    t = str(row[ti]).upper()
                    out.setdefault(t, row[ci])
            else:
                for v in d.values():
                    t = str(v.get("ticker", "")).upper()
                    if t:
                        out.setdefault(t, v.get("cik_str"))
        except Exception:
            continue
    return out


def _price_status(tickers: list) -> dict:
    """{ticker: (has_recent_trade, last_date, vendor_answered)}.

    vendor_answered distinguishes "the vendor told us this symbol has no
    trades" from "the vendor did not answer". Conflating them is how a single
    failed batch condemns 50 live stocks: on the first full run, one chunk
    came back empty and MO through SCHW — Altria, Moderna, Motorola, M&T — were
    all recorded as priceless. A symbol is only ever called dead on a real
    answer, never on silence.

    Any ticker the batch pass could not price is retried INDIVIDUALLY, because
    a batch is all-or-nothing while a single-symbol request is not.
    """
    out = {t: (False, None, False) for t in tickers}
    try:
        import yfinance as yf
    except Exception:
        return out

    def _record(t, ser):
        ser = ser.dropna()
        if len(ser):
            last = ser.index[-1].date()
            age = (datetime.now(timezone.utc).date() - last).days
            out[t] = (age <= STALE_DAYS, last.isoformat(), True)
            return True
        return False

    CHUNK = 60
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            df = yf.download(" ".join(batch), period="1mo", interval="1d",
                             auto_adjust=True, progress=False,
                             group_by="ticker", threads=True)
        except Exception:
            df = None
        got = 0
        if df is not None and len(df):
            for t in batch:
                try:
                    ser = df["Close"] if len(batch) == 1 else df[t]["Close"]
                    if _record(t, ser):
                        got += 1
                except Exception:
                    continue
        if got == 0 and len(batch) > 1:
            print(f"    batch {i//CHUNK + 1} returned nothing for all "
                  f"{len(batch)} symbols — treating as a vendor failure, "
                  f"not {len(batch)} delistings")
        time.sleep(0.2)

    # Individual retry for everything still unpriced.
    missing = [t for t in tickers if not out[t][2]]
    if missing:
        print(f"  retrying {len(missing)} unpriced symbol(s) individually…")
        for t in missing:
            try:
                h = yf.Ticker(t).history(period="1mo", interval="1d",
                                         auto_adjust=True)
                if h is not None and len(h) and "Close" in h:
                    if _record(t, h["Close"]):
                        continue
                # A real, empty answer from a single-symbol request.
                out[t] = (False, None, True)
            except Exception:
                out[t] = (False, None, False)   # still no answer
            time.sleep(0.1)
    return out


def _held_tickers() -> set:
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            return {r[0] for r in conn.execute(
                "SELECT DISTINCT ticker FROM holdings WHERE sold_date IS NULL")}
        finally:
            conn.close()
    except Exception:
        return set()


def check(tickers: list | None = None, held_only: bool = False) -> list:
    from kairos_fundamentals import load_universe
    held = _held_tickers()
    if tickers:
        universe = [t.upper() for t in tickers]
    elif held_only:
        universe = sorted(held)
    else:
        universe = load_universe()

    print(f"checking {len(universe)} ticker(s)…")
    sec = _sec_registry()
    if not sec:
        print("  WARNING: SEC registry unavailable — falling back to price "
              "signal alone, which cannot distinguish a rename from a halt.")
    prices = _price_status(universe)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    results = []
    for t in universe:
        has_price, last_date, answered = prices.get(t, (False, None, False))
        in_sec = (t in sec) if sec else None
        cik = str(sec.get(t)) if sec and t in sec else None

        if not answered:
            # The price vendor never answered. Silence is not evidence of
            # death — say so rather than guessing.
            results.append({"ticker": t, "status": "unknown", "has_price": False,
                            "last_price_date": None, "in_sec": in_sec, "cik": cik,
                            "held": t in held,
                            "detail": "price vendor did not answer — re-run; "
                                      "no conclusion drawn",
                            "checked_at": now})
            continue

        if has_price and (in_sec or in_sec is None):
            status, detail = "live", ""
        elif has_price and not in_sec:
            # Trades but does not file under this symbol: funds/ETFs legitimately
            # look like this, so it is a MISMATCH to inspect, not a death.
            status = "mismatch"
            detail = ("trades but absent from the SEC registry — normal for a "
                      "fund/ETF, otherwise check the symbol")
        elif not has_price and in_sec:
            status = "mismatch"
            detail = ("files with SEC but no recent trades — possible halt, "
                      "or a symbol the price vendor spells differently")
        elif not has_price and in_sec is False:
            status = "dead"
            detail = (f"no trade in {STALE_DAYS}d and absent from the SEC "
                      f"registry — renamed or delisted; confirm by hand")
        else:
            status, detail = "unknown", "no usable signal"

        results.append({"ticker": t, "status": status, "has_price": has_price,
                        "last_price_date": last_date, "in_sec": in_sec,
                        "cik": cik, "held": t in held, "detail": detail,
                        "checked_at": now})

    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO universe_liveness "
            "(ticker,status,has_price,last_price_date,in_sec,cik,held,detail,checked_at) "
            "VALUES (:ticker,:status,:has_price,:last_price_date,:in_sec,:cik,"
            ":held,:detail,:checked_at)",
            [{**r, "has_price": int(r["has_price"]),
              "in_sec": None if r["in_sec"] is None else int(r["in_sec"]),
              "held": int(r["held"])} for r in results])
        conn.commit()
    finally:
        conn.close()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Universe liveness check")
    ap.add_argument("--ticker", action="append")
    ap.add_argument("--held-only", action="store_true")
    ap.add_argument("--json", help="write full results to this path")
    ap.add_argument("--alert", action="store_true",
                    help="post a #kairos-alerts summary when dead symbols are held")
    args = ap.parse_args()

    results = check(args.ticker, args.held_only)
    by = {}
    for r in results:
        by.setdefault(r["status"], []).append(r)

    print()
    for status in ("dead", "mismatch", "unknown", "live"):
        rows = by.get(status, [])
        if not rows:
            continue
        if status == "live":
            print(f"live      : {len(rows)}")
            continue
        print(f"{status:<10}: {len(rows)}")
        for r in sorted(rows, key=lambda x: x["ticker"]):
            flag = "  ** HELD **" if r["held"] else ""
            last = r["last_price_date"] or "never"
            print(f"    {r['ticker']:<7} last_trade={last:<12} "
                  f"sec={r['in_sec']}{flag}")
            if r["detail"]:
                print(f"            {r['detail']}")

    dead_held = [r for r in by.get("dead", []) if r["held"]]
    print()
    if dead_held:
        print(f"*** {len(dead_held)} DEAD symbol(s) are currently HELD: "
              f"{', '.join(r['ticker'] for r in dead_held)}")
        print("    These cannot be sold under this symbol. Resolve before the "
              "next cycle.")
    else:
        print("No dead symbols are currently held.")
    print("\nNothing was auto-remapped. Confirm each replacement by hand before "
          "editing kairos_universe.json.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")

    if args.alert and (by.get("dead") or dead_held):
        try:
            from kairos_alerts import post_message
            names = ", ".join(r["ticker"] for r in by.get("dead", []))
            post_message("alerts",
                f":warning: *Universe liveness*: {len(by.get('dead', []))} symbol(s) "
                f"no longer trade — {names}\n"
                + (f"*{len(dead_held)} of these are HELD.*\n" if dead_held else "")
                + "Not auto-remapped; confirm replacements by hand.")
        except Exception as exc:
            print(f"alert post failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
