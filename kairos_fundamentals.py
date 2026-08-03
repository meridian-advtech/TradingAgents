"""Point-in-time fundamentals from SEC EDGAR XBRL company facts.

Kairos is valuation-blind: a HOT-INSIDER name at 60x earnings looks identical
to one at 12x. This is the data layer that closes that gap — deliberately ONLY
the data layer. Nothing here touches the trading path.

WHY EDGAR AND NOT A PRICE API: every XBRL fact carries the date it was FILED,
so `get_fundamentals(ticker, as_of)` can return strictly what was public on
that date. yfinance and most free fundamentals endpoints serve today's restated
figures, which silently leak future information into any backtest and make a
valuation study look better than it is. EDGAR is also free, has no vendor
dependency, and data.sec.gov is already a trusted host in this stack (Form 4
intake, IPO S-1 scanning).

WHAT IT DELIBERATELY IS NOT: a fundamentals warehouse. A small field set that
answers "is this expensive?" is enough to test whether valuation-awareness is
worth anything. Widen it only if the diagnostic says it matters.

KNOWN LIMITATION — NOT YET TTM. Flow fields (revenue, net_income, eps_diluted,
operating_cash_flow) are returned as filed, which means the value alternates
between ANNUAL (10-K) and QUARTERLY (10-Q) depending on which filing landed
most recently. AAPL revenue reads 416B as of 2026-01-10 (FY 10-K) and 111B as
of 2026-06-01 (a single quarter). Those are not comparable, and a P/E built
from a mix of the two is meaningless.

`form` is stored on every row so the two can be told apart, and the next step
before any valuation ratio is a TTM roll-up: sum the last four quarters, or
take the 10-K and add the quarters filed since. Stock fields (cash,
total_debt, equity, shares_diluted) are point-in-time balances and need no
such treatment. Do NOT compute ratios off this module until TTM lands.

Usage:
    python kairos_fundamentals.py --coverage         # universe coverage report
    python kairos_fundamentals.py --ticker AAPL      # inspect one name
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
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "kairos_universe.json")

# SEC asks for a descriptive UA and ~10 req/s max. Matches kairos_thesis_validity.
_SEC_UA = {"User-Agent": "Kairos Trading Research jason@meridiangroup.llc"}
_SEC_TIMEOUT = 20
_SEC_SLEEP = 0.15          # ~7 req/s, comfortably inside SEC's limit

# US-GAAP XBRL tags per logical field, in preference order — filers are not
# consistent, so each field tries several tags before giving up.
FIELD_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding",
                       "WeightedAverageNumberOfSharesOutstandingBasic"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "total_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "DebtLongtermAndShorttermCombinedAmount"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals_facts (
    ticker      TEXT NOT NULL,
    field       TEXT NOT NULL,
    tag         TEXT NOT NULL, -- the XBRL tag this came from
    tag_rank    INTEGER,       -- preference order within FIELD_TAGS[field]
    fy          INTEGER,
    fp          TEXT,
    unit        TEXT,
    value       REAL,
    period_end  TEXT,          -- 'end' of the reported period
    filed       TEXT NOT NULL, -- FILING date: what makes this point-in-time
    form        TEXT,
    accession   TEXT,
    -- tag is part of the key: one filing can report two tags for the same
    -- logical field, and collapsing them would drop one arbitrarily.
    PRIMARY KEY (ticker, field, tag, accession, period_end)
);
CREATE INDEX IF NOT EXISTS idx_fund_lookup
    ON fundamentals_facts (ticker, field, filed);
CREATE TABLE IF NOT EXISTS fundamentals_fetch_log (
    ticker TEXT PRIMARY KEY, cik TEXT, fetched_at TEXT,
    n_facts INTEGER, status TEXT, detail TEXT
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _conn()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ── Universe + CIK resolution ────────────────────────────────────────

def load_universe() -> list[str]:
    """Tickers Kairos actually trades (tier_a + tier_b; crypto excluded)."""
    try:
        with open(UNIVERSE_FILE) as f:
            u = json.load(f)
    except (IOError, json.JSONDecodeError):
        return []
    # The universe file nests inconsistently: tier_a holds
    # equities/etfs -> {group: [ticker, ...]}, while tier_b holds
    # tickers -> [{symbol, name, sector}, ...]. Walk it generically rather than
    # hard-coding either shape, and take symbols from dicts where present.
    out: list[str] = []

    def walk(node):
        if isinstance(node, str):
            out.append(node.upper())
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, dict):
            sym = node.get("symbol") or node.get("SYMBOL")
            if isinstance(sym, str):
                out.append(sym.upper())
                return
            for k, v in node.items():
                if k in ("added_date", "description", "last_updated", "name",
                         "sector", "promotion_reason", "promoted_from"):
                    continue
                walk(v)

    for key in ("tier_a", "tier_b"):   # crypto deliberately excluded
        walk(u.get(key))
    # Guard against metadata strings sneaking in as "tickers".
    return sorted({t for t in out if t and len(t) <= 6 and t.replace("-", "").isalnum()})


def resolve_cik(ticker: str):
    """ticker → zero-padded 10-digit CIK. Reuses the thesis-validity resolver
    so there is exactly one SEC ticker-map implementation in the stack."""
    try:
        from kairos_thesis_validity import _sec_cik
        return _sec_cik(ticker)
    except Exception:
        return None


# ── Fetch ────────────────────────────────────────────────────────────

def fetch_company_facts(ticker: str) -> dict:
    """Pull companyfacts for one ticker. Returns {"status", "facts"|"detail"}."""
    cik = resolve_cik(ticker)
    if not cik:
        return {"status": "no_cik", "detail": "ticker not in SEC mapping"}
    try:
        import requests
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=_SEC_UA, timeout=_SEC_TIMEOUT)
        if resp.status_code == 404:
            return {"status": "no_facts", "cik": cik,
                    "detail": "no XBRL company facts (often a foreign filer or ETF)"}
        if resp.status_code != 200:
            return {"status": "http_error", "cik": cik,
                    "detail": f"HTTP {resp.status_code}"}
        return {"status": "ok", "cik": cik, "facts": resp.json()}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def extract_facts(ticker: str, payload: dict) -> list[dict]:
    """Flatten companyfacts JSON into rows for the fields we care about.

    Every row keeps its `filed` date — that is the whole point. A row is only
    usable as of a date on or after it was filed.
    """
    rows: list[dict] = []
    usgaap = ((payload.get("facts") or {}).get("us-gaap") or {})
    for field, tags in FIELD_TAGS.items():
        # Collect EVERY tag that exists, not just the first. Filers migrate
        # tags over time — Apple's last "Revenues" fact is from FY2018, after
        # which it reports RevenueFromContractWithCustomerExcludingAssessedTax.
        # Stopping at the first tag present therefore pinned revenue to a
        # seven-year-old number. Preference between tags is applied at READ
        # time (most recently filed wins, ties broken by tag order), so a
        # mid-history tag switch resolves correctly in both directions.
        for rank, tag in enumerate(tags):
            entry = usgaap.get(tag)
            if not entry:
                continue
            for unit, items in (entry.get("units") or {}).items():
                for it in items:
                    filed = it.get("filed")
                    if not filed:
                        continue
                    rows.append({
                        "ticker": ticker, "field": field, "tag": tag,
                        "tag_rank": rank,
                        "fy": it.get("fy"), "fp": it.get("fp"), "unit": unit,
                        "value": it.get("val"), "period_end": it.get("end"),
                        "filed": filed, "form": it.get("form"),
                        "accession": it.get("accn") or "",
                    })
    return rows


def store_facts(rows: list[dict]) -> int:
    if not rows:
        return 0
    conn = _conn()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO fundamentals_facts "
            "(ticker, field, tag, tag_rank, fy, fp, unit, value, period_end, "
            " filed, form, accession) "
            "VALUES (:ticker,:field,:tag,:tag_rank,:fy,:fp,:unit,:value,"
            ":period_end,:filed,:form,:accession)",
            rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def sync_ticker(ticker: str) -> dict:
    """Fetch + store one ticker. Returns a status dict."""
    init_db()
    res = fetch_company_facts(ticker)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if res["status"] != "ok":
        conn = _conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO fundamentals_fetch_log "
                "(ticker, cik, fetched_at, n_facts, status, detail) VALUES (?,?,?,?,?,?)",
                (ticker, res.get("cik"), now, 0, res["status"], res.get("detail")))
            conn.commit()
        finally:
            conn.close()
        return res
    rows = extract_facts(ticker, res["facts"])
    n = store_facts(rows)
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals_fetch_log "
            "(ticker, cik, fetched_at, n_facts, status, detail) VALUES (?,?,?,?,?,?)",
            (ticker, res.get("cik"), now, n, "ok",
             f"{len({r['field'] for r in rows})} field(s)"))
        conn.commit()
    finally:
        conn.close()
    return {"status": "ok", "ticker": ticker, "n_facts": n,
            "fields": sorted({r["field"] for r in rows})}


# ── Point-in-time read ───────────────────────────────────────────────

def get_fundamentals(ticker: str, as_of: str | None = None) -> dict:
    """Fundamentals for `ticker` as they were PUBLICLY KNOWN on `as_of`.

    Only facts with filed <= as_of are considered, so a backtest evaluating a
    2026-06-01 decision cannot see a filing made in July. For each field the
    most recently FILED value is returned; ties break on the later period_end.

    Returns {field: {"value", "period_end", "filed", "form", "unit"}}. Missing
    fields are simply absent — callers must handle partial coverage, since
    filers differ in which tags they report.
    """
    as_of = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    as_of = str(as_of)[:10]
    out: dict = {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            for field in FIELD_TAGS:
                # Most recently FILED wins; then latest period covered; then
                # the preferred tag when one filing reports several. tag_rank
                # last so a newer tag never loses to a stale preferred one.
                row = conn.execute(
                    "SELECT value, period_end, filed, form, unit, tag "
                    "FROM fundamentals_facts "
                    "WHERE ticker = ? AND field = ? AND filed <= ? "
                    "  AND value IS NOT NULL "
                    "ORDER BY filed DESC, period_end DESC, "
                    "         COALESCE(tag_rank, 99) ASC LIMIT 1",
                    (ticker.upper(), field, as_of)).fetchone()
                if row is not None:
                    out[field] = dict(row)
        finally:
            conn.close()
    except Exception:
        return {}
    return out


def coverage_report(tickers: list[str] | None = None) -> dict:
    """What fraction of the universe has usable stored fundamentals."""
    tickers = tickers or load_universe()
    init_db()
    conn = _conn()
    try:
        have = {r["ticker"] for r in conn.execute(
            "SELECT DISTINCT ticker FROM fundamentals_facts")}
        logged = {r["ticker"]: dict(r) for r in conn.execute(
            "SELECT * FROM fundamentals_fetch_log")}
    finally:
        conn.close()
    missing = [t for t in tickers if t not in have]
    by_status: dict = {}
    for t in tickers:
        st = logged.get(t, {}).get("status", "not_fetched")
        by_status[st] = by_status.get(st, 0) + 1
    return {"universe": len(tickers), "with_facts": len(have & set(tickers)),
            "missing": len(missing), "by_status": by_status,
            "missing_sample": missing[:15]}


def main() -> int:
    ap = argparse.ArgumentParser(description="SEC EDGAR point-in-time fundamentals")
    ap.add_argument("--sync", action="store_true", help="fetch the whole universe")
    ap.add_argument("--ticker", action="append", help="fetch/inspect specific ticker(s)")
    ap.add_argument("--coverage", action="store_true", help="coverage report")
    ap.add_argument("--as-of", help="point-in-time date for --ticker reads (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, help="cap tickers synced (for a trial run)")
    args = ap.parse_args()

    if args.coverage:
        rep = coverage_report()
        print(f"universe          : {rep['universe']}")
        print(f"with stored facts : {rep['with_facts']}")
        print(f"missing           : {rep['missing']}")
        print(f"by fetch status   : {rep['by_status']}")
        if rep["missing_sample"]:
            print(f"missing sample    : {', '.join(rep['missing_sample'])}")
        return 0

    if args.ticker and not args.sync:
        for t in args.ticker:
            t = t.upper()
            stored = get_fundamentals(t, args.as_of)
            if not stored:
                print(f"{t}: nothing stored — fetching…")
                print(f"  {sync_ticker(t)}")
                stored = get_fundamentals(t, args.as_of)
            label = f" as of {args.as_of}" if args.as_of else ""
            print(f"\n{t}{label}:")
            if not stored:
                print("  (no facts)")
            for f, v in sorted(stored.items()):
                print(f"  {f:<20} {v['value']:>18,.2f}  "
                      f"[{v['unit']}] period_end={v['period_end']} filed={v['filed']}")
        return 0

    if args.sync:
        tickers = [t.upper() for t in args.ticker] if args.ticker else load_universe()
        if args.limit:
            tickers = tickers[:args.limit]
        print(f"syncing {len(tickers)} ticker(s) from SEC EDGAR…")
        ok = 0
        for i, t in enumerate(tickers, 1):
            r = sync_ticker(t)
            if r.get("status") == "ok":
                ok += 1
            if i % 25 == 0 or i == len(tickers):
                print(f"  {i}/{len(tickers)}  ok={ok}")
            time.sleep(_SEC_SLEEP)
        print(f"done: {ok}/{len(tickers)} fetched")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
