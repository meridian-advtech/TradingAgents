"""
Kairos Security Master — authoritative sector / industry classification

Resolves a ticker to a real GICS-style sector, separate from the universe
*bucket* it was screened under.  These are two different things that were
historically collapsed into one field:

    universe bucket   how a ticker entered the screener  ("large_cap",
                      "value_dividend", "mega_cap") — a size/style cohort
    sector            what industry the company is in    ("Financial
                      Services", "Industrials") — a risk dimension

kairos_confluence.lookup_sector() returns the *bucket*.  This module returns
the *sector*.  Both are legitimate; conflating them makes sector-concentration
risk invisible, because one real sector scatters across several buckets.

Resolution chain, highest precedence first:

    1. override    kairos_sector_overrides.json  (curated, reviewable)
    2. yfinance    .info sector / industry / quoteType
    3. sec_sic     EDGAR submissions SIC code -> sector
    4. unresolved  explicit; never silently bucketed

Every row records which source produced it, so any classification can be
explained after the fact.

This module is read-only with respect to trading behaviour: nothing in the
live path imports it yet.  Migrating call sites is a separate, deliberate step.

Usage:
    from kairos_security_master import get_sector, sector_exposure

    get_sector("JPM")                  -> "Financial Services"
    sector_exposure(positions)         -> {sector: market_value}

CLI:
    python kairos_security_master.py backfill [--limit N] [--force]
    python kairos_security_master.py resolve JPM MSFT ...
    python kairos_security_master.py status
    python kairos_security_master.py compare      # bucket vs sector, live book
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DB_PATH        = os.path.join(SCRIPT_DIR, "kairos.db")
UNIVERSE_FILE  = os.path.join(SCRIPT_DIR, "kairos_universe.json")
OVERRIDE_FILE  = os.path.join(SCRIPT_DIR, "kairos_sector_overrides.json")

# Sectors are near-static; a quarterly refresh is ample and keeps us well
# inside every upstream rate limit.
TTL_DAYS = 90

# SEC requires a User-Agent identifying the requestor.
SEC_HEADERS = {"User-Agent": "Kairos Trading System jason@meridiangroup.llc"}
SEC_THROTTLE_SEC = 0.12          # SEC asks for <= 10 req/s; stay under
YF_THROTTLE_SEC  = 0.05

UNRESOLVED = "Unclassified"

# Share-class separator differs by venue: IBKR and the Kairos universe write
# "BRK.B", while yfinance and SEC both write "BRK-B". Without normalising,
# every dual-class name silently falls through to Unclassified.
_CLASS_SEP_VENUES = str.maketrans({".": "-"})


def _vendor_symbol(ticker: str) -> str:
    """Kairos/IBKR symbol -> the form yfinance and SEC expect."""
    return ticker.upper().strip().translate(_CLASS_SEP_VENUES)

# The 11 GICS-style sector labels yfinance emits. Pinned here so a silent
# upstream rename shows up as an unrecognised value instead of quietly
# creating a 12th sector that splits exposure.
KNOWN_SECTORS = {
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
}

# Funds carry no single sector. They are classified by wrapper, not industry,
# and are excluded from sector-concentration maths rather than forced into a
# sector they don't belong to. Look-through is a later refinement.
FUND_QUOTE_TYPES = {"ETF", "MUTUALFUND", "INDEX"}
FUND_SECTOR = "Fund / ETF"

# SIC division ranges -> sector. Coarse by construction: SIC is a 1930s
# taxonomy and is only ever a fallback when yfinance has no answer.
_SIC_RANGES = [
    (100,   999,  "Basic Materials"),          # agriculture, forestry, fishing
    (1000,  1099, "Basic Materials"),          # metal mining
    (1200,  1299, "Energy"),                   # coal
    (1300,  1399, "Energy"),                   # oil & gas extraction
    (1400,  1499, "Basic Materials"),          # nonmetallic minerals
    (1500,  1799, "Industrials"),              # construction
    (2000,  2199, "Consumer Defensive"),       # food, tobacco
    (2200,  2399, "Consumer Cyclical"),        # textiles, apparel
    (2400,  2599, "Industrials"),              # lumber, furniture
    (2600,  2699, "Basic Materials"),          # paper
    (2700,  2799, "Communication Services"),   # printing & publishing
    (2800,  2829, "Basic Materials"),          # industrial chemicals
    (2830,  2836, "Healthcare"),               # drugs / biological products
    (2840,  2899, "Basic Materials"),          # soaps, other chemicals
    (2900,  2999, "Energy"),                   # petroleum refining
    (3000,  3299, "Basic Materials"),          # rubber, stone, clay, glass
    (3300,  3399, "Basic Materials"),          # primary metal
    (3400,  3569, "Industrials"),              # fabricated metal, machinery
    (3570,  3579, "Technology"),               # computer & office equipment
    (3580,  3599, "Industrials"),
    (3600,  3639, "Industrials"),              # electrical equipment
    (3640,  3699, "Technology"),               # electronic components
    (3700,  3799, "Consumer Cyclical"),        # transportation equipment
    (3800,  3829, "Technology"),               # instruments
    (3830,  3859, "Healthcare"),               # medical instruments
    (3860,  3999, "Industrials"),
    (4000,  4499, "Industrials"),              # transportation
    (4500,  4599, "Industrials"),              # air transport
    (4600,  4799, "Energy"),                   # pipelines
    (4800,  4899, "Communication Services"),   # communications
    (4900,  4999, "Utilities"),                # electric, gas, sanitary
    (5000,  5199, "Industrials"),              # wholesale
    (5200,  5599, "Consumer Cyclical"),        # retail
    (5600,  5699, "Consumer Cyclical"),        # apparel retail
    (5700,  5799, "Consumer Cyclical"),
    (5800,  5899, "Consumer Cyclical"),        # eating & drinking
    (5900,  5999, "Consumer Cyclical"),        # misc retail
    (6000,  6199, "Financial Services"),       # banking, credit
    (6200,  6299, "Financial Services"),       # brokers, exchanges
    (6300,  6499, "Financial Services"),       # insurance
    (6500,  6599, "Real Estate"),              # real estate
    (6798, 6798,  "Real Estate"),              # REITs
    (6700,  6799, "Financial Services"),       # holding & investment offices
    (7000,  7099, "Consumer Cyclical"),        # hotels
    (7200,  7299, "Consumer Cyclical"),        # personal services
    (7300,  7369, "Industrials"),              # business services
    (7370,  7379, "Technology"),               # computer services / software
    (7380,  7399, "Industrials"),
    (7800,  7999, "Communication Services"),   # entertainment, recreation
    (8000,  8099, "Healthcare"),               # health services
    (8200,  8299, "Consumer Defensive"),       # educational services
    (8300,  8399, "Healthcare"),               # social services
    (8700,  8799, "Industrials"),              # engineering, accounting, mgmt
]


# ── Schema ────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS security_master (
    ticker       TEXT PRIMARY KEY,
    sector       TEXT NOT NULL,
    industry     TEXT,
    asset_type   TEXT,               -- EQUITY / ETF / MUTUALFUND / UNKNOWN
    source       TEXT NOT NULL,      -- override | yfinance | sec_sic | unresolved
    detail       TEXT,               -- provenance breadcrumb (e.g. SIC code)
    resolved_at  TEXT NOT NULL,
    is_override  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_secmaster_sector ON security_master (sector);
"""


def _connect(readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    return sqlite3.connect(DB_PATH)


def ensure_schema() -> None:
    """Create the security_master table if it does not exist."""
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ── Overrides ─────────────────────────────────────────────────────────

def _load_overrides() -> dict[str, dict]:
    """Curated ticker -> {sector, industry, asset_type} map.

    Highest precedence in the chain. This is where funds, ADRs, and anything
    upstream gets wrong are pinned by hand.
    """
    if not os.path.exists(OVERRIDE_FILE):
        return {}
    try:
        with open(OVERRIDE_FILE) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"  WARNING: could not read {os.path.basename(OVERRIDE_FILE)}: {exc}")
        return {}

    out: dict[str, dict] = {}
    for ticker, val in (raw.get("overrides") or {}).items():
        if isinstance(val, str):
            out[ticker.upper()] = {"sector": val, "industry": None, "asset_type": None}
        elif isinstance(val, dict) and val.get("sector"):
            out[ticker.upper()] = {
                "sector":     val["sector"],
                "industry":   val.get("industry"),
                "asset_type": val.get("asset_type"),
            }
    return out


# ── Resolvers ─────────────────────────────────────────────────────────

def _resolve_yfinance(ticker: str) -> dict | None:
    """Resolve via yfinance .info. Returns None if unavailable."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        info = yf.Ticker(_vendor_symbol(ticker)).info or {}
    except Exception:
        return None

    quote_type = (info.get("quoteType") or "").upper()

    if quote_type in FUND_QUOTE_TYPES:
        return {
            "sector":     FUND_SECTOR,
            "industry":   info.get("category") or info.get("longName"),
            "asset_type": quote_type,
            "source":     "yfinance",
            "detail":     f"quoteType={quote_type}",
        }

    sector = (info.get("sector") or "").strip()
    if not sector:
        return None

    detail = None
    if sector not in KNOWN_SECTORS:
        # Not a hard failure — record it and move on, but make it visible so an
        # upstream taxonomy change can't silently fragment exposure.
        detail = f"unrecognised sector label: {sector!r}"

    return {
        "sector":     sector,
        "industry":   (info.get("industry") or "").strip() or None,
        "asset_type": quote_type or "EQUITY",
        "source":     "yfinance",
        "detail":     detail,
    }


_CIK_CACHE: dict[str, str] | None = None


def _cik_map() -> dict[str, str]:
    """ticker -> zero-padded 10-digit CIK, from SEC's public mapping file."""
    global _CIK_CACHE
    if _CIK_CACHE is not None:
        return _CIK_CACHE

    _CIK_CACHE = {}
    try:
        import requests
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        for row in resp.json().values():
            tkr = (row.get("ticker") or "").upper()
            cik = row.get("cik_str")
            if tkr and cik is not None:
                _CIK_CACHE[tkr] = str(cik).zfill(10)
    except Exception as exc:
        print(f"  WARNING: SEC CIK map unavailable: {exc}")

    return _CIK_CACHE


def _sector_from_sic(sic: int) -> str | None:
    for lo, hi, sector in _SIC_RANGES:
        if lo <= sic <= hi:
            return sector
    return None


def _resolve_sec_sic(ticker: str) -> dict | None:
    """Fallback: EDGAR submissions SIC code -> sector."""
    cik = _cik_map().get(_vendor_symbol(ticker))
    if not cik:
        return None

    try:
        import requests
        time.sleep(SEC_THROTTLE_SEC)
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=SEC_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    try:
        sic = int(data.get("sic") or 0)
    except (TypeError, ValueError):
        return None
    if not sic:
        return None

    sector = _sector_from_sic(sic)
    if not sector:
        return None

    return {
        "sector":     sector,
        "industry":   data.get("sicDescription") or None,
        "asset_type": "EQUITY",
        "source":     "sec_sic",
        "detail":     f"SIC {sic} (CIK {cik})",
    }


def resolve(ticker: str, overrides: dict | None = None) -> dict:
    """Resolve one ticker through the full chain. Always returns a dict."""
    ticker = ticker.upper().strip()
    if overrides is None:
        overrides = _load_overrides()

    ov = overrides.get(ticker)
    if ov:
        return {
            "sector":     ov["sector"],
            "industry":   ov.get("industry"),
            "asset_type": ov.get("asset_type") or "EQUITY",
            "source":     "override",
            "detail":     None,
            "is_override": 1,
        }

    for resolver in (_resolve_yfinance, _resolve_sec_sic):
        got = resolver(ticker)
        if got:
            got["is_override"] = 0
            return got

    return {
        "sector":     UNRESOLVED,
        "industry":   None,
        "asset_type": "UNKNOWN",
        "source":     "unresolved",
        "detail":     None,
        "is_override": 0,
    }


# ── Persistence ───────────────────────────────────────────────────────

def _upsert(conn: sqlite3.Connection, ticker: str, rec: dict) -> None:
    conn.execute(
        """
        INSERT INTO security_master
            (ticker, sector, industry, asset_type, source, detail, resolved_at, is_override)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            sector      = excluded.sector,
            industry    = excluded.industry,
            asset_type  = excluded.asset_type,
            source      = excluded.source,
            detail      = excluded.detail,
            resolved_at = excluded.resolved_at,
            is_override = excluded.is_override
        """,
        (
            ticker.upper(), rec["sector"], rec.get("industry"), rec.get("asset_type"),
            rec["source"], rec.get("detail"),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            rec.get("is_override", 0),
        ),
    )


def _stale_before() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=TTL_DAYS)).isoformat(timespec="seconds")


def _fresh_tickers(conn: sqlite3.Connection) -> set[str]:
    """Tickers resolved within TTL that did NOT fall through to unresolved.

    Unresolved rows are always retried — a miss is usually transient
    (rate limit, network) and should not be cached for a quarter.
    """
    rows = conn.execute(
        "SELECT ticker FROM security_master "
        "WHERE resolved_at >= ? AND source != 'unresolved'",
        (_stale_before(),),
    ).fetchall()
    return {r[0] for r in rows}


# ── Universe ──────────────────────────────────────────────────────────

def universe_tickers() -> list[str]:
    """Every ticker Kairos tracks: Tier A equities + ETFs, Tier B, Tier C."""
    out: set[str] = set()

    try:
        with open(UNIVERSE_FILE) as f:
            uni = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"  WARNING: could not read universe: {exc}")
        return []

    tier_a = uni.get("tier_a", {})
    for group in ("equities", "etfs"):
        for tickers in (tier_a.get(group) or {}).values():
            out.update(t.upper() for t in tickers if isinstance(t, str))

    # Legacy flat layout
    for group in ("equities", "etfs"):
        for tickers in (uni.get(group) or {}).values():
            out.update(t.upper() for t in tickers if isinstance(t, str))

    for entry in (uni.get("tier_b", {}).get("tickers") or []):
        if isinstance(entry, dict) and entry.get("symbol"):
            out.add(entry["symbol"].upper())
        elif isinstance(entry, str):
            out.add(entry.upper())

    tier_c_file = os.path.join(SCRIPT_DIR, "kairos_tier_c.json")
    if os.path.exists(tier_c_file):
        try:
            with open(tier_c_file) as f:
                for e in json.load(f) or []:
                    if isinstance(e, dict) and e.get("ticker"):
                        out.add(e["ticker"].upper())
        except (json.JSONDecodeError, IOError):
            pass

    return sorted(out)


# ── Public API ────────────────────────────────────────────────────────

_MEM: dict[str, dict] | None = None


def _load_all() -> dict[str, dict]:
    global _MEM
    if _MEM is not None:
        return _MEM
    _MEM = {}
    try:
        conn = _connect(readonly=True)
    except sqlite3.Error:
        return _MEM
    try:
        for t, s, i, a, src in conn.execute(
            "SELECT ticker, sector, industry, asset_type, source FROM security_master"
        ):
            _MEM[t] = {"sector": s, "industry": i, "asset_type": a, "source": src}
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return _MEM


def invalidate_cache() -> None:
    """Drop the in-process cache (call after a backfill in a long-lived proc)."""
    global _MEM
    _MEM = None


def get_sector(ticker: str) -> str:
    """Real sector for a ticker. `Unclassified` if not resolved."""
    return _load_all().get(ticker.upper(), {}).get("sector") or UNRESOLVED


def get_industry(ticker: str) -> str | None:
    return _load_all().get(ticker.upper(), {}).get("industry")


def get_asset_type(ticker: str) -> str | None:
    return _load_all().get(ticker.upper(), {}).get("asset_type")


def is_fund(ticker: str) -> bool:
    """True for ETFs / mutual funds — excluded from sector concentration."""
    return (_load_all().get(ticker.upper(), {}).get("asset_type") or "") in FUND_QUOTE_TYPES


def sector_exposure(positions: list, include_funds: bool = False) -> dict[str, float]:
    """{sector: market_value} for a list of position dicts.

    Funds are excluded by default: they carry no single sector, and folding
    them into one would misstate concentration in whichever sector they landed.
    """
    out: dict[str, float] = {}
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        if not sym or "(SIM)" in sym:
            continue
        if p.get("assetClass", "equity") == "crypto":
            continue
        mkt = float(p.get("market_value", 0) or 0)
        if mkt <= 0:
            continue
        if not include_funds and is_fund(sym):
            continue
        sec = get_sector(sym)
        out[sec] = out.get(sec, 0.0) + mkt
    return out


# ── Backfill ──────────────────────────────────────────────────────────

def backfill(tickers: list[str] | None = None, force: bool = False,
             limit: int | None = None, verbose: bool = True) -> dict:
    """Resolve and persist. Skips rows still inside TTL unless `force`."""
    ensure_schema()
    overrides = _load_overrides()
    targets = [t.upper() for t in (tickers if tickers is not None else universe_tickers())]

    conn = _connect()
    try:
        fresh = set() if force else _fresh_tickers(conn)
        todo = [t for t in targets if t not in fresh]
        if limit:
            todo = todo[:limit]

        if verbose:
            print(f"  {len(targets)} tracked · {len(targets) - len(todo)} fresh "
                  f"(TTL {TTL_DAYS}d) · {len(todo)} to resolve")

        stats: dict[str, int] = {}
        for n, ticker in enumerate(todo, 1):
            rec = resolve(ticker, overrides)
            _upsert(conn, ticker, rec)
            stats[rec["source"]] = stats.get(rec["source"], 0) + 1

            if verbose and (n % 25 == 0 or n == len(todo)):
                print(f"    {n}/{len(todo)} …")
            if n % 50 == 0:
                conn.commit()
            time.sleep(YF_THROTTLE_SEC)

        conn.commit()
    finally:
        conn.close()

    invalidate_cache()
    if verbose and todo:
        print("  by source: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return stats


# ── Reporting ─────────────────────────────────────────────────────────

def status() -> None:
    ensure_schema()
    conn = _connect(readonly=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM security_master").fetchone()[0]
        print(f"security_master: {total} rows")
        if not total:
            print("  (empty — run `backfill`)")
            return

        print("\n  by source:")
        for src, n in conn.execute(
            "SELECT source, COUNT(*) FROM security_master GROUP BY source ORDER BY 2 DESC"
        ):
            print(f"    {src:12} {n:5}")

        print("\n  by sector:")
        for sec, n in conn.execute(
            "SELECT sector, COUNT(*) FROM security_master GROUP BY sector ORDER BY 2 DESC"
        ):
            print(f"    {sec:26} {n:5}")

        stale = conn.execute(
            "SELECT COUNT(*) FROM security_master WHERE resolved_at < ?",
            (_stale_before(),),
        ).fetchone()[0]
        unres = conn.execute(
            "SELECT COUNT(*) FROM security_master WHERE source = 'unresolved'"
        ).fetchone()[0]
        print(f"\n  stale (> {TTL_DAYS}d): {stale}    unresolved: {unres}")

        if unres:
            rows = conn.execute(
                "SELECT ticker FROM security_master WHERE source = 'unresolved' "
                "ORDER BY ticker LIMIT 40"
            ).fetchall()
            print("    " + " ".join(r[0] for r in rows))
    finally:
        conn.close()


def stale_symbols() -> None:
    """Report universe tickers that no upstream source recognises.

    A ticker that neither yfinance nor SEC will resolve is almost always dead:
    renamed (BK -> BNY), or delisted after an acquisition or take-private. Both
    cases mean the screener is spending every cycle on a symbol that cannot
    trade, and any held position would fail to mark.

    Read-only. Removing or remapping a ticker changes the screening universe,
    which is a trading-behaviour change and belongs in its own reviewed step.
    """
    ensure_schema()
    conn = _connect(readonly=True)
    try:
        rows = conn.execute(
            "SELECT ticker FROM security_master WHERE source = 'unresolved' ORDER BY ticker"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("  no unresolved tickers — universe is clean")
        return

    cik_by_name = {}
    try:
        import requests
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        for r in resp.json().values():
            cik_by_name.setdefault(str(r.get("cik_str")).zfill(10), []).append(r.get("ticker"))
    except Exception as exc:
        print(f"  WARNING: SEC registrant list unavailable, "
              f"cannot suggest replacements: {exc}")

    print(f"  {len(rows)} unresolved ticker(s) in the universe:\n")
    print(f"    {'ticker':10} {'status':34} suggestion")
    for (ticker,) in rows:
        vend = _vendor_symbol(ticker)
        cik = _cik_map().get(vend)
        if cik:
            alts = [t for t in cik_by_name.get(cik, []) if t and t != vend and "-P" not in t]
            if vend != ticker.upper():
                status, sugg = "symbol format differs by venue", vend
            else:
                status, sugg = "registrant found under another ticker", ", ".join(alts[:3]) or "—"
        else:
            # Could be a rename (BK -> BNY) or a delisting. Distinguishing the
            # two needs the company name, which the universe does not carry for
            # Tier A. Guessing a replacement ticker from a fuzzy name match is
            # exactly the wrong risk to take here — flag it and let a human look.
            status, sugg = "no longer trades under this symbol", "renamed or delisted — verify"
        print(f"    {ticker:10} {status:34} {sugg}")

    print("\n  Read-only. Editing kairos_universe.json changes what gets screened;")
    print("  do that as its own reviewed change.")


def compare_live_book() -> None:
    """Side-by-side: universe bucket vs real sector, against open positions.

    Read-only. This is the evidence for whether the concentration guardrail
    is measuring what it thinks it is measuring.
    """
    try:
        from kairos_dashboard import fetch_ibkr_data
        from kairos_confluence import lookup_sector
    except ImportError as exc:
        print(f"  cannot load live book: {exc}")
        return

    ibkr = fetch_ibkr_data()
    positions = [
        p for p in (ibkr.get("positions") or [])
        if p.get("assetClass", "equity") != "crypto"
        and float(p.get("market_value", 0) or 0) > 0
        and "(SIM)" not in (p.get("symbol") or "")
    ]
    if not positions:
        print("  no open equity positions")
        return

    nlv = sum(float(p["market_value"]) for p in positions)

    bucket: dict[str, float] = {}
    for p in positions:
        b = lookup_sector(p["symbol"])
        bucket[b] = bucket.get(b, 0.0) + float(p["market_value"])

    real = sector_exposure(positions)
    funds = sum(
        float(p["market_value"]) for p in positions if is_fund(p["symbol"])
    )

    try:
        with open(os.path.join(SCRIPT_DIR, "kairos_config.json")) as f:
            limit_pct = (
                json.load(f).get("confluence", {}).get("guardrails", {})
                .get("max_sector_concentration_pct", 0.25) * 100
            )
    except (json.JSONDecodeError, IOError):
        limit_pct = 25.0

    print(f"\n  Open equity book: ${nlv:,.0f} across {len(positions)} positions")
    print(f"  Concentration limit: {limit_pct:.1f}% of NLV\n")

    print("  WHAT THE GUARDRAIL SEES (universe bucket)")
    for k, v in sorted(bucket.items(), key=lambda kv: -kv[1]):
        flag = "  <-- AT/OVER LIMIT" if v / nlv * 100 >= limit_pct else ""
        print(f"    {k:26} ${v:>12,.0f}  {v/nlv*100:5.2f}%{flag}")

    print("\n  ACTUAL SECTOR EXPOSURE (security_master)")
    for k, v in sorted(real.items(), key=lambda kv: -kv[1]):
        flag = "  <-- AT/OVER LIMIT" if v / nlv * 100 >= limit_pct else ""
        print(f"    {k:26} ${v:>12,.0f}  {v/nlv*100:5.2f}%{flag}")
    if funds:
        print(f"    {'(funds, excluded)':26} ${funds:>12,.0f}  {funds/nlv*100:5.2f}%")

    print("\n  UNDERSTATEMENT — real sector vs the largest single bucket feeding it")
    rows = []
    for sec, val in real.items():
        if sec == UNRESOLVED:
            continue
        contrib: dict[str, float] = {}
        for p in positions:
            if get_sector(p["symbol"]) != sec or is_fund(p["symbol"]):
                continue
            b = lookup_sector(p["symbol"])
            contrib[b] = contrib.get(b, 0.0) + float(p["market_value"])
        if not contrib:
            continue
        biggest = max(contrib.values())
        rows.append((val / nlv * 100, biggest / nlv * 100, sec, len(contrib)))

    rows.sort(reverse=True)
    print(f"    {'sector':26} {'real %':>7} {'seen %':>7} {'gap':>7}  buckets")
    for real_pct, seen_pct, sec, nbuckets in rows:
        gap = real_pct - seen_pct
        mark = "  <-- blind spot" if gap >= 5.0 else ""
        print(f"    {sec:26} {real_pct:6.2f}% {seen_pct:6.2f}% {gap:6.2f}pp  "
              f"{nbuckets}{mark}")


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Kairos security master")
    sub = ap.add_subparsers(dest="cmd")

    p_bf = sub.add_parser("backfill", help="resolve and persist the universe")
    p_bf.add_argument("--limit", type=int, help="stop after N tickers")
    p_bf.add_argument("--force", action="store_true", help="ignore TTL, re-resolve all")

    p_rs = sub.add_parser("resolve", help="resolve specific tickers (persists)")
    p_rs.add_argument("tickers", nargs="+")

    sub.add_parser("status", help="coverage summary")
    sub.add_parser("stale", help="universe tickers no upstream source recognises")
    sub.add_parser("compare", help="bucket vs sector against the live book")

    args = ap.parse_args()

    if args.cmd == "backfill":
        backfill(force=args.force, limit=args.limit)
        print()
        status()
    elif args.cmd == "resolve":
        backfill(tickers=args.tickers, force=True)
        for t in args.tickers:
            t = t.upper()
            print(f"  {t:8} {get_sector(t):26} {get_industry(t) or '—'}")
    elif args.cmd == "status":
        status()
    elif args.cmd == "stale":
        stale_symbols()
    elif args.cmd == "compare":
        compare_live_book()
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
