"""
Kairos security-master selftest.

Runs the whole resolve -> persist -> query path against DISPOSABLE /tmp copies
of the DB, override file, and universe file. It never touches the live
kairos.db, kairos_sector_overrides.json, or kairos_universe.json, and never
hits the network — the yfinance and SEC resolvers are stubbed so the chain's
precedence and fallback behaviour are tested deterministically.

  1. schema creation is idempotent
  2. override beats yfinance                                   (precedence)
  3. yfinance beats SIC                                        (precedence)
  4. SIC fallback fires only when yfinance is silent           (precedence)
  5. everything silent -> Unclassified, never a silent bucket  (precedence)
  6. SIC range table maps representative codes correctly
  7. funds excluded from sector_exposure by default, included on request
  8. sector_exposure skips crypto / SIM / non-positive rows
  9. TTL: fresh rows skipped, stale rows re-resolved
 10. unresolved rows are ALWAYS retried, never cached for the TTL
 11. upsert updates in place rather than duplicating
 12. universe_tickers unions tier_a equities+etfs, tier_b (dict AND str), tier_c
 13. an unrecognised upstream sector label is recorded, not silently accepted

Exit code 0 iff every assertion passes.
"""

import json
import os
import sqlite3
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import kairos_security_master as S

_checks = {"pass": 0, "fail": 0}


def check(name, cond):
    if cond:
        _checks["pass"] += 1
        print(f"  PASS  {name}")
    else:
        _checks["fail"] += 1
        print(f"  FAIL  {name}")


# ── disposable fixtures ───────────────────────────────────────────────

TMP = tempfile.mkdtemp(prefix="kairos_secmaster_selftest_")

LIVE = {
    "DB_PATH":       S.DB_PATH,
    "OVERRIDE_FILE": S.OVERRIDE_FILE,
    "UNIVERSE_FILE": S.UNIVERSE_FILE,
    "SCRIPT_DIR":    S.SCRIPT_DIR,
    "YF_THROTTLE_SEC": S.YF_THROTTLE_SEC,
}

S.DB_PATH       = os.path.join(TMP, "test.db")
S.OVERRIDE_FILE = os.path.join(TMP, "overrides.json")
S.UNIVERSE_FILE = os.path.join(TMP, "universe.json")
S.SCRIPT_DIR    = TMP          # so tier_c lookup resolves inside TMP
S.YF_THROTTLE_SEC = 0.0        # no sleeping in tests

# Guard: if any of these still point at a live artifact, abort rather than
# risk writing to production state.
assert S.DB_PATH != LIVE["DB_PATH"]
assert not S.DB_PATH.startswith(LIVE["SCRIPT_DIR"] + os.sep) or TMP in S.DB_PATH

with open(S.OVERRIDE_FILE, "w") as f:
    json.dump({"overrides": {
        "SPY":  {"sector": "Fund / ETF", "industry": "S&P 500", "asset_type": "ETF"},
        "OVER": "Utilities",                       # bare-string form
    }}, f)

with open(S.UNIVERSE_FILE, "w") as f:
    json.dump({
        "tier_a": {
            "equities": {"mega_cap": ["AAA", "BBB"], "financials": ["CCC"]},
            "etfs":     {"broad_market": ["SPY"]},
        },
        "tier_b": {"tickers": [
            {"symbol": "DDD", "name": "D Corp", "sector": "Technology"},
            "EEE",                                  # bare-string tier_b entry
        ]},
    }, f)

with open(os.path.join(TMP, "kairos_tier_c.json"), "w") as f:
    json.dump([{"ticker": "FFF", "name": "F Corp"}], f)


# ── network stubs ─────────────────────────────────────────────────────

YF_TABLE: dict[str, dict] = {}
SIC_TABLE: dict[str, dict] = {}

# Held so test 13 can exercise the genuine implementation after the stubs.
LIVE_RESOLVER = S._resolve_yfinance


def fake_yf(ticker):
    return YF_TABLE.get(ticker)


def fake_sic(ticker):
    return SIC_TABLE.get(ticker)


S._resolve_yfinance = fake_yf
S._resolve_sec_sic = fake_sic


def reset_db():
    if os.path.exists(S.DB_PATH):
        os.remove(S.DB_PATH)
    S.ensure_schema()
    S.invalidate_cache()


def yf_rec(sector, industry="Ind", qt="EQUITY", detail=None):
    return {"sector": sector, "industry": industry, "asset_type": qt,
            "source": "yfinance", "detail": detail}


def sic_rec(sector, sic=6021):
    return {"sector": sector, "industry": "Banks", "asset_type": "EQUITY",
            "source": "sec_sic", "detail": f"SIC {sic}"}


def row(ticker):
    conn = sqlite3.connect(S.DB_PATH)
    try:
        return conn.execute(
            "SELECT ticker, sector, industry, asset_type, source, is_override "
            "FROM security_master WHERE ticker = ?", (ticker,)
        ).fetchone()
    finally:
        conn.close()


# ── 1. schema idempotent ──────────────────────────────────────────────

reset_db()
S.ensure_schema()
S.ensure_schema()
conn = sqlite3.connect(S.DB_PATH)
n_tbl = conn.execute(
    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='security_master'"
).fetchone()[0]
conn.close()
check("1  schema creation is idempotent", n_tbl == 1)


# ── 2-5. resolution precedence ────────────────────────────────────────

reset_db()
YF_TABLE.clear(); SIC_TABLE.clear()
YF_TABLE["SPY"] = yf_rec("Technology")           # should LOSE to the override
SIC_TABLE["SPY"] = sic_rec("Energy")
S.backfill(tickers=["SPY"], force=True, verbose=False)
r = row("SPY")
check("2  override beats yfinance and SIC",
      r[1] == "Fund / ETF" and r[4] == "override" and r[5] == 1)

reset_db()
YF_TABLE.clear(); SIC_TABLE.clear()
YF_TABLE["AAA"] = yf_rec("Healthcare")
SIC_TABLE["AAA"] = sic_rec("Energy")             # should LOSE to yfinance
S.backfill(tickers=["AAA"], force=True, verbose=False)
r = row("AAA")
check("3  yfinance beats SIC", r[1] == "Healthcare" and r[4] == "yfinance")

reset_db()
YF_TABLE.clear(); SIC_TABLE.clear()
SIC_TABLE["BBB"] = sic_rec("Financial Services") # yfinance silent
S.backfill(tickers=["BBB"], force=True, verbose=False)
r = row("BBB")
check("4  SIC fires only when yfinance is silent",
      r[1] == "Financial Services" and r[4] == "sec_sic")

reset_db()
YF_TABLE.clear(); SIC_TABLE.clear()
S.backfill(tickers=["ZZZ"], force=True, verbose=False)
r = row("ZZZ")
check("5  all silent -> Unclassified, not a silent bucket",
      r[1] == S.UNRESOLVED and r[4] == "unresolved")


# ── 6. SIC range table ────────────────────────────────────────────────

sic_cases = [
    (6021, "Financial Services"),   # national commercial banks
    (6798, "Real Estate"),          # REITs — must beat the 6700-6799 range
    (7372, "Technology"),           # prepackaged software
    (2834, "Healthcare"),           # pharmaceutical preparations
    (4911, "Utilities"),            # electric services
    (1311, "Energy"),               # crude petroleum & natural gas
    (5812, "Consumer Cyclical"),    # eating places
    (9999, None),                   # out of range -> no guess
]
ok = all(S._sector_from_sic(code) == want for code, want in sic_cases)
check("6  SIC range table maps representative codes (incl. REIT 6798)", ok)


# ── 7. funds excluded from exposure ───────────────────────────────────

reset_db()
YF_TABLE.clear(); SIC_TABLE.clear()
YF_TABLE["AAA"] = yf_rec("Technology")
S.backfill(tickers=["AAA", "SPY"], force=True, verbose=False)
S.invalidate_cache()

positions = [
    {"symbol": "AAA", "market_value": 100.0},
    {"symbol": "SPY", "market_value": 900.0},
]
exp_default = S.sector_exposure(positions)
exp_funds   = S.sector_exposure(positions, include_funds=True)
check("7a funds excluded from sector_exposure by default",
      exp_default == {"Technology": 100.0})
check("7b funds included when explicitly requested",
      exp_funds.get("Fund / ETF") == 900.0)
check("7c is_fund() true for an ETF, false for an equity",
      S.is_fund("SPY") and not S.is_fund("AAA"))


# ── 8. exposure filtering ─────────────────────────────────────────────

noisy = [
    {"symbol": "AAA", "market_value": 100.0},
    {"symbol": "AAA (SIM)", "market_value": 500.0},
    {"symbol": "BTC", "market_value": 500.0, "assetClass": "crypto"},
    {"symbol": "AAA", "market_value": 0.0},
    {"symbol": "AAA", "market_value": -50.0},
    {"symbol": "", "market_value": 10.0},
]
check("8  sector_exposure skips SIM / crypto / non-positive / blank",
      S.sector_exposure(noisy) == {"Technology": 100.0})


# ── 9-10. TTL behaviour ───────────────────────────────────────────────

reset_db()
YF_TABLE.clear(); SIC_TABLE.clear()
YF_TABLE["AAA"] = yf_rec("Technology")
S.backfill(tickers=["AAA"], force=True, verbose=False)

# Same run again without --force: should be treated as fresh and skipped.
YF_TABLE["AAA"] = yf_rec("Energy")               # would change it IF re-resolved
S.backfill(tickers=["AAA"], force=False, verbose=False)
check("9a fresh row inside TTL is skipped", row("AAA")[1] == "Technology")

# Age the row past the TTL -> must re-resolve.
conn = sqlite3.connect(S.DB_PATH)
conn.execute("UPDATE security_master SET resolved_at = '2000-01-01T00:00:00+00:00'")
conn.commit(); conn.close()
S.backfill(tickers=["AAA"], force=False, verbose=False)
check("9b stale row past TTL is re-resolved", row("AAA")[1] == "Energy")

# An unresolved row must be retried immediately, not cached for the TTL.
reset_db()
YF_TABLE.clear(); SIC_TABLE.clear()
S.backfill(tickers=["QQQQ"], force=False, verbose=False)
check("10a unresolved row written", row("QQQQ")[4] == "unresolved")
YF_TABLE["QQQQ"] = yf_rec("Industrials")
S.backfill(tickers=["QQQQ"], force=False, verbose=False)
check("10b unresolved row is ALWAYS retried, never TTL-cached",
      row("QQQQ")[1] == "Industrials" and row("QQQQ")[4] == "yfinance")


# ── 11. upsert, not duplicate ─────────────────────────────────────────

conn = sqlite3.connect(S.DB_PATH)
n = conn.execute("SELECT COUNT(*) FROM security_master WHERE ticker='QQQQ'").fetchone()[0]
conn.close()
check("11 upsert updates in place (no duplicate rows)", n == 1)


# ── 12. universe union ────────────────────────────────────────────────

got = set(S.universe_tickers())
check("12 universe unions tier_a equities+etfs, tier_b dict+str, tier_c",
      got == {"AAA", "BBB", "CCC", "SPY", "DDD", "EEE", "FFF"})


# ── 13. the REAL yfinance branch ──────────────────────────────────────
#
# Everything above stubs _resolve_yfinance wholesale, so the function's own
# logic — fund detection, empty-sector rejection, unrecognised-label recording
# — has not actually run. Inject a fake `yfinance` module so the genuine code
# path executes without touching the network.

import types

class _FakeTicker:
    def __init__(self, symbol):
        self._info = _FAKE_INFO.get(symbol, {})
    @property
    def info(self):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info

_FAKE_INFO: dict[str, object] = {}
_fake_yf = types.ModuleType("yfinance")
_fake_yf.Ticker = _FakeTicker
sys.modules["yfinance"] = _fake_yf

S._resolve_yfinance = LIVE_RESOLVER          # restore the real implementation

_FAKE_INFO["GOOD"] = {"quoteType": "EQUITY", "sector": "Technology",
                      "industry": "Software - Infrastructure"}
r = S._resolve_yfinance("GOOD")
check("13a real resolver returns a known sector with no detail flag",
      r["sector"] == "Technology" and r["industry"] == "Software - Infrastructure"
      and r["detail"] is None)

_FAKE_INFO["ODD"] = {"quoteType": "EQUITY", "sector": "Frobnication Services"}
r = S._resolve_yfinance("ODD")
check("13b unrecognised upstream label is accepted BUT recorded in detail",
      r["sector"] == "Frobnication Services"
      and r["detail"] and "unrecognised" in r["detail"])

_FAKE_INFO["FUND"] = {"quoteType": "ETF", "sector": "Technology",
                      "category": "Sector — Technology"}
r = S._resolve_yfinance("FUND")
check("13c a fund is classified as a fund even when upstream gives a sector",
      r["sector"] == S.FUND_SECTOR and r["asset_type"] == "ETF")

_FAKE_INFO["BLANK"] = {"quoteType": "EQUITY", "sector": "   "}
check("13d blank sector is rejected so the chain falls through to SIC",
      S._resolve_yfinance("BLANK") is None)

_FAKE_INFO["BOOM"] = RuntimeError("upstream exploded")
check("13e an upstream exception falls through instead of propagating",
      S._resolve_yfinance("BOOM") is None)

# Share-class symbols: Kairos/IBKR write BRK.B, vendors write BRK-B. Without
# translation every dual-class name falls through to Unclassified.
check("13g dot share-class separator is translated for vendors",
      S._vendor_symbol("BRK.B") == "BRK-B" and S._vendor_symbol("brk.b") == "BRK-B")
check("13h plain symbols pass through untouched",
      S._vendor_symbol(" msft ") == "MSFT")
_FAKE_INFO["BRK-B"] = {"quoteType": "EQUITY", "sector": "Financial Services",
                       "industry": "Insurance - Diversified"}
r = S._resolve_yfinance("BRK.B")
check("13i BRK.B resolves via the vendor form",
      r is not None and r["sector"] == "Financial Services")

# And the pinned label set itself.
check("13f KNOWN_SECTORS holds the 11 GICS-style labels",
      len(S.KNOWN_SECTORS) == 11
      and all(isinstance(s, str) and s for s in S.KNOWN_SECTORS))


# ── teardown ──────────────────────────────────────────────────────────

for k, v in LIVE.items():
    setattr(S, k, v)

import shutil
shutil.rmtree(TMP, ignore_errors=True)

print(f"\n  {_checks['pass']} passed, {_checks['fail']} failed")
sys.exit(0 if _checks["fail"] == 0 else 1)
