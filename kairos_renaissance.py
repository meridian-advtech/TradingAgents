"""
Kairos Renaissance Capital Client — IPO Data Enrichment

A thin Python wrapper around Renaissance Capital's IPO API. Used by the
IPO intake / signals pipeline to enrich freshly-priced and pipeline
filings with offer-price ranges, calendar windows, and sector context.

Confirmed endpoints (base = https://api.renaissancecapital.com):

    GET /free/CompanyIpoDate?TickerSymbol={ticker}
    GET /basic-calendar/Calendar
    GET /basic-priced/PricedIPOs?StartDate={iso}&EndDate={iso}
    GET /basic-filings/Filings?StartDate={iso}&EndDate={iso}

Auth: Ocp-Apim-Subscription-Key header sourced from
kairos_config.json["renaissance_capital"]["api_key"].

──────────────────────────────────────────────────────────────────────
COMPLIANCE NOTE — Renaissance Capital Terms of Service §8(q)
──────────────────────────────────────────────────────────────────────
Per Renaissance Capital's ToS §8(q), data retrieved from this API
MUST NOT be fed into AI / LLM systems — neither for model training
nor as in-prompt context at inference time.

In Kairos that means:

  • DO use these helpers from numeric scoring / alerting paths
    (sector_momentum, EOD reports, Slack #kairos-alerts surfaces,
    Tier C enrichment dicts that are not pasted into prompts).
  • DO NOT pass raw Renaissance payloads into the Ollama / Claude
    reasoning prompts (kairos_router.py, kairos_reason.py,
    kairos_prompt.txt). LLM-bound flows must use EDGAR + Finnhub
    data only — see kairos_ipo_intake.fetch_all_s1_filings for
    the LLM-safe substitute.

Violations would breach the ToS and risk API-key revocation.
──────────────────────────────────────────────────────────────────────
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")

BASE_URL = "https://api.renaissancecapital.com"
TIMEOUT = 15
USER_AGENT = "Kairos Trading System jason@meridiangroup.llc"


# ── Sector → comparable-tickers map for momentum reads ───────────────
#
# Curated manually using highly-liquid GICS-sector benchmarks. Keep
# each list short (≤5) — the RSI fetcher hits Finnhub once per ticker
# and the free tier is 60 calls/min.
SECTOR_COMP_TICKERS: dict[str, list[str]] = {
    "Technology":             ["AAPL", "MSFT", "NVDA", "GOOG", "META"],
    "Financials":             ["JPM",  "BAC",  "GS",   "MS",   "WFC"],
    "Healthcare":             ["UNH",  "JNJ",  "LLY",  "PFE",  "ABBV"],
    "Energy":                 ["XOM",  "CVX",  "COP",  "OXY",  "EOG"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD",   "MCD",  "NKE"],
    "Industrials":            ["CAT",  "GE",   "BA",   "RTX",  "UPS"],
    "Communication Services": ["GOOG", "META", "NFLX", "DIS",  "TMUS"],
    "Materials":              ["LIN",  "APD",  "FCX",  "NEM",  "DOW"],
    "Real Estate":            ["PLD",  "AMT",  "EQIX", "SPG",  "WELL"],
    "Utilities":              ["NEE",  "DUK",  "SO",   "AEP",  "EXC"],
    "Consumer Staples":       ["WMT",  "PG",   "KO",   "PEP",  "COST"],
}

SECTOR_HOT_RSI = 60.0   # avg ≥ 60 → "hot"
SECTOR_COOL_RSI = 40.0  # avg ≤ 40 → "cool"


# ─────────────────────────────────────────────────────────────────────
# Config / auth
# ─────────────────────────────────────────────────────────────────────

def _load_api_key() -> Optional[str]:
    """Read renaissance_capital.api_key from kairos_config.json.

    Falls back to RENAISSANCE_CAPITAL_API_KEY when the config is
    unreadable. Returns None when neither is present.
    """
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        key = cfg.get("renaissance_capital", {}).get("api_key", "")
        if key:
            return str(key).strip()
    except (IOError, json.JSONDecodeError):
        pass
    env_key = (os.environ.get("RENAISSANCE_CAPITAL_API_KEY") or "").strip()
    return env_key or None


def _load_finnhub_key() -> Optional[str]:
    """Mirror kairos_ipo_intake's Finnhub config/env lookup."""
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        key = cfg.get("finnhub", {}).get("api_key", "")
        if key:
            return str(key).strip()
    except (IOError, json.JSONDecodeError):
        pass
    return (os.environ.get("FINNHUB_API_KEY") or "").strip() or None


# ─────────────────────────────────────────────────────────────────────
# HTTP wrapper
# ─────────────────────────────────────────────────────────────────────

def _get(path: str, params: Optional[dict] = None):
    """Issue a GET against the Renaissance Capital API.

    Returns parsed JSON (dict or list) on success, or None on transport
    error, non-2xx response, or missing API key.
    """
    api_key = _load_api_key()
    if not api_key:
        print("  WARNING: Renaissance Capital API key not configured")
        return None

    url = f"{BASE_URL}{path}"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    try:
        resp = requests.get(
            url, params=params or {}, headers=headers, timeout=TIMEOUT
        )
    except Exception as exc:
        print(f"  WARNING: Renaissance Capital GET {path} failed: {exc}")
        return None

    if resp.status_code == 401:
        print(f"  WARNING: Renaissance Capital 401 on {path} — check API key")
        return None
    if resp.status_code == 429:
        print(f"  WARNING: Renaissance Capital 429 on {path} — backoff")
        return None
    if not resp.ok:
        print(f"  WARNING: Renaissance Capital {resp.status_code} on {path}")
        return None

    try:
        return resp.json()
    except ValueError:
        return None


def _unwrap_list(data, candidate_keys: tuple) -> list:
    """Azure APIM frequently wraps arrays as {"value": [...]} — coerce."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in candidate_keys:
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


# ─────────────────────────────────────────────────────────────────────
# Confirmed endpoints
# ─────────────────────────────────────────────────────────────────────

def get_company_ipo_date(ticker: str) -> Optional[dict]:
    """GET /free/CompanyIpoDate?TickerSymbol={ticker}."""
    if not ticker:
        return None
    data = _get(
        "/free/CompanyIpoDate",
        params={"TickerSymbol": ticker.strip().upper()},
    )
    return data if isinstance(data, dict) else None


def get_calendar() -> list[dict]:
    """GET /basic-calendar/Calendar — upcoming IPO pricings."""
    data = _get("/basic-calendar/Calendar")
    return _unwrap_list(data, ("Calendar", "calendar", "value", "items"))


def get_priced_ipos(start_date: str, end_date: str) -> list[dict]:
    """GET /basic-priced/PricedIPOs in [start, end] (ISO YYYY-MM-DD)."""
    if not (start_date and end_date):
        return []
    data = _get(
        "/basic-priced/PricedIPOs",
        params={"StartDate": start_date, "EndDate": end_date},
    )
    return _unwrap_list(data, ("PricedIPOs", "value", "items"))


def get_filings(start_date: str, end_date: str) -> list[dict]:
    """GET /basic-filings/Filings in [start, end] (ISO YYYY-MM-DD)."""
    if not (start_date and end_date):
        return []
    data = _get(
        "/basic-filings/Filings",
        params={"StartDate": start_date, "EndDate": end_date},
    )
    return _unwrap_list(data, ("Filings", "value", "items"))


# ─────────────────────────────────────────────────────────────────────
# Sector momentum — reuse kairos_signals RSI fetcher
# ─────────────────────────────────────────────────────────────────────

def sector_momentum(sector: str) -> dict:
    """Score a GICS sector by avg 14-day RSI across its comp tickers.

    Reuses `_fetch_rsi_for_ticker` from kairos_signals so the Finnhub
    candle fetch + RSI computation stays in one place (the HOT-RSI
    signal path).

    Returns:
        {
          "sector":  "Technology",
          "comps":   ["AAPL", "MSFT", ...],
          "rsi":     {"AAPL": 58.2, "MSFT": 61.5, ...},
          "avg_rsi": 60.1 | None,
          "regime":  "hot" | "neutral" | "cool" | "unknown",
        }
    """
    sector_key = (sector or "").strip()
    comps = SECTOR_COMP_TICKERS.get(sector_key, [])
    out: dict = {
        "sector":  sector_key or "Unknown",
        "comps":   comps,
        "rsi":     {},
        "avg_rsi": None,
        "regime":  "unknown",
    }
    if not comps:
        return out

    finnhub_key = _load_finnhub_key()
    if not finnhub_key:
        print("  WARNING: Finnhub key missing — cannot compute sector momentum")
        return out

    try:
        from kairos_signals import _fetch_rsi_for_ticker
    except ImportError as exc:
        print(f"  WARNING: cannot import RSI helper: {exc}")
        return out

    rsi_values: list[float] = []
    for ticker in comps:
        try:
            _, rsi = _fetch_rsi_for_ticker(ticker, finnhub_key)
        except Exception:
            rsi = None
        if rsi is None:
            continue
        out["rsi"][ticker] = rsi
        rsi_values.append(rsi)

    if not rsi_values:
        return out

    avg = sum(rsi_values) / len(rsi_values)
    out["avg_rsi"] = round(avg, 2)

    if avg >= SECTOR_HOT_RSI:
        out["regime"] = "hot"
    elif avg <= SECTOR_COOL_RSI:
        out["regime"] = "cool"
    else:
        out["regime"] = "neutral"
    return out


# ─────────────────────────────────────────────────────────────────────
# S-1 financials — EDGAR primary, Renaissance Filings as supplement
# ─────────────────────────────────────────────────────────────────────

def get_s1_financials(
    *,
    ticker: Optional[str] = None,
    company: Optional[str] = None,
    lookback_days: int = 60,
) -> dict:
    """Fetch S-1 financials, EDGAR first.

    Returns:
        {
          "source":     "edgar" | "renaissance" | "none",
          "ticker":     "...",
          "company":    "...",
          "edgar":      {accession, filing_date, cik, edgar_url, form} | None,
          "renaissance": {offer_price_low, offer_price_high,
                          shares_offered, listed_date, ...} | None,
        }

    Per ToS §8(q), the `renaissance` block is for numeric rendering /
    alerting only — do not pipe this dict into LLM prompts.
    """
    result: dict = {
        "source":     "none",
        "ticker":     (ticker or "").upper().strip() or None,
        "company":    (company or "").strip() or None,
        "edgar":      None,
        "renaissance": None,
    }

    edgar = _edgar_lookup_recent_s1(
        ticker=result["ticker"],
        company=result["company"],
        lookback_days=lookback_days,
    )
    if edgar:
        result["edgar"] = edgar
        result["source"] = "edgar"
        result["company"] = result["company"] or edgar.get("company")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc)
             - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    renaissance = _renaissance_match_filing(
        ticker=result["ticker"],
        company=result["company"],
        start_date=start,
        end_date=today,
    )
    if renaissance:
        result["renaissance"] = renaissance
        if result["source"] == "none":
            result["source"] = "renaissance"

    return result


def _edgar_lookup_recent_s1(
    *,
    ticker: Optional[str],
    company: Optional[str],
    lookback_days: int,
) -> Optional[dict]:
    """Best-effort EDGAR EFTS lookup for the most recent S-1.

    Returns {company, accession, filing_date, cik, edgar_url, form} or
    None when EDGAR is unreachable / no match.
    """
    try:
        from kairos_ipo_intake import (
            SEC_HEADERS, FETCH_TIMEOUT, _strip_cik_suffix,
        )
    except Exception:
        return None

    needle = (company or ticker or "").strip()
    if not needle:
        return None

    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc)
             - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?q=%22{requests.utils.quote(needle)}%22"
        f"&dateRange=custom&startdt={start}&enddt={end}&forms=S-1"
    )

    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  WARNING: EDGAR S-1 lookup for {needle!r} failed: {exc}")
        return None

    hits = (data.get("hits", {}) or {}).get("hits", []) or []
    if not hits:
        return None

    hit = hits[0]
    src = hit.get("_source", {}) or {}
    display_names = src.get("display_names") or []
    company_name = (_strip_cik_suffix(display_names[0])
                    if display_names else needle)
    ciks = src.get("ciks") or [""]
    cik = ciks[0] if ciks else ""
    accession = (hit.get("_id") or "").split(":", 1)[0]
    adsh = accession.replace("-", "")
    filing_date = src.get("file_date") or src.get("filed") or ""
    form = src.get("form") or "S-1"
    edgar_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={cik}&type=S-1&dateb=&owner=include&count=10"
    ) if cik else (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/"
    )
    return {
        "company":     company_name,
        "accession":   accession,
        "filing_date": filing_date,
        "cik":         cik,
        "form":        form,
        "edgar_url":   edgar_url,
    }


def _renaissance_match_filing(
    *,
    ticker: Optional[str],
    company: Optional[str],
    start_date: str,
    end_date: str,
) -> Optional[dict]:
    """Scan Renaissance Filings in the window for a ticker/company match.

    Returns the normalized filing record or None. Company-name match
    falls back to substring because Renaissance Filings can pre-date
    ticker assignment.
    """
    filings = get_filings(start_date, end_date)
    if not filings:
        return None

    t_low = (ticker or "").strip().lower()
    c_low = (company or "").strip().lower()
    chosen: Optional[dict] = None
    for entry in filings:
        if not isinstance(entry, dict):
            continue
        symbol = str(
            entry.get("Symbol")
            or entry.get("Ticker")
            or entry.get("tickerSymbol")
            or ""
        ).lower()
        name = str(
            entry.get("Issuer")
            or entry.get("Company")
            or entry.get("CompanyName")
            or entry.get("companyName")
            or ""
        ).lower()
        if t_low and t_low == symbol:
            chosen = entry
            break
        if c_low and c_low in name:
            chosen = entry
            break

    if not chosen:
        return None

    return {
        "issuer":       chosen.get("Issuer") or chosen.get("CompanyName"),
        "symbol":       chosen.get("Symbol") or chosen.get("Ticker"),
        "filing_date":  chosen.get("FilingDate") or chosen.get("Date"),
        "offer_price_low":  _to_float(
            chosen.get("OfferPriceLow") or chosen.get("PriceLow")
        ),
        "offer_price_high": _to_float(
            chosen.get("OfferPriceHigh") or chosen.get("PriceHigh")
        ),
        "shares_offered": _to_int(
            chosen.get("SharesOffered") or chosen.get("Shares")
        ),
        "expected_proceeds_usd": _to_float(
            chosen.get("ExpectedProceeds") or chosen.get("Proceeds")
        ),
        "listed_date":      chosen.get("ListedDate") or chosen.get("PricingDate"),
        "exchange":         chosen.get("Exchange"),
        "lead_underwriter": chosen.get("LeadUnderwriter"),
        "raw": chosen,
    }


def _normalize_date(value: str) -> str:
    """Coerce a date string to ISO YYYY-MM-DD; pass through if unparseable.

    Renaissance frequently returns M/D/YYYY (e.g. '12/9/2020'). We try
    that first, then a few common ISO variants.
    """
    if not value:
        return value
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ", "%m/%d/%y"):
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return value


def _to_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Public enrichment helper for kairos_ipo_intake
# ─────────────────────────────────────────────────────────────────────

def enrich_live_ipo(
    ticker: str,
    *,
    company: Optional[str] = None,
    sector: Optional[str] = None,
    lookback_days: int = 90,
) -> dict:
    """Build a Renaissance-sourced enrichment record for a live IPO ticker.

    Designed for kairos_ipo_intake.add_to_tier_c: returns numeric /
    factual fields safe to attach to the Tier C entry. The caller
    decides whether to persist.

    Per ToS §8(q), the returned dict is for storage and human/alert
    surfaces — it must not be passed into LLM prompts.

    Returns:
        {
          "ticker":    "ABCD",
          "ipo_date":  "2026-05-14" | None,
          "offer_price_low":  ...   | None,
          "offer_price_high": ...   | None,
          "expected_proceeds_usd": ... | None,
          "exchange":         ...   | None,
          "lead_underwriter": ...   | None,
          "sector_regime":    {sector_momentum() output} | None,
          "source":           "renaissance",
        }
    """
    t = (ticker or "").strip().upper()
    out: dict = {
        "ticker":    t or None,
        "ipo_date":  None,
        "offer_price_low":  None,
        "offer_price_high": None,
        "expected_proceeds_usd": None,
        "exchange":         None,
        "lead_underwriter": None,
        "sector_regime":    None,
        "source":           "renaissance",
    }
    if not t:
        return out

    ipo_info = get_company_ipo_date(t)
    if isinstance(ipo_info, dict):
        for k in ("offerDate", "OfferDate",
                  "ipoDate", "IpoDate", "IPODate", "PricingDate"):
            v = ipo_info.get(k)
            if v:
                out["ipo_date"] = _normalize_date(str(v))
                break

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc)
             - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    filings = _renaissance_match_filing(
        ticker=t, company=company, start_date=start, end_date=today,
    )
    if filings:
        out["offer_price_low"]  = filings.get("offer_price_low")
        out["offer_price_high"] = filings.get("offer_price_high")
        out["expected_proceeds_usd"] = filings.get("expected_proceeds_usd")
        out["exchange"]         = filings.get("exchange")
        out["lead_underwriter"] = filings.get("lead_underwriter")
        out["ipo_date"]         = out["ipo_date"] or filings.get("listed_date")

    if sector:
        try:
            out["sector_regime"] = sector_momentum(sector)
        except Exception as exc:
            print(f"  WARNING: sector_momentum({sector}) failed: {exc}")

    return out


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _print_json(label: str, payload) -> None:
    print(f"\n── {label} ──")
    if payload in (None, [], {}):
        print("  (no data)")
        return
    print(json.dumps(payload, indent=2, default=str))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Kairos — Renaissance Capital IPO data client",
    )
    parser.add_argument("--ticker", metavar="SYMBOL",
                        help="CompanyIpoDate lookup for a given ticker")
    parser.add_argument("--calendar", action="store_true",
                        help="Fetch the upcoming IPO calendar")
    parser.add_argument("--priced", nargs=2, metavar=("START", "END"),
                        help="PricedIPOs in [START, END] (ISO dates)")
    parser.add_argument("--filings", nargs=2, metavar=("START", "END"),
                        help="Filings in [START, END] (ISO dates)")
    parser.add_argument("--sector", metavar="NAME",
                        help="Sector momentum for a GICS sector name")
    parser.add_argument("--s1", metavar="TICKER",
                        help="S-1 financials lookup (EDGAR primary)")
    parser.add_argument("--s1-company", metavar="NAME",
                        help="Company name for --s1 (or standalone)")
    parser.add_argument("--enrich", metavar="TICKER",
                        help="Build enrich_live_ipo() record for a ticker")
    parser.add_argument("--enrich-sector", metavar="SECTOR",
                        help="Sector to include with --enrich")
    args = parser.parse_args()

    did_any = False

    if args.ticker:
        _print_json(
            f"CompanyIpoDate {args.ticker.upper()}",
            get_company_ipo_date(args.ticker),
        )
        did_any = True

    if args.calendar:
        _print_json("Calendar", get_calendar())
        did_any = True

    if args.priced:
        start, end = args.priced
        _print_json(f"PricedIPOs {start} → {end}",
                    get_priced_ipos(start, end))
        did_any = True

    if args.filings:
        start, end = args.filings
        _print_json(f"Filings {start} → {end}",
                    get_filings(start, end))
        did_any = True

    if args.sector:
        _print_json(f"SectorMomentum {args.sector}",
                    sector_momentum(args.sector))
        did_any = True

    if args.s1 or args.s1_company:
        _print_json(
            f"S-1 Financials {args.s1 or args.s1_company}",
            get_s1_financials(ticker=args.s1, company=args.s1_company),
        )
        did_any = True

    if args.enrich:
        _print_json(
            f"EnrichLiveIPO {args.enrich.upper()}",
            enrich_live_ipo(args.enrich, sector=args.enrich_sector),
        )
        did_any = True

    if not did_any:
        parser.print_help()


if __name__ == "__main__":
    main()
