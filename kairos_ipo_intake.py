"""
Kairos IPO Intake — Autonomous S-1 Discovery Engine

Watchlist-FREE. Every new S-1 / S-1/A filing on EDGAR is evaluated on its
own merits — no human curation required. The daily pipeline (run_ipo_intake):

  1. fetch_all_s1_filings(30d) — enumerate ALL new S-1 / S-1/A filings via
     EDGAR full-text search (no sector pre-filter, deduped vs cache).
  2. extract_s1_content()      — BeautifulSoup over the primary document:
     revenue, revenue growth, lead underwriters, target raise, employees,
     business blurb (graceful degradation — failure still allows scoring).
  3. score_ipo_filing()        — 0–100 across revenue / growth / underwriter
     quality / raise size / sector → STRONG_BUY (≥70) / WATCH (≥45) / SKIP.
  4. process_ipo_discoveries() — STRONG_BUY → resolve ticker → Tier B
     (IPO_MOMENTUM) if live, else ipo_watchlist (PENDING) + #kairos-reports.
     WATCH → ipo_watchlist + #kairos-log. SKIP → log only.
  5. scan_watchlist_for_live_tickers() — curated PENDING entries → Tier C.

Scored filings are cached (never re-scored). Tier B entries carry the
`source: ipo_intake` marker so kairos_signals_ipo emits IPO_MOMENTUM.

Legacy helpers (fetch_edgar_s1_effectiveness, run_preipoipo_news_scan,
run_edgar_autodiscovery, get_watchlist_status) remain for the !ipo command
and Slack commander but are no longer in the autonomous critical path.

Run once per cycle from kairos_run.py after Phase 0.
"""

import difflib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

SCRIPT_DIR = "/Users/jelmore/Kairos"
sys.path.insert(0, SCRIPT_DIR)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "kairos_universe.json")
TIER_C_FILE = os.path.join(SCRIPT_DIR, "kairos_tier_c.json")
CACHE_FILE = os.path.join(SCRIPT_DIR, "kairos_ipo_cache.json")
DECISIONS_LOG = os.path.join(SCRIPT_DIR, "kairos_decisions.log")

W = 72
SEC_HEADERS = {
    "User-Agent": "Kairos Trading System jelmore@kairos.local",
    "Accept": "application/json, application/atom+xml, text/html",
}
FETCH_TIMEOUT = 20
IPO_TIER_C_TTL_DAYS = 30
NAME_MATCH_THRESHOLD = 0.62  # difflib ratio for S-1 fuzzy match

# Themes worth flagging on S-1 filings regardless of watchlist match.
# Each keyword is matched case-insensitively against the issuer name with
# WORD BOUNDARIES so short tokens (e.g. "AI", "chip") don't false-positive
# on substrings inside unrelated words like "Ambitious" or "MicroChip Co".
_S1_KEYWORDS_BROAD = [
    "artificial intelligence", "AI",
    "semiconductor", "chip",
    "fintech", "neobank",
    "SaaS", "cloud",
    "biotech",
]
# Per the spec, autodiscovery uses the same keyword set:
_S1_KEYWORDS_AUTODISCOVERY = list(_S1_KEYWORDS_BROAD)
PREIPO_NEWS_LOOKBACK_DAYS = 7
PREIPO_NEWS_VELOCITY_SPLIT_DAYS = 3
PREIPO_NEWS_ALERT_MIN_ARTICLES = 5
S1_LOOKBACK_DAYS = 30
S1_AUTODISCOVERY_LOOKBACK_DAYS = 7

# ── Autonomous S-1 discovery engine config ───────────────────────────
# These drive the watchlist-free scanner: every new S-1 / S-1/A filing on
# EDGAR is fetched, content-extracted, scored 0–100, and routed by
# recommendation (STRONG_BUY → Tier B/pipeline, WATCH → watchlist, SKIP →
# log). No human watchlist required.

EDGAR_FULL_SCAN_LOOKBACK_DAYS = 30   # trailing window for fetch_all_s1_filings
EDGAR_CONTENT_SLEEP_SEC = 1.0        # polite delay between content fetches
EDGAR_MAX_FILINGS_PER_BATCH = 20     # cap per run (top N by raise size)
IPO_STRONG_BUY_SCORE = 70
IPO_WATCH_SCORE = 45
IPO_TICKER_RESOLVE_ATTEMPTS = 5      # candidate tickers tried per company

# Known name→ticker overrides for companies whose symbol can't be inferred
# from the legal entity name. Checked first in resolve_ticker().
TICKER_OVERRIDES: dict[str, str] = {
    "Space Exploration Technologies": "SPCX",
    "SpaceX": "SPCX",
    "Klarna": "KLAR",
    "Cerebras": "CBRS",
    "Chime": "CHYM",
}

# High-signal IPO sectors for the sector-score component. Matched as
# case-insensitive substrings against the SIC description + business blurb.
_SECTOR_TIER1 = [  # 10 pts — tech / frontier
    "artificial intelligence", "ai ", "semiconductor", "chip", "space",
    "defense", "fintech", "biotech", "software", "saas", "cloud",
    "internet", "technology", "data", "cyber",
]
_SECTOR_TIER2 = [  # 6 pts — healthcare / energy / industrial tech
    "healthcare", "health care", "pharmaceutical", "medical", "energy",
    "solar", "battery", "industrial", "robotics", "manufactur",
]
_SECTOR_TIER3 = [  # 3 pts — consumer / retail / real estate
    "consumer", "retail", "restaurant", "real estate", "apparel",
    "hospitality", "food", "beverage",
]

# Bulge-bracket underwriters, tiered for the underwriter-score component.
_UW_TOP = ["Goldman Sachs", "Morgan Stanley"]                  # 20 pts
_UW_HIGH = ["J.P. Morgan", "JPMorgan", "JP Morgan",
            "BofA", "Bank of America", "Merrill Lynch"]        # 15 pts
_UW_BULGE = ["Citigroup", "Citi ", "Barclays", "Credit Suisse",
             "Deutsche Bank", "Wells Fargo", "UBS", "Jefferies",
             "Allen & Co", "Allen & Company", "Evercore"]      # 10 pts
_ALL_UNDERWRITERS = _UW_TOP + _UW_HIGH + _UW_BULGE
MARKET_OPEN_HOUR_ET = 9
MARKET_OPEN_MIN_FROM = 30
MARKET_OPEN_MIN_TO = 35


# ── Watchlist (config override + hardcoded fallback) ─────────────────

IPO_WATCHLIST: list[dict] = [
    {
        "name": "SpaceX / Starlink",
        "expected_ticker": "STRL",
        "alt_tickers": ["STRK", "SPACX", "SPC"],
        "sector": "Technology",
        "notes": "Starlink IPO expected 2026",
    },
    {
        "name": "Klarna",
        "expected_ticker": "KLAR",
        "alt_tickers": ["KLRN"],
        "sector": "Financials",
        "notes": "BNPL leader",
    },
    {
        "name": "Chime",
        "expected_ticker": "CHIM",
        "alt_tickers": [],
        "sector": "Financials",
        "notes": "Neobank",
    },
    {
        "name": "Cerebras",
        "expected_ticker": "CBRS",
        "alt_tickers": [],
        "sector": "Technology",
        "notes": "AI chip",
    },
]


def load_watchlist() -> list[dict]:
    """Load the watchlist, preferring kairos_config.json["ipo_watchlist"]."""
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        custom = cfg.get("ipo_watchlist")
        if isinstance(custom, list) and custom:
            # Normalize keys
            out = []
            for e in custom:
                if not isinstance(e, dict):
                    continue
                if not e.get("name") or not e.get("expected_ticker"):
                    continue
                out.append({
                    "name": e["name"],
                    "expected_ticker": e["expected_ticker"].upper(),
                    "alt_tickers": [t.upper() for t in (e.get("alt_tickers") or [])],
                    "sector": e.get("sector", "Unknown"),
                    "notes": e.get("notes", ""),
                })
            if out:
                return out
    except (IOError, json.JSONDecodeError):
        pass
    return list(IPO_WATCHLIST)


# ── Cache (avoid re-alerting on the same ticker) ─────────────────────

_CACHE_DEFAULTS: dict = {
    "detected": {},        # ticker -> {detected_at, name, added}
    "s1_seen": {},         # accession -> seen metadata (fuzzy effectiveness)
    "s1_filings_seen": {}, # accession -> seen metadata (autonomous full scan)
    "scored_filings": {},  # accession -> full score dict (never re-scored)
    "pre_ipo_news": {},    # company_name -> news snapshot
    "last_full_scan": "",  # ISO timestamp of last run_ipo_intake completion
}


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {k: ({} if isinstance(v, dict) else v)
                for k, v in _CACHE_DEFAULTS.items()}
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {k: ({} if isinstance(v, dict) else v)
                    for k, v in _CACHE_DEFAULTS.items()}
        for k, default in _CACHE_DEFAULTS.items():
            if isinstance(default, dict):
                data.setdefault(k, {})
                if not isinstance(data[k], dict):
                    data[k] = {}  # coerce list -> dict if type mismatch
            else:
                data.setdefault(k, default)
                if isinstance(default, list) and not isinstance(data[k], list):
                    data[k] = []  # coerce dict -> list if type mismatch
        return data
    except (IOError, json.JSONDecodeError):
        return {k: ({} if isinstance(v, dict) else v)
                for k, v in _CACHE_DEFAULTS.items()}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: IPO cache write failed: {exc}")


# ── yfinance live-ticker probe ───────────────────────────────────────

def check_ticker_live(ticker: str) -> bool:
    """Return True iff yfinance reports a positive regularMarketPrice."""
    if not ticker:
        return False
    try:
        import yfinance as yf
    except ImportError:
        print("  WARNING: yfinance not installed — cannot probe IPO tickers")
        return False

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return False

    # yfinance returns very sparse dicts for missing tickers
    price = info.get("regularMarketPrice")
    if price is None:
        price = info.get("currentPrice")
    if price is None:
        return False
    try:
        return float(price) > 0
    except (TypeError, ValueError):
        return False


def _lookup_ticker_name(ticker: str) -> Optional[str]:
    """Best-effort longName lookup. Used when announcing a new live ticker."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


def scan_watchlist_for_live_tickers() -> list[dict]:
    """Probe each watchlist entry for a live ticker.

    Returns list of {name, ticker, sector, is_new_ipo, source}. Tickers
    already announced via this module are skipped via the cache.
    """
    watchlist = load_watchlist()
    cache = _load_cache()
    seen = cache["detected"]

    found: list[dict] = []
    for entry in watchlist:
        candidates = [entry["expected_ticker"]] + list(entry.get("alt_tickers", []))
        for ticker in candidates:
            if not ticker:
                continue
            if ticker in seen:
                continue
            try:
                if check_ticker_live(ticker):
                    found.append({
                        "name": entry["name"],
                        "ticker": ticker,
                        "sector": entry.get("sector", "Unknown"),
                        "is_new_ipo": True,
                        "source": f"ipo_intake: watchlist match for {entry['name']}",
                    })
                    # Don't probe alts once we've matched the primary
                    break
            except Exception as exc:
                print(f"  WARNING: yfinance probe {ticker} failed: {exc}")
                continue
    return found


# ── EDGAR S-1 effectiveness monitor ──────────────────────────────────

def fetch_edgar_s1_effectiveness() -> list[dict]:
    """Poll EDGAR full-text search for recent S-1 / S-1/A filings.

    Fuzzy-matches issuer names against the IPO watchlist. Returns a list
    of {company, filing_type, filing_date, edgar_url, match_name, score}
    for any close matches. Empty list if EDGAR is unreachable.
    """
    try:
        import requests
    except ImportError:
        return []

    watchlist = load_watchlist()
    if not watchlist:
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    # Search for recent S-1 + S-1/A filings. EFTS treats forms=S-1 as
    # the literal form filter, which catches both base and amended.
    url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?q=%22S-1%22&dateRange=custom&startdt={start}&enddt={today}&forms=S-1"
    )

    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  WARNING: EDGAR S-1 fetch failed: {exc}")
        return []

    hits = data.get("hits", {}).get("hits", []) or []
    matches: list[dict] = []

    watch_names = [(e["name"], _name_tokens(e["name"]), e) for e in watchlist]

    for hit in hits:
        src = hit.get("_source", {}) or {}
        display_names = src.get("display_names") or []
        form = src.get("form") or "S-1"
        filing_date = src.get("file_date") or src.get("filed") or ""
        accession = (hit.get("_id") or "").split(":", 1)[0]
        adsh = accession.replace("-", "")
        ciks = src.get("ciks") or [""]
        cik = ciks[0] if ciks else ""
        edgar_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik}&type=S-1&dateb=&owner=include&count=10"
        ) if cik else ""

        for display in display_names:
            issuer = _strip_cik_suffix(display)
            best = _best_match(issuer, watch_names)
            if best is None:
                continue
            match_name, score, _watch_entry = best
            matches.append({
                "company": issuer,
                "filing_type": form,
                "filing_date": filing_date,
                "edgar_url": edgar_url or f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/",
                "match_name": match_name,
                "score": round(score, 3),
                "accession": accession,
            })

    # Dedup by (accession, match_name) — EFTS sometimes returns the same
    # filing under multiple display_names.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for m in matches:
        key = (m["accession"], m["match_name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    return deduped


def _strip_cik_suffix(display: str) -> str:
    """EDGAR display names are typically 'COMPANY NAME (0001234567) (CIK 0001234567)'.

    Strip the parenthetical CIK chunks.
    """
    # Drop trailing "(...)" once or twice
    out = display
    for _ in range(2):
        idx = out.rfind("(")
        if idx > 0 and out.endswith(")"):
            out = out[:idx].strip()
        else:
            break
    return out.strip()


def _name_tokens(name: str) -> set[str]:
    return {tok for tok in name.lower().replace("/", " ").split() if len(tok) > 2}


def _best_match(issuer: str, watch_names: list[tuple]) -> Optional[tuple]:
    """Return (match_name, score, watch_entry) above threshold, or None."""
    if not issuer:
        return None
    issuer_low = issuer.lower()
    issuer_toks = _name_tokens(issuer)
    best: Optional[tuple] = None
    for name, name_toks, entry in watch_names:
        # Token overlap is a strong positive signal
        overlap = len(issuer_toks & name_toks)
        ratio = difflib.SequenceMatcher(None, issuer_low, name.lower()).ratio()
        score = ratio + (0.15 * overlap)
        if score >= NAME_MATCH_THRESHOLD and (best is None or score > best[1]):
            best = (name, score, entry)
    return best


# ── Tier C auto-add ──────────────────────────────────────────────────

def _existing_universe_tickers() -> set[str]:
    """Tickers already in Tier A, Tier B, or Tier C — for dedup."""
    tickers: set[str] = set()
    try:
        with open(UNIVERSE_FILE) as f:
            universe = json.load(f)
        for _cat, syms in universe.get("tier_a", {}).get("equities", {}).items():
            tickers.update(syms)
        for _cat, syms in universe.get("tier_a", {}).get("etfs", {}).items():
            tickers.update(syms)
        for entry in universe.get("tier_b", {}).get("tickers", []):
            sym = entry["symbol"] if isinstance(entry, dict) else entry
            if sym:
                tickers.add(sym)
    except (IOError, json.JSONDecodeError):
        pass
    try:
        with open(TIER_C_FILE) as f:
            tier_c = json.load(f)
        for entry in tier_c:
            t = entry.get("ticker")
            if t:
                tickers.add(t)
    except (IOError, json.JSONDecodeError):
        pass
    return tickers


def _load_tier_c() -> list[dict]:
    if not os.path.exists(TIER_C_FILE):
        return []
    try:
        with open(TIER_C_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (IOError, json.JSONDecodeError):
        return []


def _save_tier_c(entries: list[dict]) -> None:
    with open(TIER_C_FILE, "w") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")


def add_to_tier_c(ticker: str, name: str, sector: str, reason: str) -> dict:
    """Add an IPO ticker to Tier C with 30-day TTL.

    Returns {"ok": True, "entry": {...}} or {"ok": False, "error": "..."}.
    No-op (ok=False) if the ticker is already present anywhere in the
    universe.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return {"ok": False, "error": "empty ticker"}

    if ticker in _existing_universe_tickers():
        return {"ok": False, "error": f"{ticker} already in universe"}

    today_dt = datetime.now(timezone.utc)
    today = today_dt.strftime("%Y-%m-%d")
    expires = (today_dt + timedelta(days=IPO_TIER_C_TTL_DAYS)).strftime("%Y-%m-%d")

    entry = {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "tier": "C",
        "added_date": today,
        "ttl_days": IPO_TIER_C_TTL_DAYS,
        "expires_date": expires,
        "reason": reason,
        "add_reason": reason,        # keep compat with existing kairos_tier_c.audit()
        "source": "ipo_intake",
        "expiry_alerted": False,
    }

    # Renaissance Capital enrichment — best-effort. Per ToS §8(q) the
    # returned record is for storage/alerting only, never LLM prompts.
    try:
        from kairos_renaissance import enrich_live_ipo
        enrichment = enrich_live_ipo(ticker, sector=sector)
        if enrichment:
            entry["renaissance"] = enrichment
    except Exception as exc:
        print(f"  WARNING: Renaissance enrichment for {ticker} failed: {exc}")

    entries = _load_tier_c()
    entries.append(entry)
    _save_tier_c(entries)

    print(f"  IPO INTAKE: added {ticker} ({name}) to Tier C "
          f"(expires {expires})")

    # Slack #kairos-alerts
    try:
        from kairos_alerts import alert_pipeline_event
        alert_pipeline_event(
            f":rocket: *IPO DETECTED:* `${ticker}` ({name}) added to Tier C universe.\n"
            f"Source: {reason}\n"
            f"Sector: {sector} | Expires: {expires}",
            channel="watchlist",
        )
    except Exception as exc:
        print(f"  WARNING: Slack IPO alert failed: {exc}")

    _append_decision_log({
        "type": "IPO_INTAKE",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "reason": reason,
        "expires_date": expires,
        "source": "ipo_intake",
    })

    return {"ok": True, "entry": entry}


def _append_decision_log(entry: dict) -> None:
    try:
        with open(DECISIONS_LOG, "a") as f:
            f.write("\n" + json.dumps(entry, indent=2))
            f.write("\n" + "=" * W + "\n")
    except IOError as exc:
        print(f"  WARNING: decision log append failed: {exc}")


# ── Broad EDGAR S-1 monitor (30-day window) ─────────────────────────

def _watchlist_name_index(watchlist: list[dict]) -> list[tuple]:
    """Pre-tokenize watchlist names for fuzzy matching."""
    return [(e["name"], _name_tokens(e["name"]), e) for e in watchlist]


_KW_REGEX_CACHE: dict[str, "re.Pattern"] = {}


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Return list of matched keywords (case-insensitive, word-boundary).

    Word boundaries prevent short tokens like "AI" or "chip" from
    matching substrings inside unrelated words (e.g. "Ambitious",
    "RAIDER", "MicroChip" in a company name). Multi-word phrases like
    "artificial intelligence" still match as a single boundary-aligned
    span.
    """
    import re
    if not text:
        return []
    hits = []
    for kw in keywords:
        pat = _KW_REGEX_CACHE.get(kw)
        if pat is None:
            # Escape, then anchor with \b on both ends.
            pat = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            _KW_REGEX_CACHE[kw] = pat
        if pat.search(text):
            hits.append(kw)
    return hits


def _edgar_get(url: str, params: Optional[dict] = None,
               as_json: bool = True):
    """GET against EDGAR with the polite User-Agent. Returns parsed JSON
    (or response text when as_json=False), or None on any failure."""
    try:
        import requests
    except ImportError:
        print("  WARNING: requests not installed — cannot fetch EDGAR")
        return None
    try:
        resp = requests.get(url, params=params, headers=SEC_HEADERS,
                            timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if as_json else resp.text
    except Exception as exc:
        print(f"  WARNING: EDGAR GET failed ({url}): {exc}")
        return None


def _parse_efts_hits(data: dict) -> list[dict]:
    """Normalize an EFTS search response into flat filing dicts.

    Each returned dict: {entity_name, cik, accession_number, filing_date,
    sic_code, sic_description, form}. Deduped within the response by
    accession_number.
    """
    out: list[dict] = []
    seen: set[str] = set()
    hits = (data or {}).get("hits", {}).get("hits", []) or []
    for hit in hits:
        src = hit.get("_source", {}) or {}
        # Prefer the explicit accession field; fall back to the _id prefix.
        accession = (src.get("adsh") or "").strip() or (
            hit.get("_id") or "").split(":", 1)[0]
        if not accession or accession in seen:
            continue
        display_names = src.get("display_names") or []
        if not display_names:
            continue
        entity_name = _strip_cik_suffix(display_names[0]).strip()
        if not entity_name:
            continue
        ciks = src.get("ciks") or [""]
        cik = (ciks[0] if ciks else "") or ""
        cik = cik.lstrip("0") or cik  # EDGAR archives want unpadded CIK
        sics = src.get("sics") or []
        sic_code = (sics[0] if sics else "") or ""
        seen.add(accession)
        out.append({
            "entity_name": entity_name,
            "cik": cik,
            "accession_number": accession,
            "filing_date": src.get("file_date") or src.get("filed") or "",
            "sic_code": sic_code,
            "sic_description": _SIC_DESCRIPTIONS.get(str(sic_code), ""),
            "form": src.get("form") or "S-1",
        })
    return out


# Minimal SIC-code → description map for the high-signal sectors. Used to
# enrich filings (EFTS returns the numeric code only) and feed the sector
# score. Unknown codes fall back to "" and score on the business blurb.
_SIC_DESCRIPTIONS: dict[str, str] = {
    "3674": "Semiconductors",
    "7372": "Prepackaged Software",
    "7370": "Computer Services / Software",
    "7389": "Computer / Technology Services",
    "3663": "Communications Equipment",
    "3812": "Defense / Search & Navigation Equipment",
    "3760": "Guided Missiles / Space Vehicles",
    "4899": "Communications Services",
    "6199": "Finance Services / Fintech",
    "6022": "Banking",
    "2836": "Biotech / Biological Products",
    "8731": "Commercial Physical & Biological Research",
    "2834": "Pharmaceutical Preparations",
    "3841": "Medical / Surgical Instruments",
    "1311": "Crude Petroleum / Energy",
    "3711": "Motor Vehicles",
    "5961": "Retail / E-commerce",
    "6798": "Real Estate Investment Trusts",
}


def fetch_all_s1_filings(lookback_days: int = EDGAR_FULL_SCAN_LOOKBACK_DAYS) -> list[dict]:
    """Autonomous full S-1 scanner — returns ALL new S-1 / S-1/A filings.

    Polls the EDGAR full-text search (EFTS) for both base S-1 and amended
    S-1/A filings across the trailing `lookback_days` window. Every filing
    is returned regardless of sector — NO pre-filtering, NO watchlist
    dependency. Deduplicated by accession_number against
    cache['s1_filings_seen']; already-seen accessions are dropped here so
    the caller only ever sees genuinely new filings.

    On EDGAR failure: logs a warning and returns [].
    """
    today_dt = datetime.now(timezone.utc)
    today = today_dt.strftime("%Y-%m-%d")
    start = (today_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    all_filings: list[dict] = []
    # Poll base S-1 and amended S-1/A. EFTS treats forms= as an exact
    # form filter, so each is queried separately. NOTE: the endpoint
    # 500s on an empty quoted query (q=%22%22) — we omit q entirely to
    # enumerate ALL filings of the form in the window. EFTS returns up to
    # 100 hits/page; we paginate with `from` until the window is drained
    # (hard cap to stay polite).
    EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
    PAGE = 100
    MAX_PAGES = 10  # ≤1000 filings/form/window — far above normal volume
    for form in ("S-1", "S-1/A"):
        for page in range(MAX_PAGES):
            params = {
                "forms": form,
                "startdt": start,
                "enddt": today,
                "from": page * PAGE,
            }
            data = _edgar_get(EFTS_URL, params=params, as_json=True)
            if data is None:
                if page == 0:
                    print(f"  WARNING: EDGAR {form} scan returned nothing")
                break
            page_hits = (data.get("hits", {}) or {}).get("hits", []) or []
            if not page_hits:
                break
            all_filings.extend(_parse_efts_hits(data))
            if len(page_hits) < PAGE:
                break

    if not all_filings:
        return []

    # Dedup across the two form queries by accession_number.
    by_accession: dict[str, dict] = {}
    for f in all_filings:
        by_accession.setdefault(f["accession_number"], f)

    # Drop accessions already processed in a prior run.
    cache = _load_cache()
    seen = cache.get("s1_filings_seen", {})
    fresh = [f for acc, f in by_accession.items() if acc not in seen]
    return fresh


def extract_s1_content(cik: str, accession_number: str) -> dict:
    """Fetch an S-1 and extract fundamentals via BeautifulSoup.

    Pulls the filing's primary document (largest .htm in the accession
    directory) and best-effort extracts: revenue, revenue growth %, lead
    underwriters, sector/business blurb, target raise, employee count.

    Returns a dict with an `extraction_quality` of "full" / "partial" /
    "failed". The entire body is wrapped in try/except — any failure
    yields {"extraction_quality": "failed"} so scoring can still proceed
    on filing metadata alone (graceful degradation).
    """
    result = {
        "revenue": None,
        "revenue_growth_pct": None,
        "lead_underwriters": [],
        "sector_description": "",
        "target_raise_usd": None,
        "employee_count": None,
        "extraction_quality": "failed",
    }
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  WARNING: bs4 not installed — S-1 content extraction disabled")
        return result

    try:
        acc_nodash = accession_number.replace("-", "")
        # Machine-readable directory listing for this accession. More
        # reliable than scraping the browse-edgar HTML index.
        idx_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
            "index.json"
        )
        idx = _edgar_get(idx_url, as_json=True)
        primary_url = None
        if idx:
            items = (idx.get("directory", {}) or {}).get("item", []) or []
            htm = [it for it in items
                   if str(it.get("name", "")).lower().endswith((".htm", ".html"))]
            if htm:
                # Primary doc is the largest .htm (skip tiny exhibit stubs).
                def _size(it):
                    try:
                        return int(it.get("size") or 0)
                    except (TypeError, ValueError):
                        return 0
                htm.sort(key=_size, reverse=True)
                primary_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{acc_nodash}/{htm[0]['name']}"
                )
        if not primary_url:
            return result

        import time
        time.sleep(EDGAR_CONTENT_SLEEP_SEC)  # respect EDGAR rate limits
        html = _edgar_get(primary_url, as_json=False)
        if not html:
            return result

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        low = text.lower()

        filled = 0

        # ── Lead underwriters ─────────────────────────────────────────
        found_uw = [uw for uw in _ALL_UNDERWRITERS if uw.lower() in low]
        # Collapse JPMorgan spelling variants etc. into canonical names.
        canon = []
        for uw in found_uw:
            c = uw.strip()
            if c not in canon:
                canon.append(c)
        if canon:
            result["lead_underwriters"] = canon
            filled += 1

        # ── Revenue (most recent annual) + growth ─────────────────────
        revs = _extract_revenue_figures(text)
        if revs:
            result["revenue"] = revs[0]
            if len(revs) >= 2 and revs[1]:
                try:
                    growth = (revs[0] - revs[1]) / abs(revs[1]) * 100.0
                    result["revenue_growth_pct"] = round(growth, 1)
                except ZeroDivisionError:
                    pass
            filled += 1

        # ── Target raise (aggregate offering price) ───────────────────
        raise_usd = _extract_dollar_near(text, "aggregate offering price")
        if raise_usd is None:
            raise_usd = _extract_dollar_near(text, "maximum aggregate offering")
        if raise_usd:
            result["target_raise_usd"] = raise_usd
            filled += 1

        # ── Employee count ────────────────────────────────────────────
        emp = _extract_employee_count(text)
        if emp:
            result["employee_count"] = emp
            filled += 1

        # ── Business / sector description (first ~500 words) ───────────
        blurb = _extract_business_blurb(text)
        if blurb:
            result["sector_description"] = blurb
            filled += 1

        if filled >= 3:
            result["extraction_quality"] = "full"
        elif filled >= 1:
            result["extraction_quality"] = "partial"
        else:
            result["extraction_quality"] = "failed"
        return result
    except Exception as exc:
        print(f"  WARNING: S-1 extraction failed for {cik}/{accession_number}: {exc}")
        return {
            "revenue": None,
            "revenue_growth_pct": None,
            "lead_underwriters": [],
            "sector_description": "",
            "target_raise_usd": None,
            "employee_count": None,
            "extraction_quality": "failed",
        }


# ── Content-extraction helpers (regex over plain text) ───────────────

_MULTIPLIERS = {
    "thousand": 1e3, "thousands": 1e3,
    "million": 1e6, "millions": 1e6,
    "billion": 1e9, "billions": 1e9,
    "trillion": 1e12, "trillions": 1e12,
}


def _money_to_float(num_str: str, unit: str = "") -> Optional[float]:
    """Convert ('1,234.5', 'million') → 1234500000.0. None on failure."""
    try:
        val = float(num_str.replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None
    mult = _MULTIPLIERS.get((unit or "").lower().strip(), 1.0)
    return val * mult


# Phrases that signal a *threshold* rather than reported revenue. The
# JOBS Act emerging-growth-company definition ("annual gross revenue of at
# least $1.235 billion", "revenues exceeded $100 million") appears in
# nearly every S-1 and must NOT be read as the issuer's actual revenue.
_REVENUE_BOILERPLATE = (
    "at least", "exceed", "in excess of", "greater than", "more than",
    "threshold", "emerging growth", "$1.235", "1.235 billion",
    "less than", "up to",
)


def _extract_revenue_figures(text: str) -> list[float]:
    """Return ACTUAL reported revenue figures, most-recent first.

    Requires reported-revenue phrasing ("revenue of/was/were/increased to
    $X") and rejects JOBS Act / threshold boilerplate so the EGC
    "$1.235 billion" definition can't masquerade as real revenue.
    """
    import re
    figures: list[float] = []
    # Must read like reported revenue, not a threshold test.
    pat = re.compile(
        r"(?:total\s+|net\s+|annual\s+)?revenues?\s+"
        r"(?:of|was|were|increased\s+to|decreased\s+to|grew\s+to|"
        r"totaled|of\s+approximately|reached)\s+"
        r"\$\s?([\d,]+(?:\.\d+)?)\s*(thousand|million|billion|trillion)?",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        # Reject if threshold/boilerplate language sits just before the hit.
        window = text[max(0, m.start() - 60):m.start()].lower()
        if any(b in window for b in _REVENUE_BOILERPLATE):
            continue
        val = _money_to_float(m.group(1), m.group(2) or "")
        if val is None or val < 1e5:   # ignore per-share / tiny figures
            continue
        figures.append(val)
        if len(figures) >= 4:
            break
    # Deduplicate while preserving order (filings repeat the same number).
    out: list[float] = []
    for v in figures:
        if v not in out:
            out.append(v)
    return out


def _extract_dollar_near(text: str, phrase: str) -> Optional[float]:
    """Find the first dollar amount within ~80 chars of `phrase`."""
    import re
    pat = re.compile(
        re.escape(phrase)
        + r"[^$]{0,80}?\$\s?([\d,]+(?:\.\d+)?)\s*(thousand|million|billion|trillion)?",
        re.IGNORECASE,
    )
    m = pat.search(text)
    if not m:
        return None
    return _money_to_float(m.group(1), m.group(2) or "")


def _extract_employee_count(text: str) -> Optional[int]:
    """Find 'we had N employees' / 'N full-time employees' style counts."""
    import re
    best = None
    pat = re.compile(
        r"(?:had|employed|approximately|of)\s+([\d,]{2,})\s+(?:full-time\s+)?employees",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        try:
            n = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if 1 <= n <= 5_000_000:
            best = n if best is None else max(best, n)
    return best


def _extract_business_blurb(text: str) -> str:
    """First ~500 words after a 'Business' / 'Overview' heading, else the
    opening of the document."""
    import re
    low = text.lower()
    anchor = -1
    for marker in ("our business", "business overview", "company overview",
                   "overview", "our company"):
        idx = low.find(marker)
        if idx != -1:
            anchor = idx
            break
    snippet = text[anchor:] if anchor != -1 else text
    words = snippet.split()
    return " ".join(words[:500]).strip()


# ── IPO scoring engine ───────────────────────────────────────────────

def _score_sector(sic_description: str, blurb: str) -> tuple[int, str]:
    hay = f"{sic_description} {blurb}".lower()
    if any(k in hay for k in _SECTOR_TIER1):
        return 10, "tier1"
    if any(k in hay for k in _SECTOR_TIER2):
        return 6, "tier2"
    if any(k in hay for k in _SECTOR_TIER3):
        return 3, "tier3"
    return 0, "other"


def _score_underwriters(underwriters: list[str]) -> int:
    joined = " ".join(underwriters).lower() if underwriters else ""
    if not joined:
        return 0
    if any(uw.lower() in joined for uw in _UW_TOP):
        return 20
    if any(uw.lower() in joined for uw in _UW_HIGH):
        return 15
    if any(uw.lower() in joined for uw in _UW_BULGE):
        return 10
    return 0


def score_ipo_filing(entity_name: str, filing_data: dict, content: dict) -> dict:
    """Score an S-1 filing 0–100 across five weighted components and map
    the total to a STRONG_BUY / WATCH / SKIP recommendation."""
    revenue = content.get("revenue")
    growth = content.get("revenue_growth_pct")
    underwriters = content.get("lead_underwriters") or []
    target_raise = content.get("target_raise_usd")
    blurb = content.get("sector_description") or ""
    sic_desc = filing_data.get("sic_description") or ""

    # Blank-check / SPAC shells (SIC 6770) have no operating history — any
    # "revenue"/"growth" we scraped is boilerplate. Null it so these don't
    # score as operating companies. They still pass through (no pre-filter)
    # but score on raise/underwriter/sector only.
    is_blank_check = str(filing_data.get("sic_code") or "") == "6770"
    if is_blank_check:
        revenue = None
        growth = None

    # Revenue score (0–30)
    if revenue is None:
        revenue_score = 0
    elif revenue > 1e9:
        revenue_score = 30
    elif revenue >= 500e6:
        revenue_score = 25
    elif revenue >= 100e6:
        revenue_score = 15
    elif revenue >= 50e6:
        revenue_score = 8
    else:
        revenue_score = 0

    # Growth score (0–25)
    if growth is None:
        growth_score = 0
    elif growth > 100:
        growth_score = 25
    elif growth >= 50:
        growth_score = 20
    elif growth >= 25:
        growth_score = 12
    elif growth >= 10:
        growth_score = 6
    else:
        growth_score = 0

    # Underwriter quality score (0–20)
    underwriter_score = _score_underwriters(underwriters)

    # Raise size score (0–15)
    if target_raise is None:
        raise_score = 0
    elif target_raise > 5e9:
        raise_score = 15
    elif target_raise >= 1e9:
        raise_score = 12
    elif target_raise >= 500e6:
        raise_score = 8
    else:
        raise_score = 3

    # Sector score (0–10)
    sector_score, _sector_tier = _score_sector(sic_desc, blurb)

    total = (revenue_score + growth_score + underwriter_score
             + raise_score + sector_score)

    if total >= IPO_STRONG_BUY_SCORE:
        recommendation = "STRONG_BUY"
    elif total >= IPO_WATCH_SCORE:
        recommendation = "WATCH"
    else:
        recommendation = "SKIP"

    rationale = _build_rationale(
        revenue, growth, underwriters, target_raise, sector_score, total
    )

    return {
        "entity_name": entity_name,
        "cik": filing_data.get("cik", ""),
        "accession_number": filing_data.get("accession_number", ""),
        "filing_date": filing_data.get("filing_date", ""),
        "total_score": total,
        "revenue_score": revenue_score,
        "growth_score": growth_score,
        "underwriter_score": underwriter_score,
        "raise_score": raise_score,
        "sector_score": sector_score,
        "revenue": revenue,
        "revenue_growth_pct": growth,
        "lead_underwriters": underwriters,
        "target_raise_usd": target_raise,
        "recommendation": recommendation,
        "rationale": rationale,
        "extraction_quality": content.get("extraction_quality", "failed"),
    }


def _fmt_money(v: Optional[float]) -> str:
    if v is None:
        return "unknown rev"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B rev"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M rev"
    return f"${v:,.0f} rev"


def _build_rationale(revenue, growth, underwriters, target_raise,
                     sector_score, total) -> str:
    bits = [_fmt_money(revenue)]
    if growth is not None:
        bits.append(f"{growth:+.0f}% growth")
    if underwriters:
        bits.append("led by " + ", ".join(underwriters[:2]))
    if target_raise:
        if target_raise >= 1e9:
            bits.append(f"${target_raise / 1e9:.1f}B raise")
        else:
            bits.append(f"${target_raise / 1e6:.0f}M raise")
    if sector_score >= 10:
        bits.append("high-signal sector")
    return f"Score {total}/100: " + "; ".join(bits)


# ── Ticker resolution engine ─────────────────────────────────────────

def _ticker_candidates(entity_name: str) -> list[str]:
    """Generate candidate tickers from a company legal name."""
    import re
    # Strip common corporate suffixes/noise.
    cleaned = re.sub(
        r"\b(inc|incorporated|corp|corporation|company|co|plc|ltd|limited|"
        r"holdings|holding|group|technologies|technology|llc|lp|sa|nv|ag|"
        r"the|class|common|stock)\b",
        " ", entity_name, flags=re.IGNORECASE,
    )
    words = [w for w in re.sub(r"[^A-Za-z0-9 ]", " ", cleaned).split() if w]
    candidates: list[str] = []

    def _add(c: str):
        c = c.upper().strip()
        if c and 1 <= len(c) <= 5 and c not in candidates:
            candidates.append(c)

    if words:
        first = words[0]
        _add(first[:4])            # first 4 letters of first word
        _add(first[:3])            # first 3 letters of first word
        if len(words) >= 2:
            _add(first[:3] + words[1][0])   # first3 + second-word initial
        # Acronym of significant words
        _add("".join(w[0] for w in words[:5]))
        _add(first)                # whole first word (if short)
    return candidates


def resolve_ticker(entity_name: str) -> Optional[str]:
    """Resolve a company name to a live trading ticker, or None.

    Checks TICKER_OVERRIDES first, then generated candidates (capped at
    IPO_TICKER_RESOLVE_ATTEMPTS), returning the first that is live per
    check_ticker_live().
    """
    if not entity_name:
        return None

    # Exact / substring override match first.
    for key, tick in TICKER_OVERRIDES.items():
        if key.lower() in entity_name.lower():
            try:
                if check_ticker_live(tick):
                    return tick
            except Exception:
                pass

    tried = 0
    for cand in _ticker_candidates(entity_name):
        if tried >= IPO_TICKER_RESOLVE_ATTEMPTS:
            break
        tried += 1
        try:
            if check_ticker_live(cand):
                return cand
        except Exception:
            continue
    return None


# ── Autonomous universe addition ─────────────────────────────────────

def _load_universe() -> dict:
    try:
        with open(UNIVERSE_FILE) as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}


def _save_universe(universe: dict) -> None:
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(universe, f, indent=2)
        f.write("\n")


def add_to_tier_b(ticker: str, name: str, sector: str, reason: str) -> dict:
    """Add a high-conviction IPO ticker directly to Tier B with the
    `source: ipo_intake` marker so kairos_signals_ipo emits IPO_MOMENTUM.

    Returns {"ok": True/False, ...}. No-op if already in the universe.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return {"ok": False, "error": "empty ticker"}
    if ticker in _existing_universe_tickers():
        return {"ok": False, "error": f"{ticker} already in universe"}

    universe = _load_universe()
    if not universe:
        return {"ok": False, "error": "universe unreadable"}
    tier_b = universe.setdefault("tier_b", {"added_date": "", "tickers": []})
    tier_b_tickers = tier_b.setdefault("tickers", [])

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tier_b_tickers.append({
        "symbol": ticker,
        "name": name,
        "sector": sector,
        "added_date": today,
        "source": "ipo_intake",        # → IPO_MOMENTUM signal in kairos_signals_ipo
        "signal_tag": "IPO_MOMENTUM",
        "add_reason": reason,
    })
    universe.setdefault("metadata", {})["last_updated"] = today
    _save_universe(universe)
    print(f"  IPO INTAKE: added {ticker} ({name}) to Tier B (IPO_MOMENTUM)")

    _append_decision_log({
        "type": "IPO_DISCOVERY_TIER_B",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "reason": reason,
        "source": "ipo_intake",
    })
    return {"ok": True, "ticker": ticker}


def _watchlist_has_name(name: str) -> bool:
    return name.lower().strip() in _watchlist_company_names()


def _add_to_ipo_watchlist(name: str, sector: str, status: str,
                          expected_ticker: str, score: dict) -> bool:
    """Persist a discovered company to kairos_config.json[ipo_watchlist]."""
    if _watchlist_has_name(name):
        return False
    entry = {
        "name": name,
        "expected_ticker": expected_ticker,
        "alt_tickers": [],
        "sector": sector,
        "status": status,
        "notes": (f"Auto-discovered S-1 (score {score.get('total_score')}/100, "
                  f"{score.get('recommendation')}). {score.get('rationale', '')}"),
        "discovered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cik": score.get("cik", ""),
        "accession": score.get("accession_number", ""),
        "ipo_score": score.get("total_score"),
    }
    return _append_watchlist_entry(entry)


def _guess_sector_label(score: dict) -> str:
    """Coarse sector label from the sector sub-score for universe tagging."""
    s = score.get("sector_score", 0)
    if s >= 10:
        return "Technology"
    if s >= 6:
        return "Healthcare"
    if s >= 3:
        return "Consumer"
    return "Unknown"


def process_ipo_discoveries(scored_filings: list[dict]) -> list[dict]:
    """Route scored filings into the universe / watchlist and announce.

      STRONG_BUY → resolve ticker; if live → Tier B (IPO_MOMENTUM) + a
                   #kairos-reports alert; else → ipo_watchlist (PENDING) +
                   a #kairos-reports pipeline alert.
      WATCH      → ipo_watchlist (status=watch) + #kairos-log only.
      SKIP       → log only.

    Returns a list of action records describing what was done.
    """
    try:
        from kairos_alerts import alert_pipeline_event
    except Exception:
        alert_pipeline_event = None

    def _post(msg: str, channel: str):
        if alert_pipeline_event is None:
            return
        try:
            alert_pipeline_event(msg, channel=channel)
        except Exception as exc:
            print(f"  WARNING: Slack post failed: {exc}")

    actions: list[dict] = []
    for score in scored_filings:
        name = score.get("entity_name", "")
        rec = score.get("recommendation")
        total = score.get("total_score", 0)
        rationale = score.get("rationale", "")
        sector = _guess_sector_label(score)

        if rec == "STRONG_BUY":
            ticker = None
            try:
                ticker = resolve_ticker(name)
            except Exception as exc:
                print(f"  WARNING: ticker resolution failed for {name}: {exc}")

            if ticker:
                res = add_to_tier_b(
                    ticker, name, sector,
                    reason=f"IPO STRONG_BUY (score {total})",
                )
                if res.get("ok"):
                    _post(
                        f":rocket: *IPO DISCOVERY:* {name} (`{ticker}`) — "
                        f"Score: {total}/100. {rationale}. Added to Tier B.",
                        "reports",
                    )
                    action_rec = {"name": name, "action": "tier_b",
                                  "ticker": ticker, "score": total}
                    # resolve_ticker only returns LIVE tickers, but re-probe
                    # defensively before committing capital.
                    is_live = False
                    try:
                        is_live = check_ticker_live(ticker)
                    except Exception:
                        is_live = False
                    if is_live:
                        offering = _current_price(ticker)
                        rsv = create_ipo_reservation(
                            ticker, name, total, offering or 0.0, rationale,
                        )
                        if rsv.get("ok"):
                            action_rec["reserved_usd"] = rsv["reserved_usd"]
                    else:
                        # Tier-B'd but not tradeable yet — let the day-1
                        # engine reserve the moment it goes live.
                        _add_to_ipo_watchlist(
                            name, sector, "pending_reservation", "PENDING", score)
                        action_rec["pending_reservation"] = True
                    actions.append(action_rec)
                else:
                    actions.append({"name": name, "action": "tier_b_skip",
                                    "ticker": ticker,
                                    "error": res.get("error")})
            else:
                # No ticker resolved → not yet live. Stage for reservation so
                # run_ipo_day1_engine() reserves the moment it starts trading.
                _add_to_ipo_watchlist(
                    name, sector, "pending_reservation", "PENDING", score)
                _post(
                    f":clipboard: *IPO PIPELINE:* {name} filed S-1 — "
                    f"Score: {total}/100. {rationale}. Watching for ticker.",
                    "reports",
                )
                actions.append({"name": name, "action": "watchlist_pending",
                                "score": total})

        elif rec == "WATCH":
            _add_to_ipo_watchlist(name, sector, "watch", "PENDING", score)
            _post(
                f":eyes: IPO WATCH: {name} filed S-1 — score {total}/100. "
                f"{rationale}",
                "log",
            )
            actions.append({"name": name, "action": "watchlist_watch",
                            "score": total})
        else:  # SKIP
            print(f"  IPO SKIP: {name} — score {total}/100 (<{IPO_WATCH_SCORE})")
            actions.append({"name": name, "action": "skip", "score": total})

    return actions


# ── EDGAR → reservation bridge + IPO decision engine ─────────────────
#
# kairos_ipo_capital.reserve_capital() can't be called as the spec sketches
# it: its signature is keyword-only (ticker=, conviction_score=), it sizes
# off a 7–10 conviction table (5/10/20/30% of *available capital*, no
# sub-5% entry) and it *requires* a live IBKR connection. The EDGAR engine
# instead sizes a small fixed % of NLV (1.5–2.0%) and must run headless in
# every cycle. So we persist the reservation row directly through
# kairos_ipo_capital's OWN schema/connection/STATUS constant — the row is
# byte-for-byte the same shape get_active_reservations() returns, so
# kairos_ipo_execute.py picks it up with zero changes. offering_price +
# source live in the JSON `notes` column (the table has no columns for them).

# IPO position sizing tiers based on EDGAR score
# Scores reflect revenue, growth, underwriter quality, raise size, sector
EDGAR_RESERVE_TIERS = [
    (95, 5.0),   # 95-100: once-in-a-decade IPO (SpaceX tier)
    (90, 3.5),   # 90-94:  exceptional
    (85, 2.5),   # 85-89:  strong
    (70, 1.5),   # 70-84:  standard STRONG_BUY
]
EDGAR_RESERVE_STD_SCORE = 70    # minimum score for any reservation
RESERVATION_SLACK_CHANNEL = "reports"


def _edgar_reservation_pct(edgar_score: int) -> Optional[float]:
    """% of NLV to reserve based on tiered EDGAR score table."""
    for min_score, pct in EDGAR_RESERVE_TIERS:
        if edgar_score >= min_score:
            return pct
    return None


def _get_nlv() -> Optional[float]:
    """Net liquidation value. Prefer IBKR (via kairos_ipo_capital), fall
    back to the most-recent net_liq_after in kairos.db. None if neither."""
    try:
        from kairos_ipo_capital import get_available_capital
        cap = get_available_capital() or {}
        nlv = cap.get("nlv")
        if nlv:
            return float(nlv)
    except Exception:
        pass
    try:
        import sqlite3
        from kairos_log_db import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute(
                "SELECT net_liq_after FROM decisions "
                "WHERE net_liq_after IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            return float(row[0])
    except Exception as exc:
        print(f"  WARNING: NLV lookup from kairos.db failed: {exc}")
    return None


def _current_price(ticker: str) -> Optional[float]:
    """Latest close via yfinance. None on any failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if hist is None or len(hist) == 0:
            return None
        px = float(hist["Close"].iloc[-1])
        return px if px > 0 else None
    except Exception:
        return None


def _reservation_exists(ticker: str, statuses=("reserved", "converted")) -> bool:
    """True iff an ipo_reservations row for `ticker` exists in any of the
    given statuses (default: reserved or converted)."""
    if not ticker:
        return False
    try:
        from kairos_ipo_capital import init_schema, _get_connection
        init_schema()
        conn = _get_connection()
        try:
            placeholders = ",".join("?" * len(statuses))
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM ipo_reservations "
                f"WHERE UPPER(ticker) = ? AND status IN ({placeholders})",
                (ticker.upper(), *statuses),
            ).fetchone()
            return bool(row and row["n"] > 0)
        finally:
            conn.close()
    except Exception as exc:
        print(f"  WARNING: reservation existence check failed for {ticker}: {exc}")
        return False


def _persist_edgar_reservation(*, ticker, entity_name, edgar_score,
                               reserved_usd, reserved_pct, nlv,
                               offering_price, rationale) -> Optional[int]:
    """Write one status='reserved' row to ipo_reservations via
    kairos_ipo_capital's schema. Returns the new id, or None on failure."""
    try:
        from kairos_ipo_capital import (
            init_schema, _get_connection, STATUS_RESERVED,
        )
    except Exception as exc:
        print(f"  WARNING: kairos_ipo_capital unavailable: {exc}")
        return None
    notes = json.dumps({
        "source": "edgar_discovery",
        "edgar_score": edgar_score,
        "offering_price": offering_price,
        "rationale": rationale,
    })
    try:
        init_schema()
        conn = _get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO ipo_reservations
                   (ticker, company_name, conviction_score, reserved_pct,
                    reserved_usd, nlv_at_reservation, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker.upper(), entity_name,
                    round(edgar_score / 10.0, 1),   # 0-100 → ~1-10 scale
                    round(reserved_pct / 100.0, 4), # store as fraction
                    round(float(reserved_usd), 2),
                    round(float(nlv), 2) if nlv else None,
                    STATUS_RESERVED, notes,
                ),
            )
            rid = int(cur.lastrowid)
            conn.commit()
            return rid
        finally:
            conn.close()
    except Exception as exc:
        print(f"  WARNING: reservation insert failed for {ticker}: {exc}")
        return None


def _annotate_scored_filing(ticker: str, entity_name: str,
                            offering_price: Optional[float]) -> None:
    """Stamp ticker + offering_price onto the matching scored_filing so
    score_ipo_for_entry() can resolve it by ticker later."""
    try:
        cache = _load_cache()
        sf = cache.get("scored_filings") or {}
        name_lc = (entity_name or "").lower()
        changed = False
        for _acc, s in sf.items():
            if (s.get("entity_name") or "").lower() == name_lc:
                s["ticker"] = ticker.upper()
                if offering_price is not None:
                    s["offering_price"] = offering_price
                changed = True
        if changed:
            _save_cache(cache)
    except Exception as exc:
        print(f"  WARNING: scored_filing annotation failed for {ticker}: {exc}")


def _init_forced_screen(ticker: str) -> None:
    """Initialize forced_screens[ticker] = 0 so the screener's IPO
    force-include window starts fresh for this reservation."""
    try:
        cache = _load_cache()
        forced = cache.get("forced_screens")
        if not isinstance(forced, dict):
            forced = {}
            cache["forced_screens"] = forced
        forced.setdefault(ticker.upper(), 0)
        _save_cache(cache)
    except Exception as exc:
        print(f"  WARNING: forced_screens init failed for {ticker}: {exc}")


def create_ipo_reservation(ticker: str, entity_name: str, edgar_score: int,
                           offering_price: float, rationale: str) -> dict:
    """Create a capital reservation for a live, high-conviction EDGAR IPO.

    Sizes reserved_usd as a % of NLV from the EDGAR score (≥85 → 2.0%,
    ≥70 → 1.5%), persists a status='reserved' row to ipo_reservations
    (consumed by kairos_ipo_execute.py), seeds forced_screens[ticker]=0,
    and posts to #kairos-reports. No-op if a reservation already exists.

    Returns {"ok": bool, "ticker": ..., "reserved_usd": ...}.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "ticker": ticker, "reserved_usd": 0.0,
                "error": "empty ticker"}

    pct = _edgar_reservation_pct(int(edgar_score))
    if pct is None:
        return {"ok": False, "ticker": ticker, "reserved_usd": 0.0,
                "error": f"score {edgar_score} below reservation threshold "
                         f"{EDGAR_RESERVE_STD_SCORE}"}

    if _reservation_exists(ticker, statuses=("reserved", "converted")):
        return {"ok": False, "ticker": ticker, "reserved_usd": 0.0,
                "error": "reservation already exists"}

    nlv = _get_nlv()
    if not nlv:
        return {"ok": False, "ticker": ticker, "reserved_usd": 0.0,
                "error": "NLV unavailable; cannot size reservation"}

    reservation_usd = round(pct / 100.0 * nlv, 2)

    rid = _persist_edgar_reservation(
        ticker=ticker, entity_name=entity_name, edgar_score=int(edgar_score),
        reserved_usd=reservation_usd, reserved_pct=pct, nlv=nlv,
        offering_price=offering_price, rationale=rationale,
    )
    if rid is None:
        return {"ok": False, "ticker": ticker, "reserved_usd": 0.0,
                "error": "reservation insert failed"}

    _init_forced_screen(ticker)
    _annotate_scored_filing(ticker, entity_name, offering_price)

    try:
        from kairos_alerts import alert_pipeline_event
        alert_pipeline_event(
            f":clipboard: *IPO RESERVATION:* {ticker} ({entity_name}) — "
            f"EDGAR score {edgar_score}/100. Reserved "
            f"${reservation_usd:,.0f} ({pct:.1f}% NLV). {rationale}",
            channel=RESERVATION_SLACK_CHANNEL,
        )
    except Exception as exc:
        print(f"  WARNING: reservation Slack post failed for {ticker}: {exc}")

    print(f"  IPO RESERVATION: {ticker} reserved ${reservation_usd:,.0f} "
          f"({pct:.1f}% NLV, score {edgar_score}, id #{rid})")
    return {"ok": True, "ticker": ticker, "reserved_usd": reservation_usd,
            "reservation_id": rid}


def _update_watchlist_status(name: str, new_status: str) -> bool:
    """Update an ipo_watchlist entry's status in kairos_config.json by name."""
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except (IOError, json.JSONDecodeError):
        return False
    wl = cfg.get("ipo_watchlist")
    if not isinstance(wl, list):
        return False
    name_lc = (name or "").lower()
    changed = False
    for e in wl:
        if isinstance(e, dict) and (e.get("name") or "").lower() == name_lc:
            e["status"] = new_status
            changed = True
    if not changed:
        return False
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        return True
    except IOError:
        return False


def _find_scored_filing(ticker: str, cache: Optional[dict] = None) -> Optional[dict]:
    """Locate the scored_filing for a ticker — by stamped ticker first,
    then by matching the detected-cache company name to entity_name."""
    cache = cache or _load_cache()
    sf = cache.get("scored_filings") or {}
    tu = ticker.upper()
    for _acc, s in sf.items():
        if (s.get("ticker") or "").upper() == tu:
            return s
    det = (cache.get("detected") or {}).get(tu) or {}
    name_lc = (det.get("name") or "").lower()
    if name_lc:
        for _acc, s in sf.items():
            if (s.get("entity_name") or "").lower() == name_lc:
                return s
    return None


def _get_offering_price(ticker: str,
                        scored_filing: Optional[dict]) -> Optional[float]:
    """Offering price from the reservation notes, else the scored_filing."""
    try:
        from kairos_ipo_capital import init_schema, _get_connection
        init_schema()
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT notes FROM ipo_reservations WHERE UPPER(ticker) = ? "
                "ORDER BY id DESC LIMIT 1", (ticker.upper(),),
            ).fetchone()
        finally:
            conn.close()
        if row and row["notes"]:
            data = json.loads(row["notes"])
            op = data.get("offering_price")
            if op:
                return float(op)
    except Exception:
        pass
    if scored_filing and scored_filing.get("offering_price"):
        try:
            return float(scored_filing["offering_price"])
        except (TypeError, ValueError):
            return None
    return None


def _days_since_ipo(ticker: str, cache: Optional[dict] = None) -> Optional[int]:
    """Days since this ticker was first detected as live."""
    cache = cache or _load_cache()
    det = (cache.get("detected") or {}).get(ticker.upper()) or {}
    ts = det.get("detected_at") or ""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts[:19] if "T" in ts else ts[:10], fmt)
            return max(0, (datetime.now(timezone.utc).replace(tzinfo=None) - dt).days)
        except (ValueError, TypeError):
            continue
    return None


def score_ipo_for_entry(ticker: str) -> dict:
    """IPO entry-opportunity score (0–100) for the reasoning layer.

    Blends the EDGAR conviction score (40%), Day-1 price action (30%),
    timing urgency since IPO (20%), and sector momentum (10%). Returns a
    dict with the inputs, a BUY/HOLD/SKIP recommendation, a one-line
    rationale, and a suggested position size (% NLV).
    """
    ticker = (ticker or "").strip().upper()
    out = {
        "ticker": ticker,
        "entry_score": 0,
        "edgar_score": None,
        "day1_performance": None,
        "days_since_ipo": None,
        "offering_price": None,
        "current_price": None,
        "recommendation": "SKIP",
        "rationale": "no EDGAR scored filing for ticker",
        "position_size_pct": 1.0,
        "entity_name": None,
        "revenue": None,
        "revenue_growth_pct": None,
        "lead_underwriters": [],
    }
    cache = _load_cache()
    sf = _find_scored_filing(ticker, cache)
    if not sf:
        return out

    edgar_score = int(sf.get("total_score") or 0)
    out["edgar_score"] = edgar_score
    out["entity_name"] = sf.get("entity_name")
    out["revenue"] = sf.get("revenue")
    out["revenue_growth_pct"] = sf.get("revenue_growth_pct")
    out["lead_underwriters"] = sf.get("lead_underwriters") or []

    offering_price = _get_offering_price(ticker, sf)
    current_price = _current_price(ticker)
    out["offering_price"] = offering_price
    out["current_price"] = current_price

    # 1. EDGAR component (40%)
    edgar_pts = edgar_score * 0.4

    # 2. Day-1 performance (30%)
    day1 = None
    if offering_price and current_price and offering_price > 0:
        day1 = (current_price - offering_price) / offering_price * 100.0
    out["day1_performance"] = round(day1, 1) if day1 is not None else None
    if day1 is None:
        day1_pts = 0
    elif day1 > 15:
        day1_pts = 30
    elif day1 >= 5:
        day1_pts = 20
    elif day1 >= 0:
        day1_pts = 12
    else:
        day1_pts = 0

    # 3. Timing (20%)
    days = _days_since_ipo(ticker, cache)
    out["days_since_ipo"] = days
    if days is None:
        timing_pts = 0
    elif days <= 3:
        timing_pts = 20
    elif days <= 7:
        timing_pts = 12
    elif days <= 14:
        timing_pts = 5
    else:
        timing_pts = 0

    # 4. Sector momentum (10%)
    sector_score = int(sf.get("sector_score") or 0)
    if sector_score >= 10:
        sector_pts = 10        # Tech / AI / Space
    elif sector_score >= 3:
        sector_pts = 5         # other growth
    else:
        sector_pts = 0         # value / unknown

    entry_score = int(round(edgar_pts + day1_pts + timing_pts + sector_pts))
    entry_score = max(0, min(100, entry_score))
    out["entry_score"] = entry_score

    if entry_score >= 80:
        out["position_size_pct"] = 2.0
    elif entry_score >= 60:
        out["position_size_pct"] = 1.5
    else:
        out["position_size_pct"] = 1.0

    if entry_score >= 60 and (day1 is None or day1 >= 0):
        out["recommendation"] = "BUY"
    elif entry_score >= 40:
        out["recommendation"] = "HOLD"
    else:
        out["recommendation"] = "SKIP"

    day1_str = f"{day1:+.1f}%" if day1 is not None else "n/a"
    out["rationale"] = (
        f"EDGAR {edgar_score}/100, day-1 {day1_str}, "
        f"{days if days is not None else '?'}d since IPO → "
        f"entry {entry_score}/100 ({out['recommendation']})"
    )
    return out


def run_ipo_day1_engine() -> dict:
    """Lightweight per-cycle engine: turn pending discoveries into
    reservations the moment their ticker starts trading.

    1. ipo_watchlist entries with status='pending_reservation' → if now
       live, reserve and flip status to 'reserved'.
    2. detected tickers with a STRONG_BUY scored_filing and no reservation
       yet → reserve if live.

    Returns {"new_reservations": int, "checked": int, "errors": [...]}.
    """
    summary = {"new_reservations": 0, "checked": 0, "errors": []}

    # 1. Pending-reservation watchlist entries.
    try:
        watchlist = load_watchlist()
    except Exception as exc:
        watchlist = []
        summary["errors"].append(f"load_watchlist: {exc}")

    cache = _load_cache()
    for entry in watchlist:
        if (entry.get("status") or "") != "pending_reservation":
            continue
        summary["checked"] += 1
        name = entry.get("name") or ""
        exp = (entry.get("expected_ticker") or "").upper()
        ticker = exp if (exp and exp != "PENDING") else None
        if not ticker:
            try:
                ticker = resolve_ticker(name)
            except Exception as exc:
                summary["errors"].append(f"resolve({name}): {exc}")
                continue
        if not ticker:
            continue
        try:
            if not check_ticker_live(ticker):
                continue
        except Exception:
            continue
        score = int(entry.get("ipo_score") or 0)
        if score < EDGAR_RESERVE_STD_SCORE:
            continue
        offering = _current_price(ticker)
        res = create_ipo_reservation(
            ticker, name, score, offering or 0.0,
            entry.get("notes") or f"EDGAR STRONG_BUY (score {score})",
        )
        if res.get("ok"):
            summary["new_reservations"] += 1
            _update_watchlist_status(name, "reserved")

    # 2. Detected tickers with a STRONG_BUY filing but no reservation yet.
    detected = (cache.get("detected") or {})
    for ticker, meta in detected.items():
        if not isinstance(meta, dict):
            continue
        tu = ticker.upper()
        if _reservation_exists(tu, statuses=("reserved", "converted")):
            continue
        sf = _find_scored_filing(tu, cache)
        if not sf:
            continue
        score = int(sf.get("total_score") or 0)
        if score < EDGAR_RESERVE_STD_SCORE:
            continue
        summary["checked"] += 1
        try:
            if not check_ticker_live(tu):
                continue
        except Exception:
            continue
        offering = _current_price(tu)
        res = create_ipo_reservation(
            tu, sf.get("entity_name") or meta.get("name") or tu, score,
            offering or 0.0, sf.get("rationale") or f"EDGAR STRONG_BUY (score {score})",
        )
        if res.get("ok"):
            summary["new_reservations"] += 1

    if summary["new_reservations"]:
        print(f"  IPO Day-1 engine: {summary['new_reservations']} new "
              f"reservation(s) from {summary['checked']} checked")
    return summary


# ── Pre-IPO news tracking (Finnhub) ───────────────────────────────────

def _load_finnhub_token() -> Optional[str]:
    """Read finnhub.api_key from kairos_config.json; fall back to env."""
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        token = cfg.get("finnhub", {}).get("api_key", "")
        if token:
            return token.strip()
    except (IOError, json.JSONDecodeError):
        pass
    return (os.environ.get("FINNHUB_API_KEY") or "").strip() or None


def fetch_preipoipo_news(company_name: str, finnhub_token: str) -> dict:
    """Search Finnhub general news for mentions of `company_name`.

    Returns: {company, article_count, avg_sentiment, velocity_7d,
              top_headlines (max 3)}.

    Strategy: Finnhub's company-news endpoint requires a symbol and is
    therefore useless for pre-IPO companies. We hit the general-news
    feed and substring-match on significant tokens of the company name.
    """
    default = {
        "company": company_name,
        "article_count": 0,
        "avg_sentiment": 0.0,
        "velocity_7d": 0.0,
        "top_headlines": [],
    }
    if not finnhub_token or not company_name:
        return default

    try:
        import requests
    except ImportError:
        return default

    name_tokens = _name_tokens(company_name)
    if not name_tokens:
        name_tokens = {company_name.lower()}

    # Primary: general-news category (broad firehose, last ~hours)
    matches: list[dict] = []
    try:
        url = "https://finnhub.io/api/v1/news"
        resp = requests.get(
            url,
            params={"category": "general", "token": finnhub_token},
            timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json() or []
    except Exception as exc:
        print(f"  WARNING: Finnhub general news fetch failed: {exc}")
        rows = []

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=PREIPO_NEWS_LOOKBACK_DAYS)).timestamp()
    for art in rows if isinstance(rows, list) else []:
        try:
            ts = float(art.get("datetime") or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts and ts < cutoff_ts:
            continue
        haystack = ((art.get("headline") or "") + " "
                    + (art.get("summary") or "")).lower()
        if any(tok in haystack for tok in name_tokens):
            matches.append(art)

    # Velocity: how many articles fell in the most-recent N days vs the
    # prior N days (positive = accelerating, zero/negative = decelerating).
    split_ts = (
        datetime.now(timezone.utc)
        - timedelta(days=PREIPO_NEWS_VELOCITY_SPLIT_DAYS)
    ).timestamp()
    recent = sum(1 for a in matches if float(a.get("datetime") or 0) >= split_ts)
    prior = len(matches) - recent
    velocity_7d = float(recent - prior)

    # Finnhub general news has no sentiment score; leave as 0.0 unless
    # we can derive one heuristically (very simple polarity from headline).
    avg_sentiment = _heuristic_sentiment(matches)

    headlines = []
    for a in sorted(matches,
                    key=lambda a: float(a.get("datetime") or 0),
                    reverse=True)[:3]:
        h = a.get("headline") or ""
        if h:
            headlines.append(h[:140])

    return {
        "company": company_name,
        "article_count": len(matches),
        "avg_sentiment": round(avg_sentiment, 3),
        "velocity_7d": velocity_7d,
        "top_headlines": headlines,
    }


_POS_WORDS = {"surge", "soar", "beat", "growth", "win", "expand", "record",
              "rally", "strong", "raise", "upgrade", "approve"}
_NEG_WORDS = {"miss", "fall", "drop", "lawsuit", "investigation", "delay",
              "downgrade", "warn", "loss", "decline", "cut", "fraud"}


def _heuristic_sentiment(articles: list[dict]) -> float:
    """Very rough polarity score in [-1, 1] from headline keywords."""
    if not articles:
        return 0.0
    score = 0
    counted = 0
    for art in articles:
        head = (art.get("headline") or "").lower()
        pos = sum(1 for w in _POS_WORDS if w in head)
        neg = sum(1 for w in _NEG_WORDS if w in head)
        if pos == 0 and neg == 0:
            continue
        score += (pos - neg)
        counted += 1
    if counted == 0:
        return 0.0
    return max(-1.0, min(1.0, score / counted))


def run_preipoipo_news_scan() -> list[dict]:
    """Scan watchlist entries whose ticker isn't live yet.

    Returns the list of news snapshots (one per scanned company). Posts
    a `:newspaper: PRE-IPO BUZZ` alert when article_count > 5 AND
    velocity_7d is positive (week-over-week acceleration).
    """
    token = _load_finnhub_token()
    if not token:
        print("  IPO news: no Finnhub token configured — skipping pre-IPO news scan")
        return []

    watchlist = load_watchlist()
    cache = _load_cache()
    news_cache = cache.setdefault("pre_ipo_news", {})
    results: list[dict] = []

    for entry in watchlist:
        # Skip entries whose ticker is already trading.
        candidates = [entry["expected_ticker"]] + list(entry.get("alt_tickers", []))
        if any(check_ticker_live(t) for t in candidates if t):
            continue

        snapshot = fetch_preipoipo_news(entry["name"], token)
        snapshot["timestamp"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        news_cache[entry["name"]] = snapshot
        results.append(snapshot)

        if (snapshot["article_count"] > PREIPO_NEWS_ALERT_MIN_ARTICLES
                and snapshot["velocity_7d"] > 0):
            try:
                from kairos_alerts import alert_pipeline_event
                lines = [
                    f":newspaper: *PRE-IPO BUZZ:* {entry['name']} — "
                    f"{snapshot['article_count']} articles this week, "
                    f"sentiment {snapshot['avg_sentiment']:.2f} "
                    f"(velocity +{int(snapshot['velocity_7d'])} w/w)"
                ]
                for h in snapshot["top_headlines"]:
                    lines.append(f"  • {h}")
                alert_pipeline_event("\n".join(lines), channel="reports")
            except Exception as exc:
                print(f"  WARNING: pre-IPO buzz alert failed: {exc}")

    _save_cache(cache)
    return results


# ── EDGAR auto-discovery ──────────────────────────────────────────────

def _watchlist_company_names() -> set[str]:
    """Lowercased names already on the watchlist, for dedup."""
    return {e["name"].lower() for e in load_watchlist()}


def _append_watchlist_entry(entry: dict) -> bool:
    """Persist a new entry to kairos_config.json["ipo_watchlist"].

    Creates the key (and copies in the current hardcoded baseline) if
    the config doesn't yet have one. Returns True on success.
    """
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except (IOError, json.JSONDecodeError) as exc:
        print(f"  WARNING: cannot read kairos_config.json: {exc}")
        return False

    existing = cfg.get("ipo_watchlist")
    if not isinstance(existing, list):
        # Seed config with the current effective watchlist so the act of
        # auto-discovery doesn't accidentally drop the hardcoded baseline.
        existing = list(IPO_WATCHLIST)

    existing.append(entry)
    cfg["ipo_watchlist"] = existing

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: cannot write kairos_config.json: {exc}")
        return False
    return True


def run_edgar_autodiscovery() -> list[dict]:
    """Scan recent S-1 filings (7d) and auto-promote keyword-matched ones.

    For each filing that:
      - is not already on the watchlist (by lowercased name), and
      - is not already in the cache's `s1_seen` dedup, and
      - matches at least one autodiscovery keyword on the issuer name,
    add a new `discovered` entry to kairos_config.json[ipo_watchlist]
    and post a `:mag: NEW IPO CANDIDATE DISCOVERED` alert.

    Returns the list of newly-discovered companies (each is the entry
    written to the watchlist).
    """
    try:
        import requests
    except ImportError:
        return []

    today_dt = datetime.now(timezone.utc)
    today = today_dt.strftime("%Y-%m-%d")
    start = (today_dt - timedelta(days=S1_AUTODISCOVERY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?q=%22%22&dateRange=custom&startdt={start}&enddt={today}&forms=S-1"
    )

    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  WARNING: EDGAR autodiscovery fetch failed: {exc}")
        return []

    hits = data.get("hits", {}).get("hits", []) or []
    cache = _load_cache()
    s1_seen = cache.setdefault("s1_seen", {})
    on_watchlist = _watchlist_company_names()

    discovered: list[dict] = []
    for hit in hits:
        src = hit.get("_source", {}) or {}
        accession = (hit.get("_id") or "").split(":", 1)[0]
        if not accession or accession in s1_seen:
            continue
        display_names = src.get("display_names") or []
        if not display_names:
            continue
        issuer = _strip_cik_suffix(display_names[0]).strip()
        if not issuer:
            continue
        if issuer.lower() in on_watchlist:
            continue

        kw_hits = _keyword_hits(issuer, _S1_KEYWORDS_AUTODISCOVERY)
        if not kw_hits:
            continue

        ciks = src.get("ciks") or [""]
        cik = ciks[0] if ciks else ""
        filing_date = src.get("file_date") or src.get("filed") or ""
        edgar_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik}&type=S-1&dateb=&owner=include&count=10"
        ) if cik else ""

        new_entry = {
            "name": issuer,
            "expected_ticker": "UNKNOWN",
            "alt_tickers": [],
            "sector": _guess_sector_from_keywords(kw_hits),
            "notes": (f"Auto-discovered via S-1 filing on {filing_date}. "
                      f"Keywords: {','.join(kw_hits)}."),
            "status": "discovered",
            "discovered_at": today_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cik": cik,
            "accession": accession,
            "edgar_url": edgar_url,
        }

        if not _append_watchlist_entry(new_entry):
            continue

        # Mark as seen so re-runs in the next 7d don't re-discover it.
        s1_seen[accession] = {
            "seen_at": today_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "company": issuer,
            "source": "autodiscovery",
            "keywords": kw_hits,
        }
        on_watchlist.add(issuer.lower())
        discovered.append(new_entry)

        try:
            from kairos_alerts import alert_pipeline_event
            alert_pipeline_event(
                f":mag: *NEW IPO CANDIDATE DISCOVERED:* {issuer} filed S-1. "
                f"Added to watchlist.\n"
                f"  Keywords: {', '.join(kw_hits)} | filed {filing_date}\n"
                f"  <{edgar_url}|EDGAR>",
                channel="watchlist",
            )
        except Exception as exc:
            print(f"  WARNING: autodiscovery alert failed for {issuer}: {exc}")

    cache["s1_seen"] = s1_seen
    _save_cache(cache)
    return discovered


_SECTOR_HINTS = {
    "artificial intelligence": "Technology",
    "semiconductor": "Technology",
    "chip": "Technology",
    "SaaS": "Technology",
    "cloud": "Technology",
    "fintech": "Financials",
    "neobank": "Financials",
    "biotech": "Healthcare",
}


def _guess_sector_from_keywords(kw_hits: list[str]) -> str:
    for kw in kw_hits:
        sector = _SECTOR_HINTS.get(kw)
        if sector:
            return sector
    return "Unknown"


# ── Once-per-day gate (called from kairos_run.py) ────────────────────

def is_full_scan_due(now_et=None) -> bool:
    """Return True iff a daily IPO scan should run now.

    Conditions:
      - We're in the 9:30–9:35 ET market-open window on a weekday.
      - We have not already completed a scan today (per
        cache['last_full_scan']).
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return False

    if now_et is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))

    if now_et.weekday() >= 5:
        return False
    if not (now_et.hour == MARKET_OPEN_HOUR_ET
            and MARKET_OPEN_MIN_FROM <= now_et.minute < MARKET_OPEN_MIN_TO):
        return False

    cache = _load_cache()
    last = (cache.get("last_full_scan") or "").strip()
    today = now_et.strftime("%Y-%m-%d")
    # last_full_scan is stored as an ET-local timestamp so a single
    # prefix comparison is unambiguous regardless of UTC offset.
    if last.startswith(today):
        return False
    return True


def _stamp_last_full_scan() -> None:
    cache = _load_cache()
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        cache["last_full_scan"] = now_et.strftime("%Y-%m-%dT%H:%M:%S ET")
    except ImportError:
        # Fall back to UTC; the date prefix may then be off by a few
        # hours but the gate still prevents same-cycle double-fires.
        cache["last_full_scan"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    _save_cache(cache)


# ── Orchestration entry point ────────────────────────────────────────

def run_ipo_intake() -> dict:
    """Autonomous IPO discovery — watchlist-free. Safe; never raises.

    Pipeline:
      1. fetch_all_s1_filings(30d)        — ALL new S-1 / S-1/A filings
      2. extract_s1_content() per filing  — revenue, growth, underwriters…
      3. score_ipo_filing() per filing    — 0–100 + STRONG_BUY/WATCH/SKIP
      4. process_ipo_discoveries()        — Tier B / watchlist + Slack
      5. scan_watchlist_for_live_tickers()— curated PENDING entries → Tier C
      6. stamp last_full_scan; return comprehensive summary

    Scored filings are cached (cache['scored_filings'][accession]) and
    never re-scored. New filings are marked in cache['s1_filings_seen'].
    """
    summary = {
        "filings_found": 0,
        "scored": 0,
        "strong_buy": 0,
        "watch": 0,
        "skip": 0,
        "tier_b_added": 0,
        "watchlist_added": 0,
        "live_detected": 0,
        "tier_c_added": 0,
        "errors": [],
    }

    cache = _load_cache()
    scored_cache = cache.setdefault("scored_filings", {})
    seen = cache.setdefault("s1_filings_seen", {})
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Fetch ALL new S-1 / S-1/A filings (no sector pre-filter).
    try:
        filings = fetch_all_s1_filings(lookback_days=EDGAR_FULL_SCAN_LOOKBACK_DAYS)
    except Exception as exc:
        filings = []
        msg = f"fetch_all_s1_filings failed: {exc}"
        print(f"  WARNING: {msg}")
        summary["errors"].append(msg)

    summary["filings_found"] = len(filings)

    # Rate-limit guard: cap the batch. We can't pre-rank by raise size
    # (that's only known post-extraction), so we cap by filing order and
    # log the drop so a backlog is never silently truncated.
    if len(filings) > EDGAR_MAX_FILINGS_PER_BATCH:
        dropped = len(filings) - EDGAR_MAX_FILINGS_PER_BATCH
        print(f"  IPO Scanner: {len(filings)} new filings — capping to "
              f"{EDGAR_MAX_FILINGS_PER_BATCH} this run ({dropped} deferred "
              f"to next cycle; not yet marked seen)")
        batch = filings[:EDGAR_MAX_FILINGS_PER_BATCH]
    else:
        batch = filings

    # 2–3. Extract + score each new filing (skip anything already scored).
    scored_filings: list[dict] = []
    for filing in batch:
        acc = filing["accession_number"]
        if acc in scored_cache:
            continue
        try:
            content = extract_s1_content(filing["cik"], acc)
        except Exception as exc:
            print(f"  WARNING: extraction crashed for {acc}: {exc}")
            content = {"extraction_quality": "failed", "lead_underwriters": []}
        try:
            score = score_ipo_filing(filing["entity_name"], filing, content)
        except Exception as exc:
            msg = f"scoring failed for {filing['entity_name']}: {exc}"
            print(f"  WARNING: {msg}")
            summary["errors"].append(msg)
            continue

        scored_filings.append(score)
        scored_cache[acc] = score
        seen[acc] = {
            "seen_at": now_iso,
            "company": filing["entity_name"],
            "filing_date": filing.get("filing_date", ""),
            "recommendation": score["recommendation"],
            "total_score": score["total_score"],
        }

    summary["scored"] = len(scored_filings)
    summary["strong_buy"] = sum(1 for s in scored_filings
                                if s["recommendation"] == "STRONG_BUY")
    summary["watch"] = sum(1 for s in scored_filings
                           if s["recommendation"] == "WATCH")
    summary["skip"] = sum(1 for s in scored_filings
                          if s["recommendation"] == "SKIP")

    _save_cache(cache)

    # 4. Route discoveries into the universe / watchlist + announce.
    try:
        actions = process_ipo_discoveries(scored_filings)
        summary["tier_b_added"] = sum(1 for a in actions
                                      if a.get("action") == "tier_b")
        summary["watchlist_added"] = sum(
            1 for a in actions
            if a.get("action") in ("watchlist_pending", "watchlist_watch")
        )
    except Exception as exc:
        msg = f"process_ipo_discoveries failed: {exc}"
        print(f"  WARNING: {msg}")
        summary["errors"].append(msg)

    # 5. Curated-watchlist live-ticker probe (PENDING entries → Tier C).
    try:
        live = scan_watchlist_for_live_tickers()
    except Exception as exc:
        live = []
        msg = f"watchlist probe failed: {exc}"
        print(f"  WARNING: {msg}")
        summary["errors"].append(msg)

    summary["live_detected"] = len(live)
    for hit in live:
        ticker = hit["ticker"]
        long_name = _lookup_ticker_name(ticker) or hit["name"]
        result = add_to_tier_c(
            ticker=ticker, name=long_name,
            sector=hit["sector"], reason=hit["source"],
        )
        cache["detected"][ticker] = {
            "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "name": hit["name"],
            "added": bool(result.get("ok")),
        }
        if result.get("ok"):
            summary["tier_c_added"] += 1
    _save_cache(cache)

    # 6. Stamp completion + log the comprehensive line.
    _stamp_last_full_scan()
    print(
        f"  IPO Scanner: {summary['filings_found']} new S-1 filings found, "
        f"{summary['scored']} scored, {summary['strong_buy']} STRONG_BUY, "
        f"{summary['watch']} WATCH "
        f"(Tier B: {summary['tier_b_added']}, watchlist: "
        f"{summary['watchlist_added']}, live→Tier C: {summary['tier_c_added']})"
    )
    return summary


# ── Helpers consumed by !ipo command ─────────────────────────────────

def get_watchlist_status() -> list[dict]:
    """For each watchlist entry, report whether the expected_ticker
    (or any alt) is currently live. Does not modify cache.
    """
    watchlist = load_watchlist()
    cache = _load_cache()
    detected = cache.get("detected", {})

    out = []
    for e in watchlist:
        live_ticker = None
        for t in [e["expected_ticker"]] + list(e.get("alt_tickers", [])):
            if not t:
                continue
            if t in detected:
                live_ticker = t
                break
            try:
                if check_ticker_live(t):
                    live_ticker = t
                    break
            except Exception:
                continue
        out.append({
            "name": e["name"],
            "expected_ticker": e["expected_ticker"],
            "alt_tickers": e.get("alt_tickers", []),
            "sector": e.get("sector", "Unknown"),
            "notes": e.get("notes", ""),
            "live_ticker": live_ticker,
            "is_live": live_ticker is not None,
        })
    return out


# ── CLI ──────────────────────────────────────────────────────────────

def demo_spacex_scoring() -> dict:
    """Show how SpaceX (SPCX) scores under the autonomous engine using
    representative S-1 fundamentals. Pure offline computation — no network.
    """
    filing_data = {
        "entity_name": "Space Exploration Technologies Corp",
        "cik": "0000000000",
        "accession_number": "0000000000-26-000001",
        "filing_date": "2026-05-01",
        "sic_code": "3760",
        "sic_description": _SIC_DESCRIPTIONS.get("3760", "Guided Missiles / Space Vehicles"),
    }
    # Representative figures (public reporting around the 2026 offering).
    content = {
        "revenue": 15.5e9,            # ~$15.5B annual revenue
        "revenue_growth_pct": 55.0,   # ~55% YoY
        "lead_underwriters": ["Goldman Sachs", "Morgan Stanley", "J.P. Morgan"],
        "sector_description": ("SpaceX designs, manufactures and launches "
                               "advanced rockets and spacecraft, and operates "
                               "the Starlink satellite internet constellation."),
        "target_raise_usd": 10e9,     # ~$10B raise
        "employee_count": 13000,
        "extraction_quality": "full",
    }
    score = score_ipo_filing("Space Exploration Technologies Corp",
                             filing_data, content)

    print("=" * W)
    print("  IPO ENGINE TEST RUN — SpaceX (SPCX)")
    print("=" * W)
    print(f"  Entity         : {score['entity_name']}")
    print(f"  Revenue        : ${content['revenue']/1e9:.1f}B  → {score['revenue_score']}/30")
    print(f"  Growth         : {content['revenue_growth_pct']:.0f}%  → {score['growth_score']}/25")
    print(f"  Underwriters   : {', '.join(content['lead_underwriters'])}  → {score['underwriter_score']}/20")
    print(f"  Target raise   : ${content['target_raise_usd']/1e9:.1f}B  → {score['raise_score']}/15")
    print(f"  Sector         : {filing_data['sic_description']}  → {score['sector_score']}/10")
    print("  " + "-" * (W - 2))
    print(f"  TOTAL SCORE    : {score['total_score']}/100")
    print(f"  RECOMMENDATION : {score['recommendation']}")
    print(f"  RATIONALE      : {score['rationale']}")
    resolved = TICKER_OVERRIDES.get("SpaceX")
    print(f"  TICKER (override) : {resolved}  → would route to Tier B (IPO_MOMENTUM) if live")
    print("=" * W)
    return score


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kairos IPO intake")
    parser.add_argument("--scan", action="store_true",
                        help="Probe watchlist tickers and print status")
    parser.add_argument("--s1", action="store_true",
                        help="Fetch + match EDGAR S-1 filings only")
    parser.add_argument("--run", action="store_true",
                        help="Full autonomous S-1 discovery run")
    parser.add_argument("--demo-spacex", action="store_true",
                        help="Show how SpaceX (SPCX) scores (offline)")
    args = parser.parse_args()

    if args.demo_spacex:
        demo_spacex_scoring()
        return

    if args.run or (not args.scan and not args.s1):
        summary = run_ipo_intake()
        print(json.dumps(summary, indent=2))
        return

    if args.scan:
        for s in get_watchlist_status():
            tag = f"LIVE @ {s['live_ticker']}" if s["is_live"] else "pending"
            print(f"  {s['name']:<24}  {s['expected_ticker']:<6}  [{tag}]")
    if args.s1:
        for m in fetch_edgar_s1_effectiveness():
            print(f"  {m['filing_type']:<7} {m['filing_date']}  "
                  f"{m['company']:<40}  → {m['match_name']} (score={m['score']})")


if __name__ == "__main__":
    main()
