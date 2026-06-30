"""
Kairos IPO Tracker — Pipeline Discovery + Conviction Scoring

Implements the discovery + scoring half of the HOT-IPO architecture
(Craft: "HOT-IPO: IPO Tracking & Capital Allocation Architecture").

  1. Discovery: pull S-1 / S-1/A filings from EDGAR EFTS and (when
     authorized) the Renaissance Capital pricing calendar. Absorb the
     manually-curated watchlist from kairos_ipo_intake.
  2. Extraction: best-effort regex on the prospectus cover page —
     underwriter, price range, shares offered.
  3. Scoring: HOT-IPO conviction score 1-10 from underwriter tier,
     sector momentum, and pricing revisions across S-1/A amendments.
     Business-quality and valuation are v2 (placeholder 0 for now).
  4. Reservation trigger: when a pipeline entry crosses score ≥7 and
     is within T-5 trading days of an expected pricing date, call
     kairos_ipo_capital.reserve_capital(dry_run=True) — actual orders
     stay opt-in until you flip --no-dry-run on the daily run.

State lives in kairos_ipo_pipeline.json, keyed by EDGAR CIK (or by a
'WATCHLIST:slug' pseudo-key for pre-S-1 watchlist entries). Each
entry tracks the full filings history so price-range revisions are
detected by comparing S-1/A amendments to the original S-1.

Renaissance ToS §8(q) — Renaissance payloads stay out of LLM prompts.
Scoring + reservation decisions are pure numeric logic.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PIPELINE_FILE = os.path.join(SCRIPT_DIR, "kairos_ipo_pipeline.json")

W = 72

SEC_HEADERS = {
    "User-Agent": "Kairos Trading System jason@meridiangroup.llc",
    "Accept": "application/json, application/atom+xml, text/html",
}
FETCH_TIMEOUT = 20
PROSPECTUS_BYTE_CAP = 524288  # ~512KB cap — cover page lives near the front

EDGAR_LOOKBACK_DAYS = 90
RESERVATION_LEAD_TRADING_DAYS = 5
RESERVATION_WINDOW_CAL_DAYS = 7  # ≈5 trading days

BASE_SCORE = 4
SCORE_MIN = 0
SCORE_MAX = 10

# Threshold gates for Slack-on-cross logic
THRESHOLD_MONITOR = 7
THRESHOLD_AGGRESSIVE = 9

MARKET_OPEN_HOUR_ET = 9
MARKET_OPEN_MIN_FROM = 30
MARKET_OPEN_MIN_TO = 40  # widen vs intake's 5-min window so both fit

SLACK_CHANNEL = "alerts"

# ── Underwriter tier map (substring match, case-insensitive) ─────────

TIER_1_UNDERWRITERS = (
    "goldman sachs", "morgan stanley", "j.p. morgan", "jpmorgan", "jp morgan",
    "bofa securities", "merrill lynch", "bank of america securities",
)
TIER_2_UNDERWRITERS = (
    "citigroup", "citi global", "barclays", "wells fargo", "deutsche bank",
    "ubs investment bank", "ubs securities", "credit suisse", "rbc capital",
)
TIER_3_UNDERWRITERS = (
    "jefferies", "cowen", "td cowen", "stifel", "raymond james",
    "piper sandler", "needham", "william blair", "canaccord", "oppenheimer",
    "wedbush", "lake street", "roth capital", "b. riley", "evercore",
    "guggenheim", "leerink", "truist",
)

# ── Sector keyword hints (mirrors kairos_ipo_intake._SECTOR_HINTS) ──

_SECTOR_HINTS = {
    "artificial intelligence": "Technology",
    "ai": "Technology",
    "semiconductor": "Technology",
    "chip": "Technology",
    "saas": "Technology",
    "cloud": "Technology",
    "fintech": "Financials",
    "neobank": "Financials",
    "bank": "Financials",
    "insurance": "Financials",
    "biotech": "Healthcare",
    "pharma": "Healthcare",
    "therapeutics": "Healthcare",
    "energy": "Energy",
    "oil": "Energy",
    "renewable": "Utilities",
    "retail": "Consumer Discretionary",
}


# ─────────────────────────────────────────────────────────────────────
# Pipeline persistence
# ─────────────────────────────────────────────────────────────────────

_PIPELINE_DEFAULTS: dict = {
    "last_run": "",
    "pipeline": {},     # key -> entry
}


def _load_pipeline() -> dict:
    if not os.path.exists(PIPELINE_FILE):
        return {k: ({} if isinstance(v, dict) else v)
                for k, v in _PIPELINE_DEFAULTS.items()}
    try:
        with open(PIPELINE_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {k: ({} if isinstance(v, dict) else v)
                    for k, v in _PIPELINE_DEFAULTS.items()}
        for k, default in _PIPELINE_DEFAULTS.items():
            data.setdefault(k, ({} if isinstance(default, dict) else default))
        return data
    except (IOError, json.JSONDecodeError):
        return {k: ({} if isinstance(v, dict) else v)
                for k, v in _PIPELINE_DEFAULTS.items()}


def _save_pipeline(state: dict) -> None:
    try:
        with open(PIPELINE_FILE, "w") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: pipeline write failed: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Discovery — EDGAR EFTS
# ─────────────────────────────────────────────────────────────────────

def _edgar_efts_filings(lookback_days: int) -> list[dict]:
    """Return normalized S-1 / S-1/A filings from EDGAR EFTS."""
    today_dt = datetime.now(timezone.utc)
    today = today_dt.strftime("%Y-%m-%d")
    start = (today_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?q=%22%22&dateRange=custom&startdt={start}&enddt={today}&forms=S-1"
    )
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  WARNING: EDGAR EFTS S-1 fetch failed: {exc}")
        return []

    hits = data.get("hits", {}).get("hits", []) or []
    out: list[dict] = []
    for hit in hits:
        src = hit.get("_source", {}) or {}
        accession = (hit.get("_id") or "").split(":", 1)[0]
        if not accession:
            continue
        display_names = src.get("display_names") or []
        if not display_names:
            continue
        ciks = src.get("ciks") or [""]
        cik = ciks[0] if ciks else ""
        out.append({
            "accession":   accession,
            "cik":         cik,
            "company":     _strip_cik_suffix(display_names[0]),
            "form":        src.get("form") or "S-1",
            "filing_date": src.get("file_date") or src.get("filed") or "",
            "source":      "edgar",
        })
    return out


def _strip_cik_suffix(display: str) -> str:
    out = display
    for _ in range(2):
        idx = out.rfind("(")
        if idx > 0 and out.endswith(")"):
            out = out[:idx].strip()
        else:
            break
    return out.strip()


# ─────────────────────────────────────────────────────────────────────
# Discovery — Renaissance Capital calendar (graceful on 401)
# ─────────────────────────────────────────────────────────────────────

def _renaissance_calendar_filings() -> list[dict]:
    """Pull the Renaissance pricing calendar. Returns [] on 401/empty."""
    try:
        from kairos_renaissance import get_calendar
    except Exception:
        return []
    try:
        rows = get_calendar()
    except Exception as exc:
        print(f"  WARNING: Renaissance calendar failed: {exc}")
        return []
    if not rows:
        return []

    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        company = (r.get("companyName") or r.get("Issuer")
                   or r.get("CompanyName") or "").strip()
        ticker = (r.get("tickerSymbol") or r.get("Ticker")
                  or r.get("Symbol") or "").strip().upper()
        pricing_date = (r.get("pricingDate") or r.get("PricingDate")
                        or r.get("expectedDate") or "")
        if not company:
            continue
        out.append({
            "accession":   "",   # not from EDGAR
            "cik":         "",
            "company":     company,
            "ticker":      ticker or None,
            "form":        "CALENDAR",
            "filing_date": pricing_date,
            "expected_pricing_date": pricing_date or None,
            "source":      "renaissance",
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# Discovery — manually-curated watchlist absorption
# ─────────────────────────────────────────────────────────────────────

_MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_RE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_RE_LONG = re.compile(
    rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:,)?\s+(\d{{4}})\b",
    re.IGNORECASE,
)


def _parse_pricing_date_hint(notes: str) -> Optional[str]:
    """Extract a YYYY-MM-DD pricing hint from a free-text notes field."""
    if not notes:
        return None
    m = _DATE_RE_ISO.search(notes)
    if m:
        try:
            y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
            return f"{y}-{mo:02d}-{d:02d}"
        except ValueError:
            pass
    m = _DATE_RE_LONG.search(notes)
    if m:
        token = f"{m.group(1)} {m.group(2)} {m.group(3)}"
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(token, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


def _watchlist_pseudo_entries() -> list[dict]:
    """Convert each intake watchlist entry into a discovery row."""
    try:
        from kairos_ipo_intake import load_watchlist
    except Exception:
        return []
    out: list[dict] = []
    try:
        wl = load_watchlist()
    except Exception:
        return []
    for w in wl:
        name = w.get("name") or ""
        if not name:
            continue
        ticker = (w.get("expected_ticker") or "").upper() or None
        pricing_date = (
            w.get("expected_pricing_date")
            or _parse_pricing_date_hint(w.get("notes", ""))
        )
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        out.append({
            "accession":   "",
            "cik":         "",
            "company":     name,
            "ticker":      ticker,
            "form":        "WATCHLIST",
            "filing_date": "",
            "sector":      w.get("sector") or None,
            "expected_pricing_date": pricing_date,
            "source":      "watchlist",
            "watchlist_slug": slug,
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# Discovery orchestration
# ─────────────────────────────────────────────────────────────────────

def discover_pending_ipos(lookback_days: int = EDGAR_LOOKBACK_DAYS) -> list[dict]:
    """Combine EDGAR + Renaissance + watchlist sources, dedup, return."""
    rows: list[dict] = []
    rows.extend(_edgar_efts_filings(lookback_days))
    rows.extend(_renaissance_calendar_filings())
    rows.extend(_watchlist_pseudo_entries())

    # Dedup: EDGAR rows are unique by accession; everything else by
    # lowercased company name. EDGAR takes precedence — when a
    # Renaissance / watchlist row maps to the same company as an EDGAR
    # row, the EDGAR data wins but we copy over the pricing date if
    # the EDGAR row didn't have one.
    seen_company: dict[str, dict] = {}
    seen_accession: set[str] = set()
    out: list[dict] = []
    for row in rows:
        accession = row.get("accession") or ""
        company_lc = (row.get("company") or "").strip().lower()
        if accession and accession in seen_accession:
            continue
        if company_lc and company_lc in seen_company:
            prior = seen_company[company_lc]
            if not prior.get("expected_pricing_date"):
                prior["expected_pricing_date"] = row.get("expected_pricing_date")
            if not prior.get("ticker"):
                prior["ticker"] = row.get("ticker")
            if not prior.get("sector"):
                prior["sector"] = row.get("sector")
            continue
        if accession:
            seen_accession.add(accession)
        if company_lc:
            seen_company[company_lc] = row
        out.append(row)
    return out


# ─────────────────────────────────────────────────────────────────────
# S-1 prospectus fetch + regex extraction
# ─────────────────────────────────────────────────────────────────────

def fetch_s1_prospectus(cik: str, accession: str) -> str:
    """Best-effort fetch of the cover-page text from an EDGAR S-1.

    Strategy: hit the filing's index.json, pick the largest .htm that
    isn't an index/header file, then Range-GET the first ~512KB. Cover
    page (underwriters, price range) is reliably in the first 512KB.

    Returns "" on any failure.
    """
    if not (cik and accession):
        return ""
    adsh = accession.replace("-", "")
    try:
        cik_int = int(cik)
    except (TypeError, ValueError):
        return ""

    base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh}"
    try:
        resp = requests.get(f"{base}/index.json", headers=SEC_HEADERS,
                            timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        idx = resp.json()
    except Exception:
        return ""

    items = (idx.get("directory") or {}).get("item") or []
    candidates = []
    for it in items:
        name = (it.get("name") or "").lower()
        if not name.endswith((".htm", ".html")):
            continue
        if "index" in name or "header" in name:
            continue
        size = 0
        try:
            size = int(it.get("size") or 0)
        except (TypeError, ValueError):
            pass
        candidates.append((size, it["name"]))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    prospectus_name = candidates[0][1]

    try:
        prosp = requests.get(
            f"{base}/{prospectus_name}",
            headers={**SEC_HEADERS,
                     "Range": f"bytes=0-{PROSPECTUS_BYTE_CAP - 1}"},
            timeout=FETCH_TIMEOUT,
        )
        # Some EDGAR servers ignore Range and return 200 with the full
        # file — still readable.
        if prosp.status_code in (200, 206):
            return prosp.text
    except Exception:
        return ""
    return ""


def extract_underwriter(text: str) -> Optional[dict]:
    """Return {name, tier} for the best-matching tier-1 underwriter.

    Substring-matches a curated list (case-insensitive). Tier 1 wins
    over Tier 2 over Tier 3 — we report the highest-tier match.
    """
    if not text:
        return None
    low = text.lower()
    for tier_no, lst in ((1, TIER_1_UNDERWRITERS),
                         (2, TIER_2_UNDERWRITERS),
                         (3, TIER_3_UNDERWRITERS)):
        for cand in lst:
            if cand in low:
                return {"name": cand.title(), "tier": tier_no}
    return None


_PRICE_RANGE_RE = re.compile(
    r"\$\s*([\d]+(?:\.[\d]+)?)\s*(?:and|to|–|—|-)\s*\$\s*([\d]+(?:\.[\d]+)?)"
    r"\s*per\s*share",
    re.IGNORECASE,
)
_SHARES_RE = re.compile(
    r"([\d,]+)\s+shares\s+of(?:\s+our)?\s+(?:common stock|Class\s+[A-Z])",
    re.IGNORECASE,
)


def extract_price_range(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _PRICE_RANGE_RE.search(text)
    if not m:
        return None
    try:
        lo = float(m.group(1))
        hi = float(m.group(2))
        if lo <= 0 or hi <= 0 or hi < lo or hi > 10000:
            return None
        return {"low": lo, "high": hi}
    except ValueError:
        return None


def extract_shares_offered(text: str) -> Optional[int]:
    if not text:
        return None
    m = _SHARES_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        val = int(raw)
        if val < 1000 or val > 10_000_000_000:
            return None
        return val
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────
# Per-filing extraction wrapper
# ─────────────────────────────────────────────────────────────────────

def extract_filing_details(discovery_row: dict) -> dict:
    """Pull underwriter / price range / shares from a discovery row.

    Returns the row augmented with `_extracted` flag and extracted
    fields. Non-EDGAR rows are returned untouched.
    """
    out = dict(discovery_row)
    out.setdefault("underwriter", None)
    out.setdefault("price_range", None)
    out.setdefault("shares_offered", None)
    out["_extracted"] = False

    if (discovery_row.get("source") != "edgar"
            or not discovery_row.get("cik")
            or not discovery_row.get("accession")):
        return out

    text = fetch_s1_prospectus(discovery_row["cik"], discovery_row["accession"])
    if not text:
        return out

    out["underwriter"] = extract_underwriter(text)
    out["price_range"] = extract_price_range(text)
    out["shares_offered"] = extract_shares_offered(text)
    out["_extracted"] = True

    # Sector hint from company name keywords
    if not out.get("sector"):
        out["sector"] = _guess_sector_from_company(discovery_row.get("company", ""))
    return out


def _guess_sector_from_company(name: str) -> Optional[str]:
    if not name:
        return None
    low = name.lower()
    for kw, sector in _SECTOR_HINTS.items():
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return sector
    return None


# ─────────────────────────────────────────────────────────────────────
# Pipeline-entry assembly + scoring
# ─────────────────────────────────────────────────────────────────────

def _entry_key(discovery_row: dict) -> str:
    """The pipeline dict key for a discovery row."""
    if discovery_row.get("source") == "watchlist":
        return f"WATCHLIST:{discovery_row.get('watchlist_slug') or 'unknown'}"
    cik = (discovery_row.get("cik") or "").strip()
    if cik:
        try:
            return f"CIK:{int(cik):010d}"
        except (TypeError, ValueError):
            pass
    company = (discovery_row.get("company") or "").strip().lower()
    return f"NAME:{re.sub(r'[^a-z0-9]+', '_', company).strip('_') or 'unknown'}"


def _filing_record(row: dict) -> dict:
    """One row of pipeline_entry['filings']."""
    return {
        "accession":   row.get("accession") or "",
        "form":        row.get("form") or "",
        "filing_date": row.get("filing_date") or "",
        "underwriter": row.get("underwriter"),
        "price_range": row.get("price_range"),
        "shares_offered": row.get("shares_offered"),
        "observed_at": datetime.now(timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _upsert_entry(pipeline: dict, discovery_row: dict) -> dict:
    """Merge a discovery row into the pipeline state, returning the entry.

    On a new CIK we create a fresh entry. On an existing CIK we append
    the new filing if the accession isn't already on file. Other
    metadata (company, sector, ticker, expected_pricing_date) is filled
    in from whichever source got there first, preferring non-empty.
    """
    key = _entry_key(discovery_row)
    entry = pipeline.setdefault(key, {
        "company":               discovery_row.get("company"),
        "cik":                   discovery_row.get("cik") or "",
        "ticker":                discovery_row.get("ticker"),
        "sector":                discovery_row.get("sector"),
        "expected_pricing_date": discovery_row.get("expected_pricing_date"),
        "source":                discovery_row.get("source"),
        "filings":               [],
        "score":                 None,
        "prior_score":           None,
        "score_components":      {},
        "reservation_id":        None,
        "last_scored_at":        None,
        "notes":                 [],
    })

    # Fill-forward unknown fields
    for k in ("company", "ticker", "sector", "expected_pricing_date"):
        if not entry.get(k) and discovery_row.get(k):
            entry[k] = discovery_row.get(k)

    if discovery_row.get("source") == "edgar" and discovery_row.get("accession"):
        seen = {f.get("accession") for f in entry["filings"]}
        if discovery_row["accession"] not in seen:
            entry["filings"].append(_filing_record(discovery_row))
            entry["filings"].sort(key=lambda f: f.get("filing_date") or "")
    return entry


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def _score_underwriter(filings: list[dict]) -> tuple[int, str]:
    """Best (highest-tier) underwriter across the entry's filings."""
    best = (0, "unknown")
    for f in filings:
        uw = f.get("underwriter") or {}
        tier = uw.get("tier")
        if tier == 1 and best[0] < 3:
            best = (3, f"tier1:{uw.get('name')}")
        elif tier == 2 and best[0] < 2:
            best = (2, f"tier2:{uw.get('name')}")
        elif tier == 3 and best[0] < 1:
            best = (1, f"tier3:{uw.get('name')}")
    return best


def _score_sector(sector: Optional[str]) -> tuple[int, str]:
    if not sector:
        return 0, "unknown_sector"
    try:
        from kairos_renaissance import sector_momentum
        sm = sector_momentum(sector)
    except Exception as exc:
        return 0, f"sector_momentum_error:{exc}"
    regime = (sm or {}).get("regime")
    if regime == "hot":     return +1, "hot"
    if regime == "cool":    return -1, "cool"
    if regime == "neutral": return  0, "neutral"
    return 0, "unknown"


def _score_pricing_revision(filings: list[dict]) -> tuple[int, str]:
    """Compare the most recent price range to the prior one."""
    ranges = [(f.get("filing_date") or "", f.get("price_range"), f.get("form"))
              for f in filings
              if f.get("price_range") and isinstance(f["price_range"], dict)]
    if len(ranges) < 2:
        return 0, "no_revisions"
    ranges.sort(key=lambda r: r[0])
    prior   = ranges[-2][1]
    current = ranges[-1][1]
    try:
        prior_mid   = (float(prior["low"])   + float(prior["high"]))   / 2
        current_mid = (float(current["low"]) + float(current["high"])) / 2
    except (TypeError, ValueError, KeyError):
        return 0, "parse_error"
    if prior_mid <= 0:
        return 0, "no_prior_mid"
    if current_mid > prior_mid * 1.02:
        return +2, f"upward_revision:{prior_mid:.2f}->{current_mid:.2f}"
    if current_mid < prior_mid * 0.98:
        return -3, f"downward_revision:{prior_mid:.2f}->{current_mid:.2f}"
    return 0, "no_change"


def score_pipeline_entry(entry: dict) -> dict:
    """Compute the HOT-IPO conviction score components for one entry.

    Returns:
        {
          "score": int (0-10),
          "components": {
              "base": 4,
              "underwriter": int,
              "sector":      int,
              "pricing_revision": int,
              "business_quality": 0,    # v2 placeholder
              "valuation":      0,      # v2 placeholder
          },
          "notes": {...},
        }
    """
    filings = entry.get("filings") or []

    uw_pts,   uw_label   = _score_underwriter(filings)
    sec_pts,  sec_label  = _score_sector(entry.get("sector"))
    rev_pts,  rev_label  = _score_pricing_revision(filings)

    # v2 placeholders — held at 0 so they don't move the score in v1.
    # TODO(v2): replace with LLM extraction over the S-1 prospectus text
    #   (fetch_s1_prospectus already pulls the cover-page-region bytes).
    #   - business_quality: have the LLM read MD&A + risk factors and
    #     score revenue growth / margins / customer concentration / moat.
    #   - valuation: have the LLM derive implied market cap from the
    #     price range x shares_offered, then compare to the sector comp
    #     set (SECTOR_COMP_TICKERS) on a revenue/earnings multiple basis.
    #   Both must stay EDGAR/Finnhub-sourced — Renaissance payloads are
    #   barred from LLM prompts per ToS section 8(q) (see kairos_renaissance).
    bq_pts, val_pts = 0, 0

    total = BASE_SCORE + uw_pts + sec_pts + rev_pts + bq_pts + val_pts
    total = max(SCORE_MIN, min(SCORE_MAX, total))

    return {
        "score": total,
        "components": {
            "base":              BASE_SCORE,
            "underwriter":       uw_pts,
            "sector":            sec_pts,
            "pricing_revision":  rev_pts,
            "business_quality":  bq_pts,
            "valuation":         val_pts,
        },
        "notes": {
            "underwriter":      uw_label,
            "sector":           sec_label,
            "pricing_revision": rev_label,
            "business_quality": "v2_deferred",
            "valuation":        "v2_deferred",
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Reservation trigger
# ─────────────────────────────────────────────────────────────────────

def _is_within_reservation_window(pricing_date_iso: Optional[str],
                                  today: Optional[datetime] = None) -> bool:
    """True iff today is in [pricing_date - 5 trading days, pricing_date]."""
    if not pricing_date_iso:
        return False
    try:
        pd = datetime.strptime(pricing_date_iso[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    today = today or datetime.now(timezone.utc)
    today_d = today.date() if hasattr(today, "date") else today
    threshold = pd - timedelta(days=RESERVATION_WINDOW_CAL_DAYS)
    return threshold <= today_d <= pd


def _existing_reservation_for(ticker: Optional[str],
                              company: Optional[str]) -> Optional[int]:
    """Return the id of an active reservation matching this IPO, or None."""
    try:
        from kairos_ipo_capital import get_active_reservations
    except Exception:
        return None
    try:
        actives = get_active_reservations()
    except Exception:
        return None
    t_low = (ticker or "").strip().upper()
    c_low = (company or "").strip().lower()
    for r in actives:
        if t_low and (r.get("ticker") or "").upper() == t_low:
            return int(r["id"])
        if c_low and (r.get("company_name") or "").strip().lower() == c_low:
            return int(r["id"])
    return None


def _maybe_reserve(entry: dict, *, dry_run: bool) -> Optional[dict]:
    """Reserve capital for an entry if it qualifies. Returns the result
    dict from reserve_capital(), or None if no reservation was attempted.
    """
    score = entry.get("score") or 0
    if score < THRESHOLD_MONITOR:
        return None
    pricing_date = entry.get("expected_pricing_date")
    if not _is_within_reservation_window(pricing_date):
        entry["notes"] = (entry.get("notes") or []) + [
            "skip_reserve:no_pricing_date_or_window"
            if not pricing_date else "skip_reserve:outside_window"
        ]
        return None

    ticker = entry.get("ticker")
    company = entry.get("company")
    if entry.get("reservation_id"):
        entry["notes"] = (entry.get("notes") or []) + ["skip_reserve:already_reserved"]
        return None
    existing = _existing_reservation_for(ticker, company)
    if existing:
        entry["reservation_id"] = existing
        entry["notes"] = (entry.get("notes") or []) + [f"skip_reserve:existing:{existing}"]
        return None

    if not ticker:
        # No ticker yet (e.g. pre-pricing watchlist row) — we can't
        # invoke reserve_capital cleanly without one. Log and skip.
        entry["notes"] = (entry.get("notes") or []) + ["skip_reserve:no_ticker_yet"]
        return None

    try:
        from kairos_ipo_capital import reserve_capital
        result = reserve_capital(
            ticker=ticker,
            conviction_score=float(score),
            expected_pricing_date=pricing_date,
            company_name=company,
            dry_run=dry_run,
        )
    except Exception as exc:
        print(f"  WARNING: reserve_capital({ticker}) failed: {exc}")
        return None

    if result and result.get("ok") and result.get("reservation_id"):
        entry["reservation_id"] = int(result["reservation_id"])
    return result


# ─────────────────────────────────────────────────────────────────────
# Daily gate
# ─────────────────────────────────────────────────────────────────────

def is_tracker_due(now_et=None) -> bool:
    """True iff the daily tracker run should fire now.

    Weekday market-open window (9:30–9:40 ET) AND not already run today.
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

    state = _load_pipeline()
    last = (state.get("last_run") or "").strip()
    today = now_et.strftime("%Y-%m-%d")
    if last.startswith(today):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────
# Slack helpers
# ─────────────────────────────────────────────────────────────────────

def _slack(msg: str) -> None:
    try:
        from kairos_alerts import alert_pipeline_event
        alert_pipeline_event(msg, channel=SLACK_CHANNEL)
    except Exception as exc:
        print(f"  WARNING: Slack post failed: {exc}")


def _slack_threshold_cross(entry: dict, threshold: int) -> None:
    icon = ":fire:" if threshold >= THRESHOLD_AGGRESSIVE else ":chart_with_upwards_trend:"
    name = entry.get("company") or entry.get("ticker") or "?"
    ticker = entry.get("ticker") or "—"
    score = entry.get("score") or 0
    comps = entry.get("score_components") or {}
    pricing = entry.get("expected_pricing_date") or "unknown"
    _slack(
        f"{icon} *HOT-IPO score ≥{threshold}:* {name} ({ticker}) "
        f"— conviction {score}/10\n"
        f"  Underwriter: {comps.get('underwriter', 0):+d}  "
        f"Sector: {comps.get('sector', 0):+d}  "
        f"Pricing: {comps.get('pricing_revision', 0):+d}\n"
        f"  Expected pricing: {pricing}"
    )


def _slack_summary(summary: dict) -> None:
    _slack(
        f":bar_chart: *HOT-IPO daily pipeline:* "
        f"{summary.get('pipeline_size', 0)} pending  |  "
        f"score≥7: {summary.get('high_conviction', 0)}  |  "
        f"reservations triggered: {summary.get('reservations_triggered', 0)}  |  "
        f"price ↑: {summary.get('pricing_revisions_up', 0)}  "
        f"↓: {summary.get('pricing_revisions_down', 0)}"
    )


# ─────────────────────────────────────────────────────────────────────
# Run orchestration
# ─────────────────────────────────────────────────────────────────────

def run_ipo_tracker(dry_run: bool = True,
                    lookback_days: int = EDGAR_LOOKBACK_DAYS) -> dict:
    """Discover → extract → score → persist → conditionally reserve.

    Safe — every step is wrapped so a single failure never aborts the
    cycle. Returns a small summary dict.
    """
    summary = {
        "pipeline_size":           0,
        "scored":                  0,
        "high_conviction":         0,
        "aggressive":              0,
        "reservations_triggered":  0,
        "pricing_revisions_up":    0,
        "pricing_revisions_down":  0,
        "extracted":               0,
        "errors":                  [],
    }

    try:
        discovery = discover_pending_ipos(lookback_days=lookback_days)
    except Exception as exc:
        summary["errors"].append(f"discover_pending_ipos: {exc}")
        return summary

    state = _load_pipeline()
    pipeline = state.setdefault("pipeline", {})

    for row in discovery:
        try:
            if row.get("source") == "edgar":
                row = extract_filing_details(row)
                if row.get("_extracted"):
                    summary["extracted"] += 1
        except Exception as exc:
            summary["errors"].append(f"extract({row.get('company')}): {exc}")
        try:
            _upsert_entry(pipeline, row)
        except Exception as exc:
            summary["errors"].append(f"upsert({row.get('company')}): {exc}")

    # Score every entry — Slack on threshold crossings + reserve if eligible
    for key, entry in list(pipeline.items()):
        try:
            sr = score_pipeline_entry(entry)
            prior = entry.get("score")
            entry["prior_score"]      = prior
            entry["score"]            = sr["score"]
            entry["score_components"] = sr["components"]
            entry["score_notes"]      = sr["notes"]
            entry["last_scored_at"]   = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            summary["scored"] += 1

            if entry["score"] >= THRESHOLD_MONITOR:
                summary["high_conviction"] += 1
            if entry["score"] >= THRESHOLD_AGGRESSIVE:
                summary["aggressive"] += 1

            rev_pts = (sr.get("components") or {}).get("pricing_revision", 0)
            if rev_pts > 0:
                summary["pricing_revisions_up"] += 1
            elif rev_pts < 0:
                summary["pricing_revisions_down"] += 1

            # Slack on first cross of MONITOR / AGGRESSIVE
            if (prior is None or prior < THRESHOLD_MONITOR) and entry["score"] >= THRESHOLD_MONITOR:
                _slack_threshold_cross(entry, THRESHOLD_MONITOR)
            elif ((prior is None or prior < THRESHOLD_AGGRESSIVE)
                  and entry["score"] >= THRESHOLD_AGGRESSIVE):
                _slack_threshold_cross(entry, THRESHOLD_AGGRESSIVE)

            # Reservation
            reserve_result = _maybe_reserve(entry, dry_run=dry_run)
            if reserve_result and reserve_result.get("ok"):
                summary["reservations_triggered"] += 1
        except Exception as exc:
            summary["errors"].append(f"score({key}): {exc}")

    summary["pipeline_size"] = len(pipeline)

    # Stamp last_run + persist
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        state["last_run"] = now_et.strftime("%Y-%m-%dT%H:%M:%S ET")
    except ImportError:
        state["last_run"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    _save_pipeline(state)

    _slack_summary(summary)
    print(
        f"  IPO tracker: pipeline={summary['pipeline_size']} "
        f"score>=7={summary['high_conviction']} "
        f"reservations={summary['reservations_triggered']} "
        f"price_up={summary['pricing_revisions_up']} "
        f"price_dn={summary['pricing_revisions_down']} "
        f"extracted={summary['extracted']}"
    )
    return summary


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
        description="Kairos HOT-IPO conviction tracker",
    )
    parser.add_argument("--scan", action="store_true",
                        help="Run discovery only (no extraction or scoring)")
    parser.add_argument("--score", action="store_true",
                        help="Score current pipeline; do not reserve")
    parser.add_argument("--run", action="store_true",
                        help="Full daily run (discovery + score + reserve)")
    parser.add_argument("--show", metavar="KEY",
                        help="Print the pipeline entry for a key")
    parser.add_argument("--lookback", type=int, default=EDGAR_LOOKBACK_DAYS,
                        help="EDGAR lookback window in days")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Issue real reservations (default is dry-run)")
    args = parser.parse_args()

    did_any = False

    if args.scan:
        _print_json("Discovery", discover_pending_ipos(lookback_days=args.lookback))
        did_any = True

    if args.show:
        state = _load_pipeline()
        _print_json(f"Entry {args.show}",
                    (state.get("pipeline") or {}).get(args.show))
        did_any = True

    if args.score:
        state = _load_pipeline()
        pipeline = state.setdefault("pipeline", {})
        for key, entry in pipeline.items():
            sr = score_pipeline_entry(entry)
            entry["prior_score"]      = entry.get("score")
            entry["score"]            = sr["score"]
            entry["score_components"] = sr["components"]
            entry["score_notes"]      = sr["notes"]
            entry["last_scored_at"]   = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        _save_pipeline(state)
        _print_json("ScoreSummary",
                    {k: {"score": v.get("score"),
                         "ticker": v.get("ticker"),
                         "company": v.get("company")}
                     for k, v in pipeline.items()})
        did_any = True

    if args.run:
        summary = run_ipo_tracker(
            dry_run=not args.no_dry_run,
            lookback_days=args.lookback,
        )
        _print_json("RunSummary", summary)
        did_any = True

    if not did_any:
        parser.print_help()


if __name__ == "__main__":
    main()
