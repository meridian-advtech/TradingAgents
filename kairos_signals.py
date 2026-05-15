"""
Kairos Buy Signal Detectors

Five independent signal detectors that run during Tier 1 screening.
Each returns a dict of {ticker: metadata} for tickers that fired.

Signals:
  HOT-EARNINGS  — beat earnings estimates by >5% in last 30 days (Finnhub)
  HOT-RSI       — 14-period RSI below 30, oversold (Finnhub candles)
  HOT-KALSHI    — Kalshi market probability shifted >10% since last snapshot
  HOT-INSIDER   — SEC Form 4 insider purchase in last 7 days (EDGAR)
  HOT-CONGRESS  — Congressional stock purchase in last 30 days (House disclosures)

All detectors are fault-tolerant: they return empty dicts on failure
so screening is never blocked by a single broken data source.
"""

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KALSHI_CACHE = os.path.join(SCRIPT_DIR, "kairos_kalshi_prev.json")
TIMEOUT = 15

W = 72


def banner(title: str) -> str:
    return f"\n{'━' * W}\n  {title}\n{'━' * W}"


# ═════════════════════════════════════════════════════════════════════
# 1. EARNINGS SURPRISE — Finnhub /calendar/earnings (one bulk call)
# ═════════════════════════════════════════════════════════════════════

def detect_earnings_surprise(
    universe: set[str],
    api_key: str,
    lookback_days: int = 30,
    surprise_threshold: float = 5.0,
) -> dict[str, dict]:
    """Return tickers that beat earnings estimates by >surprise_threshold%.

    Uses Finnhub /calendar/earnings which returns ALL earnings in a date
    range — a single API call regardless of universe size.
    """
    if not api_key:
        return {}

    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": start, "to": end, "token": api_key},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    Earnings API error: {e}")
        return {}

    results = {}
    for item in data.get("earningsCalendar", []):
        ticker = item.get("symbol", "")
        if ticker not in universe:
            continue

        actual = item.get("epsActual")
        estimate = item.get("epsEstimate")
        if actual is None or estimate is None or estimate == 0:
            continue

        surprise_pct = ((actual - estimate) / abs(estimate)) * 100
        if surprise_pct > surprise_threshold:
            results[ticker] = {
                "surprise_pct": round(surprise_pct, 1),
                "eps_actual": actual,
                "eps_estimate": estimate,
                "date": item.get("date", ""),
            }

    return results


# ═════════════════════════════════════════════════════════════════════
# 2. OVERSOLD RSI — Finnhub /stock/candle → compute 14-period RSI
# ═════════════════════════════════════════════════════════════════════

def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Compute RSI from a list of closing prices (oldest first)."""
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _fetch_rsi_for_ticker(ticker: str, api_key: str) -> tuple[str, float | None]:
    """Fetch 30 daily candles and compute RSI. Returns (ticker, rsi_or_None)."""
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (35 * 86400)  # ~35 days back for 30 trading days

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={"symbol": ticker, "resolution": "D",
                    "from": start, "to": now, "token": api_key},
            timeout=TIMEOUT,
        )
        if resp.status_code == 429:
            return ticker, None
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return ticker, None

    closes = data.get("c", [])
    if not closes or data.get("s") == "no_data":
        return ticker, None

    rsi = _compute_rsi(closes)
    return ticker, rsi


def detect_oversold_rsi(
    candidates: list[str],
    api_key: str,
    rsi_threshold: float = 30.0,
    max_workers: int = 6,
) -> dict[str, float]:
    """Check RSI for a subset of tickers (non-COLD only to save API calls).

    Returns {ticker: rsi_value} for tickers with RSI < threshold.
    """
    if not api_key or not candidates:
        return {}

    results = {}
    # Process in waves of 50 to respect Finnhub 60/min rate limit
    for wave_start in range(0, len(candidates), 50):
        wave = candidates[wave_start:wave_start + 50]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch_rsi_for_ticker, t, api_key) for t in wave]
            for fut in as_completed(futures):
                ticker, rsi = fut.result()
                if rsi is not None and rsi < rsi_threshold:
                    results[ticker] = rsi

        remaining = len(candidates) - (wave_start + 50)
        if remaining > 0:
            time.sleep(62)

    return results


# ═════════════════════════════════════════════════════════════════════
# 3. KALSHI PROBABILITY SHIFT — compare to previous snapshot
# ═════════════════════════════════════════════════════════════════════

def detect_kalshi_shift(shift_threshold: float = 10.0) -> list[dict]:
    """Detect Kalshi markets where probability shifted >threshold% since last check.

    Returns list of {title, old_prob, new_prob, shift, direction}.
    Caches current probabilities for next comparison.
    """
    try:
        resp = requests.get(
            "https://api.elections.kalshi.com/trade-api/v2/events",
            params={"limit": 20, "status": "open"},
            headers={"Accept": "application/json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception as e:
        print(f"    Kalshi API error: {e}")
        return []

    # Build current probability map
    current: dict[str, float] = {}
    event_titles: dict[str, str] = {}
    for ev in events:
        ticker = ev.get("ticker", "")
        title = ev.get("title", ticker)
        # Use yes_ask as a proxy for probability if available
        prob = ev.get("yes_ask", ev.get("last_price", 0))
        if isinstance(prob, (int, float)) and prob > 0:
            current[ticker] = prob * 100 if prob <= 1 else prob
            event_titles[ticker] = title

    # Load previous snapshot
    previous: dict[str, float] = {}
    if os.path.exists(KALSHI_CACHE):
        try:
            with open(KALSHI_CACHE) as f:
                previous = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Save current for next run
    try:
        with open(KALSHI_CACHE, "w") as f:
            json.dump(current, f)
    except IOError:
        pass

    if not previous:
        return []

    # Compare
    shifts = []
    for ticker, new_prob in current.items():
        old_prob = previous.get(ticker)
        if old_prob is None:
            continue
        shift = new_prob - old_prob
        if abs(shift) >= shift_threshold:
            direction = "UP" if shift > 0 else "DOWN"
            shifts.append({
                "ticker": ticker,
                "title": event_titles.get(ticker, ticker)[:80],
                "old_prob": round(old_prob, 1),
                "new_prob": round(new_prob, 1),
                "shift": round(shift, 1),
                "direction": direction,
            })

    return shifts


# ═════════════════════════════════════════════════════════════════════
# 4. INSIDER BUYING — SEC EDGAR Form 4 full-text search
# ═════════════════════════════════════════════════════════════════════

def detect_insider_buying(
    universe: set[str],
    lookback_days: int = 7,
) -> dict[str, list[dict]]:
    """Search SEC EDGAR for Form 4 purchase filings in the last N days.

    Returns {ticker: [{filer, date, title}]} for tickers in our universe.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": '"purchase"',
                "dateRange": "custom",
                "startdt": start_str,
                "enddt": end_str,
                "forms": "4",
            },
            headers={"User-Agent": "Kairos Trading Bot research@kairos.dev"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    SEC EDGAR error: {e}")
        return {}

    results: dict[str, list[dict]] = {}
    hits = data.get("hits", {}).get("hits", [])

    for hit in hits:
        source = hit.get("_source", {})
        display_names = source.get("display_names", [])
        file_date = source.get("file_date", "")
        entity_name = source.get("entity_name", "")

        # Extract ticker from display names or entity
        # EDGAR doesn't always have clean ticker — match against universe
        for name in display_names:
            name_upper = name.upper().strip()
            # Check if any universe ticker appears in the entity name
            for ticker in universe:
                if ticker in name_upper or name_upper.startswith(ticker + " "):
                    if ticker not in results:
                        results[ticker] = []
                    results[ticker].append({
                        "filer": entity_name[:60],
                        "date": file_date,
                    })
                    break

    # Deduplicate per ticker
    for ticker in results:
        seen = set()
        unique = []
        for entry in results[ticker]:
            key = (entry["filer"], entry["date"])
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        results[ticker] = unique[:5]  # Cap at 5 per ticker

    return results


# ═════════════════════════════════════════════════════════════════════
# 5. CONGRESSIONAL TRADING — House financial disclosure search
# ═════════════════════════════════════════════════════════════════════

def detect_congressional_trades(
    universe: set[str],
    lookback_days: int = 30,
) -> dict[str, list[dict]]:
    """Search for recent congressional stock purchases.

    Uses SEC EDGAR EFTS searching for Form 4 / periodic transaction
    reports by known congressional filers, plus House clerk disclosures.
    Returns {ticker: [{member, date, transaction_type}]}.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    results: dict[str, list[dict]] = {}

    # Try House clerk disclosure API
    try:
        year = end.year
        resp = requests.get(
            f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/",
            headers={"User-Agent": "Kairos Trading Bot research@kairos.dev"},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            _parse_house_disclosures(resp.text, universe, results, start)
    except Exception as e:
        print(f"    House disclosures error: {e}")

    # Fallback: SEC EDGAR search for political filer transaction reports
    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": '"purchase" "member" OR "representative" OR "senator"',
                "dateRange": "custom",
                "startdt": start_str,
                "enddt": end_str,
                "forms": "4",
            },
            headers={"User-Agent": "Kairos Trading Bot research@kairos.dev"},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            for hit in data.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                entity = source.get("entity_name", "")
                file_date = source.get("file_date", "")
                for name in source.get("display_names", []):
                    name_upper = name.upper().strip()
                    for ticker in universe:
                        if ticker in name_upper:
                            if ticker not in results:
                                results[ticker] = []
                            results[ticker].append({
                                "member": entity[:60],
                                "date": file_date,
                                "transaction_type": "Purchase",
                            })
                            break
    except Exception as e:
        print(f"    SEC EDGAR congressional search error: {e}")

    # Deduplicate
    for ticker in results:
        seen = set()
        unique = []
        for entry in results[ticker]:
            key = (entry.get("member", ""), entry.get("date", ""))
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        results[ticker] = unique[:5]

    return results


def _parse_house_disclosures(
    html: str,
    universe: set[str],
    results: dict[str, list[dict]],
    since: datetime,
) -> None:
    """Best-effort parse of House clerk disclosure index page.

    The page lists PDF links to periodic transaction reports. We extract
    member names and dates, then check if any known tickers appear in
    the filing titles. This is approximate — the PDFs aren't parsed.
    """
    import re

    # Pattern: links to PTR PDFs with member name and date in the URL/text
    # Format varies but commonly: LASTNAME-FirstName-YYYYMMDD.pdf
    for match in re.finditer(
        r'href="([^"]*\.pdf)"[^>]*>([^<]+)</a>', html, re.IGNORECASE
    ):
        url, text = match.groups()
        # Extract date from filename
        date_match = re.search(r'(\d{4})(\d{2})(\d{2})', url)
        if not date_match:
            continue
        try:
            filing_date = datetime(
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue

        if filing_date < since:
            continue

        member_name = text.strip()[:60]
        # We can't parse the PDF, but if the text mentions a ticker, flag it
        text_upper = text.upper()
        for ticker in universe:
            if ticker in text_upper:
                if ticker not in results:
                    results[ticker] = []
                results[ticker].append({
                    "member": member_name,
                    "date": filing_date.strftime("%Y-%m-%d"),
                    "transaction_type": "Purchase (from disclosure title)",
                })


# ═════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT FUNCTION — called by kairos_screener.py
# ═════════════════════════════════════════════════════════════════════

def run_all_signals(
    universe: set[str],
    all_scores: dict[str, str],
    api_key: str | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Run all five signal detectors and upgrade scores.

    Args:
        universe: set of all tickers being screened
        all_scores: current {ticker: score} from Ollama/reversion
        api_key: Finnhub API key (for earnings + RSI)
        dry_run: skip external API calls

    Returns:
        (updated_scores, signal_tags)
        signal_tags: {ticker: ["HOT-EARNINGS", "HOT-RSI", ...]}
    """
    print(banner("Signal Enrichment (5 detectors)"))

    signal_tags: dict[str, list[str]] = {}

    if dry_run:
        print("  DRY RUN — skipping signal detection")
        return all_scores, signal_tags

    # ── 1. Earnings surprise (one bulk Finnhub call) ─────────────────
    print("  [1/5] Earnings Surprise (Finnhub calendar)... ", end="", flush=True)
    earnings = detect_earnings_surprise(universe, api_key or "") if api_key else {}
    print(f"{len(earnings)} hit(s)")
    for t, info in earnings.items():
        signal_tags.setdefault(t, []).append("HOT-EARNINGS")
        print(f"    {t}: beat by {info['surprise_pct']}% "
              f"(actual={info['eps_actual']} vs est={info['eps_estimate']}) on {info['date']}")

    # ── 2. Oversold RSI (Finnhub candles, non-COLD only) ────────────
    # Only check tickers scored WARM or better to save API calls
    rsi_candidates = [t for t, s in all_scores.items()
                      if s not in ("COLD",) and t in universe][:60]
    print(f"  [2/5] Oversold RSI (checking {len(rsi_candidates)} non-COLD tickers)... ",
          end="", flush=True)
    oversold = detect_oversold_rsi(rsi_candidates, api_key or "") if api_key else {}
    print(f"{len(oversold)} hit(s)")
    for t, rsi in oversold.items():
        signal_tags.setdefault(t, []).append("HOT-RSI")
        print(f"    {t}: RSI={rsi:.1f} (oversold < 30)")

    # ── 3. Kalshi probability shift ──────────────────────────────────
    print("  [3/5] Kalshi Probability Shift... ", end="", flush=True)
    kalshi_shifts = detect_kalshi_shift()
    print(f"{len(kalshi_shifts)} shift(s)")
    for shift in kalshi_shifts:
        print(f"    {shift['title'][:50]}: {shift['old_prob']:.0f}% → {shift['new_prob']:.0f}% "
              f"({shift['direction']} {abs(shift['shift']):.0f}pp)")
    # Kalshi shifts are macro signals, not ticker-specific — they go into
    # the prompt context but don't upgrade individual ticker scores

    # ── 4. Insider buying (SEC EDGAR) ────────────────────────────────
    print("  [4/5] Insider Buying (SEC EDGAR Form 4)... ", end="", flush=True)
    insiders = detect_insider_buying(universe)
    print(f"{len(insiders)} ticker(s)")
    for t, filings in insiders.items():
        signal_tags.setdefault(t, []).append("HOT-INSIDER")
        for f in filings[:2]:
            print(f"    {t}: {f['filer']} on {f['date']}")

    # ── 5. Congressional trading ─────────────────────────────────────
    print("  [5/5] Congressional Trading (House/SEC)... ", end="", flush=True)
    congress = detect_congressional_trades(universe)
    print(f"{len(congress)} ticker(s)")
    for t, filings in congress.items():
        signal_tags.setdefault(t, []).append("HOT-CONGRESS")
        for f in filings[:2]:
            print(f"    {t}: {f.get('member', '?')} on {f.get('date', '?')}")

    # ── Upgrade scores for any ticker that fired a signal ────────────
    upgraded = 0
    for ticker, tags in signal_tags.items():
        current = all_scores.get(ticker, "COLD")
        if current == "COLD":
            # Promote to the first signal tag (e.g. HOT-EARNINGS)
            all_scores[ticker] = tags[0]
            upgraded += 1
        elif current == "WARM":
            all_scores[ticker] = tags[0]
            upgraded += 1
        # If already HOT or HOT-REVERSION, keep that score but record the tags

    if upgraded:
        print(f"\n  Upgraded {upgraded} ticker(s) based on signal detection")

    # Store Kalshi shifts and signal tags in a summary dict for the prompt
    signal_summary = {
        "signal_tags": signal_tags,
        "kalshi_shifts": kalshi_shifts,
        "earnings_hits": {t: info for t, info in earnings.items()},
        "insider_hits": {t: filings for t, filings in insiders.items()},
        "congress_hits": {t: filings for t, filings in congress.items()},
        "rsi_hits": oversold,
    }

    # Save for downstream prompt building
    summary_path = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(signal_summary, f, indent=2, default=str)
    except IOError:
        pass

    return all_scores, signal_tags
