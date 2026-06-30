"""
Kairos Tier 1 Screener — Two-Tier Screening Pipeline

Uses Ollama (llama3.2, local) to screen the broad universe from
kairos_universe.json. For each batch of tickers, Ollama evaluates
basic signals (price momentum, volume, news mentions via Finnhub)
and scores each as HOT, WARM, or COLD.

Only HOT and WARM tickers pass to Tier 2 (Claude reasoning).
Max 15 tickers forwarded to keep Claude's reasoning focused.

Usage:
  python kairos_screener.py                # Full screen with Finnhub data
  python kairos_screener.py --dry-run      # Score without Finnhub (fast test)
  python kairos_screener.py --max-tier2 10 # Cap shortlist at 10
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "kairos_universe.json")
TIER_C_FILE = os.path.join(SCRIPT_DIR, "kairos_tier_c.json")
SCREEN_LOG = os.path.join(SCRIPT_DIR, "kairos_screen_log.json")
FINNHUB_TIMEOUT = 10
BATCH_SIZE = 20        # tickers per Ollama call
OLLAMA_PARALLEL = 4    # concurrent Ollama batch requests
MAX_TIER2_DEFAULT = 15 # max tickers forwarded to Claude
W = 72


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Universe loader ──────────────────────────────────────────────────

def load_universe() -> list[str]:
    """Load and deduplicate all equity/ETF tickers from kairos_universe.json.

    Returns a flat sorted list (Tier A + Tier B) for backward compatibility.
    """
    _tier_a, _tier_b, _ = load_universe_tiered()
    return sorted(set(_tier_a + _tier_b))


def load_universe_tiered() -> tuple[list[str], list[str], list[str]]:
    """Load tickers grouped by tier from kairos_universe.json.

    Returns (tier_a_tickers, tier_b_tickers, crypto_tickers).
    Tier A = S&P 500 equities + ETFs.  Tier B = high-growth watchlist.
    """
    with open(UNIVERSE_FILE, "r") as f:
        data = json.load(f)

    tier_a = set()
    tier_b = set()
    crypto = []

    # Tier A — equities + ETFs
    tier_a_data = data.get("tier_a", {})
    for _cat, symbols in tier_a_data.get("equities", {}).items():
        for s in symbols:
            tier_a.add(s)
    for _cat, symbols in tier_a_data.get("etfs", {}).items():
        for s in symbols:
            tier_a.add(s)

    # Tier B — high-growth watchlist
    for entry in data.get("tier_b", {}).get("tickers", []):
        sym = entry["symbol"] if isinstance(entry, dict) else entry
        tier_b.add(sym)

    # Crypto (separate pipeline)
    crypto_section = data.get("crypto", {})
    if isinstance(crypto_section, dict):
        crypto = crypto_section.get("tickers", [])
    elif isinstance(crypto_section, list):
        crypto = crypto_section

    return sorted(tier_a), sorted(tier_b), list(crypto)


# ── Finnhub data fetcher ────────────────────────────────────────────

def fetch_finnhub_quote(ticker: str, api_key: str) -> dict | None:
    """Fetch quote data for a single ticker from Finnhub.

    Returns: {"c": close, "d": change, "dp": change_pct, "h": high,
              "l": low, "o": open, "pc": prev_close, "t": timestamp}
    or None on failure.
    """
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": ticker, "token": api_key},
            timeout=FINNHUB_TIMEOUT,
        )
        if resp.status_code == 429:
            return None  # rate limited
        resp.raise_for_status()
        data = resp.json()
        # Finnhub returns zeroes for invalid tickers
        if data.get("c", 0) == 0 and data.get("pc", 0) == 0:
            return None
        return data
    except Exception:
        return None


def fetch_quotes_yfinance(tickers: list[str]) -> dict[str, dict]:
    """Bulk-fetch quotes via a single yfinance.download() call.

    Returns {ticker: {"c": price, "d": change, "dp": change_pct, "pc": prev_close}}
    in the same format as fetch_quotes_batch. Tickers with missing/NaN data
    are skipped.
    """
    import math
    import yfinance as yf

    t0 = time.time()
    results: dict[str, dict] = {}
    total = len(tickers)

    df = yf.download(
        tickers,
        period="2d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        elapsed = round(time.time() - t0, 1)
        print(f"  yfinance bulk fetch: 0/{total} quotes in {elapsed}s")
        return results

    # yfinance returns MultiIndex columns (field, ticker) for multi-ticker
    # downloads and flat columns for a single ticker.
    multi = hasattr(df.columns, "levels") and "Close" in df.columns.get_level_values(0)

    if multi:
        closes = df["Close"]
        for t in tickers:
            if t not in closes.columns:
                continue
            series = closes[t].dropna()
            if len(series) < 2:
                continue
            prev = float(series.iloc[-2])
            cur = float(series.iloc[-1])
            if math.isnan(prev) or math.isnan(cur) or prev == 0:
                continue
            change = cur - prev
            results[t] = {
                "c": cur,
                "d": change,
                "dp": change / prev * 100.0,
                "pc": prev,
            }
    elif "Close" in df.columns and len(tickers) == 1:
        series = df["Close"].dropna()
        if len(series) >= 2:
            prev = float(series.iloc[-2])
            cur = float(series.iloc[-1])
            if not (math.isnan(prev) or math.isnan(cur) or prev == 0):
                change = cur - prev
                results[tickers[0]] = {
                    "c": cur,
                    "d": change,
                    "dp": change / prev * 100.0,
                    "pc": prev,
                }

    elapsed = round(time.time() - t0, 1)
    print(f"  yfinance bulk fetch: {len(results)}/{total} quotes in {elapsed}s")
    return results


def fetch_quotes_batch(tickers: list[str], api_key: str,
                       max_workers: int = 8) -> dict[str, dict]:
    """Fetch quotes for many tickers in parallel.

    Returns {ticker: quote_dict} for tickers that returned valid data.
    Finnhub free tier: 60 calls/min — we throttle with small batches.
    """
    results = {}
    # Process in waves of 55 to stay under 60/min rate limit
    wave_size = 55
    for wave_start in range(0, len(tickers), wave_size):
        wave = tickers[wave_start:wave_start + wave_size]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_finnhub_quote, t, api_key): t for t in wave}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    data = fut.result()
                    if data:
                        results[ticker] = data
                except Exception:
                    pass

        # If more waves remain, pause for rate limit
        remaining = len(tickers) - (wave_start + wave_size)
        if remaining > 0:
            print(f"    Rate limit pause (fetched {len(results)}/{len(tickers)})...")
            time.sleep(62)

    return results


# ── Ollama batch scorer ─────────────────────────────────────────────

def _build_batch_prompt(batch: list[dict]) -> str:
    """Build an Ollama prompt to score a batch of tickers.

    Each item in batch: {"ticker": str, "change_pct": float, "volume_signal": str}
    """
    rows = []
    for item in batch:
        ticker = item["ticker"]
        pct = item.get("change_pct", 0.0)
        vol = item.get("volume_signal", "normal")
        price = item.get("price", 0.0)
        rows.append(f"  {ticker}: price=${price:.2f}, 24h_change={pct:+.2f}%, volume={vol}")

    ticker_block = "\n".join(rows)

    return f"""You are a stock screening assistant. Score each ticker as HOT, WARM, or COLD.

Scoring rules:
- HOT: absolute price change > 3%, OR unusual volume, OR strong directional move
- WARM: absolute price change 1-3%, moderate activity, worth watching
- COLD: absolute price change < 1%, low activity, no signal

Here are the tickers with their latest data:
{ticker_block}

Respond with ONLY a JSON array. Each element: {{"ticker": "...", "score": "HOT"|"WARM"|"COLD"}}
No explanation, no markdown fences, just the JSON array:"""


def score_batch_ollama(batch: list[dict]) -> dict[str, str]:
    """Send a batch to Ollama's screen model (speed-optimized).

    Returns {ticker: "HOT"|"WARM"|"COLD"}.
    Falls back to threshold-based scoring on Ollama failure.
    """
    from kairos_ollama import ask_screen, is_available, get_screen_model

    if not is_available(get_screen_model()):
        return _score_batch_fallback(batch)

    prompt = _build_batch_prompt(batch)
    raw = ask_screen(prompt, timeout=90)

    if not raw:
        return _score_batch_fallback(batch)

    # Parse JSON from response — handle common LLM quirks
    try:
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        # Find the JSON array
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            arr = json.loads(cleaned[start:end + 1])
            scores = {}
            for item in arr:
                if not isinstance(item, dict):
                    continue
                t = item.get("ticker", "").upper()
                raw_s = item.get("score", "COLD")
                if isinstance(raw_s, list):
                    raw_s = raw_s[0] if raw_s else "COLD"
                s = str(raw_s).strip().upper()
                if t and s in ("HOT", "HOT-REVERSION", "WARM", "COLD"):
                    scores[t] = s
            if scores:
                return scores
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    return _score_batch_fallback(batch)


def _score_batch_fallback(batch: list[dict]) -> dict[str, str]:
    """Threshold-based fallback when Ollama is unavailable."""
    scores = {}
    for item in batch:
        pct = abs(item.get("change_pct", 0.0))
        if pct >= 3.0:
            scores[item["ticker"]] = "HOT"
        elif pct >= 1.0:
            scores[item["ticker"]] = "WARM"
        else:
            scores[item["ticker"]] = "COLD"
    return scores


# ── Parallel batch scoring ───────────────────────────────────────────

def _score_single_batch(args: tuple) -> tuple[int, list[str], dict[str, str]]:
    """Score one batch. Returns (batch_index, batch_tickers, scores)."""
    idx, batch = args
    tickers = [b["ticker"] for b in batch]
    scores = score_batch_ollama(batch)
    # Fill any Ollama missed
    for item in batch:
        if item["ticker"] not in scores:
            scores[item["ticker"]] = _score_batch_fallback([item])[item["ticker"]]
    return idx, tickers, scores


def score_all_batches_parallel(batches: list[list[dict]]) -> dict[str, str]:
    """Score all batches with up to OLLAMA_PARALLEL concurrent requests."""
    all_scores: dict[str, str] = {}
    total = len(batches)

    with ThreadPoolExecutor(max_workers=OLLAMA_PARALLEL) as pool:
        futures = {
            pool.submit(_score_single_batch, (i, batch)): i
            for i, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            idx, tickers, scores = fut.result()
            all_scores.update(scores)
            hot_n = sum(1 for t in tickers if scores.get(t) in ("HOT", "HOT-REVERSION"))
            warm_n = sum(1 for t in tickers if scores.get(t) == "WARM")
            print(f"  Batch {idx+1}/{total}: {tickers[0]}..{tickers[-1]} "
                  f"→ {hot_n}H {warm_n}W")

    return all_scores


# ── Mean reversion detector ──────────────────────────────────────────

REVERSION_THRESHOLD_PCT = -3.0  # drop > 3% triggers HOT-REVERSION
# Entry runway gate: a >3% drop alone is not enough — the price must also sit
# meaningfully BELOW its 30d SMA so there is real room to revert UP to the mean.
# The exit (_validity_reversion) completes the trade at price >= 0.98 * 30d SMA,
# so we require entry at <= 0.97 * 30d SMA (ratio <= 0.97), leaving >1% of runway
# to that 0.98 sell trigger. Without this, uptrending names that dip >3% while
# still ABOVE their mean were bought above the SMA and exited "reversion complete"
# minutes later (e.g. EXPE entered +10.5% above its 30d SMA).
REVERSION_MIN_BELOW_SMA_PCT = 3.0

# Lazily bound to kairos_thesis_validity._close_series on first use. Kept at module
# level (rather than a function-local import) so it can be monkeypatched in tests;
# populated lazily to avoid a circular import at module load.
_close_series = None


def apply_reversion_signals(
    all_scores: dict[str, str],
    batch_items: list[dict],
) -> dict[str, str]:
    """Override scores to HOT-REVERSION for tickers that dropped > 3%.

    Applies to both equities and crypto. This runs AFTER Ollama scoring
    so the reversion signal takes priority regardless of Ollama's score.

    A >3% drop is necessary but NOT sufficient: we additionally require the price
    to sit at least REVERSION_MIN_BELOW_SMA_PCT below the 30d SMA (real runway to
    revert), confirmed per drop-candidate via _close_series. Candidates whose 30d
    SMA cannot be fetched are skipped conservatively (not flagged).
    """
    global _close_series
    if _close_series is None:
        # Lazy import (avoid circular import at module load).
        from kairos_thesis_validity import _close_series as _cs
        _close_series = _cs

    for item in batch_items:
        pct = item.get("change_pct", 0.0)
        if pct > REVERSION_THRESHOLD_PCT:
            continue
        ticker = item["ticker"]

        # Runway gate (candidate-only — we never fetch SMA for the whole universe).
        closes = _close_series(ticker, period="45d")
        if not closes:
            # Can't confirm runway → conservative skip (don't flag on the drop alone).
            print(f"  {ticker} dropped {pct:.1f}% but 30d SMA unavailable "
                  f"— no runway confirmable, skipped")
            continue
        window = closes[-30:] if len(closes) >= 30 else closes
        sma = sum(window) / len(window)
        price = closes[-1]

        if sma > 0 and price <= sma * (1 - REVERSION_MIN_BELOW_SMA_PCT / 100):
            all_scores[ticker] = "HOT-REVERSION"
        elif sma > 0:
            print(f"  {ticker} dropped {pct:.1f}% but only "
                  f"{(price / sma - 1) * 100:+.1f}% vs 30d SMA — no runway, skipped")
        else:
            print(f"  {ticker} dropped {pct:.1f}% but 30d SMA non-positive "
                  f"— no runway confirmable, skipped")
    return all_scores


# ── Main screener ───────────────────────────────────────────────────

def run_screen(dry_run: bool = False, max_tier2: int = MAX_TIER2_DEFAULT,
              skip_tier0: bool = False) -> dict:
    original_universe_size = 0
    """Run Tier 1 screening on the full universe.

    Args:
        dry_run: If True, skip Finnhub and use random-ish test data
        max_tier2: Maximum tickers to forward to Tier 2
        skip_tier0: If True, skip Tier 0 pre-filter (for debugging/testing)

    Returns:
        {
          "timestamp": str,
          "universe_size": int,
          "quotes_fetched": int,
          "scores": {ticker: "HOT"|"WARM"|"COLD"},
          "hot": [str],
          "warm": [str],
          "cold_count": int,
          "shortlist": [str],  # HOT + WARM, capped at max_tier2
          "tier2_count": int,
          "ollama_batches": int,
          "elapsed_sec": float,
        }
    """
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS TIER 1 SCREENER — {timestamp}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    # Tier C audit — check for expiry notifications before loading
    print(banner("Tier C Audit"))
    try:
        from kairos_tier_c import audit as tier_c_audit, load_tier_c_tickers
        tier_c_audit()
        tier_c_tickers = load_tier_c_tickers()
    except Exception as exc:
        print(f"  WARNING: Tier C audit failed: {exc}")
        tier_c_tickers = []

    # Load universe (tiered)
    print(banner("Loading Universe"))
    tier_a, tier_b, _crypto = load_universe_tiered()
    # Build tier lookup for source_tier tagging
    tier_map: dict[str, str] = {}
    for t in tier_a:
        tier_map[t] = "A"
    for t in tier_b:
        tier_map[t] = "B"
    for t in tier_c_tickers:
        tier_map[t] = "C"
    # Combined equity/ETF tickers for screening (Tier A + Tier B + Tier C together)
    tickers = sorted(set(tier_a + tier_b + tier_c_tickers))
    tier_c_count = len(tier_c_tickers)
    print(f"  Loaded {len(tickers)} equity/ETF tickers "
          f"(Tier A: {len(tier_a)}, Tier B: {len(tier_b)}, Tier C: {tier_c_count})")

    # Tier 0 Pre-Filter — rules-based filtering before LLM screening
    print(banner("Tier 0 Pre-Filter"))
    tier0_passing = tickers  # Default: all pass if no filtering
    tier0_filtered_count = 0
    # Quote data captured during Tier 0 — reused below to avoid a second Finnhub call.
    prefilter_quote_data: dict[str, dict] = {}

    if not skip_tier0:
        api_key = os.environ.get("FINNHUB_API_KEY")
        from kairos_prefilter import run_tier0_filter

        if dry_run:
            # In dry run, pre-filter also uses dry run mode
            tier0_passing = run_tier0_filter(tickers, api_key=None, dry_run=True)
            tier0_filtered_count = len(tickers) - len(tier0_passing)
            print(f"  DRY RUN — pre-filter bypassed, all {len(tier0_passing)} tickers passed")
        elif api_key:
            tier0_passing, prefilter_quote_data = run_tier0_filter(
                tickers, api_key=api_key, dry_run=False, return_data=True
            )
            tier0_filtered_count = len(tickers) - len(tier0_passing)
            print(f"  Pre-filter passed {len(tier0_passing)}/{len(tickers)} tickers "
                  f"(quote data captured for {len(prefilter_quote_data)} tickers)")
        else:
            print("  WARNING: FINNHUB_API_KEY not set — skipping Tier 0 pre-filter")
            tier0_passing = tickers
    else:
        print("  Tier 0 pre-filter skipped (--skip-tier0 flag)")

    # Update tickers to only those that passed Tier 0
    tickers = tier0_passing

    # Fetch market data
    print(banner("Fetching Market Data"))
    quotes: dict[str, dict] = {}

    if dry_run:
        print("  DRY RUN — skipping Finnhub, using zero-change placeholders")
        for t in tickers:
            quotes[t] = {"c": 100.0, "d": 0.0, "dp": 0.0, "pc": 100.0}
    else:
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            print("  WARNING: FINNHUB_API_KEY not set — using fallback scoring")
        else:
            # Reuse quote data captured during Tier 0 — only call Finnhub for any
            # tickers the pre-filter didn't already cover. Prevents a duplicate
            # 13-minute fetch that was getting rate-limited to 0/726 quotes.
            missing = [t for t in tickers if t not in prefilter_quote_data]
            reused = len(tickers) - len(missing)
            if missing:
                print(f"  Pre-filter quote data covers {reused}/{len(tickers)} tickers")
                print(f"  Fetching {len(missing)} missing tickers via yfinance...")
                try:
                    quotes = fetch_quotes_yfinance(missing)
                except Exception as exc:
                    print(f"  yfinance failed ({exc}) — falling back to Finnhub")
                    quotes = fetch_quotes_batch(missing, api_key)
                print(f"  Received quotes for {len(quotes)}/{len(missing)} fallback tickers")
            else:
                print(f"  Pre-filter quote data covers all {len(tickers)} tickers — "
                      f"skipping fallback Finnhub fetch")

    # Build batch items — prefer pre-filter data, then fallback quotes, then zeros
    batch_items = []
    for t in tickers:
        pf = prefilter_quote_data.get(t)
        if pf:
            batch_items.append({
                "ticker": t,
                "price": pf["price"],
                "change_pct": pf["change_pct"],
                "volume_signal": pf["volume_signal"],
            })
            continue

        q = quotes.get(t)
        if q:
            price = q.get("c", 0.0)
            dp = q.get("dp", 0.0)
            if dp is None:
                dp = 0.0
            batch_items.append({
                "ticker": t,
                "price": price,
                "change_pct": dp,
                "volume_signal": "high" if abs(dp) > 2 else "normal",
            })
        else:
            # No data — include with zero change (will score COLD)
            batch_items.append({
                "ticker": t,
                "price": 0.0,
                "change_pct": 0.0,
                "volume_signal": "unknown",
            })

    # Total tickers with usable quote data (pre-filter or fallback fetch)
    quotes_with_data = sum(
        1 for t in tickers if t in prefilter_quote_data or t in quotes
    )

    # Score in batches via Ollama (parallel)
    print(banner("Ollama Tier 1 Scoring"))
    batches = [batch_items[i:i + BATCH_SIZE] for i in range(0, len(batch_items), BATCH_SIZE)]
    print(f"  {len(batch_items)} tickers → {len(batches)} batches of {BATCH_SIZE}"
          f" ({OLLAMA_PARALLEL} parallel)")

    all_scores = score_all_batches_parallel(batches)

    # Apply mean reversion overrides (>3% drop → HOT-REVERSION)
    print(banner("Mean Reversion Detection"))
    pre_reversion = dict(all_scores)
    all_scores = apply_reversion_signals(all_scores, batch_items)
    reversion_tickers = sorted([
        t for t, s in all_scores.items()
        if s == "HOT-REVERSION" and pre_reversion.get(t) != "HOT-REVERSION"
    ])
    if reversion_tickers:
        print(f"  HOT-REVERSION flagged: {', '.join(reversion_tickers)}")
        for rt in reversion_tickers:
            item = next((b for b in batch_items if b["ticker"] == rt), None)
            if item:
                print(f"    {rt}: {item['change_pct']:+.2f}% (threshold: {REVERSION_THRESHOLD_PCT}%)")
    else:
        print("  No mean reversion signals detected (no drops > 3%)")

    # Run five signal detectors (earnings, RSI, Kalshi, insider, congress)
    from kairos_signals import run_all_signals
    api_key = os.environ.get("FINNHUB_API_KEY")
    all_scores, signal_tags = run_all_signals(
        universe=set(tickers),
        all_scores=all_scores,
        api_key=api_key,
        dry_run=dry_run,
    )

    # Tally results (after signal enrichment)
    hot_earnings = sorted([t for t, s in all_scores.items() if s == "HOT-EARNINGS"])
    hot_rsi = sorted([t for t, s in all_scores.items() if s == "HOT-RSI"])
    hot_insider = sorted([t for t, s in all_scores.items() if s == "HOT-INSIDER"])
    hot_congress = sorted([t for t, s in all_scores.items() if s == "HOT-CONGRESS"])
    hot = sorted([t for t, s in all_scores.items() if s == "HOT"])
    hot_reversion = sorted([t for t, s in all_scores.items() if s == "HOT-REVERSION"])
    warm = sorted([t for t, s in all_scores.items() if s == "WARM"])
    cold_count = sum(1 for s in all_scores.values() if s == "COLD")

    # All signal-flagged tickers (deduplicated, preserving priority order)
    all_hot_signals = []
    seen = set()
    for group in [hot_reversion, hot_earnings, hot_rsi, hot_insider, hot_congress, hot]:
        for t in group:
            if t not in seen:
                all_hot_signals.append(t)
                seen.add(t)

    # Build shortlist: signal HOTs first, then WARM, capped at max_tier2
    shortlist = all_hot_signals[:max_tier2]
    remaining_slots = max_tier2 - len(shortlist)
    if remaining_slots > 0:
        for t in warm:
            if t not in seen:
                shortlist.append(t)
                seen.add(t)
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break

    # ── IPO momentum boost (kairos_signals_ipo) ────────────────────
    # Effects:
    #   1. Any shortlisted ticker that's a recent IPO gets the
    #      IPO_MOMENTUM signal tag appended.
    #   2. FORCE-INCLUDE (bypassing HOT/WARM/COLD classification entirely),
    #      capped at IPO_FORCE_INCLUDE_CAP=5 per cycle:
    #        a. Tier B recent-IPO tickers (first 3 screening cycles).
    #        b. EDGAR scored_filings with total_score >= 70 AND live.
    #        c. detected IPO tickers added within the last 5 days.
    #      Counters persisted in kairos_ipo_cache.json under 'forced_screens'.
    IPO_FORCE_INCLUDE_CAP = 5
    ipo_forced_added: list[str] = []

    def _force_include_ipo(tk, bump_fn, tag_const):
        """Add an IPO ticker to the shortlist, bypassing classification.
        Returns True if newly added (respects the per-cycle cap)."""
        tk = (tk or "").upper()
        if not tk or tk in shortlist:
            return False
        if len(ipo_forced_added) >= IPO_FORCE_INCLUDE_CAP:
            return False
        shortlist.append(tk)
        seen.add(tk)
        tags = signal_tags.setdefault(tk, [])
        if tag_const not in tags:
            tags.append(tag_const)
        if bump_fn:
            try:
                bump_fn(tk)
            except Exception:
                pass
        ipo_forced_added.append(tk)
        return True

    try:
        from kairos_signals_ipo import (
            is_recent_ipo, get_ipo_tickers, get_forced_screen_count,
            bump_forced_screen, IPO_SIGNAL_TAG,
        )
        # Tag existing shortlist members
        for t in shortlist:
            if is_recent_ipo(t):
                tags = signal_tags.setdefault(t, [])
                if IPO_SIGNAL_TAG not in tags:
                    tags.append(IPO_SIGNAL_TAG)

        # (a) Tier B recent-IPO tickers — first 3 forced cycles each
        tier_b_set = set(tier_b)
        for ipo_ticker in get_ipo_tickers():
            if ipo_ticker in shortlist:
                bump_forced_screen(ipo_ticker)
                continue
            if ipo_ticker not in tier_b_set:
                continue
            if get_forced_screen_count(ipo_ticker) >= 3:
                continue
            _force_include_ipo(ipo_ticker, bump_forced_screen, IPO_SIGNAL_TAG)

        # (b) + (c) EDGAR high-conviction scored filings + recent detections.
        # These bypass Tier membership and HOT/WARM/COLD entirely.
        try:
            from kairos_ipo_intake import (
                _load_cache as _ipo_load_cache, check_ticker_live,
            )
            _ipo_cache = _ipo_load_cache()

            # (b) scored_filings with score >= 70 AND live
            for _acc, sf in (_ipo_cache.get("scored_filings") or {}).items():
                if len(ipo_forced_added) >= IPO_FORCE_INCLUDE_CAP:
                    break
                if int(sf.get("total_score") or 0) < 70:
                    continue
                tk = (sf.get("ticker") or "").upper()
                if not tk or tk in shortlist:
                    continue
                try:
                    if not check_ticker_live(tk):
                        continue
                except Exception:
                    continue
                _force_include_ipo(tk, bump_forced_screen, IPO_SIGNAL_TAG)

            # (c) detected within the last 5 days
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            for tk, meta in (_ipo_cache.get("detected") or {}).items():
                if len(ipo_forced_added) >= IPO_FORCE_INCLUDE_CAP:
                    break
                if not isinstance(meta, dict):
                    continue
                ts = meta.get("detected_at") or ""
                added_recently = False
                for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
                    try:
                        dt = datetime.strptime(ts[:19] if "T" in ts else ts[:10], fmt)
                        added_recently = (now - dt).days <= 5
                        break
                    except (ValueError, TypeError):
                        continue
                if added_recently:
                    _force_include_ipo(tk, bump_forced_screen, IPO_SIGNAL_TAG)
        except Exception as edgar_exc:
            print(f"  WARNING: EDGAR IPO force-include failed: {edgar_exc}")
    except Exception as ipo_exc:
        print(f"  WARNING: IPO momentum hook failed: {ipo_exc}")

    if ipo_forced_added:
        print(f"  IPO force-include ({len(ipo_forced_added)}/{IPO_FORCE_INCLUDE_CAP}): "
              f"{', '.join(ipo_forced_added)}")

    # ── Event-driven seasonality boost (kairos_signals_events) ─────
    # For each active event (AAPL/WWDC, NVDA/GTC, etc.) whose ticker is
    # in the universe but not yet on the shortlist, force-include it.
    # Capped at EVENT_FORCED_MAX_CYCLES per event, counter persisted in
    # kairos_ipo_cache.json under 'event_forced_screens'.
    event_forced_added: list[str] = []
    try:
        from kairos_signals_events import (
            get_active_events, get_event_force_count, bump_event_force_count,
            event_key, EVENT_FORCED_MAX_CYCLES,
        )
        active_events = get_active_events()
        universe_set = set(tier_a) | set(tier_b)

        for active in active_events:
            ticker = (active.get("ticker") or "").upper()
            if not ticker:
                continue
            key = event_key(active)

            # Determine signal tag from pattern + phase
            pattern = active.get("pattern", "dip_then_rip")
            phase = active.get("phase", "pre")
            if phase == "pre":
                tag = ("PRE-EVENT-DIP" if pattern == "dip_then_rip"
                       else "PRE-EVENT-BUILDUP")
            else:
                tag = "POST-EVENT-REBOUND"

            # Always tag the ticker on the shortlist if it's already there
            if ticker in shortlist:
                tags = signal_tags.setdefault(ticker, [])
                if tag not in tags:
                    tags.append(tag)
                continue

            if ticker not in universe_set:
                continue
            if get_event_force_count(key) >= EVENT_FORCED_MAX_CYCLES:
                continue

            shortlist.append(ticker)
            seen.add(ticker)
            tags = signal_tags.setdefault(ticker, [])
            if tag not in tags:
                tags.append(tag)
            bump_event_force_count(key)
            event_forced_added.append(f"{ticker}({tag})")
    except Exception as event_exc:
        print(f"  WARNING: Event seasonality hook failed: {event_exc}")

    if event_forced_added:
        print(f"  Event force-include: {', '.join(event_forced_added)}")

    # ── AI value-chain leading-indicator boost (kairos_signals_chain) ──
    # When a Tier 1 linchpin (NVDA, MSFT, …) makes a large move, downstream
    # Tier 2/3 suppliers that haven't repriced yet are flagged HOT-CHAIN.
    #   1. Any shortlisted ticker that's HOT-CHAIN gets the tag appended.
    #   2. Up to 3 HOT-CHAIN tickers that are in the universe but off the
    #      shortlist are force-included this cycle.
    chain_forced_added: list[str] = []
    chain_conviction_boosts: dict[str, float] = {}
    try:
        from kairos_signals_chain import (
            run_chain_signal_scan, get_chain_tier, CHAIN_SIGNAL_TAG,
        )
        chain_hot = set()
        for sig in run_chain_signal_scan():
            ct = (sig.get("ticker") or "").upper()
            if ct:
                chain_hot.add(ct)

        # conviction_boost by chain tier (Tier 2 → 1.2x, Tier 3 → 1.3x)
        def _set_chain_boost(tkr: str) -> None:
            tier = get_chain_tier(tkr)
            if tier == 2:
                chain_conviction_boosts[tkr] = 1.2
            elif tier == 3:
                chain_conviction_boosts[tkr] = 1.3

        # Tag existing shortlist members
        for t in shortlist:
            if t.upper() in chain_hot:
                tags = signal_tags.setdefault(t, [])
                if CHAIN_SIGNAL_TAG not in tags:
                    tags.append(CHAIN_SIGNAL_TAG)
                _set_chain_boost(t)

        # Force-include up to 3 universe tickers not already shortlisted
        chain_universe_set = set(tier_a) | set(tier_b)
        for ct in sorted(chain_hot):
            if len(chain_forced_added) >= 3:
                break
            if ct in shortlist or ct not in chain_universe_set:
                continue
            shortlist.append(ct)
            seen.add(ct)
            tags = signal_tags.setdefault(ct, [])
            if CHAIN_SIGNAL_TAG not in tags:
                tags.append(CHAIN_SIGNAL_TAG)
            _set_chain_boost(ct)
            chain_forced_added.append(ct)
    except Exception as chain_exc:
        print(f"  WARNING: Chain signal hook failed: {chain_exc}")

    if chain_forced_added:
        print(f"  Chain force-include: {', '.join(chain_forced_added)}")

    elapsed = round(time.time() - t0, 1)

    # Print summary
    print(banner("Screening Results"))
    print(f"  Universe:       {original_universe_size} tickers loaded")
    print(f"  Tier 0 filtered: {tier0_filtered_count} tickers removed")
    print(f"  After Tier 0:    {len(tickers)} tickers screened")
    print(f"  Quotes:         {quotes_with_data} tickers with data "
          f"({len(prefilter_quote_data)} reused from pre-filter, {len(quotes)} fallback)")
    signal_counts = [
        ("HOT-REVERSION", hot_reversion), ("HOT-EARNINGS", hot_earnings),
        ("HOT-RSI", hot_rsi), ("HOT-INSIDER", hot_insider),
        ("HOT-CONGRESS", hot_congress), ("HOT", hot),
    ]
    for label, group in signal_counts:
        if group:
            print(f"  {label + ':':18s}{len(group)} — {', '.join(group[:10])}")
    print(f"  {'WARM:':18s}{len(warm)} tickers")
    print(f"  {'COLD:':18s}{cold_count} tickers (filtered out)")
    print(f"  {'Shortlist:':18s}{len(shortlist)} → Tier 2 (max {max_tier2})")
    print(f"  {'Elapsed:':18s}{elapsed}s")
    print(f"\n  Tier 2 shortlist: {', '.join(shortlist) if shortlist else '(empty — all COLD)'}")

    # Build source_tier map for each scored ticker
    source_tiers = {t: tier_map.get(t, "A") for t in all_scores}

    # Tier breakdown stats
    tier_a_screened = sum(1 for t in all_scores if tier_map.get(t) == "A")
    tier_b_screened = sum(1 for t in all_scores if tier_map.get(t) == "B")
    tier_c_screened = sum(1 for t in all_scores if tier_map.get(t) == "C")
    all_hot_set = set(hot + hot_reversion + hot_earnings + hot_rsi + hot_insider + hot_congress)
    tier_a_hot_warm = sum(1 for t in (all_hot_set | set(warm)) if tier_map.get(t) == "A")
    tier_b_hot_warm = sum(1 for t in (all_hot_set | set(warm)) if tier_map.get(t) == "B")
    tier_c_hot_warm = sum(1 for t in (all_hot_set | set(warm)) if tier_map.get(t) == "C")

    print(f"\n  Tier breakdown: A={tier_a_screened} screened ({tier_a_hot_warm} HOT/WARM), "
          f"B={tier_b_screened} screened ({tier_b_hot_warm} HOT/WARM), "
          f"C={tier_c_screened} screened ({tier_c_hot_warm} HOT/WARM)")

    # Build result
    # Note: tier0_input_count is the original universe size before filtering
    # We need to track this before tickers was overwritten
    # Actually, we should track the original count separately
    # Let's fix this by storing the original count
    
    # Get original universe size (before Tier 0 filtering)
    # This is stored in tier_a, tier_b, tier_c_tickers
    original_universe_size = len(tier_a) + len(tier_b) + tier_c_count
    
    result = {
        "timestamp": timestamp,
        "universe_size": original_universe_size,
        "tier0_filtered_count": tier0_filtered_count,
        "post_tier0_count": len(tickers),
        "quotes_fetched": quotes_with_data,
        "scores": all_scores,
        "source_tiers": source_tiers,
        "signal_tags": signal_tags,
        "conviction_boosts": chain_conviction_boosts,
        "hot": hot,
        "hot_reversion": hot_reversion,
        "hot_earnings": hot_earnings,
        "hot_rsi": hot_rsi,
        "hot_insider": hot_insider,
        "hot_congress": hot_congress,
        "warm": warm,
        "cold_count": cold_count,
        "shortlist": shortlist,
        "tier2_count": len(shortlist),
        "ollama_batches": len(batches),
        "elapsed_sec": elapsed,
        "tier_a_count": len(tier_a),
        "tier_b_count": len(tier_b),
        "tier_c_count": tier_c_count,
        "tier_a_screened": tier_a_screened,
        "tier_b_screened": tier_b_screened,
        "tier_c_screened": tier_c_screened,
        "tier_a_hot_warm": tier_a_hot_warm,
        "tier_b_hot_warm": tier_b_hot_warm,
        "tier_c_hot_warm": tier_c_hot_warm,
    }

    # Log to file
    _log_screen(result)

    return result


def _log_screen(result: dict):
    """Append screening result to kairos_screen_log.json."""
    log_entry = {
        "timestamp": result["timestamp"],
        "universe_size": result["universe_size"],
        "tier0_filtered_count": result.get("tier0_filtered_count", 0),
        "post_tier0_count": result.get("post_tier0_count", result["universe_size"]),
        "quotes_fetched": result["quotes_fetched"],
        "tier_breakdown": {
            "tier_a_screened": result.get("tier_a_screened", 0),
            "tier_b_screened": result.get("tier_b_screened", 0),
            "tier_c_screened": result.get("tier_c_screened", 0),
            "tier_a_hot_warm": result.get("tier_a_hot_warm", 0),
            "tier_b_hot_warm": result.get("tier_b_hot_warm", 0),
            "tier_c_hot_warm": result.get("tier_c_hot_warm", 0),
        },
        "signals": {
            "hot_reversion": result.get("hot_reversion", []),
            "hot_earnings": result.get("hot_earnings", []),
            "hot_rsi": result.get("hot_rsi", []),
            "hot_insider": result.get("hot_insider", []),
            "hot_congress": result.get("hot_congress", []),
            "hot": result["hot"],
        },
        "warm_count": len(result["warm"]),
        "cold_count": result["cold_count"],
        "tier2_count": result["tier2_count"],
        "shortlist": result["shortlist"],
        "source_tiers": {t: result.get("source_tiers", {}).get(t, "A") for t in result["shortlist"]},
        "elapsed_sec": result["elapsed_sec"],
    }

    # Append to log (one JSON object per line)
    with open(SCREEN_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\n  Screen log → {SCREEN_LOG}")


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kairos Tier 1 Screener")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Finnhub API calls, use placeholder data")
    parser.add_argument("--max-tier2", type=int, default=MAX_TIER2_DEFAULT,
                        help=f"Max tickers for Tier 2 (default: {MAX_TIER2_DEFAULT})")
    parser.add_argument("--skip-tier0", action="store_true",
                        help="Skip Tier 0 pre-filter (for debugging)")
    args = parser.parse_args()

    result = run_screen(dry_run=args.dry_run, max_tier2=args.max_tier2,
                        skip_tier0=args.skip_tier0)

    print("\n" + "━" * W)
    tier0_msg = f", {result['tier0_filtered_count']} filtered by Tier 0" if result.get('tier0_filtered_count', 0) > 0 else ""
    print(f"  Screener complete. {result['tier2_count']}/{result['post_tier0_count']} "
          f"passed to Tier 2{tier0_msg}.")
    print("━" * W)


if __name__ == "__main__":
    main()
