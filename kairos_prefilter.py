"""
Kairos Tier 0 Rules-Based Pre-Filter

A fast, rules-based filter that runs BEFORE the Phi-4 LLM screener.
Reduces the full ticker universe to an active candidate pool,
keeping cycle time stable as the universe grows.

Filters applied (all must pass):
  - 30-day average volume >= min_30day_avg_volume (default: 500,000)
  - Price >= min_price (default: $5.00)
  - Market cap >= min_market_cap (default: $1,000,000,000)
  - Absolute 5-day price change >= min_5day_price_change_pct (default: 1.5%)
  - Today's volume >= min_today_volume_ratio * 30-day avg volume (default: 50%)

All thresholds are read from kairos_config.json ["tier0_filter"].

Usage:
  from kairos_prefilter import run_tier0_filter
  passing_tickers = run_tier0_filter(all_tickers, api_key, dry_run=False)
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
FINNHUB_TIMEOUT = 10
W = 72

# Configure module-level logger
logger = logging.getLogger("kairos_prefilter")


def banner(text: str) -> str:
    return f"\n{'─' * W}\n  {text}\n{'─' * W}"


def load_config() -> dict:
    """Load tier0_filter config from kairos_config.json."""
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    
    tier0 = config.get("tier0_filter", {})
    
    # Validate required thresholds
    required_keys = [
        "min_30day_avg_volume",
        "min_price",
        "min_market_cap",
        "min_5day_price_change_pct",
        "min_today_volume_ratio",
    ]
    
    missing = [k for k in required_keys if k not in tier0]
    if missing:
        raise ValueError(
            f"Missing required tier0_filter config keys: {missing}. "
            f"Please add them to {CONFIG_FILE}."
        )
    
    return tier0


# ── Data fetching ─────────────────────────────────────────────────────

def fetch_quote(ticker: str, api_key: str) -> dict | None:
    """Fetch quote data from Finnhub. Returns dict with c, v, dp, etc."""
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": ticker, "token": api_key},
            timeout=FINNHUB_TIMEOUT,
        )
        if resp.status_code == 429:
            return None
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        # Finnhub returns zeroes for invalid tickers
        if data.get("c", 0) == 0 and data.get("pc", 0) == 0:
            return None
        return data
    except Exception:
        return None


def fetch_profile2(ticker: str, api_key: str) -> dict | None:
    """Fetch profile2 data from Finnhub. Returns dict with marketCapitalization, avgVolume."""
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": ticker, "token": api_key},
            timeout=FINNHUB_TIMEOUT,
        )
        if resp.status_code == 429:
            return None
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception:
        return None


def fetch_5day_candle(ticker: str, api_key: str) -> list[float] | None:
    """Fetch 5 days of closing prices. Returns list of closes (oldest first) or None."""
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (6 * 86400)  # 6 days back to get 5 trading days
    
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={
                "symbol": ticker,
                "resolution": "D",
                "from": start,
                "to": now,
                "token": api_key,
            },
            timeout=FINNHUB_TIMEOUT,
        )
        if resp.status_code == 429:
            return None
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        closes = data.get("c", [])
        if not closes or data.get("s") == "no_data":
            return None
        # Filter out None/null values
        closes = [float(c) for c in closes if c is not None and c > 0]
        return closes if len(closes) >= 2 else None
    except Exception:
        return None


def fetch_batch_data_yfinance(tickers: list[str], period: str = "1mo") -> dict[str, dict]:
    """Bulk Tier 0 data fetch via a single yfinance.download() call.

    Replaces the per-ticker Finnhub profile2/candle calls (see the
    tier0_filter._comment in kairos_config.json — profile2 was returning
    missing market_cap/avg_volume for the whole universe, and separately
    the /stock/candle endpoint was returning no data at all under this
    Finnhub plan). One bulk call gets price, today's volume, a genuine
    30-day average volume, and the 5-day close history all from data
    already proven fast (~20s for 622 tickers) and reliable.

    No market cap here — it isn't in yfinance's bulk price history and
    requires a per-ticker info() call, which is the same "N synchronous
    calls" shape that broke Finnhub. min_market_cap is 0 (disabled) as of
    this writing, so check_thresholds' zero-threshold handling makes a
    missing market_cap a no-op rather than an automatic fail. If a real
    market-cap floor is wanted later, that's worth adding as a deliberate,
    separate (slower) enrichment step — not bundled in here.

    Returns {ticker: ticker_data} in the same shape fetch_ticker_data
    produces, so run_tier0_filter / check_thresholds work unchanged.
    """
    import yfinance as yf

    results: dict[str, dict] = {}
    if not tickers:
        return results

    df = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, threads=True)
    if df is None or df.empty:
        return {t: {"ticker": t, "quote": None, "profile": None, "candle": None}
                for t in tickers}

    multi = hasattr(df.columns, "levels") and "Close" in df.columns.get_level_values(0)

    def _series(field: str, t: str):
        try:
            if multi:
                if t not in df[field].columns:
                    return None
                s = df[field][t].dropna()
            else:
                if field not in df.columns:
                    return None
                s = df[field].dropna()
            return s if len(s) else None
        except Exception:
            return None

    for t in tickers:
        closes = _series("Close", t)
        vols = _series("Volume", t)
        if closes is None or len(closes) < 2:
            results[t] = {"ticker": t, "quote": None, "profile": None, "candle": None}
            continue

        price = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
        today_vol = float(vols.iloc[-1]) if vols is not None else None
        avg_vol_30d = float(vols.mean()) if vols is not None else None
        candle = [float(c) for c in closes.tail(5).tolist()]
        dp = ((price - prev_close) / prev_close * 100) if prev_close else None

        results[t] = {
            "ticker": t,
            "quote": {"c": price, "v": today_vol, "d": None, "dp": dp},
            "profile": {"avgVolume": avg_vol_30d, "marketCapitalization": None},
            "candle": candle if len(candle) >= 2 else None,
        }

    return results


def fetch_ticker_data(ticker: str, api_key: str) -> dict:
    """Fetch all data needed for Tier 0 filtering for one ticker.
    Returns a dict with all available data. Missing fields will be None.
    """
    quote = fetch_quote(ticker, api_key)
    profile = fetch_profile2(ticker, api_key)
    candle = fetch_5day_candle(ticker, api_key)
    
    return {
        "ticker": ticker,
        "quote": quote,
        "profile": profile,
        "candle": candle,
    }


# ── Filter logic ───────────────────────────────────────────────────────

def check_thresholds(ticker_data: dict, config: dict) -> tuple[bool, dict]:
    """Check if a ticker passes all Tier 0 thresholds.
    
    Args:
        ticker_data: dict with quote, profile, candle data
        config: tier0_filter config dict
    
    Returns:
        tuple of (passes: bool, details: dict)
        details contains the actual values and which checks failed
    """
    ticker = ticker_data["ticker"]
    quote = ticker_data.get("quote", {})
    profile = ticker_data.get("profile", {})
    candle = ticker_data.get("candle", [])
    
    # Guard against None values from failed fetches
    if quote is None:
        quote = {}
    if profile is None:
        profile = {}
    if candle is None:
        candle = []
    
    details = {
        "ticker": ticker,
        "passes": True,
        "price": None,
        "market_cap": None,
        "avg_volume_30d": None,
        "today_volume": None,
        "volume_ratio": None,
        "price_change_5d_pct": None,
        "failures": [],
    }
    
    # Extract values
    price = quote.get("c")
    today_volume = quote.get("v")
    avg_volume_30d = profile.get("avgVolume")
    market_cap = profile.get("marketCapitalization")
    
    # Compute 5-day price change
    price_change_5d_pct = None
    if candle and len(candle) >= 2:
        oldest = candle[0]
        newest = candle[-1]
        if oldest and newest and oldest > 0:
            price_change_5d_pct = abs((newest - oldest) / oldest) * 100
    
    # Update details with actual values
    details["price"] = price
    details["market_cap"] = market_cap
    details["avg_volume_30d"] = avg_volume_30d
    details["today_volume"] = today_volume
    
    if today_volume and avg_volume_30d and avg_volume_30d > 0:
        details["volume_ratio"] = today_volume / avg_volume_30d
    
    details["price_change_5d_pct"] = price_change_5d_pct
    
    # Check each threshold. A threshold of 0 means "this dimension is
    # disabled" — missing data on a disabled dimension is a no-op, not an
    # automatic fail. (Previously every check unconditionally failed on
    # missing data regardless of threshold, which is what let one broken
    # Finnhub field — market_cap — silently reject the entire universe even
    # though min_market_cap was already set to 0. See tier0_filter._comment
    # in kairos_config.json.)

    # Check 1: price >= min_price
    if price is None:
        if config["min_price"] != 0:
            details["passes"] = False
            details["failures"].append("price: missing data")
    elif price < config["min_price"]:
        details["passes"] = False
        details["failures"].append(f"price: price >= ${config['min_price']} (actual: {price})")
    
    # Check 2: market_cap >= min_market_cap
    if market_cap is None:
        if config["min_market_cap"] != 0:
            details["passes"] = False
            details["failures"].append("market_cap: missing data")
    elif market_cap < config["min_market_cap"]:
        details["passes"] = False
        details["failures"].append(f"market_cap: market cap >= ${config['min_market_cap']:,.0f} (actual: {market_cap:,.0f})")
    
    # Check 3: avg_volume_30d >= min_30day_avg_volume
    if avg_volume_30d is None:
        if config["min_30day_avg_volume"] != 0:
            details["passes"] = False
            details["failures"].append("avg_volume_30d: missing data")
    elif avg_volume_30d < config["min_30day_avg_volume"]:
        details["passes"] = False
        details["failures"].append(f"avg_volume_30d: 30-day avg volume >= {config['min_30day_avg_volume']:,.0f} (actual: {avg_volume_30d:,.0f})")
    
    # Check 4: volume_ratio >= min_today_volume_ratio
    if details["volume_ratio"] is None:
        if config["min_today_volume_ratio"] != 0:
            details["passes"] = False
            details["failures"].append("volume_ratio: missing data")
    elif details["volume_ratio"] < config["min_today_volume_ratio"]:
        details["passes"] = False
        details["failures"].append(f"volume_ratio: today's volume >= {config['min_today_volume_ratio']*100:.0f}% of 30-day avg (actual: {details['volume_ratio']*100:.1f}%)")
    
    # Check 5: price_change_5d_pct >= min_5day_price_change_pct
    if price_change_5d_pct is None:
        if config["min_5day_price_change_pct"] != 0:
            details["passes"] = False
            details["failures"].append("price_change_5d_pct: missing data")
    elif price_change_5d_pct < config["min_5day_price_change_pct"]:
        details["passes"] = False
        details["failures"].append(f"price_change_5d_pct: abs 5-day price change >= {config['min_5day_price_change_pct']:.1f}% (actual: {price_change_5d_pct:.2f}%)")
    
    return details["passes"], details


# ── Batch fetching and filtering ──────────────────────────────────────

def fetch_batch_data(tickers: list[str], api_key: str, max_workers: int = 8) -> dict[str, dict]:
    """Fetch data for a batch of tickers in parallel.
    
    Returns {ticker: ticker_data} for all tickers.
    """
    results = {}
    
    # Process in waves to respect Finnhub rate limit (60/min)
    wave_size = 55
    for wave_start in range(0, len(tickers), wave_size):
        wave = tickers[wave_start:wave_start + wave_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_ticker_data, t, api_key): t for t in wave}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    results[ticker] = fut.result()
                except Exception as e:
                    logger.warning(f"Error fetching data for {ticker}: {e}")
                    results[ticker] = {"ticker": ticker, "quote": None, "profile": None, "candle": None}
        
        # Pause for rate limit if more waves remain
        remaining = len(tickers) - (wave_start + wave_size)
        if remaining > 0:
            logger.info(f"  Rate limit pause (fetched {len(results)}/{len(tickers)} tickers)...")
            time.sleep(62)
    
    return results


def run_tier0_filter(
    tickers: list[str],
    api_key: str = None,
    dry_run: bool = False,
    return_data: bool = False,
    data_source: str = "yfinance",
):
    """Run Tier 0 rules-based pre-filter on a list of tickers.

    Args:
        tickers: List of ticker symbols to filter
        api_key: Finnhub API key. Only required when data_source="finnhub".
        dry_run: If True, skip the data fetch and use placeholder data
        return_data: If True, also return a quote_data_dict mapping
            ticker -> {"price": float, "change_pct": float, "volume_signal": str}
            for every ticker with a usable quote. Lets downstream callers
            reuse pre-filter quote data instead of a second fetch.
        data_source: "yfinance" (default) — one bulk call, no per-ticker
            requests, no Finnhub dependency for this stage. "finnhub" is kept
            for comparison/testing only; it's the path that broke (see
            tier0_filter._comment in kairos_config.json) and requires api_key.

    Returns:
        If return_data is False (default): list of tickers that pass thresholds.
        If return_data is True: tuple (passing_list, quote_data_dict).

    Logs:
        - How many tickers in
        - How many passed
        - How many filtered
        - Warnings for tickers with missing data
    """
    config = load_config()
    
    logger.info(banner("Tier 0 Pre-Filter"))
    logger.info(f"  Input: {len(tickers)} tickers")
    logger.info(f"  Thresholds: 30d avg vol >= {config['min_30day_avg_volume']:,}, "
                f"price >= ${config['min_price']}, "
                f"market cap >= ${config['min_market_cap']:,.0f}, "
                f"5d change >= {config['min_5day_price_change_pct']}%, "
                f"today vol >= {config['min_today_volume_ratio']*100:.0f}% of 30d avg")
    
    if dry_run:
        logger.info("  DRY RUN — skipping Finnhub, using placeholder data")
        # In dry run mode, return all tickers (or apply simple filters on placeholder data)
        # For testing purposes, just return all tickers
        logger.info(f"  Passed: {len(tickers)} tickers (dry run bypass)")
        logger.info(f"  Filtered: 0 tickers")
        if return_data:
            return tickers, {}
        return tickers
    
    if not api_key and data_source == "finnhub":
        raise ValueError("FINNHUB_API_KEY is required when data_source='finnhub'. "
                          "Set FINNHUB_API_KEY environment variable or pass api_key, "
                          "or use data_source='yfinance' (default).")
    
    # Fetch data for all tickers
    logger.info(f"  Fetching data for {len(tickers)} tickers via {data_source}...")
    t0 = time.time()
    if data_source == "yfinance":
        all_data = fetch_batch_data_yfinance(tickers)
    else:
        all_data = fetch_batch_data(tickers, api_key)
    fetch_time = time.time() - t0
    logger.info(f"  Data fetched in {fetch_time:.1f}s")
    
    # Apply filters
    passing = []
    filtered = []
    missing_data_count = 0
    quote_data_dict: dict[str, dict] = {}

    for ticker in tickers:
        ticker_data = all_data.get(ticker)
        if not ticker_data:
            logger.warning(f"  WARNING: No data fetched for {ticker} — skipping")
            missing_data_count += 1
            filtered.append(ticker)
            continue

        # Capture quote data for any ticker with a valid Finnhub quote so
        # callers can reuse it instead of issuing a second batch fetch.
        quote = ticker_data.get("quote")
        if quote:
            price = quote.get("c")
            dp = quote.get("dp")
            if price is not None and dp is not None:
                quote_data_dict[ticker] = {
                    "price": float(price),
                    "change_pct": float(dp),
                    "volume_signal": "high" if abs(float(dp)) > 2 else "normal",
                }

        passes, details = check_thresholds(ticker_data, config)

        if passes:
            passing.append(ticker)
        else:
            filtered.append(ticker)
            if "missing data" in " ".join(details["failures"]):
                missing_data_count += 1
                # Log individual missing data warnings
                for failure in details["failures"]:
                    if "missing data" in failure:
                        logger.warning(f"  WARNING: {ticker} — {failure}")
    
    # Summary
    passed_count = len(passing)
    filtered_count = len(filtered)
    logger.info(f"  Results: {passed_count}/{len(tickers)} passed, {filtered_count} filtered out")
    
    if missing_data_count > 0:
        logger.info(f"  Tickers with missing data: {missing_data_count}")
    
    if passing:
        logger.info(f"  Passing tickers: {', '.join(sorted(passing)[:20])}" +
                    (f" ... ({len(passing) - 20} more)" if len(passing) > 20 else ""))

    if return_data:
        return passing, quote_data_dict
    return passing


# ── CLI entry point for testing ───────────────────────────────────────

def main():
    """CLI entry point for testing the pre-filter."""
    import argparse
    
    # Set up basic logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Kairos Tier 0 Pre-Filter")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Finnhub, use placeholder data")
    parser.add_argument("--universe", type=str, default="kairos_universe.json",
                        help="Path to universe JSON file (default: kairos_universe.json)")
    args = parser.parse_args()
    
    # Load universe
    from kairos_screener import load_universe
    tickers = load_universe()
    
    api_key = os.environ.get("FINNHUB_API_KEY")
    
    passing = run_tier0_filter(
        tickers=tickers,
        api_key=api_key,
        dry_run=args.dry_run
    )
    
    print(f"\n{'═' * W}")
    print(f"  Tier 0 Pre-Filter Complete: {len(passing)}/{len(tickers)} tickers passed")
    print(f"{'═' * W}")


if __name__ == "__main__":
    main()
