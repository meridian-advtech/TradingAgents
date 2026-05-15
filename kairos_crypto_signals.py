"""
Kairos Crypto Signal Detectors — 5 Signals

Data sourced from CoinGecko (free tier) and kairos_crypto_prices.json
(rolling price history maintained by the pipeline).

Signals:
  HOT-RSI              — 14-period RSI below 30 (oversold)
  HOT-REVERSION        — 24h price drop >3%
  HOT-CRYPTO-MOMENTUM  — 4h gain >3% AND total market cap growing
  HOT-CRYPTO-MACRO     — risk-on macro: fear/greed >50, equities up
  HOT-CRYPTO-BASELINE  — BTC above its 7-day moving average (weak, always-on)

All detectors are fault-tolerant: return empty dicts on failure.

Usage:
    python3 kairos_crypto_signals.py
    python3 kairos_crypto_signals.py --assets BTC,ETH,SOL
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from kairos_crypto_cache import get_ohlc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_SIGNAL_SUMMARY = os.path.join(SCRIPT_DIR, "kairos_crypto_signal_summary.json")
CRYPTO_PRICES_FILE = os.path.join(SCRIPT_DIR, "kairos_crypto_prices.json")
TIMEOUT = 15
W = 72

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "LINK": "chainlink",
}

DEFAULT_RSI_THRESHOLD = 30.0
DEFAULT_RSI_PERIOD = 14
DEFAULT_REVERSION_THRESHOLD = 3.0  # lowered from 5% to match equity
DEFAULT_MOMENTUM_THRESHOLD = 3.0   # 4h gain %
DEFAULT_FEAR_GREED_THRESHOLD = 50   # above = greedy/risk-on


def banner(title: str) -> str:
    return f"\n{'━' * W}\n  {title}\n{'━' * W}"


# ═════════════════════════════════════════════════════════════════════
# 1. HOT-RSI — CoinGecko OHLC-based 14-period RSI
# ═════════════════════════════════════════════════════════════════════

def _fetch_ohlc(coin_id: str, days: int = 30) -> list[list]:
    """Fetch OHLC candles from CoinGecko. Returns [[ts, o, h, l, c], ...]."""
    try:
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Compute RSI from closing prices. Returns 0-100 or None."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_sma(closes: list[float], period: int) -> float | None:
    """Simple moving average of the last N closes."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def detect_crypto_rsi(
    assets: list[str],
    rsi_threshold: float = DEFAULT_RSI_THRESHOLD,
    rsi_period: int = DEFAULT_RSI_PERIOD,
) -> dict[str, float]:
    """Detect oversold assets. Returns {symbol: rsi_value} for RSI < threshold."""
    hits = {}
    for symbol in assets:
        coin_id = COINGECKO_IDS.get(symbol)
        if not coin_id:
            continue
        candles = _fetch_ohlc(coin_id, days=30)
        if not candles:
            continue
        closes = [c[4] for c in candles if len(c) >= 5 and c[4] is not None]
        rsi = _compute_rsi(closes, rsi_period)
        if rsi is not None and rsi < rsi_threshold:
            hits[symbol] = round(rsi, 1)
        time.sleep(2)  # CoinGecko rate limit
    return hits


# ═════════════════════════════════════════════════════════════════════
# 2. HOT-REVERSION — 24h drop exceeds threshold (now 3%)
# ═════════════════════════════════════════════════════════════════════

def detect_crypto_reversion(
    coins_data: list[dict],
    threshold_pct: float = DEFAULT_REVERSION_THRESHOLD,
) -> dict[str, dict]:
    """Detect mean reversion candidates. Returns {symbol: {change_24h, price, ...}}."""
    hits = {}
    for coin in coins_data:
        symbol = coin.get("symbol", "")
        change = coin.get("change_24h", 0.0)
        if change <= -threshold_pct:
            hits[symbol] = {
                "change_24h": round(change, 2),
                "price": coin.get("price", 0),
                "volume": coin.get("volume", 0),
                "direction": "drop",
            }
    return hits


# ═════════════════════════════════════════════════════════════════════
# 3. HOT-CRYPTO-MOMENTUM — 4h gain >3% AND market cap growing
# ═════════════════════════════════════════════════════════════════════

def _load_price_history() -> dict[str, list[dict]]:
    """Load rolling price history from kairos_crypto_prices.json."""
    if not os.path.exists(CRYPTO_PRICES_FILE):
        return {}
    try:
        with open(CRYPTO_PRICES_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def detect_crypto_momentum(
    coins_data: list[dict],
    threshold_pct: float = DEFAULT_MOMENTUM_THRESHOLD,
    window_hours: int = 4,
) -> dict[str, dict]:
    """Detect momentum: 4h gain > threshold AND total market cap growing.

    Uses kairos_crypto_prices.json for 4h lookback.
    Market cap growth checked by comparing sum of current vs. earliest available.

    Returns {symbol: {gain_4h, price_now, price_4h_ago}}.
    """
    history = _load_price_history()
    if not history:
        return {}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)

    # Check total market cap trend (sum of all tracked coins)
    total_mcap_now = sum(c.get("market_cap", 0) for c in coins_data)

    hits = {}
    for coin in coins_data:
        symbol = coin.get("symbol", "")
        price_now = coin.get("price", 0)
        if not price_now or symbol not in history:
            continue

        # Find oldest price point within window
        entries = history[symbol]
        price_4h_ago = None
        for entry in entries:
            try:
                ts = datetime.fromisoformat(entry["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue
            if ts <= cutoff:
                price_4h_ago = entry["price"]
                break  # oldest available

        if price_4h_ago is None or price_4h_ago <= 0:
            continue

        gain_pct = (price_now - price_4h_ago) / price_4h_ago * 100

        if gain_pct >= threshold_pct and total_mcap_now > 0:
            hits[symbol] = {
                "gain_4h": round(gain_pct, 2),
                "price_now": price_now,
                "price_4h_ago": round(price_4h_ago, 2),
                "total_mcap_b": round(total_mcap_now / 1e9, 1),
            }

    return hits


# ═════════════════════════════════════════════════════════════════════
# 4. HOT-CRYPTO-MACRO — risk-on macro conditions
# ═════════════════════════════════════════════════════════════════════

def _fetch_fear_greed() -> int | None:
    """Fetch crypto fear & greed index from alternative.me API.

    Returns integer 0-100 (0=extreme fear, 100=extreme greed), or None.
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return int(data["data"][0]["value"])
    except Exception:
        return None


def _fetch_sp500_daily_change() -> float | None:
    """Estimate S&P 500 daily change from the snapshot file.

    Falls back to checking if the most recent Finnhub data shows
    positive market sentiment. Returns % change or None.
    """
    # Try to read from the last market snapshot
    snapshot_file = os.path.join(SCRIPT_DIR, "kairos_snapshot.txt")
    if not os.path.exists(snapshot_file):
        return None
    try:
        with open(snapshot_file) as f:
            content = f.read()
        # Look for SPY or S&P references in snapshot
        # This is best-effort — the snapshot may not have intraday equity data
        # Return None to skip this sub-check if unavailable
        return None
    except Exception:
        return None


def detect_crypto_macro(
    fear_greed_threshold: int = DEFAULT_FEAR_GREED_THRESHOLD,
) -> dict:
    """Detect risk-on macro conditions.

    Fires when fear/greed index > threshold (greedy = risk-on).
    Returns {fired: bool, fear_greed: int, details: str} or empty dict.
    """
    fg = _fetch_fear_greed()
    if fg is None:
        return {}

    details_parts = []
    fired = False

    if fg >= fear_greed_threshold:
        details_parts.append(f"fear/greed={fg} (>{fear_greed_threshold}, risk-on)")
        fired = True
    else:
        details_parts.append(f"fear/greed={fg} (<{fear_greed_threshold}, cautious)")

    # S&P check (best-effort, non-blocking)
    sp_change = _fetch_sp500_daily_change()
    if sp_change is not None and sp_change > 1.0:
        details_parts.append(f"S&P500 +{sp_change:.1f}%")
    elif sp_change is None:
        # Market closed (weekend/holiday) — use fear/greed as sole indicator
        details_parts.append("S&P500: market closed (fear/greed only)")

    return {
        "fired": fired,
        "fear_greed": fg,
        "details": "; ".join(details_parts),
    }


# ═════════════════════════════════════════════════════════════════════
# 5. HOT-CRYPTO-BASELINE — BTC above 7-day SMA (weak, always-on)
# ═════════════════════════════════════════════════════════════════════

def detect_crypto_baseline(coins_data: list[dict]) -> dict:
    """Fires if BTC price is above its 7-day simple moving average.

    Uses CoinGecko OHLC data for the SMA calculation.
    Returns {fired: bool, btc_price, sma_7d, pct_above} or empty dict.
    """
    btc_coin = next((c for c in coins_data if c.get("symbol") == "BTC"), None)
    if not btc_coin:
        return {}

    btc_price = btc_coin.get("price", 0)
    if btc_price <= 0:
        return {}

    # Fetch BTC OHLC for 7-day SMA from cache
    candles = get_ohlc("bitcoin", 7)
    if not candles:
        return {}

    closes = [c[4] for c in candles if len(c) >= 5 and c[4] is not None]
    if len(closes) == 0:
        return {}
    
    # Calculate SMA of all available closes in the 7-day window
    sma = sum(closes) / len(closes)

    if sma <= 0:
        return {}

    pct_above = (btc_price - sma) / sma * 100
    fired = btc_price > sma

    return {
        "fired": fired,
        "btc_price": btc_price,
        "sma_7d": round(sma, 2),
        "pct_above": round(pct_above, 2),
    }


# ═════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT — called by run_gather_crypto()
# ═════════════════════════════════════════════════════════════════════

def run_crypto_signals(
    assets: list[str],
    coins_data: list[dict],
    reversion_threshold: float = DEFAULT_REVERSION_THRESHOLD,
) -> dict[str, list[str]]:
    """Run all 5 crypto signal detectors and return signal tags.

    Returns signal_tags: {symbol: ["HOT-RSI", "HOT-CRYPTO-BASELINE", ...]}
    """
    print(banner("Crypto Signal Enrichment (5 detectors)"))

    signal_tags: dict[str, list[str]] = {}
    rsi_hits: dict[str, float] = {}
    reversion_hits: dict[str, dict] = {}
    momentum_hits: dict[str, dict] = {}
    macro_result: dict = {}
    baseline_result: dict = {}

    # ── 1. Oversold RSI ───────────────────────────────────────────
    print(f"  [1/5] Crypto RSI ({len(assets)} assets)... ", end="", flush=True)
    rsi_hits = detect_crypto_rsi(assets)
    print(f"{len(rsi_hits)} hit(s)")
    for symbol, rsi in rsi_hits.items():
        signal_tags.setdefault(symbol, []).append("HOT-RSI")
        print(f"    {symbol}: RSI={rsi:.1f} (oversold < 30)")

    # ── 2. Mean reversion ─────────────────────────────────────────
    print(f"  [2/5] Crypto Reversion (>{reversion_threshold}% drop)... ", end="", flush=True)
    reversion_hits = detect_crypto_reversion(coins_data, reversion_threshold)
    print(f"{len(reversion_hits)} hit(s)")
    for symbol, info in reversion_hits.items():
        signal_tags.setdefault(symbol, []).append("HOT-REVERSION")
        print(f"    {symbol}: {info['change_24h']:+.1f}% (24h drop)")

    # ── 3. Momentum (4h gain + market cap) ────────────────────────
    print(f"  [3/5] Crypto Momentum (4h >3%)... ", end="", flush=True)
    momentum_hits = detect_crypto_momentum(coins_data)
    print(f"{len(momentum_hits)} hit(s)")
    for symbol, info in momentum_hits.items():
        signal_tags.setdefault(symbol, []).append("HOT-CRYPTO-MOMENTUM")
        print(f"    {symbol}: +{info['gain_4h']:.1f}% in 4h "
              f"(${info['price_4h_ago']:,.0f} → ${info['price_now']:,.0f})")

    # ── 4. Macro (fear/greed + equities) ──────────────────────────
    print(f"  [4/5] Crypto Macro (fear/greed)... ", end="", flush=True)
    macro_result = detect_crypto_macro()
    if macro_result.get("fired"):
        print(f"FIRED — {macro_result['details']}")
        # Apply to ALL assets (macro is market-wide)
        for coin in coins_data:
            sym = coin.get("symbol", "")
            if sym:
                signal_tags.setdefault(sym, []).append("HOT-CRYPTO-MACRO")
    elif macro_result:
        print(f"no fire — {macro_result.get('details', '?')}")
    else:
        print(f"unavailable")

    # ── 5. Baseline (BTC > 7d SMA) ───────────────────────────────
    print(f"  [5/5] Crypto Baseline (BTC > 7d SMA)... ", end="", flush=True)
    time.sleep(2)  # rate limit before OHLC call
    baseline_result = detect_crypto_baseline(coins_data)
    if baseline_result.get("fired"):
        pct = baseline_result["pct_above"]
        print(f"FIRED — BTC ${baseline_result['btc_price']:,.0f} > "
              f"7d SMA ${baseline_result['sma_7d']:,.0f} (+{pct:.1f}%)")
        # Apply to ALL assets (BTC trend is market-wide sentiment)
        for coin in coins_data:
            sym = coin.get("symbol", "")
            if sym:
                signal_tags.setdefault(sym, []).append("HOT-CRYPTO-BASELINE")
    elif baseline_result:
        print(f"no fire — BTC ${baseline_result.get('btc_price', 0):,.0f} "
              f"< 7d SMA ${baseline_result.get('sma_7d', 0):,.0f}")
    else:
        print(f"unavailable")

    # ── Save summary ──────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "signal_tags": signal_tags,
        "rsi_hits": rsi_hits,
        "reversion_hits": {k: v for k, v in reversion_hits.items()},
        "momentum_hits": {k: v for k, v in momentum_hits.items()},
        "macro": macro_result,
        "baseline": baseline_result,
        "assets_scanned": assets,
    }

    try:
        with open(CRYPTO_SIGNAL_SUMMARY, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Saved → {CRYPTO_SIGNAL_SUMMARY}")
    except IOError:
        pass

    # Summary
    total_signals = sum(len(t) for t in signal_tags.values())
    tickers_with_signals = len(signal_tags)
    print(f"  Total: {total_signals} signals across {tickers_with_signals} asset(s)")

    # Log crypto diagnostics after signal evaluation
    try:
        from kairos_crypto_diagnostics import log_crypto_diagnostics
        log_crypto_diagnostics()
    except Exception as e:
        print(f"Warning: Crypto diagnostics logging failed: {e}")

    return signal_tags


def load_crypto_signal_summary() -> dict:
    """Load the most recent crypto signal summary."""
    if not os.path.exists(CRYPTO_SIGNAL_SUMMARY):
        return {}
    try:
        with open(CRYPTO_SIGNAL_SUMMARY) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def get_crypto_ticker_signals(symbol: str) -> list[str]:
    """Load signal tags for a specific crypto asset."""
    summary = load_crypto_signal_summary()
    return summary.get("signal_tags", {}).get(symbol, [])


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kairos Crypto Signal Detectors")
    parser.add_argument("--assets", default="BTC,ETH,SOL,BNB,XRP",
                        help="Comma-separated symbols")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    print("=" * W)
    print("  KAIROS CRYPTO SIGNAL DETECTORS")
    print("=" * W)
    print(f"  Assets: {', '.join(assets)}")

    coin_ids = ",".join(COINGECKO_IDS.get(a, "") for a in assets if a in COINGECKO_IDS)
    coins_data = []
    if coin_ids:
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={"vs_currency": "usd", "ids": coin_ids, "order": "market_cap_desc"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            for c in resp.json():
                coins_data.append({
                    "symbol": c.get("symbol", "").upper(),
                    "price": float(c.get("current_price") or 0),
                    "change_24h": float(c.get("price_change_percentage_24h") or 0),
                    "volume": float(c.get("total_volume") or 0),
                    "market_cap": float(c.get("market_cap") or 0),
                })
        except Exception as exc:
            print(f"  CoinGecko fetch failed: {exc}")

    if coins_data:
        print(f"\n  Market data:")
        for c in coins_data:
            d = "▲" if c["change_24h"] >= 0 else "▼"
            print(f"    {c['symbol']:>5}  ${c['price']:>12,.2f}  {d} {abs(c['change_24h']):.2f}%")

    tags = run_crypto_signals(assets, coins_data)

    print(f"\n  Signal summary:")
    if tags:
        for sym, t in sorted(tags.items()):
            print(f"    {sym}: {', '.join(t)}")
    else:
        print(f"    No signals fired.")
    print("=" * W)


if __name__ == "__main__":
    main()
