#!/usr/bin/env python3
"""
Crypto API Cache — Reduces CoinGecko API calls by caching responses.

Centralizes all CoinGecko API calls to enforce rate limits and cache responses.
Cached data is valid for 5 minutes (300 seconds) to stay within free tier limits.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

CACHE_FILE = os.path.join(os.path.dirname(__file__), "kairos_crypto_cache.json")
CACHE_EXPIRY_SECONDS = 300  # 5 minutes


def _load_cache() -> Dict[str, Any]:
    """Load cache from file, return empty dict if not exists or invalid."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Save cache to file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass  # Silent failure - cache is optional


def _is_cache_valid(cached_item: Optional[Dict[str, Any]]) -> bool:
    """Check if cached item is still valid (not expired)."""
    if not cached_item or 'timestamp' not in cached_item:
        return False
    
    cache_age = time.time() - cached_item['timestamp']
    return cache_age < CACHE_EXPIRY_SECONDS


def _make_api_call(url: str, params: Dict[str, Any], cache_key: str) -> Optional[Any]:
    """Make API call with rate limit handling and caching."""
    # Try to return cached data first
    cache = _load_cache()
    cached_item = cache.get(cache_key)
    
    if _is_cache_valid(cached_item):
        return cached_item['data']
    
    # Make fresh API call
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Update cache
            cache[cache_key] = {
                'timestamp': time.time(),
                'data': data
            }
            _save_cache(cache)
            return data
        elif response.status_code == 429:
            # Rate limited - return stale cache if available
            if cached_item:
                return cached_item['data']
            return None
        else:
            return None
    except Exception:
        # Return stale cache if available
        if cached_item:
            return cached_item['data']
        return None


def get_simple_price(ids: str) -> Optional[Dict[str, Any]]:
    """Get simple price for one or more coins (comma-separated)."""
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': ids, 'vs_currencies': 'usd'}
    cache_key = f'simple_price_{ids}'
    return _make_api_call(url, params, cache_key)


def get_ohlc(coin_id: str, days: int) -> Optional[List[List[Any]]]:
    """Get OHLC candles for a coin."""
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc'
    params = {'vs_currency': 'usd', 'days': str(days)}
    cache_key = f'ohlc_{coin_id}_{days}d'
    return _make_api_call(url, params, cache_key)


def get_fear_greed_index() -> Optional[str]:
    """Get fear & greed index."""
    url = 'https://api.alternative.me/fng/'
    params = {'limit': '1'}
    cache_key = 'fear_greed'
    result = _make_api_call(url, params, cache_key)
    return result


def get_spy_data() -> Optional[Dict[str, Any]]:
    """Get S&P 500 market data (placeholder for now)."""
    # This would need a proper market data API
    # For now, return None to use the fallback logic
    return None