"""
Kairos IPO Signals — Post-IPO Momentum Boost

Treats a freshly-listed ticker as a higher-conviction trade for its first
five trading days. Sources truth from two places:

  1. kairos_ipo_cache.json — written by kairos_ipo_intake. The `detected`
     map records when a watchlist ticker first started trading.
  2. kairos_universe.json — any Tier B entry written by ipo_intake
     (carries `source: ipo_intake` + `added_date`).

Public API:
    IPO_HOLD_DAYS               5-day eligibility window
    get_ipo_tickers()           Recent IPO catalog
    is_recent_ipo(ticker)       Bool gate for boost
    get_ipo_context(ticker)     Price/momentum dict for prompt + checkpoints
    score_ipo_momentum(ticker)  {signal, conviction_boost, size_multiplier,
                                 context}
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

SCRIPT_DIR = "/Users/jelmore/TradingAgents"

CACHE_FILE = os.path.join(SCRIPT_DIR, "kairos_ipo_cache.json")
UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "kairos_universe.json")

IPO_HOLD_DAYS = 5
IPO_SIGNAL_TAG = "IPO_MOMENTUM"
IPO_CONVICTION_BOOST = 1.5
IPO_SIZE_MULTIPLIER = 1.5

_DATE_FMTS = (
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S UTC",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


# ── Date helpers ─────────────────────────────────────────────────────

def _parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _days_since(dt: datetime, now: Optional[datetime] = None) -> int:
    now = now or datetime.utcnow()
    delta = now - dt
    return max(0, int(delta.total_seconds() // 86400))


# ── Catalog ───────────────────────────────────────────────────────────

def _load_cache() -> dict:
    """Best-effort read of kairos_ipo_cache.json. Empty dict on failure."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: IPO cache write failed: {exc}")


def _load_universe() -> dict:
    try:
        with open(UNIVERSE_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def get_ipo_tickers() -> dict[str, dict]:
    """All currently-recent IPO tickers, keyed by symbol.

    Sources:
      1. Cache: kairos_ipo_cache.json `detected` map — each entry has
         {detected_at, name, added}.
      2. Universe: any tier_b entry with `source == "ipo_intake"` and
         `added_date` within IPO_HOLD_DAYS.

    Each value includes:
        {
            "detected_date": "YYYY-MM-DD",
            "company_name": "...",
            "sector": "...",
            "days_since_ipo": int,
        }
    """
    now = datetime.utcnow()
    out: dict[str, dict] = {}

    # Cache-sourced detections
    cache = _load_cache()
    detected = cache.get("detected") or {}
    if isinstance(detected, dict):
        for ticker, meta in detected.items():
            if not isinstance(meta, dict):
                continue
            ts = meta.get("detected_at") or ""
            dt = _parse_date(ts)
            if dt is None:
                continue
            days = _days_since(dt, now)
            if days > IPO_HOLD_DAYS:
                continue
            out[ticker.upper()] = {
                "detected_date": dt.strftime("%Y-%m-%d"),
                "company_name": meta.get("name") or "",
                "sector": meta.get("sector") or "",
                "days_since_ipo": days,
            }

    # Universe-sourced (tier_b entries promoted by ipo_intake)
    universe = _load_universe()
    for entry in universe.get("tier_b", {}).get("tickers", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("source") != "ipo_intake":
            continue
        added = entry.get("added_date") or ""
        dt = _parse_date(added)
        if dt is None:
            continue
        days = _days_since(dt, now)
        if days > IPO_HOLD_DAYS:
            continue
        ticker = (entry.get("symbol") or "").upper()
        if not ticker:
            continue
        # Prefer cache record if richer; otherwise add a new row
        if ticker not in out:
            out[ticker] = {
                "detected_date": dt.strftime("%Y-%m-%d"),
                "company_name": entry.get("name") or "",
                "sector": entry.get("sector") or "",
                "days_since_ipo": days,
            }
        else:
            # Fill in sector/name from universe if cache lacked them
            out[ticker]["company_name"] = (
                out[ticker]["company_name"] or entry.get("name") or ""
            )
            out[ticker]["sector"] = (
                out[ticker]["sector"] or entry.get("sector") or ""
            )

    return out


def is_recent_ipo(ticker: str) -> bool:
    if not ticker:
        return False
    catalog = get_ipo_tickers()
    rec = catalog.get(ticker.upper())
    if rec is None:
        return False
    return rec["days_since_ipo"] <= IPO_HOLD_DAYS


# ── Context (price + momentum) ───────────────────────────────────────

def _ipo_price_from_cache(ticker: str) -> Optional[float]:
    """Pull a cached IPO price if previously stored."""
    cache = _load_cache()
    rec = (cache.get("detected") or {}).get(ticker.upper())
    if not isinstance(rec, dict):
        return None
    for key in ("ipo_price", "offer_price", "price"):
        val = rec.get(key)
        if val is None:
            continue
        try:
            v = float(val)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _fetch_day1_history(ticker: str) -> Optional[dict]:
    """Return {date, open, high, low, close} for the first listed day.

    Uses yfinance's history(period="max") and picks the earliest row.
    Returns None if yfinance is unavailable or the ticker has no data.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    try:
        first = df.iloc[0]
        return {
            "date": str(df.index[0].date()),
            "open": float(first.get("Open") or 0),
            "high": float(first.get("High") or 0),
            "low": float(first.get("Low") or 0),
            "close": float(first.get("Close") or 0),
        }
    except Exception:
        return None


def _fetch_current_price(ticker: str) -> Optional[float]:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    for key in ("regularMarketPrice", "currentPrice", "previousClose"):
        v = info.get(key)
        if v is None:
            continue
        try:
            f = float(v)
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return None


def get_ipo_context(ticker: str) -> dict:
    """Pull IPO-specific context for the reasoning prompt + checkpoints.

    All fields are best-effort — fields that can't be derived are set
    to None or 0.0 so callers can render the dict without guarding
    every key.
    """
    ticker = ticker.upper()
    catalog = get_ipo_tickers()
    rec = catalog.get(ticker) or {}

    day1 = _fetch_day1_history(ticker)
    ipo_price = _ipo_price_from_cache(ticker)
    if ipo_price is None and day1 is not None:
        ipo_price = day1.get("open") or None

    day1_open = day1.get("open") if day1 else None
    day1_close = day1.get("close") if day1 else None
    day1_gain_pct = None
    if day1_open and day1_close and day1_open > 0:
        day1_gain_pct = (day1_close - day1_open) / day1_open * 100.0

    current_price = _fetch_current_price(ticker)
    current_vs_ipo_pct = None
    if current_price is not None and ipo_price and ipo_price > 0:
        current_vs_ipo_pct = (current_price - ipo_price) / ipo_price * 100.0

    return {
        "ticker": ticker,
        "ipo_price": ipo_price,
        "day1_open": day1_open,
        "day1_close": day1_close,
        "day1_gain_pct": day1_gain_pct,
        "current_price": current_price,
        "current_vs_ipo_pct": current_vs_ipo_pct,
        "company_name": rec.get("company_name") or "",
        "sector": rec.get("sector") or "",
        "days_since_ipo": rec.get("days_since_ipo", 0),
        "detected_date": rec.get("detected_date") or "",
    }


# ── Scoring ──────────────────────────────────────────────────────────

def score_ipo_momentum(ticker: str) -> dict:
    """Return the IPO boost payload for the given ticker.

    If the ticker is not a recent IPO, returns a neutral payload
    (boost=1.0, multiplier=1.0, signal=None) so callers can apply it
    unconditionally.
    """
    if not is_recent_ipo(ticker):
        return {
            "signal": None,
            "conviction_boost": 1.0,
            "size_multiplier": 1.0,
        }
    return {
        "signal": IPO_SIGNAL_TAG,
        "conviction_boost": IPO_CONVICTION_BOOST,
        "size_multiplier": IPO_SIZE_MULTIPLIER,
        "context": get_ipo_context(ticker),
    }


# ── Force-screen helpers (used by the screener) ──────────────────────

def get_forced_screen_count(ticker: str) -> int:
    """Read `forced_screens[ticker]` from kairos_ipo_cache.json (0 if absent)."""
    cache = _load_cache()
    forced = cache.get("forced_screens") or {}
    if not isinstance(forced, dict):
        return 0
    try:
        return int(forced.get(ticker.upper(), 0))
    except (TypeError, ValueError):
        return 0


def bump_forced_screen(ticker: str) -> int:
    """Increment the forced-screen counter for a ticker. Returns new value."""
    cache = _load_cache()
    forced = cache.get("forced_screens")
    if not isinstance(forced, dict):
        forced = {}
        cache["forced_screens"] = forced
    key = ticker.upper()
    new_val = int(forced.get(key, 0)) + 1
    forced[key] = new_val
    _save_cache(cache)
    return new_val


def format_ipo_prompt_block(tickers: list[str]) -> str:
    """Build the IPO context block to prepend to the reasoning prompt.

    Returns an empty string when no shortlisted ticker is a recent IPO.
    """
    lines: list[str] = []
    for t in tickers:
        if not is_recent_ipo(t):
            continue
        ctx = get_ipo_context(t)
        ipo_price = ctx.get("ipo_price")
        day1_gain = ctx.get("day1_gain_pct")
        cur_vs_ipo = ctx.get("current_vs_ipo_pct")
        ipo_price_str = f"${ipo_price:.2f}" if ipo_price else "n/a"
        day1_str = f"{day1_gain:.1f}%" if day1_gain is not None else "n/a"
        cur_str = (
            f"{cur_vs_ipo:+.1f}%" if cur_vs_ipo is not None else "n/a"
        )
        lines.append(
            f"IPO CONTEXT for {t}: Listed {ctx.get('days_since_ipo', 0)} "
            f"days ago. IPO price: {ipo_price_str}. Day-one gain: "
            f"{day1_str}. Currently {cur_str} vs IPO price. "
            f"ARK/institutional buying: check HOT signals."
        )
    if not lines:
        return ""
    header = (
        "================================================================\n"
        "RECENT IPO CONTEXT (high-conviction boost applies)\n"
        "================================================================\n"
        "The following shortlisted tickers IPO'd within the last "
        f"{IPO_HOLD_DAYS} trading days. For each, apply a "
        f"~{IPO_CONVICTION_BOOST:.1f}x conviction boost and treat "
        f"position size as ~{IPO_SIZE_MULTIPLIER:.1f}x normal "
        "(encode this by raising the conviction value in your JSON).\n"
    )
    return header + "\n".join(lines) + "\n\n"
