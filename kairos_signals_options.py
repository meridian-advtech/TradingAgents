"""
Kairos Options Activity Detector — HOT-OPTIONS Signal

Detects unusual options activity on shortlisted tickers using IBKR's
native options data via ib_insync.  Runs during Phase 1 (gather) since
it reuses the existing IBKR connection.

Three independent sub-signals (any one triggers HOT-OPTIONS):

  vol_spike    — volume / open_interest ratio > 3.0 on near-term ATM chain
  iv_rank      — current IV vs. recent range > 80th percentile
  skew         — call_vol / put_vol ratio > 2.5 (bullish) or < 0.4 (bearish)

Designed for the final 15-ticker shortlist only to stay within IBKR's
~50 simultaneous market data subscription limit.  Each ticker is
processed sequentially: qualify chain → fetch ATM data → release → next.

Usage (standalone test):
    python3 kairos_signals_options.py           # Requires IBKR on port 7497
    python3 kairos_signals_options.py --tickers NET,FDX,PLTR
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONS_CACHE = os.path.join(SCRIPT_DIR, "kairos_options_activity.json")

W = 72

# Tickers we watch especially closely — relaxed volume-spike threshold
# (1.5x instead of 2.0x) so smaller unusual moves still flag, and kairos_run
# runs a second yfinance sweep on any of these that miss the main shortlist.
EXECUTIVE_WATCHLIST = ["NVDA", "AMD", "ORCL", "PLTR", "BA", "AAPL", "TSLA", "META", "MSFT"]
VOLUME_SPIKE_THRESHOLD = 2.0
VOLUME_SPIKE_THRESHOLD_WATCHLIST = 1.5


# ── Suppress IBKR error 10091 from console output ────────────────────
# When running standalone, install the same filter as kairos_run.py.
# When imported from kairos_run.py, the filter is already installed.

class _Suppress10091(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "10091" not in record.getMessage()


def _install_ibkr_error_filter():
    suppress = _Suppress10091()
    for name in ("ib_insync", "ib_insync.wrapper", "ib_insync.client", "ib_insync.ib"):
        logger = logging.getLogger(name)
        logger.addFilter(suppress)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(suppress)
    debug_log = os.path.join(SCRIPT_DIR, "kairos_ibkr_debug.log")
    try:
        fh = logging.FileHandler(debug_log, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logging.getLogger("ib_insync").addHandler(fh)
        logging.getLogger("ib_insync").setLevel(logging.DEBUG)
    except IOError:
        pass


_install_ibkr_error_filter()

# Default thresholds (overridable via kairos_config.json)
DEFAULT_VOL_OI_THRESHOLD = 3.0
DEFAULT_IV_RANK_THRESHOLD = 0.80
DEFAULT_SKEW_BULL_THRESHOLD = 2.5
DEFAULT_SKEW_BEAR_THRESHOLD = 0.4
DEFAULT_DTE_MIN = 7
DEFAULT_DTE_MAX = 45


def _load_options_config() -> dict:
    """Load options signal config from kairos_config.json if available."""
    config_file = os.path.join(SCRIPT_DIR, "kairos_config.json")
    defaults = {
        "vol_oi_threshold": DEFAULT_VOL_OI_THRESHOLD,
        "iv_rank_threshold": DEFAULT_IV_RANK_THRESHOLD,
        "skew_bull_threshold": DEFAULT_SKEW_BULL_THRESHOLD,
        "skew_bear_threshold": DEFAULT_SKEW_BEAR_THRESHOLD,
        "dte_min": DEFAULT_DTE_MIN,
        "dte_max": DEFAULT_DTE_MAX,
    }
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            opts = cfg.get("options_signal", {})
            for k in defaults:
                if k in opts:
                    defaults[k] = opts[k]
        except (json.JSONDecodeError, IOError):
            pass
    return defaults


def _get_front_month_expiry(
    ib,
    stock_contract,
    dte_min: int = 7,
    dte_max: int = 45,
) -> str | None:
    """Find the nearest monthly expiration within the DTE window.

    Returns expiry string (YYYYMMDD) or None if no valid expiry found.
    """
    try:
        chains = ib.reqSecDefOptParams(
            stock_contract.symbol, "", stock_contract.secType, stock_contract.conId
        )
    except Exception:
        return None

    if not chains:
        return None

    today = datetime.now(timezone.utc).date()
    min_date = today + timedelta(days=dte_min)
    max_date = today + timedelta(days=dte_max)

    # Collect all expiries across all exchanges
    all_expiries: set[str] = set()
    for chain in chains:
        all_expiries.update(chain.expirations)

    valid = []
    for exp_str in all_expiries:
        try:
            exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
        except ValueError:
            continue
        if min_date <= exp_date <= max_date:
            valid.append(exp_str)

    if not valid:
        return None

    valid.sort()
    return valid[0]  # nearest valid expiry


def _get_atm_strikes(
    ib,
    chains,
    spot_price: float,
    pct_range: float = 0.05,
) -> list[float]:
    """Return strikes within pct_range of spot from available chains."""
    all_strikes: set[float] = set()
    for chain in chains:
        all_strikes.update(chain.strikes)

    lo = spot_price * (1 - pct_range)
    hi = spot_price * (1 + pct_range)
    atm = sorted(s for s in all_strikes if lo <= s <= hi)
    return atm


class _OptionsSubscriptionError(Exception):
    """Raised when IBKR returns error 10091 (no active options data subscription)."""
    pass


def _fetch_option_data(ib, symbol: str, expiry: str, strikes: list[float]):
    """Fetch volume, OI, and IV for calls and puts at given strikes.

    Returns (calls_data, puts_data) where each is a list of dicts:
        {strike, volume, open_interest, implied_vol}

    Raises _OptionsSubscriptionError if IBKR error 10091 is encountered.
    """
    from ib_insync import Option

    calls_data = []
    puts_data = []

    for strike in strikes:
        for right, data_list in [("C", calls_data), ("P", puts_data)]:
            try:
                contract = Option(symbol, expiry, strike, right, "SMART")
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    continue

                ib.reqMarketDataType(4)  # delayed/frozen OK
                ticker = ib.reqMktData(contract, genericTickList="100,101,106")
                ib.sleep(2)

                # Check for error 10091 before processing data
                if hasattr(ticker, "contract") and ticker.hasTicks is False:
                    # Check ib.wrapper error events for 10091
                    pass
                for err in getattr(ib.wrapper, "errors", []):
                    if len(err) >= 2 and err[1] == 10091:
                        ib.cancelMktData(contract)
                        raise _OptionsSubscriptionError(
                            f"options data subscription not active (error 10091)"
                        )

                vol = 0
                oi = 0
                iv = 0.0

                # Volume
                v = getattr(ticker, "volume", None)
                if v is not None and v == v and v >= 0:
                    vol = int(v)

                # Open interest (generic tick 101)
                for attr in ("callOpenInterest", "putOpenInterest", "openInterest"):
                    val = getattr(ticker, attr, None)
                    if val is not None and val == val and val > 0:
                        oi = int(val)
                        break

                # Implied volatility (generic tick 106)
                for attr in ("impliedVolatility", "modelGreeks"):
                    if attr == "modelGreeks":
                        greeks = getattr(ticker, attr, None)
                        if greeks and hasattr(greeks, "impliedVol"):
                            iv_val = greeks.impliedVol
                            if iv_val is not None and iv_val == iv_val:
                                iv = float(iv_val)
                                break
                    else:
                        val = getattr(ticker, attr, None)
                        if val is not None and val == val and val > 0:
                            iv = float(val)
                            break

                ib.cancelMktData(contract)

                data_list.append({
                    "strike": strike,
                    "volume": vol,
                    "open_interest": oi,
                    "implied_vol": round(iv, 4),
                })
            except _OptionsSubscriptionError:
                raise  # propagate to per-ticker handler
            except Exception:
                continue

    return calls_data, puts_data


def _compute_iv_rank(current_iv: float) -> float:
    """Estimate IV rank as a percentile.

    Without historical IV data from IBKR, we use a heuristic:
    typical equity IV ranges from ~0.15 (low) to ~0.80 (high).
    This gives a rough rank.  A future enhancement could cache
    daily IV snapshots for a true 52-week rank.
    """
    if current_iv <= 0:
        return 0.0
    # Typical range for equity options
    iv_floor = 0.15
    iv_ceil = 0.80
    rank = (current_iv - iv_floor) / (iv_ceil - iv_floor)
    return max(0.0, min(1.0, rank))


def _detect_options_with_yfinance(sym: str, config: dict) -> Optional[dict]:
    """Fallback options detection using yfinance when IBKR is unavailable.
    
    Detects the same three sub-signals using yfinance data:
    1. vol_spike — total call volume today vs historical average > 2.0x
    2. skew — call_vol / put_vol ratio > 2.5 (bullish unusual flow)
    3. iv_spike — implied volatility elevated vs recent average
    
    Returns None if yfinance fails, otherwise returns signal dict.
    """
    try:
        import yfinance as yf
        
        # Get the ticker and its options chain
        tk = yf.Ticker(sym)
        
        # Check if options exist for this ticker
        if not tk.options or len(tk.options) == 0:
            print(f"  [options] {sym}: no options available via yfinance")
            return None
        
        # Use the nearest expiry (first in the list)
        try:
            nearest_expiry = tk.options[0]
            chain = tk.option_chain(nearest_expiry)
        except Exception as e:
            print(f"  [options] {sym}: failed to get options chain — {e}")
            return None
        
        if chain is None or len(chain.calls) == 0 or len(chain.puts) == 0:
            print(f"  [options] {sym}: empty options chain from yfinance")
            return None
        
        # Extract calls and puts
        calls = chain.calls
        puts = chain.puts
        
        if calls.empty or puts.empty:
            print(f"  [options] {sym}: no calls/puts data from yfinance")
            return None
        
        # Calculate sub-signals
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        call_put_ratio = total_call_vol / total_put_vol if total_put_vol > 0 else float('inf')
        
        # 1. Volume spike detection (call volume vs historical)
        # For yfinance, we'll use open interest as a proxy for historical volume.
        # Watchlist tickers get a relaxed 1.5x threshold so smaller moves still flag.
        avg_call_oi = calls['openInterest'].mean() if len(calls['openInterest']) > 0 else 1
        on_watchlist = sym.upper() in EXECUTIVE_WATCHLIST
        vol_spike_mult = 1.5 if on_watchlist else 2.0
        vol_spike = total_call_vol > (vol_spike_mult * avg_call_oi)
        
        # 2. Skew detection (call/put ratio)
        skew_fired = call_put_ratio > 2.5  # bullish unusual flow
        
        # 3. IV spike detection
        # Use average IV of near-ATM calls as current IV
        atm_calls = calls[abs(calls['strike'] - calls['strike'].mean()) < (calls['strike'].std() * 0.5)]
        current_iv = atm_calls['impliedVolatility'].mean() if len(atm_calls) > 0 else 0
        
        # Compare to historical IV (use all calls as proxy)
        historical_iv_avg = calls['impliedVolatility'].mean() if len(calls['impliedVolatility']) > 0 else 1
        iv_spike = current_iv > (historical_iv_avg * 1.2)  # 20% above average
        
        # Determine which sub-signals fired
        triggered_by = []
        if vol_spike:
            triggered_by.append("vol_spike")
        if skew_fired:
            triggered_by.append("skew")
        if iv_spike:
            triggered_by.append("iv_spike")
        
        if not triggered_by:
            print(f"  [options] {sym}: yfinance data available but no signals fired")
            return None
        
        # Return signal result in the same format as IBKR path
        result = {
            'vol_oi_ratio': total_call_vol / avg_call_oi if avg_call_oi > 0 else 0,
            'iv_rank': 0.85 if iv_spike else 0.5,  # Approximate
            'call_put_ratio': call_put_ratio,
            'direction': 'bullish' if skew_fired else 'neutral',
            'front_expiry': nearest_expiry,
            'total_call_vol': int(total_call_vol),
            'total_put_vol': int(total_put_vol),
            'triggered_by': triggered_by,
            'executive_watchlist': on_watchlist,
            'vol_spike_threshold': vol_spike_mult,
        }
        
        print(f"  [options] {sym}: yfinance fallback → {triggered_by}")
        return result
        
    except ImportError:
        print(f"  [options] {sym}: yfinance not installed (pip install yfinance)")
        return None
    except Exception as yf_exc:
        print(f"  [options] {sym}: yfinance error — {yf_exc}")
        return None


def fetch_volume_spike_signal(ticker: str) -> dict:
    """Detect underlying volume spike + unusual call-side options flow via yfinance.

    Threshold is relaxed to 1.5x for EXECUTIVE_WATCHLIST tickers (vs 2.0x default)
    so smaller-but-significant moves on closely watched names still register.

    Returns:
        {
            "volume_ratio":       float,  # current_volume / averageVolume
            "volume_spike":       bool,   # ratio > per-ticker threshold
            "options_call_ratio": float,  # call_volume / call_open_interest (near-term expiry)
            "signal":             "STRONG" | "NEUTRAL" | "WEAK",
        }

    STRONG = volume spike AND call_volume/OI > 1.0 (fresh call buying on heavy tape).
    NEUTRAL = exactly one of the two fires.
    WEAK = neither (or yfinance is unavailable / threw).
    """
    default = {
        "volume_ratio": 0.0,
        "volume_spike": False,
        "options_call_ratio": 0.0,
        "signal": "WEAK",
    }

    sym = (ticker or "").upper()
    threshold = (
        VOLUME_SPIKE_THRESHOLD_WATCHLIST
        if sym in EXECUTIVE_WATCHLIST
        else VOLUME_SPIKE_THRESHOLD
    )

    try:
        import yfinance as yf
    except ImportError:
        return default

    try:
        tk = yf.Ticker(sym)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        current_vol = (
            info.get("volume")
            or info.get("regularMarketVolume")
            or 0
        )
        avg_vol = (
            info.get("averageVolume")
            or info.get("averageVolume10days")
            or 0
        )

        volume_ratio = (current_vol / avg_vol) if avg_vol else 0.0
        volume_spike = volume_ratio > threshold

        # Near-term expiry call volume vs open interest
        options_call_ratio = 0.0
        try:
            expirations = tk.options
            if expirations:
                chain = tk.option_chain(expirations[0])
                calls = chain.calls
                if calls is not None and not calls.empty:
                    total_call_vol = float(calls["volume"].fillna(0).sum())
                    total_call_oi = float(calls["openInterest"].fillna(0).sum())
                    if total_call_oi > 0:
                        options_call_ratio = total_call_vol / total_call_oi
        except Exception:
            pass  # leave at 0.0 — partial signal is fine

        call_flow_hot = options_call_ratio > 1.0
        if volume_spike and call_flow_hot:
            signal = "STRONG"
        elif volume_spike or call_flow_hot:
            signal = "NEUTRAL"
        else:
            signal = "WEAK"

        return {
            "volume_ratio": round(volume_ratio, 2),
            "volume_spike": volume_spike,
            "options_call_ratio": round(options_call_ratio, 2),
            "signal": signal,
        }
    except Exception:
        return default


def detect_options_activity(
    tickers: list[str],
    ib=None,
    config: dict | None = None,
) -> dict[str, dict]:
    """Screen tickers for unusual options activity via IBKR.

    Args:
        tickers: shortlisted tickers to screen (max ~15)
        ib:      live ib_insync.IB connection (caller manages lifecycle)
        config:  threshold overrides (from kairos_config.json)

    Returns:
        {ticker: {vol_oi_ratio, iv_rank, call_put_ratio, direction,
                  front_expiry, total_call_vol, total_put_vol,
                  triggered_by: [sub-signal names]}}
        Only tickers that fired at least one sub-signal are included.
    """
    if config is None:
        config = _load_options_config()

    vol_oi_thresh = config["vol_oi_threshold"]
    iv_rank_thresh = config["iv_rank_threshold"]
    skew_bull = config["skew_bull_threshold"]
    skew_bear = config["skew_bear_threshold"]
    dte_min = config["dte_min"]
    dte_max = config["dte_max"]

    own_connection = False
    if ib is None:
        try:
            from ib_insync import IB
            ib = IB()
            ib.connect("127.0.0.1", 7497, clientId=7, timeout=10)
            own_connection = True
        except Exception as exc:
            print(f"  [options] IBKR unavailable: {exc}")
            return {}

    from ib_insync import Stock

    results: dict[str, dict] = {}

    for sym in tickers:
        try:
            # Qualify the underlying stock
            stock = Stock(sym, "SMART", "USD")
            qualified = ib.qualifyContracts(stock)
            if not qualified:
                continue

            # Get spot price for ATM determination
            ib.reqMarketDataType(4)
            mkt = ib.reqMktData(stock)
            ib.sleep(2)
            spot = None
            for attr in ("last", "close", "bid", "ask"):
                val = getattr(mkt, attr, None)
                if val is not None and val == val and val > 0:
                    spot = float(val)
                    break
            ib.cancelMktData(stock)

            if not spot:
                continue

            # Find front-month expiry
            expiry = _get_front_month_expiry(ib, stock, dte_min, dte_max)
            if not expiry:
                continue

            # Get available chains for ATM strikes
            chains = ib.reqSecDefOptParams(
                stock.symbol, "", stock.secType, stock.conId
            )
            atm_strikes = _get_atm_strikes(ib, chains, spot)
            if not atm_strikes:
                continue

            # Limit to 5 ATM strikes to keep data requests reasonable
            atm_strikes = atm_strikes[:5]

            # Fetch option data
            calls, puts = _fetch_option_data(ib, sym, expiry, atm_strikes)

            if not calls and not puts:
                continue

            # ── Aggregate metrics ────────────────────────────────────
            total_call_vol = sum(c["volume"] for c in calls)
            total_put_vol = sum(p["volume"] for p in puts)
            total_call_oi = sum(c["open_interest"] for c in calls)
            total_put_oi = sum(p["open_interest"] for p in puts)
            total_vol = total_call_vol + total_put_vol
            total_oi = total_call_oi + total_put_oi

            # Vol/OI ratio
            vol_oi = total_vol / total_oi if total_oi > 0 else 0.0

            # Average IV across all fetched contracts
            all_ivs = [c["implied_vol"] for c in calls + puts if c["implied_vol"] > 0]
            avg_iv = sum(all_ivs) / len(all_ivs) if all_ivs else 0.0
            iv_rank = _compute_iv_rank(avg_iv)

            # Call/put volume ratio
            cp_ratio = (total_call_vol / total_put_vol
                        if total_put_vol > 0
                        else (10.0 if total_call_vol > 0 else 0.0))

            # ── Check sub-signal triggers ────────────────────────────
            triggered: list[str] = []

            if vol_oi >= vol_oi_thresh and total_oi >= 100:
                triggered.append("vol_spike")

            if iv_rank >= iv_rank_thresh and avg_iv > 0:
                triggered.append("iv_rank")

            if cp_ratio >= skew_bull:
                triggered.append("skew_bullish")
            elif cp_ratio > 0 and cp_ratio <= skew_bear:
                triggered.append("skew_bearish")

            # yfinance volume-spike check runs alongside the IBKR sub-signals.
            # STRONG signal independently fires HOT-OPTIONS; data is always
            # attached so reviewers can see the underlying tape context.
            vol_spike_yf = fetch_volume_spike_signal(sym)
            if vol_spike_yf.get("signal") == "STRONG":
                triggered.append("volume_spike_yf")

            if not triggered:
                continue

            # Determine directional bias
            if "skew_bearish" in triggered:
                direction = "bearish"
            elif "skew_bullish" in triggered:
                direction = "bullish"
            elif total_call_vol > total_put_vol:
                direction = "leaning_bullish"
            else:
                direction = "neutral"

            results[sym] = {
                "vol_oi_ratio": round(vol_oi, 2),
                "iv_rank": round(iv_rank, 2),
                "call_put_ratio": round(cp_ratio, 2),
                "avg_iv": round(avg_iv, 4),
                "direction": direction,
                "front_expiry": expiry,
                "total_call_vol": total_call_vol,
                "total_put_vol": total_put_vol,
                "total_oi": total_oi,
                "triggered_by": triggered,
                "strikes_checked": len(atm_strikes),
                "volume_spike_yf": vol_spike_yf,
            }

        except _OptionsSubscriptionError:
            print(f"  [options] HOT-OPTIONS: IBKR unavailable, trying yfinance fallback for {sym}")
            yf_result = _detect_options_with_yfinance(sym, config)
            if yf_result:
                results[sym] = yf_result
            continue
        except Exception as exc:
            # Check if the error message contains 10091 (some ib_insync versions
            # surface it as a generic exception rather than a distinct error code)
            if "10091" in str(exc):
                print(f"  [options] HOT-OPTIONS: IBKR unavailable, trying yfinance fallback for {sym}")
                yf_result = _detect_options_with_yfinance(sym, config)
                if yf_result:
                    results[sym] = yf_result
            else:
                print(f"  [options] {sym}: error — {exc}")
            continue

    if own_connection:
        try:
            ib.disconnect()
        except Exception:
            pass

    return results


def save_options_activity(results: dict[str, dict]) -> None:
    """Persist options activity results for downstream prompt assembly."""
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tickers_screened": len(results),
        "hits": results,
    }
    with open(OPTIONS_CACHE, "w") as f:
        json.dump(payload, f, indent=2)


def load_options_activity() -> dict[str, dict]:
    """Load cached options activity results. Returns {ticker: data}."""
    if not os.path.exists(OPTIONS_CACHE):
        return {}
    try:
        with open(OPTIONS_CACHE) as f:
            data = json.load(f)
        return data.get("hits", {})
    except (json.JSONDecodeError, IOError):
        return {}


# ── CLI for standalone testing ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kairos Options Activity Detector")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers (default: load shortlist)")
    args = parser.parse_args()

    print("=" * W)
    print("  KAIROS OPTIONS ACTIVITY DETECTOR")
    print("=" * W)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        screen_file = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")
        if os.path.exists(screen_file):
            with open(screen_file) as f:
                tickers = json.load(f).get("shortlist", [])
            print(f"  Loaded {len(tickers)} tickers from screening shortlist")
        else:
            print("  No shortlist found and no --tickers specified")
            sys.exit(1)

    print(f"  Screening {len(tickers)} ticker(s): {', '.join(tickers)}")
    print()

    config = _load_options_config()
    print(f"  Thresholds:")
    print(f"    vol/OI      ≥ {config['vol_oi_threshold']}")
    print(f"    IV rank     ≥ {config['iv_rank_threshold']:.0%}")
    print(f"    skew bull   ≥ {config['skew_bull_threshold']}")
    print(f"    skew bear   ≤ {config['skew_bear_threshold']}")
    print(f"    DTE range:    {config['dte_min']}–{config['dte_max']} days")
    print()

    results = detect_options_activity(tickers, config=config)

    if results:
        print(f"\n  HOT-OPTIONS hits: {len(results)}")
        for sym, data in results.items():
            triggers = ", ".join(data["triggered_by"])
            print(f"    {sym}: vol/OI={data['vol_oi_ratio']:.1f}x  "
                  f"IV rank={data['iv_rank']:.0%}  "
                  f"C/P={data['call_put_ratio']:.1f}x  "
                  f"({data['direction']})  [{triggers}]")
        save_options_activity(results)
        print(f"\n  Saved to {OPTIONS_CACHE}")
    else:
        print("\n  No unusual options activity detected.")

    print("=" * W)


if __name__ == "__main__":
    main()
