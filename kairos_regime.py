"""
Kairos Macro Regime Detection — Phase 0R

Classifies the current market regime into one of four states based on
VIX, SPY vs moving averages, put/call ratio, Kalshi recession probability,
and the 10Y-2Y Treasury spread.

Regimes:
    NORMAL        — business as usual
    CAUTION       — tighten sizing, raise conviction bar
    RISK-OFF      — defensive posture, restrict sectors
    EXTREME-FEAR  — halt new equity buys

Each regime carries guardrails that downstream phases enforce.

Usage:
    from kairos_regime import detect_regime
    result = detect_regime()
    # result["regime"], result["guardrails"], result["prompt_section"]
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

REGIME_STATE_FILE = os.path.join(SCRIPT_DIR, ".kairos_regime_state.json")

# ── Regime thresholds ────────────────────────────────────────────────

REGIMES = {
    "EXTREME-FEAR": {
        "description": "Extreme fear — halt new equity buys",
        "guardrails": {
            "equity_buys_allowed": False,
            "crypto_size_mult": 0.50,
            "max_position_mult": 0.25,
            "mode_c_allowed": False,
            "mode_c_min_conviction": 10,   # effectively disabled
            "stop_loss_pct": 5.0,
            "blocked_sectors": [],         # moot — no buys
        },
    },
    "RISK-OFF": {
        "description": "Risk-off — defensive posture, restrict sectors",
        "guardrails": {
            "equity_buys_allowed": True,
            "crypto_size_mult": 0.50,
            "max_position_mult": 0.25,
            "mode_c_allowed": False,
            "mode_c_min_conviction": 10,
            "stop_loss_pct": 8.0,
            "blocked_sectors": [
                "energy_materials", "financials", "consumer_discretionary",
            ],
        },
    },
    "CAUTION": {
        "description": "Caution — reduced sizing, higher conviction bar",
        "guardrails": {
            "equity_buys_allowed": True,
            "crypto_size_mult": 0.75,
            "max_position_mult": 0.50,
            "mode_c_allowed": True,
            "mode_c_min_conviction": 6,
            "stop_loss_pct": 10.0,
            "blocked_sectors": [],
        },
    },
    "NORMAL": {
        "description": "Normal — no restrictions",
        "guardrails": {
            "equity_buys_allowed": True,
            "crypto_size_mult": 1.0,
            "max_position_mult": 1.0,
            "mode_c_allowed": True,
            "mode_c_min_conviction": 4,
            "stop_loss_pct": 10.0,
            "blocked_sectors": [],
        },
    },
}


# ── Data fetchers ────────────────────────────────────────────────────

def _fetch_vix_and_spy_ibkr(client_id: int = 15) -> dict:
    """Pull VIX current value, SPY price, and SPY 50/200-day MAs from IBKR.

    client_id lets callers use a dedicated IBKR clientId so concurrent regime
    lookups (e.g. the scheduler and the Slack commander) don't collide on 15.
    """
    result = {
        "vix": None, "vix_5d_ago": None,
        "spy_price": None, "spy_50ma": None, "spy_200ma": None,
        "spy_put_call_ratio": None,
    }

    try:
        from ib_insync import IB, Index, Stock
        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=client_id, timeout=10)

        # ── VIX current ──────────────────────────────────────────
        vix_contract = Index("VIX", "CBOE")
        ib.qualifyContracts(vix_contract)
        ib.reqMarketDataType(4)
        mkt = ib.reqMktData(vix_contract)
        ib.sleep(2)
        for attr in ("last", "close", "bid", "ask"):
            val = getattr(mkt, attr, None)
            if val is not None and val == val and val > 0:
                result["vix"] = round(float(val), 2)
                break
        ib.cancelMktData(vix_contract)

        # ── VIX 5-day history (for trend) ────────────────────────
        try:
            bars = ib.reqHistoricalData(
                vix_contract,
                endDateTime="",
                durationStr="10 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
            )
            if bars and len(bars) >= 6:
                result["vix_5d_ago"] = round(float(bars[-6].close), 2)
        except Exception:
            pass

        # ── SPY current price ────────────────────────────────────
        spy_contract = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(spy_contract)
        mkt = ib.reqMktData(spy_contract)
        ib.sleep(2)
        for attr in ("last", "close", "bid", "ask"):
            val = getattr(mkt, attr, None)
            if val is not None and val == val and val > 0:
                result["spy_price"] = round(float(val), 2)
                break
        ib.cancelMktData(spy_contract)

        # ── SPY 200-day history → compute 50MA & 200MA ───────────
        try:
            bars = ib.reqHistoricalData(
                spy_contract,
                endDateTime="",
                durationStr="1 Y",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
            )
            closes = [float(b.close) for b in bars]
            if len(closes) >= 200:
                result["spy_200ma"] = round(sum(closes[-200:]) / 200, 2)
            if len(closes) >= 50:
                result["spy_50ma"] = round(sum(closes[-50:]) / 50, 2)
        except Exception:
            pass

        # ── SPY put/call ratio from options chain ────────────────
        # Suppress IBKR error callbacks during options probing (paper
        # accounts often lack options data subscriptions).
        try:
            _orig_error = ib.errorEvent
            ib.errorEvent.clear()

            chains = ib.reqSecDefOptParams(
                spy_contract.symbol, "", spy_contract.secType, spy_contract.conId)
            if chains:
                chain = chains[0]
                # Pick the nearest monthly expiry
                expirations = sorted(chain.expirations)
                target_exp = None
                today_str = datetime.now().strftime("%Y%m%d")
                for exp in expirations:
                    if exp >= today_str:
                        target_exp = exp
                        break
                if target_exp:
                    from ib_insync import Option
                    # Sample a few strikes around current price for volume
                    spy_px = result["spy_price"] or 500
                    strikes = [s for s in sorted(chain.strikes)
                               if abs(s - spy_px) <= 20][:20]
                    put_vol, call_vol = 0, 0
                    contracts = []
                    for s in strikes:
                        for right in ("P", "C"):
                            c = Option("SPY", target_exp, s, right, "SMART")
                            contracts.append(c)
                    qualified = ib.qualifyContracts(*contracts)
                    for c in qualified:
                        ib.reqMktData(c, genericTickList="", snapshot=True)
                    ib.sleep(3)
                    for c in qualified:
                        tk = ib.ticker(c)
                        vol = getattr(tk, "volume", 0)
                        if vol and vol == vol and vol > 0:
                            if c.right == "P":
                                put_vol += vol
                            else:
                                call_vol += vol
                        ib.cancelMktData(c)
                    if call_vol > 0:
                        result["spy_put_call_ratio"] = round(put_vol / call_vol, 2)

            ib.errorEvent = _orig_error
        except Exception:
            pass

        ib.disconnect()

    except Exception as exc:
        print(f"  WARNING: IBKR regime data fetch failed: {exc}")

    return result


def _fetch_fred_spread() -> float | None:
    """Fetch 10Y-2Y Treasury spread from FRED (DGS10 - DGS2)."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("  WARNING: FRED_API_KEY not set — skipping Treasury spread")
        return None

    import requests

    values = {}
    for series in ("DGS10", "DGS2"):
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series,
                    "api_key": key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                },
                timeout=10,
            )
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            # Skip "." placeholder values
            for o in obs:
                if o.get("value", ".") != ".":
                    values[series] = float(o["value"])
                    break
        except Exception as exc:
            print(f"  WARNING: FRED {series} fetch failed: {exc}")

    if "DGS10" in values and "DGS2" in values:
        return round(values["DGS10"] - values["DGS2"], 3)
    return None


def _fetch_kalshi_recession() -> float | None:
    """Query Kalshi for recession probability on any active recession market.

    Returns probability as a percentage (0-100), or None if unavailable.
    """
    import requests, os, time, base64
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    key_id = os.environ.get("KALSHI_KEY_ID")
    key_path = os.environ.get("KALSHI_KEY_PATH")
    if not key_id or not key_path:
        return None

    try:
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
    except Exception:
        return None

    def _sign_request(method, path):
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode()
        sig = private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
        return ts, base64.b64encode(sig).decode()

    endpoints = [
        "https://api.elections.kalshi.com/trade-api/v2/events",
        "https://api.elections.kalshi.com/trade-api/v2/events",
    ]

    events = []
    for url in endpoints:
        try:
            path = "/trade-api/v2/events"
            ts, sig = _sign_request("GET", path)
            headers = {
                "Accept": "application/json",
                "KALSHI-ACCESS-KEY": key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": sig,
            }
            resp = requests.get(url, params={"limit": 50, "status": "open"},
                                headers=headers, timeout=10)
            resp.raise_for_status()
            all_events = resp.json().get("events", [])
            # Filter for recession-related markets
            events = [e for e in all_events if any(
                kw in (e.get("event_ticker","") + e.get("title","") + e.get("sub_title","")).upper()
                for kw in ["RECESSION", "GDP", "RECES"]
            )]
            if not events:
                # Try dedicated recession endpoint
                resp2 = requests.get(
                    url.replace("/events", "/markets"),
                    params={"limit": 50, "status": "open", "series_ticker": "KXRECESSION"},
                    headers=headers, timeout=10
                )
                if resp2.status_code == 200:
                    markets = resp2.json().get("markets", [])
                    if markets:
                        return markets[0].get("last_price", None)
            break
        except Exception:
            events = []
            continue

    if not events:
        return None


def build_prompt_section(regime: str, reason: str, data: dict) -> str:
    """Build the SECTION 0: MACRO REGIME block for Claude prompts."""
    W = 72
    guardrails = REGIMES[regime]["guardrails"]

    lines = [
        f"{'=' * W}",
        f"SECTION 0: MACRO REGIME — {regime}",
        f"{'=' * W}",
        f"",
        f"Classification: {regime} — {reason}",
        f"",
        f"Indicators:",
    ]

    if data.get("vix") is not None:
        trend = ""
        if data.get("vix_5d_ago") is not None:
            delta = data["vix"] - data["vix_5d_ago"]
            trend = f"  (5d trend: {delta:+.1f})"
        lines.append(f"  VIX:              {data['vix']}{trend}")
    if data.get("spy_price") is not None:
        lines.append(f"  SPY price:        ${data['spy_price']}")
    if data.get("spy_50ma") is not None:
        rel = "above" if (data.get("spy_price") or 0) >= data["spy_50ma"] else "BELOW"
        lines.append(f"  SPY 50-day MA:    ${data['spy_50ma']} (SPY {rel})")
    if data.get("spy_200ma") is not None:
        rel = "above" if (data.get("spy_price") or 0) >= data["spy_200ma"] else "BELOW"
        lines.append(f"  SPY 200-day MA:   ${data['spy_200ma']} (SPY {rel})")
    if data.get("spy_put_call_ratio") is not None:
        lines.append(f"  SPY put/call:     {data['spy_put_call_ratio']}")
    if data.get("treasury_spread") is not None:
        inv = " (INVERTED)" if data["treasury_spread"] < 0 else ""
        lines.append(f"  10Y-2Y spread:    {data['treasury_spread']:+.3f}{inv}")
    if data.get("kalshi_recession_pct") is not None:
        lines.append(f"  Kalshi recession: {data['kalshi_recession_pct']}%")

    lines.append("")
    lines.append("Active guardrails:")

    if not guardrails["equity_buys_allowed"]:
        lines.append("  *** NEW EQUITY BUYS HALTED — extreme fear regime ***")
    if guardrails["max_position_mult"] < 1.0:
        pct = int(guardrails["max_position_mult"] * 100)
        lines.append(f"  Position sizing: {pct}% of normal maximum")
    if guardrails["crypto_size_mult"] < 1.0:
        pct = int(guardrails["crypto_size_mult"] * 100)
        lines.append(f"  Crypto sizing: {pct}% of normal")
    if not guardrails["mode_c_allowed"]:
        lines.append("  Mode C (conviction-only): DISABLED")
    elif guardrails["mode_c_min_conviction"] > 4:
        lines.append(f"  Mode C min conviction: {guardrails['mode_c_min_conviction']}/10 (raised)")
    if guardrails["blocked_sectors"]:
        sectors = ", ".join(guardrails["blocked_sectors"])
        lines.append(f"  Blocked sectors (no new buys): {sectors}")
    if guardrails["stop_loss_pct"] < 10:
        lines.append(f"  Stop-loss threshold: {guardrails['stop_loss_pct']}% (tightened)")

    if regime == "NORMAL":
        lines.append("  (no restrictions — normal operation)")

    lines.append("")
    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────

def _classify(data: dict) -> tuple:
    """Classify macro regime from indicator data. Returns (regime, reason)."""
    vix = data.get("vix")
    spy_price = data.get("spy_price")
    spy_200ma = data.get("spy_200ma")
    kalshi = data.get("kalshi_recession_pct")
    treasury_spread = data.get("treasury_spread")

    reasons = []

    if vix is not None and vix > 35:
        reasons.append(f"VIX {vix} >35")
        return "EXTREME-FEAR", "; ".join(reasons)

    if spy_price is not None and spy_200ma is not None and spy_200ma > 0:
        pct_below_200 = (spy_200ma - spy_price) / spy_200ma * 100
        if pct_below_200 > 10:
            reasons.append(f"SPY {pct_below_200:.1f}% below 200MA")
            return "EXTREME-FEAR", "; ".join(reasons)

    if vix is not None and vix > 25:
        reasons.append(f"VIX {vix} >25")
    if spy_price is not None and spy_200ma is not None and spy_price < spy_200ma:
        reasons.append(f"SPY ${spy_price} below 200MA ${spy_200ma}")
    if kalshi is not None and kalshi > 40:
        reasons.append(f"Kalshi recession {kalshi}% >40%")
    if reasons:
        return "RISK-OFF", "; ".join(reasons)

    reasons = []
    if vix is not None and vix > 20:
        reasons.append(f"VIX {vix} >20")
    if treasury_spread is not None and treasury_spread < 0:
        reasons.append(f"Inverted yield curve {treasury_spread:.3f}")
    if kalshi is not None and kalshi > 25:
        reasons.append(f"Kalshi recession {kalshi}% >25%")
    if reasons:
        return "CAUTION", "; ".join(reasons)

    spread_str = f"+{treasury_spread:.3f}" if treasury_spread else "N/A"
    vix_str = f"{vix:.2f}" if vix is not None else "N/A"
    return "NORMAL", f"VIX {vix_str}; SPY above 50MA; spread {spread_str}"


def _alert_regime_change(previous: str, current: str, reason: str) -> None:
    """Post a regime change alert to #kairos-alerts."""
    try:
        from kairos_alerts import alert_pipeline_event
        emoji_map = {"NORMAL": "green_circle", "CAUTION": "yellow_circle",
                     "RISK-OFF": "orange_circle", "EXTREME-FEAR": "red_circle"}
        icon = emoji_map.get(current, "white_circle")
        msg = (":" + icon + ": *REGIME CHANGE: " + previous + " -> " + current + "*\n"
               "Reason: " + reason + "\n"
               "_Position sizing and conviction thresholds have been adjusted._")
        alert_pipeline_event(msg, channel="alerts")
    except Exception as e:
        print(f"  WARNING: regime alert failed: {e}")


def _load_previous_regime() -> str | None:
    """Load the last recorded regime from kairos.db."""
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "kairos.db")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT regime FROM regime_log ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _save_regime_state(regime: str, reason: str, data: dict) -> None:
    """Persist regime to kairos.db regime_log table."""
    try:
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "kairos.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                regime TEXT,
                reason TEXT,
                vix REAL,
                spy_price REAL,
                treasury_spread REAL,
                kalshi_recession_pct REAL
            )
        """)
        cur.execute("""
            INSERT INTO regime_log (timestamp, regime, reason, vix, spy_price, treasury_spread, kalshi_recession_pct)
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)
        """, (
            regime, reason,
            data.get("vix"), data.get("spy_price"),
            data.get("treasury_spread"), data.get("kalshi_recession_pct")
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"WARNING: Could not save regime state: {e}")


def _log_regime_to_db(regime: str, reason: str, data: dict) -> None:
    """Log regime to DB - delegates to _save_regime_state."""
    _save_regime_state(regime, reason, data)


def detect_regime(verbose: bool = True, client_id: int = 15) -> dict:
    """Run full macro regime detection. Returns dict with:

        regime:         str — NORMAL / CAUTION / RISK-OFF / EXTREME-FEAR
        reason:         str — human-readable classification reason
        guardrails:     dict — downstream enforcement rules
        prompt_section: str — text block to prepend to Claude prompts
        data:           dict — raw indicator values
        changed:        bool — True if regime shifted since last cycle
    """
    if verbose:
        print("\n  Fetching macro indicators...")

    # ── Gather data in sequence (IBKR first, then HTTP) ──────────
    ibkr_data = _fetch_vix_and_spy_ibkr(client_id=client_id)
    treasury_spread = _fetch_fred_spread()
    kalshi_pct = _fetch_kalshi_recession()

    data = {
        **ibkr_data,
        "treasury_spread": treasury_spread,
        "kalshi_recession_pct": kalshi_pct,
    }

    if verbose:
        vix_str = str(data.get("vix") or "N/A")
        spy_str = f"${data['spy_price']}" if data.get("spy_price") else "N/A"
        spread_str = f"{treasury_spread:+.3f}" if treasury_spread is not None else "N/A"
        kalshi_str = f"{kalshi_pct}%" if kalshi_pct is not None else "N/A"
        print(f"  VIX: {vix_str}  SPY: {spy_str}  "
              f"10Y-2Y: {spread_str}  Kalshi recession: {kalshi_str}")

    # ── Classify ─────────────────────────────────────────────────
    regime, reason = _classify(data)

    if verbose:
        print(f"  Regime: {regime} — {reason}")

    # ── Change detection + alerting ──────────────────────────────
    previous = _load_previous_regime()
    changed = previous is not None and previous != regime

    if changed:
        print(f"  ⚠ REGIME CHANGE: {previous} → {regime}")
        _alert_regime_change(previous, regime, reason)

    # ── Persist ──────────────────────────────────────────────────
    _save_regime_state(regime, reason, data)

    # ── Build prompt section ─────────────────────────────────────
    prompt_section = build_prompt_section(regime, reason, data)

    return {
        "regime": regime,
        "reason": reason,
        "guardrails": REGIMES[regime]["guardrails"],
        "prompt_section": prompt_section,
        "data": data,
        "changed": changed,
        "previous_regime": previous,
    }


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kairos Macro Regime Detection")
    parser.add_argument("--history", action="store_true",
                        help="Show recent regime_log entries from kairos.db")
    parser.add_argument("--last", type=int, default=20)
    args = parser.parse_args()

    if args.history:
        import sqlite3
        DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM regime_log ORDER BY id DESC LIMIT ?",
            (args.last,),
        ).fetchall()
        conn.close()
        if not rows:
            print("No regime_log entries.")
        else:
            W = 72
            print("╔" + "═" * W + "╗")
            print(f"║  REGIME HISTORY — Last {len(rows)} entries".ljust(W + 1) + "║")
            print("╚" + "═" * W + "╝")
            print(f"  {'Timestamp':<22} {'Regime':<14} {'VIX':>5} {'SPY':>8} "
                  f"{'50MA':>8} {'200MA':>8} {'Spread':>7} {'Kalshi':>7}")
            print("  " + "─" * 70)
            for r in reversed(rows):
                ts = (r["timestamp"] or "")[:19]
                vix = f"{r['vix']:.0f}" if r["vix"] else "—"
                spy = f"${r['spy_price']:.0f}" if r["spy_price"] else "—"
                ma50 = f"${r['spy_50ma']:.0f}" if r["spy_50ma"] else "—"
                ma200 = f"${r['spy_200ma']:.0f}" if r["spy_200ma"] else "—"
                spread = f"{r['treasury_spread']:+.2f}" if r["treasury_spread"] is not None else "—"
                kalshi = f"{r['kalshi_recession_pct']:.0f}%" if r["kalshi_recession_pct"] is not None else "—"
                print(f"  {ts:<22} {r['regime']:<14} {vix:>5} {spy:>8} "
                      f"{ma50:>8} {ma200:>8} {spread:>7} {kalshi:>7}")
            print()
    else:
        result = detect_regime(verbose=True)
        print(f"\n  Result: {result['regime']}")
        print(f"  Changed: {result['changed']}")
        if result["changed"]:
            print(f"  Previous: {result['previous_regime']}")
        print(f"\n  Guardrails:")
        for k, v in result["guardrails"].items():
            print(f"    {k}: {v}")
