#!/usr/bin/env python3
"""
Kairos Performance Dashboard Generator

Pulls data from IBKR paper account (port 7497) and kairos.db, then generates
kairos_dashboard.html — a self-contained dark-theme dashboard comparing Kairos
returns against J's 14.4% advisor benchmark and 29% annualized target.

Usage:
    python3 kairos_dashboard.py           # Full update (requires IBKR on port 7497)
    python3 kairos_dashboard.py --no-ibkr # Use cached performance data only
"""

import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(SCRIPT_DIR, "kairos.db")
PERF_FILE     = os.path.join(SCRIPT_DIR, "kairos_performance.json")
DASHBOARD_OUT = os.path.join(SCRIPT_DIR, "kairos_dashboard.html")
LEDGER_FILE   = os.path.join(SCRIPT_DIR, "kairos_ledger.txt")
SCREEN_LOG    = os.path.join(SCRIPT_DIR, "kairos_screen_log.json")

ADVISOR_RATE = 0.144  # 14.4% annualized — J's financial advisor benchmark
TARGET_RATE  = 0.290  # 29.0% annualized — 2× advisor target


# ── IBKR ──────────────────────────────────────────────────────────────

# ── Ticker name resolution ────────────────────────────────────────────

_TICKER_NAMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticker_names (
    ticker  TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    updated TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _load_universe_names() -> dict[str, str]:
    """Extract ticker→name mappings from kairos_universe.json (Tier B has names)."""
    universe_file = os.path.join(SCRIPT_DIR, "kairos_universe.json")
    names: dict[str, str] = {}
    if not os.path.exists(universe_file):
        return names
    try:
        with open(universe_file) as f:
            universe = json.load(f)
        for entry in universe.get("tier_b", {}).get("tickers", []):
            if isinstance(entry, dict) and entry.get("name"):
                names[entry["symbol"]] = entry["name"]
    except (json.JSONDecodeError, IOError):
        pass
    return names


def _resolve_ticker_names(symbols: list[str]) -> dict[str, str]:
    """Resolve ticker symbols to company names.

    Priority: kairos_universe.json → ticker_names DB cache → yfinance.
    New yfinance lookups are cached in the ticker_names table.
    """
    if not symbols:
        return {}

    names = _load_universe_names()
    result: dict[str, str] = {}
    missing = []

    for sym in symbols:
        if sym in names:
            result[sym] = names[sym]
        else:
            missing.append(sym)

    if not missing:
        return result

    # Check DB cache
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        conn.execute(_TICKER_NAMES_SCHEMA)
        conn.commit()
        rows = conn.execute(
            f"SELECT ticker, name FROM ticker_names WHERE ticker IN ({','.join('?' * len(missing))})",
            missing,
        ).fetchall()
        for ticker, name in rows:
            result[ticker] = name
            missing.remove(ticker)
        conn.close()
    except Exception:
        pass

    if not missing:
        return result

    # Fetch from yfinance (batch, with timeout)
    try:
        import yfinance as yf
        fetched = {}
        for sym in missing:
            try:
                info = yf.Ticker(sym).info
                name = info.get("longName") or info.get("shortName") or sym
                fetched[sym] = name
            except Exception:
                fetched[sym] = sym

        # Cache to DB
        if fetched:
            try:
                import sqlite3
                conn = sqlite3.connect(DB_PATH)
                conn.execute(_TICKER_NAMES_SCHEMA)
                for ticker, name in fetched.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO ticker_names (ticker, name) VALUES (?, ?)",
                        (ticker, name),
                    )
                conn.commit()
                conn.close()
            except Exception:
                pass
            result.update(fetched)
    except ImportError:
        # yfinance not installed — use ticker as name
        for sym in missing:
            result[sym] = sym

    return result


# ── IBKR data ─────────────────────────────────────────────────────────

def _ibkr_marks(ib, symbols: list[str]) -> dict:
    """Pull per-symbol marks from IBKR, live-preferred with a delayed fallback.

    Returns {sym: (price, mdtype)} where mdtype is 1=live, 3=delayed, 4=frozen,
    or (None, None) if no price. Replaces the yfinance per-position lookup, which
    gets rate-limited (429) and silently falls back to avg cost — freezing the
    dashboard. Delayed data flows even when the live entitlement is impaired.
    """
    from ib_insync import Stock
    import logging
    # Silence the expected per-ticker 10089 "delayed available" / 300 notices so
    # they don't flood the cron log (57 tickers x 2 passes).
    _iblog = logging.getLogger("ib_insync")
    _iblog_prev = _iblog.level
    _iblog.setLevel(logging.CRITICAL)

    def _pick(t):
        for v in (t.last, t.close):
            if v is not None and v == v and v > 0:   # not None, not NaN, positive
                return round(float(v), 2)
        return None

    syms = sorted({s for s in symbols if s})
    contracts = {}
    for s in syms:
        try:
            c = Stock(s, "SMART", "USD")
            ib.qualifyContracts(c)
            contracts[s] = c
        except Exception:
            pass

    out: dict = {}
    # Pass 1 — live (type 1). Whatever doesn't tick gets retried as delayed.
    ib.reqMarketDataType(1)
    tickers = {s: ib.reqMktData(c, "", False, False) for s, c in contracts.items()}
    ib.sleep(4)
    missing = []
    for s, t in tickers.items():
        px = _pick(t)
        if px is not None:
            out[s] = (px, t.marketDataType)
        else:
            missing.append(s)
        try:
            ib.cancelMktData(contracts[s])
        except Exception:
            pass
    # Pass 2 — delayed (type 3) for anything still missing.
    if missing:
        ib.reqMarketDataType(3)
        t2 = {s: ib.reqMktData(contracts[s], "", False, False) for s in missing}
        ib.sleep(4)
        for s, t in t2.items():
            px = _pick(t)
            out[s] = (px, t.marketDataType) if px is not None else (None, None)
            try:
                ib.cancelMktData(contracts[s])
            except Exception:
                pass
    _iblog.setLevel(_iblog_prev)
    return out


def _ibkr_live_available(ib) -> bool:
    """True only if IBKR is serving LIVE (type-1) ticks right now.

    Probes a liquid reference (SPY) so staleness detection never depends on how
    many positions happened to price. Returns False on delayed-only / no-tick /
    error (e.g. 10089 when the live entitlement is down).
    """
    from ib_insync import Stock
    import logging
    _lg = logging.getLogger("ib_insync")
    _prev = _lg.level
    _lg.setLevel(logging.CRITICAL)
    try:
        ib.reqMarketDataType(1)
        c = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(c)
        t = ib.reqMktData(c, "", False, False)
        live = False
        for _ in range(3):
            ib.sleep(2)
            if (t.last is not None and t.last == t.last
                    and t.marketDataType == 1):
                live = True
                break
        try:
            ib.cancelMktData(c)
        except Exception:
            pass
        return live
    except Exception:
        return False
    finally:
        _lg.setLevel(_prev)


def fetch_ibkr_data() -> dict:
    """Connect to IBKR TWS on port 7497, pull account summary and positions."""
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=3, timeout=10)

        wanted = {"NetLiquidation", "TotalCashValue", "GrossPositionValue",
                  "UnrealizedPnL", "RealizedPnL", "BuyingPower"}
        acct = {v.tag: float(v.value)
                for v in ib.accountSummary() if v.tag in wanted}

        net_liq = acct.get("NetLiquidation", 0.0)
        gross   = acct.get("GrossPositionValue", 0.0)

        try:
            from kairos_confluence import lookup_sector
        except ImportError:
            lookup_sector = lambda sym: "unknown"

        # Resilient marks: one IBKR pull (live-preferred, delayed fallback),
        # replacing the rate-limited yfinance per-position lookup.
        _all_pos = ib.positions()
        if not _all_pos and float(acct.get("GrossPositionValue", 0.0) or 0) > 0:
            # Empty positions but the account holds value → a slow/just-restarted
            # Gateway returned nothing; force a fresh request before trusting 0.
            try:
                ib.reqPositions()
                ib.sleep(3)
                _all_pos = ib.positions()
            except Exception:
                pass
        _marks = _ibkr_marks(ib, [pp.contract.symbol for pp in _all_pos
                                  if pp.contract.secType not in ("CRYPTO", "CFD")])
        _mark_modes = []
        positions, crypto_val = [], 0.0
        for p in _all_pos:
            sym = p.contract.symbol
            sec = p.contract.secType
            qty = float(p.position)
            avg = float(p.avgCost)
            is_crypto = sec in ("CRYPTO", "CFD") or sym in ("BTC", "ETH", "BTCUSD", "ETHUSD")
            raw_sector = "crypto" if is_crypto else lookup_sector(sym)
            
            # Price source: IBKR mark (live/delayed) → yfinance → avg cost.
            _mk = _marks.get(sym)
            if _mk and _mk[0] is not None:
                mkt_price = _mk[0]
                if _mk[1] is not None:
                    _mark_modes.append(_mk[1])
            else:
                try:
                    import yfinance as yf
                    hist = yf.Ticker(sym).history(period="1d", interval="1m")
                    mkt_price = float(hist['Close'].iloc[-1]) if not hist.empty else avg
                except Exception:
                    mkt_price = avg

            mkt = round(qty * mkt_price, 2)
            
            # Calculate unrealized P&L
            cost_basis = qty * avg
            unrealized_pnl = mkt - cost_basis
            
            positions.append({
                "symbol":        sym,
                "secType":       sec,
                "assetClass":    "crypto" if is_crypto else "equity",
                "sector":        _normalize_sector(raw_sector, sym) if not is_crypto else "Crypto",
                "quantity":      qty,
                "avg_cost":      round(avg, 2),
                "market_value":  mkt,
                "unrealized_pnl": round(unrealized_pnl, 2),
            })
            if is_crypto:
                crypto_val += mkt

        # Decide live-vs-stale from an ACTUAL live-data check, not from how many
        # positions happened to price (a slow/empty positions fetch must never be
        # read as "live"). If any position priced live we trust that; otherwise
        # probe a liquid reference (SPY) directly while still connected.
        if _mark_modes and all(m == 1 for m in _mark_modes):
            _live_ok = True
        elif _mark_modes and any(m in (3, 4) for m in _mark_modes):
            _live_ok = False
        else:
            _live_ok = _ibkr_live_available(ib)
        ib.disconnect()

        data_mode = "live" if _live_ok else "delayed"
        marks_stale = not _live_ok

        _equity_mv = sum(p["market_value"] for p in positions
                         if p.get("assetClass") != "crypto")
        cash_val = acct.get("TotalCashValue", 0.0)   # cash is feed-independent
        computed_nlv = round(cash_val + _equity_mv + crypto_val, 2)
        unreal = acct.get("UnrealizedPnL", 0.0)
        nlv_source = "ibkr_account"
        # Override with freshly-marked (delayed) figures ONLY when positions are
        # actually loaded — never when the positions fetch came back empty (that
        # would collapse NLV to just cash and badly understate the account).
        if marks_stale and positions and _equity_mv > 0:
            net_liq = computed_nlv
            gross = round(_equity_mv + crypto_val, 2)
            unreal = round(sum(p["unrealized_pnl"] for p in positions
                               if p.get("unrealized_pnl") is not None), 2)
            nlv_source = "computed_delayed"
        elif marks_stale:
            # Stale, but positions did not load — keep the account NLV (frozen) and
            # still flag the data as stale so the dashboard warns rather than lies.
            nlv_source = "ibkr_account_stale"

        return {
            "connected":       True,
            "data_mode":       data_mode,
            "marks_stale":     marks_stale,
            "nlv_source":      nlv_source,
            "computed_nlv":    computed_nlv,
            "net_liquidation": net_liq,
            "cash":            cash_val,
            "gross_positions": gross,
            "unrealized_pnl":  unreal,
            "realized_pnl":    acct.get("RealizedPnL", 0.0),
            "buying_power":    acct.get("BuyingPower", 0.0),
            "equity_value":    gross - crypto_val,
            "crypto_value":    crypto_val,
            "positions":       sorted(positions, key=lambda p: p["symbol"]),
        }
    except Exception as e:
        print(f"  IBKR unavailable: {e}")
        return {"connected": False}


# Map Kairos-internal sector labels to GICS-aligned display names.
# value_dividend is a mixed bag — map individual tickers where possible,
# otherwise fall back to the broad category.
_VALUE_DIVIDEND_OVERRIDES = {
    # Energy
    "CVX": "Energy", "COP": "Energy", "OXY": "Energy", "HAL": "Energy",
    "DVN": "Energy", "PSX": "Energy", "VLO": "Energy", "MRO": "Energy",
    "KMI": "Energy", "WMB": "Energy", "ET": "Energy", "EPD": "Energy",
    "PXD": "Energy", "DOW": "Materials",
    # Financials
    "WFC": "Financials", "C": "Financials", "MS": "Financials",
    "AXP": "Financials", "TRV": "Financials",
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials",
    "LMT": "Industrials", "GE": "Industrials", "WM": "Industrials",
    "SWK": "Industrials", "MMM": "Industrials",
    # Communication Services
    "T": "Communication Services", "VZ": "Communication Services",
    "IBM": "Technology",
}

_SECTOR_NORMALIZE = {
    # Kairos internal labels → GICS-aligned title case
    "mega_cap":               "Large Cap",
    "large_cap":              "Large Cap",
    "mid_cap_growth":         "Mid Cap Growth",
    "consumer_tech":          "Consumer Tech",
    "energy_materials":       "Energy & Materials",
    "healthcare_biotech":     "Healthcare",
    "financials":             "Financials",
    "industrials_transport":  "Industrials",
    "reits_real_estate":      "Real Estate",
    "etf_broad_market":       "ETF — Broad Market",
    "etf_sector":             "ETF — Sector",
    "etf_thematic":           "ETF — Thematic",
    "etf_fixed_income":       "ETF — Fixed Income",
    "etf_volatility_commodity": "ETF — Commodity/Vol",
    "unknown":                "Other",
}


def _normalize_sector(raw: str, symbol: str = "") -> str:
    """Normalize a sector label to title-case GICS-aligned name."""
    # value_dividend: use per-ticker GICS override
    if raw == "value_dividend":
        return _VALUE_DIVIDEND_OVERRIDES.get(symbol, "Value & Dividend")

    # Check explicit mapping
    if raw in _SECTOR_NORMALIZE:
        return _SECTOR_NORMALIZE[raw]

    # Already looks like a proper name (from Tier B) — just title-case it
    return raw.replace("_", " ").title()


def _compute_sector_exposure(positions: list) -> dict[str, float]:
    """Return {sector: total_market_value} for equity (non-crypto, non-sim) positions."""
    try:
        from kairos_confluence import lookup_sector
    except ImportError:
        return {}

    exposure: dict[str, float] = {}
    for p in positions:
        sym = p.get("symbol", "")
        if "(SIM)" in sym:
            continue
        if p.get("assetClass", "equity") == "crypto":
            continue
        mkt_val = float(p.get("market_value", 0) or 0)
        if mkt_val <= 0:
            continue
        raw_sector = lookup_sector(sym)
        sector = _normalize_sector(raw_sector, sym)
        exposure[sector] = exposure.get(sector, 0.0) + mkt_val

    return exposure


# Donut buckets for the AI Value Chain Exposure chart. Order is fixed so the
# chart slices and colors line up deterministically; colors match Kairos brand.
_CHAIN_TIER_BUCKETS = [
    ("tier1", "Tier 1 — Direct AI",        "#00D4FF"),  # bright cyan
    ("tier2", "Tier 2 — Infrastructure",   "#7B61FF"),  # purple
    ("tier3", "Tier 3 — Suppliers",        "#FF6B35"),  # orange
    ("none",  "Non-Chain",                 "#4A5568"),  # muted grey
]


def compute_chain_tier_breakdown(price_map: dict[str, float] | None = None) -> dict:
    """Build donut-chart data bucketing open holdings by AI value-chain tier.

    Reads open holdings (sold_date IS NULL) from kairos.db and classifies each
    ticker with kairos_signals_chain.get_chain_tier(). Market value per bucket is
    quantity * current_price, where current_price comes from the live IBKR price
    map (ticker -> price/share) when available, falling back to the holding's
    entry_price (the holdings table has no current_price column).

    Returns a dict with a 'title' and four ordered 'buckets' (label/value/color).
    Falls back to four zero-value buckets if kairos_signals_chain can't be
    imported or the holdings table is empty/unreadable, rather than raising.
    """
    title = "AI Value Chain Exposure"

    def _empty() -> dict:
        return {
            "title": title,
            "buckets": [
                {"label": label, "value": 0.0, "color": color}
                for _key, label, color in _CHAIN_TIER_BUCKETS
            ],
        }

    try:
        from kairos_signals_chain import get_chain_tier
    except Exception as exc:  # ImportError or any load-time failure
        print(f"  WARNING: chain-tier import failed: {exc}")
        return _empty()

    price_map = {str(k).upper(): v for k, v in (price_map or {}).items()}

    if not os.path.exists(DB_PATH):
        return _empty()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ticker, quantity, entry_price FROM holdings "
            "WHERE sold_date IS NULL"
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"  WARNING: chain-tier holdings query failed: {exc}")
        return _empty()

    if not rows:
        return _empty()

    totals = {key: 0.0 for key, _label, _color in _CHAIN_TIER_BUCKETS}
    for r in rows:
        ticker = (r["ticker"] or "").strip()
        if not ticker:
            continue
        qty = float(r["quantity"] or 0)
        price = price_map.get(ticker.upper())
        if price is None:
            price = float(r["entry_price"] or 0)  # fallback: cost basis
        mkt_val = qty * float(price)

        tier = get_chain_tier(ticker)
        if tier == 1:
            totals["tier1"] += mkt_val
        elif tier == 2:
            totals["tier2"] += mkt_val
        elif tier == 3:
            totals["tier3"] += mkt_val
        else:
            totals["none"] += mkt_val

    return {
        "title": title,
        "buckets": [
            {"label": label, "value": round(totals[key], 2), "color": color}
            for key, label, color in _CHAIN_TIER_BUCKETS
        ],
    }


def _merge_sim_positions(ibkr_positions: list) -> list:
    """Append simulated crypto positions from crypto_holdings to the IBKR list."""
    try:
        import requests
        from kairos_log_db import get_crypto_holdings
        open_lots = get_crypto_holdings()
        
        # If IBKR positions are empty (IBKR offline), we cannot show accurate positions
        # since we don't have real-time market data. Leave positions empty to avoid
        # showing misleading fabricated data.
        # Note: This is intentional - we show "No open positions (or IBKR offline)" 
        # rather than inventing prices that would give false P&L figures.
        
        if not open_lots:
            return ibkr_positions

        # Aggregate open lots by asset
        agg: dict[str, dict] = {}
        for lot in open_lots:
            a = lot["asset"]
            if a not in agg:
                agg[a] = {"qty": 0.0, "cost": 0.0}
            agg[a]["qty"] += lot["quantity"]
            agg[a]["cost"] += lot["entry_price"] * lot["quantity"]

        # Fetch current prices from CoinGecko for all sim assets
        coingecko_ids = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
            "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "matic-network",
            "LINK": "chainlink",
        }
        ids_str = ",".join(coingecko_ids[a] for a in agg if a in coingecko_ids)
        prices = {}
        if ids_str:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": ids_str, "vs_currencies": "usd"}, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            for asset, cg_id in coingecko_ids.items():
                if cg_id in data:
                    prices[asset] = data[cg_id]["usd"]

        positions = list(ibkr_positions)
        for asset, info in agg.items():
            qty = info["qty"]
            avg_cost = round(info["cost"] / qty, 2) if qty else 0
            mkt_price = prices.get(asset, avg_cost)
            mkt_value = round(qty * mkt_price, 2)
            unrealized = round(mkt_value - info["cost"], 2)
            positions.append({
                "symbol": f"{asset} (SIM)",
                "secType": "CRYPTO",
                "assetClass": "crypto",
                "sector": "Crypto",
                "quantity": qty,
                "avg_cost": avg_cost,
                "market_value": mkt_value,
                "unrealized_pnl": unrealized,
            })

        return sorted(positions, key=lambda p: p["symbol"])
    except Exception as e:
        print(f"  Sim positions unavailable: {e}")
        return ibkr_positions


# ── Performance history ────────────────────────────────────────────────

def load_perf() -> dict:
    if os.path.exists(PERF_FILE):
        with open(PERF_FILE) as f:
            return json.load(f)
    return {"start_date": None, "start_value": None, "snapshots": []}


def save_perf(perf: dict):
    with open(PERF_FILE, "w") as f:
        json.dump(perf, f, indent=2)


def upsert_snapshot(perf: dict, ibkr: dict) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    net   = ibkr.get("net_liquidation", 0.0)

    if not perf["start_date"]:
        perf["start_date"]  = today
        perf["start_value"] = net

    snap = {
        "date":         today,
        "value":        round(net, 2),
        "equity_value": round(ibkr.get("equity_value", 0.0), 2),
        "crypto_value": round(ibkr.get("crypto_value", 0.0), 2),
        "cash":         round(ibkr.get("cash", 0.0), 2),
    }
    perf["snapshots"] = [s for s in perf["snapshots"] if s["date"] != today]
    perf["snapshots"].append(snap)
    perf["snapshots"].sort(key=lambda x: x["date"])
    return perf


# ── Database ───────────────────────────────────────────────────────────

def query_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {"decisions": [], "holdings": []}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Equity decisions (with optional outcomes join)
    equity_rows = conn.execute("""
        SELECT d.id, d.timestamp, d.ticker, d.action, d.quantity,
               d.rationale, d.execution_price, d.execution_status,
               d.commission, d.net_liq_after, d.conviction_trade,
               'equity' AS asset_class,
               o.pnl, o.hold_pnl
        FROM decisions d
        LEFT JOIN outcomes o ON o.decision_id = d.id
        WHERE (d.execution_status != 'Skipped' OR d.execution_status IS NULL)
        ORDER BY d.id DESC LIMIT 50
    """).fetchall()

    # Crypto decisions (separate table, different column names)
    crypto_rows = conn.execute("""
        SELECT id, timestamp, asset AS ticker, action, quantity,
               rationale, execution_price, execution_status,
               commission, NULL AS net_liq_after, 0 AS conviction_trade,
               'crypto' AS asset_class,
               NULL AS pnl, NULL AS hold_pnl
        FROM crypto_decisions
        WHERE (execution_status != 'Skipped' OR execution_status IS NULL)
        ORDER BY id DESC LIMIT 50
    """).fetchall()

    # Merge and sort by timestamp descending, limit to 50
    decisions = [dict(r) for r in equity_rows] + [dict(r) for r in crypto_rows]
    decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    decisions = decisions[:50]

    # Counts for the summary line (includes skipped, both tables)
    eq_counts = conn.execute("""
        SELECT
            SUM(CASE WHEN execution_status != 'Skipped' THEN 1 ELSE 0 END) AS executed,
            SUM(CASE WHEN execution_status = 'Skipped' THEN 1 ELSE 0 END) AS skipped
        FROM decisions
    """).fetchone()
    cr_counts = conn.execute("""
        SELECT
            SUM(CASE WHEN execution_status != 'Skipped' THEN 1 ELSE 0 END) AS executed,
            SUM(CASE WHEN execution_status = 'Skipped' THEN 1 ELSE 0 END) AS skipped
        FROM crypto_decisions
    """).fetchone()
    decisions_counts = {
        "executed": (eq_counts["executed"] or 0) + (cr_counts["executed"] or 0),
        "skipped":  (eq_counts["skipped"] or 0) + (cr_counts["skipped"] or 0),
    }

    holdings = [dict(r) for r in conn.execute("""
        SELECT *,
            CAST(julianday(COALESCE(sold_date, datetime('now')))
                 - julianday(entry_date) AS INTEGER) AS holding_days
        FROM holdings ORDER BY entry_date DESC
    """).fetchall()]

    conn.close()
    return {"decisions": decisions, "holdings": holdings,
            "decisions_counts": decisions_counts}


# ── Metrics ────────────────────────────────────────────────────────────

def compute_metrics(perf: dict, db: dict) -> dict:
    snaps      = perf.get("snapshots", [])
    start_val  = perf.get("start_value") or 1_000_000.0
    start_date = perf.get("start_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    curr_val   = snaps[-1]["value"] if snaps else start_val

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_run = max(1, (datetime.now(timezone.utc) - start_dt).days)
    years    = days_run / 365.25

    total_ret  = (curr_val - start_val) / start_val * 100
    if days_run >= 7 and years > 0 and curr_val > 0 and start_val > 0:
        annualized = ((curr_val / start_val) ** (1.0 / years) - 1.0) * 100
    else:
        annualized = None  # Insufficient data for annualized calc

    equity_val = snaps[-1].get("equity_value", 0) if snaps else 0
    crypto_val = snaps[-1].get("crypto_value", 0) if snaps else 0
    cash_val   = snaps[-1].get("cash", 0) if snaps else 0
    if cash_val == 0 and curr_val > 0:
        cash_val = curr_val - equity_val - crypto_val  # fallback: derive from NLV

    equity_start = next((s["equity_value"] for s in snaps if s.get("equity_value", 0) > 0), None)
    crypto_start = next((s["crypto_value"] for s in snaps if s.get("crypto_value", 0) > 0), None)
    equity_return = (equity_val - equity_start) / equity_start * 100 if equity_start else None
    crypto_return = (crypto_val - crypto_start) / crypto_start * 100 if crypto_start else None

    decs   = db.get("decisions", [])
    trades = [d for d in decs if d.get("action", "HOLD").upper() in ("BUY", "SELL")]
    filled = [d for d in trades if d.get("execution_status") == "Filled"]

    # Win rate: realized (closed) positions only — exclude open/unrealized
    closed = [h for h in db.get("holdings", [])
              if h.get("sold_date") and h.get("sold_price")]
    wins   = [h for h in closed if h["sold_price"] > h["entry_price"]]
    win_rt = len(wins) / len(closed) * 100 if closed else None

    max_dd, peak = 0.0, start_val
    for s in snaps:
        v      = s["value"]
        peak   = max(peak, v)
        dd     = (peak - v) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    sharpe = None
    if len(snaps) >= 10:
        dr = [(snaps[i]["value"] - snaps[i-1]["value"]) / snaps[i-1]["value"]
              for i in range(1, len(snaps)) if snaps[i-1]["value"] > 0]
        if len(dr) >= 5:
            avg_r  = statistics.mean(dr)
            std_r  = statistics.stdev(dr) if len(dr) > 1 else 0
            rf     = 0.05 / 252
            sharpe = round((avg_r - rf) / std_r * math.sqrt(252), 2) if std_r > 0 else 0.0

    pnls         = [d["pnl"] for d in decs if d.get("pnl") is not None]
    largest_loss = round(min(pnls), 2) if pnls else 0.0
    total_pnl    = round(sum(pnls), 2) if pnls else 0.0

    return {
        "start_value":       round(start_val, 2),
        "current_value":     round(curr_val, 2),
        "equity_value":      round(equity_val, 2),
        "crypto_value":      round(crypto_val, 2),
        "cash_value":        round(cash_val, 2),
        "cash_pct":          round(cash_val / curr_val * 100, 1) if curr_val > 0 else 0.0,
        "total_return_pct":  round(total_ret, 3),
        "total_return_usd":  round(curr_val - start_val, 2),
        "annualized_return": round(annualized, 2) if annualized is not None else None,
        "days_running":      days_run,
        "advisor_rate":      round(ADVISOR_RATE * 100, 1),
        "target_rate":       round(TARGET_RATE  * 100, 1),
        "vs_advisor":        round(annualized - ADVISOR_RATE * 100, 2) if annualized is not None else None,
        "vs_target":         round(annualized - TARGET_RATE  * 100, 2) if annualized is not None else None,
        "advisor_weekly":    round(ADVISOR_RATE / 52 * 100, 4),
        "target_weekly":     round(TARGET_RATE  / 52 * 100, 4),
        "total_trades":      len(trades),
        "filled_trades":     len(filled),
        "win_rate":          round(win_rt, 1) if win_rt is not None else None,
        "closed_trades":     len(closed),
        "max_drawdown":      round(max_dd, 2),
        "sharpe_ratio":      sharpe,
        "largest_loss":      largest_loss,
        "realized_pnl":      total_pnl,
        "equity_return":     round(equity_return, 2) if equity_return is not None else None,
        "crypto_return":     round(crypto_return, 2) if crypto_return is not None else None,
    }


# ── Chart series ───────────────────────────────────────────────────────

def make_series(perf: dict) -> dict:
    snaps      = perf.get("snapshots", [])
    start_val  = perf.get("start_value") or 1_000_000.0
    start_date = perf.get("start_date")
    if not start_date:
        start_date = snaps[0]["date"] if snaps else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dates, actual, advisor, target, equity, crypto = [], [], [], [], [], []

    for s in snaps:
        dt  = datetime.strptime(s["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        yrs = (dt - start_dt).days / 365.25
        dates.append(s["date"])
        actual.append(round(s["value"], 2))
        advisor.append(round(start_val * (1 + ADVISOR_RATE) ** yrs, 2))
        target.append(round(start_val  * (1 + TARGET_RATE)  ** yrs, 2))
        equity.append(round(s.get("equity_value", 0), 2))
        crypto.append(round(s.get("crypto_value", 0), 2))

    # Always include today with projections even if no snapshot yet
    if not dates:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dates   = [today]
        actual  = [start_val]
        advisor = [start_val]
        target  = [start_val]
        equity  = [0]
        crypto  = [0]

    return {
        "dates": dates, "actual": actual,
        "advisor": advisor, "target": target,
        "equity": equity, "crypto": crypto,
    }


def make_weekly(perf: dict) -> dict:
    snaps = perf.get("snapshots", [])
    if len(snaps) < 2:
        return {"labels": [], "returns": [], "advisor": [], "target": []}

    wk_data: dict = {}
    for s in snaps:
        dt = datetime.strptime(s["date"], "%Y-%m-%d")
        wk = dt.strftime("%Y-W%V")
        if wk not in wk_data:
            wk_data[wk] = {"start": s["value"], "end": s["value"]}
        else:
            wk_data[wk]["end"] = s["value"]

    labels, rets = [], []
    for k in sorted(wk_data):
        labels.append(k)
        w   = wk_data[k]
        ret = (w["end"] - w["start"]) / w["start"] * 100 if w["start"] > 0 else 0
        rets.append(round(ret, 4))

    n = len(labels)
    return {
        "labels":  labels,
        "returns": rets,
        "advisor": [round(ADVISOR_RATE / 52 * 100, 4)] * n,
        "target":  [round(TARGET_RATE  / 52 * 100, 4)] * n,
    }


# ── Ledger + Screen log loaders ────────────────────────────────────────

def load_ledger() -> dict:
    """Parse kairos_ledger.txt into pattern summary + trade lines."""
    if not os.path.exists(LEDGER_FILE):
        return {"pattern_summary": "", "trades": []}

    with open(LEDGER_FILE, "r") as f:
        raw = f.read()

    # Split into header and trade lines
    header_lines = []
    trade_lines = []
    in_trades = False
    for line in raw.splitlines():
        if line.strip() == "--- TRADE LOG ---":
            in_trades = True
            continue
        if line.strip() == "--- PATTERN SUMMARY ---":
            continue
        if in_trades:
            if line.strip():
                trade_lines.append(line.strip())
        else:
            header_lines.append(line)

    # Parse trade lines into structured dicts
    trades = []
    for tl in trade_lines:
        parts = [p.strip() for p in tl.split("|")]
        if len(parts) >= 6:
            trades.append({
                "date": parts[0],
                "ticker": parts[1],
                "action": parts[2],
                "signals": parts[3],
                "pnl_pct": parts[4],
                "verdict": parts[5],
            })

    trades.sort(key=lambda t: t["date"])

    return {
        "pattern_summary": "\n".join(header_lines).strip(),
        "trades": trades,
    }


def load_screen_log() -> dict | None:
    """Load the most recent entry from kairos_screen_log.json."""
    if not os.path.exists(SCREEN_LOG):
        return None

    last_line = None
    with open(SCREEN_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        return None

    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


# ── Payload builder ────────────────────────────────────────────────────

def load_nlv_snapshots() -> list[dict]:
    """Daily NLV snapshots from kairos.db, oldest→newest. Empty list on any error."""
    try:
        from kairos_log_db import get_connection, init_db
        init_db()
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT snapshot_date, nlv, total_cash, invested, num_positions, "
                "unrealized_pnl, realized_pnl_cum FROM nlv_snapshots "
                "ORDER BY snapshot_date ASC"
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"  WARNING: nlv_snapshots load failed: {exc}")
        return []


def _snapshot_on_or_before(snaps: list[dict], target_date) -> dict | None:
    """Latest snapshot whose date is on or before target_date (a date object)."""
    chosen = None
    for s in snaps:
        try:
            d = datetime.strptime(s["snapshot_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError, KeyError):
            continue
        if d <= target_date:
            chosen = s
        else:
            break
    return chosen


def compute_portfolio_metrics(ibkr: dict, snaps: list[dict]) -> dict:
    """Curated portfolio-metrics panel: Snapshot (now) + Trends (time-series).

    "Now" metrics come from live IBKR when connected, else the latest snapshot.
    Trend metrics need the nlv_snapshots history and stay None (rendered "—")
    until >=2 days exist. No projections.
    """
    latest = snaps[-1] if snaps else None
    connected = bool(ibkr.get("connected"))

    # Equity positions (exclude crypto) for concentration + best/worst.
    positions = [p for p in ibkr.get("positions", [])
                 if p.get("assetClass", "equity") != "crypto"]

    # ── NLV / cash: prefer live, fall back to latest snapshot ─────────
    nlv = ibkr.get("net_liquidation") if connected else (latest or {}).get("nlv")
    cash = ibkr.get("cash") if connected else (latest or {}).get("total_cash")
    unrealized = ibkr.get("unrealized_pnl") if connected else (latest or {}).get("unrealized_pnl")

    cash_pct = round(cash / nlv * 100.0, 2) if (nlv and cash is not None) else None
    invested_pct = round((nlv - cash) / nlv * 100.0, 2) if (nlv and cash is not None) else None

    # realized P&L (cumulative closed): authoritative from the closed-lot ledger.
    realized_cum = None
    try:
        from kairos_log_db import get_connection
        from kairos_execute import _realized_pnl_cumulative
        conn = get_connection()
        try:
            realized_cum = _realized_pnl_cumulative(conn)
        finally:
            conn.close()
    except Exception:
        realized_cum = (latest or {}).get("realized_pnl_cum")

    # ── Concentration + best/worst (live positions only) ──────────────
    largest_pct = top5_pct = None
    best = worst = None
    if positions and nlv:
        mvals = sorted((float(p.get("market_value") or 0.0) for p in positions), reverse=True)
        if mvals:
            largest_pct = round(mvals[0] / nlv * 100.0, 2)
            top5_pct = round(sum(mvals[:5]) / nlv * 100.0, 2)

    ranked = []
    n_profit = n_loss = 0
    open_cost_basis_sum = 0.0
    open_upnl_sum = 0.0
    for p in positions:
        upnl = p.get("unrealized_pnl")
        mv = float(p.get("market_value") or 0.0)
        if upnl is None:
            continue
        # Book breadth: count winners vs losers across open positions.
        if float(upnl) >= 0:
            n_profit += 1
        else:
            n_loss += 1
        cost_basis = mv - float(upnl)
        if cost_basis <= 0:
            continue
        # Aggregate for blended open-book return %.
        open_cost_basis_sum += cost_basis
        open_upnl_sum += float(upnl)
        pct = float(upnl) / cost_basis * 100.0
        ranked.append({"ticker": p.get("symbol", "").split(" ")[0], "pct": round(pct, 2)})
    if ranked:
        best = max(ranked, key=lambda r: r["pct"])
        worst = min(ranked, key=lambda r: r["pct"])
    # Blended unrealized return on the open book (distinct from total return,
    # which blends in realized P&L). None until we have positions with basis.
    unrealized_return_pct = (round(open_upnl_sum / open_cost_basis_sum * 100.0, 2)
                             if open_cost_basis_sum > 0 else None)

    # ── Trend metrics (need history) ──────────────────────────────────
    daily = weekly = drawdown = ret_30d = None
    if latest and len(snaps) >= 2:
        try:
            latest_date = datetime.strptime(latest["snapshot_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError, KeyError):
            latest_date = None
        cur_nlv = latest.get("nlv")

        # Daily: latest vs the immediately prior snapshot.
        prior = snaps[-2]
        if cur_nlv and prior.get("nlv"):
            d_usd = round(cur_nlv - prior["nlv"], 2)
            daily = {"usd": d_usd, "pct": round(d_usd / prior["nlv"] * 100.0, 2)}

        # Weekly: latest vs the snapshot on/before 7 calendar days ago.
        if latest_date and cur_nlv:
            wk = _snapshot_on_or_before(snaps, latest_date - timedelta(days=7))
            if wk and wk is not latest and wk.get("nlv"):
                w_usd = round(cur_nlv - wk["nlv"], 2)
                weekly = {"usd": w_usd, "pct": round(w_usd / wk["nlv"] * 100.0, 2)}

            # 30-day return.
            m = _snapshot_on_or_before(snaps, latest_date - timedelta(days=30))
            if m and m is not latest and m.get("nlv"):
                ret_30d = round((cur_nlv - m["nlv"]) / m["nlv"] * 100.0, 2)

        # Drawdown from peak NLV across the whole series.
        peak = max((s.get("nlv") or 0.0) for s in snaps)
        if peak and cur_nlv:
            drawdown = round((cur_nlv - peak) / peak * 100.0, 2)

    return {
        "have_live": connected,
        "as_of": (latest or {}).get("snapshot_date"),
        "snapshot_days": len(snaps),
        # Snapshot (now)
        "open_positions": len(positions) if (connected or not latest)
                          else (latest or {}).get("num_positions"),
        "positions_in_profit": n_profit,
        "positions_in_loss": n_loss,
        "unrealized_return_pct": unrealized_return_pct,
        "nlv": round(nlv, 2) if nlv is not None else None,
        "cash": round(cash, 2) if cash is not None else None,
        "cash_pct": cash_pct,
        "invested_pct": invested_pct,
        "unrealized_pnl": round(unrealized, 2) if unrealized is not None else None,
        "realized_pnl_cum": realized_cum,
        "largest_pct": largest_pct,
        "top5_pct": top5_pct,
        "best": best,
        "worst": worst,
        # Trends (None until enough history)
        "daily": daily,
        "weekly": weekly,
        "drawdown_pct": drawdown,
        "return_30d_pct": ret_30d,
    }


def build_payload(perf: dict, db: dict, ibkr: dict) -> dict:
    decisions_display = []
    for d in db.get("decisions", [])[:20]:
        decisions_display.append({
            "date":      (d.get("timestamp") or "")[:10],
            "ticker":    d.get("ticker", ""),
            "action":    d.get("action", "HOLD"),
            "quantity":  d.get("quantity", 0),
            "price":     d.get("execution_price"),
            "status":    d.get("execution_status", ""),
            "pnl":       d.get("pnl"),
            "rationale": (d.get("rationale") or "")[:150],
            "conviction_trade": bool(d.get("conviction_trade", 0)),
            "asset_class": d.get("asset_class", "equity"),
        })

    positions = _merge_sim_positions(ibkr.get("positions", []))

    # Resolve company names for all positions
    try:
        raw_symbols = [p["symbol"].split(" ")[0] for p in positions]  # strip (SIM) suffix
        name_map = _resolve_ticker_names(raw_symbols)
        for p in positions:
            raw_sym = p["symbol"].split(" ")[0]
            p["name"] = name_map.get(raw_sym, "")
    except Exception as _name_err:
        print(f"  WARNING: Name resolution failed: {_name_err}")

    # Live price/share map (ticker -> current price) from IBKR positions, used to
    # value open holdings at market for the AI value-chain donut. Strips (SIM).
    price_map: dict[str, float] = {}
    for p in positions:
        raw_sym = p.get("symbol", "").split(" ")[0]
        qty = float(p.get("quantity", 0) or 0)
        mkt_val = float(p.get("market_value", 0) or 0)
        if raw_sym and qty:
            price_map[raw_sym.upper()] = mkt_val / qty

    # Build the payload dictionary
    return {
        "metrics":        compute_metrics(perf, db),
        "portfolio_metrics": compute_portfolio_metrics(ibkr, load_nlv_snapshots()),
        "series":         make_series(perf),
        "weekly":         make_weekly(perf),
        "decisions":      decisions_display,
        "decisions_counts": db.get("decisions_counts", {"executed": 0, "skipped": 0}),
        "positions":      positions,
        "sector_breakdown": _compute_sector_exposure(positions),
        "chain_tier_breakdown": compute_chain_tier_breakdown(price_map),
        "ibkr_connected": ibkr.get("connected", False),
        "marks_stale":    ibkr.get("marks_stale", False),
        "data_mode":      ibkr.get("data_mode", "unknown"),
        "nlv_source":     ibkr.get("nlv_source", ""),
        "start_date":     perf.get("start_date", ""),
        "snapshot_count": len(perf.get("snapshots", [])),
        "ledger":         load_ledger(),
        "screen_log":     load_screen_log(),
    }


# ── HTML template ──────────────────────────────────────────────────────

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kairos Performance Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #080d1c;
      --surface: #0d1426;
      --surface2: #111c35;
      --border: #1a2845;
      --border2: #243560;
      --text: #c8d8f0;
      --dim: #6a80a8;
      --muted: #3a4a68;
      --cyan: #00ccff;
      --cyan-dim: rgba(0,204,255,0.15);
      --green: #00e676;
      --green-dim: rgba(0,230,118,0.15);
      --amber: #ffaa00;
      --amber-dim: rgba(255,170,0,0.15);
      --red: #ff4466;
      --red-dim: rgba(255,68,102,0.15);
      --purple: #bb66ff;
      --purple-dim: rgba(187,102,255,0.15);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "SF Mono","Consolas","Monaco","Courier New",monospace;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 24px 28px;
      font-size: 13px;
    }
    /* ── Header ── */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--border);
    }
    .logo { font-size: 20px; font-weight: 700; color: var(--cyan); letter-spacing: 4px; }
    .logo span { color: var(--dim); font-weight: 400; }
    .header-sub { font-size: 11px; color: var(--dim); margin-top: 5px; letter-spacing: 1px; }
    .header-right { text-align: right; font-size: 11px; color: var(--dim); }
    .dot {
      display: inline-block; width: 7px; height: 7px;
      border-radius: 50%; margin-right: 5px; vertical-align: middle;
    }
    .dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); animation: blink 2s infinite; }
    .dot-amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
    @keyframes blink { 0%,100% { opacity:1 } 50% { opacity:.3 } }
    /* ── Refresh button ── */
    .refresh-row { display: flex; align-items: center; gap: 10px; margin-top: 8px; justify-content: flex-end; }
    .refresh-btn {
      background: var(--surface2); color: var(--cyan); border: 1px solid var(--border2);
      border-radius: 6px; padding: 5px 14px; font-size: 11px; font-family: inherit;
      cursor: pointer; letter-spacing: 0.5px; transition: all 0.2s;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .refresh-btn:hover { background: var(--cyan-dim); border-color: var(--cyan); }
    .refresh-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .refresh-btn .spinner {
      display: none; width: 12px; height: 12px; border: 2px solid var(--border2);
      border-top-color: var(--cyan); border-radius: 50%; animation: spin 0.8s linear infinite;
    }
    .refresh-btn.loading .spinner { display: inline-block; }
    .refresh-btn.loading .btn-icon { display: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .last-updated { font-size: 10px; color: var(--muted); }
    /* ── Metric grid ── */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 18px;
    }
    .mcard {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      position: relative;
      overflow: hidden;
    }
    .mcard::after {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
    }
    .mcard.c-cyan::after   { background: var(--cyan); }
    .mcard.c-green::after  { background: var(--green); }
    .mcard.c-amber::after  { background: var(--amber); }
    .mcard.c-red::after    { background: var(--red); }
    .mcard.c-purple::after { background: var(--purple); }
    .mcard.c-white::after  { background: var(--border2); }
    .mlabel { font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--dim); margin-bottom: 7px; }
    .mval { font-size: 22px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .mval.cyan   { color: var(--cyan); }
    .mval.green  { color: var(--green); }
    .mval.amber  { color: var(--amber); }
    .mval.red    { color: var(--red); }
    .mval.purple { color: var(--purple); }
    .mval.white  { color: var(--text); }
    .msub { font-size: 10px; color: var(--dim); }
    /* ── Portfolio metrics panel ── */
    .pm-subhdr { font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin: 4px 0 12px; }
    .pm-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 16px; }
    .pm-tile { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 11px 13px; }
    .pm-label { font-size: 8.5px; letter-spacing: 1.2px; text-transform: uppercase; color: var(--dim); margin-bottom: 6px; }
    .pm-val { font-size: 17px; font-weight: 700; line-height: 1.05; color: var(--text); }
    .pm-sub { font-size: 9.5px; color: var(--dim); margin-top: 3px; }
    @media(max-width:1100px) { .pm-grid { grid-template-columns: repeat(3,1fr); } }
    @media(max-width:600px)  { .pm-grid { grid-template-columns: repeat(2,1fr); } }
    /* ── Cards ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px 22px;
      margin-bottom: 18px;
    }
    .section-hdr {
      font-size: 9px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--dim);
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .section-hdr::after { content: ""; flex: 1; height: 1px; background: var(--border); }
    /* ── Charts ── */
    .chart-h280 { position: relative; height: 280px; }
    .chart-h340 { position: relative; height: 340px; }
    .charts-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
    .charts-2col .card { margin-bottom: 0; }
    .risk-3col { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }
    .rcard {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      text-align: center;
    }
    .rlabel { font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--dim); margin-bottom: 8px; }
    .rval { font-size: 28px; font-weight: 700; }
    .rsub { font-size: 10px; color: var(--dim); margin-top: 4px; }
    /* ── Tables ── */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th {
      text-align: left; color: var(--dim); font-size: 9px;
      letter-spacing: 1.2px; text-transform: uppercase;
      padding: 8px 12px; border-bottom: 1px solid var(--border);
    }
    td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255,255,255,0.02); }
    .tag-BUY  { color: var(--green); font-weight: 700; }
    .tag-SELL { color: var(--red); font-weight: 700; }
    .tag-HOLD { color: var(--dim); }
    .tag-Filled    { color: var(--green); font-size: 10px; }
    .tag-Submitted { color: var(--amber); font-size: 10px; }
    .tag-Skipped   { color: var(--dim); font-size: 10px; }
    .tag-Cancelled { color: var(--red); font-size: 10px; }
    .tag-Pending   { color: var(--amber); font-size: 10px; }
    .pnl-pos { color: var(--green); }
    .pnl-neg { color: var(--red); }
    .rat { color: var(--dim); font-size: 11px; line-height: 1.4; max-width: 380px; }
    .no-data { text-align: center; color: var(--muted); padding: 28px; letter-spacing: 1px; }
    /* ── Responsive ── */
    @media(max-width:1100px) {
      .metrics-grid { grid-template-columns: repeat(2,1fr); }
      .charts-2col  { grid-template-columns: 1fr; }
      .risk-3col    { grid-template-columns: repeat(2,1fr); }
    }
    @media(max-width:600px) { body { padding: 12px; } .metrics-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>

<!-- ── Header ── -->
<div class="header">
  <div>
    <div class="logo">KAIROS <span>PERFORMANCE</span></div>
    <div class="header-sub">Paper Trading &middot; IBKR &middot; AI-driven decisions</div>
  </div>
  <div class="header-right">
    <div id="ibkr-status-line"><span class="dot dot-amber" id="status-dot"></span><span id="status-txt">Loading&hellip;</span></div>
    <div style="margin-top:5px">__TIMESTAMP__</div>
    <div style="margin-top:3px" id="snap-count"></div>
    <div class="refresh-row">
      <span class="last-updated" id="last-updated"></span>
      <button class="refresh-btn" id="refresh-btn" onclick="refreshDashboard()">
        <span class="spinner"></span>
        <span class="btn-icon">&#x21bb;</span>
        Refresh
      </button>
    </div>
  </div>
</div>

<!-- ── Metric Cards ── -->
<div class="metrics-grid" id="metrics-grid"></div>

<!-- ── Portfolio Metrics (curated) ── -->
<div class="card" id="portfolio-metrics-card">
  <div class="section-hdr">Portfolio Metrics</div>
  <div class="pm-subhdr">Snapshot</div>
  <div class="pm-grid" id="pm-snapshot"></div>
  <div class="pm-subhdr">Trends</div>
  <div class="pm-grid" id="pm-trends"></div>
</div>

<!-- ── Portfolio Value Chart ── -->
<div class="card">
  <div class="section-hdr">Portfolio Value Over Time</div>
  <div class="chart-h340"><canvas id="portfolioChart"></canvas></div>
</div>

<!-- ── Weekly Returns + Asset Breakdown ── -->
<div class="charts-2col">
  <div class="card">
    <div class="section-hdr">Weekly Returns vs Benchmarks</div>
    <div class="chart-h280"><canvas id="weeklyChart"></canvas></div>
  </div>
  <div class="card">
    <div class="section-hdr">AI Value Chain Exposure</div>
    <div id="chainKpi" style="padding:18px 8px;"></div>
  </div>
</div>

<!-- ── Sector Breakdown ── -->
<div class="card" id="sector-card">
  <div class="section-hdr">Sector Exposure (Equity)</div>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;font-size:10px;color:var(--dim)">
    <span style="display:inline-flex;align-items:center;gap:5px">
      <span style="display:inline-block;width:12px;height:12px;background:rgba(0,204,255,0.75);border:1px solid #00ccff;border-radius:2px"></span>Normal
    </span>
    <span style="display:inline-flex;align-items:center;gap:5px">
      <span style="display:inline-block;width:12px;height:12px;background:rgba(255,68,102,0.8);border:1px solid #ff4466;border-radius:2px"></span>&gt;25% — over concentration limit
    </span>
  </div>
  <div class="chart-h280"><canvas id="sectorChart"></canvas></div>
</div>

<!-- ── Risk Metrics ── -->
<div class="risk-3col" id="risk-row"></div>

<!-- ── Positions ── -->
<div class="card">
  <div class="section-hdr">Current Positions</div>
  <div id="positions-wrap"></div>
</div>

<!-- ── Decision Log ── -->
<div class="card">
  <div class="section-hdr">Decision Log &mdash; Recent 20</div>
  <div id="decisions-wrap"></div>
</div>

<!-- ── Trading Lessons + Screening Stats ── -->
<div class="charts-2col">
  <div class="card">
    <div class="section-hdr">Trading Lessons &mdash; kairos_ledger.txt</div>
    <div id="ledger-wrap"></div>
  </div>
  <div class="card">
    <div class="section-hdr">Screening Stats &mdash; Last Run</div>
    <div id="screen-wrap"></div>
  </div>
</div>

<div style="height:32px"></div>

<script>
  const DATA = __DATA_JSON__;
  const M = DATA.metrics;
  const S = DATA.series;
  const W = DATA.weekly;
  const positions = DATA.positions || [];

  // ── Status ─────────────────────────────────────────────────────────
  document.getElementById("status-txt").textContent =
    DATA.ibkr_connected ? "IBKR Connected" : "IBKR Offline \u2014 cached data";
  document.getElementById("status-dot").className =
    "dot " + (DATA.ibkr_connected ? "dot-green" : "dot-amber");
  // Override with a visible DELAYED-DATA badge when marks are stale (no live
  // IBKR entitlement) — the displayed NLV is computed from delayed marks.
  if (DATA.ibkr_connected && DATA.marks_stale) {
    // Connection is healthy even on delayed data → leave the dot GREEN (set
    // above). Show only a small, muted 'delayed' note instead of a loud banner.
    document.getElementById("status-txt").innerHTML =
      "IBKR Connected <span style='color:#8a8f99;font-weight:400;font-size:0.8em;margin-left:5px;opacity:0.85;'>"
      + "&middot; delayed marks</span>";
  }
  document.getElementById("snap-count").textContent =
    DATA.snapshot_count + " day" + (DATA.snapshot_count !== 1 ? "s" : "") + " of data";

  // ── Helpers ────────────────────────────────────────────────────────
  const fmtN = (n, d=2) => n == null ? "\u2014" :
    n.toLocaleString("en-US", {minimumFractionDigits:d, maximumFractionDigits:d});
  const fmtSign = (n, d=2) => n == null ? "\u2014" : (n >= 0 ? "+" : "") + fmtN(n, d);
  const fmtUSD  = (n) => n == null ? "\u2014" : "$" + fmtN(n, 0);
  const fmtPct  = (n, d=2) => n == null ? "\u2014" : fmtSign(n, d) + "%";
  const colClass = (n) => n == null ? "white" : n > 0 ? "green" : n < 0 ? "red" : "white";
  const accentOf = (n) => n == null ? "c-white" : n > 0 ? "c-green" : n < 0 ? "c-red" : "c-white";

  // ── Metric cards ───────────────────────────────────────────────────
  const cardDefs = [
    {
      label: "Total Return",
      val:   () => fmtPct(M.total_return_pct),
      sub:   () => fmtSign(M.total_return_usd, 0).replace(/^([+-])/, "$1$") + " net P&L",
      color: () => colClass(M.total_return_pct),
      accent:() => accentOf(M.total_return_pct),
    },
    {
      label: "Annualized Return",
      val:   () => fmtPct(M.annualized_return),
      sub:   () => M.annualized_return == null ? "Insufficient data (<7 days)" : "from " + M.days_running + " day" + (M.days_running !== 1 ? "s" : "") + " of data",
      color: () => colClass(M.annualized_return),
      accent:() => accentOf(M.annualized_return),
    },
    {
      label: "vs Advisor (14.4%)",
      val:   () => fmtPct(M.vs_advisor),
      sub:   () => M.vs_advisor >= 0 ? "\u25b2 Ahead of benchmark" : "\u25bc Behind benchmark",
      color: () => colClass(M.vs_advisor),
      accent:() => accentOf(M.vs_advisor),
    },
    {
      label: "vs Target (29.0%)",
      val:   () => fmtPct(M.vs_target),
      sub:   () => M.vs_target >= 0 ? "\u25b2 On track to double" : "\u25bc Gap to close",
      color: () => colClass(M.vs_target),
      accent:() => accentOf(M.vs_target),
    },
    {
      label: "Portfolio Value",
      val:   () => fmtUSD(M.current_value),
      sub:   () => "Started at " + fmtUSD(M.start_value),
      color: () => "cyan",
      accent:() => "c-cyan",
    },
    {
      label: "Cash Available",
      val:   () => fmtUSD(M.cash_value),
      sub:   () => fmtN(M.cash_pct, 1) + "% of NLV",
      color: () => "white",
      accent:() => "c-white",
    },
    {
      label: "Win Rate",
      val:   () => M.win_rate != null ? fmtN(M.win_rate, 1) + "%" : "\u2014",
      sub:   () => (M.closed_trades || 0) + " closed trade" + ((M.closed_trades || 0) !== 1 ? "s" : "") + (M.win_rate == null ? " (no exits yet)" : ""),
      color: () => M.win_rate != null && M.win_rate >= 50 ? "green" : M.win_rate != null && M.win_rate > 0 ? "amber" : "white",
      accent:() => M.win_rate != null && M.win_rate >= 50 ? "c-green" : M.win_rate != null && M.win_rate > 0 ? "c-amber" : "c-white",
    },
    {
      label: "Total Trades",
      val:   () => String(M.total_trades),
      sub:   () => M.filled_trades + " filled, " + (M.total_trades - M.filled_trades) + " other",
      color: () => "white",
      accent:() => "c-white",
    },
    {
      label: "Days Running",
      val:   () => String(M.days_running),
      sub:   () => "Since " + (DATA.start_date || "\u2014"),
      color: () => "purple",
      accent:() => "c-purple",
    },
  ];

  const grid = document.getElementById("metrics-grid");
  cardDefs.forEach(def => {
    const c = def.color(), a = def.accent();
    grid.innerHTML += `
      <div class="mcard ${a}">
        <div class="mlabel">${def.label}</div>
        <div class="mval ${c}">${def.val()}</div>
        <div class="msub">${def.sub()}</div>
      </div>`;
  });

  // ── Risk row ───────────────────────────────────────────────────────
  const riskDefs = [
    {
      label: "Max Drawdown",
      val:   () => fmtN(M.max_drawdown, 2) + "%",
      sub:   () => "Peak-to-trough decline",
      color: () => M.max_drawdown > 10 ? "var(--red)" : M.max_drawdown > 5 ? "var(--amber)" : "var(--green)",
    },
    {
      label: "Sharpe Ratio",
      val:   () => M.sharpe_ratio != null ? fmtN(M.sharpe_ratio, 2) : "< 10 days",
      sub:   () => M.sharpe_ratio != null
        ? (M.sharpe_ratio > 1 ? "Good risk-adjusted return" : "Building track record")
        : "Need 10+ snapshots",
      color: () => M.sharpe_ratio != null
        ? (M.sharpe_ratio > 1 ? "var(--green)" : "var(--amber)") : "var(--dim)",
    },
    {
      label: "Largest Single Loss",
      val:   () => M.largest_loss < 0 ? "$" + fmtN(Math.abs(M.largest_loss), 2) : "\u2014",
      sub:   () => "Closed trade P&L",
      color: () => M.largest_loss < 0 ? "var(--red)" : "var(--dim)",
    },
  ];

  const riskRow = document.getElementById("risk-row");
  riskDefs.forEach(r => {
    riskRow.innerHTML += `
      <div class="rcard">
        <div class="rlabel">${r.label}</div>
        <div class="rval" style="color:${r.color()}">${r.val()}</div>
        <div class="rsub">${r.sub()}</div>
      </div>`;
  });

  // ── Portfolio metrics panel ────────────────────────────────────────
  (function renderPortfolioMetrics() {
    const P = DATA.portfolio_metrics;
    if (!P) return;

    // Money with thousands separators + 2dp; signed money keeps the sign.
    const money    = (n) => n == null ? "—" : "$" + fmtN(n, 2);
    const moneySig = (n) => n == null ? "—" :
      (n >= 0 ? "+$" : "-$") + fmtN(Math.abs(n), 2);
    const pctSig   = (n) => n == null ? "—" : fmtSign(n, 2) + "%";
    const cls      = (n) => n == null ? "" : (n >= 0 ? "pnl-pos" : "pnl-neg");
    const tile = (label, valHtml, sub) =>
      `<div class="pm-tile"><div class="pm-label">${label}</div>` +
      `<div class="pm-val">${valHtml}</div>` +
      `<div class="pm-sub">${sub || ""}</div></div>`;

    // ── Snapshot (computable now) ──
    const src = P.have_live ? "live IBKR" : (P.as_of ? "snapshot " + P.as_of : "—");
    const best  = P.best  ? `${P.best.ticker} <span class="${cls(P.best.pct)}">${pctSig(P.best.pct)}</span>`   : "—";
    const worst = P.worst ? `${P.worst.ticker} <span class="${cls(P.worst.pct)}">${pctSig(P.worst.pct)}</span>` : "—";
    const snap = [
      tile("Open Positions", P.open_positions != null ? String(P.open_positions) : "—", src),
      tile("In Profit / Loss",
           `<span class="pnl-pos">${P.positions_in_profit != null ? P.positions_in_profit : "—"} \u25B2</span>`
           + ` / <span class="pnl-neg">${P.positions_in_loss != null ? P.positions_in_loss : "—"} \u25BC</span>`,
           "Open book breadth"),
      tile("Unrealized Return",
           `<span class="${cls(P.unrealized_return_pct)}">${pctSig(P.unrealized_return_pct)}</span>`,
           "Blended, open book"),
      tile("Invested", P.invested_pct != null ? fmtN(P.invested_pct, 2) + "%" : "—", "of NLV"),
      tile("Unrealized P&L", `<span class="${cls(P.unrealized_pnl)}">${moneySig(P.unrealized_pnl)}</span>`, "Open positions"),
      tile("Realized P&L", `<span class="${cls(P.realized_pnl_cum)}">${moneySig(P.realized_pnl_cum)}</span>`, "Cumulative, closed"),
      tile("Largest Position", P.largest_pct != null ? fmtN(P.largest_pct, 2) + "%" : "—", "of NLV"),
      tile("Top-5 Concentration", P.top5_pct != null ? fmtN(P.top5_pct, 2) + "%" : "—", "of NLV"),
      tile("Best Open", best, "Unrealized %"),
      tile("Worst Open", worst, "Unrealized %"),
    ];
    document.getElementById("pm-snapshot").innerHTML = snap.join("");

    // ── Trends (— until >=2 days of snapshots) ──
    const dd  = P.daily, wk = P.weekly;
    const trendTile = (label, t, sub) => t
      ? tile(label, `<span class="${cls(t.usd)}">${moneySig(t.usd)}</span>`,
             `<span class="${cls(t.pct)}">${pctSig(t.pct)}</span>` + (sub ? " · " + sub : ""))
      : tile(label, "—", "Needs ≥2 days");
    const trends = [
      trendTile("Daily P&L", dd),
      trendTile("Weekly P&L", wk),
      tile("Drawdown from Peak",
           P.drawdown_pct != null ? `<span class="${cls(P.drawdown_pct)}">${pctSig(P.drawdown_pct)}</span>` : "—",
           P.drawdown_pct != null ? "Current vs peak NLV" : "Needs ≥2 days"),
      tile("30-Day Return",
           P.return_30d_pct != null ? `<span class="${cls(P.return_30d_pct)}">${pctSig(P.return_30d_pct)}</span>` : "—",
           P.return_30d_pct != null ? "Trailing 30 days" : "Needs 30 days"),
      tile("History", String(P.snapshot_days || 0), "day" + ((P.snapshot_days||0) !== 1 ? "s" : "") + " of snapshots"),
    ];
    document.getElementById("pm-trends").innerHTML = trends.join("");
  })();

  // ── Chart defaults ─────────────────────────────────────────────────
  Chart.defaults.color = "#6a80a8";
  Chart.defaults.borderColor = "#1a2845";
  Chart.defaults.font.family = "SF Mono, Consolas, Monaco, Courier New, monospace";
  Chart.defaults.font.size = 11;

  const gridOpts = {
    color: "rgba(26,40,69,0.8)",
    drawBorder: false,
  };
  const tickOpts = { color: "#6a80a8" };

  // ── Portfolio value chart ──────────────────────────────────────────
  new Chart(document.getElementById("portfolioChart"), {
    type: "line",
    data: {
      labels: S.dates,
      datasets: [
        {
          label: "Kairos Actual",
          data: S.actual,
          borderColor: "#00ccff",
          backgroundColor: "rgba(0,204,255,0.08)",
          borderWidth: 2.5,
          pointRadius: S.dates.length > 30 ? 0 : 4,
          pointBackgroundColor: "#00ccff",
          tension: 0.3,
          fill: true,
          order: 1,
        },
        {
          label: "Advisor 14.4%",
          data: S.advisor,
          borderColor: "#ffaa00",
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          tension: 0.1,
          fill: false,
          order: 2,
        },
        {
          label: "Target 29.0%",
          data: S.target,
          borderColor: "#00e676",
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointRadius: 0,
          tension: 0.1,
          fill: false,
          order: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#6a80a8", boxWidth: 20, padding: 16 },
        },
        tooltip: {
          backgroundColor: "#0d1426",
          borderColor: "#1a2845",
          borderWidth: 1,
          titleColor: "#c8d8f0",
          bodyColor: "#6a80a8",
          callbacks: {
            label: ctx => {
              const val = ctx.parsed.y;
              const fmtVal = "$" + val.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2});
              if (ctx.dataset.label === "Kairos Actual" && M.start_value > 0) {
                const pct = ((val - M.start_value) / M.start_value * 100).toFixed(2);
                return " " + ctx.dataset.label + ": " + fmtVal + " (" + (pct >= 0 ? "+" : "") + pct + "%)";
              }
              return " " + ctx.dataset.label + ": " + fmtVal;
            },
          },
        },
      },
      scales: {
        x: { grid: gridOpts, ticks: tickOpts },
        y: {
          grid: gridOpts,
          ticks: {
            ...tickOpts,
            callback: v => "$" + (v >= 1000 ? (v/1000).toFixed(0) + "k" : v),
          },
        },
      },
    },
  });

  // ── Weekly returns chart ───────────────────────────────────────────
  if (W.labels.length > 0) {
    const barColors = W.returns.map(r => r >= M.advisor_weekly
      ? "rgba(0,230,118,0.75)" : "rgba(255,68,102,0.75)");
    new Chart(document.getElementById("weeklyChart"), {
      type: "bar",
      data: {
        labels: W.labels,
        datasets: [
          {
            label: "Kairos Weekly %",
            data: W.returns,
            backgroundColor: barColors,
            borderRadius: 3,
            order: 1,
          },
          {
            label: "Advisor (0.277%/wk)",
            data: W.advisor,
            borderColor: "#ffaa00",
            borderWidth: 1.5,
            borderDash: [5,3],
            type: "line",
            pointRadius: 0,
            fill: false,
            order: 0,
          },
          {
            label: "Target (0.558%/wk)",
            data: W.target,
            borderColor: "#00e676",
            borderWidth: 1.5,
            borderDash: [3,3],
            type: "line",
            pointRadius: 0,
            fill: false,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top", labels: { color: "#6a80a8", boxWidth: 16, padding: 12 } },
          tooltip: {
            backgroundColor: "#0d1426", borderColor: "#1a2845", borderWidth: 1,
            titleColor: "#c8d8f0", bodyColor: "#6a80a8",
            callbacks: {
              label: ctx => " " + ctx.dataset.label + ": " +
                (ctx.parsed.y >= 0 ? "+" : "") + ctx.parsed.y.toFixed(3) + "%",
            },
          },
        },
        scales: {
          x: { grid: gridOpts, ticks: tickOpts },
          y: {
            grid: gridOpts,
            ticks: { ...tickOpts, callback: v => v.toFixed(2) + "%" },
          },
        },
      },
    });
  } else {
    document.getElementById("weeklyChart").closest(".card").querySelector(".chart-h280")
      .innerHTML = \'<div class="no-data">No weekly data yet &mdash; check back after the first week</div>\';
  }

  // ── AI Value Chain Exposure KPI table ──────────────────────────────
  (function() {
    const chain   = DATA.chain_tier_breakdown || {};
    const buckets = (chain.buckets || []).filter(b => b.label !== "Non-Chain");
    const nlv     = DATA.metrics ? (DATA.metrics.nlv || 0) : 0;
    const wrap    = document.getElementById("chainKpi");
    if (!wrap) return;

    if (!buckets.length) {
      wrap.innerHTML = '<div class="no-data">No chain positions yet</div>';
      return;
    }

    let rows = buckets.map(b => {
      const val  = b.value || 0;
      const pct  = nlv > 0 ? (val / nlv * 100).toFixed(1) : "0.0";
      const fmtVal = "$" + val.toLocaleString("en-US", {minimumFractionDigits:0, maximumFractionDigits:0});
      return `<tr>
        <td style="padding:10px 8px;">
          <span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${b.color};margin-right:8px;"></span>
          <span style="color:#c8d8f0;font-size:13px;">${b.label}</span>
        </td>
        <td style="padding:10px 8px;text-align:right;color:#c8d8f0;font-size:13px;font-weight:600;">${fmtVal}</td>
        <td style="padding:10px 8px;text-align:right;color:#6a80a8;font-size:12px;">${pct}% of NLV</td>
      </tr>`;
    }).join("");

    wrap.innerHTML = `<table style="width:100%;border-collapse:collapse;">${rows}</table>`;
  })();

  // ── Sector breakdown chart ─────────────────────────────────────────
  (function() {
    const sectorRaw  = DATA.sector_breakdown || {};
    const WARN_PCT   = 25;
    const equityBase = M.equity_value > 0 ? M.equity_value : M.current_value;

    const entries = Object.entries(sectorRaw)
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1]);

    const sectorCard = document.getElementById("sector-card");

    if (entries.length === 0 || equityBase <= 0) {
      const el = sectorCard ? sectorCard.querySelector(".chart-h280") : null;
      if (el) el.innerHTML = \'<div class="no-data">No equity positions to display</div>\';
      return;
    }

    const labels  = entries.map(([k]) => k.replace(/_/g, " "));
    const pcts    = entries.map(([, v]) => parseFloat((v / equityBase * 100).toFixed(1)));
    const dollars = entries.map(([, v]) => v);

    const bgColors  = pcts.map(p => p >= WARN_PCT ? "rgba(255,68,102,0.8)"  : "rgba(0,204,255,0.75)");
    const bdColors  = pcts.map(p => p >= WARN_PCT ? "#ff4466"               : "#00ccff");

    // Warn border on the card itself if any sector is over limit
    if (pcts.some(p => p >= WARN_PCT) && sectorCard) {
      sectorCard.style.borderColor = "#ff4466";
      sectorCard.style.boxShadow   = "0 0 0 1px rgba(255,68,102,0.4)";
    }

    new Chart(document.getElementById("sectorChart"), {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "% of Equity",
            data: pcts,
            backgroundColor: bgColors,
            borderColor: bdColors,
            borderWidth: 1.5,
            borderRadius: 3,
            order: 1,
          },
          {
            label: "25% limit",
            data: labels.map(() => WARN_PCT),
            type: "line",
            borderColor: "rgba(255,68,102,0.55)",
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false,
            order: 0,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#6a80a8", boxWidth: 16, padding: 12 },
          },
          tooltip: {
            backgroundColor: "#0d1426", borderColor: "#1a2845", borderWidth: 1,
            titleColor: "#c8d8f0", bodyColor: "#6a80a8",
            callbacks: {
              label: ctx => {
                if (ctx.datasetIndex === 1) return null;
                const idx = ctx.dataIndex;
                const over = pcts[idx] >= WARN_PCT ? " \u26a0 OVER LIMIT" : "";
                return ` ${pcts[idx].toFixed(1)}%  ($${dollars[idx].toLocaleString("en-US", {maximumFractionDigits:0})})${over}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: gridOpts,
            ticks: { ...tickOpts, callback: v => v + "%" },
            max: Math.min(100, Math.max(35, ...pcts) + 5),
          },
          y: { grid: { display: false }, ticks: tickOpts },
        },
      },
    });
  })();

  // ── Positions table ────────────────────────────────────────────────
  const posWrap = document.getElementById("positions-wrap");
  if (positions.length === 0) {
    posWrap.innerHTML = \'<div class="no-data">No open positions (or IBKR offline)</div>\';
  } else {
    let html = \'<div class="tbl-wrap"><table><thead><tr>\' +
      \'<th>Symbol</th><th>Name</th><th>Class</th><th>Sector</th><th>Qty</th><th>Avg Cost</th><th>Market Value</th><th>Unreal P&amp;L</th><th>P&L %</th>\' +
      \'</tr></thead><tbody>\';
    positions.forEach(p => {
      const isSim = p.symbol.includes("(SIM)");
      const cls = p.assetClass === "crypto" ? "purple" : "cyan";
      const upnl = p.unrealized_pnl;
    const costBasis = (p.avg_cost || 0) * (p.quantity || 0);
    const upnlPct = (costBasis > 0 && upnl != null) ? (upnl / costBasis * 100) : null;
    const upnlPctStr = upnlPct != null
      ? `<span class="${upnlPct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${upnlPct >= 0 ? '+' : ''}${upnlPct.toFixed(2)}%</span>`
      : '<span style="color:var(--muted)">—</span>';
      const upnlStr = upnl != null
        ? `<span class="${upnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${upnl >= 0 ? '+' : ''}$${Math.abs(upnl).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</span>`
        : \'<span style="color:var(--muted)">\u2014</span>\';
      html += `<tr>
        <td style="color:var(--${cls});font-weight:700">${p.symbol}</td>
        <td style="color:var(--dim);font-size:11px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.name || ""}</td>
        <td style="color:var(--dim);text-transform:uppercase;font-size:10px">${p.assetClass}</td>
        <td style="color:var(--dim);font-size:11px">${p.sector || "\u2014"}</td>
        <td>${p.quantity.toLocaleString()}</td>
        <td>$${p.avg_cost.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</td>
        <td>$${p.market_value.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</td>
        <td>${upnlStr}</td>
    <td>${upnlPctStr}</td>
      </tr>`;
    });
    html += "</tbody></table></div>";
    posWrap.innerHTML = html;
  }

  // ── Decisions table ────────────────────────────────────────────────
  const decWrap = document.getElementById("decisions-wrap");
  const decs = DATA.decisions || [];
  const dCounts = DATA.decisions_counts || {};
  const summaryLine = `<div style="font-size:12px;color:var(--dim);margin-bottom:8px">`
    + `${dCounts.executed || 0} trades executed, ${dCounts.skipped || 0} skipped</div>`;
  if (decs.length === 0) {
    decWrap.innerHTML = summaryLine + \'<div class="no-data">No actionable decisions logged yet</div>\';
  } else {
    let html = summaryLine + \'<div class="tbl-wrap"><table><thead><tr>\' +
      \'<th>Date</th><th>Ticker</th><th>Action</th><th>Qty</th>\' +
      \'<th>Price</th><th>Status</th><th>P&amp;L</th><th>Rationale</th>\' +
      \'</tr></thead><tbody>\';
    decs.forEach(d => {
      const pnl = d.pnl;
      const pnlStr = pnl != null
        ? `<span class="${pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${pnl >= 0 ? "+" : ""}$${Math.abs(pnl).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</span>`
        : \'<span style="color:var(--muted)">—</span>\';
      const priceStr = d.price != null ? "$" + d.price.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2}) : "—";
      const convTag = d.conviction_trade ? \'<span style="color:#bb66ff;font-weight:700">[C]</span> \' : "";
      const tickerColor = d.asset_class === "crypto" ? "var(--purple, #bb66ff)" : "var(--text)";
      const cryptoTag = d.asset_class === "crypto" ? \'<span style="color:var(--purple, #bb66ff);font-size:9px;margin-left:3px">\u20BF</span>\' : "";
      html += `<tr>
        <td style="color:var(--dim)">${d.date}</td>
        <td style="font-weight:700;color:${tickerColor}">${d.ticker}${cryptoTag}</td>
        <td><span class="tag-${d.action}">${d.action}</span></td>
        <td>${d.quantity}</td>
        <td>${priceStr}</td>
        <td><span class="tag-${d.status || "HOLD"}">${d.status || "—"}</span></td>
        <td>${pnlStr}</td>
        <td><div class="rat">${convTag}${d.rationale || "—"}</div></td>
      </tr>`;
    });
    html += "</tbody></table></div>";
    decWrap.innerHTML = html;
  }

  // ── Trading Lessons panel ─────────────────────────────────────────
  const ledgerWrap = document.getElementById("ledger-wrap");
  const ledger = DATA.ledger || {};
  const ledgerTrades = ledger.trades || [];
  const patternSummary = ledger.pattern_summary || "";

  if (ledgerTrades.length === 0 && !patternSummary) {
    ledgerWrap.innerHTML = '<div class="no-data">No closed trades yet &mdash; the ledger populates as positions close</div>';
  } else {
    let lhtml = "";
    // Pattern summary block
    if (patternSummary) {
      lhtml += '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:12px 14px;margin-bottom:14px;white-space:pre-wrap;font-size:12px;line-height:1.6;color:var(--text)">';
      const escaped = patternSummary
        .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
        .replace(/(AVOID)/g, '<span style="color:var(--red);font-weight:700">$1</span>')
        .replace(/(REPEAT)/g, '<span style="color:var(--green);font-weight:700">$1</span>')
        .replace(/(Win rate: \\d+%)/g, '<span style="color:var(--cyan)">$1</span>')
        .replace(/(LOSING PATTERNS[^:]*:)/g, '<span style="color:var(--red);font-weight:700">$1</span>')
        .replace(/(WINNING PATTERNS[^:]*:)/g, '<span style="color:var(--green);font-weight:700">$1</span>');
      lhtml += escaped + "</div>";
    }
    // Trade log table
    if (ledgerTrades.length > 0) {
      lhtml += '<div class="tbl-wrap"><table><thead><tr>' +
        '<th>Date</th><th>Ticker</th><th>Action</th><th>Signals</th><th>P&amp;L</th><th>Result</th>' +
        '</tr></thead><tbody>';
      ledgerTrades.slice().reverse().forEach(t => {
        const isPass = t.verdict === "PASS";
        const verdictCls = isPass ? "pnl-pos" : "pnl-neg";
        const pnlCls = t.pnl_pct.startsWith("+") ? "pnl-pos" : "pnl-neg";
        lhtml += `<tr>
          <td style="color:var(--dim)">${t.date}</td>
          <td style="font-weight:700">${t.ticker}</td>
          <td><span class="tag-${t.action}">${t.action}</span></td>
          <td style="font-size:10px;color:var(--dim)">${t.signals}</td>
          <td><span class="${pnlCls}">${t.pnl_pct}</span></td>
          <td><span class="${verdictCls}" style="font-weight:700">${t.verdict}</span></td>
        </tr>`;
      });
      lhtml += "</tbody></table></div>";
    }
    ledgerWrap.innerHTML = lhtml;
  }

  // ── Screening Stats panel ─────────────────────────────────────────
  const screenWrap = document.getElementById("screen-wrap");
  const scr = DATA.screen_log;

  if (!scr) {
    screenWrap.innerHTML = '<div class="no-data">No screening data yet &mdash; run kairos_run.py with screening enabled</div>';
  } else {
    const passRate = scr.universe_size > 0
      ? ((scr.tier2_count / scr.universe_size) * 100).toFixed(1) : "0.0";
    const shortlistStr = (scr.shortlist || []).join(", ") || "None";
    const hotStr = (scr.hot || []).join(", ") || "None";

    let shtml = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px">';

    // Mini stat cards
    const miniCards = [
      { label: "Universe Scanned", val: String(scr.universe_size), color: "var(--text)" },
      { label: "Passed to Tier 2", val: String(scr.tier2_count) + " (" + passRate + "%)", color: "var(--cyan)" },
      { label: "Elapsed", val: scr.elapsed_sec + "s", color: "var(--dim)" },
    ];
    miniCards.forEach(mc => {
      shtml += `<div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;text-align:center">
        <div style="font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--dim);margin-bottom:5px">${mc.label}</div>
        <div style="font-size:20px;font-weight:700;color:${mc.color}">${mc.val}</div>
      </div>`;
    });
    shtml += "</div>";

    // Score breakdown bar
    const hotPct = scr.universe_size > 0 ? (scr.hot_count / scr.universe_size * 100) : 0;
    const warmPct = scr.universe_size > 0 ? (scr.warm_count / scr.universe_size * 100) : 0;
    const coldPct = scr.universe_size > 0 ? (scr.cold_count / scr.universe_size * 100) : 0;

    shtml += '<div style="margin-bottom:14px">';
    shtml += '<div style="font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--dim);margin-bottom:6px">Score Distribution</div>';
    shtml += '<div style="display:flex;height:24px;border-radius:4px;overflow:hidden;border:1px solid var(--border)">';
    if (hotPct > 0) shtml += `<div style="width:${Math.max(hotPct,2)}%;background:var(--red);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff" title="HOT: ${scr.hot_count}">${scr.hot_count} HOT</div>`;
    if (warmPct > 0) shtml += `<div style="width:${Math.max(warmPct,3)}%;background:var(--amber);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#000" title="WARM: ${scr.warm_count}">${scr.warm_count} WARM</div>`;
    shtml += `<div style="flex:1;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--dim)" title="COLD: ${scr.cold_count}">${scr.cold_count} COLD</div>`;
    shtml += "</div></div>";

    // HOT tickers
    shtml += '<div style="margin-bottom:10px">';
    shtml += '<div style="font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--dim);margin-bottom:5px">HOT Tickers</div>';
    shtml += '<div style="font-size:13px;color:var(--red);font-weight:700">' + (hotStr) + '</div>';
    shtml += "</div>";

    // Shortlist table with Tier column
    shtml += '<div style="margin-bottom:8px">';
    shtml += '<div style="font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--dim);margin-bottom:5px">Tier 2 Shortlist</div>';
    const srcTiers = scr.source_tiers || {};
    if (scr.shortlist && scr.shortlist.length > 0) {
      shtml += '<div class="tbl-wrap"><table style="font-size:12px"><thead><tr>' +
        '<th>Symbol</th><th>Tier</th>' +
        '</tr></thead><tbody>';
      scr.shortlist.forEach(function(t) {
        const tier = srcTiers[t] || "A";
        const tierColor = tier === "C" ? "#f59e0b" : tier === "B" ? "var(--amber)" : "var(--cyan)";
        const tierLabel = tier === "C" ? "C (Opport.)" : tier === "B" ? "B (Growth)" : "A (Core)";
        const rowBg = tier === "C" ? "background:rgba(245,158,11,0.1);" : "";
        shtml += '<tr style="' + rowBg + '"><td style="font-weight:700;color:var(--cyan)">' + t + '</td>' +
          '<td style="color:' + tierColor + ';font-weight:600">' + tierLabel + '</td></tr>';
      });
      shtml += '</tbody></table></div>';
    } else {
      shtml += '<div style="font-size:12px;color:var(--cyan);line-height:1.6">None</div>';
    }
    const tb = scr.tier_breakdown;
    if (tb) {
      shtml += '<div style="font-size:10px;color:var(--dim);margin-top:6px">' +
        'Tier A: ' + (tb.tier_a_screened || 0) + ' screened, ' + (tb.tier_a_hot_warm || 0) + ' HOT/WARM &middot; ' +
        'Tier B: ' + (tb.tier_b_screened || 0) + ' screened, ' + (tb.tier_b_hot_warm || 0) + ' HOT/WARM' +
        ((tb.tier_c_screened || 0) > 0 ? ' &middot; Tier C: ' + tb.tier_c_screened + ' screened, ' + (tb.tier_c_hot_warm || 0) + ' HOT/WARM' : '') +
        '</div>';
    }
    shtml += "</div>";

    // Timestamp
    shtml += '<div style="font-size:10px;color:var(--muted);margin-top:10px">Last screened: ' + (scr.timestamp || "unknown") + '</div>';

    screenWrap.innerHTML = shtml;
  }

  // ── Refresh button + auto-refresh ────────────────────────────────
  const REFRESH_URL = "/refresh";
  const STATUS_URL  = "/api/status";
  const AUTO_REFRESH_MS = 5 * 60 * 1000; // 5 minutes

  // Show initial "Last updated" from the generation timestamp
  (function initLastUpdated() {
    const el = document.getElementById("last-updated");
    if (el) el.textContent = "Last updated: __TIMESTAMP__";
  })();

  function refreshDashboard() {
    const btn = document.getElementById("refresh-btn");
    const updEl = document.getElementById("last-updated");
    if (!btn || btn.classList.contains("loading")) return;

    btn.classList.add("loading");
    btn.disabled = true;
    if (updEl) updEl.textContent = "Refreshing\u2026";

    fetch(REFRESH_URL, { method: "POST", redirect: "follow" })
      .then(function(resp) {
        if (resp.redirected) {
          window.location.href = resp.url;
        } else {
          // Non-redirect: reload the page to pick up new HTML
          window.location.reload();
        }
      })
      .catch(function(err) {
        btn.classList.remove("loading");
        btn.disabled = false;
        if (updEl) updEl.textContent = "Refresh failed \u2014 is the server running?";
        console.error("Refresh failed:", err);
      });
  }

  // Auto-refresh: reload the page every 5 minutes
  // Only when served from the Flask server (not file://)
  if (window.location.protocol !== "file:") {
    setInterval(function() {
      // Use the /refresh endpoint so data is regenerated, not just HTML reloaded
      fetch(REFRESH_URL, { method: "POST", redirect: "follow" })
        .then(function() { window.location.reload(); })
        .catch(function() { /* silent — will retry next interval */ });
    }, AUTO_REFRESH_MS);
  }

  // Countdown display for next auto-refresh
  (function autoRefreshCountdown() {
    if (window.location.protocol === "file:") return;
    let remaining = AUTO_REFRESH_MS / 1000;
    const tick = function() {
      remaining--;
      if (remaining <= 0) { remaining = AUTO_REFRESH_MS / 1000; }
      const updEl = document.getElementById("last-updated");
      const mins = Math.floor(remaining / 60);
      const secs = remaining % 60;
      const base = "Last updated: __TIMESTAMP__";
      if (updEl && !document.getElementById("refresh-btn").classList.contains("loading")) {
        updEl.textContent = base + "  \u00b7  next in " + mins + ":" + (secs < 10 ? "0" : "") + secs;
      }
    };
    setInterval(tick, 1000);
  })();
</script>
</body>
</html>'''


def generate_dashboard(payload: dict, now_str: str) -> str:
    data_json = json.dumps(payload, indent=2)
    html = HTML_TEMPLATE
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("__TIMESTAMP__", now_str)
    return html


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kairos Performance Dashboard Generator")
    parser.add_argument("--no-ibkr", action="store_true", help="Skip IBKR connection, use cached data")
    args = parser.parse_args()

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    W = 60
    print("=" * W)
    print(f"  KAIROS DASHBOARD GENERATOR — {now_str}")
    print("=" * W)

    # Load performance history
    perf = load_perf()
    snap_count = len(perf.get("snapshots", []))
    print(f"  Performance history: {snap_count} snapshot(s)")

    # Fetch IBKR data
    ibkr = {"connected": False}
    if not args.no_ibkr:
        print("  Connecting to IBKR (port 7497)...")
        ibkr = fetch_ibkr_data()
        if ibkr["connected"]:
            _mode = ibkr.get("data_mode", "?")
            _src = ibkr.get("nlv_source", "?")
            print(f"  Net liquidation: ${ibkr['net_liquidation']:,.2f}  "
                  f"(marks={_mode}, nlv_source={_src})")
            if ibkr.get("marks_stale"):
                print(f"  ⚠ marks STALE — using delayed/computed NLV "
                      f"(acct-summary NLV is frozen; live data entitlement impaired)")
            print(f"  Equity: ${ibkr['equity_value']:,.2f}  Crypto: ${ibkr['crypto_value']:,.2f}")
            perf = upsert_snapshot(perf, ibkr)
            save_perf(perf)
            print(f"  Snapshot saved → {PERF_FILE}")
        else:
            print("  IBKR unavailable — using cached performance data")
    else:
        print("  --no-ibkr: skipping IBKR connection")

    # Seed with demo data if nothing exists (first run with no IBKR)
    if not perf.get("snapshots"):
        print("  No snapshots found — seeding with $1,000,000 starting value")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        perf = {
            "start_date": today,
            "start_value": 1_000_000.0,
            "snapshots": [{
                "date": today, "value": 1_000_000.0,
                "equity_value": 0.0, "crypto_value": 0.0, "cash": 1_000_000.0,
            }],
        }
        save_perf(perf)

    # Query DB
    print("  Querying kairos.db...")
    db = query_db()
    print(f"  Decisions: {len(db['decisions'])}  Holdings: {len(db['holdings'])}")

    # Build and write dashboard
    payload = build_payload(perf, db, ibkr)
    html    = generate_dashboard(payload, now_str)

    with open(DASHBOARD_OUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Dashboard written → {DASHBOARD_OUT}")
    print("=" * W)
    print("  Open kairos_dashboard.html in your browser to view.")
    print("=" * W)


if __name__ == "__main__":
    main()
