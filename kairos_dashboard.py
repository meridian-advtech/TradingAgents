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
ML_DB_PATH    = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
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
                # Real sector, same source as the exposure panel. Previously
                # this was the universe screening bucket, so the positions
                # table showed "Large Cap" and "Financials" in the same column
                # the exposure panel labelled "Financial Services" — two
                # provenances for one column, in one view.
                "sector":        "Crypto" if is_crypto else _position_sector(sym, raw_sector),
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


def _position_sector(symbol: str, raw_bucket: str) -> str:
    """Real sector for a single position, matching the exposure panel.

    Falls back to the normalised universe bucket if the security master is
    unavailable or has not resolved this ticker, so the column always says
    something rather than going blank.
    """
    try:
        import kairos_security_master as sm
    except ImportError:
        return _normalize_sector(raw_bucket, symbol)

    sector = sm.get_sector(symbol)
    if sector and sector != sm.UNRESOLVED:
        return sector
    return _normalize_sector(raw_bucket, symbol)


def _eligible_equity(positions: list):
    """Yield (symbol, market_value) for equity positions that can carry exposure.

    Excludes sim rows, crypto, and non-positive values — a short cannot be a
    slice of a part-to-whole breakdown.
    """
    for p in positions:
        sym = p.get("symbol", "")
        if not sym or "(SIM)" in sym:
            continue
        if p.get("assetClass", "equity") == "crypto":
            continue
        mkt_val = float(p.get("market_value", 0) or 0)
        if mkt_val <= 0:
            continue
        yield sym, mkt_val


def _compute_sector_exposure(positions: list) -> dict[str, float]:
    """Return {sector: market_value} using REAL sectors from the security master.

    kairos_confluence.lookup_sector() returns the universe *screening bucket*
    ("large_cap", "value_dividend"), not a sector. Grouping exposure by it
    scatters one real sector across several buckets and understates
    concentration. Size/style is still a legitimate view — it is reported
    separately by _compute_size_style_exposure(), not conflated with this one.

    Falls back to the bucket labels if the security master is unavailable, so a
    missing table degrades the breakdown rather than emptying the panel.
    """
    try:
        import kairos_security_master as sm
    except ImportError:
        sm = None

    exposure: dict[str, float] = {}

    if sm is not None:
        for sym, mkt_val in _eligible_equity(positions):
            if sm.is_fund(sym):
                continue          # funds carry no single sector
            exposure[sm.get_sector(sym)] = exposure.get(sm.get_sector(sym), 0.0) + mkt_val
        # An empty security_master table resolves everything to Unclassified;
        # that is worse than the old behaviour, so fall through to it.
        if exposure and set(exposure) != {sm.UNRESOLVED}:
            return exposure
        exposure = {}

    try:
        from kairos_confluence import lookup_sector
    except ImportError:
        return {}
    for sym, mkt_val in _eligible_equity(positions):
        sector = _normalize_sector(lookup_sector(sym), sym)
        exposure[sector] = exposure.get(sector, 0.0) + mkt_val
    return exposure


# Tier A universe categories → display labels. Anything not in here that comes
# back from lookup_sector is a Tier B entry carrying its own sector string, not
# a size/style cohort, so it is grouped rather than listed alongside them.
_SIZE_STYLE_LABELS = {
    "mega_cap":              "Mega cap",
    "large_cap":             "Large cap",
    "mid_cap_growth":        "Mid cap growth",
    "value_dividend":        "Value & dividend",
    "healthcare_biotech":    "Healthcare / biotech",
    "financials":            "Financials",
    "energy_materials":      "Energy & materials",
    "reits_real_estate":     "REITs & real estate",
    "industrials_transport": "Industrials & transport",
    "consumer_tech":         "Consumer tech",
}


def _compute_size_style_exposure(positions: list) -> dict[str, float]:
    """Return {bucket: market_value} — the universe screening axis.

    This is what lookup_sector actually measures. Kept as its own dimension
    because size/style exposure is worth seeing; it just is not a sector.
    """
    try:
        from kairos_confluence import lookup_sector
    except ImportError:
        return {}

    exposure: dict[str, float] = {}
    for sym, mkt_val in _eligible_equity(positions):
        raw = lookup_sector(sym)
        if raw in _SIZE_STYLE_LABELS:
            label = _SIZE_STYLE_LABELS[raw]
        elif raw.startswith("etf_"):
            label = "ETF"
        else:
            label = "Tier B / unbucketed"
        exposure[label] = exposure.get(label, 0.0) + mkt_val
    return exposure


def _compute_exit_coverage(closed_trades: list) -> dict:
    """Exit-reason coverage, so share-of-exits can use an honest denominator.

    Reason capture did not exist for the earliest trades. Those trades cannot
    carry a type, and leaving them in the denominator understates every real
    exit reason — "Not recorded" is an absence of data, not a way of exiting.
    A date cutoff is the wrong instrument here because capture ramped up rather
    than switching on: any single date either readmits unrecorded trades or
    discards recorded ones from the same window.
    """
    total = len(closed_trades)
    recorded = [t for t in closed_trades if t.get("exit_reason")]
    days = [str(t.get("sold_date") or "")[:10] for t in recorded if t.get("sold_date")]
    return {
        "total":          total,
        "recorded":       len(recorded),
        "unrecorded":     total - len(recorded),
        "capture_from":   min(days) if days else None,
        "pct":            round(len(recorded) / total * 100, 1) if total else 0.0,
    }


def _compute_short_positions(positions: list) -> list[dict]:
    """Open shorts. Netted into totals correctly, but they cannot appear in a
    part-to-whole breakdown, so surface them rather than silently dropping."""
    out = []
    for p in positions:
        sym = p.get("symbol", "")
        if not sym or "(SIM)" in sym:
            continue
        if float(p.get("quantity", 0) or 0) < 0:
            out.append({
                "symbol":       sym,
                "quantity":     float(p.get("quantity", 0) or 0),
                "market_value": float(p.get("market_value", 0) or 0),
            })
    return out


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


# ── ML trade ledger (single source of truth for closed-trade stats) ────

def load_ml_trade_stats(since: str | None = None) -> dict | None:
    """Closed-trade stats from the reconciled ML ledger (kairos_ml_outcomes.db).

    `since` ("YYYY-MM-DD") restricts the set to trades that EXITED on or after
    that date — used for the current-system era view. Stored timestamps come in
    two shapes ("2026-05-27 13:37:55 UTC" and "2026-07-31T17:36:09Z"); both lead
    with an ISO date, so a lexicographic compare on the prefix is exact.
    Filtering here is display-only — no row is ever hidden from the DB itself.

    This is the authoritative trade universe: one row per attributable round
    trip, P&L frozen at exit. The legacy kairos.db holdings lots are NOT — the
    daily reconciler rewrites/merges closed lots at broker cost, which erases
    their realized P&L over time.

    Opened READ-ONLY (uri mode=ro) so the dashboard can never lock the ML DB
    against the live trading writers. Returns None on any error, so callers
    fall back to their legacy computation instead of showing nothing.

    Returns {closed_trades, win_rate, realized_pnl, monthly[]} where monthly is
    ascending by "YYYY-MM" with {month, n, win_rate, avg_pct, total_usd}.
    """
    if not os.path.exists(ML_DB_PATH):
        return None
    try:
        conn = sqlite3.connect(f"file:{ML_DB_PATH}?mode=ro", uri=True, timeout=2)
        try:
            conn.execute("PRAGMA busy_timeout = 2000")
            rows = conn.execute(
                "SELECT timestamp_exit, pnl_pct, pnl_dollar FROM trade_outcomes "
                "WHERE timestamp_exit IS NOT NULL AND pnl_pct IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        print(f"  WARNING: ML trade ledger unavailable ({exc}) — using legacy stats")
        return None

    if since:
        rows = [r for r in rows if str(r[0])[:10] >= since]

    if not rows:
        return None

    n_total = len(rows)
    wins    = sum(1 for _, pct, _ in rows if pct is not None and pct > 0)
    realized = sum(float(usd or 0.0) for _, _, usd in rows)

    buckets: dict[str, list] = {}
    for ts_exit, pct, usd in rows:
        month = str(ts_exit)[:7]          # "YYYY-MM-DD ..." → "YYYY-MM"
        if len(month) != 7:
            continue
        buckets.setdefault(month, []).append((float(pct), float(usd or 0.0)))

    monthly = []
    for month in sorted(buckets):
        b = buckets[month]
        monthly.append({
            "month":     month,
            "n":         len(b),
            "win_rate":  round(sum(1 for pct, _ in b if pct > 0) / len(b) * 100, 1),
            "avg_pct":   round(sum(pct for pct, _ in b) / len(b), 2),
            "total_usd": round(sum(usd for _, usd in b), 2),
        })

    return {
        "closed_trades": n_total,
        "win_rate":      round(wins / n_total * 100, 1),
        "realized_pnl":  round(realized, 2),
        "monthly":       monthly,
    }


# ── Era baseline (measure the CURRENT system, keep full history) ───────

def load_performance_config() -> dict:
    """The `performance` block of kairos_config.json: {baseline_date, baseline_note}.

    baseline_date marks where the current system begins — everything before it
    ran with known critical bugs (zero-qty sizing, unlogged trades, the exit
    ratchet), so an inception-blended headline measures software that no longer
    exists. It lives in config precisely because it MOVES: when the next
    material change ships, J re-points it and every era figure follows.

    Returns {} when the block is absent or unreadable, which makes every era
    metric None and leaves the dashboard on its inception-only behavior.
    """
    try:
        with open(os.path.join(SCRIPT_DIR, "kairos_config.json")) as f:
            block = json.load(f).get("performance") or {}
    except (json.JSONDecodeError, IOError) as exc:
        print(f"  WARNING: performance config unreadable ({exc}) — era metrics off")
        return {}
    date = str(block.get("baseline_date") or "").strip()
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        if date:
            print(f"  WARNING: bad performance.baseline_date {date!r} — era metrics off")
        return {}
    return {"baseline_date": date,
            "baseline_note": str(block.get("baseline_note") or "").strip()}


def _era_baseline_value(baseline_date: str, perf_snaps: list[dict]) -> tuple[float, str] | None:
    """(value, actual_date) for the era baseline, taken from the PERF series.

    SAME-SERIES BY CONSTRUCTION: both ends of the era return come from
    kairos_performance.json — this baseline and the current value the inception
    hero tiles already use. An earlier revision took the baseline from
    nlv_snapshots and the current value from PERF; those two series are captured
    at different times of day and disagreed by ~0.23% on 2026-07-01
    ($1,082,667 vs $1,080,230), so the era return carried a phantom delta that
    was an artifact of the source mix rather than a real move. nlv_snapshots
    remains the source for the metrics panel and history — only these two
    endpoints changed.

    The series is dense (daily), so "nearest" (by absolute day distance, earlier
    wins a tie) only matters for a baseline outside its range. Returns None when
    no usable row exists, which drops the caller back to inception.
    """
    try:
        target = datetime.strptime(baseline_date, "%Y-%m-%d").date()
    except ValueError:
        return None

    best = None
    for snap in perf_snaps:
        value = snap.get("value")
        if not value or value <= 0:
            continue
        try:
            d = datetime.strptime(str(snap.get("date"))[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        dist = abs((d - target).days)
        if best is None or dist < best[0]:
            best = (dist, float(value), str(snap.get("date"))[:10])
    return (best[1], best[2]) if best else None


def compute_era_metrics(perf: dict, curr_val: float) -> dict | None:
    """Metrics for the current-system era only — the same shape as the headline
    figures, measured from performance.baseline_date instead of inception.

    Returns None (callers fall back to inception) when no baseline is configured
    or no usable baseline value exists. Purely a presentation-layer view: it
    reads the same snapshot series and ML ledger the inception numbers read, and
    writes nothing anywhere.
    """
    cfg = load_performance_config()
    baseline_date = cfg.get("baseline_date")
    if not baseline_date or not curr_val:
        return None

    baseline = _era_baseline_value(baseline_date, perf.get("snapshots", []))
    if not baseline:
        print(f"  WARNING: no PERF snapshot near {baseline_date} — era metrics off")
        return None
    base_val, base_actual = baseline

    base_dt  = datetime.strptime(base_actual, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days     = max(1, (datetime.now(timezone.utc) - base_dt).days)
    years    = days / 365.25
    total_ret = (curr_val - base_val) / base_val * 100

    # Same >=7-day guard the inception figure uses: annualizing a handful of
    # days extrapolates noise into a headline number.
    annualized = None
    if days >= 7 and curr_val > 0 and base_val > 0:
        annualized = ((curr_val / base_val) ** (1.0 / years) - 1.0) * 100

    ml = load_ml_trade_stats(since=baseline_date)

    return {
        "baseline_date":     baseline_date,
        "baseline_actual":   base_actual,     # PERF row actually used
        "baseline_note":     cfg.get("baseline_note", ""),
        "baseline_value":    round(base_val, 2),
        "baseline_source":   "PERF series, same-source",
        "days":              days,
        "total_return_pct":  round(total_ret, 3),
        "total_return_usd":  round(curr_val - base_val, 2),
        "annualized_return": round(annualized, 2) if annualized is not None else None,
        "vs_advisor":        round(annualized - ADVISOR_RATE * 100, 2) if annualized is not None else None,
        "vs_target":         round(annualized - TARGET_RATE  * 100, 2) if annualized is not None else None,
        "win_rate":          ml["win_rate"] if ml else None,
        "closed_trades":     ml["closed_trades"] if ml else None,
        "realized_pnl":      ml["realized_pnl"] if ml else None,
    }


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

    # Win rate / closed count / realized P&L: the reconciled ML ledger is the
    # single source of truth (see load_ml_trade_stats). The legacy JSON-holdings
    # computation below is the explicit fallback when that ledger is unreadable
    # — it counts a different trade universe and its lots decay as the daily
    # reconciler rewrites them at cost.
    ml = load_ml_trade_stats()

    closed = [h for h in db.get("holdings", [])
              if h.get("sold_date") and h.get("sold_price")]
    wins   = [h for h in closed if h["sold_price"] > h["entry_price"]]
    win_rt = len(wins) / len(closed) * 100 if closed else None

    closed_n   = len(closed)
    stats_src  = "legacy"
    monthly    = []
    if ml:
        win_rt    = ml["win_rate"]
        closed_n  = ml["closed_trades"]
        monthly   = ml["monthly"]
        stats_src = "ml"

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
    total_pnl    = ml["realized_pnl"] if ml else (round(sum(pnls), 2) if pnls else 0.0)

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
        "closed_trades":     closed_n,
        "trade_stats_source": stats_src,
        "monthly_performance": monthly,
        # Current-system era view (None → the UI shows inception only). The
        # monthly table above deliberately stays full-history: the era answers
        # "what is this system doing now", the months show how it got there.
        "era":               compute_era_metrics(perf, curr_val),
        "max_drawdown":      round(max_dd, 2),
        "sharpe_ratio":      sharpe,
        "largest_loss":      largest_loss,
        "realized_pnl":      total_pnl,
        "equity_return":     round(equity_return, 2) if equity_return is not None else None,
        "crypto_return":     round(crypto_return, 2) if crypto_return is not None else None,
    }


# ── Closed-positions trade list (individual realized trades) ───────────

# Prefix (the token before the first ':') of a stored exit_reason → the
# friendly condition label + a color key matching the dashboard palette.
# Only these three conditions are actually emitted by the exit engine today;
# anything else falls through to a title-cased prefix with a neutral color.
_EXIT_TYPE_MAP = {
    "STOP-LOSS":          ("Hard Stop",     "red"),
    "TRAILING-STOP":      ("Trailing Stop", "amber"),
    "REVERSION-COMPLETE": ("Reversion",     "cyan"),
}


# A real exit type is a short label ("TRAILING-STOP", "HARD-STOP"). Anything
# longer, or carrying sentence punctuation, is a free-text rationale that had no
# "TYPE:" prefix — taking everything before the first colon then promotes an
# entire paragraph to a category of its own. Six such paragraphs were each
# appearing as a distinct exit type in the distribution and the filter dropdown.
_MAX_EXIT_TYPE_LEN = 28


def _parse_exit_type(reason: str) -> tuple[str, str]:
    """Map a raw exit_reason string to (friendly_label, color_key)."""
    if not reason:
        return ("Not recorded", "muted")
    prefix = str(reason).split(":", 1)[0].strip().upper()
    if prefix in _EXIT_TYPE_MAP:
        return _EXIT_TYPE_MAP[prefix]
    if len(prefix) > _MAX_EXIT_TYPE_LEN or any(ch in prefix for ch in ".;("):
        return ("Unlabelled rationale", "muted")
    return (prefix.replace("-", " ").title() or "Exit", "white")


def _load_exit_reasons() -> dict[tuple, dict]:
    """Load stored exit reasons keyed by (TICKER, sold-day) for a precise join.

    Reads the append-only position_exits_history: a repeat-traded ticker now has
    a row per close, so keying on ticker + same calendar day (exit_date vs. a
    holding's sold_date) attributes each closed lot to its OWN exit reason
    instead of borrowing the latest. A day with no matching history row renders
    "Not recorded" rather than an unrelated reason. (On the rare two-closes-same-
    day collision the later id wins — acceptable for this display join.)
    """
    reasons: dict[tuple, dict] = {}
    if not os.path.exists(DB_PATH):
        return reasons
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # Ordered by id so that, on a same-(ticker,day) collision, the later
        # (higher-id) exit is the one that survives in the dict.
        rows = conn.execute(
            "SELECT ticker, exit_date, exit_reason FROM position_exits_history "
            "ORDER BY id ASC"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return reasons

    for r in rows:
        tkr = (r["ticker"] or "").strip().upper()
        day = (r["exit_date"] or "")[:10]
        if not tkr or not day:
            continue
        label, color = _parse_exit_type(r["exit_reason"])
        reasons[(tkr, day)] = {
            "exit_reason": r["exit_reason"],
            "exit_type":   label,
            "exit_color":  color,
        }
    return reasons


def _load_entry_signals() -> dict[tuple, list]:
    """Entry signals per BUY, keyed by (TICKER, timestamp) for a precise join.

    A closed lot's entry_date matches its BUY decision's timestamp exactly (to
    the second), so this keys off both — correctly attributing signals even for
    repeat-traded tickers. Signals live in decisions.data_inputs →
    confluence.signals; conviction/no-signal buys yield an empty list.
    """
    signals: dict[tuple, list] = {}
    if not os.path.exists(DB_PATH):
        return signals
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ticker, timestamp, data_inputs FROM decisions WHERE action = 'BUY'"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return signals

    for r in rows:
        tkr = (r["ticker"] or "").strip().upper()
        ts = r["timestamp"] or ""
        if not tkr or not ts:
            continue
        try:
            di = json.loads(r["data_inputs"]) if r["data_inputs"] else {}
            sigs = (di.get("confluence", {}) or {}).get("signals") or di.get("signals") or []
            sigs = [str(s) for s in sigs if s]
        except (json.JSONDecodeError, TypeError):
            sigs = []
        # Last BUY at a given (ticker, ts) wins — timestamps are second-precise
        # so collisions are effectively the same decision.
        signals[(tkr, ts)] = sigs
    return signals


def _parse_trade_date(raw) -> datetime | None:
    """Parse a holdings date string, tolerating ' UTC' and '[RECON-merged]'
    suffixes that break SQLite's julianday() (hence null holding_days)."""
    if not raw:
        return None
    s = str(raw).split("[", 1)[0].replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _days_held(entry_date, sold_date, fallback) -> int | None:
    """Whole days between entry and sale, computed robustly; falls back to the
    SQL-provided value only when both dates fail to parse."""
    ed, sd = _parse_trade_date(entry_date), _parse_trade_date(sold_date)
    if ed and sd:
        return max(0, (sd - ed).days)
    try:
        return int(fallback) if fallback is not None else None
    except (TypeError, ValueError):
        return None


def build_closed_trades(holdings: list[dict]) -> list[dict]:
    """Individual realized trades, newest-first by sold_date, for the trade list.

    A trade is closed when sold_date/sold_price are set. Realized P&L is
    (sold_price - entry_price) * quantity; % is against that lot's cost basis.
    Exit reason/type is a best-effort join to position_exits_history by ticker +
    day; absent, exit_reason is None and exit_type "Not recorded" (never faked).
    """
    exit_reasons = _load_exit_reasons()
    entry_signals = _load_entry_signals()
    trades = []
    for h in (holdings or []):
        if not (h.get("sold_date") and h.get("sold_price") is not None):
            continue
        # Reconciliation lot-merges aren't economic exits (sold at cost, zero
        # P&L) — they carry a [RECON-merged] marker on sold_date. Skip them so
        # the closed-trade list shows only real exits.
        if "[RECON-merged]" in str(h.get("sold_date") or ""):
            continue
        try:
            entry = float(h["entry_price"])
            qty   = float(h["quantity"])
            sold  = float(h["sold_price"])
        except (TypeError, ValueError, KeyError):
            continue

        pnl_usd = (sold - entry) * qty
        pnl_pct = (sold - entry) / entry * 100.0 if entry else None
        tkr = (h.get("ticker") or "").strip().upper()
        sold_day = (h.get("sold_date") or "")[:10]

        ex = exit_reasons.get((tkr, sold_day))
        sigs = entry_signals.get((tkr, h.get("entry_date") or ""), [])
        trades.append({
            "id":           h.get("id"),
            "ticker":       tkr,
            "entry_date":   h.get("entry_date"),
            "sold_date":    h.get("sold_date"),
            "days_held":    _days_held(h.get("entry_date"), h.get("sold_date"),
                                       h.get("holding_days")),
            "entry_price":  round(entry, 2),
            "sold_price":   round(sold, 2),
            "quantity":     qty,
            "realized_pnl_usd": round(pnl_usd, 2),
            "realized_pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "exit_reason":  ex["exit_reason"] if ex else None,
            "exit_type":    ex["exit_type"] if ex else "Not recorded",
            "exit_color":   ex["exit_color"] if ex else "muted",
            "entry_signals": sigs,
        })

    # Newest first. sold_date is an ISO-ish string (some carry a " UTC" or
    # "[RECON-merged]" suffix) — lexical sort on the raw string orders correctly.
    trades.sort(key=lambda t: t.get("sold_date") or "", reverse=True)
    return trades


# ── Slide-over price series (native thesis_reviews + IBKR fallback) ─────

def build_price_series(ticker: str, entry_date, entry_price: float,
                       end_date, end_price: float, conn=None) -> list[dict]:
    """One point per day for a ticker's price over its holding window.

    Native source: thesis_reviews.current_price (~1 point per trading day since
    entry). Anchored with the entry price at the entry day and the end price
    (current for open, sold for closed) at the end day. Returns [{t, p}] sorted
    by day. Empty on error / no data (the caller then shows a 'building history'
    state or tries the IBKR fallback).
    """
    ed = _parse_trade_date(entry_date)
    if ed is None:
        return []
    # _parse_trade_date is naive → keep the window bound naive too.
    end_dt = _parse_trade_date(end_date) if end_date else datetime.now(timezone.utc).replace(tzinfo=None)
    if end_dt is None:
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    day2price: dict[str, float] = {}

    own = False
    if conn is None:
        if not os.path.exists(DB_PATH):
            return []
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            own = True
        except sqlite3.Error:
            return []
    try:
        rows = conn.execute(
            "SELECT timestamp, current_price FROM thesis_reviews "
            "WHERE ticker = ? AND current_price IS NOT NULL AND current_price > 0 "
            "ORDER BY id ASC", (ticker,)).fetchall()
        for r in rows:
            d = _parse_trade_date(r["timestamp"])
            if d is None or d < ed or d > end_dt:
                continue
            day2price[d.strftime("%Y-%m-%d")] = round(float(r["current_price"]), 2)
    except sqlite3.Error:
        pass
    finally:
        if own:
            conn.close()

    # Anchor the ends: entry price at the entry day, end price at the end day.
    if entry_price:
        day2price[ed.strftime("%Y-%m-%d")] = round(float(entry_price), 2)
    if end_price and end_dt:
        day2price[end_dt.strftime("%Y-%m-%d")] = round(float(end_price), 2)

    return [{"t": d, "p": day2price[d]} for d in sorted(day2price)]


def _ibkr_historical_series(tickers: list[str]) -> dict:
    """Fallback: 30 calendar days of daily closes per ticker via IBKR, in ONE
    connection. Returns {ticker: [{t, p}]}. Empty on any failure (ib_insync
    missing, gateway down, etc.) — the caller keeps whatever native series it had.
    """
    out: dict = {}
    if not tickers:
        return out
    try:
        from ib_insync import IB, Stock
    except Exception:
        return out
    ib = IB()
    try:
        ib.connect("127.0.0.1", 7497, clientId=17, timeout=8)
    except Exception:
        return out
    try:
        ib.reqMarketDataType(3)
        for t in tickers:
            try:
                c = Stock(t, "SMART", "USD")
                ib.qualifyContracts(c)
                bars = ib.reqHistoricalData(
                    c, endDateTime="", durationStr="30 D", barSizeSetting="1 day",
                    whatToShow="TRADES", useRTH=True, formatDate=1)
                pts = [{"t": str(b.date), "p": round(float(b.close), 2)}
                       for b in bars if b.close and b.close > 0]
                if len(pts) >= 3:
                    out[t] = pts
            except Exception:
                continue
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
    return out


def build_closed_details(closed_trades: list[dict], name_map: dict) -> dict:
    """Detail records for the slide-over panel, keyed by str(holding id).

    Numeric-string keys never collide with the open-position records (keyed by
    uppercase ticker), so both live in the one position_details map the existing
    panel already reads. Marked closed:true so the panel renders realized
    figures (exit price, realized P&L, exit reason) instead of live/unrealized.
    """
    details: dict = {}
    # One shared connection for all price-series lookups (indexed by ticker).
    conn = None
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            conn = None
    try:
        for t in closed_trades:
            hid = t.get("id")
            if hid is None:
                continue
            entry, sold, qty = t["entry_price"], t["sold_price"], t["quantity"]
            # Native price line for the holding window; entry + exit anchored.
            series = build_price_series(t["ticker"], t["entry_date"], entry,
                                        t["sold_date"], sold, conn=conn)
            details[str(hid)] = {
                "closed":       True,
                "ticker":       t["ticker"],
                "name":         name_map.get(t["ticker"], ""),
                "asset_class":  "equity",
                "entry_date":   t["entry_date"],
                "sold_date":    t["sold_date"],
                "days_held":    t["days_held"],
                "entry_price":  entry,
                "sold_price":   sold,
                "quantity":     qty,
                "cost_basis":   round(entry * qty, 2),
                "proceeds":     round(sold * qty, 2),
                "realized_pnl": t["realized_pnl_usd"],
                "realized_pct": t["realized_pnl_pct"],
                "entry_signals": t.get("entry_signals", []),
                "exit_reason":  t["exit_reason"],
                "exit_type":    t["exit_type"],
                "exit_color":   t["exit_color"],
                # Chart: series + marked points; closed → no live stop lines.
                "price_series": series,
                "entry_t":      (_parse_trade_date(t["entry_date"]).strftime("%Y-%m-%d")
                                 if _parse_trade_date(t["entry_date"]) else None),
                "exit_t":       (_parse_trade_date(t["sold_date"]).strftime("%Y-%m-%d")
                                 if _parse_trade_date(t["sold_date"]) else None),
                "hard_stop_price":  None,
                "trail_stop_price": None,
            }
    finally:
        if conn:
            conn.close()
    return details


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

    # Realized P&L (cumulative closed): the ML ledger is the source of truth —
    # same number kairos_execute._realized_pnl_cumulative() now writes into
    # nlv_snapshots. Read it directly here rather than importing that module,
    # which drags in ib_insync and would silently fail to the stale snapshot
    # value on any host without it. Fallbacks, in order: the shared helper (its
    # own kairos.db lot-sum fallback included), then the last snapshot value.
    ml_stats = load_ml_trade_stats()
    realized_cum = ml_stats["realized_pnl"] if ml_stats else None
    if realized_cum is None:
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
    # Breadth is only real when at least one position carried an unrealized P&L.
    # With IBKR offline nothing is enriched, and "0 ▲ / 0 ▼" next to 59 open
    # positions is false data — report None so the tile renders "—" instead.
    have_breadth = (n_profit + n_loss) > 0
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

        # Use LIVE IBKR NLV as "current" when connected, so trend metrics reflect
        # today's real value even if today's snapshot hasn't been written yet
        # (a missed snapshot used to freeze every trend number against a stale
        # baseline). Fall back to the latest snapshot when disconnected.
        from datetime import date as _date
        today = datetime.now(timezone.utc).astimezone().date()
        if connected and nlv:
            cur_nlv = nlv
            cur_date = today
            # Baseline = most recent snapshot strictly BEFORE today, so we never
            # compare live-now against a snapshot already taken today (or a stale
            # one two days back). If today's snapshot exists it's snaps[-1]; the
            # correct prior is then snaps[-2]; otherwise it's snaps[-1].
            prior = _snapshot_on_or_before(snaps, today - timedelta(days=1))
        else:
            cur_nlv = latest.get("nlv")
            cur_date = latest_date
            prior = snaps[-2]

        # Daily: current vs the prior-day baseline. Snapshots are sparse, so
        # carry the baseline date — the tile labels the real span rather than
        # implying one calendar day.
        if cur_nlv and prior and prior.get("nlv"):
            d_usd = round(cur_nlv - prior["nlv"], 2)
            daily = {"usd": d_usd, "pct": round(d_usd / prior["nlv"] * 100.0, 2),
                     "since": prior.get("snapshot_date")}

        # Weekly: current vs the snapshot on/before 7 calendar days ago. Two
        # honesty guards, because a short/sparse history used to make this tile
        # silently duplicate the daily figure:
        #   1. the baseline must be >= 5 days older than "now"; and
        #   2. it must not be the SAME snapshot the daily figure used — a
        #      weekly number identical to the daily one is not a weekly number.
        if cur_date and cur_nlv:
            wk = _snapshot_on_or_before(snaps, cur_date - timedelta(days=7))
            wk_date = None
            if wk:
                try:
                    wk_date = datetime.strptime(wk["snapshot_date"], "%Y-%m-%d").date()
                except (ValueError, TypeError, KeyError):
                    wk_date = None
            if (wk and wk is not latest and wk is not prior and wk.get("nlv")
                    and wk_date and (cur_date - wk_date).days >= 5):
                w_usd = round(cur_nlv - wk["nlv"], 2)
                weekly = {"usd": w_usd, "pct": round(w_usd / wk["nlv"] * 100.0, 2),
                          "since": wk.get("snapshot_date")}

            # 30-day return.
            m = _snapshot_on_or_before(snaps, cur_date - timedelta(days=30))
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
        "positions_in_profit": n_profit if have_breadth else None,
        "positions_in_loss": n_loss if have_breadth else None,
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


# ── Per-ticker detail (slide-over panel) ───────────────────────────────

def _parse_signal_list(raw) -> list:
    """Coerce a stored signals field ('[]', JSON list, or CSV) to a list."""
    if not raw:
        return []
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(s) for s in v]
    except (json.JSONDecodeError, TypeError):
        pass
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _exit_status_for(ticker, avg_cost, current_price, peak_gain_pct,
                     days_held, entry_signals):
    """Build the Exit Status block: peak gain, active trail, the single
    closest-to-firing condition, and the full five-condition breakdown.

    Mirrors kairos_exits' conditions 1/2/5 math. The exit engine is imported
    lazily with a local fallback so dashboard generation never hard-depends
    on that module importing cleanly (matches the defensive style elsewhere).
    """
    conv_status = "Fresh"
    try:
        from kairos_exits import (
            _exits_config, _get_regime, get_position_signal,
            hard_stop_threshold, trailing_stop_threshold,
        )
        cfg = _exits_config()
        regime = _get_regime()
        signal = get_position_signal(ticker, entry_signals)
        closing_stop, _intraday = hard_stop_threshold(signal, regime, cfg)
        trail = trailing_stop_threshold(peak_gain_pct, cfg)
        decay_cfg = cfg.get("conviction_decay", {})
    except Exception:
        regime, signal = "NORMAL", "STANDARD"
        closing_stop = -8.0
        trail = None
        for min_gain, t in ([15, 8], [30, 10], [50, 12]):
            if peak_gain_pct >= min_gain:
                trail = float(t)
        decay_cfg = {"decay_days": {"HOT-EARNINGS": 10, "HOT-INSIDER": 365,
                     "HOT-CONGRESS": 90, "HOT-OPTIONS": 7, "HOT-REVERSION": 5,
                     "DEFAULT": 7}, "liberation_threshold": 0.5}

    gain_pct = ((current_price - avg_cost) / avg_cost * 100.0) if avg_cost else 0.0

    def _clamp(x):
        return max(0.0, min(100.0, x))

    conditions = []

    # 1) Hard stop-loss — proximity within one stop-width above the trigger.
    span = abs(closing_stop) or 8.0
    hard_prox = _clamp(100.0 * (1 - (gain_pct - closing_stop) / span))
    conditions.append({
        "name": "Hard stop-loss",
        "proximity": round(hard_prox, 1),
        "measurable": True,
        "detail": f"Now {gain_pct:+.1f}% · closing stop {closing_stop:.1f}% ({signal})",
    })

    # 2) Trailing stop — armed only once a peak-gain tier is reached.
    if trail is not None:
        retreat = peak_gain_pct - gain_pct
        trail_prox = _clamp(100.0 * retreat / trail) if trail else 0.0
        conditions.append({
            "name": "Trailing stop",
            "proximity": round(trail_prox, 1),
            "measurable": True,
            "detail": f"Retreated {retreat:.1f}% of {trail:.0f}% trail from peak {peak_gain_pct:.1f}%",
        })
    else:
        conditions.append({
            "name": "Trailing stop",
            "proximity": None,
            "measurable": False,
            "detail": f"Arms at +15% peak (peak so far {peak_gain_pct:.1f}%)",
        })

    # 3) Reversion target reached (condition 5) — governs reversion-led entries.
    entry_set = {str(s).upper() for s in (entry_signals or [])}
    reversion_governs = ("HOT-REVERSION" in entry_set
                         and not (entry_set & {"HOT-INSIDER", "HOT-CONGRESS"}))
    conditions.append({
        "name": "Reversion target reached",
        "proximity": None,
        "measurable": False,
        "detail": ("Active — sells when price reverts to its 30-day mean"
                   if reversion_governs
                   else "Not governing (no reversion-led entry signal)"),
    })

    # 4) Conviction decay / liberation — derived from the longest-horizon
    #    entry signal's decay window vs. days held.
    decay_days = decay_cfg.get("decay_days", {})
    lib_thresh = float(decay_cfg.get("liberation_threshold", 0.5))
    windows = [int(decay_days.get(s, 0)) for s in entry_set if decay_days.get(s)]
    decay_window = max(windows) if windows else int(decay_days.get("DEFAULT", 7))
    remaining = (1.0 - min(1.0, days_held / decay_window)) if decay_window else 0.0
    if days_held >= decay_window:
        conv_status = "Liberated"
    elif remaining <= lib_thresh:
        conv_status = "Decaying"
    else:
        conv_status = "Fresh"
    conditions.append({
        "name": "Conviction decay",
        "proximity": round(_clamp(100.0 * (1 - remaining)), 1),
        "measurable": False,
        "detail": f"{conv_status} — held {days_held}d of {decay_window}d window",
    })

    # 5) Tax-aware hold gate — delays (never triggers) a profitable exit.
    days_to_anniv = 365 - days_held
    near_anniv = 0 < days_to_anniv <= 30 and gain_pct > 0
    conditions.append({
        "name": "Tax-aware hold gate",
        "proximity": None,
        "measurable": False,
        "detail": (f"Delaying profit-taking — {days_to_anniv}d to 1-yr mark"
                   if near_anniv
                   else f"Inactive — {days_to_anniv}d to 1-yr long-term mark"),
    })

    measurable = [c for c in conditions if c["measurable"] and c["proximity"] is not None]
    closest = max(measurable, key=lambda c: c["proximity"]) if measurable else None

    return {
        "gain_pct": round(gain_pct, 2),
        "peak_gain_pct": round(peak_gain_pct, 2),
        "trail_pct": (round(trail, 1) if trail is not None else None),
        "closing_stop_pct": round(closing_stop, 1),
        "signal": signal,
        "regime": regime,
        "closest": closest,
        "conditions": conditions,
        "conviction": {
            "status": conv_status,
            "remaining_pct": round(_clamp(100.0 * remaining), 1),
            "window_days": decay_window,
            "days_held": days_held,
        },
    }


def build_position_details(positions: list, holdings: list,
                           ibkr_connected: bool = False) -> dict:
    """Assemble the full read-only detail record for each open position,
    keyed by raw symbol (no '(SIM)' suffix), for the slide-over panel.

    Read-only: pulls entry rationale/signals from `decisions`, peak gain and
    entry date from `holdings`, and the latest `thesis_reviews` row. Degrades
    gracefully (missing pieces become null) so crypto/sim or freshly-opened
    positions still render a header and facts.
    """
    details: dict = {}

    holding_by_tkr: dict = {}
    for h in holdings or []:
        if h.get("sold_date"):
            continue
        t = (h.get("ticker") or "").upper()
        if t and t not in holding_by_tkr:   # holdings arrive ORDER BY entry_date DESC
            holding_by_tkr[t] = h

    conn = None
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            conn = None

    def _entry_decision(tkr):
        if not conn:
            return None
        try:
            return conn.execute(
                "SELECT timestamp, rationale, data_inputs FROM decisions "
                "WHERE ticker = ? AND action = 'BUY' ORDER BY id DESC LIMIT 1",
                (tkr,)).fetchone()
        except sqlite3.Error:
            return None

    def _thesis(tkr):
        if not conn:
            return None
        try:
            return conn.execute(
                "SELECT timestamp, return_pct, holding_days, current_signals, "
                "sell_triggered, trigger_type, sell_reason FROM thesis_reviews "
                "WHERE ticker = ? ORDER BY id DESC LIMIT 1", (tkr,)).fetchone()
        except sqlite3.Error:
            return None

    try:
        for p in positions or []:
            sym = p.get("symbol", "")
            tkr = sym.split(" ")[0].upper()
            if not tkr or tkr in details:
                continue

            qty = float(p.get("quantity", 0) or 0)
            avg_cost = float(p.get("avg_cost", 0) or 0)
            mkt_val = float(p.get("market_value", 0) or 0)
            upnl = p.get("unrealized_pnl")
            cur_price = (mkt_val / qty) if qty else None
            cost_basis = avg_cost * qty
            upnl_pct = (upnl / cost_basis * 100.0) if (cost_basis and upnl is not None) else None

            h = holding_by_tkr.get(tkr, {})
            entry_date = h.get("entry_date")
            days_held = h.get("holding_days")
            if days_held is None and entry_date:
                try:
                    ed = datetime.strptime(
                        entry_date.replace(" UTC", "").strip(), "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                    days_held = (datetime.now(timezone.utc) - ed).days
                except (ValueError, AttributeError):
                    days_held = None
            peak_gain = float(h.get("peak_gain_pct", 0) or 0)

            entry_signals: list = []
            confluence: dict = {}
            rationale = ""
            dec = _entry_decision(tkr)
            if dec:
                rationale = dec["rationale"] or ""
                try:
                    di = json.loads(dec["data_inputs"]) if dec["data_inputs"] else {}
                    conf = di.get("confluence", {}) or {}
                    entry_signals = conf.get("signals") or di.get("signals") or []
                    confluence = {"score": conf.get("score"), "tier": conf.get("tier"),
                                  "nlv_pct": conf.get("nlv_pct")}
                except (json.JSONDecodeError, TypeError):
                    pass

            thesis = None
            th = _thesis(tkr)
            if th:
                thesis = {
                    "timestamp": th["timestamp"],
                    "return_pct": th["return_pct"],
                    "holding_days": th["holding_days"],
                    "current_signals": _parse_signal_list(th["current_signals"]),
                    "sell_triggered": bool(th["sell_triggered"]),
                    "trigger_type": th["trigger_type"],
                    "sell_reason": th["sell_reason"],
                }

            exit_status = None
            if avg_cost and cur_price is not None and days_held is not None:
                exit_status = _exit_status_for(
                    tkr, avg_cost, cur_price, peak_gain, int(days_held), entry_signals)

            # Chart: native price line since entry, plus stop-level prices for
            # the horizontal reference lines. Crypto/SIM have no thesis_reviews.
            is_crypto = p.get("assetClass", "equity") == "crypto"
            series = ([] if is_crypto else
                      build_price_series(tkr, entry_date, avg_cost, None, cur_price, conn=conn))
            hard_stop_price = trail_stop_price = None
            if exit_status and avg_cost:
                csp = exit_status.get("closing_stop_pct")
                if csp is not None:
                    hard_stop_price = round(avg_cost * (1 + csp / 100.0), 2)
                tp = exit_status.get("trail_pct")
                pk = exit_status.get("peak_gain_pct")
                if tp is not None and pk is not None:
                    trail_stop_price = round(avg_cost * (1 + pk / 100.0) * (1 - tp / 100.0), 2)

            details[tkr] = {
                "symbol": sym,
                "ticker": tkr,
                "name": p.get("name", ""),
                "asset_class": p.get("assetClass", "equity"),
                "sector": p.get("sector", ""),
                "current_price": round(cur_price, 2) if cur_price is not None else None,
                "quantity": qty,
                "avg_cost": avg_cost,
                "market_value": round(mkt_val, 2),
                "unrealized_pnl": upnl,
                "unrealized_pct": round(upnl_pct, 2) if upnl_pct is not None else None,
                "entry_date": entry_date,
                "days_held": int(days_held) if days_held is not None else None,
                "entry_signals": entry_signals,
                "confluence": confluence,
                "rationale": rationale,
                "exit_status": exit_status,
                "thesis": thesis,
                # Chart data.
                "price_series": series,
                "entry_t": (_parse_trade_date(entry_date).strftime("%Y-%m-%d")
                            if _parse_trade_date(entry_date) else None),
                "exit_t": None,
                "hard_stop_price": hard_stop_price,
                "trail_stop_price": trail_stop_price,
            }
    finally:
        if conn:
            conn.close()

    # IBKR fallback: freshly-opened equity positions have too few native
    # points to chart. If IBKR is connected, backfill true 30-day daily closes
    # in one connection (keeping the entry marker + stop lines).
    if ibkr_connected:
        need = [tkr for tkr, d in details.items()
                if d.get("asset_class") != "crypto" and len(d.get("price_series") or []) < 3]
        if need:
            hist = _ibkr_historical_series(need)
            for tkr, pts in hist.items():
                d = details.get(tkr)
                if not d:
                    continue
                # Extend to 'now' with the live price so the line reaches today.
                cur = d.get("current_price")
                if cur is not None:
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    pts = [pt for pt in pts if pt["t"] != today] + [{"t": today, "p": cur}]
                d["price_series"] = pts

    return details


# ── System health strip ────────────────────────────────────────────────

_SCHED_LOG   = os.path.join(SCRIPT_DIR, "kairos_scheduler.log")
_COMMANDER_PID = "/tmp/kairos_commander.pid"
_REGIME_STATE  = os.path.join(SCRIPT_DIR, ".kairos_regime_state.json")

# Regime → health level (green/amber/red language).
_REGIME_LEVEL = {"NORMAL": "ok", "CAUTION": "warn",
                 "RISK-OFF": "down", "EXTREME-FEAR": "down"}


def _et_zone():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception:
        return None


def _in_rth(dt_utc: datetime) -> bool:
    """True if dt (UTC) falls in the equity window (09:30–16:00 ET, Mon–Fri)."""
    et = _et_zone()
    if et is None:
        return False
    d = dt_utc.astimezone(et)
    if d.weekday() >= 5:
        return False
    hm = d.hour * 100 + d.minute
    return 930 <= hm < 1600


def _et_hhmm(dt_utc: datetime) -> str:
    et = _et_zone()
    d = dt_utc.astimezone(et) if et else dt_utc
    return d.strftime("%H:%M ET")


def _typical_cycle_minutes(default: float = 30.0) -> float:
    """Median gap (minutes) between the last N completed cycles, from the
    scheduler log. Falls back to `default` if there's not enough same-session
    data. Replaces a hardcoded 30-min cadence assumption that drifted out of
    sync with actual cycle runtime (observed ~50min as of 2026-07-01) —
    self-adjusts instead of needing another manual edit next time it drifts.
    """
    try:
        stamps = []
        with open(_SCHED_LOG) as f:
            for line in f:
                if "] PASS —" in line or "] FAIL —" in line:
                    s = line[1:line.index("]")].rstrip("Z")
                    try:
                        stamps.append(datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
                                      .replace(tzinfo=timezone.utc))
                    except ValueError:
                        pass
        stamps = stamps[-15:]  # recent window only
        gaps = []
        for a, b in zip(stamps, stamps[1:]):
            g = (b - a).total_seconds() / 60.0
            if 5 <= g <= 120:  # exclude overnight/weekend rollovers
                gaps.append(g)
        if len(gaps) >= 3:
            return statistics.median(gaps)
    except Exception:
        pass
    return default


def _next_rth_open(dt_utc: datetime):
    """Next weekday 09:30 ET open at/after dt_utc. Returns (open_dt_utc, same_day)."""
    et = _et_zone()
    if et is None:
        return None, False
    d = dt_utc.astimezone(et)
    step = 1 if (d.hour * 100 + d.minute) >= 1600 or d.weekday() >= 5 else 0
    nd = d + timedelta(days=step) if step else d
    while nd.weekday() >= 5:
        nd = nd + timedelta(days=1)
    same_day = nd.date() == d.date()
    open_et = nd.replace(hour=9, minute=30, second=0, microsecond=0)
    return open_et.astimezone(timezone.utc), same_day


def build_system_health(ibkr: dict) -> dict:
    """Ambient system-health facts for the top strip. Read-only; each item is
    {label, value, level} with level ∈ {ok, warn, down, info}. Every probe is
    defensive so a missing log / pid / table never breaks dashboard generation.
    """
    now = datetime.now(timezone.utc)
    items: dict = {}
    last_ts = None   # UTC datetime of the last cycle, for the next-cycle estimate
    cadence = _typical_cycle_minutes()  # observed, not assumed — see docstring above

    # ── Last cycle: last PASS/FAIL line in the scheduler log ─────────────
    last = {"label": "Last cycle", "value": "unknown", "level": "warn"}
    try:
        last_line = None
        with open(_SCHED_LOG) as f:
            for line in f:
                if "] PASS —" in line or "] FAIL —" in line:
                    last_line = line.strip()
        if last_line and last_line.startswith("["):
            stamp = last_line[1:last_line.index("]")].rstrip("Z")
            ts = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            last_ts = ts
            ok = "] PASS —" in last_line
            mins = (now - ts).total_seconds() / 60.0
            level = "ok" if ok else "down"
            # A green run that's gone well past the observed cadence during
            # RTH is a warning — threshold floats with actual cycle time
            # instead of a hardcoded 40min that drifts out of sync.
            if ok and _in_rth(now) and mins > cadence * 1.5:
                level = "warn"
            last = {"label": "Last cycle",
                    "value": f"{_et_hhmm(ts)} · {'success' if ok else 'ERROR'}",
                    "level": level}
    except FileNotFoundError:
        last["value"] = "no log"
    except Exception:
        pass
    items["last_cycle"] = last

    # ── Next scheduled cycle: DERIVED estimate (not tracked anywhere) ────
    # launchd fires every 1800s from load time, but a still-running cycle
    # blocks the next fire, so real cadence runs longer — use the observed
    # median (`cadence`) rather than the nominal 30min interval.
    nxt = {"label": "Next", "value": "—", "level": "info"}
    et = _et_zone()
    if et is not None:
        if _in_rth(now):
            base = last_ts if last_ts else now
            cand = base + timedelta(minutes=cadence)
            if cand < now:
                cand = now + timedelta(minutes=cadence)
            if not _in_rth(cand):
                cand = None  # would fall past the close → next open
            nxt["value"] = (f"~{_et_hhmm(cand)} (est)" if cand
                            else "~09:30 ET next day (est)")
        else:
            open_dt, same_day = _next_rth_open(now)
            if open_dt is not None:
                nxt["value"] = ("~09:30 ET (est)" if same_day
                                else f"~{open_dt.astimezone(et).strftime('%a')} 09:30 ET (est)")
    items["next_cycle"] = nxt

    # ── Commander daemon: PID file + signal-0 probe ─────────────────────
    cmd = {"label": "Commander", "value": "dead", "level": "down"}
    try:
        with open(_COMMANDER_PID) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            cmd = {"label": "Commander", "value": "alive", "level": "ok"}
        except PermissionError:
            cmd = {"label": "Commander", "value": "alive", "level": "ok"}
        except (ProcessLookupError, OSError):
            cmd = {"label": "Commander", "value": "dead (stale pid)", "level": "down"}
    except (FileNotFoundError, ValueError):
        cmd = {"label": "Commander", "value": "dead (no pid)", "level": "down"}
    items["commander"] = cmd

    # ── IBKR connection (from the payload's ibkr dict) ──────────────────
    if ibkr.get("connected"):
        if ibkr.get("marks_stale"):
            items["ibkr"] = {"label": "IBKR", "value": "connected · delayed", "level": "warn"}
        else:
            items["ibkr"] = {"label": "IBKR", "value": "connected", "level": "ok"}
    else:
        items["ibkr"] = {"label": "IBKR", "value": "disconnected", "level": "down"}

    # ── Regime: fresh regime_log row; flag if the state file has diverged ─
    reg = {"label": "Regime", "value": "unknown", "level": "warn"}
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT regime, timestamp FROM regime_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row and row["regime"]:
            regime = str(row["regime"]).upper()
            level = _REGIME_LEVEL.get(regime, "info")
            note = ""
            # Divergence check: the on-disk state file the exit engine reads.
            try:
                with open(_REGIME_STATE) as f:
                    js = json.load(f)
                if str(js.get("regime", "")).upper() != regime:
                    note = "state file stale"
                    if level == "ok":
                        level = "warn"
            except Exception:
                pass
            reg = {"label": "Regime",
                   "value": regime + (f" · {note}" if note else ""),
                   "level": level}
    except Exception:
        pass
    items["regime"] = reg

    # ── Market hours: open/closed + count up/down against the close/open ──
    mkt = {"label": "Market", "value": "—", "level": "info"}
    if et is not None:
        if _in_rth(now):
            close_et = now.astimezone(et).replace(hour=16, minute=0, second=0, microsecond=0)
            remain = (close_et.astimezone(timezone.utc) - now).total_seconds() / 60.0
            h, m = int(remain // 60), int(remain % 60)
            mkt = {"label": "Market",
                   "value": f"OPEN · closes in {h}h {m:02d}m",
                   "level": "ok"}
        else:
            open_dt, same_day = _next_rth_open(now)
            if open_dt is not None:
                until = (open_dt - now).total_seconds() / 60.0
                h, m = int(until // 60), int(until % 60)
                when = f"in {h}h {m:02d}m" if same_day else f"{open_dt.astimezone(et).strftime('%a')} 09:30 ET"
                mkt = {"label": "Market",
                       "value": f"closed · opens {when}",
                       "level": "info"}
    items["market"] = mkt

    return items


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

    # Closed-trade list + slide-over detail records. Resolve company names for
    # closed tickers too (the open-position name map above only covers holdings
    # still open), then merge the closed detail records — keyed by numeric
    # holding id — into the same position_details map the panel reads.
    closed_trades = build_closed_trades(db.get("holdings", []))
    position_details = build_position_details(
        positions, db.get("holdings", []), ibkr_connected=bool(ibkr.get("connected")))
    try:
        closed_name_map = _resolve_ticker_names(
            sorted({t["ticker"] for t in closed_trades if t.get("ticker")}))
    except Exception as _cn_err:
        print(f"  WARNING: closed-name resolution failed: {_cn_err}")
        closed_name_map = {}
    position_details.update(build_closed_details(closed_trades, closed_name_map))

    # Build the payload dictionary
    return {
        "metrics":        compute_metrics(perf, db),
        "portfolio_metrics": compute_portfolio_metrics(ibkr, load_nlv_snapshots()),
        "series":         make_series(perf),
        "weekly":         make_weekly(perf),
        "decisions":      decisions_display,
        "decisions_counts": db.get("decisions_counts", {"executed": 0, "skipped": 0}),
        "positions":      positions,
        "position_details": position_details,
        "closed_trades":  closed_trades,
        "sector_breakdown": _compute_sector_exposure(positions),
        "size_style_breakdown": _compute_size_style_exposure(positions),
        "short_positions": _compute_short_positions(positions),
        "exit_coverage":  _compute_exit_coverage(closed_trades),
        "chain_tier_breakdown": compute_chain_tier_breakdown(price_map),
        "system_health":  build_system_health(ibkr),
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
    /* ── Design tokens ──────────────────────────────────────────────────
       Variable NAMES are unchanged from the original terminal palette so
       every existing rule inherits the new look without markup edits; only
       the VALUES move. Saturated neon (#00e676 / #ff4466) reads as a gaming
       HUD; institutional reporting wants desaturated semantic colour and a
       single accent. --cyan is retained as the accent slot name.            */
    :root {
      --bg: #0b1020;
      --surface: #121829;
      --surface2: #171e33;
      --border: #242d47;
      --border2: #2f3a58;
      --text: #e7ecf6;
      --dim: #98a5be;
      --muted: #67748f;
      --cyan: #5b8def;
      --cyan-dim: rgba(91,141,239,0.14);
      --green: #3fa87a;
      --green-dim: rgba(63,168,122,0.13);
      --amber: #c9973f;
      --amber-dim: rgba(201,151,63,0.13);
      --red: #d1656b;
      --red-dim: rgba(209,101,107,0.13);
      --purple: #8a7bd8;
      --purple-dim: rgba(138,123,216,0.13);
      --hover: rgba(255,255,255,0.035);
      --shadow: 0 1px 2px rgba(0,0,0,.35), 0 10px 26px -16px rgba(0,0,0,.7);
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
      --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    }
    /* Light theme: allocators read in daylight and print to PDF. The toggle
       must beat the OS preference in both directions, so the media query is
       scoped with :where() to keep it below the explicit data-theme stamp. */
    :root[data-theme="light"] {
      --bg: #f5f7fb;
      --surface: #ffffff;
      --surface2: #f4f6fa;
      --border: #dee3ed;
      --border2: #cdd5e3;
      --text: #111726;
      --dim: #525d73;
      --muted: #7d8799;
      --cyan: #2f62d6;
      --cyan-dim: rgba(47,98,214,0.10);
      --green: #1c7a52;
      --green-dim: rgba(28,122,82,0.09);
      --amber: #8f6b1c;
      --amber-dim: rgba(143,107,28,0.09);
      --red: #b3373f;
      --red-dim: rgba(179,55,63,0.09);
      --purple: #5b49b0;
      --purple-dim: rgba(91,73,176,0.09);
      --hover: rgba(16,24,40,0.035);
      --shadow: 0 1px 2px rgba(16,24,40,.04), 0 10px 26px -18px rgba(16,24,40,.3);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    /* Equal-width digits wherever figures align vertically — table rows and
       stat lists. Large standalone figures take proportional digits, which is
       handled per-element below. */
    table, .pm-val, .mval, .rval, .so-fact-val, .so-kv-val,
    .cl-count, .sh-val, .last-updated { font-variant-numeric: tabular-nums; }
    .mono, .tkr { font-family: var(--mono); }
    body {
      font-family: var(--sans);
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
    /* 4px letterspacing on a 20px wordmark was wide enough to wrap on narrow
       viewports and read as terminal chrome. Tightened, with the accent
       carrying the brand rather than the tracking. */
    .logo {
      font-size: 18px; font-weight: 640; color: var(--text);
      letter-spacing: -0.2px; white-space: nowrap;
    }
    .logo span { color: var(--dim); font-weight: 450; }
    .header-sub { font-size: 12px; color: var(--dim); margin-top: 2px; }
    .header-right { text-align: right; font-size: 11px; color: var(--dim); }
    .dot {
      display: inline-block; width: 7px; height: 7px;
      border-radius: 50%; margin-right: 5px; vertical-align: middle;
    }
    .dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); animation: blink 2s infinite; }
    .dot-amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
    .dot-red   { background: var(--red);   box-shadow: 0 0 6px var(--red); }
    @keyframes blink { 0%,100% { opacity:1 } 50% { opacity:.3 } }
    /* ── System health strip ── */
    .sys-health {
      display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 9px 16px; margin-bottom: 18px;
    }
    .sh-title {
      font-size: 11px; font-weight: 600; color: var(--muted); margin-right: 4px;
    }
    .sh-item { display: inline-flex; align-items: center; font-size: 11px; color: var(--dim); }
    .sh-item .dot { animation: none; box-shadow: none; }
    .sh-item.ok   .dot { background: var(--green); }
    .sh-item.warn .dot { background: var(--amber); }
    .sh-item.down .dot { background: var(--red); }
    .sh-item.info .dot { background: var(--border2); }
    .sh-label { color: var(--muted); margin-right: 5px; }
    .sh-val { color: var(--text); }
    .sh-item.down .sh-val { color: var(--red); }
    .sh-item.warn .sh-val { color: var(--amber); }
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
    .closed-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px;
    }
    .closed-empty { color: var(--muted); padding: 20px 4px; letter-spacing: 1px; font-size: 12px; }
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
    .mlabel { font-size: 11.5px; color: var(--dim); margin-bottom: 7px; font-weight: 500; }
    .mval { font-size: 22px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .mval.cyan   { color: var(--cyan); }
    .mval.green  { color: var(--green); }
    .mval.amber  { color: var(--amber); }
    .mval.red    { color: var(--red); }
    .mval.purple { color: var(--purple); }
    .mval.white  { color: var(--text); }
    .msub { font-size: 10px; color: var(--dim); }
    /* ── Hero: one focal figure + the equity curve ──────────────────────
       Replaces the 9-tile metrics grid, which had no rank (every tile the
       same weight) and left an orphan on a third row. The era ambiguity that
       the old .era-note paragraph apologised for in prose is now a control. */
    .hero {
      display: grid; grid-template-columns: 260px minmax(0, 1fr);
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; box-shadow: var(--shadow);
      margin-bottom: 12px; overflow: hidden;
    }
    .hero > * { min-width: 0; }
    .hl { padding: 18px 20px; border-right: 1px solid var(--border); }
    .eyebrow { font-size: 11px; font-weight: 600; color: var(--dim); margin-bottom: 6px; }
    /* Proportional figures: equal-width digits read loose at display sizes. */
    .hv {
      font-size: 33px; font-weight: 660; letter-spacing: -1.1px; line-height: 1;
      color: var(--text); font-variant-numeric: proportional-nums;
    }
    .hd { display: flex; align-items: baseline; gap: 8px; margin-top: 8px;
          font-size: 12.5px; color: var(--dim); flex-wrap: wrap; }
    .chip { font-size: 12px; font-weight: 620; padding: 2px 7px; border-radius: 5px; }
    .chip.up   { background: var(--green-dim); color: var(--green); }
    .chip.down { background: var(--red-dim);   color: var(--red); }
    .chip.flat { background: var(--cyan-dim);  color: var(--dim); }
    .hmeta { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }
    .hmeta div { display: flex; justify-content: space-between; gap: 10px;
                 padding: 3px 0; font-size: 12px; }
    .hmeta dt { color: var(--dim); }
    .hmeta dd { color: var(--text); font-weight: 550; font-variant-numeric: tabular-nums; }
    .hr { padding: 14px 18px 12px; }
    .crow { display: flex; justify-content: space-between; align-items: center;
            gap: 12px; margin-bottom: 10px; }
    .hr-title { font-size: 12px; color: var(--dim); }
    /* Segmented control — the era switch */
    .seg { display: inline-flex; padding: 2px; gap: 2px; background: var(--bg);
           border: 1px solid var(--border); border-radius: 7px; }
    .seg button {
      font: inherit; font-size: 11px; font-weight: 550; padding: 4px 11px;
      border: 0; border-radius: 5px; background: transparent; color: var(--dim);
      cursor: pointer; white-space: nowrap;
    }
    .seg button[aria-pressed="true"] {
      background: var(--surface2); color: var(--text); box-shadow: var(--shadow);
    }
    .seg button:hover { color: var(--text); }
    .seg:empty { display: none; }
    /* ── Tile strip ─────────────────────────────────────────────────────
       For a small parts-of-a-whole set where one slice dominates. The AI
       chain is ~92% non-chain, so a stacked bar renders the three tiers as
       invisible slivers; the readable content is the total and its split.
       Magnitude, not identity — one hue, no categorical palette needed.      */
    .tilestrip {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 1px; background: var(--border); border: 1px solid var(--border);
      border-radius: 10px; overflow: hidden; box-shadow: var(--shadow);
      margin-bottom: 18px;
    }
    .tile { background: var(--surface); padding: 13px 16px; min-width: 0; }
    .tile.lead { background: var(--surface2); }
    .tile.lead .tval { color: var(--cyan); }
    .tlabel { font-size: 11.5px; color: var(--dim); font-weight: 550; margin-bottom: 5px; }
    .tval { font-size: 19px; font-weight: 640; letter-spacing: -.4px; color: var(--text);
            font-variant-numeric: proportional-nums; }
    .tbar { height: 4px; border-radius: 2px; background: var(--bg); margin: 9px 0 8px; overflow: hidden; }
    .tbar i { display: block; height: 100%; border-radius: 2px; background: var(--cyan); }
    .tmeta { display: flex; justify-content: space-between; gap: 8px;
             font-size: 11px; color: var(--dim); font-variant-numeric: tabular-nums; }
    /* ── Positions table ── */
    .postbl th.r, .postbl td.r { text-align: right; }
    .postbl td { white-space: nowrap; }
    .postbl td.tkr { font-weight: 640; color: var(--cyan); }
    .postbl td.tkr.crypto { color: var(--purple); }
    .postbl td.nm {
      color: var(--dim); max-width: 200px;
      overflow: hidden; text-overflow: ellipsis;
    }
    .postbl td.sec { color: var(--dim); font-size: 11.5px; }
    .dimval { color: var(--muted); }
    .wcell { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
    .wbar { display: inline-block; width: 52px; height: 4px; border-radius: 2px;
            background: var(--border2); flex: none; }
    .wbar i { display: block; height: 100%; border-radius: 2px; background: var(--cyan); }
    /* Totals row sticks to the bottom of the scroll box so it stays readable
       while the book scrolls behind it. */
    .postbl tfoot td {
      position: sticky; bottom: 0; background: var(--surface2);
      border-top: 1px solid var(--border2); border-bottom: 0;
      font-weight: 640; z-index: 1;
    }
    .postbl tfoot td.nm { color: var(--dim); font-weight: 450; }
    .postbl tfoot .wbar { background: transparent; }
    /* ── Stat list: label left, value right ── */
    .stats { padding: 2px 0; }
    .st {
      display: flex; justify-content: space-between; align-items: baseline; gap: 14px;
      padding: 7px 14px; border-bottom: 1px solid var(--border);
    }
    .st:last-child { border-bottom: 0; }
    .st:hover { background: var(--hover); }
    .st dt { font-size: 12px; color: var(--dim); }
    .st dd {
      font-size: 13.5px; font-weight: 600; color: var(--text); text-align: right;
      font-variant-numeric: tabular-nums; white-space: nowrap;
    }
    .stnote { display: block; font-size: 10.5px; color: var(--muted); margin-top: 1px; }
    .stsub  { display: block; font-size: 10.5px; color: var(--dim); font-weight: 450; margin-top: 1px; }
    .tbar i.green { background: var(--green); }
    .tbar i.red   { background: var(--red); }
    /* ── Three-column section, cards sized to content ── */
    .cols3 {
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px 18px; margin-bottom: 18px; align-items: start;
    }
    @media(max-width:1100px) { .cols3 { grid-template-columns: 1fr; } }
    /* ── Two-column section, cards sized to content ── */
    .cols2 {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px 18px; margin-bottom: 18px; align-items: start;
    }
    .card.np { padding: 0; overflow: hidden; }
    .ph {
      display: flex; align-items: baseline; justify-content: space-between; gap: 10px;
      font-size: 12.5px; font-weight: 600; color: var(--text);
      padding: 11px 14px 10px; border-bottom: 1px solid var(--border);
    }
    .phn { font-size: 11px; font-weight: 450; color: var(--dim); }
    /* Exposure rows: label, value, share, magnitude bar. */
    .exp { width: 100%; border-collapse: collapse; }
    .exp td { padding: 7px 14px; font-size: 12.5px; border-bottom: 1px solid var(--border);
              white-space: nowrap; }
    .exp tr:last-child td { border-bottom: 0; }
    .exp tr:hover td { background: var(--hover); }
    .exp .v, .exp .p { text-align: right; font-variant-numeric: tabular-nums; }
    .exp .p { color: var(--dim); width: 62px; }
    .exp tr.over .p { color: var(--red); }
    .expbar { width: 56px; }
    .expbar span { display: block; height: 4px; border-radius: 2px; background: var(--border2); }
    .expbar i { display: block; height: 100%; border-radius: 2px; background: var(--cyan); }
    .exp tr.over .expbar i { background: var(--red); }
    .overtag {
      font-size: 10.5px; font-weight: 600; color: var(--red); background: var(--red-dim);
      padding: 1px 6px; border-radius: 4px; margin-left: 7px;
    }
    .warnbar {
      font-size: 11.5px; color: var(--text); background: var(--surface2);
      border: 1px solid var(--border); border-left: 2px solid var(--amber);
      border-radius: 6px; padding: 8px 13px; margin-bottom: 14px;
    }
    .warnbar b { font-weight: 600; }
    @media(max-width:900px) { .cols2 { grid-template-columns: 1fr; } }
    /* ── KPI strip ── */
    .kpis {
      display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 1px;
      background: var(--border); border: 1px solid var(--border);
      border-radius: 10px; overflow: hidden; margin-bottom: 18px;
      box-shadow: var(--shadow);
    }
    .kpis > * { min-width: 0; }
    .k { background: var(--surface); padding: 11px 13px; }
    .k dt { font-size: 11px; color: var(--dim); margin-bottom: 5px; font-weight: 500; }
    .k dd { font-size: 17px; font-weight: 620; letter-spacing: -.4px; color: var(--text);
            font-variant-numeric: proportional-nums; }
    .k dd.green { color: var(--green); } .k dd.red { color: var(--red); }
    .k dd.amber { color: var(--amber); } .k dd.cyan { color: var(--cyan); }
    .kf { font-size: 10.5px; color: var(--dim); margin-top: 3px; line-height: 1.4; }
    @media(max-width:1250px) { .kpis { grid-template-columns: repeat(4, minmax(0,1fr)); } }
    @media(max-width:900px) {
      .hero { grid-template-columns: 1fr; }
      .hl { border-right: 0; border-bottom: 1px solid var(--border); }
      .kpis { grid-template-columns: repeat(2, minmax(0,1fr)); }
    }
    /* ── Portfolio metrics panel ── */
    .pm-subhdr { font-size: 11px; color: var(--muted); margin: 4px 0 12px; font-weight: 600; }
    .pm-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 16px; }
    .pm-tile { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 11px 13px; }
    .pm-label { font-size: 11.5px; color: var(--dim); margin-bottom: 6px; font-weight: 500; }
    .pm-val { font-size: 17px; font-weight: 700; line-height: 1.05; color: var(--text); }
    .pm-sub { font-size: 9.5px; color: var(--dim); margin-top: 3px; }
    @media(max-width:1100px) { .pm-grid { grid-template-columns: repeat(3,1fr); } }
    @media(max-width:600px)  { .pm-grid { grid-template-columns: repeat(2,1fr); } }
    /* ── Monthly performance table ── */
    .pm-monthly { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
    .pm-monthly th {
      font-size: 11.5px; color: var(--dim); text-align: right; font-weight: 600;
      padding: 8px 10px; border-bottom: 1px solid var(--border);
    }
    .pm-monthly th:first-child, .pm-monthly td:first-child { text-align: left; }
    .pm-monthly td {
      font-size: 12px; text-align: right; color: var(--text);
      padding: 8px 10px; border-bottom: 1px solid var(--border);
      font-variant-numeric: tabular-nums;
    }
    .pm-monthly tr:last-child td { border-bottom: none; }
    .pm-monthly td.pm-month { color: var(--cyan); font-weight: 600; }
    .pm-monthly-note { font-size: 9.5px; color: var(--dim); margin-bottom: 16px; }
    /* ── Cards ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px 22px;
      margin-bottom: 18px;
    }
    .section-hdr {
      font-size: 14px;
      letter-spacing: -0.1px;
      color: var(--text);
      font-weight: 620;
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
    .rlabel { font-size: 11.5px; color: var(--dim); margin-bottom: 8px; font-weight: 500; }
    .rval { font-size: 28px; font-weight: 700; }
    .rsub { font-size: 10px; color: var(--dim); margin-top: 4px; }
    /* ── Tables ── */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th {
      text-align: left; color: var(--dim); font-size: 11.5px; font-weight: 600;
      padding: 9px 12px; border-bottom: 1px solid var(--border);
      background: var(--surface2);
    }
    td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--hover); }
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
      .charts-2col  { grid-template-columns: 1fr; }
      .risk-3col    { grid-template-columns: repeat(2,1fr); }
      .closed-grid  { grid-template-columns: repeat(3,1fr); }
    }
    @media(max-width:600px) { body { padding: 12px; } .closed-grid { grid-template-columns: repeat(2,1fr); } }
    /* ── Position rows are clickable ── */
    #positions-wrap tbody tr { cursor: pointer; transition: background 0.12s; }
    #positions-wrap tbody tr:hover td { background: var(--cyan-dim); }
    #positions-wrap tbody tr td:first-child { position: relative; }
    #positions-wrap tbody tr:hover td:first-child::before {
      content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: var(--cyan);
    }
    /* ── Closed-positions filters ── */
    .cl-filters {
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 12px; margin-bottom: 18px;
    }
    .cl-filter-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
    .cl-select {
      background: var(--bg); color: var(--text);
      border: 1px solid var(--border); border-radius: 6px;
      padding: 7px 30px 7px 11px; font-family: inherit; font-size: 11px;
      letter-spacing: 0.5px; cursor: pointer; appearance: none;
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path d='M1 1l4 4 4-4' stroke='%236a80a8' stroke-width='1.5' fill='none'/></svg>");
      background-repeat: no-repeat; background-position: right 11px center;
      transition: border-color 0.12s;
    }
    .cl-select:hover, .cl-select:focus { border-color: var(--cyan); outline: none; }
    .cl-select.active { border-color: var(--cyan); color: var(--cyan); }
    .cl-pills { display: inline-flex; gap: 4px; }
    .cl-pill {
      background: var(--bg); color: var(--dim);
      border: 1px solid var(--border); border-radius: 6px;
      padding: 7px 12px; font-family: inherit; font-size: 11px;
      letter-spacing: 0.5px; cursor: pointer; transition: color 0.12s, border-color 0.12s;
    }
    .cl-pill:hover { color: var(--text); border-color: var(--border2); }
    .cl-pill.active { color: var(--cyan); border-color: var(--cyan); background: var(--cyan-dim); }
    .cl-filter-state { display: flex; align-items: center; gap: 12px; }
    .cl-count { font-size: 11px; color: var(--dim); letter-spacing: 0.5px; }
    .cl-clear {
      background: none; border: none; color: var(--muted);
      font-family: inherit; font-size: 11px; letter-spacing: 0.5px;
      cursor: pointer; padding: 4px 2px; transition: color 0.12s;
    }
    .cl-clear:hover { color: var(--red); }
    .cl-filter-state.hidden { display: none; }
    @media(max-width:600px) { .cl-filters { flex-direction: column; align-items: stretch; } }
    /* ── Closed-trade list ── */
    #closed-list { margin-top: 18px; }
    #closed-list tbody tr { cursor: pointer; transition: background 0.12s; }
    #closed-list tbody tr:hover td { background: var(--cyan-dim); }
    #closed-list tbody tr td:first-child { position: relative; }
    #closed-list tbody tr:hover td:first-child::before {
      content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: var(--cyan);
    }
    #closed-list .cl-tkr  { font-weight: 700; color: var(--cyan); }
    #closed-list .cl-date { color: var(--dim); font-size: 11px; white-space: nowrap; }
    #closed-list .cl-days { color: var(--text); }
    #closed-list td, #closed-list th { white-space: nowrap; }
    /* ── Signal analytics table ── */
    /* ── Diverging bar list — bars grow left/right from a centre baseline ── */
    .dvrow {
      display: grid; grid-template-columns: 150px 150px minmax(0, 1fr) 100px;
      align-items: center; gap: 12px; padding: 7px 4px;
      border-bottom: 1px solid var(--border);
    }
    .dvrow:last-child { border-bottom: 0; }
    .dvrow:not(.dvhead):hover { background: var(--hover); }
    .dvlabel {
      font-size: 12px; font-weight: 620; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis;
    }
    /* Fixed tracks, right-aligned: space-between sizes each cell to its own
       text, so a 6-char header drifts off the 2-char figure below it. */
    .dvmeta {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px;
      font-size: 11px; color: var(--dim); text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .dvtrack { position: relative; height: 14px; }
    .dvaxis { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--border2); }
    .dvbar { position: absolute; top: 3px; height: 8px; border-radius: 3px; }
    /* 2px gap at the baseline so a bar never touches the axis rule. */
    .dvbar.p { left: calc(50% + 2px); background: var(--green); }
    .dvbar.n { right: calc(50% + 2px); background: var(--red); }
    .dvval { font-size: 12px; font-weight: 600; text-align: right;
             font-variant-numeric: tabular-nums; }
    .dvhead { background: var(--surface2); border-bottom: 1px solid var(--border2); }
    .dvhead .dvmeta span, .dvhead .dvval { font-size: 11px; font-weight: 600; color: var(--dim); }
    .caveat {
      font-size: 11px; color: var(--amber); line-height: 1.5;
      padding: 10px 4px 2px; border-top: 1px solid var(--border); margin-top: 8px;
    }
    /* A coverage note is context, not a warning — it must not compete with the
       attribution caveat, which flags data you genuinely cannot trust. */
    .caveat.neutral { color: var(--dim); padding: 10px 14px 12px; margin-top: 0; }
    .exitbl th { font-size: 11.5px; }
    .exitbl th.r, .exitbl td.r { text-align: right; }
    .exitbl td { padding: 7px 14px; }
    .exitbl .wcell { justify-content: flex-end; }
    .sig-pill {
      display: inline-flex; align-items: center; gap: 6px;
      font-weight: 700; color: var(--text); font-size: 11px;
    }
    .sig-pill .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; background: var(--cyan); }
    .sig-pill.muted { color: var(--muted); font-weight: 400; }
    .sig-pill.muted .dot { background: var(--muted); }
    .exit-chip {
      display: inline-flex; align-items: center; gap: 6px;
      font-size: 11px; color: var(--text);
    }
    .exit-chip .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
    .exit-chip.cyan  .dot { background: var(--cyan); }
    .exit-chip.amber .dot { background: var(--amber); }
    .exit-chip.red   .dot { background: var(--red); }
    .exit-chip.green .dot { background: var(--green); }
    .exit-chip.white .dot { background: var(--border2); }
    .exit-chip.muted { color: var(--muted); }
    .exit-chip.muted .dot { background: var(--muted); }
    .cl-more-wrap { text-align: center; margin-top: 14px; }
    .cl-more-btn {
      background: var(--surface2); color: var(--dim);
      border: 1px solid var(--border); border-radius: 6px;
      padding: 8px 20px; font-family: inherit; font-size: 12px; cursor: pointer;
      transition: color 0.12s, border-color 0.12s;
    }
    .cl-more-btn:hover { color: var(--cyan); border-color: var(--cyan); }
    /* ── Slide-over panel ── */
    .so-overlay {
      position: fixed; inset: 0; z-index: 90;
      background: color-mix(in srgb, var(--bg) 78%, transparent); backdrop-filter: blur(3px); -webkit-backdrop-filter: blur(3px);
      opacity: 0; visibility: hidden; transition: opacity 0.28s ease, visibility 0.28s ease;
    }
    .so-overlay.open { opacity: 1; visibility: visible; }
    .so-panel {
      position: fixed; top: 0; right: 0; bottom: 0; z-index: 91;
      width: min(40%, 560px); max-width: 100vw;
      background: var(--surface2); border-left: 1px solid var(--border2);
      box-shadow: -24px 0 60px rgba(0,0,0,0.55);
      transform: translateX(100%); transition: transform 0.32s cubic-bezier(0.22,0.61,0.36,1);
      display: flex; flex-direction: column; overflow: hidden;
    }
    .so-overlay.open .so-panel { transform: translateX(0); }
    .so-body { overflow-y: auto; padding: 0 26px 32px; flex: 1; }
    .so-body::-webkit-scrollbar { width: 8px; }
    .so-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
    .so-body::-webkit-scrollbar-track { background: transparent; }
    /* Price mini-chart */
    .so-chart-wrap { position: relative; height: 110px; margin: 16px 0 6px; }
    .so-chart-empty {
      position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
      color: var(--muted); font-size: 11px; letter-spacing: 1px;
    }
    .so-chart-legend {
      display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px;
      font-size: 9.5px; color: var(--dim); letter-spacing: 0.5px;
    }
    .so-chart-legend span { display: inline-flex; align-items: center; gap: 5px; }
    .so-chart-legend .lg-dot { width: 7px; height: 7px; border-radius: 50%; }
    .so-chart-legend .lg-dash { width: 12px; border-top: 2px dashed; height: 0; }
    /* Header */
    .so-head {
      position: sticky; top: 0; z-index: 2; background: var(--surface2);
      padding: 24px 26px 18px; border-bottom: 1px solid var(--border);
    }
    .so-close {
      position: absolute; top: 20px; right: 22px; width: 28px; height: 28px;
      border: 1px solid var(--border2); border-radius: 6px; background: var(--surface);
      color: var(--dim); font-size: 15px; line-height: 1; cursor: pointer;
      display: flex; align-items: center; justify-content: center; transition: all 0.18s;
    }
    .so-close:hover { color: var(--text); border-color: var(--red); background: var(--red-dim); }
    .so-sym { font-size: 26px; font-weight: 660; letter-spacing: -0.5px; font-variant-numeric: proportional-nums; }
    .so-sym.equity { color: var(--cyan); }
    .so-sym.crypto { color: var(--purple); }
    .so-class {
      font-size: 11px; color: var(--dim);
      border: 1px solid var(--border2); border-radius: 4px; padding: 2px 7px; margin-left: 10px;
      vertical-align: middle;
    }
    .so-name { font-size: 12px; color: var(--dim); margin-top: 4px; }
    .so-price-row { display: flex; align-items: baseline; gap: 16px; margin-top: 14px; flex-wrap: wrap; }
    .so-price { font-size: 20px; font-weight: 700; color: var(--text); }
    .so-pnl-big { font-size: 20px; font-weight: 700; }
    .so-pnl-pct { font-size: 13px; opacity: 0.85; margin-left: 6px; }
    /* Sections */
    .so-sec { margin-top: 26px; }
    .so-sec-hdr {
      font-size: 12.5px; font-weight: 620; color: var(--text);
      margin-bottom: 14px; display: flex; align-items: center; gap: 10px;
    }
    .so-sec-hdr::after { content: ""; flex: 1; height: 1px; background: var(--border); }
    /* Facts grid */
    .so-facts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 22px; }
    .so-fact-label { font-size: 11.5px; color: var(--dim); margin-bottom: 5px; }
    .so-fact-val { font-size: 15px; font-weight: 700; color: var(--text); }
    /* Pills */
    .so-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
    .so-pill {
      font-size: 11px; font-weight: 600; padding: 5px 11px; border-radius: 20px;
      background: var(--cyan-dim); color: var(--cyan); border: 1px solid var(--cyan);
    }
    .so-pill.rev { background: var(--purple-dim); color: var(--purple); border-color: var(--purple); }
    .so-conf-row { display: flex; align-items: center; gap: 12px; }
    .so-conf-dots { display: flex; gap: 4px; }
    .so-conf-dot { width: 18px; height: 5px; border-radius: 3px; background: var(--border2); }
    .so-conf-dot.on { background: var(--cyan); }
    .so-conf-txt { font-size: 11px; color: var(--dim); }
    /* Rationale */
    .so-prose { font-size: 12.5px; line-height: 1.65; color: var(--text); opacity: 0.92; }
    .so-prose.empty { color: var(--muted); font-style: italic; }
    /* Bars */
    .so-bar-track { height: 8px; border-radius: 5px; background: var(--bg); overflow: hidden; border: 1px solid var(--border); }
    .so-bar-fill { height: 100%; border-radius: 5px; transition: width 0.4s ease; }
    .so-bar-meta { display: flex; justify-content: space-between; font-size: 11px; margin: 9px 0 5px; }
    .so-bar-detail { font-size: 10.5px; color: var(--dim); margin-top: 6px; line-height: 1.4; }
    .so-kv-row { display: flex; gap: 26px; margin-bottom: 16px; flex-wrap: wrap; }
    .so-kv-label { font-size: 11.5px; color: var(--dim); margin-bottom: 4px; }
    .so-kv-val { font-size: 16px; font-weight: 700; }
    /* Expandable all-conditions */
    .so-expand { margin-top: 14px; }
    .so-expand-btn {
      font-size: 10px; letter-spacing: 0.5px; color: var(--cyan); background: none; border: none;
      cursor: pointer; padding: 4px 0; font-family: inherit; display: inline-flex; align-items: center; gap: 6px;
    }
    .so-expand-btn:hover { color: var(--text); }
    .so-expand-body { display: none; margin-top: 12px; }
    .so-expand.open .so-expand-body { display: block; }
    .so-expand.open .so-caret { transform: rotate(90deg); }
    .so-caret { transition: transform 0.2s; display: inline-block; }
    .so-cond {
      display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--border);
    }
    .so-cond:last-child { border-bottom: none; }
    .so-cond-name { font-size: 12px; font-weight: 620; color: var(--text); width: 150px; flex-shrink: 0; }
    .so-cond-detail { font-size: 10px; color: var(--dim); line-height: 1.35; flex: 1; }
    .so-cond-prox { font-size: 11px; font-weight: 700; width: 42px; text-align: right; flex-shrink: 0; }
    /* Conviction status chip */
    .so-chip {
      font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 4px;
    }
    .so-chip.fresh { background: var(--green-dim); color: var(--green); }
    .so-chip.decaying { background: var(--amber-dim); color: var(--amber); }
    .so-chip.liberated { background: var(--red-dim); color: var(--red); }
    .so-empty { font-size: 11.5px; color: var(--muted); font-style: italic; padding: 4px 0; }
    @media(max-width:768px) { .so-panel { width: 100vw; } }
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
      <button class="refresh-btn" id="theme-btn" type="button"
              aria-label="Toggle light or dark theme">Light</button>
      <button class="refresh-btn" id="refresh-btn" onclick="refreshDashboard()">
        <span class="spinner"></span>
        <span class="btn-icon">&#x21bb;</span>
        Refresh
      </button>
    </div>
  </div>
</div>

<!-- ── System health strip ── -->
<div class="sys-health" id="sys-health"></div>

<!-- ── Headline: hero figure + equity curve ── -->
<section class="hero">
  <div class="hl">
    <div class="eyebrow">Net liquidation value</div>
    <div class="hv" id="hero-val">&mdash;</div>
    <div class="hd" id="hero-delta"></div>
    <dl class="hmeta" id="hero-meta"></dl>
  </div>
  <div class="hr">
    <div class="crow">
      <div class="seg" id="era-seg" role="group" aria-label="Measurement period"></div>
      <div class="hr-title" id="chart-title">Portfolio value</div>
    </div>
    <div class="chart-h280"><canvas id="portfolioChart"></canvas></div>
  </div>
</section>

<!-- ── Headline KPIs ── -->
<dl class="kpis" id="kpis"></dl>

<!-- ── Portfolio metrics: three stat panels ── -->
<div class="section-hdr">Portfolio metrics</div>
<div class="cols3">
  <div class="card np">
    <div class="ph">Performance <span class="phn" id="pm-src">&nbsp;</span></div>
    <div class="stats" id="pm-performance"></div>
  </div>
  <div class="card np">
    <div class="ph">Risk &amp; exposure <span class="phn">current</span></div>
    <div class="stats" id="pm-risk"></div>
  </div>
  <div class="card np">
    <div class="ph">Activity <span class="phn">book &amp; trades</span></div>
    <div class="stats" id="pm-activity"></div>
  </div>
</div>

<!-- ── Monthly performance ── -->
<div class="section-hdr">Monthly performance</div>
<div class="tilestrip" id="pm-monthly"></div>

<!-- ── Weekly Returns ── -->
<div class="card">
  <div class="section-hdr">Weekly returns vs benchmarks</div>
  <div class="chart-h280"><canvas id="weeklyChart"></canvas></div>
</div>

<!-- ── AI Value Chain ── -->
<div class="section-hdr">AI value chain</div>
<div class="tilestrip" id="chainKpi"></div>

<!-- ── Exposure: two independent axes over the same book ── -->
<div class="section-hdr">Exposure</div>
<div id="short-note"></div>
<div class="cols2">
  <div class="card np" id="sector-card">
    <div class="ph">Sector <span class="phn" id="sector-phn">GICS</span></div>
    <div id="sector-table"></div>
  </div>
  <div class="card np">
    <div class="ph">Size / style <span class="phn">screening bucket, not a sector</span></div>
    <div id="sizestyle-table"></div>
  </div>
</div>

<!-- ── Positions ── -->
<div class="card">
  <div class="section-hdr">Current Positions</div>
  <div id="positions-wrap"></div>
</div>

<!-- ── Closed Positions ── -->
<div class="card">
  <div class="section-hdr">Closed Positions</div>
  <div class="cl-filters" id="cl-filters">
    <div class="cl-filter-controls">
      <select class="cl-select" id="f-signal" aria-label="Filter by entry signal"></select>
      <select class="cl-select" id="f-exit" aria-label="Filter by exit condition"></select>
      <div class="cl-pills" id="f-period" role="group" aria-label="Filter by time period"></div>
    </div>
    <div class="cl-filter-state" id="cl-filter-state">
      <span class="cl-count" id="cl-count"></span>
      <button class="cl-clear" id="cl-clear">&times; Clear</button>
    </div>
  </div>
  <div class="closed-grid" id="closed-summary"></div>
  <div id="closed-list"></div>
</div>

<!-- ── Attribution: why we entered, how we exited ── -->
<div class="cols2">
  <div class="card np">
    <div class="ph">Signal attribution <span class="phn">realized P&amp;L by entry signal</span></div>
    <div id="signal-analytics" style="padding:4px 14px 12px"></div>
  </div>
  <div class="card np">
    <div class="ph">Exit distribution <span class="phn" id="exit-phn">how positions closed</span></div>
    <div id="exit-distribution"></div>
  </div>
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

<!-- ── Ticker detail slide-over ── -->
<div class="so-overlay" id="so-overlay">
  <aside class="so-panel" id="so-panel" role="dialog" aria-modal="true" aria-labelledby="so-sym">
    <div class="so-head">
      <button class="so-close" id="so-close" aria-label="Close">&times;</button>
      <div><span class="so-sym" id="so-sym"></span><span class="so-class" id="so-class"></span></div>
      <div class="so-name" id="so-name"></div>
      <div class="so-price-row">
        <span class="so-price" id="so-price"></span>
        <span><span class="so-pnl-big" id="so-pnl"></span><span class="so-pnl-pct" id="so-pnl-pct"></span></span>
      </div>
    </div>
    <div class="so-body" id="so-body"></div>
  </aside>
</div>

<script>
  // Chart.js cannot read CSS custom properties, so resolve the palette once
  // per render into an object the chart configs reference. Rebuilt whenever
  // the theme changes, otherwise charts keep the colours of the mode they
  // were first drawn in.
  const CH = {};
  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n, fallback) => (cs.getPropertyValue(n) || "").trim() || fallback;
    CH.cyan    = v("--cyan",    "#5b8def");
    CH.green   = v("--green",   "#3fa87a");
    CH.amber   = v("--amber",   "#c9973f");
    CH.red     = v("--red",     "#d1656b");
    CH.purple  = v("--purple",  "#8a7bd8");
    CH.dim     = v("--dim",     "#98a5be");
    CH.border  = v("--border",  "#242d47");
    CH.surface = v("--surface", "#121829");
    CH.text    = v("--text",    "#e7ecf6");
  }

  // ── Theme ──────────────────────────────────────────────────────────
  // Stamped on <html> so the data-theme rules win over the OS preference in
  // both directions. Persisted, because a viewer who picked light does not
  // want it reset on every cron-driven page refresh.
  (function initTheme() {
    const root = document.documentElement;
    const btn  = document.getElementById("theme-btn");
    const prefersLight = window.matchMedia
      && window.matchMedia("(prefers-color-scheme: light)").matches;
    let mode;
    try { mode = localStorage.getItem("kairos-theme"); } catch (e) { mode = null; }
    if (!mode) mode = prefersLight ? "light" : "dark";
    const apply = (m, redraw) => {
      root.setAttribute("data-theme", m);
      if (btn) btn.textContent = m === "dark" ? "Light" : "Dark";
      try { localStorage.setItem("kairos-theme", m); } catch (e) {}
      readPalette();
      if (redraw && window.Chart) {
        // Re-tint live charts in place rather than rebuilding them, so zoom
        // and any open tooltip survive the switch.
        Object.values(Chart.instances || {}).forEach(c => {
          try {
            const o = c.options || {};
            if (o.plugins && o.plugins.legend && o.plugins.legend.labels)
              o.plugins.legend.labels.color = CH.dim;
            if (o.plugins && o.plugins.tooltip) {
              o.plugins.tooltip.backgroundColor = CH.surface;
              o.plugins.tooltip.borderColor = CH.border;
            }
            Object.values(o.scales || {}).forEach(sc => {
              if (sc.ticks) sc.ticks.color = CH.dim;
              if (sc.grid)  sc.grid.color  = CH.border;
            });
            c.update("none");
          } catch (e) {}
        });
      }
    };
    apply(mode, false);
    if (btn) btn.addEventListener("click", () => {
      apply(root.getAttribute("data-theme") === "dark" ? "light" : "dark", true);
    });
  })();

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

  // ── System health strip ────────────────────────────────────────────
  (function renderSystemHealth() {
    const el = document.getElementById("sys-health");
    const H = DATA.system_health;
    if (!el) return;
    if (!H) { el.style.display = "none"; return; }
    const esc = (s) => String(s == null ? "" : s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const order = ["last_cycle", "next_cycle", "market", "commander", "ibkr", "regime"];
    let html = `<span class="sh-title">System</span>`;
    order.forEach(k => {
      const it = H[k];
      if (!it) return;
      const lvl = it.level || "info";
      html += `<span class="sh-item ${lvl}"><span class="dot"></span>`
            + `<span class="sh-label">${esc(it.label)}</span>`
            + `<span class="sh-val">${esc(it.value)}</span></span>`;
    });
    el.innerHTML = html;
  })();

  // ── Helpers ────────────────────────────────────────────────────────
  const fmtN = (n, d=2) => n == null ? "\u2014" :
    n.toLocaleString("en-US", {minimumFractionDigits:d, maximumFractionDigits:d});
  const fmtSign = (n, d=2) => n == null ? "\u2014" : (n >= 0 ? "+" : "") + fmtN(n, d);
  const fmtUSD  = (n) => n == null ? "\u2014" : "$" + fmtN(n, 0);
  const fmtPct  = (n, d=2) => n == null ? "\u2014" : fmtSign(n, d) + "%";
  const colClass = (n) => n == null ? "white" : n > 0 ? "green" : n < 0 ? "red" : "white";
  const usdSig  = (n) => n == null ? "\u2014" : (n >= 0 ? "+$" : "-$") + fmtN(Math.abs(n), 0);
  // Shared escaper. Several renderers previously each declared their own
  // function-scoped copy; anything outside those closures had none.
  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  // ── Era view ───────────────────────────────────────────────────────
  // Two measurement windows over the same book: since inception, and the
  // current-system era from the configured baseline forward. Previously the
  // era was force-applied and a paragraph of prose explained the ambiguity;
  // it is now a control, and every era-dependent figure re-renders on switch.
  const E     = M.era || null;
  const eraOn = !!E;
  let   view  = eraOn ? "era" : "all";          // default to the current system
  const inEra = () => view === "era" && eraOn;
  const eraTag = eraOn ? "since " + E.baseline_date : "";
  const eraVal = (k) => inEra() ? E[k] : M[k];

  // ── Hero ───────────────────────────────────────────────────────────
  function renderHero() {
    const PM = DATA.portfolio_metrics || {};
    document.getElementById("hero-val").textContent = fmtUSD(M.current_value);

    const ret = eraVal("total_return_pct");
    const usd = inEra() ? E.total_return_usd : M.total_return_usd;
    const cls = ret == null ? "flat" : ret > 0 ? "up" : ret < 0 ? "down" : "flat";
    const when = inEra()
      ? "current system \u00b7 " + eraTag.replace("since ", "since ")
      : "since inception \u00b7 " + (DATA.start_date || "\u2014");
    document.getElementById("hero-delta").innerHTML =
      `<span class="chip ${cls}">${fmtPct(ret)}</span>`
      + `<span>${usdSig(usd)} \u00b7 ${when}</span>`;

    const rows = [
      ["Invested",       PM.invested_pct != null ? fmtN(PM.invested_pct, 1) + "%" : "\u2014", ""],
      ["Cash",           fmtUSD(M.cash_value), ""],
      ["Open positions", PM.open_positions != null ? String(PM.open_positions) : "\u2014", ""],
      ["Unrealized",     usdSig(PM.unrealized_pnl), colClass(PM.unrealized_pnl)],
      ["Realized",       usdSig(inEra() ? E.realized_pnl : M.realized_pnl),
                         colClass(inEra() ? E.realized_pnl : M.realized_pnl)],
      ["Sharpe",         M.sharpe_ratio != null ? fmtN(M.sharpe_ratio, 2) : "\u2014", ""],
    ];
    document.getElementById("hero-meta").innerHTML = rows.map(
      ([k, v, c]) => `<div><dt>${k}</dt><dd class="${c === "white" ? "" : c}">${v}</dd></div>`
    ).join("");
  }

  // ── KPI strip ──────────────────────────────────────────────────────
  function renderKpis() {
    const tiles = [
      {
        label: "Total return",
        val:   () => fmtPct(eraVal("total_return_pct")),
        cls:   () => colClass(eraVal("total_return_pct")),
        foot:  () => usdSig(inEra() ? E.total_return_usd : M.total_return_usd)
                   + " \u00b7 " + (inEra() ? E.days : M.days_running) + "d",
      },
      {
        label: "Annualized",
        val:   () => fmtPct(eraVal("annualized_return")),
        cls:   () => colClass(eraVal("annualized_return")),
        foot:  () => {
          const d = inEra() ? E.days : M.days_running;
          return d + "-day basis" + (d < 90 ? " \u00b7 low confidence" : "");
        },
      },
      {
        label: "vs Advisor (" + fmtN(M.advisor_rate, 1) + "%)",
        val:   () => fmtPct(eraVal("vs_advisor")),
        cls:   () => colClass(eraVal("vs_advisor")),
        foot:  () => "annualized spread",
      },
      {
        label: "vs Target (" + fmtN(M.target_rate, 1) + "%)",
        val:   () => fmtPct(eraVal("vs_target")),
        cls:   () => colClass(eraVal("vs_target")),
        foot:  () => "annualized spread",
      },
      {
        label: "Win rate",
        val:   () => {
          const wr = inEra() ? E.win_rate : M.win_rate;
          return wr != null ? fmtN(wr, 1) + "%" : "\u2014";
        },
        cls:   () => {
          const wr = inEra() ? E.win_rate : M.win_rate;
          return wr == null ? "white" : wr >= 50 ? "green" : "amber";
        },
        foot:  () => {
          const n = inEra() ? E.closed_trades : M.closed_trades;
          return (n == null ? 0 : n) + " closed trades";
        },
      },
      {
        label: "Realized P&L",
        val:   () => usdSig(inEra() ? E.realized_pnl : M.realized_pnl),
        cls:   () => colClass(inEra() ? E.realized_pnl : M.realized_pnl),
        foot:  () => inEra() ? "in era" : "cumulative",
      },
      // Not era-scoped: the same figure under either window.
      {
        label: "Max drawdown",
        val:   () => M.max_drawdown != null ? "-" + fmtN(M.max_drawdown, 2) + "%" : "\u2014",
        cls:   () => M.max_drawdown > 10 ? "red" : M.max_drawdown > 5 ? "amber" : "white",
        foot:  () => "peak to trough",
      },
      {
        label: "Total trades",
        val:   () => String(M.total_trades),
        cls:   () => "white",
        foot:  () => M.filled_trades + " filled, "
                   + (M.total_trades - M.filled_trades) + " other",
      },
    ];
    document.getElementById("kpis").innerHTML = tiles.map(t => {
      const c = t.cls();
      return `<div class="k"><dt>${t.label}</dt>`
           + `<dd class="${c === "white" ? "" : c}">${t.val()}</dd>`
           + `<div class="kf">${t.foot()}</div></div>`;
    }).join("");
  }

  // ── Era switch ─────────────────────────────────────────────────────
  function renderSeg() {
    const el = document.getElementById("era-seg");
    if (!el) return;
    if (!eraOn) { el.innerHTML = ""; return; }   // :empty hides it
    el.innerHTML =
        `<button type="button" data-view="all" aria-pressed="${view === "all"}">`
      + `Since inception</button>`
      + `<button type="button" data-view="era" aria-pressed="${view === "era"}">`
      + `Current system</button>`;
    el.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      if (view === b.dataset.view) return;
      view = b.dataset.view;
      renderAllEraViews();
    }));
  }

  function renderAllEraViews() {
    renderSeg();
    renderHero();
    renderKpis();
    if (typeof applyChartWindow === "function") applyChartWindow();
    const t = document.getElementById("chart-title");
    if (t) {
      t.textContent = inEra()
        ? "Portfolio value \u00b7 " + eraTag
        : "Portfolio value \u00b7 since inception";
    }
  }

  // ── Portfolio metrics: three stat panels ───────────────────────────
  // Replaces 15 equal-weight tiles plus a separate 3-card risk row. Those
  // carried real information at one visual rank, so scanning for a specific
  // figure meant reading all eighteen. Grouped into performance / risk /
  // activity as label-value lists, which is denser and actually scannable.
  (function renderPortfolioMetrics() {
    const P = DATA.portfolio_metrics || {};
    const money    = (n) => n == null ? "—" : "$" + fmtN(n, 2);
    const moneySig = (n) => n == null ? "—" : (n >= 0 ? "+$" : "-$") + fmtN(Math.abs(n), 2);
    const pctSig   = (n) => n == null ? "—" : fmtSign(n, 2) + "%";
    const cls      = (n) => n == null ? "" : (n >= 0 ? "pnl-pos" : "pnl-neg");
    const row = (label, valHtml, note) =>
      '<div class="st"><dt>' + label + (note ? '<span class="stnote">' + note + '</span>' : "")
      + '</dt><dd>' + valHtml + '</dd></div>';
    const wrap = (v, c) => c ? '<span class="' + c + '">' + v + '</span>' : v;

    const src = P.have_live ? "live IBKR" : (P.as_of ? "snapshot " + P.as_of : "—");
    const srcEl = document.getElementById("pm-src");
    if (srcEl) srcEl.textContent = src;

    const trend = (t, fallback) => t
      ? wrap(moneySig(t.usd), cls(t.usd))
        + '<span class="stsub">' + pctSig(t.pct)
        + (t.since ? " · since " + t.since : "") + '</span>'
      : '—<span class="stsub">' + (fallback || "needs ≥2 days") + '</span>';

    document.getElementById("pm-performance").innerHTML = [
      row("Net liquidation value", money(M.current_value)),
      row("Starting capital",      money(M.start_value)),
      row("Total return",          wrap(pctSig(M.total_return_pct), cls(M.total_return_pct))),
      row("Realized P&L",          wrap(moneySig(P.realized_pnl_cum), cls(P.realized_pnl_cum))),
      row("Unrealized P&L",        wrap(moneySig(P.unrealized_pnl), cls(P.unrealized_pnl))),
      row("Unrealized return",     wrap(pctSig(P.unrealized_return_pct), cls(P.unrealized_return_pct)),
          P.unrealized_return_pct == null ? "needs live marks" : ""),
      row("Daily P&L",             trend(P.daily)),
      row("Weekly P&L",            trend(P.weekly, "no baseline ≥5 days back")),
    ].join("");

    document.getElementById("pm-risk").innerHTML = [
      row("Sharpe ratio",        M.sharpe_ratio != null ? fmtN(M.sharpe_ratio, 2) : "—",
          M.sharpe_ratio == null ? "needs 10+ snapshots" : ""),
      row("Max drawdown",        wrap("-" + fmtN(M.max_drawdown, 2) + "%", "pnl-neg")),
      row("Drawdown from peak",  wrap(pctSig(P.drawdown_pct), cls(P.drawdown_pct))),
      row("30-day return",       wrap(pctSig(P.return_30d_pct), cls(P.return_30d_pct))),
      row("Invested",            P.invested_pct != null ? fmtN(P.invested_pct, 2) + "%" : "—"),
      row("Cash",                money(M.cash_value)),
      row("Largest position",    P.largest_pct != null ? fmtN(P.largest_pct, 2) + "%" : "—"),
      row("Top-5 concentration", P.top5_pct != null ? fmtN(P.top5_pct, 2) + "%" : "—"),
      row("Largest single loss", M.largest_loss < 0
            ? wrap("-$" + fmtN(Math.abs(M.largest_loss), 2), "pnl-neg") : "—"),
    ].join("");

    // Breadth needs live marks; with IBKR offline it is unknowable, and
    // "0 up / 0 down" would read as real data.
    const breadth = (P.positions_in_profit == null || P.positions_in_loss == null)
      ? "—"
      : '<span class="pnl-pos">' + P.positions_in_profit + '</span> / '
        + '<span class="pnl-neg">' + P.positions_in_loss + '</span>';
    const bw = (o) => o ? esc(o.ticker) + " " + wrap(pctSig(o.pct), cls(o.pct)) : "—";

    document.getElementById("pm-activity").innerHTML = [
      row("Open positions",  P.open_positions != null ? String(P.open_positions) : "—"),
      row("In profit / loss", breadth,
          P.positions_in_profit == null ? "needs live marks" : ""),
      row("Best open",       bw(P.best)),
      row("Worst open",      bw(P.worst)),
      row("Closed trades",   String(M.closed_trades || 0)),
      row("Win rate",        M.win_rate != null ? fmtN(M.win_rate, 1) + "%" : "—",
          "since inception"),
      row("Total trades",    String(M.total_trades),
          M.filled_trades + " filled"),
      row("Snapshot history", String(P.snapshot_days || 0),
          "day" + ((P.snapshot_days || 0) !== 1 ? "s" : "")),
    ].join("");

    // ── Monthly performance ──
    // A tile strip rather than a table: three rows in a full-width table wastes
    // the row, and tiles keep the section one card tall as months accumulate.
    (function renderMonthly() {
      const host = document.getElementById("pm-monthly");
      if (!host) return;
      const rows = (M.monthly_performance) || [];
      if (!rows.length) {
        host.innerHTML = '<div class="tile"><div class="no-data">'
          + 'No closed trades in the ML ledger yet</div></div>';
        return;
      }
      const max = Math.max.apply(null, rows.map(function (r) {
        return Math.abs(r.total_usd || 0); })) || 1;
      host.innerHTML = rows.map(function (r) {
        const pos = (r.total_usd || 0) >= 0;
        return '<div class="tile"><div class="tlabel">' + esc(r.month) + '</div>'
          + '<div class="tval ' + (pos ? "pnl-pos" : "pnl-neg") + '">'
          + usdSig(r.total_usd) + '</div>'
          + '<div class="tbar"><i class="' + (pos ? "green" : "red") + '" style="width:'
          + (Math.abs(r.total_usd || 0) / max * 100).toFixed(0) + '%"></i></div>'
          + '<div class="tmeta"><span>' + r.n + ' trades</span>'
          + '<span>' + fmtN(r.win_rate, 0) + '% win</span>'
          + '<span class="' + ((r.avg_pct || 0) >= 0 ? "pnl-pos" : "pnl-neg") + '">'
          + fmtSign(r.avg_pct, 2) + '%</span></div></div>';
      }).join("");
    })();
  })();

  // ── Realized-trade aggregation (shared) ────────────────────────────
  // Pure function reused by the Closed Positions summary band (over the
  // filtered set) and the Signal Performance Analytics section (per signal
  // group). Includes avg_days (mean hold time) for the analytics table.
  function computeSummary(trades) {
    let winners = 0, losers = 0, breakeven = 0;
    let totalPnl = 0, totalCost = 0;
    const winPcts = [], lossPcts = [], days = [];
    trades.forEach(t => {
      const pnl = t.realized_pnl_usd, cost = (t.entry_price || 0) * (t.quantity || 0);
      const pct = t.realized_pnl_pct;
      if (t.days_held != null) days.push(t.days_held);
      if (pnl == null) return;
      totalPnl += pnl; totalCost += cost;
      if (pnl > 0) { winners++; if (pct != null) winPcts.push(pct); }
      else if (pnl < 0) { losers++; if (pct != null) lossPcts.push(pct); }
      else breakeven++;
    });
    const n = trades.length;
    const mean = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : null;
    return {
      n, winners, losers, breakeven,
      total_pnl_usd: n ? totalPnl : null,
      total_pnl_pct: totalCost ? totalPnl / totalCost * 100 : null,
      win_rate: n ? winners / n * 100 : null,
      avg_win_pct: mean(winPcts),
      avg_loss_pct: mean(lossPcts),
      avg_days: mean(days),
      win_loss_ratio: losers ? winners / losers : null,
    };
  }

  // ── Closed positions: filters + live-recomputing summary + list ────
  (function initClosedPositions() {
    const ALL = DATA.closed_trades || [];
    const summaryEl = document.getElementById("closed-summary");
    const listEl = document.getElementById("closed-list");
    const filtersEl = document.getElementById("cl-filters");
    if (!summaryEl || !listEl) return;

    // No closed trades at all → original single-line empty state, hide filters.
    if (!ALL.length) {
      if (filtersEl) filtersEl.style.display = "none";
      summaryEl.className = "";
      summaryEl.innerHTML = `<div class="closed-empty">No closed positions yet.</div>`;
      listEl.innerHTML = "";
      return;
    }

    const BATCH = 12;
    const NO_SIGNAL = "__none__";
    const filters = { signal: "", exit: "", period: "all" };

    const moneySig = (n) => n == null ? "—" : (n >= 0 ? "+$" : "-$") + fmtN(Math.abs(n), 2);
    const money    = (n) => n == null ? "—" : (n >= 0 ? "+$" : "−$") + Math.abs(n).toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2});
    const pctSig   = (n) => n == null ? "—" : fmtSign(n, 2) + "%";
    const signCls  = (n) => n == null ? "white" : (n >= 0 ? "green" : "red");
    const day      = (s) => s ? String(s).slice(0, 10) : "—";
    const esc      = (s) => String(s == null ? "" : s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

    function renderSummary(C, filtered) {
      const card = (accent, label, valHtml, valCls, sub) =>
        `<div class="mcard c-${accent}">` +
        `<div class="mlabel">${label}</div>` +
        `<div class="mval ${valCls || "white"}">${valHtml}</div>` +
        `<div class="msub">${sub}</div></div>`;
      const scope = filtered ? "filtered" : "all-time";
      const cards = [
        card(signCls(C.total_pnl_usd), "Realized P&L",
             moneySig(C.total_pnl_usd), signCls(C.total_pnl_usd),
             `<span class="${C.total_pnl_pct >= 0 ? "pnl-pos" : "pnl-neg"}">${pctSig(C.total_pnl_pct)}</span> · holdings-sum, may differ from top`),
        card("cyan", "Win Rate",
             C.win_rate != null ? fmtN(C.win_rate, 1) + "%" : "—", "cyan",
             `${C.winners} of ${C.n} up`),
        card("green", "Avg Win", pctSig(C.avg_win_pct), "green", "Winners, avg return"),
        card("red", "Avg Loss", pctSig(C.avg_loss_pct), "red", "Losers, avg return"),
        card("white", "Win / Loss", `${C.winners} / ${C.losers}`, "white",
             C.win_loss_ratio != null ? `ratio ${fmtN(C.win_loss_ratio, 2)}`
               : (C.breakeven ? `${C.breakeven} breakeven` : "no losers")),
        card("cyan", "Closed Trades", String(C.n), "cyan",
             C.breakeven ? `${C.breakeven} breakeven · ${scope}` : scope),
      ];
      summaryEl.className = "closed-grid";
      summaryEl.innerHTML = cards.join("");
    }

    // ── Trade list with load-more pagination ────────────────────────
    function rowHtml(t) {
      const pnlCls = t.realized_pnl_usd == null ? "" : (t.realized_pnl_usd >= 0 ? "pnl-pos" : "pnl-neg");
      const pctCls = t.realized_pnl_pct == null ? "" : (t.realized_pnl_pct >= 0 ? "pnl-pos" : "pnl-neg");
      const days = t.days_held != null ? t.days_held + "d" : "—";
      const chip = `<span class="exit-chip ${t.exit_color || "muted"}"><span class="dot"></span>${esc(t.exit_type || "Not recorded")}</span>`;
      return `<tr onclick="openPanel('${t.id}')">
        <td class="cl-tkr">${esc(t.ticker)}</td>
        <td class="cl-date">${day(t.entry_date)}</td>
        <td class="cl-date">${day(t.sold_date)}</td>
        <td class="cl-days">${days}</td>
        <td class="${pnlCls}">${money(t.realized_pnl_usd)}</td>
        <td class="${pctCls}">${pctSig(t.realized_pnl_pct)}</td>
        <td>${chip}</td>
      </tr>`;
    }

    function renderList(trades) {
      if (!trades.length) {
        listEl.innerHTML = `<div class="no-data">No closed trades match these filters</div>`;
        return;
      }
      listEl.innerHTML =
        '<div class="tbl-wrap"><table><thead><tr>' +
        '<th>Ticker</th><th>Entry</th><th>Exit</th><th>Days</th>' +
        '<th>Realized P&amp;L</th><th>P&L %</th><th>Exit Reason</th>' +
        '</tr></thead><tbody id="cl-body"></tbody></table></div>' +
        '<div class="cl-more-wrap" id="cl-more-wrap"></div>';
      const body = document.getElementById("cl-body");
      const moreWrap = document.getElementById("cl-more-wrap");
      let shown = 0;
      function renderMore() {
        const next = trades.slice(shown, shown + BATCH);
        body.insertAdjacentHTML("beforeend", next.map(rowHtml).join(""));
        shown += next.length;
        const remaining = trades.length - shown;
        moreWrap.innerHTML = remaining > 0
          ? `<button class="cl-more-btn" id="cl-more-btn">Load more (${remaining} remaining)</button>`
          : "";
        const btn = document.getElementById("cl-more-btn");
        if (btn) btn.addEventListener("click", renderMore);
      }
      renderMore();
    }

    // ── Filter matching ─────────────────────────────────────────────
    function inPeriod(soldDate, period) {
      if (period === "all") return true;
      const s = String(soldDate || "").slice(0, 10).split("-");
      if (s.length !== 3) return false;
      const d = new Date(+s[0], +s[1] - 1, +s[2]);
      if (isNaN(d)) return false;
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      if (period === "month")
        return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth();
      if (period === "week") {
        const dow = (now.getDay() + 6) % 7; // Mon=0 … Sun=6
        const weekStart = new Date(today); weekStart.setDate(today.getDate() - dow);
        return d >= weekStart;
      }
      if (period === "30d") {
        const cutoff = new Date(today); cutoff.setDate(today.getDate() - 30);
        return d >= cutoff;
      }
      return true;
    }

    function matches(t) {
      if (filters.signal) {
        const sigs = t.entry_signals || [];
        if (filters.signal === NO_SIGNAL) { if (sigs.length) return false; }
        else if (!sigs.includes(filters.signal)) return false;
      }
      if (filters.exit && t.exit_type !== filters.exit) return false;
      if (!inPeriod(t.sold_date, filters.period)) return false;
      return true;
    }

    // ── Build filter controls from the data actually present ────────
    const signalSel = document.getElementById("f-signal");
    const exitSel   = document.getElementById("f-exit");
    const periodEl  = document.getElementById("f-period");
    const clearBtn  = document.getElementById("cl-clear");
    const countEl   = document.getElementById("cl-count");
    const stateEl   = document.getElementById("cl-filter-state");

    (function populate() {
      const sigSet = new Set(), exitSet = new Set();
      let hasNoSignal = false;
      ALL.forEach(t => {
        (t.entry_signals || []).forEach(s => sigSet.add(s));
        if (!(t.entry_signals || []).length) hasNoSignal = true;
        exitSet.add(t.exit_type || "Not recorded");
      });
      const sigs = [...sigSet].sort();
      // Exit types sorted, but keep "Not recorded" last.
      const exits = [...exitSet].filter(e => e !== "Not recorded").sort();
      if (exitSet.has("Not recorded")) exits.push("Not recorded");

      let sHtml = `<option value="">All signals</option>`;
      sigs.forEach(s => sHtml += `<option value="${esc(s)}">${esc(s)}</option>`);
      if (hasNoSignal) sHtml += `<option value="${NO_SIGNAL}">No signal</option>`;
      signalSel.innerHTML = sHtml;

      let eHtml = `<option value="">All exits</option>`;
      exits.forEach(e => eHtml += `<option value="${esc(e)}">${esc(e)}</option>`);
      exitSel.innerHTML = eHtml;

      const periods = [["all","All time"],["week","This week"],["month","This month"],["30d","Last 30 days"]];
      periodEl.innerHTML = periods.map(([v,l]) =>
        `<button class="cl-pill${v==="all"?" active":""}" data-period="${v}">${l}</button>`).join("");
    })();

    // ── Apply + wire ────────────────────────────────────────────────
    function apply() {
      const filtered = ALL.filter(matches);
      const isFiltered = !!(filters.signal || filters.exit || filters.period !== "all");
      renderSummary(computeSummary(filtered), isFiltered);
      renderList(filtered);

      signalSel.classList.toggle("active", !!filters.signal);
      exitSel.classList.toggle("active", !!filters.exit);
      periodEl.querySelectorAll(".cl-pill").forEach(p =>
        p.classList.toggle("active", p.dataset.period === filters.period));

      if (isFiltered) {
        countEl.textContent = `Showing ${filtered.length} of ${ALL.length}`;
        stateEl.classList.remove("hidden");
      } else {
        stateEl.classList.add("hidden");
      }
    }

    signalSel.addEventListener("change", () => { filters.signal = signalSel.value; apply(); });
    exitSel.addEventListener("change", () => { filters.exit = exitSel.value; apply(); });
    periodEl.addEventListener("click", (e) => {
      const btn = e.target.closest(".cl-pill");
      if (!btn) return;
      filters.period = btn.dataset.period;
      apply();
    });
    clearBtn.addEventListener("click", () => {
      filters.signal = ""; filters.exit = ""; filters.period = "all";
      signalSel.value = ""; exitSel.value = "";
      apply();
    });

    stateEl.classList.add("hidden");
    apply();
  })();

  // ── Signal attribution ─────────────────────────────────────────────
  // Realized P&L by entry signal is a magnitude above/below a zero baseline —
  // the diverging case. A diverging bar reads the sign at a glance in a way a
  // column of signed numbers does not, and it breaks up a page that is
  // otherwise all tables. Green/red here is polarity, not series identity.
  (function renderSignalAttribution() {
    const wrap = document.getElementById("signal-analytics");
    if (!wrap) return;
    const ALL = DATA.closed_trades || [];
    if (!ALL.length) {
      wrap.innerHTML = '<div class="no-data">No closed trades to analyze yet</div>';
      return;
    }

    const NO_SIGNAL = "(unattributed)";
    const groups = {};
    ALL.forEach(function (t) {
      const sigs = (t.entry_signals && t.entry_signals.length)
        ? t.entry_signals : [NO_SIGNAL];
      sigs.forEach(function (sig) {
        (groups[sig] = groups[sig] || []).push(t);
      });
    });

    const stats = Object.keys(groups).map(function (sig) {
      const ts = groups[sig];
      let pnl = 0, wins = 0, sumPct = 0, holdN = 0, holdSum = 0;
      ts.forEach(function (t) {
        pnl += t.realized_pnl_usd || 0;
        if ((t.realized_pnl_pct || 0) > 0) wins++;
        sumPct += t.realized_pnl_pct || 0;
        if (t.days_held != null) { holdSum += t.days_held; holdN++; }
      });
      return {
        name: sig, n: ts.length, pnl: pnl,
        win: wins / ts.length * 100,
        avg: sumPct / ts.length,
        hold: holdN ? holdSum / holdN : null,
      };
    }).sort(function (a, b) { return b.pnl - a.pnl; });

    const max = Math.max.apply(null, stats.map(function (r) {
      return Math.abs(r.pnl); })) || 1;

    // Column headers sit directly above the figures they name; a legend
    // stranded in the panel header leaves four unlabelled numbers per row.
    let html = '<div class="dv"><div class="dvrow dvhead"><div></div>'
      + '<div class="dvmeta"><span>Trades</span><span>Win</span>'
      + '<span>Avg</span><span>Hold</span></div>'
      + '<div class="dvtrack"></div><div class="dvval">Realized</div></div>';

    html += stats.map(function (r) {
      const side = r.pnl >= 0 ? "p" : "n";
      const w = 50 * Math.abs(r.pnl) / max;
      return '<div class="dvrow">'
        + '<div class="dvlabel">' + esc(r.name) + '</div>'
        + '<div class="dvmeta"><span>' + r.n + '</span>'
        + '<span>' + fmtN(r.win, 0) + '%</span>'
        + '<span class="' + (r.avg >= 0 ? "pnl-pos" : "pnl-neg") + '">'
        + (r.avg >= 0 ? "+" : "") + fmtN(r.avg, 1) + '%</span>'
        + '<span>' + (r.hold == null ? "—" : fmtN(r.hold, 0) + "d") + '</span></div>'
        + '<div class="dvtrack"><div class="dvaxis"></div>'
        + '<div class="dvbar ' + side + '" style="width:' + w.toFixed(1) + '%"></div></div>'
        + '<div class="dvval ' + (r.pnl >= 0 ? "pnl-pos" : "pnl-neg") + '">'
        + usdSig(r.pnl) + '</div></div>';
    }).join("");

    html += '</div>';

    // The chart is persuasive and the underlying attribution is not clean —
    // say so next to it, with the real recency so it does not read as a
    // closed historical gap.
    const unattr = ALL.filter(function (t) {
      return !(t.entry_signals && t.entry_signals.length); });
    let last = null;
    unattr.forEach(function (t) {
      const d = String(t.entry_date || "").slice(0, 10);
      if (d && (!last || d > last)) last = d;
    });
    if (unattr.length) {
      html += '<div class="caveat">Attribution is reconstructed partly from '
        + 'rationale text, so confluence trades count under every signal that '
        + 'fired — the bars sum to more than the book. ' + unattr.length
        + ' trades carry no entry signal at all'
        + (last ? ', the most recent entered ' + esc(last) : "")
        + ', so this is not a closed historical gap. Directional only.</div>';
    }

    wrap.innerHTML = html;
  })();

  // ── Exit distribution ──────────────────────────────────────────────
  // Share is computed over trades that carry a recorded reason, not over every
  // closed trade. Exit-reason capture did not exist for the earliest trades;
  // "Not recorded" is an absence of data, not a way of exiting, and leaving
  // those trades in the denominator understated every real exit type by ~1.25x
  // (Trailing Stop read 23.5% where it is actually 37.3%).
  (function renderExitDistribution() {
    const host = document.getElementById("exit-distribution");
    if (!host) return;
    const ALL = DATA.closed_trades || [];
    const cov = DATA.exit_coverage || {};
    const pool = ALL.filter(function (t) { return t.exit_reason; });

    if (!pool.length) {
      host.innerHTML = '<div class="no-data">No exit reasons recorded yet</div>';
      return;
    }

    const groups = {};
    pool.forEach(function (t) {
      const k = t.exit_type || "Unlabelled rationale";
      (groups[k] = groups[k] || []).push(t);
    });
    const rows = Object.keys(groups).map(function (k) {
      const ts = groups[k];
      let pnl = 0, sumPct = 0;
      ts.forEach(function (t) {
        pnl += t.realized_pnl_usd || 0;
        sumPct += t.realized_pnl_pct || 0;
      });
      return { name: k, n: ts.length, pnl: pnl, avg: sumPct / ts.length };
    }).sort(function (a, b) { return b.n - a.n; });

    const maxN = rows[0].n || 1;
    let html = '<table class="exp exitbl"><thead><tr><th>Exit type</th>'
      + '<th class="r">Trades</th><th class="r">Share</th>'
      + '<th class="r">Avg return</th><th class="r">Realized</th></tr></thead><tbody>';
    html += rows.map(function (r) {
      return '<tr><td>' + esc(r.name) + '</td>'
        + '<td class="r wcell"><span>' + r.n + '</span>'
        + '<span class="wbar"><i style="width:' + (r.n / maxN * 100).toFixed(0) + '%"></i></span></td>'
        + '<td class="p">' + fmtN(r.n / pool.length * 100, 1) + '%</td>'
        + '<td class="r ' + (r.avg >= 0 ? "pnl-pos" : "pnl-neg") + '">'
        + (r.avg >= 0 ? "+" : "") + fmtN(r.avg, 2) + '%</td>'
        + '<td class="r ' + (r.pnl >= 0 ? "pnl-pos" : "pnl-neg") + '">'
        + usdSig(r.pnl) + '</td></tr>';
    }).join("");
    html += '</tbody></table>';

    const missing = cov.unrecorded != null ? cov.unrecorded : (ALL.length - pool.length);
    if (missing > 0) {
      html += '<div class="caveat neutral">Share is of the ' + pool.length
        + ' trades that carry a recorded exit reason ('
        + fmtN(pool.length / ALL.length * 100, 0) + '% of ' + ALL.length
        + (cov.capture_from ? '; capture began ' + esc(cov.capture_from) : "")
        + '). The other ' + missing + ' closed before the instrumentation '
        + 'existed and cannot carry a type — counting them would understate '
        + 'every real exit reason.</div>';
    }

    host.innerHTML = html;
    const phn = document.getElementById("exit-phn");
    if (phn) phn.textContent = pool.length + " of " + ALL.length + " closed trades";
  })();

  // ── Chart defaults ─────────────────────────────────────────────────
  Chart.defaults.color = CH.dim;
  Chart.defaults.borderColor = CH.border;
  Chart.defaults.font.family = "SF Mono, Consolas, Monaco, Courier New, monospace";
  Chart.defaults.font.size = 11;

  const gridOpts = {
    color: CH.border,
    drawBorder: false,
  };
  const tickOpts = { color: CH.dim };

  // ── Portfolio value chart ──────────────────────────────────────────
  const portfolioChart = new Chart(document.getElementById("portfolioChart"), {
    type: "line",
    data: {
      labels: S.dates,
      datasets: [
        {
          label: "Kairos Actual",
          data: S.actual,
          borderColor: CH.cyan,
          backgroundColor: "rgba(91,141,239,0.10)",
          borderWidth: 2.5,
          pointRadius: S.dates.length > 30 ? 0 : 4,
          pointBackgroundColor: CH.cyan,
          tension: 0.3,
          fill: true,
          order: 1,
        },
        {
          label: "Advisor 14.4%",
          data: S.advisor,
          borderColor: CH.amber,
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
          borderColor: CH.green,
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
          labels: { color: CH.dim, boxWidth: 20, padding: 16 },
        },
        tooltip: {
          backgroundColor: CH.surface,
          borderColor: CH.border,
          borderWidth: 1,
          titleColor: CH.text,
          bodyColor: CH.dim,
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
        // 72 daily labels rotated 45deg is an unreadable picket fence. Cap the
        // count and keep them horizontal; the tooltip carries the exact date.
        x: {
          grid: gridOpts,
          ticks: { ...tickOpts, maxTicksLimit: 8, maxRotation: 0, autoSkip: true },
        },
        y: {
          grid: gridOpts,
          ticks: {
            ...tickOpts,
            maxTicksLimit: 6,
            callback: v => "$" + (v >= 1000 ? (v/1000).toFixed(0) + "k" : v),
          },
        },
      },
    },
  });


  // Clip the equity curve to the selected measurement window. Slicing the
  // arrays (rather than drawing a marker on the full series) is what makes the
  // era view actually re-scale — otherwise 33 days of data sit compressed
  // against 77 days of axis and the shape is unreadable.
  function applyChartWindow() {
    if (!portfolioChart) return;
    let i0 = 0;
    if (inEra() && E && E.baseline_date) {
      const found = S.dates.findIndex(d => d >= E.baseline_date);
      if (found > 0) i0 = found;
    }
    portfolioChart.data.labels = S.dates.slice(i0);
    const keys = ["actual", "advisor", "target"];
    portfolioChart.data.datasets.forEach((ds, n) => {
      const src = S[keys[n]] || [];
      ds.data = src.slice(i0);
      ds.pointRadius = (S.dates.length - i0) > 30 ? 0 : 4;
    });
    portfolioChart.update("none");
  }

  // First paint. Deferred to here because the hero and KPI renderers drive the
  // chart window, and the chart must exist before they run.
  renderAllEraViews();
  // Chart.js lays out its category scale on its own first frame. A window
  // applied before that lands is measured against the pre-layout scale, so the
  // series gets compressed into the left ~40% of the plot while the axis spans
  // the full width. Re-apply once laid out; text has already painted, so this
  // costs nothing visible.
  requestAnimationFrame(() => applyChartWindow());

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
            borderColor: CH.amber,
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
            borderColor: CH.green,
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
          legend: { position: "top", labels: { color: CH.dim, boxWidth: 16, padding: 12 } },
          tooltip: {
            backgroundColor: CH.surface, borderColor: CH.border, borderWidth: 1,
            titleColor: CH.text, bodyColor: CH.dim,
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

  // ── AI value chain ─────────────────────────────────────────────────
  // Non-Chain is ~92% of the book, so a stacked bar or donut renders the three
  // tiers as invisible slivers. The readable content is the in-chain total and
  // how it splits — a magnitude comparison, so one hue and no categorical
  // palette. Tier bars scale against the largest tier, not NLV, to stay legible.
  (function renderChain() {
    const chain = DATA.chain_tier_breakdown || {};
    const all   = chain.buckets || [];
    const wrap  = document.getElementById("chainKpi");
    if (!wrap) return;

    const tiers = all.filter(b => b.label !== "Non-Chain");
    const total = all.reduce((a, b) => a + (b.value || 0), 0);
    if (!tiers.length || total <= 0) {
      wrap.innerHTML = '<div class="tile"><div class="no-data">No chain positions yet</div></div>';
      return;
    }
    const inChain = tiers.reduce((a, b) => a + (b.value || 0), 0);
    const maxTier = Math.max.apply(null, tiers.map(b => b.value || 0)) || 1;
    const pctOf = (v, base) => base > 0 ? (v / base * 100) : 0;

    let html =
        '<div class="tile lead"><div class="tlabel">In-chain exposure</div>'
      + '<div class="tval">' + fmtN(pctOf(inChain, total), 1) + '%</div>'
      + '<div class="tbar"><i style="width:'
      + Math.min(100, pctOf(inChain, total)).toFixed(0) + '%"></i></div>'
      + '<div class="tmeta"><span>' + fmtUSD(inChain) + '</span>'
      + '<span>' + fmtUSD(total - inChain) + ' non-chain</span></div></div>';

    html += tiers.map(function (b) {
      const v = b.value || 0;
      return '<div class="tile"><div class="tlabel">' + esc(b.label) + '</div>'
           + '<div class="tval">' + fmtN(pctOf(v, total), 1) + '%</div>'
           + '<div class="tbar"><i style="width:' + (v / maxTier * 100).toFixed(0) + '%"></i></div>'
           + '<div class="tmeta"><span>' + fmtUSD(v) + '</span>'
           + '<span>' + fmtN(pctOf(v, inChain), 0) + '% of chain</span></div></div>';
    }).join("");

    wrap.innerHTML = html;
  })();

  // ── Exposure: sector (GICS) and size/style, as two separate axes ───
  // These measure different things. lookup_sector() returns the universe
  // screening bucket, which scatters one real sector across several cohorts;
  // sector_breakdown now resolves through the security master instead. Both
  // are shown because size/style is a legitimate view — just not a sector one.
  //
  // A table rather than a chart is deliberate: past ~7 classes that all carry
  // meaning, adjacent colours blur, and 11 sectors would need a categorical
  // palette beyond the safe ceiling. Magnitude bars in a table read better.
  (function renderExposure() {
    const WARN_PCT = 25;
    const base = M.equity_value > 0 ? M.equity_value : M.current_value;

    function table(elId, data, flagLimit) {
      const host = document.getElementById(elId);
      if (!host) return;
      const rows = Object.entries(data || {})
        .filter(function (e) { return e[1] > 0; })
        .sort(function (a, b) { return b[1] - a[1]; });
      if (!rows.length || base <= 0) {
        host.innerHTML = '<div class="no-data">No equity positions to display</div>';
        return;
      }
      const max = rows[0][1];
      const body = rows.map(function (e) {
        const k = e[0], v = e[1];
        const pct = v / base * 100;
        const isOver = flagLimit && pct >= WARN_PCT;
        return '<tr class="' + (isOver ? "over" : "") + '">'
             + '<td>' + esc(k)
             + (isOver ? '<span class="overtag">over watermark</span>' : "") + '</td>'
             + '<td class="v">' + fmtUSD(v) + '</td>'
             + '<td class="p">' + fmtN(pct, 1) + '%</td>'
             + '<td class="expbar"><span><i style="width:'
             + (v / max * 100).toFixed(0) + '%"></i></span></td></tr>';
      }).join("");
      host.innerHTML = '<table class="exp"><tbody>' + body + '</tbody></table>';
    }

    table("sector-table", DATA.sector_breakdown, true);
    table("sizestyle-table", DATA.size_style_breakdown, false);

    const phn = document.getElementById("sector-phn");
    if (phn) phn.textContent = "GICS · watermark " + WARN_PCT + "%";

    // Shorts net into totals correctly but cannot be a slice of a
    // part-to-whole view, so say so rather than silently dropping them.
    const shorts = DATA.short_positions || [];
    const note = document.getElementById("short-note");
    if (!note) return;
    note.innerHTML = shorts.length
      ? '<div class="warnbar">' + shorts.length + " short position"
        + (shorts.length === 1 ? "" : "s") + " — "
        + shorts.map(function (p) {
            return "<b>" + esc(p.symbol) + "</b> " + fmtN(p.quantity, 0)
                 + " sh " + fmtUSD(p.market_value);
          }).join(", ")
        + " · netted into totals, excluded from the breakdowns above</div>"
      : "";
  })();

  // ── Positions table ────────────────────────────────────────────────
  // Sorted A-Z: the book is a reference list you scan for a known name, not a
  // ranking. Weight bars still scale off the largest holding, so relative size
  // survives the reorder. A sticky totals row closes the table — without it
  // the reader has to trust that 56 rows add up to the headline.
  const posWrap = document.getElementById("positions-wrap");
  if (positions.length === 0) {
    posWrap.innerHTML = '<div class="no-data">No open positions (or IBKR offline)</div>';
  } else {
    const nlvBase = M.current_value > 0 ? M.current_value : 1;
    const rows = positions.slice().sort(function (a, b) {
      return String(a.symbol).localeCompare(String(b.symbol));
    });
    const weightOf = (p) => (p.market_value || 0) / nlvBase * 100;
    const maxW = Math.max.apply(null, rows.map(weightOf)) || 1;

    let totMV = 0, totPnl = 0, totBasis = 0;
    rows.forEach(function (p) {
      totMV    += p.market_value || 0;
      totPnl   += p.unrealized_pnl || 0;
      totBasis += (p.avg_cost || 0) * (p.quantity || 0);
    });
    const totPct = totBasis > 0 ? (totPnl / totBasis * 100) : null;

    const pnlCell = (v, isPct) => {
      if (v == null) return '<span class="dimval">—</span>';
      const c = v >= 0 ? "pnl-pos" : "pnl-neg";
      const t = isPct ? (v >= 0 ? "+" : "") + fmtN(v, 2) + "%" : usdSig(v);
      return '<span class="' + c + '">' + t + '</span>';
    };

    let html = '<div class="tbl-wrap"><table class="postbl"><thead><tr>'
      + '<th>Ticker</th><th>Name</th><th>Sector</th><th class="r">Qty</th>'
      + '<th class="r">Avg cost</th><th class="r">Market value</th>'
      + '<th class="r">Unrealized</th><th class="r">Return</th>'
      + '<th class="r">Weight</th></tr></thead><tbody>';

    rows.forEach(function (p) {
      const upnl = p.unrealized_pnl;
      const basis = (p.avg_cost || 0) * (p.quantity || 0);
      const pct = (basis > 0 && upnl != null) ? (upnl / basis * 100) : null;
      const w = weightOf(p);
      const rawTkr = String(p.symbol).split(" ")[0];
      const isCrypto = p.assetClass === "crypto";
      html += `<tr data-ticker="${esc(rawTkr)}" onclick="openPanel('${esc(rawTkr)}')">`
        + '<td class="tkr' + (isCrypto ? " crypto" : "") + '">' + esc(p.symbol) + '</td>'
        + '<td class="nm">' + esc(p.name || "") + '</td>'
        + '<td class="sec">' + esc(p.sector || "—") + '</td>'
        + '<td class="r">' + (p.quantity || 0).toLocaleString() + '</td>'
        + '<td class="r">$' + fmtN(p.avg_cost, 2) + '</td>'
        + '<td class="r">' + fmtUSD(p.market_value) + '</td>'
        + '<td class="r">' + pnlCell(upnl, false) + '</td>'
        + '<td class="r">' + pnlCell(pct, true) + '</td>'
        + '<td class="r wcell"><span>' + fmtN(w, 2) + '%</span>'
        + '<span class="wbar"><i style="width:' + (w / maxW * 100).toFixed(0) + '%"></i></span></td>'
        + '</tr>';
    });

    html += '</tbody><tfoot><tr>'
      + '<td class="tkr">TOTAL</td>'
      + '<td class="nm">' + rows.length + ' positions</td><td></td><td></td><td></td>'
      + '<td class="r">' + fmtUSD(totMV) + '</td>'
      + '<td class="r">' + pnlCell(totPnl, false) + '</td>'
      + '<td class="r">' + pnlCell(totPct, true) + '</td>'
      + '<td class="r wcell"><span>' + fmtN(totMV / nlvBase * 100, 2) + '%</span>'
      + '<span class="wbar"></span></td>'
      + '</tr></tfoot></table></div>';
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

  // ── Ticker detail slide-over ───────────────────────────────────────
  (function initSlideOver() {
    const overlay = document.getElementById("so-overlay");
    const body    = document.getElementById("so-body");
    const DETAILS = DATA.position_details || {};

    function money(v, dp) {
      if (v == null) return "—";
      dp = dp == null ? 2 : dp;
      return "$" + Math.abs(v).toLocaleString("en-US", {minimumFractionDigits:dp, maximumFractionDigits:dp});
    }
    function signMoney(v) {
      if (v == null) return "—";
      return (v >= 0 ? "+" : "−") + money(v);
    }
    function pct(v, dp) {
      if (v == null) return "—";
      dp = dp == null ? 2 : dp;
      return (v >= 0 ? "+" : "") + v.toFixed(dp) + "%";
    }
    function esc(s) {
      return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    function barColor(p) {            // proximity 0 (calm) -> 100 (firing)
      if (p >= 75) return "var(--red)";
      if (p >= 45) return "var(--amber)";
      return "var(--green)";
    }
    function fact(label, val) {
      return `<div><div class="so-fact-label">${label}</div><div class="so-fact-val">${val}</div></div>`;
    }

    // ── Price mini-chart (top of the panel body) ──────────────────────
    let soChart = null;

    function chartSection() {
      return `<div class="so-chart-wrap"><canvas id="so-chart"></canvas>`
           + `<div class="so-chart-empty" id="so-chart-empty" style="display:none">`
           + `Building price history…</div>`
           + `<div class="so-chart-legend" id="so-chart-legend"></div></div>`;
    }

    function destroyChart() {
      if (soChart) { try { soChart.destroy(); } catch (e) {} soChart = null; }
    }

    function buildSoChart(d) {
      destroyChart();
      const series = d.price_series || [];
      const canvas = document.getElementById("so-chart");
      const empty  = document.getElementById("so-chart-empty");
      const legend = document.getElementById("so-chart-legend");
      if (!canvas) return;
      if (series.length < 3) {           // too sparse to be meaningful
        canvas.style.display = "none";
        if (empty) empty.style.display = "block";
        return;
      }
      const labels = series.map(s => s.t);
      const data   = series.map(s => s.p);
      const net = d.closed ? d.realized_pnl : d.unrealized_pnl;
      const up = net == null ? true : net >= 0;
      const lineColor = up ? CH.green : CH.red;

      // Highlight the entry (cyan) and, for closed trades, the exit point.
      const ptR = [], ptBg = [];
      labels.forEach(t => {
        if (t === d.entry_t)      { ptR.push(4); ptBg.push(CH.cyan); }
        else if (t === d.exit_t)  { ptR.push(4); ptBg.push(CH.text); }
        else                      { ptR.push(0); ptBg.push(lineColor); }
      });

      const datasets = [{
        label: "Price", data, borderColor: lineColor,
        backgroundColor: (up ? "rgba(0,230,118,0.08)" : "rgba(255,68,102,0.08)"),
        borderWidth: 2, tension: 0.25, fill: true,
        pointRadius: ptR, pointBackgroundColor: ptBg, pointHoverRadius: 5,
      }];

      const hline = (val, color) => ({
        label: "", data: labels.map(() => val), borderColor: color,
        borderWidth: 1, borderDash: [4, 3], pointRadius: 0, fill: false, tension: 0,
      });
      let legendHtml = `<span><span class="lg-dot" style="background:#00ccff"></span>Entry</span>`;
      if (d.exit_t) legendHtml += `<span><span class="lg-dot" style="background:#c8d8f0"></span>Exit</span>`;
      if (d.hard_stop_price != null) {
        datasets.push(hline(d.hard_stop_price, "#ff4466"));
        legendHtml += `<span><span class="lg-dash" style="border-color:#ff4466"></span>Hard stop</span>`;
      }
      if (d.trail_stop_price != null) {
        datasets.push(hline(d.trail_stop_price, CH.amber));
        legendHtml += `<span><span class="lg-dash" style="border-color:#ffaa00"></span>Trailing stop</span>`;
      }
      if (legend) legend.innerHTML = legendHtml;

      soChart = new Chart(canvas, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: CH.surface, borderColor: CH.border, borderWidth: 1,
              titleColor: CH.text, bodyColor: CH.dim,
              filter: item => item.datasetIndex === 0,
              callbacks: { label: ctx => " $" + ctx.parsed.y.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2}) },
            },
          },
          scales: {
            x: { display: false },
            y: {
              position: "right",
              grid: { color: CH.border, drawBorder: false },
              ticks: { color: CH.dim, font: { size: 9 }, maxTicksLimit: 4,
                       callback: v => "$" + v },
            },
          },
        },
      });
    }

    function renderClosedBody(d) {
      const pnlCls = d.realized_pnl == null ? "" : (d.realized_pnl >= 0 ? "pnl-pos" : "pnl-neg");
      const pctCls = d.realized_pct == null ? "" : (d.realized_pct >= 0 ? "pnl-pos" : "pnl-neg");
      let h = chartSection();

      // Realized Trade facts
      h += `<div class="so-sec"><div class="so-sec-hdr">Realized Trade</div><div class="so-facts">`;
      h += fact("Entry date", d.entry_date ? esc(d.entry_date.slice(0,10)) : "—");
      h += fact("Exit date", d.sold_date ? esc(d.sold_date.slice(0,10)) : "—");
      h += fact("Days held", d.days_held != null ? d.days_held + "d" : "—");
      h += fact("Shares", d.quantity != null ? d.quantity.toLocaleString() : "—");
      h += fact("Entry price", money(d.entry_price));
      h += fact("Exit price", money(d.sold_price));
      h += fact("Cost basis", money(d.cost_basis));
      h += fact("Proceeds", money(d.proceeds));
      h += `</div></div>`;

      // Realized P&L
      h += `<div class="so-sec"><div class="so-sec-hdr">Realized P&amp;L</div><div class="so-kv-row">`;
      h += `<div><div class="so-kv-label">Net P&L</div><div class="so-kv-val ${pnlCls}">${signMoney(d.realized_pnl)}</div></div>`;
      h += `<div><div class="so-kv-label">Return</div><div class="so-kv-val ${pctCls}">${pct(d.realized_pct)}</div></div>`;
      h += `</div></div>`;

      // Entry Signals
      h += `<div class="so-sec"><div class="so-sec-hdr">Entry Signals</div>`;
      const sigs = d.entry_signals || [];
      if (sigs.length) {
        h += `<div class="so-pills">`;
        sigs.forEach(s => {
          const rev = String(s).toUpperCase().indexOf("REVERSION") >= 0 ? " rev" : "";
          h += `<span class="so-pill${rev}">${esc(s)}</span>`;
        });
        h += `</div>`;
      } else {
        h += `<div class="so-empty">No entry signals recorded</div>`;
      }
      h += `</div>`;

      // Exit
      h += `<div class="so-sec"><div class="so-sec-hdr">Exit</div>`;
      if (d.exit_reason) {
        h += `<div class="so-bar-meta"><span style="color:var(--dim)">Condition</span>`
          + `<span class="exit-chip ${d.exit_color || "white"}"><span class="dot"></span>${esc(d.exit_type)}</span></div>`;
        h += `<div class="so-prose" style="margin-top:10px">${esc(d.exit_reason)}</div>`;
      } else {
        h += `<div class="so-empty">Exit reason not recorded for this trade.</div>`;
      }
      h += `</div>`;

      return h;
    }

    function renderBody(d) {
      if (d.closed) return renderClosedBody(d);
      let h = chartSection();

      // Position Facts
      h += `<div class="so-sec"><div class="so-sec-hdr">Position Facts</div><div class="so-facts">`;
      h += fact("Entry date", d.entry_date ? esc(d.entry_date.slice(0,10)) : "—");
      h += fact("Days held", d.days_held != null ? d.days_held : "—");
      h += fact("Avg cost", money(d.avg_cost));
      h += fact("Shares held", d.quantity != null ? d.quantity.toLocaleString() : "—");
      h += fact("Market value", money(d.market_value));
      h += fact("Sector", d.sector ? esc(d.sector) : "—");
      h += `</div></div>`;

      // Entry Signals
      h += `<div class="so-sec"><div class="so-sec-hdr">Entry Signals</div>`;
      const sigs = d.entry_signals || [];
      if (sigs.length) {
        h += `<div class="so-pills">`;
        sigs.forEach(s => {
          const rev = String(s).toUpperCase().indexOf("REVERSION") >= 0 ? " rev" : "";
          h += `<span class="so-pill${rev}">${esc(s)}</span>`;
        });
        h += `</div>`;
      } else {
        h += `<div class="so-empty">No entry signals recorded</div>`;
      }
      const conf = d.confluence || {};
      if (conf.score != null) {
        const score = Math.max(0, Math.min(5, conf.score));
        let dots = "";
        for (let i = 0; i < 5; i++) dots += `<span class="so-conf-dot${i < score ? " on" : ""}"></span>`;
        h += `<div class="so-conf-row"><div class="so-conf-dots">${dots}</div>`
          + `<div class="so-conf-txt">Confluence ${conf.score}${conf.tier ? " · " + esc(conf.tier) : ""}</div></div>`;
      }
      h += `</div>`;

      // Entry Rationale
      h += `<div class="so-sec"><div class="so-sec-hdr">Entry Rationale</div>`;
      h += d.rationale
        ? `<div class="so-prose">${esc(d.rationale)}</div>`
        : `<div class="so-prose empty">No rationale recorded for this entry.</div>`;
      h += `</div>`;

      // Exit Status
      h += `<div class="so-sec"><div class="so-sec-hdr">Exit Status</div>`;
      const ex = d.exit_status;
      if (ex) {
        const peakCls = ex.peak_gain_pct >= 0 ? "pnl-pos" : "pnl-neg";
        h += `<div class="so-kv-row">`;
        h += `<div><div class="so-kv-label">Peak gain</div><div class="so-kv-val ${peakCls}">${pct(ex.peak_gain_pct,1)}</div></div>`;
        h += `<div><div class="so-kv-label">Trailing stop</div><div class="so-kv-val">${ex.trail_pct != null ? ex.trail_pct.toFixed(0) + "% trail" : "Not armed"}</div></div>`;
        h += `<div><div class="so-kv-label">Hard stop</div><div class="so-kv-val">${ex.closing_stop_pct != null ? ex.closing_stop_pct.toFixed(0) + "%" : "—"}</div></div>`;
        h += `</div>`;
        if (ex.closest) {
          const p = ex.closest.proximity;
          h += `<div class="so-bar-meta"><span style="color:var(--dim)">Closest trigger: <b style="color:var(--text)">${esc(ex.closest.name)}</b></span>`
            + `<span style="color:${barColor(p)};font-weight:700">${p.toFixed(0)}%</span></div>`;
          h += `<div class="so-bar-track"><div class="so-bar-fill" style="width:${p}%;background:${barColor(p)}"></div></div>`;
          h += `<div class="so-bar-detail">${esc(ex.closest.detail)}</div>`;
        }
        h += `<div class="so-expand" id="so-expand"><button class="so-expand-btn" id="so-expand-btn">`
          + `<span class="so-caret">▸</span> Show all exit conditions</button><div class="so-expand-body">`;
        (ex.conditions || []).forEach(c => {
          const pv = c.proximity;
          const proxTxt = (c.measurable && pv != null)
            ? `<span class="so-cond-prox" style="color:${barColor(pv)}">${pv.toFixed(0)}%</span>`
            : `<span class="so-cond-prox" style="color:var(--muted)">—</span>`;
          h += `<div class="so-cond"><div class="so-cond-name">${esc(c.name)}</div><div class="so-cond-detail">${esc(c.detail)}</div>${proxTxt}</div>`;
        });
        h += `</div></div>`;
      } else {
        h += `<div class="so-empty">Exit metrics unavailable for this position.</div>`;
      }
      h += `</div>`;

      // Thesis & Conviction
      h += `<div class="so-sec"><div class="so-sec-hdr">Thesis &amp; Conviction</div>`;
      const conv = ex ? ex.conviction : null;
      if (conv) {
        const st = (conv.status || "").toLowerCase();
        const rem = conv.remaining_pct;
        const cColor = st === "liberated" ? "var(--red)" : (st === "decaying" ? "var(--amber)" : "var(--green)");
        h += `<div class="so-bar-meta"><span style="color:var(--dim)">Conviction <span class="so-chip ${st}">${esc(conv.status)}</span></span>`
          + `<span style="color:var(--text);font-weight:700">${rem.toFixed(0)}%</span></div>`;
        h += `<div class="so-bar-track"><div class="so-bar-fill" style="width:${rem}%;background:${cColor}"></div></div>`;
        h += `<div class="so-bar-detail">Held ${conv.days_held}d of ${conv.window_days}d decay window</div>`;
      }
      const th = d.thesis;
      if (th) {
        const retCls = (th.return_pct != null && th.return_pct < 0) ? "pnl-neg" : "pnl-pos";
        h += `<div style="margin-top:14px"><div class="so-bar-detail" style="font-size:11px">Last review ${th.timestamp ? esc(th.timestamp.slice(0,10)) : "—"}`
          + (th.return_pct != null ? ` · <span class="${retCls}">${pct(th.return_pct,1)}</span>` : "")
          + (th.holding_days != null ? ` · ${th.holding_days}d held` : "") + `</div>`;
        if (th.sell_triggered && th.sell_reason) {
          h += `<div class="so-bar-detail" style="color:var(--amber);margin-top:6px">⚠ ${esc(th.trigger_type || "Exit")} — ${esc(th.sell_reason)}</div>`;
        }
        h += `</div>`;
      } else if (!conv) {
        h += `<div class="so-empty">No thesis review recorded yet.</div>`;
      }
      h += `</div>`;

      return h;
    }

    window.openPanel = function(ticker) {
      const d = DETAILS[(ticker || "").toUpperCase()];
      if (!d) return;
      const symEl = document.getElementById("so-sym");
      symEl.textContent = d.ticker;
      symEl.className = "so-sym " + (d.asset_class === "crypto" ? "crypto" : "equity");
      document.getElementById("so-class").textContent =
        d.closed ? "closed" : (d.asset_class || "equity");
      document.getElementById("so-name").textContent = d.name || "";
      // Closed → final exit price + realized P&L; open → live price + unrealized.
      const price   = d.closed ? d.sold_price     : d.current_price;
      const pnlVal  = d.closed ? d.realized_pnl   : d.unrealized_pnl;
      const pnlPct  = d.closed ? d.realized_pct   : d.unrealized_pct;
      document.getElementById("so-price").textContent = price != null ? money(price) : "—";
      const neg = pnlVal != null && pnlVal < 0;
      const pnlEl = document.getElementById("so-pnl");
      const pctEl = document.getElementById("so-pnl-pct");
      pnlEl.textContent = signMoney(pnlVal);
      pnlEl.className = "so-pnl-big " + (neg ? "pnl-neg" : "pnl-pos");
      pctEl.textContent = pnlPct != null ? "(" + pct(pnlPct,2) + ")" : "";
      pctEl.className = "so-pnl-pct " + (neg ? "pnl-neg" : "pnl-pos");
      body.innerHTML = renderBody(d);
      body.scrollTop = 0;
      overlay.classList.add("open");
      document.body.style.overflow = "hidden";
      buildSoChart(d);
      const exBtn = document.getElementById("so-expand-btn");
      if (exBtn) exBtn.addEventListener("click", function() {
        document.getElementById("so-expand").classList.toggle("open");
      });
    };

    function closePanel() {
      overlay.classList.remove("open");
      document.body.style.overflow = "";
      destroyChart();
    }
    document.getElementById("so-close").addEventListener("click", closePanel);
    overlay.addEventListener("click", function(e) { if (e.target === overlay) closePanel(); });
    document.addEventListener("keydown", function(e) { if (e.key === "Escape") closePanel(); });
  })();

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
