"""
Kairos HOT-CATALYST Options Executor — long options only

Takes the long-option signals from kairos_catalyst_signals.detect_catalyst_signals(),
selects a contract (expiry + strike) in the configured DTE / delta band,
sizes it against the options risk budget, and either SIMULATES + logs the
trade (default) or places a real IBKR order. It also manages open options
positions (50% stop / 100% take-profit / 21-DTE close).

LONG OPTIONS ONLY. Entries are BUY calls / BUY puts. The only SELL this
module ever issues is a sell-to-CLOSE on a position we already own — it
never opens a short or writes premium.

Execution gating (two locks, both required for a live order):
    1. kairos_config.json -> hot_catalyst.dry_run == false
    2. --no-dry-run passed on the command line
If either is missing, every order is simulated (logged with simulated=1).

Greeks / OPRA degradation:
    Level 2 / OPRA market data is pending. When live greeks are
    unavailable, delta is derived from a Black-Scholes model on the
    underlying + an IV estimate, and the decision row is tagged
    degraded=1 so downstream analysis knows the strike was BS-selected.

Strike selection:
    Nearest expiry inside [dte_min_entry, dte_max_entry] (closest to the
    midpoint), then the strike whose |delta| lands in the signal's delta
    band (default 30-40 / reversion 40-50 / flow 25-35), preferring strikes
    inside the band and otherwise nearest the band midpoint.

Usage:
    python3 kairos_options_execute.py                 # simulate entries + manage (dry-run)
    python3 kairos_options_execute.py --tickers NVDA,FDX
    python3 kairos_options_execute.py --manage-only   # only manage open positions
    python3 kairos_options_execute.py --no-dry-run    # real orders (also needs config dry_run=false)
    python3 kairos_options_execute.py --nlv 1000000   # NLV override when not connected to IBKR
"""

import argparse
import math
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

W = 72
RISK_FREE_RATE = 0.04
DEFAULT_SIM_NLV = 1_000_000.0


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Black-Scholes (delta + price fallback when OPRA greeks unavailable) ──

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_d1(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_delta(right: str, S: float, K: float, T: float, sigma: float,
             r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes delta. Calls in [0,1], puts in [-1,0]."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, sigma, r)
    if right == "C":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_price(right: str, S: float, K: float, T: float, sigma: float,
             r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes option price — premium fallback when no quote exists."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ── Chain / quote data (yfinance — works without OPRA) ───────────────

def _dte(expiry_iso: str) -> int:
    """Days to expiry from an ISO (YYYY-MM-DD) expiry string."""
    try:
        exp = datetime.strptime(expiry_iso, "%Y-%m-%d").date()
    except ValueError:
        return -1
    return (exp - datetime.now(timezone.utc).date()).days


def _mid(bid, ask, last) -> float | None:
    """Mid of bid/ask, falling back to last. None if nothing usable."""
    try:
        b = float(bid) if bid is not None and bid == bid else 0.0
        a = float(ask) if ask is not None and ask == ask else 0.0
        if b > 0 and a > 0:
            return round((b + a) / 2.0, 2)
        l = float(last) if last is not None and last == last else 0.0
        if l > 0:
            return round(l, 2)
    except (TypeError, ValueError):
        pass
    return None


def _select_expiry(ticker, dte_min: int, dte_max: int):
    """Pick the expiry inside [dte_min, dte_max] closest to the window midpoint.

    Returns (spot, expiry_iso, dte) or (None, None, None).
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, None, None
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", auto_adjust=False)
        if hist is None or len(hist) == 0:
            return None, None, None
        spot = float(hist["Close"].iloc[-1])
        if spot <= 0 or not tk.options:
            return None, None, None
        target = (dte_min + dte_max) / 2.0
        candidates = []
        for exp in tk.options:
            d = _dte(exp)
            if dte_min <= d <= dte_max:
                candidates.append((abs(d - target), exp, d))
        if not candidates:
            return spot, None, None
        candidates.sort()
        _, expiry, d = candidates[0]
        return spot, expiry, d
    except Exception:
        return None, None, None


def select_contract(signal: dict, cfg: dict):
    """Pick expiry + strike for a signal. Returns a contract dict or None.

    Returns:
        {
          "ticker", "right", "expiry" (ISO), "dte", "strike",
          "delta", "iv", "premium", "spot", "degraded" (bool)
        }
    degraded is True whenever delta came from the Black-Scholes fallback
    rather than live (OPRA) greeks.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    ticker = signal["ticker"]
    right = signal["right"]
    band = cfg["delta_bands"].get(signal.get("delta_band", "default"), [0.30, 0.40])
    band_lo, band_hi = float(band[0]), float(band[1])
    band_mid = (band_lo + band_hi) / 2.0

    spot, expiry, dte = _select_expiry(
        ticker, int(cfg["dte_min_entry"]), int(cfg["dte_max_entry"])
    )
    if not spot or not expiry:
        return None

    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
    except Exception:
        return None
    df = chain.calls if right == "C" else chain.puts
    if df is None or df.empty:
        return None

    T = max(dte, 1) / 365.0
    # Restrict to strikes within ±25% of spot to keep delta calc sane.
    lo, hi = spot * 0.75, spot * 1.25

    best = None
    best_dist = None
    for _, row in df.iterrows():
        try:
            strike = float(row["strike"])
        except (TypeError, ValueError):
            continue
        if not (lo <= strike <= hi):
            continue
        iv = float(row["impliedVolatility"]) if row["impliedVolatility"] == row["impliedVolatility"] else 0.0
        if iv <= 0:
            iv = signal.get("iv") or 0.0
        if iv <= 0:
            continue
        # OPRA pending -> no live greeks; delta from Black-Scholes (degraded).
        delta = bs_delta(right, spot, strike, T, iv)
        adelta = abs(delta)
        premium = _mid(row.get("bid"), row.get("ask"), row.get("lastPrice"))
        if premium is None or premium <= 0:
            premium = round(bs_price(right, spot, strike, T, iv), 2)
        if premium <= 0:
            continue
        # Prefer strikes inside the band; rank by distance to band midpoint.
        in_band = band_lo <= adelta <= band_hi
        dist = abs(adelta - band_mid) - (1.0 if in_band else 0.0)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = {
                "ticker": ticker,
                "right": right,
                "expiry": expiry,
                "dte": dte,
                "strike": strike,
                "delta": round(delta, 4),
                "iv": round(iv, 4),
                "premium": premium,
                "spot": round(spot, 2),
                "degraded": True,  # BS-selected; flip to False when OPRA greeks land
            }
    return best


def current_premium(ticker: str, expiry_iso: str, strike: float, right: str) -> float | None:
    """Live mid premium for an existing position (yfinance). None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        chain = yf.Ticker(ticker).option_chain(expiry_iso)
        df = chain.calls if right == "C" else chain.puts
        if df is None or df.empty:
            return None
        row = df.iloc[(df["strike"] - strike).abs().argsort()[:1]]
        if row.empty:
            return None
        r = row.iloc[0]
        return _mid(r.get("bid"), r.get("ask"), r.get("lastPrice"))
    except Exception:
        return None


# ── Sizing ───────────────────────────────────────────────────────────

def current_options_exposure() -> float:
    """Total cost basis of all open options positions (the 15% budget user)."""
    from kairos_log_db import get_open_options_positions
    total = 0.0
    for p in get_open_options_positions():
        cb = p.get("cost_basis")
        if cb:
            total += float(cb)
    return total


def size_position(premium: float, nlv: float, cfg: dict) -> tuple[int, dict]:
    """Number of contracts honoring per-position (2.5%) + portfolio (15%) caps.

    Returns (contracts, detail). contracts==0 means no room / too expensive.
    """
    per_pos_budget = nlv * float(cfg["max_single_position_pct"])
    options_budget = nlv * float(cfg["max_options_portfolio_pct"])
    exposure = current_options_exposure()
    remaining = options_budget - exposure
    budget = min(per_pos_budget, max(0.0, remaining))

    cost_per_contract = premium * 100.0
    contracts = int(budget // cost_per_contract) if cost_per_contract > 0 else 0

    detail = {
        "per_pos_budget": round(per_pos_budget, 2),
        "options_budget": round(options_budget, 2),
        "options_exposure": round(exposure, 2),
        "budget_used": round(budget, 2),
        "cost_per_contract": round(cost_per_contract, 2),
    }
    return max(0, contracts), detail


# ── IBKR helpers (real orders + NLV) ─────────────────────────────────

def fetch_nlv(ib) -> float | None:
    try:
        for v in ib.accountSummary():
            if v.tag == "NetLiquidation":
                return float(v.value)
    except Exception:
        pass
    return None


def _place_option_order(ib, contract_spec: dict, action: str, contracts: int,
                        limit_price: float) -> dict:
    """Place a real IBKR options order (BUY to open, SELL to close a long).

    SELL here is strictly sell-to-close on a position we already hold —
    never a naked write.
    """
    import time
    from ib_insync import Option, LimitOrder

    expiry_ibkr = contract_spec["expiry"].replace("-", "")  # YYYYMMDD
    contract = Option(contract_spec["ticker"], expiry_ibkr,
                      contract_spec["strike"], contract_spec["right"], "SMART")
    ib.qualifyContracts(contract)
    order = LimitOrder(action, contracts, round(limit_price, 2), tif="DAY")
    trade = ib.placeOrder(contract, order)

    start = time.time()
    while time.time() - start < 45:
        ib.sleep(1)
        if trade.isDone():
            break

    result = {"status": trade.orderStatus.status, "order_id": trade.order.orderId}
    if trade.fills:
        result["fill_price"] = trade.fills[0].execution.price
        result["commission"] = sum(
            f.commissionReport.commission for f in trade.fills
            if f.commissionReport.commission < 1e6
        )
    return result


# ── Entry ────────────────────────────────────────────────────────────

def execute_signal(signal: dict, cfg: dict, nlv: float, dry_run: bool,
                   ib=None) -> dict:
    """Select, size, and (simulate or place) a single long-option entry."""
    from kairos_log_db import insert_options_decision, open_options_position

    ticker = signal["ticker"]
    setup = signal["setup"]
    right = signal["right"]
    action = "BUY_CALL" if right == "C" else "BUY_PUT"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Long-only invariant.
    if right not in ("C", "P"):
        return {"ticker": ticker, "status": "rejected", "reason": "non-long signal"}

    contract = select_contract(signal, cfg)
    if contract is None:
        return {"ticker": ticker, "status": "skipped",
                "reason": "no contract in DTE/delta window"}

    contracts, sizing = size_position(contract["premium"], nlv, cfg)
    if contracts < 1:
        reason = ("options budget exhausted (15% cap)"
                  if sizing["budget_used"] <= 0
                  else "premium too large for 2.5% position budget")
        return {"ticker": ticker, "status": "skipped", "reason": reason,
                "sizing": sizing}

    premium = contract["premium"]
    cost_basis = premium * contracts * 100
    target_delta = (cfg["delta_bands"][signal.get("delta_band", "default")][0]
                    + cfg["delta_bands"][signal.get("delta_band", "default")][1]) / 2.0

    simulated = dry_run
    fill_price = premium  # simulate fill at mid
    commission = None
    exec_status = "Simulated"

    if not dry_run:
        if ib is None:
            return {"ticker": ticker, "status": "error",
                    "reason": "real order requested but no IBKR connection"}
        order_res = _place_option_order(ib, contract, "BUY", contracts, premium)
        exec_status = order_res.get("status", "Unknown")
        fill_price = order_res.get("fill_price", premium)
        commission = order_res.get("commission")
        simulated = False

    decision_id = insert_options_decision(
        timestamp=timestamp, ticker=ticker, setup=setup, action=action,
        right=right, strike=contract["strike"], expiry=contract["expiry"],
        contracts=contracts, target_delta=round(target_delta, 3),
        limit_price=premium, rationale=signal.get("rationale", ""),
        signals_fired=signal.get("signals_fired"),
        conviction=signal.get("conviction"),
        execution_status=exec_status, fill_price=fill_price,
        commission=commission, degraded=contract["degraded"],
        simulated=simulated,
    )

    position_id = open_options_position(
        decision_id=decision_id, ticker=ticker, setup=setup, right=right,
        strike=contract["strike"], expiry=contract["expiry"],
        contracts=contracts, entry_premium=fill_price, opened_at=timestamp,
        dte_at_entry=contract["dte"], delta_at_entry=contract["delta"],
        entry_underlying=contract["spot"], entry_iv=contract["iv"],
        direction=signal.get("direction"), simulated=simulated,
    )

    return {
        "ticker": ticker, "setup": setup, "action": action,
        "status": "simulated" if simulated else exec_status.lower(),
        "right": right, "strike": contract["strike"], "expiry": contract["expiry"],
        "dte": contract["dte"], "delta": contract["delta"], "iv": contract["iv"],
        "premium": fill_price, "contracts": contracts,
        "cost_basis": round(cost_basis, 2), "degraded": contract["degraded"],
        "decision_id": decision_id, "position_id": position_id,
        "sizing": sizing,
    }


# ── Management (stop / take-profit / DTE close) ──────────────────────

def manage_positions(cfg: dict, dry_run: bool, ib=None) -> list[dict]:
    """Scan open options positions and close on stop / TP / DTE rules."""
    from kairos_log_db import get_open_options_positions, close_options_position

    stop_pct = float(cfg["stop_loss_pct"])
    tp_pct = float(cfg["take_profit_pct"])
    close_dte = int(cfg["close_at_dte"])
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    actions: list[dict] = []
    for pos in get_open_options_positions():
        ticker = pos["ticker"]
        entry = float(pos["entry_premium"])
        contracts = int(pos["contracts"])
        right = pos["right"]
        expiry = pos["expiry"]
        strike = float(pos["strike"])
        dte_now = _dte(expiry)

        cur = current_premium(ticker, expiry, strike, right)

        reason = None
        if cur is not None and entry > 0:
            if cur <= entry * (1.0 - stop_pct):
                reason = "stop_50"
            elif cur >= entry * (1.0 + tp_pct):
                reason = "tp_100"
        if reason is None and 0 <= dte_now <= close_dte:
            reason = "dte_21"

        if reason is None:
            continue

        close_prem = cur if cur is not None else entry  # DTE close w/o quote: flat
        realized = (close_prem - entry) * contracts * 100

        if not dry_run and ib is not None and not pos["simulated"]:
            # Sell-to-CLOSE the long — not a new short.
            spec = {"ticker": ticker, "expiry": expiry, "strike": strike, "right": right}
            order_res = _place_option_order(ib, spec, "SELL", contracts, close_prem)
            close_prem = order_res.get("fill_price", close_prem)
            realized = (close_prem - entry) * contracts * 100

        close_options_position(
            position_id=pos["id"], close_premium=close_prem,
            close_reason=reason, closed_at=timestamp, realized_pnl=realized,
        )
        actions.append({
            "position_id": pos["id"], "ticker": ticker, "right": right,
            "strike": strike, "expiry": expiry, "reason": reason,
            "entry_premium": entry, "close_premium": close_prem,
            "contracts": contracts, "realized_pnl": round(realized, 2),
            "dte": dte_now, "simulated": bool(pos["simulated"]),
        })
    return actions


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kairos HOT-CATALYST options executor")
    parser.add_argument("--tickers", default="",
                        help="Comma-separated tickers for the Setup 2 crush scan")
    parser.add_argument("--manage-only", action="store_true",
                        help="Only manage open positions, no new entries")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Allow real orders (also requires config dry_run=false)")
    parser.add_argument("--nlv", type=float, default=None,
                        help="NLV override when not connected to IBKR (sim only)")
    args = parser.parse_args()

    from kairos_catalyst_signals import load_hot_catalyst_config
    from kairos_log_db import init_db
    init_db()
    cfg = load_hot_catalyst_config()

    allow_real = (not cfg.get("dry_run", True)) and args.no_dry_run
    dry_run = not allow_real

    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS OPTIONS EXECUTOR — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")
    mode = "LIVE ORDERS" if allow_real else "DRY-RUN (simulate + log)"
    print(f"  Mode: {mode}   (config.dry_run={cfg.get('dry_run', True)}, "
          f"--no-dry-run={args.no_dry_run})")

    # IBKR connection only when placing real orders.
    ib = None
    nlv = args.nlv
    if allow_real:
        try:
            import random
            from ib_insync import IB
            ib = IB()
            ib.connect("127.0.0.1", 7497, clientId=random.randint(60, 69), timeout=10)
            nlv = fetch_nlv(ib) or nlv
        except Exception as exc:
            print(f"  IBKR connect failed: {exc} — aborting live run.")
            return
    if nlv is None:
        nlv = DEFAULT_SIM_NLV
        print(f"  NLV: ${nlv:,.0f} (simulation default — pass --nlv to override)")
    else:
        print(f"  NLV: ${nlv:,.0f}")

    # ── Manage existing positions first (free up budget) ──────────
    print(banner("Managing Open Positions"))
    closes = manage_positions(cfg, dry_run, ib)
    if not closes:
        print("  No positions hit stop / take-profit / DTE rules.")
    else:
        for c in closes:
            tag = " [sim]" if c["simulated"] else ""
            print(f"  CLOSE {c['ticker']} {c['strike']}{c['right']} {c['expiry']} "
                  f"— {c['reason']}{tag}: {c['entry_premium']:.2f} -> "
                  f"{c['close_premium']:.2f}  PnL ${c['realized_pnl']:,.0f}")

    # ── Entries ───────────────────────────────────────────────────
    if args.manage_only:
        print(banner("Manage-only mode — skipping entries"))
        if ib:
            ib.disconnect()
        return

    print(banner("Detecting Catalyst Signals"))
    from kairos_catalyst_signals import detect_catalyst_signals
    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else None)
    signals = detect_catalyst_signals(tickers=tickers, config=cfg)
    print(f"  {len(signals)} signal(s) detected.")

    print(banner("Executing Entries"))
    executed = 0
    skipped = 0
    for s in signals:
        res = execute_signal(s, cfg, nlv, dry_run, ib)
        status = res.get("status")
        if status in ("simulated", "filled", "submitted"):
            executed += 1
            tag = " [sim]" if status == "simulated" else ""
            deg = " (degraded/BS)" if res.get("degraded") else ""
            print(f"  {res['action']} {res['ticker']} {res['strike']}{res['right']} "
                  f"{res['expiry']} x{res['contracts']} @ ${res['premium']:.2f}{tag}{deg}")
            print(f"      [{res['setup']}] dte={res['dte']} Δ={res['delta']} "
                  f"cost=${res['cost_basis']:,.0f}  decision#{res['decision_id']} "
                  f"pos#{res['position_id']}")
        else:
            skipped += 1
            print(f"  SKIP {res['ticker']}: {res.get('reason', status)}")

    if ib:
        ib.disconnect()

    print("\n" + "━" * W)
    print(f"  OPTIONS EXECUTOR COMPLETE  ({mode})")
    print(f"  Closed: {len(closes)}  Entered: {executed}  Skipped: {skipped}")
    print("━" * W)


if __name__ == "__main__":
    main()
