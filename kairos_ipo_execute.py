"""
Kairos HOT-IPO Day 1 Executor — open-market execution (Phase 2 / "Option B")

Implements the execution half of the HOT-IPO architecture (Craft: "HOT-IPO:
IPO Tracking & Capital Allocation Architecture"). The capital engine
(kairos_ipo_capital.py) reserves dollars against a priced-but-not-yet-trading
IPO. This module watches those reservations and, the moment the ticker begins
trading, buys it on the open market sized to the reserved dollars, then calls
convert_reservation_to_position() to flip the reservation to 'converted'.

"Has it begun trading?" — there is no explicit pipeline flag for this. The
authoritative signal is a live, qualifiable IBKR quote: if qualifyContracts()
resolves the symbol AND a positive reference price comes back, the IPO is
trading. No quote / unqualifiable contract -> not trading yet -> leave the
reservation alone and skip.

Sizing — deliberate guardrail deviation: standard single-position caps
(kairos_execute.check_guardrails, 10% NLV) are NOT applied here. The HOT-IPO
architecture explicitly overrides the conservative caps for IPOs (up to 30%
of available capital at conviction 10) and reserved_usd already encodes that
aggressive sizing. The only sanity check is affordability (>= 1 share).

Execution gating (two locks, both required for a live order):
    1. kairos_config.json -> hot_ipo.dry_run == false
    2. --no-dry-run passed on the command line
If either is missing every entry is simulated: sizing/fill are computed and
logged, but no order is placed, the reservation is NOT mutated, and
convert_reservation_to_position() is NOT called.

Paper trading only: connects to IBKR at 127.0.0.1:7497 (paper). A live order
here still routes to the paper account.

Usage:
    python3 kairos_ipo_execute.py                 # dry-run: probe + plan
    python3 kairos_ipo_execute.py --list          # show active reservations
    python3 kairos_ipo_execute.py --tickers KLAR  # restrict to these reservations
    python3 kairos_ipo_execute.py --no-dry-run    # real paper orders (also needs config dry_run=false)
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

W = 72

IB_HOST = "127.0.0.1"
IB_PORT = 7497            # paper
IB_CLIENTID_LO = 70       # 70-79 reserved for the IPO executor
IB_CLIENTID_HI = 79

# BUY limit buffer over the reference price — matches kairos_execute.execute_order.
LIMIT_BUFFER = 1.005


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Config ───────────────────────────────────────────────────────────

def _hot_ipo_dry_run_default() -> bool:
    """Read hot_ipo.dry_run from kairos_config.json (defaults True)."""
    try:
        with open(os.path.join(SCRIPT_DIR, "kairos_config.json")) as f:
            return bool(json.load(f).get("hot_ipo", {}).get("dry_run", True))
    except Exception:
        return True


# ── Trading detection ────────────────────────────────────────────────

def reference_price_if_trading(ib, ticker: str):
    """Return a positive reference price iff `ticker` is trading, else None.

    Doubles as the "has the IPO begun trading?" probe: a symbol that hasn't
    started trading won't qualify or won't return a price.
    """
    from kairos_execute import get_reference_price
    try:
        return get_reference_price(ib, ticker)
    except Exception:
        return None


def detect_tradeable_reservations(ib, reservations: list[dict]) -> list[dict]:
    """Annotate each reservation with a live ref price; keep only those trading.

    Returns a list of {"reservation": <row>, "ref_price": float}.
    """
    out: list[dict] = []
    for r in reservations:
        ticker = (r.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        price = reference_price_if_trading(ib, ticker)
        if price and price > 0:
            out.append({"reservation": r, "ref_price": float(price)})
    return out


# ── Sizing ───────────────────────────────────────────────────────────

def size_ipo_order(reserved_usd: float, limit_price: float) -> int:
    """Whole shares the reserved dollars buy at limit_price. 0 if unaffordable."""
    if not reserved_usd or not limit_price or limit_price <= 0:
        return 0
    return max(0, int(math.floor(float(reserved_usd) / float(limit_price))))


# ── Entry ────────────────────────────────────────────────────────────

def execute_ipo_entry(ib, reservation: dict, ref_price: float,
                      dry_run: bool) -> dict:
    """Size and (simulate or place) a single Day 1 open-market BUY.

    On a real fill, converts the reservation to a position. In dry-run, the
    reservation is left untouched and convert is not called.
    """
    from kairos_ipo_capital import convert_reservation_to_position

    ticker = (reservation.get("ticker") or "").strip().upper()
    reserved_usd = float(reservation.get("reserved_usd") or 0.0)
    limit_price = round(ref_price * LIMIT_BUFFER, 2)
    shares = size_ipo_order(reserved_usd, limit_price)

    base = {
        "ticker": ticker,
        "reservation_id": reservation.get("id"),
        "reserved_usd": round(reserved_usd, 2),
        "ref_price": round(ref_price, 2),
        "limit_price": limit_price,
        "shares": shares,
    }

    if shares < 1:
        base.update({"status": "skipped",
                     "reason": f"reserved ${reserved_usd:,.0f} < 1 share @ ${limit_price:.2f}"})
        return base

    if dry_run:
        base.update({
            "status": "simulated",
            "est_cost": round(shares * limit_price, 2),
            "converted": False,
        })
        return base

    # Live (paper) order — reuse the stock executor's limit/fill pattern.
    from kairos_execute import execute_order
    exec_res = execute_order(ib, ticker, "BUY", shares)
    base["execution"] = exec_res
    status = (exec_res.get("status") or "").lower()

    if status == "filled" and exec_res.get("fill_price"):
        filled_qty = shares
        new_pos = exec_res.get("new_position") or {}
        if new_pos.get("quantity"):
            filled_qty = int(abs(new_pos["quantity"]))
        fill_price = float(exec_res["fill_price"])
        conv = convert_reservation_to_position(
            ticker=ticker, shares=filled_qty, avg_price=fill_price,
        )
        base.update({
            "status": "filled",
            "shares": filled_qty,
            "fill_price": round(fill_price, 4),
            "converted": bool(conv.get("ok")),
            "convert_result": conv,
        })
    else:
        base.update({
            "status": exec_res.get("status", "unknown").lower(),
            "converted": False,
            "reason": exec_res.get("reason", "no fill"),
        })
    return base


# ── Cycle orchestration (pipeline entry point) ───────────────────────

def run_ipo_execution_cycle(dry_run: bool = True, ib=None,
                            tickers=None) -> dict:
    """Probe active reservations and execute Day 1 buys for those now trading.

    Caller owns the IBKR connection (pass `ib`); responsible for disconnect.
    Returns {"executed", "converted", "skipped", "not_trading", "actions"}.
    """
    from kairos_ipo_capital import get_active_reservations

    reservations = get_active_reservations()
    if tickers:
        want = {t.strip().upper() for t in tickers if t.strip()}
        reservations = [r for r in reservations
                        if (r.get("ticker") or "").upper() in want]

    print(banner("Active IPO Reservations"))
    if not reservations:
        print("  No active (reserved) IPO reservations.")
        return {"executed": 0, "converted": 0, "skipped": 0,
                "not_trading": 0, "actions": []}
    for r in reservations:
        print(f"  {r.get('ticker'):<8} reserved ${float(r.get('reserved_usd') or 0):,.0f}  "
              f"score {r.get('conviction_score')}  (id #{r.get('id')})")

    if ib is None:
        print("\n  No IBKR connection — cannot probe trading status.")
        return {"executed": 0, "converted": 0, "skipped": 0,
                "not_trading": len(reservations), "actions": []}

    # Don't stack a buy on a reservation that already has a live order.
    try:
        from kairos_execute import get_open_order_tickers
        open_orders = get_open_order_tickers(ib)
    except Exception:
        open_orders = set()

    print(banner("Probing Trading Status"))
    tradeable = detect_tradeable_reservations(ib, reservations)
    trading_tickers = {t["reservation"].get("ticker", "").upper() for t in tradeable}
    not_trading = len(reservations) - len(tradeable)
    for r in reservations:
        tk = (r.get("ticker") or "").upper()
        if tk not in trading_tickers:
            print(f"  {tk:<8} not trading yet — left reserved")

    print(banner("Executing Day 1 Entries"))
    actions: list[dict] = []
    executed = converted = skipped = 0
    for item in tradeable:
        r = item["reservation"]
        ticker = (r.get("ticker") or "").upper()
        if ticker in open_orders:
            print(f"  SKIP {ticker}: open order already live")
            skipped += 1
            actions.append({"ticker": ticker, "status": "skipped",
                            "reason": "open order already live"})
            continue

        res = execute_ipo_entry(ib, r, item["ref_price"], dry_run)
        actions.append(res)
        status = res.get("status")
        if status in ("simulated", "filled", "submitted"):
            executed += 1
            if res.get("converted"):
                converted += 1
            tag = " [sim]" if status == "simulated" else ""
            price = res.get("fill_price", res.get("limit_price"))
            conv = " → converted" if res.get("converted") else ""
            print(f"  BUY {ticker} x{res['shares']} @ ${price:.2f}{tag}  "
                  f"(reserved ${res['reserved_usd']:,.0f}){conv}")
        else:
            skipped += 1
            print(f"  SKIP {ticker}: {res.get('reason', status)}")

    return {"executed": executed, "converted": converted, "skipped": skipped,
            "not_trading": not_trading, "actions": actions}


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kairos HOT-IPO Day 1 open-market executor")
    parser.add_argument("--list", action="store_true",
                        help="List active reservations and exit")
    parser.add_argument("--tickers", default="",
                        help="Comma-separated reservation tickers to restrict to")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Allow real paper orders (also requires config dry_run=false)")
    args = parser.parse_args()

    from kairos_ipo_capital import get_active_reservations, init_schema
    init_schema()

    if args.list:
        rows = get_active_reservations()
        print(banner("Active IPO Reservations"))
        if not rows:
            print("  (none)")
        for r in rows:
            print(f"  {r.get('ticker'):<8} ${float(r.get('reserved_usd') or 0):,.0f}  "
                  f"score {r.get('conviction_score')}  pricing {r.get('expected_pricing_date')}  "
                  f"(id #{r.get('id')})")
        return

    allow_real = (not _hot_ipo_dry_run_default()) and args.no_dry_run
    dry_run = not allow_real

    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS HOT-IPO DAY 1 EXECUTOR — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")
    mode = "LIVE ORDERS (paper)" if allow_real else "DRY-RUN (probe + plan)"
    print(f"  Mode: {mode}   (config.dry_run={_hot_ipo_dry_run_default()}, "
          f"--no-dry-run={args.no_dry_run})")

    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else None)

    ib = None
    try:
        import random
        from ib_insync import IB
        ib = IB()
        ib.connect(IB_HOST, IB_PORT,
                   clientId=random.randint(IB_CLIENTID_LO, IB_CLIENTID_HI),
                   timeout=10)
    except Exception as exc:
        print(f"  IBKR connect failed: {exc} — cannot probe trading status.")
        ib = None

    try:
        summary = run_ipo_execution_cycle(dry_run=dry_run, ib=ib, tickers=tickers)
    finally:
        if ib:
            try:
                ib.disconnect()
            except Exception:
                pass

    print("\n" + "━" * W)
    print(f"  HOT-IPO DAY 1 EXECUTOR COMPLETE  ({mode})")
    print(f"  Executed: {summary['executed']}  Converted: {summary['converted']}  "
          f"Skipped: {summary['skipped']}  Not trading: {summary['not_trading']}")
    print("━" * W)


if __name__ == "__main__":
    main()
