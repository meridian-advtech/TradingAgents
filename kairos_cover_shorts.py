"""Flatten unintended SHORT stock positions. Long-only remediation tool.

Kairos is long-only. Six oversell events between 2026-07-02 and 2026-07-29
(root cause: entry sizing applied to SELLs — see kairos_sell_guard and the
2026-08-03 fix) left short stock positions at the broker. This script buys
them back to flat.

SAFETY MODEL — read before running:
  • Dry-run is the DEFAULT. Nothing is sent without --execute.
  • --execute additionally requires typing the confirmation phrase.
  • It covers ONLY negative STK positions, and only up to abs(position).
    It can never open a long, and never sells anything.
  • The broker is the sole source of truth; positions are re-read immediately
    before each order, so a position covered by hand in the meantime is skipped.

Usage:
    python kairos_cover_shorts.py              # dry run — show the plan
    python kairos_cover_shorts.py --execute    # place buy-to-cover orders
    python kairos_cover_shorts.py --execute --ticker ETN --ticker EME
"""

from __future__ import annotations

import argparse
import random
import sys
import time

CONFIRM_PHRASE = "COVER SHORTS"
LIMIT_BUFFER = 1.005     # marketable buy limit: 0.5% through the offer
FILL_TIMEOUT_S = 45


def _connect():
    from ib_insync import IB
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=random.randint(90, 99), timeout=10)
    return ib


def _short_positions(ib, only: set | None = None) -> list[dict]:
    """Negative STK positions at the broker, as {ticker, qty_short, avg_cost}."""
    ib.reqPositions()
    ib.sleep(1.5)
    out = []
    for p in ib.positions():
        try:
            if p.contract.secType != "STK":
                continue
            qty = float(p.position)
            if qty >= 0:
                continue
            sym = p.contract.symbol
            if only and sym not in only:
                continue
            out.append({"ticker": sym, "qty_short": int(abs(qty)),
                        "avg_cost": round(float(p.avgCost), 2)})
        except Exception as exc:
            print(f"  WARNING: could not parse a position: {exc}")
    return sorted(out, key=lambda d: d["ticker"])


def _reference_price(ib, ticker: str) -> float | None:
    from ib_insync import Stock
    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqMarketDataType(4)
        mkt = ib.reqMktData(contract)
        ib.sleep(2)
        px = None
        for attr in ("ask", "last", "close", "bid"):
            val = getattr(mkt, attr, None)
            if val is not None and val == val and val > 0:
                px = round(float(val), 2)
                break
        ib.cancelMktData(contract)
        return px
    except Exception:
        return None


def _buy_to_cover(ib, ticker: str, qty: int, ref_price: float) -> dict:
    from ib_insync import Stock, LimitOrder
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    limit_price = round(ref_price * LIMIT_BUFFER, 2)
    order = LimitOrder("BUY", qty, limit_price, tif="DAY", outsideRth=True)
    order.overridePercentageConstraints = True

    print(f"    placing BUY {qty} {ticker} @ limit ${limit_price:,.2f}")
    trade = ib.placeOrder(contract, order)
    print(f"    order id {trade.order.orderId}")

    start = time.time()
    while time.time() - start < FILL_TIMEOUT_S:
        ib.sleep(1)
        if trade.isDone():
            break

    res = {"status": trade.orderStatus.status, "order_id": trade.order.orderId}
    if trade.fills:
        res["fill_price"] = trade.fills[0].execution.price
        print(f"    FILLED @ ${res['fill_price']:,.2f}")
    else:
        print(f"    status: {res['status']} — no fill within {FILL_TIMEOUT_S}s")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Buy-to-cover unintended short stock positions")
    ap.add_argument("--execute", action="store_true",
                    help="actually place orders (default is a dry run)")
    ap.add_argument("--ticker", action="append",
                    help="limit to specific ticker(s); repeatable")
    args = ap.parse_args()

    only = set(args.ticker) if args.ticker else None

    try:
        ib = _connect()
    except Exception as exc:
        print(f"IBKR connection failed: {exc}")
        return 1

    try:
        shorts = _short_positions(ib, only)
        if not shorts:
            print("No short STK positions at the broker — nothing to cover.")
            return 0

        print(f"\n{'TICKER':<8}{'SHORT':>8}{'SOLD@':>11}{'REF':>11}{'EST COST':>13}")
        print("-" * 51)
        total = 0.0
        for s in shorts:
            s["ref"] = _reference_price(ib, s["ticker"])
            est = (s["ref"] or 0) * s["qty_short"]
            total += est
            ref_txt = f"{s['ref']:,.2f}" if s["ref"] else "n/a"
            print(f"{s['ticker']:<8}{-s['qty_short']:>8}{s['avg_cost']:>11,.2f}"
                  f"{ref_txt:>11}{est:>13,.2f}")
        print("-" * 51)
        print(f"{'TOTAL':<8}{'':>8}{'':>11}{'':>11}{total:>13,.2f}\n")

        if not args.execute:
            print("DRY RUN — no orders placed. Re-run with --execute to cover.")
            return 0

        print(f"About to BUY-TO-COVER {len(shorts)} position(s), ~${total:,.2f}.")
        try:
            typed = input(f'Type "{CONFIRM_PHRASE}" to proceed: ').strip()
        except EOFError:
            typed = ""
        if typed != CONFIRM_PHRASE:
            print("Confirmation did not match — aborted. Nothing was sent.")
            return 1

        for s in shorts:
            ticker = s["ticker"]
            print(f"\n  {ticker}:")
            if not s["ref"]:
                print("    no reference price — SKIPPED")
                continue

            # Re-read the broker immediately before sending: the position may
            # have been covered by hand since the plan above was printed.
            live = {d["ticker"]: d["qty_short"] for d in _short_positions(ib)}
            qty = live.get(ticker, 0)
            if qty <= 0:
                print("    no longer short — SKIPPED")
                continue
            if qty != s["qty_short"]:
                print(f"    position changed ({s['qty_short']} -> {qty}) — "
                      f"covering {qty}")
            _buy_to_cover(ib, ticker, qty, s["ref"])

        print("\n  Re-reading broker positions…")
        ib.sleep(2)
        remaining = _short_positions(ib)
        if remaining:
            print("  STILL SHORT:")
            for r in remaining:
                print(f"    {r['ticker']}: -{r['qty_short']}")
            print("  Re-run to cover the remainder (unfilled limits expire at "
                  "the close).")
        else:
            print("  ✅ All short positions flat.")
        return 0
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
