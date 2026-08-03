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
# Outside regular hours the book is thin and IBKR's quote is routinely stale or
# absurdly wide — a 2026-08-03 06:00 ET dry run returned a $32.99 ask on KARD
# against a $20.82 close. Any quote this far from the last daily close is
# treated as untrustworthy and refused rather than traded on.
MAX_QUOTE_DIVERGENCE_PCT = 10.0


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


def _working_buy_qty(ib, only: set | None = None) -> dict:
    """{ticker: shares} already working as unfilled BUY orders at the broker.

    THE DOUBLE-COVER HAZARD: a working BUY has not reduced the short position
    yet, so a naive re-run reads the ticker as still fully short and places a
    SECOND order for the same shares. If both then fill, the account flips from
    short to LONG — the mirror image of the oversell this tool exists to undo.
    Orders are counted across ALL client ids via reqAllOpenOrders(), because
    the original order was placed on a different clientId than this run.
    """
    working: dict = {}
    try:
        ib.reqAllOpenOrders()
        ib.sleep(1.5)
        active = {"ApiPending", "PendingSubmit", "PreSubmitted", "Submitted",
                  "PendingCancel"}
        for t in ib.openTrades():
            try:
                if not t.orderStatus or t.orderStatus.status not in active:
                    continue
                if (t.order.action or "").upper() != "BUY":
                    continue
                sym = getattr(t.contract, "symbol", None)
                if not sym or (only and sym not in only):
                    continue
                filled = float(getattr(t.orderStatus, "filled", 0) or 0)
                remaining = float(t.order.totalQuantity) - filled
                if remaining > 0:
                    working[sym] = working.get(sym, 0.0) + remaining
            except Exception:
                continue
    except Exception as exc:
        # Cannot prove there are no working orders -> refuse to guess. The
        # caller treats an unreadable order book as "assume in flight".
        print(f"  WARNING: could not read open orders ({exc}) — "
              f"treating all tickers as having orders in flight")
        return {"__unreadable__": 1.0}
    return working


def _cancel_working_buys(ib, ticker: str) -> int:
    """Cancel working BUY orders for one ticker. Returns how many were cancelled."""
    n = 0
    try:
        for t in ib.openTrades():
            if (getattr(t.contract, "symbol", None) == ticker
                    and (t.order.action or "").upper() == "BUY"
                    and t.orderStatus
                    and t.orderStatus.status in ("PreSubmitted", "Submitted",
                                                 "PendingSubmit", "ApiPending")):
                ib.cancelOrder(t.order)
                n += 1
        if n:
            ib.sleep(2)
    except Exception as exc:
        print(f"    WARNING: cancel failed for {ticker}: {exc}")
    return n


def _prev_close(ticker: str) -> float | None:
    """Last confirmed daily close, as an independent check on the IBKR quote."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist is None or hist.empty or "Close" not in hist:
            return None
        vals = [float(v) for v in hist["Close"] if v == v and v > 0]
        return round(vals[-1], 2) if vals else None
    except Exception:
        return None


def _divergence_pct(ref: float | None, prev: float | None) -> float | None:
    if not ref or not prev:
        return None
    return (ref - prev) / prev * 100.0


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
    ap.add_argument("--replace", action="store_true",
                    help="cancel any working BUY orders for these tickers and "
                         "re-price at the current market (use when a previous "
                         "run's limits are stranded away from the market)")
    ap.add_argument("--allow-wide", action="store_true",
                    help="override the quote-sanity refusal (use only if you have "
                         "independently confirmed the price is real)")
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

        print(f"\n{'TICKER':<8}{'SHORT':>8}{'SOLD@':>11}{'REF':>11}"
              f"{'PREV CLOSE':>12}{'DIVERGE':>10}{'EST COST':>13}")
        print("-" * 73)
        total = 0.0
        suspect = []
        for s in shorts:
            s["ref"] = _reference_price(ib, s["ticker"])
            s["prev"] = _prev_close(s["ticker"])
            s["div"] = _divergence_pct(s["ref"], s["prev"])
            est = (s["ref"] or 0) * s["qty_short"]
            total += est
            ref_txt = f"{s['ref']:,.2f}" if s["ref"] else "n/a"
            prev_txt = f"{s['prev']:,.2f}" if s["prev"] else "n/a"
            if s["div"] is None:
                div_txt = "?"
            else:
                div_txt = f"{s['div']:+.1f}%"
                if abs(s["div"]) > MAX_QUOTE_DIVERGENCE_PCT:
                    div_txt += " !"
                    suspect.append(s["ticker"])
            print(f"{s['ticker']:<8}{-s['qty_short']:>8}{s['avg_cost']:>11,.2f}"
                  f"{ref_txt:>11}{prev_txt:>12}{div_txt:>10}{est:>13,.2f}")
        print("-" * 73)
        print(f"{'TOTAL':<8}{'':>8}{'':>11}{'':>11}{'':>12}{'':>10}{total:>13,.2f}\n")

        if suspect:
            print(f"  ⚠ QUOTE SANITY: {', '.join(suspect)} diverge more than "
                  f"{MAX_QUOTE_DIVERGENCE_PCT:.0f}% from the last daily close.")
            print("    Outside regular hours the book is thin and the IBKR quote "
                  "is often stale or\n    wildly wide — this is exactly what a "
                  "pre-market reading looks like. Orders for\n    these tickers "
                  "are REFUSED unless --allow-wide is passed.\n")

        working = _working_buy_qty(ib, {s["ticker"] for s in shorts})
        if working and "__unreadable__" not in working:
            print("  ⓘ WORKING BUY ORDERS already at the broker:")
            for k, v in sorted(working.items()):
                print(f"      {k}: {v:g} share(s) unfilled")
            print("    These have NOT yet reduced the short. Placing more without "
                  "cancelling\n    them risks over-covering into a LONG position, so "
                  "they are skipped\n    unless --replace is passed.\n")

        if not args.execute:
            print("DRY RUN — no orders placed. Re-run with --execute to cover.")
            if suspect:
                print("Positions above are exact; the price columns are not "
                      "trustworthy right now.")
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
            if (s["div"] is not None
                    and abs(s["div"]) > MAX_QUOTE_DIVERGENCE_PCT
                    and not args.allow_wide):
                print(f"    quote ${s['ref']:,.2f} is {s['div']:+.1f}% from the "
                      f"last close ${s['prev']:,.2f} — REFUSED")
                print("    (stale or thin book; re-run once the market is open, "
                      "or pass --allow-wide)")
                continue

            # Re-read the broker immediately before sending: the position may
            # have been covered by hand since the plan above was printed.
            live = {d["ticker"]: d["qty_short"] for d in _short_positions(ib)}
            qty = live.get(ticker, 0)
            if qty <= 0:
                print("    no longer short — SKIPPED")
                continue

            # Net off anything already working, or cancel it with --replace.
            wk = _working_buy_qty(ib, {ticker})
            pending = wk.get(ticker, 0.0) or (
                qty if "__unreadable__" in wk else 0.0)
            if pending > 0:
                if args.replace:
                    n = _cancel_working_buys(ib, ticker)
                    print(f"    cancelled {n} working BUY order(s) before re-pricing")
                    live = {d["ticker"]: d["qty_short"] for d in _short_positions(ib)}
                    qty = live.get(ticker, 0)
                    if qty <= 0:
                        print("    no longer short after cancel — SKIPPED")
                        continue
                else:
                    print(f"    {pending:g} share(s) already working as an unfilled "
                          f"BUY — SKIPPED to avoid double-covering")
                    print("    (let it fill or expire, or re-run with --replace "
                          "to cancel and re-price)")
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
