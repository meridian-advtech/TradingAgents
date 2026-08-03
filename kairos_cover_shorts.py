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


def _connect(master: bool = False):
    """Connect to IB Gateway.

    master=True uses clientId 0, which is the ONLY client permitted to cancel
    orders placed by a different clientId. Without it, cancelOrder() on someone
    else's order fails with "Error 10147: OrderId N that needs to be cancelled
    is not found" — and, worse, ib_insync raises nothing, so the caller can
    believe a cancel succeeded when the order is still live at the broker.
    """
    from ib_insync import IB
    ib = IB()
    ib.connect("127.0.0.1", 7497,
               clientId=0 if master else random.randint(90, 99), timeout=10)
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
    """Cancel working BUY orders for one ticker. Returns how many actually died.

    VERIFIES the cancel rather than trusting it. A cancel request for an order
    owned by another clientId fails at the broker with error 10147 while
    ib_insync raises nothing, so counting requests SENT reports success while
    the order is still live — which is how a "cancelled" order can later fill
    and flip a flat account long.
    """
    active = ("PreSubmitted", "Submitted", "PendingSubmit", "ApiPending")
    targets = []
    try:
        for t in ib.openTrades():
            if (getattr(t.contract, "symbol", None) == ticker
                    and (t.order.action or "").upper() == "BUY"
                    and t.orderStatus and t.orderStatus.status in active):
                targets.append(t)
                ib.cancelOrder(t.order)
        if not targets:
            return 0
        ib.sleep(3)

        # Re-read from the broker and count what is genuinely gone.
        ib.reqAllOpenOrders()
        ib.sleep(1.5)
        still = {t.order.orderId for t in ib.openTrades()
                 if getattr(t.contract, "symbol", None) == ticker
                 and t.orderStatus and t.orderStatus.status in active}
        killed = [t for t in targets if t.order.orderId not in still]
        survived = [t for t in targets if t.order.orderId in still]
        if survived:
            ids = ", ".join(str(t.order.orderId) for t in survived)
            print(f"    *** CANCEL FAILED for {ticker} order(s) {ids} — they are "
                  f"STILL LIVE at the broker.")
            print(f"    They were placed by another clientId; only clientId 0 may "
                  f"cancel those.\n    Re-run with --cancel-stale, or cancel them "
                  f"in the IB Gateway UI.")
        return len(killed)
    except Exception as exc:
        print(f"    WARNING: cancel failed for {ticker}: {exc}")
        return 0


_ACTIVE_STATES = ("PreSubmitted", "Submitted", "PendingSubmit", "ApiPending")


def _list_working_buys(ib, tickers: set | None = None) -> list:
    ib.reqAllOpenOrders()
    ib.sleep(2)
    return [t for t in ib.openTrades()
            if (t.order.action or "").upper() == "BUY"
            and t.orderStatus and t.orderStatus.status in _ACTIVE_STATES
            and (not tickers or getattr(t.contract, "symbol", None) in tickers)]


def cancel_stale_buys(tickers: set | None = None, allow_global: bool = False) -> int:
    """Cancel working BUY orders. Places no orders — only withdraws instructions.

    IBKR will not let one client cancel another client's order. Neither a
    random clientId nor clientId 0 is sufficient: Gateway's master client must
    be CONFIGURED for 0 to have that power, and it is not here. What does work
    is reconnecting as the clientId that actually placed the order, which each
    order carries. That is precise — it touches only the intended orders.

    reqGlobalCancel() is the last resort behind --allow-global, because it
    cancels EVERY open order in the account, including any the trading engine
    has legitimately working.
    """
    ib = _connect()
    try:
        targets = _list_working_buys(ib, tickers)
        if not targets:
            print("  No working BUY orders to cancel.")
            return 0
        by_client: dict = {}
        for t in targets:
            print(f"  found {t.contract.symbol} BUY {t.order.totalQuantity:g} "
                  f"@ {t.order.lmtPrice} (order {t.order.orderId}, "
                  f"placed by clientId {t.order.clientId})")
            by_client.setdefault(int(t.order.clientId), []).append(t)
        want = {t.order.orderId for t in targets}
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    # Reconnect as each owning client and cancel its own orders.
    for cid, trades in by_client.items():
        print(f"\n  reconnecting as clientId {cid} to cancel its {len(trades)} order(s)…")
        try:
            from ib_insync import IB
            owner = IB()
            owner.connect("127.0.0.1", 7497, clientId=cid, timeout=10)
        except Exception as exc:
            print(f"    could not connect as clientId {cid}: {exc}")
            continue
        try:
            owner.reqAllOpenOrders()
            owner.sleep(1.5)
            ids = {t.order.orderId for t in trades}
            for t in owner.openTrades():
                if t.order.orderId in ids and t.orderStatus.status in _ACTIVE_STATES:
                    owner.cancelOrder(t.order)
            owner.sleep(3)
        finally:
            try:
                owner.disconnect()
            except Exception:
                pass

    # Verify against the broker from a fresh connection.
    ib = _connect()
    try:
        still = [(t.contract.symbol, t.order.orderId)
                 for t in _list_working_buys(ib, tickers) if t.order.orderId in want]
        if not still:
            print(f"\n  ✅ all {len(want)} order(s) cancelled and verified gone.")
            return len(want)
        print(f"\n  *** STILL LIVE: {still}")
        if not allow_global:
            print("  Per-client cancel did not take. Re-run with --allow-global to "
                  "use\n  reqGlobalCancel(), which cancels EVERY open order in the "
                  "account.")
            return 0
        others = [t for t in _list_working_buys(ib) if t.order.orderId not in want]
        if others:
            print(f"  NOTE: global cancel will ALSO kill {len(others)} unrelated "
                  f"order(s): "
                  f"{[(t.contract.symbol, t.order.orderId) for t in others]}")
        print("  issuing reqGlobalCancel()…")
        ib.reqGlobalCancel()
        ib.sleep(4)
        left = [(t.contract.symbol, t.order.orderId)
                for t in _list_working_buys(ib, tickers) if t.order.orderId in want]
        if left:
            print(f"  *** STILL LIVE after global cancel: {left} — cancel from the "
                  f"IBKR Client Portal or mobile app.")
            return 0
        print("  ✅ cancelled and verified gone via global cancel.")
        return len(want)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


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
    ap.add_argument("--cancel-stale", action="store_true",
                    help="cancel working BUY orders (as clientId 0, so it can "
                         "reach orders placed by any client) and exit. Places "
                         "no orders.")
    ap.add_argument("--allow-global", action="store_true",
                    help="with --cancel-stale: fall back to reqGlobalCancel(), "
                         "which cancels EVERY open order in the account")
    ap.add_argument("--replace", action="store_true",
                    help="cancel any working BUY orders for these tickers and "
                         "re-price at the current market (use when a previous "
                         "run's limits are stranded away from the market)")
    ap.add_argument("--allow-wide", action="store_true",
                    help="override the quote-sanity refusal (use only if you have "
                         "independently confirmed the price is real)")
    args = ap.parse_args()

    only = set(args.ticker) if args.ticker else None

    if args.cancel_stale:
        try:
            cancel_stale_buys(only, allow_global=args.allow_global)
        except Exception as exc:
            print(f"IBKR connection failed: {exc}")
            return 1
        return 0

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
            print("  Unfilled limits stay WORKING until the close. A plain "
                  "re-run will SKIP them\n  rather than place a duplicate "
                  "(that is what --replace is for):")
            print("    kairos_cover_shorts.py --execute --replace   "
                  "# cancel + re-price at market")
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
