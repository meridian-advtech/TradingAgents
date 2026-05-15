"""One-shot script: cancel all open crypto orders on the IBKR paper account."""

import random
import time
from ib_insync import IB

CRYPTO_SYMBOLS = {"ETH", "BTC", "SOL", "BNB", "XRP"}
PORT = 7497
CLIENT_ID = random.randint(80, 89)


def main():
    ib = IB()
    print(f"Connecting to IBKR paper (port {PORT}, clientId {CLIENT_ID})...")
    ib.connect("127.0.0.1", PORT, clientId=CLIENT_ID, timeout=10)
    print("Connected.\n")

    open_trades = ib.openTrades()
    print(f"Open trades on account: {len(open_trades)}")

    cancelled = 0
    for trade in open_trades:
        symbol = trade.contract.symbol
        sec_type = trade.contract.secType
        status = trade.orderStatus.status

        if symbol not in CRYPTO_SYMBOLS:
            continue

        oid = trade.order.orderId
        action = trade.order.action
        qty = trade.order.totalQuantity
        limit = trade.order.lmtPrice
        tif = trade.order.tif

        print(f"  Cancelling: {action} {qty} {symbol} "
              f"(secType={sec_type}, limit=${limit}, tif={tif}, "
              f"status={status}, orderId={oid})")

        ib.cancelOrder(trade.order)
        cancelled += 1

    if cancelled:
        # Give IBKR a moment to process cancellations
        ib.sleep(3)
        print(f"\nCancelled {cancelled} crypto order(s).")
    else:
        print("\nNo open crypto orders found.")

    ib.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
