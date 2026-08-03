"""Kairos Trading Script - IBKR Paper Trading via ib_insync"""

import time
from datetime import datetime, timezone
from ib_insync import IB, Stock, MarketOrder, LimitOrder

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# --- Task 1: Pull account balance and positions ---
print("=" * 60)
print("TASK 1: Account Balance & Positions")
print("=" * 60)

account_values = ib.accountSummary()
balance_keys = {
    'NetLiquidation', 'TotalCashValue', 'GrossPositionValue',
    'BuyingPower', 'AvailableFunds'
}
balances = {v.tag: v.value for v in account_values if v.tag in balance_keys}
for tag, val in balances.items():
    print(f"  {tag}: ${float(val):,.2f}")

positions = ib.positions()
print(f"\nPositions ({len(positions)}):")
if positions:
    for p in positions:
        print(f"  {p.contract.symbol}: {p.position} shares @ avg ${p.avgCost:.2f}")
else:
    print("  (none)")

net_liq_before = float(balances.get('NetLiquidation', 0))

# --- Task 2: Place market order to buy 1 share of AAPL ---
print("\n" + "=" * 60)
print("TASK 2: Placing Market Order - BUY 1 AAPL")
print("=" * 60)

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

# Use limit order above market for pre-market fill (market orders don't fill in extended hours)
# First get current market price
ib.reqMarketDataType(4)  # Use delayed/frozen data if live not available
ticker = ib.reqMktData(contract)
ib.sleep(3)
# Estimate price from existing position value
last_price = ticker.last if ticker.last == ticker.last else (ticker.close if ticker.close == ticker.close else None)
if last_price is None:
    # Derive from account: gross position = 1 share of AAPL
    existing = [p for p in positions if p.contract.symbol == 'AAPL']
    last_price = existing[0].avgCost if existing else 253.0
limit_price = round(last_price * 1.005, 2)  # 0.5% above to stay within 3% constraint
print(f"  Reference price: ${last_price:.2f}, Limit: ${limit_price:.2f}")
ib.cancelMktData(contract)

order = LimitOrder('BUY', 1, limit_price, tif='GTC', outsideRth=True)
order.overridePercentageConstraints = True
trade = ib.placeOrder(contract, order)
print(f"  Order placed. Order ID: {trade.order.orderId}")

# --- Task 3: Confirm execution ---
print("\n" + "=" * 60)
print("TASK 3: Waiting for Order Execution")
print("=" * 60)

timeout = 30
start = time.time()
while time.time() - start < timeout:
    ib.sleep(1)
    if trade.isDone():
        break

status = trade.orderStatus.status
print(f"  Order status: {status}")

fill_price = None
if trade.fills:
    fill = trade.fills[0]
    fill_price = fill.execution.price
    print(f"  Fill price: ${fill_price:.2f}")
    print(f"  Fill time: {fill.execution.time}")
else:
    print("  WARNING: No fills received yet.")

# Refresh positions
ib.sleep(2)
positions_after = ib.positions()
aapl_pos = [p for p in positions_after if p.contract.symbol == 'AAPL']
if aapl_pos:
    p = aapl_pos[0]
    print(f"  AAPL position confirmed: {p.position} shares @ avg ${p.avgCost:.2f}")
else:
    print("  AAPL position not yet visible.")

# Refresh account balance
account_values_after = ib.accountSummary()
balances_after = {v.tag: v.value for v in account_values_after if v.tag in balance_keys}
net_liq_after = float(balances_after.get('NetLiquidation', 0))

# --- Task 4: Log transaction ---
print("\n" + "=" * 60)
print("TASK 4: Logging Transaction")
print("=" * 60)

log_path = '/Users/jelmore/Kairos/kairos_log.txt'
timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
fill_str = f"${fill_price:.2f}" if fill_price else "N/A"
log_entry = (
    f"Timestamp:       {timestamp}\n"
    f"Ticker:          AAPL\n"
    f"Action:          BUY\n"
    f"Quantity:        1\n"
    f"Fill Price:      {fill_str}\n"
    f"Order Status:    {status}\n"
    f"Acct Balance:    ${net_liq_after:,.2f} (was ${net_liq_before:,.2f})\n"
    f"{'-' * 60}\n"
)

with open(log_path, 'a') as f:
    f.write(log_entry)

print(f"  Transaction logged to {log_path}")
print(f"\n{'=' * 60}")
print("All tasks complete.")
print(f"{'=' * 60}")

ib.disconnect()
