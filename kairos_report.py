"""
Kairos Performance Report — Milestone 6

Queries kairos.db and prints a performance summary:
  - Total decisions and breakdown by action
  - Execution stats (fill rate, avg price)
  - Win rate, avg gain/loss (from outcomes table)
  - Best and worst decisions
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
W = 72


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


def main():
    if not os.path.exists(DB_PATH):
        print("  ERROR: kairos.db not found. Run kairos_log_db.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS PERFORMANCE REPORT — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    # ── Decision Summary ─────────────────────────────────────────────
    print(banner("Decision Summary"))

    total = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    print(f"  Total decisions: {total}")

    if total == 0:
        print("  No decisions recorded yet.")
        conn.close()
        return

    actions = conn.execute(
        "SELECT action, COUNT(*) as cnt, SUM(quantity) as total_qty "
        "FROM decisions GROUP BY action ORDER BY cnt DESC"
    ).fetchall()
    for a in actions:
        print(f"    {a['action']:>5}: {a['cnt']} decision(s), {a['total_qty'] or 0} total shares")

    tickers = conn.execute(
        "SELECT ticker, COUNT(*) as cnt FROM decisions GROUP BY ticker ORDER BY cnt DESC"
    ).fetchall()
    print(f"\n  Tickers traded: {', '.join(t['ticker'] for t in tickers)}")

    # ── Execution Summary ────────────────────────────────────────────
    print(banner("Execution Summary"))

    executed = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE execution_status = 'Filled'"
    ).fetchone()[0]
    skipped = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE execution_status = 'Skipped'"
    ).fetchone()[0]
    failed = total - executed - skipped

    print(f"  Filled:    {executed}")
    print(f"  Skipped:   {skipped}")
    if failed:
        print(f"  Failed:    {failed}")
    print(f"  Fill rate: {executed / max(total - no_action, 1) * 100:.0f}% (of actionable)")

    # Avg execution price and commission
    stats = conn.execute(
        "SELECT AVG(execution_price) as avg_price, SUM(commission) as total_comm, "
        "SUM(quantity * execution_price) as total_value "
        "FROM decisions WHERE execution_status = 'Filled'"
    ).fetchone()
    if stats["avg_price"]:
        print(f"\n  Avg fill price:    ${stats['avg_price']:.2f}")
        print(f"  Total value traded: ${stats['total_value']:,.2f}")
        print(f"  Total commissions:  ${stats['total_comm']:.2f}")

    # ── Individual Trades ────────────────────────────────────────────
    print(banner("Trade Log"))

    trades = conn.execute(
        "SELECT id, timestamp, ticker, action, quantity, execution_price, "
        "execution_status, commission, net_liq_after "
        "FROM decisions ORDER BY id"
    ).fetchall()

    print(f"  {'#':>3}  {'Time':<22} {'Action':<6} {'Qty':>4} {'Ticker':<6} "
          f"{'Price':>9} {'Status':<10} {'Net Liq':>14}")
    print(f"  {'─'*3}  {'─'*22} {'─'*6} {'─'*4} {'─'*6} {'─'*9} {'─'*10} {'─'*14}")

    for t in trades:
        price_str = f"${t['execution_price']:.2f}" if t['execution_price'] else "N/A"
        liq_str = f"${t['net_liq_after']:,.2f}" if t['net_liq_after'] else "N/A"
        print(f"  {t['id']:>3}  {t['timestamp']:<22} {t['action']:<6} "
              f"{t['quantity']:>4} {t['ticker']:<6} {price_str:>9} "
              f"{t['execution_status'] or 'N/A':<10} {liq_str:>14}")

    # ── Outcomes / P&L ───────────────────────────────────────────────
    print(banner("Outcome Analysis"))

    outcome_count = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    if outcome_count == 0:
        print("  No outcomes recorded yet.")
        print("  (Outcomes are recorded when positions are closed or reviewed.)")
    else:
        outcomes = conn.execute(
            "SELECT o.*, d.ticker, d.action, d.quantity, d.execution_price, d.rationale "
            "FROM outcomes o JOIN decisions d ON o.decision_id = d.id "
            "ORDER BY o.pnl DESC"
        ).fetchall()

        pnls = [o["pnl"] for o in outcomes if o["pnl"] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        print(f"  Outcomes recorded: {len(pnls)}")
        print(f"  Win rate:          {len(wins) / max(len(pnls), 1) * 100:.1f}%")
        print(f"  Total P&L:         ${sum(pnls):,.2f}")
        print(f"  Avg gain (wins):   ${sum(wins) / max(len(wins), 1):,.2f}" if wins else "")
        print(f"  Avg loss (losses): ${sum(losses) / max(len(losses), 1):,.2f}" if losses else "")

        # Best and worst
        if outcomes:
            best = outcomes[0]
            worst = outcomes[-1]
            print(f"\n  Best:  {best['action']} {best['quantity']} {best['ticker']} "
                  f"@ ${best['execution_price']:.2f} → P&L ${best['pnl']:+,.2f}")
            print(f"         {best['rationale'][:80]}...")
            if len(outcomes) > 1:
                print(f"  Worst: {worst['action']} {worst['quantity']} {worst['ticker']} "
                      f"@ ${worst['execution_price']:.2f} → P&L ${worst['pnl']:+,.2f}")
                print(f"         {worst['rationale'][:80]}...")

    # ── Portfolio Snapshot (latest net liq) ──────────────────────────
    print(banner("Latest Portfolio State"))
    latest = conn.execute(
        "SELECT net_liq_after, position_after, timestamp "
        "FROM decisions WHERE net_liq_after IS NOT NULL "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()

    if latest:
        print(f"  Net Liquidation: ${latest['net_liq_after']:,.2f}")
        print(f"  As of:           {latest['timestamp']}")
        if latest["position_after"]:
            pos = json.loads(latest["position_after"])
            print(f"  Position:        {pos.get('quantity', '?')} shares @ ${pos.get('avg_cost', '?')}")
    else:
        print("  No portfolio data recorded yet.")

    print("\n" + "━" * W)
    print("  Report complete.")
    print("━" * W)

    conn.close()


if __name__ == "__main__":
    main()
