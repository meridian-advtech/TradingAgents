"""Correct decisions.quantity on the six oversell SELLs (2026-07-02 → 07-29).

The entry-sizing bug fixed in a2a1fa2 wrote SELL decisions at an ENTRY-sized
quantity rather than the held position, so six rows in kairos.db `decisions`
record more shares than were actually sold:

    2026-07-02 EQIX  16 vs 10      2026-07-29 CARR 266 vs 162
    2026-07-08 KARD 714 vs 413     2026-07-29 ETN   44 vs  28
    2026-07-08 CCL  831 vs 585     2026-07-29 EME   23 vs  15

Scope of the contamination is narrow and worth stating precisely:
  • kairos_ml_outcomes.db is CLEAN — the ML ledger recorded the true closed
    quantities and P&L, so July's +$41,133.59 needs no restatement.
  • kairos.db `holdings` is CLEAN — sell_holdings only ever closed lots that
    existed.
  • Only decisions.quantity is wrong. Its sole consumer is the total-shares
    line in kairos_report.py, so no P&L figure moves. This corrects the audit
    trail, not the accounting.

The original value is preserved in data_inputs.oversell_correction rather than
overwritten, so the record of the bug survives the correction of its effect.

Usage:
    python kairos_fix_oversell_quantities.py            # dry run
    python kairos_fix_oversell_quantities.py --apply    # back up, then write
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")

FIND_SQL = """
WITH closed AS (
    SELECT sold_date, ticker, SUM(quantity) AS lots
    FROM holdings WHERE sold_date IS NOT NULL
    GROUP BY sold_date, ticker
)
SELECT d.id, d.timestamp, d.ticker, d.quantity AS logged_qty,
       c.lots AS actual_qty, d.data_inputs
FROM decisions d
JOIN closed c ON c.ticker = d.ticker AND c.sold_date = d.timestamp
WHERE d.action = 'SELL' AND d.execution_status = 'Filled'
  AND d.quantity > c.lots
ORDER BY d.timestamp
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Correct oversold decisions.quantity rows")
    ap.add_argument("--apply", action="store_true",
                    help="write the correction (default is a dry run)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(FIND_SQL).fetchall()

    if not rows:
        print("No oversold decision rows found — nothing to correct.")
        conn.close()
        return 0

    print(f"{'ID':>6}  {'TIMESTAMP':<25}{'TICKER':<8}{'LOGGED':>8}{'ACTUAL':>8}{'DELTA':>8}")
    print("-" * 65)
    total_delta = 0
    for r in rows:
        delta = r["logged_qty"] - int(r["actual_qty"])
        total_delta += delta
        print(f"{r['id']:>6}  {r['timestamp']:<25}{r['ticker']:<8}"
              f"{r['logged_qty']:>8}{int(r['actual_qty']):>8}{delta:>8}")
    print("-" * 65)
    print(f"{'':>6}  {'':<25}{'TOTAL':<8}{'':>8}{'':>8}{total_delta:>8}\n")

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to correct.")
        conn.close()
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = f"{DB_PATH}.bak_PRE_OVERSELL_QTY_FIX_{ts}"
    conn.close()
    shutil.copy(DB_PATH, backup)
    print(f"backup: {os.path.basename(backup)}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    corrected = 0
    try:
        for r in conn.execute(FIND_SQL).fetchall():
            actual = int(r["actual_qty"])
            try:
                di = json.loads(r["data_inputs"]) if r["data_inputs"] else {}
            except (TypeError, ValueError):
                di = {}
            if not isinstance(di, dict):
                di = {"_original_data_inputs": r["data_inputs"]}
            di["oversell_correction"] = {
                "logged_quantity": r["logged_qty"],
                "actual_quantity": actual,
                "oversold_by": r["logged_qty"] - actual,
                "cause": "entry sizing applied to a SELL (fixed 2026-08-03, a2a1fa2)",
                "corrected_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"),
            }
            conn.execute(
                "UPDATE decisions SET quantity = ?, data_inputs = ? WHERE id = ?",
                (actual, json.dumps(di), r["id"]),
            )
            corrected += 1
        conn.commit()
    finally:
        conn.close()

    print(f"corrected {corrected} row(s); originals preserved in "
          f"data_inputs.oversell_correction")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    remaining = conn.execute(FIND_SQL).fetchall()
    conn.close()
    print(f"re-check: {len(remaining)} oversold row(s) remaining "
          f"{'✅' if not remaining else '— UNEXPECTED'}")
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
