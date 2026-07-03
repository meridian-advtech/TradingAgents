#!/usr/bin/env python3
"""
migrate_position_exits_history.py — one-off migration of exit records from the
per-ticker upsert table `position_exits` (PRIMARY KEY ticker, retains only each
ticker's LATEST exit) to the append-only `position_exits_history` table (one row
per exit EVENT, so re-traded tickers keep every close).

Sources merged into position_exits_history
-------------------------------------------
1. Every current position_exits row (1 per ticker today — trivial copy). These
   legacy rows have no lot context, so entry_date/lot_ids are left NULL.
2. Additional reasoned exits that survive ONLY in kairos_ml_outcomes.db
   trade_outcomes: any CLOSED (timestamp_exit NOT NULL) row that HAS an
   exit_reason and is NOT within 1h of a position_exits row for the same ticker
   (i.e. an earlier exit position_exits overwrote). timestamp_entry is carried
   into entry_date for attribution. Trade_outcomes lives in a SEPARATE database,
   so it is read read-only and never modified here.

Rows are inserted in ascending exit_date order so that, per ticker, the append
id order matches chronological order — get_position_exit reads the highest id as
the most-recent exit, which the re-entry guard depends on.

The old position_exits table is left FULLY INTACT (frozen safety net) — not
dropped, not renamed. Nothing is written to it or to the ML DB.

Safety
------
DRY-RUN by default: prints the exact counts and the rows it would insert, writes
nothing. Pass --apply to write (inside one transaction). Refuses to --apply if
position_exits_history already contains rows (re-run guard) unless --force.

    python3 migrate_position_exits_history.py           # dry-run, show counts
    python3 migrate_position_exits_history.py --apply    # write the migration
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KAIROS_DB = os.path.join(SCRIPT_DIR, "kairos.db")
ML_DB = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")

# A trade_outcomes exit within this window of a position_exits row is the SAME
# exit already captured by source 1 — only farther-off reasoned rows are the
# extra history worth pulling in. Matches backfill_exit_reasons.MAX_MATCH_DELTA.
SAME_EXIT_DELTA_SECS = 3600  # 1 hour

_TS_FMTS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S UTC",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def _parse_ts(ts):
    if not ts:
        return None
    s = str(ts).strip()
    for fmt in _TS_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _delta_secs(a, b):
    da, db = _parse_ts(a), _parse_ts(b)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds())


def plan_migration():
    """Return (rows, from_pe_count, from_to_count) without writing.

    rows: list of dicts ready to INSERT, already sorted by exit_date asc.
    """
    kconn = sqlite3.connect(KAIROS_DB)
    kconn.row_factory = sqlite3.Row
    pe = [dict(r) for r in kconn.execute(
        "SELECT ticker, exit_date, exit_price, exit_reason, exit_signals FROM position_exits")]
    kconn.close()

    pe_by_ticker = {}
    for p in pe:
        pe_by_ticker.setdefault(p["ticker"], []).append(p)

    rows = []
    for p in pe:
        rows.append({
            "ticker": p["ticker"],
            "exit_date": p["exit_date"],
            "exit_price": p["exit_price"],
            "exit_reason": p["exit_reason"],
            "exit_signals": p["exit_signals"],
            "entry_date": None,
            "lot_ids": None,
            "src": "position_exits",
        })
    from_pe_count = len(rows)

    from_to_count = 0
    if os.path.exists(ML_DB):
        mconn = sqlite3.connect(ML_DB)
        mconn.row_factory = sqlite3.Row
        to_closed = mconn.execute(
            """SELECT ticker, timestamp_entry, timestamp_exit, price_exit, exit_reason
               FROM trade_outcomes
               WHERE timestamp_exit IS NOT NULL AND exit_reason IS NOT NULL""").fetchall()
        mconn.close()
        for r in to_closed:
            t = r["ticker"]
            near = False
            for p in pe_by_ticker.get(t, []):
                d = _delta_secs(r["timestamp_exit"], p["exit_date"])
                if d is not None and d <= SAME_EXIT_DELTA_SECS:
                    near = True
                    break
            if near:
                continue  # same exit already carried in from position_exits
            rows.append({
                "ticker": t,
                "exit_date": r["timestamp_exit"],
                "exit_price": r["price_exit"],
                "exit_reason": r["exit_reason"],
                "exit_signals": None,
                "entry_date": r["timestamp_entry"],
                "lot_ids": None,
                "src": "trade_outcomes",
            })
            from_to_count += 1

    # Chronological insert order → per-ticker id order == time order. Unparseable
    # dates sort last (stable).
    rows.sort(key=lambda x: (_parse_ts(x["exit_date"]) or datetime.max))
    return rows, from_pe_count, from_to_count


def apply_rows(rows):
    conn = sqlite3.connect(KAIROS_DB)
    try:
        for r in rows:
            conn.execute(
                """INSERT INTO position_exits_history
                   (ticker, exit_date, exit_price, exit_reason, exit_signals, entry_date, lot_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (r["ticker"], r["exit_date"], r["exit_price"], r["exit_reason"],
                 r["exit_signals"], r["entry_date"], r["lot_ids"]),
            )
        conn.commit()
    finally:
        conn.close()


def ensure_table():
    """Create position_exits_history if missing (idempotent, same DDL as app)."""
    from kairos_log_db import SCHEMA_POSITION_EXITS_HISTORY
    conn = sqlite3.connect(KAIROS_DB)
    conn.executescript(SCHEMA_POSITION_EXITS_HISTORY)
    conn.commit()
    conn.close()


def existing_count():
    conn = sqlite3.connect(KAIROS_DB)
    try:
        return conn.execute("SELECT COUNT(*) FROM position_exits_history").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write the migration (default dry-run).")
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run (default).")
    parser.add_argument("--force", action="store_true",
                        help="Allow --apply even if history table already has rows.")
    args = parser.parse_args()

    apply = args.apply and not args.dry_run
    mode = "APPLY" if apply else "DRY-RUN"

    if not os.path.exists(KAIROS_DB):
        print(f"ERROR: database not found: {KAIROS_DB}", file=sys.stderr)
        return 1

    ensure_table()
    rows, from_pe, from_to = plan_migration()
    already = existing_count()

    print(f"migrate_position_exits_history.py — mode: {mode}")
    print(f"  DB    : {KAIROS_DB}")
    print(f"  ML DB : {ML_DB} (read-only cross-reference)")
    print()
    print(f"  from position_exits (1/ticker copy) : {from_pe}")
    print(f"  from trade_outcomes (extra history) : {from_to}")
    print(f"  TOTAL rows to insert                : {len(rows)}")
    print(f"  position_exits_history existing rows: {already}")
    print()

    extra = [r for r in rows if r["src"] == "trade_outcomes"]
    if extra:
        print("  --- extra exits recovered from trade_outcomes ---")
        for r in extra:
            print(f"  {r['ticker']:<8} exit {r['exit_date']}  entry {r['entry_date']}  "
                  f"{(r['exit_reason'] or '')[:50]}")
        print()

    if apply:
        if already and not args.force:
            print(f"  REFUSING to apply: position_exits_history already has {already} row(s). "
                  f"Re-run with --force to append anyway.")
            return 1
        apply_rows(rows)
        print(f"  APPLIED — inserted {len(rows)} row(s). "
              f"position_exits_history now has {existing_count()} row(s).")
    else:
        print("  DRY-RUN — no changes written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
