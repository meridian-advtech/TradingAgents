"""Stamp existing trade_outcomes rows with their attribution provenance.

Until 2026-08-03, _derive_entry_signals documented a "first non-empty source
wins" priority but implemented a UNION of all five sources. Trade-level truth
(the entry path's own trigger), ticker-level context (what else happened to be
firing for that ticker that day), and tags parsed out of rationale prose were
merged into one indistinguishable list. 38% of the corpus carries multiple
tags as a result, and HOT-CONGRESS's population became mostly trades it never
drove.

Those rows cannot be un-mixed after the fact — the inputs (that day's
signal_summary.json and screen_result.json) are gone. What we CAN do is stop
them from being mistaken for clean attribution: every pre-fix row is stamped
'legacy_mixed', so per-signal analysis can exclude them explicitly rather than
silently averaging them in with trustworthy rows.

New rows carry a real source ('explicit' / 'confluence' / 'ticker_context' /
'rationale_text' / 'none'), and only the first two support a causal claim.

Usage:
    python kairos_backfill_attribution_source.py            # dry run
    python kairos_backfill_attribution_source.py --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
LEGACY = "legacy_mixed"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stamp pre-fix trade_outcomes rows as legacy_mixed")
    ap.add_argument("--apply", action="store_true",
                    help="write the stamp (default is a dry run)")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"ML database not found: {DB_PATH}")
        return 1

    # Ensure the column exists before touching it.
    sys.path.insert(0, SCRIPT_DIR)
    from kairos_ml_outcomes import init_db
    init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) c FROM trade_outcomes").fetchone()["c"]
    unstamped = conn.execute(
        "SELECT COUNT(*) c FROM trade_outcomes "
        "WHERE signal_attribution_source IS NULL").fetchone()["c"]
    with_sigs = conn.execute(
        "SELECT COUNT(*) c FROM trade_outcomes "
        "WHERE signal_attribution_source IS NULL "
        "  AND signals_fired IS NOT NULL AND signals_fired != ''").fetchone()["c"]

    print(f"trade_outcomes rows          : {total}")
    print(f"  unstamped (pre-fix)        : {unstamped}")
    print(f"    of which carry signals   : {with_sigs}")
    print(f"  would stamp as             : {LEGACY!r}\n")

    if not unstamped:
        print("Nothing to stamp — every row already carries a provenance.")
        conn.close()
        return 0

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply.")
        conn.close()
        return 0

    conn.close()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = f"{DB_PATH}.bak_PRE_ATTRIBUTION_STAMP_{ts}"
    shutil.copy(DB_PATH, backup)
    print(f"backup: {os.path.basename(backup)}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "UPDATE trade_outcomes SET signal_attribution_source = ? "
            "WHERE signal_attribution_source IS NULL", (LEGACY,))
        conn.commit()
        print(f"stamped {cur.rowcount} row(s) as {LEGACY!r}")
        print("\nprovenance breakdown now:")
        for r in conn.execute(
            "SELECT COALESCE(signal_attribution_source,'(null)') s, COUNT(*) c "
            "FROM trade_outcomes GROUP BY s ORDER BY c DESC"
        ):
            print(f"  {r['s']:<16}{r['c']:>5}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
