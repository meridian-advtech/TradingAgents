#!/usr/bin/env python3
"""
backfill_exit_reasons.py — one-off historical backfill of exit metadata.

Populates exit_reason (and give_back_pct where mfe_pct is already computed) on
historical, already-closed rows in kairos_ml_outcomes.db `trade_outcomes`, using
the exit reasons recorded in kairos.db `position_exits_history`.

This is the historical counterpart to the live-path fix in
kairos_ml_outcomes.record_exit_outcome (wired at the sell_holdings chokepoint):
going forward every close records its own exit_reason; this script fills the
rows that closed before that wiring existed.

Matching
--------
position_exits_history is APPEND-ONLY — a row per exit EVENT, so a re-traded
ticker has several exits, not just its latest. For each history exit row (that
has a reason) we find the trade_outcomes rows for that ticker that are already
CLOSED (timestamp_exit IS NOT NULL) and still MISSING an exit_reason, then claim
the one whose timestamp_exit is nearest that exit's date. Each trade_outcomes
row is claimed by at most ONE exit (nearest-first), so two exits of the same
ticker map to two distinct closed rows. Only exit_reason (and give_back_pct when
mfe_pct is present) are written — the pnl / timestamp_exit / price_exit already
populated by write_trade_close are left untouched.

Open rows (timestamp_exit IS NULL) are NEVER matched here: those are current
holdings, and a past exit for a since-rebought ticker must not close the live
position. Fully-open historical exits (if any) stay for the live path or a
targeted follow-up.

Safety
------
DRY-RUN by default: prints every proposed update and the matched/unmatched
counts, writes nothing. Pass --apply to actually write (inside one transaction).

    python3 backfill_exit_reasons.py            # dry-run (default), show counts
    python3 backfill_exit_reasons.py --dry-run  # same, explicit
    python3 backfill_exit_reasons.py --apply     # write the backfill
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DB = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
KAIROS_DB = os.path.join(SCRIPT_DIR, "kairos.db")

# position_exits_history keeps every exit per ticker, and a ticker traded more
# than once has several closed trade_outcomes rows. A correct match is temporally
# exact (both systems stamp the same close instant — observed Δt = 0.0h). Any
# candidate whose nearest closed row is far from the exit's date is a DIFFERENT
# exit whose real reason we do not have, so matching it would mislabel it. Only
# match within this tolerance; report the rest as ambiguous.
MAX_MATCH_DELTA_SECS = 3600  # 1 hour

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


def _ts_delta_secs(a, b):
    da, db = _parse_ts(a), _parse_ts(b)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds())


def load_exit_history():
    """Every recorded exit (a row per event) from position_exits_history,
    ordered oldest-first so nearest-first claiming is deterministic."""
    conn = sqlite3.connect(KAIROS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ticker, exit_date, exit_price, exit_reason FROM position_exits_history "
        "ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def plan_backfill():
    """Return (updates, unmatched) without writing.

    updates: list of dicts {trade_id, ticker, exit_reason, exit_date,
             timestamp_exit, delta_secs, set_give_back, give_back_pct}
    unmatched: list of dicts {ticker, why}

    A ticker now has multiple history exits; each closed trade_outcomes row is
    claimed by at most one exit (nearest-first), so two exits map to two rows.
    """
    exits = load_exit_history()

    conn = sqlite3.connect(ML_DB)
    conn.row_factory = sqlite3.Row

    # Preload null-reason closed candidates per ticker once; a shared claimed set
    # prevents two exits of the same ticker from grabbing the same trade_id.
    cand_by_ticker = {}
    for e in exits:
        t = e["ticker"]
        if t in cand_by_ticker:
            continue
        cand_by_ticker[t] = conn.execute(
            """SELECT trade_id, timestamp_exit, price_exit, pnl_pct, mfe_pct, give_back_pct
               FROM trade_outcomes
               WHERE ticker = ? AND exit_reason IS NULL AND timestamp_exit IS NOT NULL""",
            (t,),
        ).fetchall()

    updates = []
    unmatched = []
    claimed = set()  # trade_ids already matched by an earlier exit this run

    for e in exits:
        ticker = e["ticker"]
        reason = e["exit_reason"]
        if not reason:
            unmatched.append({"ticker": ticker, "why": "history row has no exit_reason"})
            continue

        candidates = [c for c in cand_by_ticker.get(ticker, []) if c["trade_id"] not in claimed]

        if not candidates:
            # Distinguish "already has a reason / all claimed" from "no closed row".
            has_closed = conn.execute(
                "SELECT COUNT(*) FROM trade_outcomes WHERE ticker = ? AND timestamp_exit IS NOT NULL",
                (ticker,),
            ).fetchone()[0]
            if has_closed:
                unmatched.append({"ticker": ticker, "why": "all closed rows already have exit_reason (or claimed)"})
            else:
                unmatched.append({"ticker": ticker, "why": "no closed trade_outcomes row (only open / none)"})
            continue

        # Claim the closed row whose timestamp_exit is nearest THIS exit's date.
        best = None
        best_delta = None
        for c in candidates:
            d = _ts_delta_secs(c["timestamp_exit"], e["exit_date"])
            # None delta (unparseable) sorts last.
            key = d if d is not None else float("inf")
            if best is None or key < best_delta:
                best, best_delta = c, key

        # Guard: only accept a temporally-exact match. A far-off nearest row is a
        # different exit of the same ticker whose reason we do not have.
        if best_delta is None or best_delta > MAX_MATCH_DELTA_SECS:
            dt = "unparseable" if (best_delta is None or best_delta == float("inf")) \
                else f"{best_delta / 3600:.1f}h"
            unmatched.append({
                "ticker": ticker,
                "why": f"nearest null-reason closed row is Δt={dt} from history exit "
                       f"(likely a different exit) — skipped",
            })
            continue

        claimed.add(best["trade_id"])
        set_give_back = best["mfe_pct"] is not None and best["give_back_pct"] is None
        give_back_val = None
        if set_give_back:
            pnl = best["pnl_pct"] if best["pnl_pct"] is not None else 0.0
            give_back_val = round(best["mfe_pct"] - pnl, 4)

        updates.append({
            "trade_id": best["trade_id"],
            "ticker": ticker,
            "exit_reason": reason,
            "exit_date": e["exit_date"],
            "timestamp_exit": best["timestamp_exit"],
            "delta_secs": None if best_delta == float("inf") else best_delta,
            "set_give_back": set_give_back,
            "give_back_pct": give_back_val,
        })

    conn.close()
    return updates, unmatched


def apply_updates(updates):
    conn = sqlite3.connect(ML_DB)
    try:
        for u in updates:
            if u["set_give_back"]:
                conn.execute(
                    "UPDATE trade_outcomes SET exit_reason = ?, give_back_pct = ? WHERE trade_id = ?",
                    (u["exit_reason"], u["give_back_pct"], u["trade_id"]),
                )
            else:
                conn.execute(
                    "UPDATE trade_outcomes SET exit_reason = ? WHERE trade_id = ?",
                    (u["exit_reason"], u["trade_id"]),
                )
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write the backfill (default is dry-run).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Explicit dry-run (default). Writes nothing.")
    args = parser.parse_args()

    apply = args.apply and not args.dry_run
    mode = "APPLY" if apply else "DRY-RUN"

    for path in (ML_DB, KAIROS_DB):
        if not os.path.exists(path):
            print(f"ERROR: database not found: {path}", file=sys.stderr)
            return 1

    updates, unmatched = plan_backfill()

    print(f"backfill_exit_reasons.py — mode: {mode}")
    print(f"  ML DB : {ML_DB}")
    print(f"  src   : {KAIROS_DB} position_exits_history")
    print()
    print(f"  history exits scanned       : {len(updates) + len(unmatched)}")
    print(f"  MATCHED (would update)      : {len(updates)}")
    print(f"  UNMATCHED (skipped)         : {len(unmatched)}")
    give_back_n = sum(1 for u in updates if u["set_give_back"])
    print(f"    of matched, also set give_back_pct: {give_back_n}")
    print()

    if updates:
        print("  --- matched (exit_reason ← position_exits_history) ---")
        print(f"  {'ticker':<8} {'trade_id':<10} {'Δt':>10}  reason")
        for u in sorted(updates, key=lambda x: x["ticker"]):
            dt = "—" if u["delta_secs"] is None else f"{u['delta_secs']/3600:.1f}h"
            gb = "  +gb" if u["set_give_back"] else ""
            print(f"  {u['ticker']:<8} {u['trade_id'][:8]:<10} {dt:>10}  "
                  f"{u['exit_reason'][:70]}{gb}")
        print()

    if unmatched:
        print("  --- unmatched (no update) ---")
        for m in sorted(unmatched, key=lambda x: x["ticker"]):
            print(f"  {m['ticker']:<8} {m['why']}")
        print()

    if apply:
        apply_updates(updates)
        print(f"  APPLIED {len(updates)} update(s).")
    else:
        print("  DRY-RUN — no changes written. Re-run with --apply to write.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
