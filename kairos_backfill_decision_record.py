#!/usr/bin/env python3
"""Re-attribute legacy_mixed trade_outcomes rows from the decision-time record.

Why (2026-09-28): 270 rows were quarantined as 'legacy_mixed' because their
signals_fired was built by unioning every tagging source. Per-signal consumers
were changed to read trusted rows only, which discarded most of the history.

But decisions.data_inputs.confluence.signals — the field every 'confluence'
row is written from — exists for these trades too, and it matched the ledger
on 217 of 217 'confluence' rows checked. Re-attributing from it recovers 239
rows. It changed only 25 tags (dropped HOT-REVERSION x23, HOT-EARNINGS x2),
confirming the legacy tags were mostly right and the quarantine was costing
far more evidence than it protected.

Match: same ticker, filled BUY within 15 minutes of timestamp_entry, nearest
wins, non-empty signal list. Unmatched rows stay legacy_mixed.

The original tags are preserved in legacy_signals_fired for audit.

    python3 kairos_backfill_decision_record.py           # dry run
    python3 kairos_backfill_decision_record.py --apply   # write
"""
import json, os, re, sqlite3, sys
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "kairos_ml_outcomes.db")
KDB = os.path.join(HERE, "kairos.db")
WINDOW_MIN = 15


def _ts(s):
    s = re.sub(r"\s*(UTC|Z|[+-]\d{2}:?\d{2})\s*$", "", str(s).strip()).replace("T", " ")
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:26], f)
        except ValueError:
            pass
    return None


def main(apply: bool) -> int:
    k = sqlite3.connect(KDB); k.row_factory = sqlite3.Row
    buys = defaultdict(list)
    for r in k.execute("SELECT timestamp, ticker, data_inputs FROM decisions "
                       "WHERE action='BUY' AND execution_status='Filled'"):
        dt = _ts(r["timestamp"])
        if not dt:
            continue
        try:
            di = json.loads(r["data_inputs"] or "{}")
            sig = (di.get("confluence") or {}).get("signals") or di.get("signals") or []
        except Exception:
            sig = []
        sig = [str(s).strip().upper() for s in sig if s]
        buys[r["ticker"]].append((dt, list(dict.fromkeys(sig))))
    k.close()

    ml = sqlite3.connect(ML); ml.row_factory = sqlite3.Row
    cols = {c[1] for c in ml.execute("PRAGMA table_info(trade_outcomes)")}
    rows = ml.execute("SELECT trade_id, ticker, timestamp_entry, signals_fired "
                      "FROM trade_outcomes WHERE signal_attribution_source='legacy_mixed'").fetchall()

    plan, unmatched, changes = [], 0, Counter()
    for r in rows:
        e = _ts(r["timestamp_entry"])
        best = None
        for dt, sig in buys.get(r["ticker"], []):
            if not e or not sig:
                continue
            gap = abs((dt - e).total_seconds()) / 60
            if gap <= WINDOW_MIN and (best is None or gap < best[0]):
                best = (gap, sig)
        if not best:
            unmatched += 1
            continue
        old = set(json.loads(r["signals_fired"] or "[]"))
        new = set(best[1])
        for s in old - new: changes["dropped " + s] += 1
        for s in new - old: changes["added " + s] += 1
        plan.append((r["trade_id"], r["signals_fired"], json.dumps(best[1])))

    print(f"legacy_mixed rows: {len(rows)} | re-attributable: {len(plan)} | unmatched (stay legacy): {unmatched}")
    for c, n in changes.most_common():
        print(f"   {n:>4}  {c}")

    if not apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        ml.close()
        return 0

    if "legacy_signals_fired" not in cols:
        ml.execute("ALTER TABLE trade_outcomes ADD COLUMN legacy_signals_fired TEXT")
    n = 0
    for tid, old_raw, new_raw in plan:
        cur = ml.execute(
            "UPDATE trade_outcomes SET legacy_signals_fired = ?, signals_fired = ?, "
            "signal_attribution_source = 'decision_record' "
            "WHERE trade_id = ? AND signal_attribution_source = 'legacy_mixed'",
            (old_raw, new_raw, tid))
        n += cur.rowcount
    ml.commit()
    left = ml.execute("SELECT COUNT(*) FROM trade_outcomes WHERE signal_attribution_source='legacy_mixed'").fetchone()[0]
    ml.close()
    print(f"\nAPPLIED: {n} rows -> decision_record | legacy_mixed remaining: {left}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
