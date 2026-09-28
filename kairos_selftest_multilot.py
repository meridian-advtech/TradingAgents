#!/usr/bin/env python3
"""
kairos_selftest_multilot.py — multi-lot close harness for the fills ledger.

Drives the REAL write path (kairos_ledger.record_decision + record_fills +
rebuild_trades) against a throwaway COPY of kairos.db and asserts which trades
a sale closes:

  a) 1-lot full sale        1 trade closed, fully stamped, nothing else touched
  b) 3-lot full sale        all 3 closed + stamped, 0 ghosts — including a SELL
                            that fills in two executions, and add-on lots
                            entered after the position armed
  c) partial trim, 3 lots   FIFO: the oldest shares close first; a partly-sold
                            trade stays OPEN and UNSTAMPED; holdings agree
  d) rebuild is deterministic; the removed close paths (sell_holdings,
     record_exit_outcome, write_trade_close, find_open_trade,
     label_unlabeled_closes, reconcile_provisional_entries) are gone

Never writes the source database: the copy is taken with the SQLite backup API
into a temp dir and every module path is repointed there before any write.

Usage:  python kairos_selftest_multilot.py        exit 0 = all green
"""
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kairos_ledger as K          # noqa: E402
import kairos_log_db as LDB        # noqa: E402
import kairos_ml_outcomes as M     # noqa: E402

RESULTS = []
CLOSE_FIELDS = ("timestamp_exit", "price_exit", "pnl_dollar", "pnl_pct",
                "outcome_label", "exit_reason", "exit_params_snapshot")
_seq = [0]


def check(case, name, ok, detail=""):
    RESULTS.append((case, name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {case}: {name}"
          + (f"  — {detail}" if detail and not ok else ""))


def _copy_db(src, dst):
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    s.backup(d)
    d.close()
    s.close()


def outcomes(where="1=1", args=()):
    c = K.get_connection()
    try:
        return {r["trade_id"]: dict(r) for r in c.execute(
            f"SELECT * FROM trade_outcomes WHERE {where}", args)}
    finally:
        c.close()


def rows_for(ticker):
    c = K.get_connection()
    try:
        return [dict(r) for r in c.execute(
            "SELECT * FROM trade_outcomes WHERE ticker = ? ORDER BY timestamp_entry", (ticker,))]
    finally:
        c.close()


def fully_stamped(r):
    return all(r.get(f) is not None for f in CLOSE_FIELDS)


def unstamped(r):
    return all(r.get(f) is None for f in CLOSE_FIELDS)


def _fill(ticker, side, qty, price, at, perm):
    _seq[0] += 1
    return dict(exec_id=f"ML-{ticker}-{_seq[0]}", perm_id=perm, ticker=ticker, side=side,
                quantity=qty, price=price, commission=0.0, executed_at=at)


def setup(ticker, qtys, arm_after=None):
    """One BUY decision + fill per qty, a day apart. arm_after: index of the
    entry after which the position arms (an armed_trail_context row)."""
    base = 9000 + len(ticker) * 100 + sum(map(ord, ticker))
    tids = []
    for i, q in enumerate(qtys):
        day = f"2030-01-{i + 1:02d}"
        did, tid = K.record_decision(
            timestamp=f"{day} 15:00:00 UTC", ticker=ticker, action="BUY", quantity=q,
            execution_status="Filled", perm_id=base * 10 + i,
            entry={"signals": ["HOT-INSIDER"], "signal_attribution_source": "confluence"})
        K.record_fills([_fill(ticker, "BUY", q, 100.0 + i, f"{day}T15:00:00Z", base * 10 + i)], did)
        tids.append(tid)
        if arm_after is not None and i == arm_after:
            c = LDB.get_connection()
            c.execute("INSERT INTO armed_trail_context (ticker, armed_at, atr_pct, raw_trail_pct, "
                      "trail_pct, bind_state, atr_enabled, created_at) VALUES "
                      "(?, ?, 2.0, 3.0, 3.0, 'atr', 0, ?)",
                      (ticker, f"{day} 16:00:00 UTC", f"{day} 16:00:00 UTC"))
            c.commit()
            c.close()
    K.rebuild_trades()
    return tids


def sell(ticker, qtys, price=110.0, day="2030-02-01", reason="SELFTEST-EXIT: multilot"):
    """One SELL decision filled in len(qtys) executions."""
    perm = 7_000_000 + _seq[0]
    did, _ = K.record_decision(
        timestamp=f"{day} 15:00:00 UTC", ticker=ticker, action="SELL", quantity=sum(qtys),
        execution_status="Filled", perm_id=perm,
        exit=K.build_exit_annotation(ticker, reason))
    K.record_fills([_fill(ticker, "SELL", q, price, f"{day}T15:00:{i:02d}Z", perm)
                    for i, q in enumerate(qtys)], did)
    K.rebuild_trades()
    return did


def main():
    tmp = tempfile.mkdtemp(prefix="kairos_multilot_")
    k_copy = os.path.join(tmp, "kairos.db")
    print(f"Copying kairos.db into {tmp} ...")
    _copy_db(K.DB_PATH, k_copy)
    K.DB_PATH = LDB.DB_PATH = M.DB_PATH = M.KAIROS_DB_PATH = k_copy
    assert all(p.startswith(tmp) for p in (K.DB_PATH, LDB.DB_PATH, M.DB_PATH))
    c = K.get_connection()
    if not K.is_migrated(c):
        c.close()
        print("SKIP: kairos.db is not migrated to the fills ledger yet")
        return 1
    c.close()
    K.rebuild_trades()

    test_tickers = ("ZZA", "ZZB", "ZZB2", "ZZC", "ZZC2", "ZZC3")
    untouched = lambda: outcomes("ticker NOT IN (%s)" % ",".join("?" * len(test_tickers)),
                                 test_tickers)
    before_other = untouched()

    # ── a) 1-lot full sale ───────────────────────────────────────────
    print("\na) 1-lot full sale")
    setup("ZZA", [10], arm_after=0)
    sell("ZZA", [10])
    r = rows_for("ZZA")
    check("a", "exactly 1 trade, and it closed", len(r) == 1 and r[0]["timestamp_exit"])
    check("a", "trade fully stamped (exit/pnl/label/reason/snapshot)", fully_stamped(r[0]))
    check("a", "label matches pnl (entry 100 -> 110 = WIN)", r[0]["outcome_label"] == "WIN")
    check("a", "armed_trail arm context stamped",
          "armed_trail" in json.loads(r[0]["exit_params_snapshot"] or "{}"))
    check("a", "no other trade touched", untouched() == before_other)

    # ── b) 3-lot full sale ───────────────────────────────────────────
    print("\nb) 3-lot full sale")
    setup("ZZB", [10, 20, 30])
    sell("ZZB", [60])
    r = rows_for("ZZB")
    check("b", "3 lots: all 3 closed + stamped", len(r) == 3 and all(fully_stamped(x) for x in r))
    check("b", "0 ghosts", not any(x["timestamp_exit"] is None for x in r))

    setup("ZZB2", [10, 20, 30], arm_after=0)
    sell("ZZB2", [25, 35])
    r = rows_for("ZZB2")
    check("b", "SELL filled in two executions: all 3 closed + stamped",
          len(r) == 3 and all(fully_stamped(x) for x in r))
    check("b", "add-on lots entered AFTER the arm still carry armed_trail",
          all("armed_trail" in json.loads(x["exit_params_snapshot"]) for x in r))
    check("b", "no other trade touched", untouched() == before_other)

    # ── c) partial trims of a 3-lot position ─────────────────────────
    print("\nc) partial trim of a 3-lot position")
    setup("ZZC", [10, 20, 30])
    sell("ZZC", [30])
    r = rows_for("ZZC")
    check("c", "trim 30 of 10/20/30: FIFO-oldest two trades closed + stamped",
          fully_stamped(r[0]) and fully_stamped(r[1]))
    check("c", "trim 30: newest trade still open and UNSTAMPED", unstamped(r[2]))

    setup("ZZC2", [10, 20, 30])
    sell("ZZC2", [15])
    r = rows_for("ZZC2")
    check("c", "trim 15: only the oldest trade closed", fully_stamped(r[0]))
    check("c", "trim 15: partly-sold trade 2 and trade 3 open and UNSTAMPED",
          unstamped(r[1]) and unstamped(r[2]))
    c = K.get_connection()
    q2 = c.execute("SELECT qty_open, qty_closed FROM trades WHERE trade_id = ?",
                   (r[1]["trade_id"],)).fetchone()
    c.close()
    check("c", "trim 15: trade 2 has 5 closed, 15 still open", q2["qty_closed"] == 5 and q2["qty_open"] == 15)

    setup("ZZC3", [10, 20, 30])
    sell("ZZC3", [5])
    r = rows_for("ZZC3")
    check("c", "trim 5 (< oldest lot): no trade closed", all(unstamped(x) for x in r))
    c = K.get_connection()
    held = c.execute("SELECT SUM(quantity) FROM holdings WHERE ticker='ZZC3' "
                     "AND sold_date IS NULL").fetchone()[0]
    pos = c.execute("SELECT qty FROM positions WHERE ticker='ZZC3'").fetchone()[0]
    c.close()
    check("c", "holdings and positions both show the position open (55 sh)", held == 55 and pos == 55)
    check("c", "no other trade touched", untouched() == before_other)

    # ── d) determinism + removed close paths ─────────────────────────
    print("\nd) determinism / removed close paths")
    c = K.get_connection()
    snap = lambda: (
        [tuple(r) for r in c.execute("SELECT trade_id, ticker_current, quantity_adj, qty_open, "
                                     "price_exit, pnl_dollar, outcome_label FROM trades ORDER BY trade_id")],
        [tuple(r) for r in c.execute("SELECT trade_id, exit_ref, qty, exit_price, entry_cost "
                                     "FROM trade_matches ORDER BY trade_id, exit_ref")])
    first = snap()
    K.rebuild_trades(c)
    check("d", "rebuild_trades twice → identical trades and matches", snap() == first)
    c.close()
    removed = ("sell_holdings", "record_exit_outcome", "write_trade_close", "write_trade_open",
               "find_open_trade", "label_unlabeled_closes", "reconcile_provisional_entries",
               "insert_holding", "upsert_position_exit", "reconcile_positions_against_broker")
    offenders = []
    for f in sorted(os.listdir(HERE)):
        if not (f.startswith("kairos_") and f.endswith(".py")) or "selftest" in f:
            continue
        src = open(os.path.join(HERE, f)).read()
        for name in removed:
            if re.search(rf"\b{name}\s*\(", src):
                offenders.append(f"{f}:{name}")
    check("d", "no module defines or calls a removed close path", not offenders, ", ".join(offenders))

    shutil.rmtree(tmp, ignore_errors=True)
    failed = [r for r in RESULTS if not r[2]]
    print("\nSummary by case:")
    for case in "abcd":
        cs = [r for r in RESULTS if r[0] == case]
        print(f"  {case}: {'PASS' if all(r[2] for r in cs) else 'FAIL'} "
              f"({sum(r[2] for r in cs)}/{len(cs)})")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
