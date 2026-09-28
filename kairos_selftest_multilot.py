#!/usr/bin/env python3
"""
kairos_selftest_multilot.py — multi-lot ML-ledger close harness.

Drives the REAL close path (kairos_log_db.sell_holdings -> kairos_ml_outcomes.
record_exit_outcome) against throwaway COPIES of kairos.db and
kairos_ml_outcomes.db, and asserts which trade_outcomes rows a sale closes:

  a) 1-lot full sale        1 row closed, fully stamped, nothing else touched
  b) 3-lot full sale        all 3 closed + stamped, 0 ghosts (also: one
                            consolidated holdings lot, and ledger quantities
                            that disagree with holdings — the AMZN shape)
  c) partial trim, 3 lots   exactly the FIFO-oldest whole rows closed; the rest
                            still open and UNSTAMPED
  d) write_trade_close afterwards changes nothing; no close path still calls
     the newest-row find_open_trade guess

Never writes the source databases: copies are taken with the SQLite backup API
into a temp dir and every module path is repointed there before any write.

Usage:  python kairos_selftest_multilot.py        exit 0 = all green
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kairos_log_db as L          # noqa: E402
import kairos_ml_outcomes as M     # noqa: E402
import kairos_atr_trail as ATR     # noqa: E402

RESULTS = []
CLOSE_FIELDS = ("timestamp_exit", "price_exit", "pnl_dollar", "pnl_pct",
                "outcome_label", "exit_reason", "exit_params_snapshot")


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


def ledger(where="1=1", args=()):
    c = M.get_connection()
    try:
        return {r["trade_id"]: dict(r) for r in c.execute(
            f"SELECT * FROM trade_outcomes WHERE {where}", args)}
    finally:
        c.close()


def rows_for(ticker):
    c = M.get_connection()
    try:
        return [dict(r) for r in c.execute(
            "SELECT * FROM trade_outcomes WHERE ticker = ? "
            "ORDER BY timestamp_entry", (ticker,))]
    finally:
        c.close()


def setup(ticker, qtys, consolidated=False, holdings_qty=None, arm_after=None):
    """Ledger rows (one per qty, a day apart) + matching holdings lot(s).

    consolidated: one holdings lot (as the reconciler leaves it) instead of one
    per ledger row. holdings_qty overrides its size (ledger/broker mismatch).
    arm_after: index of the ledger row after whose entry the position arms.
    """
    entries = [f"2026-09-{10 + i:02d} 15:00:00 UTC" for i in range(len(qtys))]
    tids = [M.write_trade_open(ticker, "BUY", q, 100.0 + i,
                               timestamp_entry=entries[i])
            for i, q in enumerate(qtys)]
    c = L.get_connection()
    if consolidated:
        c.execute("INSERT INTO holdings (ticker, entry_date, entry_price, quantity) "
                  "VALUES (?,?,?,?)",
                  (ticker, entries[0], 100.0, holdings_qty or sum(qtys)))
    else:
        for i, q in enumerate(qtys):
            c.execute("INSERT INTO holdings (ticker, entry_date, entry_price, quantity) "
                      "VALUES (?,?,?,?)", (ticker, entries[i], 100.0 + i, q))
    c.commit()
    c.close()
    if arm_after is not None:
        rid = ATR.record_arm(ticker, {"atr_pct": 2.5, "raw_trail_pct": 3.0,
                                      "trail_pct": 3.0, "bind_state": ATR.BIND_FREE,
                                      "atr_mult": 1.2, "trail_lo_pct": 2.0,
                                      "trail_hi_pct": 4.0, "fallback_trail_pct": 8.0})
        armed_at = entries[arm_after].replace(" UTC", "").replace(" ", "T")[:11] + "18:00:00Z"
        c = L.get_connection()
        c.execute("UPDATE armed_trail_context SET armed_at = ? WHERE id = ?",
                  (armed_at, rid))
        c.commit()
        c.close()
    return tids


def fully_stamped(r):
    if any(r.get(f) is None for f in CLOSE_FIELDS):
        return False
    snap = json.loads(r["exit_params_snapshot"])
    return bool(snap.get("params")) and snap.get("reconstructed") is False


def unstamped(r):
    return all(r.get(f) is None for f in CLOSE_FIELDS)


def sell(ticker, qty, price=110.0, ts="2026-09-28 20:30:00 UTC",
         reason="SELFTEST-EXIT: multilot"):
    return L.sell_holdings(ticker, qty, ts, price, reason)


def main():
    tmp = tempfile.mkdtemp(prefix="kairos_multilot_")
    k_copy = os.path.join(tmp, "kairos.db")
    m_copy = os.path.join(tmp, "kairos_ml_outcomes.db")
    print(f"Copying DBs into {tmp} ...")
    _copy_db(L.DB_PATH, k_copy)
    _copy_db(M.DB_PATH, m_copy)
    L.DB_PATH = k_copy
    M.DB_PATH, M.KAIROS_DB_PATH = m_copy, k_copy
    assert L.DB_PATH.startswith(tmp) and M.DB_PATH.startswith(tmp)
    M.init_db()

    test_tickers = ("ZZA", "ZZB", "ZZB2", "ZZB3", "ZZC", "ZZC2", "ZZC3")
    untouched = lambda: ledger("ticker NOT IN (%s)" % ",".join("?" * len(test_tickers)),
                               test_tickers)
    before_other = untouched()
    unlabelled_before = M.label_unlabeled_closes(dry_run=True, grace_minutes=0)

    # ── a) 1-lot full sale ───────────────────────────────────────────
    print("\na) 1-lot full sale")
    setup("ZZA", [10], arm_after=0)
    sell("ZZA", 10)
    r = rows_for("ZZA")
    check("a", "exactly 1 row, and it closed", len(r) == 1 and r[0]["timestamp_exit"])
    check("a", "row fully stamped (exit/pnl/label/reason/snapshot)", fully_stamped(r[0]))
    check("a", "label matches pnl (entry 100 -> 110 = WIN)", r[0]["outcome_label"] == "WIN")
    check("a", "armed_trail arm context stamped",
          "armed_trail" in json.loads(r[0]["exit_params_snapshot"] or "{}"))
    check("a", "no other ledger row touched", untouched() == before_other)

    # ── b) 3-lot full sale ───────────────────────────────────────────
    print("\nb) 3-lot full sale")
    setup("ZZB", [10, 20, 30])
    sell("ZZB", 60)
    r = rows_for("ZZB")
    check("b", "3 lots / one holdings lot each: all 3 closed + stamped",
          len(r) == 3 and all(fully_stamped(x) for x in r))
    check("b", "0 ghosts", not any(x["timestamp_exit"] is None for x in r))

    setup("ZZB2", [10, 20, 30], consolidated=True, arm_after=0)
    sell("ZZB2", 60)
    r = rows_for("ZZB2")
    check("b", "consolidated holdings lot: all 3 closed + stamped",
          len(r) == 3 and all(fully_stamped(x) for x in r))
    check("b", "add-on rows entered AFTER the arm still carry armed_trail",
          all("armed_trail" in json.loads(x["exit_params_snapshot"]) for x in r))

    # AMZN shape: ledger says 65+126+168=359, broker/holdings says 168.
    setup("ZZB3", [65, 126, 168], consolidated=True, holdings_qty=168)
    sell("ZZB3", 168)
    r = rows_for("ZZB3")
    check("b", "ledger qty > holdings qty: flat exit still closes all 3 (no ghost)",
          len(r) == 3 and all(fully_stamped(x) for x in r))
    check("b", "no other ledger row touched", untouched() == before_other)

    # ── c) partial trims of a 3-lot position ─────────────────────────
    print("\nc) partial trim of a 3-lot position")
    setup("ZZC", [10, 20, 30], consolidated=True)
    sell("ZZC", 30)
    r = rows_for("ZZC")
    check("c", "trim 30 of 10/20/30: FIFO-oldest two rows closed + stamped",
          fully_stamped(r[0]) and fully_stamped(r[1]))
    check("c", "trim 30: newest row still open and UNSTAMPED", unstamped(r[2]))

    setup("ZZC2", [10, 20, 30], consolidated=True)
    sell("ZZC2", 15)
    r = rows_for("ZZC2")
    check("c", "trim 15: only the oldest row closed", fully_stamped(r[0]))
    check("c", "trim 15: partly-sold row 2 and row 3 open and UNSTAMPED",
          unstamped(r[1]) and unstamped(r[2]))

    setup("ZZC3", [10, 20, 30], consolidated=True)
    sell("ZZC3", 5)
    r = rows_for("ZZC3")
    check("c", "trim 5 (< oldest row): no row closed", all(unstamped(x) for x in r))
    c = L.get_connection()
    held = c.execute("SELECT SUM(quantity) FROM holdings WHERE ticker='ZZC3' "
                     "AND sold_date IS NULL").fetchone()[0]
    c.close()
    check("c", "holdings still show the position open (55 sh)", held == 55)
    check("c", "no other ledger row touched", untouched() == before_other)

    # ── d) write_trade_close afterwards is a no-op ───────────────────
    print("\nd) write_trade_close after (a)-(c)")
    before_all = ledger()
    closed_test = [t for t, x in before_all.items()
                   if x["ticker"] in test_tickers and x["timestamp_exit"]]
    results = [M.write_trade_close(t, 999.0, timestamp_exit="2030-01-01 00:00:00")
               for t in closed_test]
    after_all = ledger()
    check("d", f"write_trade_close on all {len(closed_test)} sale-closed rows: all no-op",
          all(x.get("noop") for x in results))
    check("d", "no row in the ledger changed (price 999 / 2030 ignored)",
          after_all == before_all)
    check("d", "open-row count unchanged",
          sum(x["timestamp_exit"] is None for x in after_all.values())
          == sum(x["timestamp_exit"] is None for x in before_all.values()))
    check("d", "sales created no new unlabelled closed rows",
          M.label_unlabeled_closes(dry_run=True, grace_minutes=0) == unlabelled_before)

    # Pre-fix row shape (exit recorded, label + snapshot missing): fills only.
    tid = M.write_trade_open("ZZD", "BUY", 5, 50.0,
                             timestamp_entry="2026-09-01 15:00:00 UTC")
    c = M.get_connection()
    c.execute("UPDATE trade_outcomes SET timestamp_exit='2026-09-02T15:00:00Z', "
              "price_exit=55.0, pnl_dollar=25.0, pnl_pct=10.0 WHERE trade_id=?", (tid,))
    c.commit()
    c.close()
    M.write_trade_close(tid, 999.0)
    x = ledger("trade_id = ?", (tid,))[tid]
    check("d", "half-stamped legacy row: label + snapshot filled, exit price kept",
          x["outcome_label"] == "WIN" and x["exit_params_snapshot"]
          and x["price_exit"] == 55.0 and x["timestamp_exit"] == "2026-09-02T15:00:00Z")

    offenders = []
    for f in ("kairos_execute.py", "kairos_exits.py", "kairos_reallocation.py",
              "kairos_stoploss.py", "kairos_thesis_review.py"):
        src = open(os.path.join(HERE, f)).read()
        if "find_open_trade(" in src or "write_trade_close(" in src:
            offenders.append(f)
    check("d", "no close path calls find_open_trade / write_trade_close",
          not offenders, ", ".join(offenders))

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
