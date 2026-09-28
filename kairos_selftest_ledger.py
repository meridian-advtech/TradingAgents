#!/usr/bin/env python3
"""Fills-ledger selftest — simulated scenarios against a throwaway kairos.db.

Touches no real database: every scenario runs in a temp directory with its own
schema (kairos_log_db.init_db on a temp DB_PATH). The broker is a fake object
exposing only what broker_check calls.

  a) BUY in 2 executions → 2 fills, 1 trade at the qty-weighted price
  b) partial SELL closes the OLDEST entry first (FIFO), rest stays open
  c) a late fill (arrives after the live record) is caught by the broker check
     and linked to its decision by perm_id; positions then match the broker
  d) recording the same execution twice leaves one row
  e) a BUY decision without annotations cannot be stored (API and raw SQL)
  f) DD 1-for-3 reverse split and EA cash merger → right lots, CORPORATE-ACTION
  g) fills / position_events refuse UPDATE and DELETE
  h) exit_signals '[]' is refused without a note
  i) a SELL beyond the open quantity opens a short (long-only violation) that a
     later BUY covers first

Run: python3 kairos_selftest_ledger.py
"""

import os
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kairos_ledger as L  # noqa: E402
import kairos_log_db  # noqa: E402

_results = []


def check(label, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  — {detail}" if detail and not cond else ""))


def fresh_db():
    d = tempfile.mkdtemp(prefix="kairos_ledger_selftest_")
    path = os.path.join(d, "kairos.db")
    kairos_log_db.DB_PATH = path
    L.DB_PATH = path
    kairos_log_db.init_db()
    return L.get_connection(path)


def entry(sigs=("HOT-INSIDER",)):
    return dict(signals=list(sigs), signal_attribution_source="confluence",
                confluence_score=3, conviction=7)


def buy(conn, ticker, qty, ts, perm, fills=(), status="Filled"):
    did, tid = L.record_decision(timestamp=ts, ticker=ticker, action="BUY", quantity=qty,
                                 execution_status=status, perm_id=perm, entry=entry(), conn=conn)
    if fills:
        L.record_fills(list(fills), did, source="live", conn=conn)
    return did, tid


def sell(conn, ticker, qty, ts, perm, fills=(), reason="TRAILING-STOP: test"):
    did, _ = L.record_decision(timestamp=ts, ticker=ticker, action="SELL", quantity=qty,
                               execution_status="Filled", perm_id=perm,
                               exit=dict(exit_reason=reason, exit_signals=["HOT-INSIDER"]),
                               conn=conn)
    if fills:
        L.record_fills(list(fills), did, source="live", conn=conn)
    return did


def f(exec_id, ticker, side, qty, px, at, perm, comm=1.0):
    return dict(exec_id=exec_id, perm_id=perm, ticker=ticker, side=side, quantity=qty,
                price=px, commission=comm, executed_at=at)


def main():
    # ── a) + b) ────────────────────────────────────────────────────────
    print("\na) BUY in two executions / b) partial SELL is FIFO")
    c = fresh_db()
    _, t1 = buy(c, "ZZA", 100, "2026-09-01 14:00:00 UTC", 1001, [
        f("e1", "ZZA", "BUY", 60, 10.0, "2026-09-01T14:00:01Z", 1001),
        f("e2", "ZZA", "BUY", 40, 11.0, "2026-09-01T14:00:03Z", 1001)])
    _, t2 = buy(c, "ZZA", 50, "2026-09-02 14:00:00 UTC", 1002, [
        f("e3", "ZZA", "BUY", 50, 12.0, "2026-09-02T14:00:01Z", 1002)])
    L.rebuild_trades(c)
    n_fills = c.execute("SELECT COUNT(*) FROM fills WHERE perm_id = 1001").fetchone()[0]
    tr = c.execute("SELECT * FROM trades WHERE trade_id = ?", (t1,)).fetchone()
    check("2 executions → 2 fills", n_fills == 2)
    check("… and ONE trade", c.execute("SELECT COUNT(*) FROM trades WHERE ticker='ZZA'").fetchone()[0] == 2
          and tr is not None and tr["quantity"] == 100)
    check("trade entry price is the qty-weighted fill price (10.40)",
          abs(tr["price_entry"] - 10.40) < 1e-9, tr["price_entry"] if tr else None)
    sell(c, "ZZA", 70, "2026-09-03 14:00:00 UTC", 1003, [
        f("e4", "ZZA", "SELL", 70, 13.0, "2026-09-03T14:00:01Z", 1003)])
    L.rebuild_trades(c)
    a = c.execute("SELECT qty_open, qty_closed FROM trades WHERE trade_id = ?", (t1,)).fetchone()
    b = c.execute("SELECT qty_open, qty_closed FROM trades WHERE trade_id = ?", (t2,)).fetchone()
    check("partial SELL of 70 closes 70 of the OLDEST trade", a["qty_closed"] == 70 and a["qty_open"] == 30)
    check("… and leaves the newer trade untouched", b["qty_closed"] == 0 and b["qty_open"] == 50)
    open_ids = [r[0] for r in c.execute("SELECT trade_id FROM trade_outcomes WHERE ticker='ZZA' "
                                        "AND timestamp_exit IS NULL")]
    check("both trades still open in trade_outcomes (neither fully sold)", set(open_ids) == {t1, t2})
    h = c.execute("SELECT SUM(quantity) FROM holdings WHERE ticker='ZZA' AND sold_date IS NULL").fetchone()[0]
    check("holdings view open qty = 80", abs(h - 80) < 1e-9, h)
    sell(c, "ZZA", 80, "2026-09-04 14:00:00 UTC", 1004, [
        f("e5", "ZZA", "SELL", 80, 9.0, "2026-09-04T14:00:01Z", 1004, comm=2.0)])
    L.rebuild_trades(c)
    t = c.execute("SELECT * FROM trade_outcomes WHERE trade_id = ?", (t1,)).fetchone()
    # t1: 70 @13 (comm 1.0 of e4) + 30 @9 (30/80 of 2.0); cost 1040 + entry comm 2.0
    exp = 70 * 13 + 30 * 9 - 1040 - 2.0 - 1.0 - 2.0 * 30 / 80
    check("closed trade P&L is net of both legs' commissions",
          t["pnl_dollar"] is not None and abs(t["pnl_dollar"] - round(exp, 4)) < 1e-6,
          f"{t['pnl_dollar']} vs {exp}")
    check("closed trade labelled and exit_reason from its LAST exit",
          t["outcome_label"] == "WIN" and t["exit_reason"] == "TRAILING-STOP: test")
    check("exit_params_snapshot / signals carried through the view",
          t["signals_fired"] == '["HOT-INSIDER"]' and t["signal_attribution_source"] == "confluence")

    # ── c) late fill caught by broker check ───────────────────────────
    print("\nc) late fill is caught by the broker check")
    did, tid = buy(c, "ZZC", 100, "2026-09-05 14:00:00 UTC", 2001, [
        f("c1", "ZZC", "BUY", 30, 20.0, "2026-09-05T14:00:01Z", 2001)], status="Submitted")
    L.rebuild_trades(c)

    def ibfill(exec_id, sym, side, qty, px, perm, when):
        return SimpleNamespace(
            contract=SimpleNamespace(secType="STK", symbol=sym),
            execution=SimpleNamespace(execId=exec_id, permId=perm, side=side, shares=qty,
                                      price=px, time=when),
            commissionReport=SimpleNamespace(execId="", commission=0.0))
    from datetime import datetime, timezone
    when1 = datetime(2026, 9, 5, 14, 0, 1, tzinfo=timezone.utc)
    when2 = datetime(2026, 9, 5, 15, 30, 0, tzinfo=timezone.utc)
    ledger_now = {r[0]: r[1] for r in c.execute("SELECT ticker, qty FROM positions")}
    broker = dict(ledger_now)
    broker["ZZC"] = 100.0

    class FakeIB:
        def reqExecutions(self, _flt=None):
            return [ibfill("c1", "ZZC", "BOT", 30, 20.0, 2001, when1),
                    ibfill("c2", "ZZC", "BOT", 70, 20.5, 2001, when2)]

        def reqPositions(self):
            return None

        def sleep(self, _s):
            return None

        def positions(self):
            return [SimpleNamespace(contract=SimpleNamespace(secType="STK", symbol=k), position=v)
                    for k, v in broker.items()]
    res = L.broker_check(ib=FakeIB(), conn=c, post=False)
    row = c.execute("SELECT decision_id, link_method, source, commission FROM fills "
                    "WHERE exec_id = 'c2'").fetchone()
    check("broker check recorded the late fill", res["new_fills"] == 1 and row is not None)
    check("… linked to its decision by perm_id", row["decision_id"] == did and row["link_method"] == "perm_id")
    check("… source 'broker_check', commission unknown (NULL, not 0)",
          row["source"] == "broker_check" and row["commission"] is None)
    tr = c.execute("SELECT quantity, price_entry FROM trades WHERE trade_id = ?", (tid,)).fetchone()
    check("… and the trade now holds both fills (100 @ 20.35)",
          tr["quantity"] == 100 and abs(tr["price_entry"] - 20.35) < 1e-9, dict(tr))
    check("positions match the broker afterwards", res["mismatches"] == [], res["mismatches"])
    broker["ZZC"] = 90.0
    res2 = L.broker_check(ib=FakeIB(), conn=c, post=False)
    check("a mismatch is reported and NOTHING is changed",
          res2["mismatches"] == [("ZZC", 100.0, 90.0)]
          and c.execute("SELECT qty FROM positions WHERE ticker='ZZC'").fetchone()[0] == 100.0)

    # ── d) duplicate execution ────────────────────────────────────────
    print("\nd) recording the same execution twice")
    st = L.record_fills([f("e1", "ZZA", "BUY", 60, 10.0, "2026-09-01T14:00:01Z", 1001)], None,
                        source="flex_nightly", conn=c)
    check("second insert ignored, one row", st["inserted"] == 0 and
          c.execute("SELECT COUNT(*) FROM fills WHERE exec_id='e1'").fetchone()[0] == 1)
    st = L.record_fills([f("c2", "ZZC", "BUY", 70, 20.5, "20260905;113000", 2001, comm=0.35)],
                        None, source="flex_nightly", conn=c)
    check("a late commission for an exec stored without one lands in fill_commissions",
          st["commissions"] == 1 and c.execute(
              "SELECT commission FROM fill_commissions WHERE exec_id='c2'").fetchone()[0] == 0.35)

    # ── e) BUY without annotations ────────────────────────────────────
    print("\ne) a BUY decision without annotations cannot be stored")
    try:
        L.record_decision(timestamp="2026-09-06 14:00:00 UTC", ticker="ZZE", action="BUY",
                          quantity=1, execution_status="Filled", conn=c)
        ok = False
    except ValueError:
        ok = True
    check("record_decision refuses it", ok)
    try:
        with c:
            c.execute("INSERT INTO decisions (timestamp, ticker, action, quantity, execution_status) "
                      "VALUES ('2026-09-06 14:00:00 UTC','ZZE','BUY',1,'Filled')")
        ok = False
    except sqlite3.IntegrityError as exc:
        ok = "entry_annotations" in str(exc)
    check("raw SQL INSERT refused by the decisions trigger", ok)
    did_skip, tid_skip = L.record_decision(timestamp="2026-09-06 14:00:00 UTC", ticker="ZZE",
                                           action="BUY", quantity=1, execution_status="Skipped",
                                           conn=c)
    check("a Skipped BUY (never sent) needs no annotations and gets no trade_id",
          did_skip and tid_skip is None)

    # ── f) corporate actions ──────────────────────────────────────────
    print("\nf) DD reverse split and EA cash merger")
    buy(c, "DD", 326, "2026-05-19 14:14:43 UTC", 3001, [
        f("dd1", "DD", "BUY", 326, 46.84, "2026-05-19T14:14:42Z", 3001, comm=1.0)])
    buy(c, "EA", 79, "2026-07-06 20:06:49 UTC", 3002, [
        f("ea1", "EA", "BUY", 79, 205.89, "2026-07-06T20:06:47Z", 3002, comm=1.0)])
    L.record_position_events([
        dict(event_id="T-DD-OLD", ticker="DD", event_type="split", qty_change=-326, ratio=1 / 3,
             effective_at="2026-06-24T00:25:00Z", source="selftest"),
        dict(event_id="T-DD-NEW", ticker="DD", event_type="split", qty_change=108.6667, ratio=1 / 3,
             effective_at="2026-06-24T00:25:00Z", source="selftest"),
        dict(event_id="T-EA", ticker="EA", event_type="merger_cash", qty_change=-79,
             cash_per_share=210.0, proceeds=16590.0, effective_at="2026-08-05T00:25:00Z",
             source="selftest")], conn=c)
    sell(c, "DD", 108, "2026-07-01 13:38:45 UTC", 3003, [
        f("dd2", "DD", "SELL", 100, 134.53, "2026-07-01T13:38:35Z", 3003, comm=1.0),
        f("dd3", "DD", "SELL", 8, 134.53, "2026-07-01T13:38:44Z", 3003, comm=0.0)])
    L.rebuild_trades(c)
    dd = c.execute("SELECT * FROM trades WHERE ticker='DD'").fetchone()
    check("DD lot rescaled 326 → 108.6667, cost preserved (unit cost 140.52)",
          abs(dd["quantity_adj"] - 108.6667) < 1e-9 and abs(dd["price_entry_adj"] - 15269.84 / 108.6667) < 1e-6)
    check("DD: 108 sold at 134.53 against 140.52 cost — a LOSS, not a +28.5K phantom",
          abs(dd["realized_pnl"] - round(108 * 134.53 - 108 * 15269.84 / 108.6667 - 108 / 108.6667 - 1.0, 4)) < 1e-3,
          dd["realized_pnl"])
    check("DD: 0.6667 still open", abs(dd["qty_open"] - 0.6667) < 1e-9)
    check("DD positions view = 0.6667",
          c.execute("SELECT qty FROM positions WHERE ticker='DD'").fetchone()[0] == 0.6667)
    ea = c.execute("SELECT * FROM trade_outcomes WHERE ticker='EA'").fetchone()
    check("EA closed at $210 by the merger with exit_reason CORPORATE-ACTION",
          ea["price_exit"] == 210.0 and ea["exit_reason"] == "CORPORATE-ACTION")
    check("EA P&L = 79×(210−205.89) − $1 commission = 323.69 (IBKR FifoPnlRealized)",
          abs(ea["pnl_dollar"] - 323.69) < 0.005, ea["pnl_dollar"])
    check("EA gone from positions", c.execute("SELECT COUNT(*) FROM positions WHERE ticker='EA'").fetchone()[0] == 0)

    # ── g) append-only ────────────────────────────────────────────────
    print("\ng) append-only tables")
    for sql in ("UPDATE fills SET price = 0 WHERE exec_id = 'e1'",
                "DELETE FROM fills WHERE exec_id = 'e1'",
                "UPDATE position_events SET qty_change = 0 WHERE event_id = 'T-EA'",
                "DELETE FROM position_events WHERE event_id = 'T-EA'"):
        try:
            with c:
                c.execute(sql)
            ok = False
        except sqlite3.IntegrityError as exc:
            ok = "append-only" in str(exc)
        check(f"refused: {sql.split(' WHERE')[0]}", ok)

    # ── h) empty exit signals ─────────────────────────────────────────
    print("\nh) exit_signals '[]' needs a note")
    try:
        L.record_decision(timestamp="2026-09-07 14:00:00 UTC", ticker="ZZA", action="SELL",
                          quantity=1, execution_status="Filled",
                          exit=dict(exit_reason="MANUAL", exit_signals=[]), conn=c)
        ok = False
    except ValueError:
        ok = True
    check("record_decision refuses [] without exit_signals_note", ok)
    try:
        with c:
            c.execute("INSERT INTO exit_annotations (decision_id, ticker, exit_reason, exit_signals, "
                      "created_at) VALUES (?, 'ZZA', 'MANUAL', '[]', 'now')", (did_skip,))
        ok = False
    except sqlite3.IntegrityError:
        ok = True
    check("the table CHECK refuses it too", ok)

    # ── i) short ──────────────────────────────────────────────────────
    print("\ni) oversell opens a short; a later BUY covers it first")
    sell(c, "ZZS", 10, "2026-09-08 14:00:00 UTC", 4001, [
        f("s1", "ZZS", "SELL", 10, 50.0, "2026-09-08T14:00:01Z", 4001, comm=0.0)])
    buy(c, "ZZS", 15, "2026-09-09 14:00:00 UTC", 4002, [
        f("s2", "ZZS", "BUY", 15, 48.0, "2026-09-09T14:00:01Z", 4002, comm=0.0)])
    L.rebuild_trades(c)
    sh = c.execute("SELECT * FROM trades WHERE ticker='ZZS' AND action='SELL'").fetchone()
    lg = c.execute("SELECT * FROM trades WHERE ticker='ZZS' AND action='BUY'").fetchone()
    check("short trade closed by the cover, +$20", sh and sh["qty_open"] == 0 and sh["pnl_dollar"] == 20.0)
    check("the remaining 5 open a long trade", lg and lg["quantity"] == 5 and lg["qty_open"] == 5)
    check("positions view ZZS = 5", c.execute("SELECT qty FROM positions WHERE ticker='ZZS'").fetchone()[0] == 5)

    c.close()
    n, p = len(_results), sum(_results)
    print(f"\n{p}/{n} passed")
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
