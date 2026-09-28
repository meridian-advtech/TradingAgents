#!/usr/bin/env python3
"""Trade-path integration selftest — the REAL call sites, on a throwaway copy.

Drives kairos_execute.log_execution (BUY and SELL) and kairos_stoploss._log_sell
with synthetic execution dicts shaped exactly like execute_order /
_place_market_sell return them (incl. the new perm_id + fills keys), against a
backup COPY of kairos.db in a temp dir. Asserts the decision, annotations,
fills, derived trade and thesis row land — and that a failing ledger write
alerts instead of raising into the order path.

Run: python3 kairos_selftest_tradepath.py   (needs a migrated kairos.db)
"""

import os
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kairos_ledger as K        # noqa: E402
import kairos_log_db as LDB      # noqa: E402
import kairos_ml_outcomes as M   # noqa: E402
import kairos_ml_thesis as T     # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="kairos_tradepath_")
    db = os.path.join(tmp, "kairos.db")
    s = sqlite3.connect(f"file:{K.DB_PATH}?mode=ro", uri=True)
    d = sqlite3.connect(db)
    s.backup(d)
    s.close()
    d.close()
    K.DB_PATH = LDB.DB_PATH = M.DB_PATH = M.KAIROS_DB_PATH = T.DB_PATH = db
    import kairos_execute as E
    import kairos_stoploss as S
    E.LOG_FILE = os.path.join(tmp, "decisions.log")
    posted = []
    import kairos_alerts
    kairos_alerts.post_message = lambda ch, text, *a, **k: posted.append((ch, text)) or True
    kairos_alerts.alert_trade_executed = lambda *a, **k: None
    import kairos_reason
    kairos_reason.record_ledger_entry = lambda *a, **k: None

    c = K.get_connection(db)
    n_fills0 = c.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    c.close()

    print("\nlog_execution — BUY filled in two executions")
    trade = {"ticker": "ZZTP", "action": "BUY", "quantity": 30, "rationale": "selftest buy",
             "sector": "tech", "conviction": 8, "_confluence": {"score": 3, "signals": ["HOT-INSIDER"]},
             "predicted_direction": "UP", "predicted_return_pct": 8.0, "predicted_timeframe_days": 10,
             "invalidation_conditions": "closes below $90"}
    execution = {"status": "Filled", "order_id": 4242, "perm_id": 99000001, "fill_price": 100.0,
                 "commission": 1.0, "new_position": {"quantity": 30.0, "avg_cost": 100.33},
                 "fills": [dict(exec_id="TP-1", perm_id=99000001, ticker="ZZTP", side="BUY",
                                quantity=20, price=100.0, commission=0.5,
                                executed_at="2030-03-01T15:00:00Z"),
                           dict(exec_id="TP-2", perm_id=99000001, ticker="ZZTP", side="BUY",
                                quantity=10, price=101.0, commission=0.5,
                                executed_at="2030-03-01T15:00:02Z")]}
    E.log_execution({"tickers_evaluated": ["ZZTP"]}, trade, execution)
    c = K.get_connection(db)
    dec = c.execute("SELECT * FROM decisions WHERE ticker='ZZTP' AND action='BUY'").fetchone()
    check("decision written with trade_id, order_id and perm_id",
          dec and dec["trade_id"] and dec["order_id"] == 4242 and dec["perm_id"] == 99000001)
    ea = c.execute("SELECT * FROM entry_annotations WHERE trade_id=?", (dec["trade_id"],)).fetchone()
    check("entry_annotations in the same transaction (signals, source, conviction)",
          ea and ea["signals"] == '["HOT-INSIDER"]' and ea["signal_attribution_source"] == "confluence"
          and ea["conviction"] == 8)
    fl = c.execute("SELECT * FROM fills WHERE decision_id=?", (dec["id"],)).fetchall()
    check("BOTH executions recorded as fills, linked 'live'",
          len(fl) == 2 and all(f["link_method"] == "live" and f["source"] == "live" for f in fl))
    tr = c.execute("SELECT * FROM trades WHERE trade_id=?", (dec["trade_id"],)).fetchone()
    check("one trade at the qty-weighted price (100.3333)",
          tr and tr["quantity"] == 30 and abs(tr["price_entry"] - 100.333333) < 1e-5)
    tp = c.execute("SELECT * FROM thesis_predictions WHERE decision_id=?", (dec["trade_id"],)).fetchone()
    check("thesis prediction keyed by the trade_id", tp is not None)
    check("positions view shows ZZTP 30",
          c.execute("SELECT qty FROM positions WHERE ticker='ZZTP'").fetchone()[0] == 30)
    c.close()

    print("\n_log_sell — exit-engine style SELL, partially filled now")
    ex2 = {"status": "Filled", "order_id": 4243, "perm_id": 99000002, "fill_price": 110.0,
           "commission": 0.4,
           "fills": [dict(exec_id="TP-3", perm_id=99000002, ticker="ZZTP", side="SELL",
                          quantity=20, price=110.0, commission=0.4,
                          executed_at="2030-03-05T15:00:00Z")]}
    S._log_sell("ZZTP", 30, 100.33, 110.0, 4, "TRAILING-STOP: selftest", ex2,
                exit_signals=["HOT-REVERSION"])
    c = K.get_connection(db)
    sd = c.execute("SELECT * FROM decisions WHERE ticker='ZZTP' AND action='SELL'").fetchone()
    xa = c.execute("SELECT * FROM exit_annotations WHERE decision_id=?", (sd["id"],)).fetchone()
    check("SELL decision + exit_annotations (reason, signals ∪ entry signals, snapshot)",
          xa and xa["exit_reason"] == "TRAILING-STOP: selftest"
          and set(__import__("json").loads(xa["exit_signals"])) == {"HOT-REVERSION", "HOT-INSIDER"}
          and xa["exit_params_snapshot"])
    check("only the 20 FILLED shares closed; 10 still open (nothing closed by assumption)",
          c.execute("SELECT qty FROM positions WHERE ticker='ZZTP'").fetchone()[0] == 10)
    o = c.execute("SELECT timestamp_exit FROM trade_outcomes WHERE trade_id=?", (dec["trade_id"],)).fetchone()
    check("the trade stays open until the rest fills", o["timestamp_exit"] is None)
    c.close()

    print("\nbroker check picks up the late remainder")
    from types import SimpleNamespace
    from datetime import datetime, timezone

    class FakeIB:
        def reqExecutions(self, _f=None):
            return [SimpleNamespace(
                contract=SimpleNamespace(secType="STK", symbol="ZZTP"),
                execution=SimpleNamespace(execId="TP-4", permId=99000002, side="SLD", shares=10,
                                          price=109.5, time=datetime(2030, 3, 5, 16, tzinfo=timezone.utc)),
                commissionReport=SimpleNamespace(execId="TP-4", commission=0.2))]

        def reqPositions(self):
            pass

        def sleep(self, _s):
            pass

        def positions(self):
            c2 = K.get_connection(db)
            rows = [SimpleNamespace(contract=SimpleNamespace(secType="STK", symbol=t), position=q)
                    for t, q in c2.execute("SELECT ticker, qty FROM positions WHERE ticker <> 'ZZTP'")]
            c2.close()
            return rows
    r = K.broker_check(ib=FakeIB(), post=False)
    c = K.get_connection(db)
    lf = c.execute("SELECT decision_id, link_method, commission FROM fills WHERE exec_id='TP-4'").fetchone()
    check("late SELL fill recorded, linked to the SELL decision by perm_id, commission kept",
          lf and lf["decision_id"] == sd["id"] and lf["link_method"] == "perm_id" and lf["commission"] == 0.2)
    o = c.execute("SELECT * FROM trade_outcomes WHERE trade_id=?", (dec["trade_id"],)).fetchone()
    check("trade now closed: labelled, exit_reason from the SELL, snapshot present",
          o["timestamp_exit"] and o["outcome_label"] == "WIN"
          and o["exit_reason"] == "TRAILING-STOP: selftest" and o["exit_params_snapshot"])
    check("positions match the (fake) broker — ZZTP flat", r["mismatches"] == [], r["mismatches"])
    c.close()

    print("\na failed ledger write alerts, never raises")
    real = K.record_decision
    K.record_decision = lambda **kw: (_ for _ in ()).throw(RuntimeError("disk full (simulated)"))
    try:
        E.log_execution({}, dict(trade, ticker="ZZTQ"), dict(execution, fills=[]))
        raised = False
    except Exception:
        raised = True
    K.record_decision = real
    check("log_execution did not raise", not raised)
    check("an #alerts post named the failure",
          any(ch == "alerts" and "record_decision" in t and "disk full" in t for ch, t in posted))

    c = K.get_connection(db)
    check("no fill outside the test tickers was written",
          c.execute("SELECT COUNT(*) FROM fills WHERE ticker NOT IN ('ZZTP')").fetchone()[0] == n_fills0)
    c.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
