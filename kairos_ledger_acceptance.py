#!/usr/bin/env python3
"""
kairos_ledger_acceptance.py — acceptance tests for the fills-ledger migration.

Runs every Phase 1 acceptance test against a migrated kairos.db (the rehearsal
copy in Phase 1). The database is opened READ-ONLY; the append-only trigger
test runs on a throwaway backup copy. Broker reads use a clientId Kairos never
uses (918), readonly=True.

    python kairos_ledger_acceptance.py --db ~/kairos-rehearsal/kairos.db

Readers and selftests run as subprocesses from THIS checkout, whose kairos.db
must resolve to --db (the script refuses otherwise), with SLACK_BOT_TOKEN set to
a dummy so nothing can post.
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal as D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kairos_ledger as L  # noqa: E402

PY = sys.executable
RESULTS = []
REQUIRED_OPEN = ("ACHR", "DD", "EXPE", "JNJ", "KO", "Q")
LEGACY_VIEW_TABLES = (("holdings", "holdings_legacy"), ("trade_outcomes", "trade_outcomes_legacy"))


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


def ro(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def affinity(decl: str) -> str:
    """SQLite column affinity rules (§3.1 of the datatype doc)."""
    t = (decl or "").upper()
    if "INT" in t:
        return "INTEGER"
    if any(k in t for k in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in t or not t:
        return "BLOB"
    if any(k in t for k in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


# ── independent FIFO (Decimal) — shares no code with rebuild_trades ──

def independent_pnl(c) -> dict:
    """{trade_id: realized P&L} recomputed from raw fills + position_events."""
    comm = {r[0]: D(str(r[1])) if r[1] is not None else D(0) for r in c.execute(
        "SELECT f.exec_id, COALESCE(f.commission, fc.commission) FROM fills f "
        "LEFT JOIN fill_commissions fc ON fc.exec_id = f.exec_id")}
    key_to_tid = {}
    for r in c.execute("SELECT trade_id, order_key, action FROM trades WHERE order_key IS NOT NULL"):
        key_to_tid[(r["order_key"], r["action"])] = r["trade_id"]

    def okey(f):
        if f["perm_id"]:
            return f"P{f['perm_id']}"
        return f"O{f['order_id']}" if f["order_id"] else f"X{f['exec_id']}"

    items = [(f["executed_at"], 1, f["exec_id"], dict(f)) for f in c.execute("SELECT * FROM fills")]
    items += [(e["effective_at"], 0, e["event_id"], dict(e)) for e in c.execute("SELECT * FROM position_events")]
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    longs, shorts = defaultdict(list), defaultdict(list)   # [tid, qty, unit_cost|None, unit_comm]
    pnl, known = defaultdict(D), defaultdict(lambda: True)
    done_split = set()

    def take(book, tk, qty, px, fee, total, sign):
        lots = book[tk]
        while qty > D("0.000001") and lots:
            lot = lots[0]
            m = min(qty, lot[1])
            if lot[2] is None:
                known[lot[0]] = False
            else:
                pnl[lot[0]] += sign * (m * px - m * lot[2]) - m * lot[3] - fee * m / total
            lot[1] -= m
            qty -= m
            if lot[1] <= D("0.000001"):
                lots.pop(0)
        return qty

    for at, pri, ref, x in items:
        if pri == 1:
            tk, q, px, fee = x["ticker"], D(str(x["quantity"])), D(str(x["price"])), comm[x["exec_id"]]
            if x["side"] == "BUY":
                rest = take(shorts, tk, q, px, fee, q, -1)
                if rest > 0:
                    tid = key_to_tid.get((okey(x), "BUY"))
                    lot = next((l for l in longs[tk] if l[0] == tid), None)
                    if lot:
                        tot = lot[1] + rest
                        lot[2] = (lot[2] * lot[1] + px * rest) / tot
                        lot[3] = (lot[3] * lot[1] + fee * rest / q) / tot
                        lot[1] = tot
                    else:
                        longs[tk].append([tid, rest, px, fee / q])
            else:
                rest = take(longs, tk, q, px, fee, q, 1)
                if rest > 0:
                    tid = key_to_tid.get((okey(x), "SELL"))
                    lot = next((l for l in shorts[tk] if l[0] == tid), None)
                    if lot:
                        tot = lot[1] + rest
                        lot[2] = (lot[2] * lot[1] + px * rest) / tot
                        lot[1] = tot
                    else:
                        shorts[tk].append([tid, rest, px, fee / q])
            continue
        et, tk = x["event_type"], x["ticker"]
        if et == "opening_balance":
            longs[tk].append([x["event_id"], D(str(x["qty_change"])), None, D(0)])
        elif et == "split" and (tk, at) not in done_split:
            done_split.add((tk, at))
            rows = list(c.execute("SELECT qty_change FROM position_events WHERE event_type='split' "
                                  "AND ticker=? AND effective_at=?", (tk, at)))
            new = sum(D(str(r[0])) for r in rows if r[0] > 0)
            old = -sum(D(str(r[0])) for r in rows if r[0] < 0)
            for lot in longs[tk]:
                nq = lot[1] * new / old
                if lot[2] is not None:
                    lot[2] = lot[2] * lot[1] / nq
                lot[3] = lot[3] * lot[1] / nq
                lot[1] = nq
        elif et == "symbol_change":
            longs[x["new_ticker"]] += longs.pop(tk, [])
            shorts[x["new_ticker"]] += shorts.pop(tk, [])
        elif et == "merger_cash":
            q = -D(str(x["qty_change"]))
            take(longs, tk, q, D(str(x["cash_per_share"])), D(0), q, 1)
    return {t: (v if known[t] else None) for t, v in pnl.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--skip-readers", action="store_true")
    args = ap.parse_args()
    db = os.path.realpath(os.path.expanduser(args.db))
    here_db = os.path.realpath(os.path.join(HERE, "kairos.db"))
    if not args.skip_readers and here_db != db:
        sys.exit(f"REFUSED: {HERE}/kairos.db resolves to {here_db}, not {db}")
    c = ro(db)

    # ── 1. positions == IBKR; reqExecutions all in fills ───────────────
    print("\n1. Positions view vs IBKR (clientId 918, readonly)")
    from ib_insync import IB, ExecutionFilter
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=918, readonly=True, timeout=20)
    try:
        execs = L.normalize_fills(ib.reqExecutions(ExecutionFilter()))
        ib.reqPositions()
        ib.sleep(1)
        broker = L.broker_positions(ib)
    finally:
        ib.disconnect()
    bad = L.compare_positions(c, broker)
    check(f"positions view == IBKR for all {len(broker)} STK positions (zero mismatches)",
          not bad and broker, str(bad) if bad else "")
    dd = c.execute("SELECT qty FROM positions WHERE ticker='DD'").fetchone()
    check("DD fractional 0.6667 matches", dd and dd[0] == 0.6667 == broker.get("DD"))

    print("\n2. Fills integrity")
    have = {r[0] for r in c.execute("SELECT exec_id FROM fills")}
    missing = [e["exec_id"] for e in execs if e["exec_id"] not in have]
    check(f"every reqExecutions() exec_id ({len(execs)}) is in fills", not missing, str(missing))
    n, nd = c.execute("SELECT COUNT(*), COUNT(DISTINCT exec_id) FROM fills").fetchone()
    check(f"no duplicate exec_ids ({n} fills)", n == nd)
    bad_ts = c.execute("SELECT COUNT(*) FROM fills WHERE executed_at NOT GLOB "
                       "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'"
                       ).fetchone()[0]
    check("one timestamp format (UTC 'YYYY-MM-DDTHH:MM:SSZ') on every fill", bad_ts == 0)

    # ── 3. closed trades ─────────────────────────────────────────────
    print("\n3. Closed trades")
    closed = c.execute("SELECT trade_id, origin, pnl_dollar, outcome_label FROM trades "
                       "WHERE timestamp_exit IS NOT NULL").fetchall()
    unknown_basis = [r["trade_id"] for r in closed if r["origin"] == "opening_balance"]
    unlabelled = [r["trade_id"] for r in closed
                  if r["outcome_label"] is None and r["origin"] != "opening_balance"]
    check(f"every closed trade with a known cost basis has outcome_label ({len(closed)} closed)",
          not unlabelled, str(unlabelled[:10]))
    check("… the only unlabelled closes are the opening balances (basis unknown by design)",
          sorted(unknown_basis) == ["MANUAL-OPEN-AAPL-20260401", "MANUAL-OPEN-CI-20260401"],
          str(unknown_basis))
    ind = independent_pnl(c)
    off = []
    for r in closed:
        if r["origin"] == "opening_balance":
            continue
        x = ind.get(r["trade_id"])
        if x is None or r["pnl_dollar"] is None or abs(D(str(r["pnl_dollar"])) - x) >= D("0.005"):
            off.append((r["trade_id"], r["pnl_dollar"], None if x is None else float(round(x, 4))))
    check(f"P&L of all {len(closed) - len(unknown_basis)} closed trades recomputes from fills "
          f"to the cent (independent Decimal FIFO)", not off, str(off[:10]))
    tot_ind = sum(v for v in ind.values() if v is not None)
    tot_led = c.execute("SELECT SUM(realized_pnl) FROM trades").fetchone()[0]
    check("total realized P&L (open + closed trades) agrees to the cent",
          abs(D(str(tot_led)) - tot_ind) < D("0.01"), f"ledger {tot_led:.2f} vs independent {tot_ind:.2f}")

    print("\n4. Required open trades")
    for tk in REQUIRED_OPEN:
        n = c.execute("SELECT COUNT(*) FROM trade_outcomes WHERE ticker = ? AND timestamp_exit IS NULL",
                      (tk,)).fetchone()[0]
        check(f"{tk} has an open trade", n >= 1, f"{n} open")

    print("\n5. Append-only (on a throwaway copy)")
    tmp = tempfile.mkdtemp(prefix="kairos_accept_")
    cp = os.path.join(tmp, "k.db")
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dst = sqlite3.connect(cp)
    src.backup(dst)
    src.close()
    for sql in ("UPDATE fills SET price = price + 1", "DELETE FROM fills",
                "UPDATE position_events SET qty_change = 0", "DELETE FROM position_events"):
        try:
            dst.execute(sql)
            dst.commit()
            ok = False
        except sqlite3.IntegrityError as exc:
            ok = "append-only" in str(exc)
        check(f"refused on the migrated DB: {sql}", ok)
    for t in ("holdings_legacy", "trade_outcomes_legacy", "position_exits_history_legacy"):
        try:
            dst.execute(f"DELETE FROM {t}")
            dst.commit()
            ok = False
        except sqlite3.IntegrityError:
            ok = True
        check(f"{t} is read-only", ok)
    dst.close()
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n6. View schemas reproduce the replaced tables")
    for view, legacy in LEGACY_VIEW_TABLES:
        a = [(r[1], affinity(r[2])) for r in c.execute(f"PRAGMA table_info({view})")]
        b = [(r[1], affinity(r[2])) for r in c.execute(f"PRAGMA table_info({legacy})")]
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        check(f"{view}: {len(a)} columns, same names / order / affinity as {legacy} ({len(b)})",
              len(a) == len(b) and not diff, str(diff))
    c.close()

    env = dict(os.environ, SLACK_BOT_TOKEN="xoxb-rehearsal-disabled")

    def run(label, cmd, must=None, timeout=900):
        try:
            p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout)
            out = p.stdout + p.stderr
            ok = p.returncode == 0 and (must is None or re.search(must, out))
            tail = "" if ok else out[-1500:]
        except subprocess.TimeoutExpired:
            ok, tail = False, "timeout"
        check(label, ok, tail)
        return ok

    print("\n7. Simulated scenarios")
    run("kairos_selftest_ledger.py (2-exec BUY, FIFO trim, late fill, duplicate, "
        "unannotated BUY, DD split, EA cash-out, append-only, shorts)",
        [PY, "kairos_selftest_ledger.py"], must=r"(\d+)/\1 passed")
    run("kairos_selftest_tradepath.py (real call sites: log_execution BUY/SELL, _log_sell, "
        "broker check late fill, failed write alerts without raising)",
        [PY, "kairos_selftest_tradepath.py"], must=r"(\d+)/\1 passed")

    if not args.skip_readers:
        print("\n8. Readers, unchanged, against the migrated DB")
        run("dashboard renders (kairos_dashboard.py --no-ibkr)",
            [PY, "kairos_dashboard.py", "--no-ibkr"], must=r"Dashboard written")
        run("Arbiter daily --dry-run", [PY, "kairos_arbiter.py", "--mode", "daily", "--dry-run"],
            must=r"DRY RUN")
        run("council prompt build — SECTION 4b + 5b present (kairos_reason.write_prompt)",
            [PY, "-c", (
                "import json,kairos_reason as R\n"
                "sl=[c['ticker'] for c in json.load(open('kairos_ml_result.json')).get('candidates',[])][:8]\n"
                "p=R.write_prompt('', {}, shortlist=sl)\n"
                "t=open(R.PROMPT_FILE).read() if hasattr(R,'PROMPT_FILE') else open('kairos_prompt.txt').read()\n"
                "assert 'SECTION 4b' in t, '4b missing'\nassert 'SECTION 5b' in t, '5b missing'\n"
                "print('SECTIONS-OK', len(t))")], must=r"SECTIONS-OK")
        run("kairos_ml.train_model(force_retrain=True) (model written to a temp path)",
            [PY, "-c", (
                "import tempfile,os,kairos_ml as K\n"
                "K.MODEL_PATH=os.path.join(tempfile.mkdtemp(),'m.pkl')\n"
                "r=K.train_model(force_retrain=True)\n"
                "print('TRAINED', r.get('trade_count'), r.get('accuracy'))\n"
                "assert r.get('model') is not None")], must=r"TRAINED")
        run("kairos_scorecard", [PY, "kairos_scorecard.py"], must=r"MECHANISM SCORECARD")
        run("kairos_axis_weights --compute-params --dry-run",
            [PY, "kairos_axis_weights.py", "--compute-params", "--dry-run"])

        print("\n9. Selftests")
        run("selftest: learning", [PY, "kairos_selftest_learning.py"], must=r"\b0 failed")
        run("selftest: multilot", [PY, "kairos_selftest_multilot.py"], must=r"(\d+)/\1 checks passed")
        run("selftest: healthcheck (79/79)", [PY, "kairos_selftest_healthcheck.py"],
            must=r"79/79 assertions passed")
        run("selftest: watch", [PY, "kairos_selftest_watch.py"], must=r"\b0 failed")

    print("\n10. Static proof")
    writers, ml_refs, fixtures = [], [], []
    rx_w = re.compile(r"(INSERT\s+(OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM|REPLACE\s+INTO)\s+\"?"
                      r"(holdings|trade_outcomes)\b(?!_legacy)", re.I)
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".py"):
            continue
        src = open(os.path.join(HERE, f)).read()
        if f.startswith("kairos_") and "selftest" not in f:
            for m in rx_w.finditer(src):
                # A module-embedded _selftest that builds its OWN temp table of
                # the same name (kairos_axis_weights._selftest) is a fixture,
                # not a writer of kairos.db — reported, not failed.
                fn = re.findall(r"^def (\w+)", src[:m.start()], re.M)
                if fn and fn[-1].startswith("_selftest"):
                    fixtures.append(f"{f}:{fn[-1]}: {m.group(0)}")
                    continue
                writers.append(f"{f}: {m.group(0)}")
        if "kairos_ml_outcomes.db" in src and f not in (
                "kairos_migrate_fills.py", "kairos_ledger.py", "kairos_ml_outcomes.py",
                "kairos_ledger_acceptance.py"):
            for i, line in enumerate(src.splitlines(), 1):
                if "kairos_ml_outcomes.db" in line and not line.strip().startswith("#"):
                    ml_refs.append(f"{f}:{i}: {line.strip()[:90]}")
    check("no code writes holdings / trade_outcomes (they are views)", not writers,
          "\n".join(writers) or ("temp-DB selftest fixtures (not kairos.db): " + "; ".join(fixtures)
                                 if fixtures else ""))
    check("nothing opens kairos_ml_outcomes.db (only the migration script names it)",
          not ml_refs, "\n".join(ml_refs))

    n_fail = sum(1 for r in RESULTS if not r[1])
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} acceptance checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
