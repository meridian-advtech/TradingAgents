#!/usr/bin/env python3
"""
kairos_reconcile_ghost_rows.py — close ML-ledger rows whose position was sold.

Until 2026-09-28 a sale closed at most two trade_outcomes rows (record_exit_outcome
took the oldest, find_open_trade -> write_trade_close the newest), so the third
and later lots of a multi-lot position never closed. Those "ghost" rows still
read timestamp_exit IS NULL although the position is gone; their outcomes are
missing from training, stats and the Arbiter.

DRY-RUN BY DEFAULT: prints what --apply would write and writes nothing.

  ghost           open ledger row, ticker has NO open lot in kairos.db holdings
  hidden ghost    open ledger row, ticker was re-entered — but the position the
                  row belonged to went flat after its entry. Not closed unless
                  --include-reentered: after the forward fix the next flat exit
                  of that ticker would otherwise close it at the WRONG price.

Match rule, per row: the FIRST filled SELL in kairos.db decisions after the
row's entry that left the ticker flat (no open holdings lot afterwards) —
exactly the event the forward fix closes every open row on. Price and time
come from that decision's execution_price / timestamp; exit_reason from the
matching position_exits_history row (else the decision rationale). Rows with
no such SELL are reported and left alone — never guessed. Trims between entry
and the matched sale are counted per row as an ambiguity flag: the ledger
cannot tell whether an earlier trim consumed that lot.

Each close is stamped through kairos_ml_outcomes._stamp_close (same fields as
a live close) with an exit_params_snapshot flagged reconstructed=True — the
params / axis weights in force at the exit, rebuilt from axis_weight_history
(kairos_backfill_evidence) — plus a ghost_reconcile block naming the decision.

--audit-trims (read-only) reports past partial sales whose newest-row guess
stamped a lot that, by FIFO, was still held, and LATE closes: rows that sat as
ghosts and were later closed by another position's sale, at its price.

    python kairos_reconcile_ghost_rows.py                       # dry run
    python kairos_reconcile_ghost_rows.py --audit-trims         # + trim audit
    python kairos_reconcile_ghost_rows.py --apply               # backs up first
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import kairos_ml_outcomes as M                              # noqa: E402
from kairos_backfill_evidence import (                      # noqa: E402
    _parse_dt, _current_param_values, _current_axis_weights,
    _load_approved_history, _value_in_force,
)

# A holdings sold_date and its SELL decision timestamp are written by the same
# exit path moments apart; ledger exits stamped by write_trade_close used now().
MATCH_TOLERANCE = timedelta(minutes=15)


def _dt(s):
    """Parse any Kairos timestamp, including holdings' '[RECON-merged]' suffix."""
    return _parse_dt((s or "").replace("[RECON-merged]", "").strip())


def _ro(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_state():
    k = _ro(M.KAIROS_DB_PATH)
    m = _ro(M.DB_PATH)
    try:
        ledger = [dict(r) for r in m.execute(
            "SELECT * FROM trade_outcomes ORDER BY timestamp_entry, rowid")]
        holdings = defaultdict(list)
        for r in k.execute("SELECT * FROM holdings"):
            holdings[r["ticker"]].append(dict(r))
        sells = defaultdict(list)
        for r in k.execute(
                "SELECT id, timestamp, ticker, quantity, execution_price, rationale "
                "FROM decisions WHERE action = 'SELL' AND execution_status = 'Filled' "
                "ORDER BY timestamp"):
            sells[r["ticker"]].append(dict(r))
        exits = defaultdict(list)
        for r in k.execute("SELECT * FROM position_exits_history"):
            exits[r["ticker"]].append(dict(r))
    finally:
        k.close()
        m.close()
    return ledger, holdings, sells, exits


def is_flat_after(lots, sale_dt):
    """True if no holdings lot of the ticker was still open just after sale_dt."""
    horizon = sale_dt + MATCH_TOLERANCE
    for lot in lots:
        e = _dt(lot["entry_date"])
        if e is None or e > sale_dt or (lot["quantity"] or 0) <= 1e-9:
            continue
        sd = _dt(lot["sold_date"]) if lot["sold_date"] else None
        if sd is None or sd > horizon:
            return False
    return True


def classify_sells(holdings, sells):
    """{ticker: [(sell, sale_dt, flat)]} for every filled SELL."""
    out = {}
    for t, ss in sells.items():
        out[t] = [(s, _dt(s["timestamp"]), is_flat_after(holdings.get(t, []), _dt(s["timestamp"])))
                  for s in ss if _dt(s["timestamp"])]
    return out


def exit_reason_for(exits, ticker, sale_dt, fallback):
    best = None
    for x in exits.get(ticker, []):
        d = _dt(x["exit_date"])
        if d and abs(d - sale_dt) <= MATCH_TOLERANCE:
            if best is None or abs(d - sale_dt) < abs(_dt(best["exit_date"]) - sale_dt):
                best = x
    return (best["exit_reason"] if best else None) or (fallback or "").strip() or "SELL (unspecified)"


def find_ghosts(ledger, holdings, sells_cls, exits):
    rows = []
    for r in ledger:
        if r["timestamp_exit"] is not None:
            continue
        t = r["ticker"]
        entry = _dt(r["timestamp_entry"])
        held_now = any(l["sold_date"] is None and (l["quantity"] or 0) > 1e-9
                       for l in holdings.get(t, []))
        after = [(s, d, f) for (s, d, f) in sells_cls.get(t, []) if entry and d > entry]
        flat_sale = next(((s, d) for (s, d, f) in after if f), None)
        if held_now and flat_sale is None:
            continue  # genuinely still held
        kind = "hidden" if held_now else "ghost"
        rec = {"row": r, "kind": kind, "match": None,
               "trims_before": 0, "why": None}
        if flat_sale is None:
            rec["why"] = (f"no flat-making filled SELL after entry "
                          f"({len(after)} filled SELL(s) after entry, none left it flat)")
        else:
            s, d = flat_sale
            rec["match"] = {"decision_id": s["id"], "sale_dt": d,
                            "timestamp": s["timestamp"],
                            "price": s["execution_price"],
                            "reason": exit_reason_for(exits, t, d, s["rationale"])}
            rec["trims_before"] = sum(1 for (_, d2, f) in after if d2 < d and not f)
            if not s["execution_price"]:
                rec["match"] = None
                rec["why"] = f"matched SELL #{s['id']} has no execution_price"
        rows.append(rec)
    return rows


def reconstructed_snapshot(exit_dt, cur_params, cur_weights, hist, decision_id, trims):
    params = {p: _value_in_force("param:" + p, exit_dt, hist, cur_params.get(p))
              for p in M.SNAPSHOT_PARAM_PATHS}
    weights = {a: _value_in_force(a, exit_dt, hist, w) for a, w in cur_weights.items()}
    snap = M.build_exit_params_snapshot(
        reconstructed=True, param_overrides=params, weight_overrides=weights,
        as_of=exit_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    snap["ghost_reconcile"] = {
        "decision_id": decision_id,
        "matched_by": "first flat-making filled SELL after entry",
        "trims_between": trims,
        "reconciled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return snap


def audit_trims(ledger, sells_cls):
    """Past trims where a ledger row closed while an OLDER row stayed open
    (the newest-row guess stamped a FIFO-still-held lot), or where the only
    open row closed although the position stayed open."""
    by_t = defaultdict(list)
    for r in ledger:
        by_t[r["ticker"]].append(r)
    findings = []
    for t, cls in sells_cls.items():
        for s, d, flat in cls:
            if flat:
                continue
            rows = by_t.get(t, [])
            closed_here = [r for r in rows if r["timestamp_exit"]
                           and abs(_dt(r["timestamp_exit"]) - d) <= MATCH_TOLERANCE]
            if not closed_here:
                continue
            open_after = [r for r in rows
                          if _dt(r["timestamp_entry"]) and _dt(r["timestamp_entry"]) < d
                          and (r["timestamp_exit"] is None or _dt(r["timestamp_exit"]) > d + MATCH_TOLERANCE)]
            for r in closed_here:
                older_open = [o for o in open_after if o["timestamp_entry"] < r["timestamp_entry"]]
                findings.append({
                    "ticker": t, "decision_id": s["id"], "sale": s["timestamp"],
                    "sold_qty": s["quantity"], "trade_id": r["trade_id"],
                    "entry": r["timestamp_entry"], "row_qty": r["quantity"],
                    "label": r["outcome_label"],
                    "snapshot": r["exit_params_snapshot"] is not None,
                    "older_rows_left_open": len(older_open),
                    "open_rows_after": len(open_after),
                })
    return findings


def audit_late_closes(ledger, sells_cls):
    """Closed rows whose position went flat EARLIER than the row's recorded
    exit: the row sat as a ghost and was later closed by a DIFFERENT
    position's sale, at that sale's price (the FIFO pick of the old
    record_exit_outcome). Their pnl / label are wrong, not just late."""
    out = []
    for r in ledger:
        if not r["timestamp_exit"]:
            continue
        entry, ex = _dt(r["timestamp_entry"]), _dt(r["timestamp_exit"])
        if not entry or not ex:
            continue
        flat = next(((s, d) for (s, d, f) in sells_cls.get(r["ticker"], [])
                     if f and d > entry), None)
        if flat and ex > flat[1] + MATCH_TOLERANCE:
            s, d = flat
            out.append({"ticker": r["ticker"], "trade_id": r["trade_id"],
                        "entry": r["timestamp_entry"], "qty": r["quantity"],
                        "recorded_exit": r["timestamp_exit"], "recorded_price": r["price_exit"],
                        "flat_sale": s["timestamp"], "flat_price": s["execution_price"],
                        "decision_id": s["id"], "label": r["outcome_label"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--apply", action="store_true", help="write (backs up first)")
    ap.add_argument("--include-reentered", action="store_true",
                    help="also close hidden ghosts on re-entered tickers")
    ap.add_argument("--audit-trims", action="store_true",
                    help="report past trims that stamped a still-held lot")
    args = ap.parse_args()

    print(f"kairos.db:             {M.KAIROS_DB_PATH}  (read-only)")
    print(f"kairos_ml_outcomes.db: {M.DB_PATH}  ({'WRITE' if args.apply else 'read-only'})")
    ledger, holdings, sells, exits = load_state()
    sells_cls = classify_sells(holdings, sells)
    ghosts = find_ghosts(ledger, holdings, sells_cls, exits)

    cur_params, cur_weights = _current_param_values(), _current_axis_weights()
    hist = _load_approved_history()

    todo = [g for g in ghosts if g["match"]
            and (g["kind"] == "ghost" or args.include_reentered)]
    for kind, title in (("ghost", "GHOSTS — ticker has no open lot"),
                        ("hidden", "HIDDEN GHOSTS — ticker re-entered after this row's position went flat")):
        grp = [g for g in ghosts if g["kind"] == kind]
        print(f"\n=== {title}: {len(grp)} row(s), "
              f"{len({g['row']['ticker'] for g in grp})} ticker(s) ===")
        per = defaultdict(lambda: [0, 0])
        for g in sorted(grp, key=lambda g: (g["row"]["ticker"], g["row"]["timestamp_entry"])):
            r = g["row"]
            per[r["ticker"]][0 if g["match"] else 1] += 1
            head = (f"  {r['ticker']:<6} {r['trade_id'][:8]}  entry {r['timestamp_entry'][:16]}"
                    f"  {r['quantity']:>6} @ {r['price_entry']:<9}")
            if g["match"]:
                mt = g["match"]
                pnl = (mt["price"] - r["price_entry"]) * (r["quantity"] or 0)
                pct = (mt["price"] - r["price_entry"]) / r["price_entry"] * 100 if r["price_entry"] else 0
                lbl = "SCRATCH" if abs(pnl) < 0.01 else ("WIN" if pnl > 0 else "LOSS")
                print(f"{head} -> SELL #{mt['decision_id']} {mt['timestamp'][:16]} @ {mt['price']}"
                      f"  pnl {pnl:+,.2f} ({pct:+.2f}%) {lbl}"
                      + (f"  [!{g['trims_before']} trim(s) between]" if g["trims_before"] else "")
                      + f"\n{'':10}reason: {mt['reason'][:90]}")
            else:
                print(f"{head} -> UNMATCHED: {g['why']}")
        if per:
            print("  per ticker (matched / unmatched): "
                  + "  ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(per.items())))

    if args.audit_trims:
        f = audit_trims(ledger, sells_cls)
        bad = [x for x in f if x["older_rows_left_open"] or x["open_rows_after"] == 0]
        print(f"\n=== TRIM AUDIT: {len(f)} ledger row(s) closed at a partial sale; "
              f"{len(bad)} stamped a lot FIFO says was still held ===")
        for x in f:
            flag = ("NEWER-THAN-OPEN-ROW" if x["older_rows_left_open"]
                    else "ONLY-ROW-CLOSED-POSITION-OPEN" if x["open_rows_after"] == 0 else "ok (FIFO)")
            print(f"  {x['ticker']:<6} SELL #{x['decision_id']} {x['sale'][:16]} qty {x['sold_qty']}"
                  f" -> {x['trade_id'][:8]} (entry {x['entry'][:10]}, qty {x['row_qty']}, "
                  f"label {x['label']}, snap {'Y' if x['snapshot'] else 'N'})  {flag}")

    if args.audit_trims:
        late = audit_late_closes(ledger, sells_cls)
        print(f"\n=== LATE CLOSES: {len(late)} closed row(s) recorded AFTER their position "
              f"had already gone flat (closed later at another sale's price; not touched) ===")
        for x in late:
            print(f"  {x['ticker']:<6} {x['trade_id'][:8]} entry {x['entry'][:10]} qty {x['qty']}: "
                  f"recorded exit {x['recorded_exit'][:10]} @ {x['recorded_price']}  vs  "
                  f"flat SELL #{x['decision_id']} {x['flat_sale'][:10]} @ {x['flat_price']}  "
                  f"(label {x['label']})")

    print(f"\n{'Closing' if args.apply else 'Would close'} {len(todo)} row(s)"
          + ("" if args.include_reentered else " (hidden ghosts excluded; --include-reentered to add)")
          + f"; {sum(1 for g in ghosts if not g['match'])} unmatched left alone.")
    if not args.apply or not todo:
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = M.DB_PATH.replace(".db", f"_PRE_GHOST_RECON_{stamp}.db")
    src, dst = sqlite3.connect(M.DB_PATH), sqlite3.connect(backup)
    src.backup(dst)
    dst.close()
    src.close()
    print(f"Backup: {backup}")

    conn = M.get_connection()
    try:
        for g in todo:
            row = conn.execute("SELECT * FROM trade_outcomes WHERE trade_id = ? "
                               "AND timestamp_exit IS NULL", (g["row"]["trade_id"],)).fetchone()
            if row is None:
                continue  # closed since the scan
            mt = g["match"]
            M._stamp_close(conn, row, mt["price"],
                           mt["sale_dt"].strftime("%Y-%m-%dT%H:%M:%SZ"), mt["reason"],
                           snapshot=reconstructed_snapshot(mt["sale_dt"], cur_params, cur_weights,
                                                           hist, mt["decision_id"], g["trims_before"]))
        conn.commit()
    finally:
        conn.close()
    print(f"Closed {len(todo)} row(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
