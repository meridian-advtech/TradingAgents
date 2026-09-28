#!/usr/bin/env python3
"""
kairos_trade_diff.py — old (legacy trade_outcomes) vs new (fills ledger) trades.

Read-only. Compares trade_outcomes_legacy (the pre-migration ML ledger, copied
into kairos.db by kairos_migrate_fills) with the new trade_outcomes view, per
trade_id, and puts every difference in ONE primary category (precedence order):

  corporate action   the trade touches a split / symbol change / cash merger
  opening balance    built from a manual opening balance (basis unknown)
  missing row        exists only in the new ledger — pre-Kairos-era fills,
                     orders that filled after their decision was marked
                     Expired, and short positions (long-only violations)
  ghost row          legacy row whose state disagrees with the fills: open in
                     one ledger and closed in the other, or voided
  quantity mismatch  legacy qty ≠ the shares the order actually filled
  wrong-price close  exit price differs by > $0.01 (legacy closed at a price
                     other than the matched exit fills)
  wrong-price entry  entry price differs by > $0.01 (legacy took the first
                     fill / an estimate instead of the qty-weighted fills)
  commission         P&L differs only by the commissions the new ledger nets
  rounding           |ΔP&L| < $0.01 and nothing else differs
  identical          no difference

    python kairos_trade_diff.py --db ~/kairos-rehearsal/kairos.db \\
        --csv ~/kairos-rehearsal/trade_diff.csv --summary ~/kairos-rehearsal/trade_diff_summary.md
"""

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict

CORP_TICKERS = {"DD", "EA", "SATS", "ECHO"}
KAIROS_ERA_START = "2026-05-18"
TRUSTED = ("explicit", "confluence", "decision_record")


def sigs(raw):
    try:
        return [str(s) for s in (json.loads(raw) if raw else []) if s]
    except (TypeError, json.JSONDecodeError):
        return []


def f(x):
    return None if x is None else float(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    old = {r["trade_id"]: dict(r) for r in c.execute("SELECT * FROM trade_outcomes_legacy")}
    new = {r["trade_id"]: dict(r) for r in c.execute(
        "SELECT o.*, t.origin, t.quantity_adj, t.qty_open, t.realized_pnl, t.entry_commission, "
        "t.exit_commission, t.commission_complete, t.decision_id AS entry_decision_id, "
        "t.ticker_current FROM trade_outcomes o JOIN trades t ON t.trade_id = o.trade_id")}

    rows = []
    for tid in sorted(set(old) | set(new), key=lambda t: ((new.get(t) or old.get(t))["timestamp_entry"], t)):
        o, n = old.get(tid), new.get(tid)
        cat, detail = "identical", ""
        if n and (n["ticker"] in CORP_TICKERS or (n["ticker_current"] or "") in CORP_TICKERS):
            cat = "corporate action"
            detail = {"DD": "1-for-3 reverse split 2026-06-23",
                      "EA": "cash merger $210/sh 2026-08-04",
                      "SATS": "SATS→ECHO symbol change (same conid)"}.get(n["ticker"], "")
        elif n and n["origin"] == "opening_balance":
            cat, detail = "opening balance", "pre-2026-04-01 shares, basis unknown"
        elif o is None:
            cat = "missing row"
            if n["action"] == "SELL":
                detail = "short position (long-only violation) — not in the legacy ledger"
            elif n["timestamp_entry"][:10] < KAIROS_ERA_START:
                detail = "pre-Kairos-era fills (before 2026-05-18, no decisions exist)"
            elif n["entry_decision_id"] is None:
                detail = ("Kairos-era BUY order with no same-session decision (e.g. a GTC "
                          "order that filled on a later day) — unlinked")
            else:
                st = c.execute("SELECT execution_status FROM decisions WHERE id = ?",
                               (n["entry_decision_id"],)).fetchone()
                detail = (f"BUY order filled; no legacy row (its decision is "
                          f"'{st[0] if st else '?'}')")
        elif n is None:
            cat, detail = "ghost row", "legacy row with no IBKR order behind it"
        else:
            o_closed, n_closed = o["timestamp_exit"] is not None, n["timestamp_exit"] is not None
            void = (o["exit_reason"] or "").startswith("VOID") or "[RECON" in (o["timestamp_exit"] or "")
            d_pnl = (f(n["pnl_dollar"]) or 0) - (f(o["pnl_dollar"]) or 0)
            comm = (f(n["entry_commission"]) or 0) + (f(n["exit_commission"]) or 0)
            if o_closed != n_closed or void or (o_closed and o["outcome_label"] is None):
                cat = "ghost row"
                detail = (f"legacy {'closed' if o_closed else 'open'}"
                          f"{' (void/recon)' if void else ''}, fills say "
                          f"{'closed' if n_closed else 'open'}")
            elif abs((f(o["quantity"]) or 0) - (f(n["quantity_adj"]) or 0)) > 1e-6:
                cat, detail = "quantity mismatch", f"legacy {o['quantity']} vs filled {n['quantity_adj']}"
            elif n_closed and abs((f(o["price_exit"]) or 0) - (f(n["price_exit"]) or 0)) > 0.01:
                cat, detail = "wrong-price close", f"exit {o['price_exit']} → {n['price_exit']:.4f}"
                lag = (o["timestamp_exit"] or "")[:10], (n["timestamp_exit"] or "")[:10]
                if lag[0] != lag[1]:
                    detail += f"; legacy closed {lag[0]}, fills sold it {lag[1]}"
            elif abs((f(o["price_entry"]) or 0) - (f(n["price_entry"]) or 0)) > 0.01:
                cat, detail = "wrong-price entry", f"entry {o['price_entry']} → {n['price_entry']:.4f}"
            elif n_closed and abs(d_pnl) >= 0.01:
                if abs(d_pnl + comm) < 0.02:
                    cat, detail = "commission", f"commissions ${comm:.2f}"
                else:
                    cat, detail = "wrong-price close", f"P&L Δ {d_pnl:+.2f} not explained by commissions"
            elif n_closed and abs(d_pnl) > 1e-9:
                cat = "rounding"
        rows.append(dict(
            trade_id=tid, ticker=(n or o)["ticker"], category=cat, detail=detail,
            old_qty=o and o["quantity"], new_qty=n and n["quantity_adj"],
            old_entry=o and o["price_entry"], new_entry=n and n["price_entry"],
            old_exit=o and o["price_exit"], new_exit=n and n["price_exit"],
            old_exit_ts=o and o["timestamp_exit"], new_exit_ts=n and n["timestamp_exit"],
            old_pnl=o and o["pnl_dollar"], new_pnl=n and n["pnl_dollar"],
            new_realized_incl_partial=n and n["realized_pnl"],
            old_label=o and o["outcome_label"], new_label=n and n["outcome_label"],
            old_attr=o and o["signal_attribution_source"], new_attr=n and n["signal_attribution_source"],
            signals=json.dumps(sigs((n or o)["signals_fired"]))))

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── summary ──────────────────────────────────────────────────────
    cnt = defaultdict(int)
    dpnl = defaultdict(float)
    for r in rows:
        cnt[r["category"]] += 1
        dpnl[r["category"]] += (f(r["new_pnl"]) or 0) - (f(r["old_pnl"]) or 0)
    old_real = sum(f(o["pnl_dollar"]) or 0 for o in old.values() if o["timestamp_exit"])
    new_closed = sum(f(n["pnl_dollar"]) or 0 for n in new.values() if n["timestamp_exit"])
    new_all = sum(f(n["realized_pnl"]) or 0 for n in new.values())
    mapped_new = sum(f(new[t]["pnl_dollar"]) or 0 for t in old if t in new and new[t]["timestamp_exit"])
    kairos_era_new = sum(f(n["pnl_dollar"]) or 0 for n in new.values()
                         if n["timestamp_exit"] and n["timestamp_entry"][:10] >= KAIROS_ERA_START)
    commissions = sum((f(n["entry_commission"]) or 0) + (f(n["exit_commission"]) or 0)
                      for n in new.values() if n["timestamp_exit"])

    def per_signal(src, trusted_only):
        agg = defaultdict(lambda: [0, 0.0])
        for r in src.values():
            if not r["timestamp_exit"] or r["pnl_dollar"] is None:
                continue
            if trusted_only and r["signal_attribution_source"] not in TRUSTED:
                continue
            for s in sigs(r["signals_fired"]):
                agg[s][0] += 1
                agg[s][1] += float(r["pnl_dollar"])
        return agg
    o_all, n_all = per_signal(old, False), per_signal(new, False)
    o_tr, n_tr = per_signal(old, True), per_signal(new, True)
    top = [s for s, _ in sorted(n_tr.items(), key=lambda kv: -kv[1][0])][:5]
    shown = ["HOT-INSIDER"] + [s for s in top if s != "HOT-INSIDER"]

    L = []
    L.append("# Old vs new trade record — diff summary\n")
    L.append(f"Source: `{args.db}` — legacy `trade_outcomes_legacy` ({len(old)} rows) vs the new "
             f"`trade_outcomes` view ({len(new)} trades). Per-trade detail: `{args.csv}`.\n")
    L.append("P&L convention: new P&L is **net of commissions on both legs**; the legacy ledger "
             "was gross. Every closed legacy row's commission gap is therefore a (small) "
             "difference, categorised `commission` when that is the only difference.\n")
    L.append("## Realized P&L\n")
    L.append("| measure | old | new | Δ |\n|---|---:|---:|---:|")
    L.append(f"| all closed trades | {old_real:,.2f} | {new_closed:,.2f} | {new_closed - old_real:+,.2f} |")
    L.append(f"| closed trades, Kairos era (entry ≥ {KAIROS_ERA_START}) | {old_real:,.2f} | "
             f"{kairos_era_new:,.2f} | {kairos_era_new - old_real:+,.2f} |")
    L.append(f"| same trade_ids only (legacy rows, new values) | {old_real:,.2f} | {mapped_new:,.2f} | "
             f"{mapped_new - old_real:+,.2f} |")
    L.append(f"| incl. partial exits of still-open trades (trades.realized_pnl) | — | {new_all:,.2f} | |")
    L.append(f"\nCommissions netted in new closed-trade P&L: ${commissions:,.2f}.\n")
    L.append("## Count per category\n")
    L.append("| category | trades | Σ ΔP&L (new − old, closed) |\n|---|---:|---:|")
    order = ["corporate action", "opening balance", "missing row", "ghost row", "quantity mismatch",
             "wrong-price close", "wrong-price entry", "commission", "rounding", "identical"]
    for k in order + sorted(set(cnt) - set(order)):
        if cnt.get(k):
            L.append(f"| {k} | {cnt[k]} | {dpnl[k]:+,.2f} |")
    L.append(f"| **total** | **{len(rows)}** | {sum(dpnl.values()):+,.2f} |\n")
    sub = defaultdict(int)
    for r in rows:
        if r["category"] == "missing row":
            sub[r["detail"]] += 1
    if sub:
        L.append("`missing row` breakdown: " + "; ".join(f"{v} × {k}" for k, v in sub.items()) + "\n")
    L.append("## Largest individual differences (|ΔP&L|)\n")
    L.append("| trade_id | ticker | category | old P&L | new P&L | detail |\n|---|---|---|---:|---:|---|")
    big = sorted(rows, key=lambda r: -abs((f(r["new_pnl"]) or 0) - (f(r["old_pnl"]) or 0)))[:12]
    money = lambda x: "" if x is None else f"{float(x):,.2f}"
    for r in big:
        L.append(f"| {r['trade_id'][:13]} | {r['ticker']} | {r['category']} | "
                 f"{money(r['old_pnl'])} | {money(r['new_pnl'])} | {r['detail']} |")
    L.append("\n## Per-signal realized P&L (closed trades)\n")
    L.append("Trusted = signal_attribution_source ∈ {explicit, confluence, decision_record}. "
             "HOT-INSIDER plus the top 5 signals by trusted closed-trade count (new).\n")
    L.append("| signal | old n | old P&L | new n | new P&L | old n (trusted) | old P&L (trusted) | "
             "new n (trusted) | new P&L (trusted) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in shown:
        a, b, ta, tb = o_all.get(s, [0, 0.0]), n_all.get(s, [0, 0.0]), o_tr.get(s, [0, 0.0]), n_tr.get(s, [0, 0.0])
        L.append(f"| {s} | {a[0]} | {a[1]:,.2f} | {b[0]} | {b[1]:,.2f} | {ta[0]} | {ta[1]:,.2f} | "
                 f"{tb[0]} | {tb[1]:,.2f} |")
    open(args.summary, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
