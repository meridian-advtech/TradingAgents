"""Shadow-log the sector concentration guardrail under BOTH classifications.

Step 2 of the sector-classification plan. This module OBSERVES and RECORDS. It
never changes a trading decision — the live gate keeps using universe buckets
exactly as before, and every function here is wrapped so a failure degrades to
"log nothing" rather than affecting an order.

WHY THIS EXISTS RATHER THAN A DIRECT CUTOVER
`check_sector_concentration` is built on `lookup_sector`, which returns
universe SCREENING BUCKETS (`mega_cap`, `value_dividend`, …) — size/style
cohorts, not industries. Real Financial Services exposure scatters across six
buckets, so on 2026-08-03 the book held 25.12% while the guardrail read
11.72%.

The tempting move is to swap in real sectors immediately. That would be wrong
for a reason that has nothing to do with correctness of the classifier:
buckets aggregate LOWER than sectors by construction, so the existing
threshold was implicitly calibrated against an understated number. Swapping
the measurement while keeping the limit silently tightens the gate — a real
behaviour change disguised as a bug fix, on a limit J has explicitly called a
watermark rather than a hard cap.

So: collect evidence first about how far the two readings diverge, across
tickers and regimes, then let J set the limit that belongs with real sectors.
That decision needs data this table is here to produce.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sector_shadow_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    proposed_spend  REAL,
    nlv             REAL,
    max_pct         REAL,
    -- live path (universe buckets) — what actually gated the trade
    bucket_sector   TEXT,
    bucket_current  REAL,
    bucket_new_pct  REAL,
    bucket_passed   INTEGER,
    -- shadow path (security_master real sectors) — recorded only
    real_sector     TEXT,
    real_current    REAL,
    real_new_pct    REAL,
    real_passed     INTEGER,
    -- divergence
    gap_pp          REAL,
    verdicts_differ INTEGER,
    is_fund         INTEGER,
    regime          TEXT,
    detail          TEXT
);
CREATE INDEX IF NOT EXISTS idx_sector_shadow_ts
    ON sector_shadow_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_sector_shadow_differ
    ON sector_shadow_log (verdicts_differ, timestamp);
"""


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _real_sector_exposure(positions: dict) -> dict:
    """{real_sector: market_value} from broker positions, funds excluded.

    Funds carry no single sector; folding an ETF into one would misstate
    concentration in whichever sector it landed on.
    """
    from kairos_security_master import get_sector, is_fund
    out: dict = {}
    for sym, p in (positions or {}).items():
        try:
            if is_fund(sym):
                continue
            mv = float(p.get("market_value") or 0)
            if mv <= 0:
                continue
            out[get_sector(sym)] = out.get(get_sector(sym), 0.0) + mv
        except Exception:
            continue
    return out


def log_shadow_decision(ticker: str, proposed_spend: float, portfolio: dict,
                        guardrails: dict | None = None,
                        bucket_passed: bool | None = None,
                        bucket_sector: str | None = None,
                        regime: str | None = None) -> str | None:
    """Record what the concentration gate decides under both classifications.

    Returns a one-line human summary when the two readings DISAGREE on the
    verdict (worth printing), else None. Never raises.
    """
    try:
        from kairos_confluence import lookup_sector, load_guardrails
        from kairos_security_master import get_sector, is_fund

        guardrails = guardrails or load_guardrails()
        max_pct = float(guardrails.get("max_sector_concentration_pct", 0.25))
        nlv = float(portfolio.get("nlv") or 0)
        if nlv <= 0:
            return None
        spend = float(proposed_spend or 0)
        max_value = nlv * max_pct

        # ── live path: universe buckets (unchanged, this is what gates) ──
        b_sector = bucket_sector or lookup_sector(ticker)
        b_current = float((portfolio.get("sector_exposure") or {}).get(b_sector, 0))
        b_new = b_current + spend
        b_pass = b_new <= max_value if bucket_passed is None else bool(bucket_passed)

        # ── shadow path: real sectors ────────────────────────────────────
        r_sector = get_sector(ticker)
        fund = is_fund(ticker)
        r_exposure = _real_sector_exposure(portfolio.get("positions") or {})
        r_current = float(r_exposure.get(r_sector, 0))
        # A fund has no single sector, so a concentration test on it is
        # meaningless — recorded, never treated as a breach.
        r_new = r_current + spend
        r_pass = True if fund else (r_new <= max_value)

        b_new_pct = 100.0 * b_new / nlv
        r_new_pct = 100.0 * r_new / nlv
        differ = bool(b_pass) != bool(r_pass)

        init_db()
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "INSERT INTO sector_shadow_log "
                "(timestamp, ticker, proposed_spend, nlv, max_pct, bucket_sector, "
                " bucket_current, bucket_new_pct, bucket_passed, real_sector, "
                " real_current, real_new_pct, real_passed, gap_pp, "
                " verdicts_differ, is_fund, regime, detail) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                 ticker, spend, nlv, max_pct, b_sector, b_current, b_new_pct,
                 int(b_pass), r_sector, r_current, r_new_pct, int(r_pass),
                 r_new_pct - b_new_pct, int(differ), int(fund), regime,
                 json.dumps({"max_value": round(max_value, 2)})),
            )
            conn.commit()
        finally:
            conn.close()

        if differ:
            return (f"SECTOR SHADOW: buckets say {'PASS' if b_pass else 'BLOCK'} "
                    f"({b_sector} → {b_new_pct:.2f}%), real sectors say "
                    f"{'PASS' if r_pass else 'BLOCK'} ({r_sector} → "
                    f"{r_new_pct:.2f}%, limit {max_pct:.0%}) — observation only, "
                    f"live gate unchanged")
        return None
    except Exception:
        return None


def report(limit: int = 20) -> dict:
    """Summarise collected divergence. Read-only."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        n = conn.execute("SELECT COUNT(*) c FROM sector_shadow_log").fetchone()["c"]
        if not n:
            return {"rows": 0}
        differ = conn.execute(
            "SELECT COUNT(*) c FROM sector_shadow_log WHERE verdicts_differ=1"
        ).fetchone()["c"]
        blocked_by_real = conn.execute(
            "SELECT COUNT(*) c FROM sector_shadow_log "
            "WHERE bucket_passed=1 AND real_passed=0").fetchone()["c"]
        worst = [dict(r) for r in conn.execute(
            "SELECT ticker, bucket_sector, bucket_new_pct, real_sector, "
            "       real_new_pct, gap_pp, timestamp "
            "FROM sector_shadow_log ORDER BY ABS(gap_pp) DESC LIMIT ?", (limit,))]
        by_sector = [dict(r) for r in conn.execute(
            "SELECT real_sector, COUNT(*) n, AVG(gap_pp) avg_gap, "
            "       MAX(real_new_pct) max_real_pct "
            "FROM sector_shadow_log GROUP BY real_sector ORDER BY avg_gap DESC")]
        return {"rows": n, "verdicts_differ": differ,
                "would_block_but_passed": blocked_by_real,
                "worst_gaps": worst, "by_sector": by_sector}
    finally:
        conn.close()


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Sector shadow-log report")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()
    r = report(args.limit)
    if not r.get("rows"):
        print("No shadow observations recorded yet.")
        print("Rows accumulate as the executor evaluates BUYs.")
        return 0
    print(f"observations                     : {r['rows']}")
    print(f"verdicts differ                  : {r['verdicts_differ']}")
    print(f"passed on buckets, would block   : {r['would_block_but_passed']}")
    print("\nlargest divergences (real % minus bucket %):")
    print(f"  {'ticker':<8}{'bucket sector':<22}{'buck%':>7}"
          f"{'real sector':<24}{'real%':>7}{'gap pp':>8}")
    for w in r["worst_gaps"]:
        print(f"  {w['ticker']:<8}{(w['bucket_sector'] or '')[:21]:<22}"
              f"{w['bucket_new_pct']:>7.2f}{(w['real_sector'] or '')[:23]:<24}"
              f"{w['real_new_pct']:>7.2f}{w['gap_pp']:>8.2f}")
    print("\nby real sector:")
    for s in r["by_sector"]:
        print(f"  {(s['real_sector'] or '')[:26]:<28}n={s['n']:<5}"
              f"avg gap {s['avg_gap']:+7.2f}pp   peak real {s['max_real_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
