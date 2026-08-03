"""Sector shadow-log tests. Runs against a disposable /tmp copy of kairos.db."""
import os, shutil, sqlite3, sys, tempfile
sys.path.insert(0, "/Users/jelmore/Kairos")

tmp = tempfile.mkdtemp(prefix="sector_shadow_")
db = os.path.join(tmp, "kairos.db")
# Real security_master rows are needed, so copy the live DB rather than stub it.
shutil.copy("/Users/jelmore/Kairos/kairos.db", db)

import kairos_sector_shadow as S
S.DB_PATH = db
S.init_db()
# Start from a clean log in the sandbox.
c = sqlite3.connect(db); c.execute("DELETE FROM sector_shadow_log"); c.commit(); c.close()

res = []
def ck(label, cond, detail=""):
    res.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"\n         {detail}" if detail else ""))

# A book concentrated in Financial Services across several universe buckets.
PORT = {
    "nlv": 1_000_000.0,
    "cash": 50_000.0,
    "positions": {
        "JPM":  {"market_value": 90_000.0},
        "BAC":  {"market_value": 80_000.0},
        "MS":   {"market_value": 60_000.0},
        "GS":   {"market_value": 40_000.0},
        "SPY":  {"market_value": 50_000.0},   # fund — must be excluded
        "MSFT": {"market_value": 40_000.0},
    },
    # Buckets scatter the same names, so the live view sees far less.
    "sector_exposure": {"financials": 90_000.0, "mega_cap": 80_000.0,
                        "large_cap": 60_000.0, "value_dividend": 40_000.0},
}
GR = {"max_sector_concentration_pct": 0.25, "min_cash_reserve_pct": 0.01,
      "max_single_position_pct": 0.10}

print("\n=== A. Divergence is detected and recorded ===")
note = S.log_shadow_decision("WFC", 30_000.0, PORT, GR, bucket_passed=True,
                             bucket_sector="financials")
row = sqlite3.connect(db).execute(
    "SELECT ticker,bucket_sector,bucket_new_pct,real_sector,real_new_pct,"
    "bucket_passed,real_passed,verdicts_differ FROM sector_shadow_log "
    "ORDER BY id DESC LIMIT 1").fetchone()
print(f"  row: {row}")
ck("a row was written", row is not None)
ck("real sector resolved (not a bucket name)", row[3] == "Financial Services",
   f"got {row[3]!r}")
ck("real exposure exceeds bucket exposure", row[4] > row[2],
   f"real {row[4]:.2f}% vs bucket {row[2]:.2f}%")
ck("buckets PASS while real sectors BLOCK", row[5] == 1 and row[6] == 0)
ck("divergence flagged", row[7] == 1)
ck("a human-readable note is returned on disagreement", bool(note), note)

print("\n=== B. Funds are excluded from real-sector exposure ===")
exp = S._real_sector_exposure(PORT["positions"])
print(f"  real exposure: { {k: round(v) for k,v in exp.items()} }")
# JPM 90k + BAC 80k + MS 60k + GS 40k = 270k Financial Services,
# MSFT 40k Technology, SPY 50k excluded entirely -> 310k total.
ck("fund value excluded from the total",
   sum(exp.values()) == 310_000.0,
   f"total {sum(exp.values()):,.0f} (expected 310,000 = everything but SPY)")
ck("no sector absorbed SPY's 50k",
   exp.get("Financial Services") == 270_000.0 and exp.get("Technology") == 40_000.0,
   f"{ {k: round(v) for k, v in exp.items()} }")
S.log_shadow_decision("SPY", 10_000.0, PORT, GR, bucket_passed=True)
r2 = sqlite3.connect(db).execute(
    "SELECT is_fund, real_passed FROM sector_shadow_log ORDER BY id DESC LIMIT 1").fetchone()
ck("a fund is marked and never counted as a breach", r2 == (1, 1), f"got {r2}")

print("\n=== C. Agreement produces no note (only disagreement is surfaced) ===")
quiet = S.log_shadow_decision("MSFT", 5_000.0, PORT, GR, bucket_passed=True)
ck("no note when both agree", quiet is None, f"got {quiet!r}")

print("\n=== D. It CANNOT change a trading decision ===")
import kairos_execute as E
src = open("/Users/jelmore/Kairos/kairos_execute.py").read()
seg = src[src.index("# 2b. SHADOW ONLY"):src.index("# 3. Single position")]
ck("shadow block never touches `failures`", "failures" not in seg)
ck("shadow block never assigns `ok`", "ok =" not in seg and "ok=" not in seg)
ck("shadow block is wrapped in try/except", "try:" in seg and "except Exception:" in seg)

print("\n=== E. Broken shadow layer degrades to silence ===")
orig = S._real_sector_exposure
S._real_sector_exposure = lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
safe = S.log_shadow_decision("JPM", 1_000.0, PORT, GR, bucket_passed=True)
S._real_sector_exposure = orig
ck("an internal failure returns None rather than raising", safe is None)

print("\n=== F. Report summarises without crashing ===")
rep = S.report(5)
print(f"  rows={rep['rows']} differ={rep['verdicts_differ']} "
      f"would_block_but_passed={rep['would_block_but_passed']}")
ck("report counts the divergence", rep["verdicts_differ"] >= 1)
ck("report counts pass-but-would-block", rep["would_block_but_passed"] >= 1)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{'='*60}\n{sum(res)}/{len(res)} checks passed")
sys.exit(0 if all(res) else 1)
