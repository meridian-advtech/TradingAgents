# Handoff — Sector Classification & Concentration Guardrail

**To:** Kairos Maturity Build — 1
**From:** dashboard design-pass session, 2026-08-03
**Status:** Step 1 complete and inert. Steps 2–4 not started.

---

## TL;DR

`lookup_sector()` returns **universe screening buckets**, not sectors. The
portfolio concentration guardrail is built on it, so it measures a grouping
that has no risk meaning. As of 2026-08-03 real Financial Services exposure is
**25.12%** while the guardrail reads it as **11.72%**.

A replacement classification layer (`kairos_security_master.py`) is built,
backfilled, and tested, but **nothing imports it yet**. Trading behaviour is
unchanged.

> **Scope note from J (2026-08-03):** the 25% figure is a **watermark, not a
> hard operating limit**. Treat the current reading as an *observation point*,
> not a breach requiring remediation. Do **not** rush an enforcement cutover on
> the strength of that number. The defect worth fixing is that the measurement
> is meaningless, not that a threshold was crossed.

---

## Root cause

`kairos_confluence._build_sector_map()` builds `{ticker: sector}` from the
category keys in `kairos_universe.json`. Those keys are screening cohorts:

```
mega_cap  large_cap  mid_cap_growth  value_dividend  consumer_tech
healthcare_biotech  financials  energy_materials  reits_real_estate
industrials_transport
```

Two different concepts were collapsed into one field:

| | meaning | legitimate use |
|---|---|---|
| universe bucket | how a ticker entered the screener (size/style) | universe management, screening |
| sector | what industry the company is in | **concentration risk** |

Tier B tickers carry a real sector string and fall through a title-case path in
`kairos_dashboard._normalize_sector()`, which is why the dashboard showed both
`Financials` ($144k) and `Financial Services` ($16k) — two provenances, one
column.

## Why it matters

`lookup_sector` is load-bearing in three places, all of which inherit the defect:

| Call site | Use | Consequence |
|---|---|---|
| `kairos_execute.py:361` | `check_sector_concentration()` pre-trade gate | **Blind to real sector risk** |
| `kairos_sector_monitor.py:52,94` | pre-trade warnings into the reasoning prompt | Council reasons on wrong exposure |
| `kairos_tax_harvest.py:284,412` | wash-sale replacement selection | "Same sector" replacement may hold different exposure |

### Measured, live book (2026-08-03, 56 equity positions, $1,045,603)

```
                              real %   seen %      gap   buckets
Financial Services            25.12%   11.72%   13.40pp   6   <-- widest gap
Consumer Cyclical             12.65%    6.30%    6.35pp   3
Industrials                   12.84%    9.64%    3.20pp   2
Basic Materials                5.39%    2.65%    2.73pp   3
Technology                     7.91%    5.31%    2.60pp   3
Healthcare                     7.73%    5.15%    2.58pp   2
```

Financial Services scatters across six buckets (`financials`, `mega_cap`,
`large_cap`, `value_dividend`, plus two Tier B labels). The guardrail counts
only the largest and believes ~13pp of headroom remains.

Reproduce:

```bash
source ~/Kairos-env/bin/activate && python kairos_security_master.py compare
```

---

## What is already built (step 1 — complete, inert)

| File | Purpose |
|---|---|
| `kairos_security_master.py` | Resolver, cache, public API, CLI |
| `kairos_sector_overrides.json` | 38 curated fund/ETF entries |
| `kairos_selftest_security_master.py` | 25 assertions, all passing |

**Table** `security_master` in `kairos.db`:
`ticker, sector, industry, asset_type, source, detail, resolved_at, is_override`.
The `source` column makes every classification attributable — relevant to the
B2B/RIA auditability requirement.

**Resolution chain:** `override` → `yfinance` → `sec_sic` → `Unclassified`.
Never silently buckets an unknown.

**Backfill result — 646 tickers:**

| source | n |
|---|---|
| yfinance | 594 |
| override | 38 |
| sec_sic | 2 |
| unresolved | 11 |

**Public API** (read-only, no side effects):

```python
from kairos_security_master import get_sector, get_industry, is_fund, sector_exposure

get_sector("JPM")                    # "Financial Services"
is_fund("SPY")                       # True — excluded from concentration
sector_exposure(positions)           # {sector: market_value}, funds excluded
```

**CLI:**

```bash
python kairos_security_master.py backfill [--limit N] [--force]
python kairos_security_master.py resolve JPM MSFT
python kairos_security_master.py status
python kairos_security_master.py stale
python kairos_security_master.py compare
```

Refresh TTL is 90 days. `unresolved` rows are always retried, never TTL-cached.

---

## Remaining work

### Step 2 — observation only (no behaviour change)

Log, for every BUY the guardrail evaluates, what it decides under **both**
classifications. Do not enforce the new one. The purpose is to build a record
of how far apart the two readings run over time and across regimes — not to
stage an imminent cutover.

Per J: 25% is a watermark. Before any enforcement change, the question to answer
from the collected data is *what the right limit is when it means real sectors* —
buckets aggregate lower by construction, so the number set against buckets is
not transferable. That calibration is J's call and needs evidence first.

### Step 3 — enforcement cutover (deferred; needs J's sign-off)

`kairos_execute.py:361` → `security_master.get_sector()`, excluding funds via
`is_fund()`. **Blocked on step 2 evidence and an explicit threshold decision.**
Ship alone when it happens, per one-change-at-a-time.

### Step 4 — remaining call sites

`kairos_sector_monitor.py` (both call sites) and `kairos_tax_harvest.py`
(`_find_replacement`). Tax harvest deserves its own thought: "same sector" for
a wash-sale replacement may want *industry*, not sector.

### Step 5 — dashboard (owned by the design track, not this thread)

`_compute_sector_exposure()` in `kairos_dashboard.py` → `security_master`.
Already done in the design prototype; show sector and size/style as two axes.

---

## Secondary findings

### A. Share-class symbols were silently unclassified — FIXED

Kairos/IBKR write `BRK.B`; yfinance and SEC write `BRK-B`. Every dual-class
name fell through to `Unclassified`. `_vendor_symbol()` now translates, with
tests. **Worth checking whether the same mismatch affects other vendor calls
in the codebase** — this module is unlikely to be the only place.

### B. 11 universe tickers no longer trade under those symbols — NOT FIXED

```
BK  CFLT  COUP  HOLX  HZNP  MMC  MRO  PXD  SQ  TTM  WBA
```

Hand-verified against the SEC registrant list:

| symbol | finding |
|---|---|
| `BK` | renamed → **BNY** (Bank of New York Mellon) |
| `MMC` | renamed → **MRSH** (Marsh & McLennan) |
| `SQ` | renamed → **XYZ** (Block, Inc.) |
| `CFLT` `COUP` `HOLX` `HZNP` `MRO` `PXD` `TTM` `WBA` | absent from SEC registrant list — apparently delisted; **each needs individual confirmation** |

Also: Tier B lists `TTM` as "TTM Technologies Inc." — that company trades as
`TTMI`, so this looks like a wrong-ticker mapping rather than a rename.

None are currently held, so there is no marking risk today — but they are
screened every cycle, and a renamed symbol would fail to trade if selected.

Deliberately **not** auto-remapped: guessing a replacement ticker from a fuzzy
name match is the wrong risk in a system that places orders. `stale` flags them
for a human and stops.

**Suggested follow-up:** a periodic universe liveness check, since this will
recur. There is no current mechanism that would have caught it.

---

## Constraints observed

- Nothing in the live path imports the new module; step 1 changed no behaviour.
- No existing file was modified.
- Selftests run against disposable `/tmp` copies with the network stubbed;
  they never touch `kairos.db` or the live config.
- Uncommitted at handoff. Work sits on `fix/reallocation-atomic-swap`, which is
  unrelated — wants its own branch.
