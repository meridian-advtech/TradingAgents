# Kairos — Engineering Context

Consolidated from Chat project history (Kairos Build Phase 1-1 through 1-6, Kairos Maturity Phase 1-2,
Arbiter Build threads). This file gives Claude Code the standing context that used to live only in Chat.

## What Kairos Is

An autonomous AI-powered equity/crypto trading system built under Meridian Technologies (J / Jason,
jelmore@meridiantech.global), targeting B2B licensing to RIAs and family offices. Core thesis:
**information synthesis arbitrage** — the edge comes from synthesizing non-price signals (SEC insider
filings, congressional trades, unusual options flow, earnings data, AI value-chain cascades, IPO
intelligence, event-driven seasonality) *before* the market prices them in. This operates on
hours-to-days timeframes, not milliseconds — speed of synthesis, not speed of execution. This framing
governs all signal and architecture decisions; don't reach for latency/execution-speed optimizations,
that's not the game being played here.

Currently in **paper trading Phase 1** on a ~$1.08M NLV IBKR paper account (~55-65 open positions).
Live capital deployment targeted ~September 2026, funded from a pending business exit (Tuearis).
J's explicit performance floor is **29% annualized** — his stated threshold above a ~14-15% passive
alternative below which the project isn't worth the operational cost and stress.

## Environment

- Runs on **Prometheus** (MacBook Air M5, 32GB RAM), lid-closed, plugged in permanently. Production
  target **Olympus** (Mac Studio M5 Ultra, 256GB RAM) arriving mid-to-late 2026.
- Codebase: `/Users/jelmore/Kairos`. Python 3.11 venv at `~/Kairos-env` — always
  `source ~/Kairos-env/bin/activate` before running Kairos scripts directly (not needed inside Claude
  Code sessions using its own tool execution, but relevant when shelling out).
- IB Gateway on port 7497 (paper). Replaced TWS entirely — TWS updates silently reset the "Enable
  ActiveX and Socket Clients" checkbox, which caused a full trading outage early on. IBC
  (Interactive Brokers Controller) automates login; Launch Agent at
  `~/Library/LaunchAgents/com.kairos.ibcgateway.plist` (RunAtLoad only, no KeepAlive, to avoid
  restart loops) restarts it nightly at 11:59 PM ET with a 5-minute grace period.
- Database: `kairos.db` (SQLite) — primary state *and*, since 2026-09-28, the trade record and ML
  corpus (see Trade Record below). `kairos_ml_outcomes.db` is retired: nothing opens it except
  `kairos_migrate_fills.py`; archive it, don't write to it.
- Slack workspace: Meridian Technologies. Channels: `#kairos-trades` (fills only),
  `#kairos-alerts` (system health), `#kairos-reports` (cycle summaries), `#kairos-watchlist`
  (Tier C / universe changes), `#kairos-log` (verbose/debug), `#kairos-arbiter`, `#kairos-commands`.
- Tailscale enables remote dashboard access from J's iPhone; dashboard binds to `0.0.0.0` not
  `localhost` for this to work. Dashboard server runs on port 5001.
- Little Snitch blocks Ollama's outbound traffic by default; temporary rule exceptions are opened for
  model pulls, then re-locked.

## Pipeline Architecture (current)

- **Phase 0** — Ollama orchestration / macro regime detection (`kairos_regime.py`). Four states:
  NORMAL / CAUTION / RISK-OFF / EXTREME-FEAR, with hard programmatic enforcement of guardrails
  (e.g. stop-loss thresholds tighten as regime worsens: 10% NORMAL → 5% EXTREME-FEAR).
- **Phase 0.5** — Tier 1 screener (`kairos_screener.py`, now `phi4-mini` via Ollama). Batch HOT/WARM/COLD
  classification across the 727-ticker universe (grew from an original 279 through tiered additions —
  Tier A: S&P 500 + Nasdaq-100 core; Tier B: curated high-growth/recent-IPO watchlist; Tier C:
  opportunistic temporary holdings with a 14-day TTL, self-populating via an automated intake engine
  polling SEC EDGAR Form 4, OpenInsider, and House/Senate Stock Watcher).
- **Phase 1** — Gather snapshot: IBKR state, tax context, history, shortlist.
- **Phase 2** — Council reasoning via the Anthropic API. Model IDs live only in `kairos_config.json`
  `claude.*`: `model` = decision call (`claude-opus-5-5`), `router_model` / `thesis_model` /
  `watch_model` = `claude-sonnet-5-5` (upgraded 2026-09-29 from Opus 5 / Sonnet 5). No module
  hardcodes a model ID; change models in config only.
  Effort levels per call type: high for decisions, medium for routing, low for thesis review.
  Reasoning timeout 240s, max_tokens 8192 (bumped from 180s/4096 after cold-cache timeouts and
  extended-thinking token exhaustion post-Sonnet-5 switch).
- **Phase 3** — Execute via IBKR.
- **Router/analyst layer** — local Ollama `reason_model` (`qwen3.8:27b-mlx` as of 2026-08-19; see the
  `ollama._comment` in config for the swap history), handles news filtering/routing.

### Active signals
HOT-REVERSION, HOT-INSIDER, HOT-CONGRESS, HOT-OPTIONS, HOT-CATALYST, HOT-CHAIN (AI value-chain
cascade — fires when a Tier 1 AI name moves >3% and downstream hasn't repriced), HOT-EARNINGS,
HOT-EVENT (event-driven seasonality around known calendar catalysts), HOT-IPO (autonomous EDGAR S-1
scanning + scored sizing tiers). Confluence scoring cross-references co-firing signals to dynamically
size positions (1.0x base, up to 1.25x for Very High confluence, 5+ points).

Crypto (BTC/ETH via IBKR PAXOS) is currently **disabled** for active trading — J decided to hold
BTC/ETH long-term separately rather than actively trade them; crypto execution is deferred to live
IBKR when going live in September.

## Exit / Sell Logic Stack

Four layers, in order of when they run:

1. **Phase 0.1 — Stop-loss** (`kairos_stoploss.py`), every cycle. Regime-adjusted drawdown thresholds.
   Tax override protects positions 340-365 days old if drawdown < 15%.
2. **Daily 9:35am ET — Thesis review** (`kairos_thesis_review.py`). Checks TAKE-PROFIT,
   STALE-THESIS, THESIS-INVALID conditions. Primary discretionary exit path.
3. **Every cycle — Reallocation engine** (`kairos_reallocation.py`). Sells weakest-thesis position to
   fund a higher-conviction new opportunity (conviction ≥7 required to trigger). Five sequential gates,
   each a hard stop with a logged rejection reason:
   - Gate 1: `MIN_HOLD_DAYS_FOR_REALLOCATION = 5` (anti-churn)
   - Gate 2: conviction delta ≥ `MIN_CONVICTION_DELTA` (raised 3 → 4.5 after churn analysis)
   - Gate 2.5: thesis validity check against `thesis_reviews` table
   - Gate 2.6: remaining thesis runway by signal-type hold window (HOT-EARNINGS 10d, HOT-INSIDER 21d,
     HOT-CONGRESS 30d, HOT-OPTIONS 7d, HOT-REVERSION 5d, HOT-CATALYST 14d)
   - Gate 3.5 (formerly missing, now fixed): unrealized loss floor at -15%
   - Gate 4: tax cost — foregone long-term savings — must not exceed `MAX_TAX_SAVINGS_FOR_EXIT`
   - Gate 5: wash sale lookback check
4. **Pre-execution — Tax efficiency** (`kairos_tax_efficiency.py`). After-tax NPV optimization:
   compares tax savings from waiting for long-term treatment against the opportunity cost of waiting
   (`opportunity_cost_rate: 0.08` annual, configurable), replacing a blunt "$500 threshold" heuristic
   that was provably wrong in several test cases. Wash-sale prevention and tax-loss harvesting (with
   replacement-candidate injection) also live here. Tax context gets injected into the Council's
   reasoning prompt (Section 0b) for any position held 180-364 days, with an explicit instruction not
   to override a HOLD-FOR-TAX flag except for stop-loss/thesis-invalid conditions.

**Key architectural insight (repeatedly load-bearing):** *entry signal ≠ holding thesis.* The signal
that justified buying a position isn't necessarily the right criterion for continuing to hold it —
holding theses can be longer-term/fundamental (e.g. a product-cycle recovery), not reducible to signal
freshness scoring. The reallocation engine historically ran far more often (every ~cycle) than thesis
review (once daily), letting it dominate exits and recycle positions mid-thesis at small gains before
Gates 2.5/2.6 were added.

**Recent addition:** target-armed trailing stop — arms only after a position's own thesis target is
reached (backtested +31.6 pts over actual on 49 clean trades).

5. **Price-level thesis invalidation** (`kairos_exits.price_invalidation_exit`, condition 1.5 — after
   the hard stop, before the trailing stop). Shipped 2026-08-03, **`enabled: false`**. Exits when the
   last N consecutive daily CLOSES are all at or below a price level parsed from the position's own
   logged `thesis_predictions.invalidation_conditions`. Reason is prefixed `PRICE-INVALIDATION:` and
   tagged `[price-invalid]`, with its own `_log_sell` trigger — deliberately **distinct from the
   reasoning-driven `THESIS-INVALID`** exit (the LLM thesis-review path), so per-mechanism attribution
   stays separable. Backtest n=56: +12.7 pts, fired 15/56, 9 helped / 4 hurt.
   *Constraint worth knowing before relying on it:* it can only act on theses that name a number, and
   at ship time only 37% did — the Council prompt's own worked example contained no price at all. The
   prompt now REQUIRES the first invalidation bullet to read `closes below $N`, so coverage should
   climb for new theses; existing positions are out of scope by design. Zero of 56 live positions
   would have fired at ship time.

**Never-oversell guard (long-only enforcement).** `kairos_sell_guard.clamp_sell_quantity` sits at both
order-submission choke points — `kairos_execute.execute_order` and `kairos_stoploss._place_market_sell`
(which covers stop-loss, the exit engine, thesis review, tax harvest, and the reallocation SELL leg).
It clamps any SELL to the shares actually held, aborts at held ≤ 0, and alerts `#kairos-alerts` either
way. It fails OPEN only when broker *and* DB are both unreadable — blocking every stop-loss is the
larger risk — and says so loudly when it does. Aborted sells return `oversell_blocked` so callers skip
close logging; no phantom rows. Entry guardrails (cash reserve / sector concentration / single
position) now run on **BUY only** — all three model `proposed_spend` as capital being *added*, so
applying them to a SELL could block an exit.

## Trade Record (fills ledger — live since 2026-09-28)

The trade record is **derived from broker fills**, not written by the trading path. Replaced the
legacy `holdings` / `trade_outcomes` tables, whose write-time bookkeeping had drifted badly from IBKR
(ghost closes at other trades' prices, a DD split booked as +$28.6K that never existed, 68 filled BUYs
never recorded, wrong quantities). Code: `kairos_ledger.py`. Deployed as `23e726a` + `7b5712c`; git
tag `pre-fills-migration` marks the last pre-ledger commit; pre-migration DB backups in
`~/kairos-backups/premigration_20260928/` (and the tar.gz on Box). Reports:
`~/kairos-rehearsal/PHASE1_REPORT.md`, `PHASE2_REPORT.md`.

- **Facts (append-only, UPDATE/DELETE raise):** `fills` (one row per IBKR execution, keyed on
  `exec_id`, UTC ISO-Z timestamps), `fill_commissions` (late commissions — `fills` is never updated;
  an unknown commission is NULL, not 0), `position_events` (splits, mergers, symbol changes, opening
  balances).
- **Annotations (why, not what):** `decisions`, `entry_annotations` (a BUY without them is refused),
  `exit_annotations` (exit_signals required; `[]` only with a note).
- **Derived — never hand-edit, rebuild instead:** `trades`, `trade_matches`, `lots` are rewritten by
  `rebuild_trades()` (deterministic FIFO over fills + position_events). P&L is **net of both legs'
  commissions**. `trade_features` (MFE / give-back / forgone) comes from the nightly job.
- **Views with the old names and exact old column shapes:** `holdings`, `trade_outcomes`, `positions`,
  `signal_performance` — readers kept working unchanged. `position_state` (peak gain / protected /
  DRIP flags) is the one mutable table; writers are `kairos_exits.update_peak_gain` and
  `kairos_exits.set_position_flags` only.
- **Five write paths:** `record_decision` and `record_fills` (via `record_execution` from every order
  site); `broker_check()` at the **end** of each equity cycle (reqExecutions → missing fills → rebuild →
  compare `positions` to `ib.positions()`; on mismatch it alerts and changes **nothing** — a mismatch
  means a missing fill or corporate action to be found, never papered over); `rebuild_trades()`; and
  the nightly `kairos_outcome_features.py --backfill` (launchd `com.kairos.outcome_features`, 19:30 ET),
  which pulls the IBKR Flex query first. The Flex token/query id reach launchd through the untracked
  wrapper `~/.local/bin/kairos_with_flex_env.sh`, which reads only those two `export` lines from
  `~/.zshrc`. The **token expires 2027-09-28**.
- **Attribution:** `TRUSTED_ATTRIBUTION_SOURCES = explicit, confluence, decision_record`. Opening
  balances, corporate-action exits and SATS→ECHO are `excluded`; pre-Kairos-era trades (before
  2026-05-18) are `none`. 5 Kairos-era orders stay unlinked under the strict same-session rule.
- **Numbers that changed at migration** (the old ones were wrong; don't cite them): Kairos-era realized
  P&L $37.4K, not $67.9K; HOT-INSIDER lifetime ≈ −$8.8K, not +$22K (the old figure is also in the
  `kairos_scorecard` docstring and SECTION 4b evidence up to 2026-09-28). Holding ages are now true FIFO
  lot ages (the old reconciler reset `entry_date` on consolidation).
- `decisions.execution_status` is **unreliable**: `reconcile_submitted_orders` / `cancel_stale_orders`
  match by ticker + quantity, so GTC-style limits that fill after the 45 s wait get marked
  Expired/Cancelled. Use fills, not status, to decide whether something traded. Not yet fixed.
- The `*_legacy` tables are read-only snapshots; drop them on/after **2026-10-28**.
- Checks: `kairos_ledger_acceptance.py --db kairos.db` (38 checks, opens the DB read-only),
  `kairos_selftest_ledger.py`, `kairos_selftest_tradepath.py`.

## The Arbiter (Feedback / Learning Loop)

`kairos_arbiter.py` — **Mistral Medium 3.5**, chosen deliberately for EU jurisdiction and decorrelated
training lineage from the Claude-based council (lineage diversity is a design principle, not an
accident). Runs daily (8 PM ET weekdays, retrospective on the day) and weekly (7 AM ET Saturdays,
broader pattern review), posts to `#kairos-arbiter`.

**Auto-apply (allowlist last widened 2026-09-12):** the Arbiter self-applies changes to the axes in
`kairos_autonomy.AUTO_APPLY_AXES` (`exit_timing`, `profit_floor_pp`, `target_armed.trail_pct`) —
a hardcoded allowlist, widened only by a reviewed code change. Each applied change is logged in
`autonomy_log` with its baseline error and judged after `ROLLBACK_MIN_CLOSES` (10) new closes;
>20% worse → automatic revert (verified by re-reading the live value), then a 14-day cooldown.
Changes to any other axis are still `proposed` rows for J. Two-way Slack interface
(`kairos_arbiter_commander.py`) for questions and verdicts; Approve/Reject buttons via Socket Mode
(`kairos_slack_cards.py`); `!pending` re-posts current proposals from `#kairos-commands`.
Known false alarm: `_judge` reports a refused apply as "ROLLBACK FAILED" in #kairos-alerts even when
the live value is correct (seen 2026-09-22 and 2026-09-28) — check the live value before acting.
2026-09-29: six verdicts (autonomy_log ids 25–30) judged across the old/new ledger were reset to
`pending` and re-baselined on the fills ledger (closes_at_apply 599).

**The ratchet incident** (important cautionary precedent — do not repeat this failure mode): daily
proposal generation ran on frozen/null evidence (28/30 `mfe_pct` rows NULL, so averages were computed
from 2 trades while `n` counted 30) with a **one-sided objective** (only measured giveback, no
forgone-gain term) and no regime memory. This let `trail_pct` compound 8.0 → 4.0 and `profit_floor_pp`
1.0 → 1.906 across three days, producing a wave of premature exits. Remediated with:
- `PROPOSALS_FROZEN` guard in both `propose_all` and `propose_all_params`
- Data layer (landed earlier): `mfe`/`forgone_gain_5d_pct` backfill, `exit_params_snapshot` regime
  tagging, Arbiter ghost-position fix
- Logic layer (landed **2026-08-03** — `kairos_axis_weights` had still been the pre-redesign version,
  which is why the committed 26-assertion selftest was red): two-sided objective, regime window,
  contributing-row filter, cumulative 7-day ±40% band, freshness gate via evidence hash. **26/26 green.**

*The ratchet in one line:* the old objective's error term was give-back, which is never negative — so
every proposal it was physically capable of emitting was a tightening. Netting give-back against
forgone gain is what makes "loosen" representable at all. On live evidence the two objectives disagree
in **direction**: the old one would have ratcheted `profit_floor_pp` UP (1.155 → 1.373) on evidence
that, measured two-sidedly, says loosen. The cumulative band is the guard the ratchet defeated — each
of its three steps was legally within ±25% of what the previous step had just written, so the band is
anchored at the value in force when the window opened. Replaying the ratchet now halts at 4.8 (−40%)
instead of 4.0 (−50%).

**`PROPOSALS_FROZEN = False`** since commit `92d2809`: the loop proposes, and self-applies the
allowlisted axes (see Auto-apply above).

**Known contamination issue (historical — see Trade Record and Known Open Items; kept for context):**
the `signals_fired` field in legacy `trade_outcomes` was populated
from five sources including rationale-text reconstruction in `kairos_execute.py`. This means per-signal
P&L on the dashboard *and* in the Arbiter's learning loop is not clean — trades can be attributed to
signals that didn't actually drive them. HOT-CONGRESS's 105-trade population is overwhelmingly
confluence trades (84 confluence-driven vs. 8 congress-only vs. 13 unstructured), so isolated
HOT-CONGRESS P&L is not trustworthy as-is. **Fixing signal attribution is a high-priority open item**
before drawing further conclusions from per-signal performance data or adjusting axis weights based on it.

**Conviction calibration:** an audit of 63 matured trades showed higher conviction correlated with
*worse* outcomes (Spearman rho -0.26), but this is confounded by hold-duration effects and range
compression (63% of scores cluster 5-7). J rejected recalibrating on this alone; the newly-approved
`exit_timing` weight (+0.28) may organically resolve the confound. Root-cause the dispersion before
touching calibration weights.

**Auto-apply / Phase 3:** granted for the three allowlisted axes (see above). Hermes Agent was
evaluated and rejected: self-modification against live capital is a compliance liability for
B2B/RIA licensing; the structured propose→apply→judge→revert pipeline is the foundation to widen.

## Security / Git Hygiene

Recurring failure pattern: credentials get committed accidentally (Slack bot tokens, Mistral API keys,
Renaissance Capital API key have each leaked into git history at different points). Standard remediation
sequence: move to environment variables → `git filter-repo` to scrub history → rotate the credential →
force-push clean. **Caution**: `git filter-repo` runs have previously wiped uncommitted working-tree
changes as a side effect — commit or stash before running it, not after.

API keys belong in environment variables, never in `kairos_config.json` or other tracked files.
`.gitignore` should exclude both databases, `kalshi_private_key.pem`, runtime state files, and
`.claude/` worktrees.

## Design Principles (apply these by default, don't relitigate)

- **Diagnose against live data before writing code; dry-run before live writes; IBKR is source of
  truth.** J enforces this sequencing strictly.
- **One change at a time**, with explicit attribution tracking — J pushes back hard on bundling
  changes that should be sequenced, because it destroys the ability to attribute an outcome to a cause.
- **Systematic over arbitrary.** No hardcoded thresholds where a self-scaling/data-driven approach is
  possible (e.g. the dashboard's self-calibrating cycle-cadence function replaced a hardcoded 30-minute
  assumption).
- **Tier 0 pre-filtering is a sanity floor only** — never filter on price/volume activity to reduce
  screener batch size. Quiet pre-move setups are exactly what HOT-INSIDER and HOT-CONGRESS need to see
  before the market reprices them; activity-based filtering defeats the core thesis.
- **No Google products/services anywhere in the stack.** Chinese-origin models are excluded from API
  use on trust grounds; open-weight Chinese models run *locally* remain eligible.
- **Model/lineage diversity is intentional**, not incidental — council seats and the Arbiter are chosen
  for decorrelated training lineage, not just capability.
- **Auditable decision trails are non-negotiable** — this is a hard compliance requirement for the
  eventual B2B/RIA licensing business, and should bias every architecture decision toward
  inspectability over cleverness.
- **Direct, honest technical assessments are preferred over reassuring ones.** If something is broken
  or a number is fake/misleading (e.g. a dashboard stat computed from a broken source), say so plainly
  rather than softening it.

## Known Open Items (as of 2026-09-29)

- **Trade-record corrections (2026-09-28 fills-ledger migration).** HOT-INSIDER lifetime realized is
  about −$8.8K (the legacy ledger showed +$22K; +$28.6K of that was a phantom DD split gain). It is
  soft-paused; hard-pausing is J's open decision. Kairos-era realized P&L is $37.4K (legacy said $67.9K).
- **ML model is at chance** (CV 46% on 449 trusted trades after the migration). It now retrains nightly
  in `kairos_outcome_features.py --backfill` (before 2026-09-29 it never retrained unless forced by
  hand). SECTION 5b tells the council to discount scores while CV < 55%.
- **Order-status bug:** `reconcile_submitted_orders` / `cancel_stale_orders` match by ticker+qty, so 60
  BUYs marked Expired/Cancelled/Submitted actually filled. The ledger no longer depends on
  `decisions.execution_status`, but the status column is still wrong going forward.
- **Reallocation BUY leg logs `new_position.quantity`** (post-fill total) as the filled quantity. Ledger
  unaffected; the logging bug remains.
- **5 Kairos-era orders unlinked** to a decision (PRLD, TTWO, HOOD, TWLO, EQIX cover): GTC fills outside
  the same session. Counted in P&L, excluded from learning.
- Signal attribution: **fixed forward 2026-08-03**, but the corpus is split. `_derive_entry_signals`
  documented a source priority and implemented a *union* of all five sources, merging trade-level
  causation with ticker-level coincidence and prose guesses. It is now genuinely tiered (first tier
  wins outright), and the tier is persisted as `trade_outcomes.signal_attribution_source`
  (`explicit`/`confluence` = causal, per `TRUSTED_ATTRIBUTION_SOURCES`; `ticker_context`/
  `rationale_text` = not). All 270 pre-fix rows are stamped `legacy_mixed` and **must be excluded**
  from per-signal analysis — they cannot be un-mixed, the source files are gone. Per-signal P&L stays
  untrustworthy until enough clean rows accumulate; re-check the provenance breakdown before drawing
  conclusions. Note the **param** loop was never affected (it reads `exit_reason` + mfe/give-back/
  forgone, not `signals_fired`).
- Invalidation-level coverage: only 37% of logged theses named a parseable price. The Council prompt is
  fixed going forward; worth re-measuring coverage after a few weeks of new theses before judging the
  price-invalidation mechanism's value.
- Split/data-integrity scan: a CRWD pre-split-vs-post-split yfinance mismatch created a backtest
  anomaly; a full 727-ticker scan for split adjustments during the paper-trading window is warranted
- Conviction dispersion root cause investigation (before any calibration weight changes)
- Tier 0 pre-filter (`skip_tier0: true`) validated in dry-run (603/622 pass, 21.5s) but not yet
  reactivated in production — deferred pending router validation sequencing
- Fundamentals/valuation blind spot: Kairos has no point-in-time fundamentals source, making every
  signal valuation-blind — a HOT-INSIDER name at 60x earnings looks identical to one at 12x. Top
  signal-sharpening item. **Recommended path (scoped 2026-08-03, not started):** SEC EDGAR XBRL
  `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` — free, no vendor, and *point-in-time by
  construction* (every fact carries its `filed` date, so backtests can't leak future data, which
  yfinance fundamentals cannot promise). The plumbing already exists: `kairos_thesis_validity._sec_cik`
  resolves ticker→CIK from `company_tickers.json`, and `data.sec.gov` is already a trusted host in the
  stack for Form 4 intake. Sequence, per diagnose-before-code:
  1. `kairos_fundamentals.py` — pull a SMALL field set (revenue TTM, net income, EPS, shares out, cash,
     debt, operating cash flow), cache to a table, expose `get_fundamentals(ticker, as_of)`; report
     universe coverage before anything consumes it.
  2. **Diagnose** — across the closed-trade corpus, does entry valuation percentile relate to outcome?
     If there is no relationship, the blind spot is not costing anything and this stops here.
  3. Only if (2) shows signal: add a valuation section to the Council prompt (context only, like the
     tax and axis-calibration sections) and observe.
  4. Only after that: let valuation gate or size positions — behavior-changing, needs its own backtest.
- Price-invalidation exit is enabled (`exits.price_invalidation.enabled: true` since 2026-09-28); watch its
  fires in the weekly scorecard.
