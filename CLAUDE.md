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
- Databases: `kairos.db` (primary state) and `kairos_ml_outcomes.db` (ML training corpus), both SQLite.
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
- **Phase 2** — Council reasoning (`claude-sonnet-5` via Anthropic API, upgraded from Sonnet 4).
  Effort levels per call type: high for decisions, medium for routing, low for thesis review.
  Reasoning timeout 240s, max_tokens 8192 (bumped from 180s/4096 after cold-cache timeouts and
  extended-thinking token exhaustion post-Sonnet-5 switch).
- **Phase 3** — Execute via IBKR.
- **Router/analyst layer** — `qwen3.6:35b-a3b` via Ollama, handles news filtering/routing
  (upgraded from qwen3:14b → qwen3:30b-a3b → qwen3.6:35b-a3b).

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

## The Arbiter (Feedback / Learning Loop)

`kairos_arbiter.py` — **Mistral Medium 3.5**, chosen deliberately for EU jurisdiction and decorrelated
training lineage from the Claude-based council (lineage diversity is a design principle, not an
accident). Runs daily (8 PM ET weekdays, retrospective on the day) and weekly (7 AM ET Saturdays,
broader pattern review), posts to `#kairos-arbiter`.

**Mandatory human gate**: the Arbiter only ever writes `proposed` rows and Slack summaries — it never
auto-applies changes to live config. Two-way Slack interface (`kairos_arbiter_commander.py`) lets J
ask questions ("why did you flag AMAT?"), review data, and issue verdicts conversationally. Interactive
Approve/Reject buttons via Socket Mode (`kairos_slack_cards.py`); `!pending` re-posts current proposals
on demand from `#kairos-commands`.

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

**`PROPOSALS_FROZEN` is currently STILL SET** (`kairos_axis_weights.py`, a truthy string). The loop
therefore proposes nothing. Unfreezing (set it to `False`) is J's call and should follow a review of
the live dry-run, not be done as a side effect of other work. Note on unfreeze: `profit_floor_pp` is
currently *gated* (5 contributing in-regime trades < the 10 required), so only `trail_pct` would
propose — that is the regime window working, not a bug.

**Known contamination issue (unresolved):** the `signals_fired` field in `trade_outcomes` is populated
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

**Auto-apply / Phase 3 criteria:** J has expressed intent to eventually grant the Arbiter loop
auto-apply rights (removing the human gate) once evidence quality is proven — explicitly *not* by
adopting a self-modifying agent runtime (Hermes Agent was evaluated and rejected for exactly this
reason: self-modification against live capital is a compliance liability for B2B/RIA licensing, and
the existing structured propose→approve→inject pipeline is the right foundation to graduate rather
than replace). Explicit Phase 3 evidence thresholds for earning auto-apply rights are scoped in
concept but not yet formally documented — worth doing before this becomes urgent.

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

## Known Open Items (as of 2026-08-03)

- **Short positions to flatten** (long-only violation): CARR -104, EME -8, ETN -16, KARD -301. Cover
  with `python kairos_cover_shorts.py` (dry run) then `--execute`, during regular hours — the tool
  refuses quotes more than 10% off the last daily close, because a 06:00 ET read returned a $32.99 ask
  on KARD against a $20.82 close. The root cause is fixed; these are the residue.
- **`PROPOSALS_FROZEN` unfreeze decision** — logic is repaired and 26/26 green; flipping it is J's call
  (see the Arbiter section).
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
- Price-invalidation exit: shipped `enabled: false` — review a dry-run with real fire candidates
  before flipping it on (same deployment pattern as target-armed)
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
- Phase 3 (Arbiter auto-apply) evidence thresholds: scoped conceptually, not yet formally documented
- Sonnet 5 pricing transitions from intro ($2/$10 per M tokens) to standard ($3/$15) on August 31, and
  the updated tokenizer maps the same text to 1.0-1.35x more tokens — factor into any cost projections
