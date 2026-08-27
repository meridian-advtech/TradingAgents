"""
Kairos Confluence Scoring — NLV-Based Conviction Sizing

Computes a confluence score based on how many independent signals fired
on the same ticker.  Higher confluence = stronger conviction = larger
position as a percentage of Net Liquidation Value (NLV).

Conviction tiers (% of NLV):
    Score 1       → Single Signal  (1.0% NLV)    ~$10K at $1M NLV
    Score 2–3     → Medium         (1.5% NLV)    ~$15K
    Score 4–5     → High           (2.0% NLV)    ~$20K
    Score 6+      → Very High      (2.5% NLV)    ~$25K

Portfolio-level guardrails (enforced in kairos_execute.py):
    Cash reserve:         1% of NLV must remain as cash (commissions + settlement timing)
    Sector concentration: Max 25% of NLV in any single sector
    Single position:      Max 10% of NLV in any single stock

Usage:
    from kairos_confluence import compute_confluence, compute_position_size

    conf = compute_confluence(["HOT-EARNINGS", "HOT-INSIDER", "HOT-OPTIONS"])
    qty  = compute_position_size(conf, nlv=1_000_000, ref_price=197.04)
    # → 126 shares (~$25K at Very High tier)
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Signal point weights
DEFAULT_SIGNAL_POINTS = {
    "HOT-EARNINGS":  2,
    "HOT-RSI":       1,
    "HOT-INSIDER":   2,
    "HOT-CONGRESS":  1,
    "HOT-REVERSION": 1,
    "HOT-KALSHI":    1,
    "HOT-OPTIONS":   2,
}

# NLV-based conviction tiers
DEFAULT_TIERS = [
    {"min_score": 1, "name": "Single Signal", "nlv_pct": 0.010},  # 1.0%
    {"min_score": 2, "name": "Medium",        "nlv_pct": 0.015},  # 1.5%
    {"min_score": 4, "name": "High",          "nlv_pct": 0.020},  # 2.0%
    {"min_score": 6, "name": "Very High",     "nlv_pct": 0.025},  # 2.5%
]

# Portfolio guardrail defaults
DEFAULT_GUARDRAILS = {
    "min_cash_reserve_pct": 0.01,       # 1% of NLV — covers commissions + settlement timing only
    "max_sector_concentration_pct": 0.25,  # 25% of NLV in any sector
    "max_single_position_pct": 0.10,    # 10% of NLV in any stock
}

# Sector mapping: universe category → sector label
# Built from kairos_universe.json categories
_SECTOR_CACHE: dict[str, str] | None = None


def _load_confluence_config() -> dict:
    """Load confluence config from kairos_config.json, with defaults."""
    config_file = os.path.join(SCRIPT_DIR, "kairos_config.json")
    result = {
        "signal_points": dict(DEFAULT_SIGNAL_POINTS),
        "tiers": list(DEFAULT_TIERS),
        "guardrails": dict(DEFAULT_GUARDRAILS),
    }
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            conf = cfg.get("confluence", {})
            if "signal_points" in conf:
                result["signal_points"].update(conf["signal_points"])
            if "tiers" in conf:
                result["tiers"] = conf["tiers"]
            if "guardrails" in conf:
                result["guardrails"].update(conf["guardrails"])
        except (json.JSONDecodeError, IOError):
            pass
    result["tiers"].sort(key=lambda t: t["min_score"])
    return result


# ── Sector lookup ─────────────────────────────────────────────────────

def _build_sector_map() -> dict[str, str]:
    """Build {ticker: sector} mapping from kairos_universe.json categories."""
    global _SECTOR_CACHE
    if _SECTOR_CACHE is not None:
        return _SECTOR_CACHE

    universe_file = os.path.join(SCRIPT_DIR, "kairos_universe.json")
    mapping: dict[str, str] = {}

    if os.path.exists(universe_file):
        try:
            with open(universe_file) as f:
                universe = json.load(f)

            # Map equity categories to sector labels
            category_to_sector = {
                "mega_cap": "mega_cap",
                "large_cap": "large_cap",
                "mid_cap_growth": "mid_cap_growth",
                "value_dividend": "value_dividend",
                "healthcare_biotech": "healthcare_biotech",
                "financials": "financials",
                "energy_materials": "energy_materials",
                "reits_real_estate": "reits_real_estate",
                "industrials_transport": "industrials_transport",
                "consumer_tech": "consumer_tech",
            }

            # Support tiered structure (tier_a.equities) and legacy (equities)
            equities = universe.get("tier_a", {}).get("equities", universe.get("equities", {}))
            for cat, sector_label in category_to_sector.items():
                tickers = equities.get(cat, [])
                for t in tickers:
                    # First category wins (some tickers may appear in multiple)
                    if t not in mapping:
                        mapping[t] = sector_label

            # ETFs get their own category
            etfs = universe.get("tier_a", {}).get("etfs", universe.get("etfs", {}))
            for cat, tickers in etfs.items():
                for t in tickers:
                    if t not in mapping:
                        mapping[t] = f"etf_{cat}"

            # Tier B tickers — use their sector field
            for entry in universe.get("tier_b", {}).get("tickers", []):
                sym = entry["symbol"] if isinstance(entry, dict) else entry
                sector = entry.get("sector", "tier_b") if isinstance(entry, dict) else "tier_b"
                if sym not in mapping:
                    mapping[sym] = sector

        except (json.JSONDecodeError, IOError):
            pass

    _SECTOR_CACHE = mapping
    return mapping


def lookup_sector(ticker: str) -> str:
    """Return sector label for a ticker. Returns 'unknown' if not found."""
    mapping = _build_sector_map()
    return mapping.get(ticker, "unknown")


# ── Confluence scoring ────────────────────────────────────────────────

def compute_confluence(
    signal_tags: list[str],
    config: dict | None = None,
) -> dict:
    """Compute confluence score and conviction tier.

    Returns:
        {
            "score": int,
            "tier": str,
            "nlv_pct": float,        # e.g. 0.025 for Very High
            "signals_detail": dict,   # {signal: points}
        }
    """
    if config is None:
        config = _load_confluence_config()

    points_table = config["signal_points"]
    tiers = config["tiers"]

    # Defense-in-depth: exclude HARD-paused signals from confluence entirely,
    # even if a hard tag reaches here via some path (the generation chokepoint
    # already strips them from the summary, but confluence must not credit a
    # genuinely-cut signal under any circumstance). SOFT-paused signals are
    # intentionally allowed to score — "never fires alone" still lets them
    # confirm another healthy signal's trade, which is the point of soft mode.
    try:
        from kairos_signals import signal_pause_mode
        _pm = signal_pause_mode
    except Exception:
        _pm = lambda s: None

    signals_detail = {}
    for tag in signal_tags:
        tag = tag.strip().upper()
        if _pm(tag) == "hard":
            continue
        if tag in points_table:
            signals_detail[tag] = points_table[tag]
        elif tag.startswith("HOT"):
            signals_detail[tag] = 1

    score = sum(signals_detail.values())

    tier_name, nlv_pct = _tier_and_size(score, tiers)

    return {
        "score": score,
        "tier": tier_name,
        "nlv_pct": nlv_pct,
        "signals_detail": signals_detail,
    }


def _tier_and_size(score: float, tiers: list) -> tuple[str, float]:
    """Map a confluence score to (tier name, nlv_pct), interpolating size.

    The tier NAME is still a step function — it labels a band, and the
    reasoning prompt, alerts and the ML ledger all read it as a category.

    The SIZE is piecewise-linear between tier anchors rather than a step.
    Signal points are floats (evidence adjusts them; see kairos_axis_weights),
    and a step function makes that lever lumpy in exactly the wrong way: with
    anchors at 1/2/4/6, moving HOT-INSIDER 2.0 -> 1.9 would cut a single-signal
    position 33% (1.5% -> 1.0% NLV) while 2.0 -> 3.0 would change nothing at
    all. Small evidence, no effect; slightly larger evidence, a cliff. That is
    the same shape as the July ratchet incident.

    Interpolation preserves current behaviour EXACTLY at whole-number scores
    (1 -> 1.0%, 2 -> 1.5%, 4 -> 2.0%, 6 -> 2.5%) — it is a strict superset of
    the step behaviour, so nothing changes until a point value moves off an
    integer. Below the first anchor: no position. Above the last: clamped.
    """
    if score <= 0 or not tiers:
        return "None", 0.0

    anchors = sorted(
        ((float(t["min_score"]), float(t["nlv_pct"]), str(t["name"])) for t in tiers),
        key=lambda a: a[0],
    )

    # Below the lowest anchor is not a fractional position — it is no signal.
    if score < anchors[0][0]:
        return "None", 0.0

    # Tier name: highest anchor whose threshold the score has reached.
    tier_name = anchors[0][2]
    for min_score, _pct, name in anchors:
        if score >= min_score:
            tier_name = name

    # At or past the top anchor, clamp — do not extrapolate size upward.
    if score >= anchors[-1][0]:
        return tier_name, anchors[-1][1]

    for (lo_s, lo_p, _ln), (hi_s, hi_p, _hn) in zip(anchors, anchors[1:]):
        if lo_s <= score < hi_s:
            span = hi_s - lo_s
            frac = 0.0 if span <= 0 else (score - lo_s) / span
            return tier_name, lo_p + frac * (hi_p - lo_p)

    return tier_name, anchors[-1][1]


def compute_position_size(
    confluence: dict,
    nlv: float,
    ref_price: float,
) -> int:
    """Compute share quantity from confluence tier and NLV.

    Args:
        confluence: output of compute_confluence()
        nlv:        net liquidation value in USD
        ref_price:  current share price

    Returns:
        integer share quantity (0 if no signals or price unavailable)
    """
    nlv_pct = confluence.get("nlv_pct", 0.0)
    if nlv_pct <= 0 or ref_price is None or ref_price <= 0 or nlv <= 0:
        return 0

    target_usd = nlv * nlv_pct
    return int(target_usd / ref_price)


def check_mode_c_regime_eligible() -> tuple[bool, int, str]:
    """Check if Mode C (conviction-only trades) is allowed by the current regime.

    Reads .kairos_regime_state.json to determine regime guardrails.
    This is a backstop — Claude should already respect the prompt, but
    the executor calls this to hard-block Mode C when the regime forbids it.

    Returns:
        (allowed, min_conviction, regime_name)
        - allowed: False if Mode C is disabled by regime
        - min_conviction: minimum conviction score required (raised in CAUTION)
        - regime_name: current regime string for logging
    """
    regime_state_file = os.path.join(SCRIPT_DIR, ".kairos_regime_state.json")
    try:
        from kairos_regime import REGIMES
        with open(regime_state_file) as f:
            state = json.load(f)
        regime = state.get("regime", "NORMAL")
        guardrails = REGIMES.get(regime, REGIMES["NORMAL"])["guardrails"]
        return (
            guardrails.get("mode_c_allowed", True),
            guardrails.get("mode_c_min_conviction", 4),
            regime,
        )
    except (IOError, json.JSONDecodeError, ImportError, KeyError):
        return True, 4, "NORMAL"


def format_confluence_for_display(confluence: dict) -> str:
    """One-line summary for logs and Slack alerts."""
    signals = list(confluence.get("signals_detail", {}).keys())
    nlv_pct = confluence.get("nlv_pct", 0)
    return (
        f"{confluence['tier']} (score {confluence['score']}, "
        f"{nlv_pct:.1%} NLV) — "
        f"{', '.join(signals) if signals else 'no signals'}"
    )


# ── Signal tag lookup ─────────────────────────────────────────────────

def get_ticker_signals(ticker: str) -> list[str]:
    """Load signal tags for a specific ticker from kairos_signal_summary.json."""
    summary_file = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
    if not os.path.exists(summary_file):
        return []
    try:
        with open(summary_file) as f:
            data = json.load(f)
        signal_tags = data.get("signal_tags", {})
        if signal_tags is None:
            signal_tags = {}
        tags = signal_tags.get(ticker, [])
        if tags is None:
            tags = []
        tags = list(tags)

        options_file = os.path.join(SCRIPT_DIR, "kairos_options_activity.json")
        if os.path.exists(options_file):
            with open(options_file) as f:
                opts = json.load(f)
            hits = opts.get("hits", {})
            if hits is None:
                hits = {}
            if ticker in hits:
                if "HOT-OPTIONS" not in tags:
                    tags.append("HOT-OPTIONS")

        return tags
    except (json.JSONDecodeError, IOError):
        return []


# ── Portfolio guardrails ──────────────────────────────────────────────

def load_guardrails(config: dict | None = None) -> dict:
    """Load portfolio guardrail thresholds."""
    if config is None:
        config = _load_confluence_config()
    return config["guardrails"]


def check_cash_reserve(
    cash: float,
    nlv: float,
    proposed_spend: float,
    guardrails: dict | None = None,
) -> tuple[bool, str]:
    """Check if proposed spend would breach cash reserve.

    Returns (allowed, reason).
    """
    if guardrails is None:
        guardrails = load_guardrails()
    min_pct = guardrails["min_cash_reserve_pct"]
    min_cash = nlv * min_pct
    remaining = cash - proposed_spend

    # Guard 1: Prevent cash from going negative
    if remaining < 0:
        return False, f"Trade would make cash negative: ${remaining:,.0f} remaining"

    # Guard 2: Prevent proposed spend from exceeding available cash
    if remaining - proposed_spend < 0:
        return False, f"Insufficient cash for trade: ${cash:,.0f} available, ${proposed_spend:,.0f} needed"

    if remaining < min_cash:
        return False, (
            f"Cash reserve breach: ${remaining:,.0f} remaining "
            f"< ${min_cash:,.0f} ({min_pct:.0%} of ${nlv:,.0f} NLV)"
        )
    return True, f"Cash OK: ${remaining:,.0f} remaining (reserve: ${min_cash:,.0f})"


def check_sector_concentration(
    sector: str,
    current_sector_value: float,
    proposed_spend: float,
    nlv: float,
    guardrails: dict | None = None,
) -> tuple[bool, str]:
    """Check if proposed trade would breach sector concentration limit.

    Returns (allowed, reason).
    """
    if guardrails is None:
        guardrails = load_guardrails()
    max_pct = guardrails["max_sector_concentration_pct"]
    max_value = nlv * max_pct
    new_value = current_sector_value + proposed_spend

    if new_value > max_value:
        return False, (
            f"Sector '{sector}' concentration breach: "
            f"${new_value:,.0f} > ${max_value:,.0f} ({max_pct:.0%} of NLV)"
        )
    return True, f"Sector '{sector}' OK: ${new_value:,.0f} / ${max_value:,.0f}"


def check_single_position(
    ticker: str,
    current_position_value: float,
    proposed_spend: float,
    nlv: float,
    guardrails: dict | None = None,
) -> tuple[bool, str]:
    """Check if proposed trade would breach single-position limit.

    Returns (allowed, reason).
    """
    if guardrails is None:
        guardrails = load_guardrails()
    max_pct = guardrails["max_single_position_pct"]
    max_value = nlv * max_pct
    new_value = current_position_value + proposed_spend

    if new_value > max_value:
        return False, (
            f"Single position breach for {ticker}: "
            f"${new_value:,.0f} > ${max_value:,.0f} ({max_pct:.0%} of NLV)"
        )
    return True, f"{ticker} position OK: ${new_value:,.0f} / ${max_value:,.0f}"
