"""
Kairos Tax Efficiency Module — After-Tax NPV Decision Engine

For every open position, calculates after-tax net present value (NPV)
of selling now vs. holding until the long-term capital gains threshold:

  NPV(sell now)  = after-tax proceeds + opportunity cost of freed capital
  NPV(hold)      = after-tax proceeds at long-term rate (conservative 0% growth)

If NPV(sell now) > NPV(hold) → SELL-OK
If NPV(hold) > NPV(sell now) → HOLD-FOR-TAX

Also serves as a pre-execution filter: overrides SELL signals when NPV
favors holding, unless the position has lost more than the stop-loss
percentage from its high (configurable).

Config: kairos_config.json → tax.{short_term_rate, long_term_rate,
        long_term_days, opportunity_cost_rate, stop_loss_override_pct}
"""

import json
import os
import sys
from dataclasses import dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
W = 72

# ── Config ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "short_term_rate": 0.37,
    "long_term_rate": 0.20,
    "long_term_days": 365,
    "opportunity_cost_rate": 0.08,
    "stop_loss_override_pct": 10.0,
}


def load_tax_config() -> dict:
    """Load tax config from kairos_config.json, falling back to defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            cfg.update(data.get("tax", {}))
        except (json.JSONDecodeError, IOError):
            pass
    return cfg


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class LotAnalysis:
    """Tax efficiency analysis for a single holding lot."""
    ticker: str
    quantity: float
    entry_price: float
    entry_date: str
    current_price: float
    holding_days: int
    days_to_long_term: int
    unrealized_gain: float          # total $ gain
    unrealized_gain_pct: float      # % gain
    tax_if_sell_now: float          # $ tax at current rate
    tax_if_long_term: float         # $ tax if waited for LT rate
    tax_savings: float              # $ saved by waiting (raw, before opp cost)
    opportunity_cost: float         # $ opportunity cost of waiting
    npv_sell_now: float             # after-tax proceeds + opportunity value
    npv_hold: float                 # after-tax proceeds if held to LT
    daily_decay_rate: float         # $ tax savings lost per 1% daily drop
    recommendation: str             # "HOLD-FOR-TAX" or "SELL-OK"
    reason: str


@dataclass
class TaxEfficiencyResult:
    """Aggregated tax efficiency analysis for a ticker."""
    ticker: str
    lots: list[LotAnalysis] = field(default_factory=list)
    total_tax_savings: float = 0.0
    hold_for_tax: bool = False
    stop_loss_triggered: bool = False
    override_reason: str = ""


# ── Core analysis ────────────────────────────────────────────────────

def calculate_aftertax_npv(
    entry_price: float,
    current_price: float,
    quantity: float,
    holding_days: int,
    cfg: dict | None = None,
) -> dict:
    """Calculate after-tax NPV of selling now vs. holding to long-term.

    Returns dict with all intermediate values and the recommendation.
    """
    if cfg is None:
        cfg = load_tax_config()

    st_rate = cfg["short_term_rate"]
    lt_rate = cfg["long_term_rate"]
    lt_days = cfg["long_term_days"]
    opp_rate = cfg.get("opportunity_cost_rate", 0.08)

    cost_basis = entry_price * quantity
    market_value = current_price * quantity
    unrealized_gain = market_value - cost_basis
    days_to_lt = max(0, lt_days - holding_days)
    is_long_term = holding_days >= lt_days

    # 1. After-tax proceeds if selling NOW
    if unrealized_gain > 0:
        if is_long_term:
            tax_now = unrealized_gain * lt_rate
        else:
            tax_now = unrealized_gain * st_rate
    else:
        tax_now = 0.0
    aftertax_now = market_value - tax_now

    # 2. After-tax proceeds if holding to long-term (0% growth assumption)
    projected_gain = unrealized_gain  # conservative: same price
    if projected_gain > 0:
        tax_lt = projected_gain * lt_rate
    else:
        tax_lt = 0.0
    aftertax_lt = market_value - tax_lt

    # 3. Opportunity cost of waiting
    # Capital freed by selling now could earn opp_rate annually
    years_waiting = days_to_lt / 365.0
    opportunity_cost = aftertax_now * opp_rate * years_waiting

    # 4. NPV comparison
    npv_sell_now = aftertax_now + opportunity_cost
    npv_hold = aftertax_lt

    # Raw tax savings (before opportunity cost)
    tax_savings = tax_now - tax_lt if unrealized_gain > 0 and not is_long_term else 0.0

    return {
        "aftertax_now": round(aftertax_now, 2),
        "aftertax_lt": round(aftertax_lt, 2),
        "opportunity_cost": round(opportunity_cost, 2),
        "npv_sell_now": round(npv_sell_now, 2),
        "npv_hold": round(npv_hold, 2),
        "tax_now": round(tax_now, 2),
        "tax_lt": round(tax_lt, 2),
        "tax_savings": round(tax_savings, 2),
        "days_to_lt": days_to_lt,
        "is_long_term": is_long_term,
        "hold_recommended": npv_hold > npv_sell_now,
    }


def analyze_lot(
    ticker: str,
    quantity: float,
    entry_price: float,
    entry_date: str,
    current_price: float,
    holding_days: int,
    high_price: float | None = None,
    cfg: dict | None = None,
) -> LotAnalysis:
    """Analyze tax efficiency for a single lot using after-tax NPV."""
    if cfg is None:
        cfg = load_tax_config()

    st_rate = cfg["short_term_rate"]
    lt_rate = cfg["long_term_rate"]
    lt_days = cfg["long_term_days"]

    cost_basis = entry_price * quantity
    market_value = current_price * quantity
    unrealized_gain = market_value - cost_basis
    unrealized_gain_pct = ((current_price - entry_price) / entry_price * 100
                           if entry_price > 0 else 0.0)

    days_to_lt = max(0, lt_days - holding_days)

    # NPV calculation
    npv = calculate_aftertax_npv(entry_price, current_price, quantity,
                                 holding_days, cfg)

    # Daily decay rate: tax savings lost per 1% daily price drop
    one_pct_of_value = market_value * 0.01
    if unrealized_gain > 0 and not npv["is_long_term"]:
        rate_spread = st_rate - lt_rate
        daily_decay = one_pct_of_value * rate_spread
    else:
        daily_decay = 0.0

    # Recommendation based on NPV
    if unrealized_gain <= 0:
        recommendation = "SELL-OK"
        reason = "Position at a loss — no tax savings from waiting"
    elif npv["is_long_term"]:
        recommendation = "SELL-OK"
        reason = "Already qualifies for long-term rate"
    elif npv["hold_recommended"]:
        net_benefit = npv["npv_hold"] - npv["npv_sell_now"]
        recommendation = "HOLD-FOR-TAX"
        reason = (f"NPV(hold)=${npv['npv_hold']:,.0f} > NPV(sell)=${npv['npv_sell_now']:,.0f} "
                  f"(+${net_benefit:,.0f}, {days_to_lt}d to LT)")
    else:
        net_cost = npv["npv_sell_now"] - npv["npv_hold"]
        recommendation = "SELL-OK"
        reason = (f"NPV(sell)=${npv['npv_sell_now']:,.0f} > NPV(hold)=${npv['npv_hold']:,.0f} "
                  f"(opp cost ${npv['opportunity_cost']:,.0f} > tax savings ${npv['tax_savings']:,.0f})")

    return LotAnalysis(
        ticker=ticker,
        quantity=quantity,
        entry_price=round(entry_price, 2),
        entry_date=entry_date,
        current_price=round(current_price, 2),
        holding_days=holding_days,
        days_to_long_term=days_to_lt,
        unrealized_gain=round(unrealized_gain, 2),
        unrealized_gain_pct=round(unrealized_gain_pct, 2),
        tax_if_sell_now=npv["tax_now"],
        tax_if_long_term=npv["tax_lt"],
        tax_savings=npv["tax_savings"],
        opportunity_cost=npv["opportunity_cost"],
        npv_sell_now=npv["npv_sell_now"],
        npv_hold=npv["npv_hold"],
        daily_decay_rate=round(daily_decay, 2),
        recommendation=recommendation,
        reason=reason,
    )


def analyze_ticker(
    ticker: str,
    current_price: float,
    high_price: float | None = None,
) -> TaxEfficiencyResult:
    """Run tax efficiency analysis for all open lots of a ticker.

    Pulls lot data from kairos.db via kairos_log_db.
    """
    from kairos_log_db import get_open_holdings, init_db
    init_db()

    cfg = load_tax_config()
    lots = get_open_holdings(ticker)
    result = TaxEfficiencyResult(ticker=ticker)

    if not lots:
        return result

    for lot in lots:
        la = analyze_lot(
            ticker=ticker,
            quantity=lot["quantity"],
            entry_price=lot["entry_price"],
            entry_date=lot["entry_date"],
            current_price=current_price,
            holding_days=lot["holding_days"] or 0,
            high_price=high_price,
            cfg=cfg,
        )
        result.lots.append(la)
        result.total_tax_savings += la.tax_savings

    # Aggregate recommendation — NPV-based (no dollar threshold)
    stop_loss_pct = cfg["stop_loss_override_pct"]

    any_hold = any(la.recommendation == "HOLD-FOR-TAX" for la in result.lots)
    result.hold_for_tax = any_hold

    # Check stop-loss override: if price has dropped >stop_loss_pct from
    # high, allow the sell even if tax says hold
    if result.hold_for_tax and high_price and high_price > 0:
        drop_from_high = (high_price - current_price) / high_price * 100
        if drop_from_high >= stop_loss_pct:
            result.hold_for_tax = False
            result.stop_loss_triggered = True
            result.override_reason = (
                f"Stop-loss override: {ticker} dropped {drop_from_high:.1f}% from "
                f"high of ${high_price:.2f} (threshold: {stop_loss_pct}%)"
            )

    if result.hold_for_tax:
        result.override_reason = (
            f"HOLD-FOR-TAX: ${result.total_tax_savings:,.0f} tax savings "
            f"exceed ${threshold:,} threshold"
        )

    return result


# ── Pre-execution filter ─────────────────────────────────────────────

def check_tax_override(decision: dict, current_price: float,
                       high_price: float | None = None) -> dict:
    """Pre-execution filter: override SELL if tax savings warrant holding.

    Args:
        decision: the trading decision dict with action, ticker, quantity
        current_price: current market price of the ticker
        high_price: recent high price (for stop-loss check)

    Returns:
        Modified decision dict (action may change to HOLD).
    """
    action = decision.get("action", "").upper()
    ticker = decision.get("ticker", "")

    if action != "SELL":
        return decision

    result = analyze_ticker(ticker, current_price, high_price)

    if result.hold_for_tax:
        print(f"  TAX FILTER: {result.override_reason}")
        for la in result.lots:
            if la.recommendation == "HOLD-FOR-TAX":
                print(f"    Lot {la.entry_date[:10]}: {la.quantity:.0f} shares, "
                      f"savings ${la.tax_savings:,.0f}, {la.days_to_long_term}d to LT")
        decision["action"] = "HOLD"
        decision["quantity"] = 0
        decision["tax_override"] = result.override_reason
        decision["rationale"] = (
            f"[TAX OVERRIDE] {decision.get('rationale', '')} "
            f"— Overridden: {result.override_reason}"
        )
    elif result.stop_loss_triggered:
        print(f"  TAX FILTER: {result.override_reason}")
        print(f"  TAX FILTER: SELL allowed despite tax savings — stop-loss triggered")

    return decision


# ── Prompt section formatter ─────────────────────────────────────────

def format_tax_efficiency_section(tickers_prices: dict[str, float],
                                  tickers_highs: dict[str, float] | None = None,
                                  ) -> str:
    """Build the 'Tax Efficiency Analysis' prompt section.

    Args:
        tickers_prices: {ticker: current_price} for each open position
        tickers_highs: {ticker: high_price} for stop-loss context
    """
    if not tickers_prices:
        return "No open positions to analyze."

    cfg = load_tax_config()
    opp_rate = cfg.get("opportunity_cost_rate", 0.08)
    lines = []
    lines.append(f"Tax rates: ST {cfg['short_term_rate']*100:.0f}% / "
                 f"LT {cfg['long_term_rate']*100:.0f}%  |  "
                 f"Opp cost: {opp_rate*100:.0f}% annual  |  "
                 f"Stop-loss override: {cfg['stop_loss_override_pct']}%")
    lines.append(f"Decision method: after-tax NPV (sell now + opp cost vs. hold to LT at 0% growth)")
    lines.append("")

    total_savings_all = 0.0
    any_hold_for_tax = False

    for ticker, price in tickers_prices.items():
        high = (tickers_highs or {}).get(ticker)
        result = analyze_ticker(ticker, price, high)

        if not result.lots:
            continue

        lines.append(f"  {ticker} (current: ${price:.2f}"
                     + (f", high: ${high:.2f}" if high else "")
                     + ")")

        # Table header
        lines.append(f"    {'Lot':>3}  {'Entry':>10}  {'Shares':>6}  {'Days':>5}  "
                     f"{'toLT':>5}  {'Gain%':>7}  {'TaxSaved':>9}  "
                     f"{'OppCost':>9}  {'NPV Sell':>10}  {'NPV Hold':>10}  {'Action'}")
        lines.append(f"    {'─'*3}  {'─'*10}  {'─'*6}  {'─'*5}  "
                     f"{'─'*5}  {'─'*7}  {'─'*9}  "
                     f"{'─'*9}  {'─'*10}  {'─'*10}  {'─'*14}")

        for i, la in enumerate(result.lots, 1):
            dtl = f"{la.days_to_long_term}d" if la.days_to_long_term > 0 else "—"
            lines.append(
                f"    {i:>3}  {la.entry_date[:10]:>10}  {la.quantity:>6.0f}  "
                f"{la.holding_days:>5}  {dtl:>5}  {la.unrealized_gain_pct:>+6.1f}%  "
                f"${la.tax_savings:>8.0f}  "
                f"${la.opportunity_cost:>8.0f}  "
                f"${la.npv_sell_now:>9,.0f}  ${la.npv_hold:>9,.0f}  "
                f"{la.recommendation}"
            )

        total_savings_all += result.total_tax_savings

        if result.hold_for_tax:
            any_hold_for_tax = True
            lines.append(f"    >>> {result.override_reason}")
        elif result.stop_loss_triggered:
            lines.append(f"    >>> {result.override_reason}")
        lines.append("")

    # Summary
    lines.append(f"  Total tax savings from waiting: ${total_savings_all:,.0f}")
    if any_hold_for_tax:
        lines.append(f"\n  IMPORTANT: HOLD-FOR-TAX is active on positions above.")
        lines.append(f"  SELL signals will be overridden unless stop-loss "
                     f"({cfg['stop_loss_override_pct']}% from high) triggers.")
    else:
        lines.append(f"  No positions favor HOLD-FOR-TAX on NPV basis.")

    return "\n".join(lines)
