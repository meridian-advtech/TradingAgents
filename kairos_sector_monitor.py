"""
Kairos Sector Concentration Monitor

Checks if the current shortlist contains tickers from sectors where the
portfolio already holds >20% of NLV, and returns a warning block to inject
into the reasoning prompt before Claude reasons about the shortlist.

Usage:
    from kairos_sector_monitor import build_sector_warnings
    warning = build_sector_warnings(shortlist, portfolio)
    # Insert warning string into prompt before the reasoning framework

The 20% warn threshold is intentionally lower than the 25% hard limit
enforced by kairos_confluence.check_sector_concentration() — it gives
Claude a heads-up so it can factor the risk in, even when the guardrail
won't block the trade outright.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

WARN_THRESHOLD_PCT = 20.0   # warn if sector already > 20% of NLV
HARD_LIMIT_PCT     = 25.0   # hard cap enforced by kairos_confluence


def get_sector_exposure(portfolio: dict) -> dict[str, float]:
    """Return {sector: total_market_value} for all tracked equity positions.

    Uses avg_cost × quantity as a proxy for market value when live prices
    are not in the portfolio dict.
    """
    try:
        from kairos_confluence import lookup_sector
    except ImportError:
        return {}

    positions    = portfolio.get("positions", [])
    ticker_prices = portfolio.get("ticker_prices", {})
    exposure: dict[str, float] = {}

    for p in positions:
        sym = p["symbol"]
        qty = float(p.get("quantity", 0))
        # Prefer live price from ticker_prices; fall back to avg_cost
        price = float(ticker_prices.get(sym) or p.get("avg_cost", 0))
        mkt_val = qty * price
        if mkt_val <= 0:
            continue
        sector = lookup_sector(sym)
        exposure[sector] = exposure.get(sector, 0.0) + mkt_val

    return exposure


def build_sector_warnings(shortlist: list, portfolio: dict) -> str:
    """Build a warning block for the reasoning prompt.

    For each shortlisted ticker whose sector already exceeds WARN_THRESHOLD_PCT
    of NLV, emits a line:
        WARNING: Portfolio already has X% exposure to [sector].
        New positions in [ticker] increase concentration risk.

    Returns the full block as a string, or "" if no warnings triggered.

    Args:
        shortlist:  tickers being evaluated this cycle (from kairos_screener)
        portfolio:  portfolio dict from gather_portfolio()
                    (must contain 'account' with NetLiquidation)
    """
    if not shortlist:
        return ""

    try:
        from kairos_confluence import lookup_sector
    except ImportError:
        return ""

    acct = portfolio.get("account", {})
    nlv  = float(acct.get("NetLiquidation", 0))
    if nlv <= 0:
        return ""

    sector_exposure = get_sector_exposure(portfolio)
    if not sector_exposure:
        return ""

    seen_sectors: set = set()
    warnings: list[tuple[str, str, float]] = []   # (ticker, sector, pct)

    for ticker in shortlist:
        sector = lookup_sector(ticker)
        if not sector or sector == "unknown" or sector in seen_sectors:
            continue
        seen_sectors.add(sector)

        sector_val = sector_exposure.get(sector, 0.0)
        sector_pct = sector_val / nlv * 100.0

        if sector_pct > WARN_THRESHOLD_PCT:
            warnings.append((ticker, sector, sector_pct))

    if not warnings:
        return ""

    W = 72
    lines = [
        "",
        "╔" + "═" * W + "╗",
        ("║  ⚠  SECTOR CONCENTRATION WARNINGS" + " " * (W - 35) + "║"),
        ("║  Review before trading — these sectors are approaching the 25% cap." + " " * (W - 68) + "║"),
        "╠" + "═" * W + "╣",
    ]
    for ticker, sector, pct in warnings:
        msg1 = f"  WARNING: Portfolio already has {pct:.1f}% exposure to {sector}."
        msg2 = f"  New positions in {ticker} increase concentration risk (max: {HARD_LIMIT_PCT:.0f}%)."
        lines.append(f"║{msg1:<{W}}║")
        lines.append(f"║{msg2:<{W}}║")
        lines.append("║" + " " * W + "║")
    lines.append("╚" + "═" * W + "╝")
    lines.append("")

    return "\n".join(lines)
