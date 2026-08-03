"""
Kairos Tax Loss Harvesting — Daily evaluation at 9:35 AM ET

For each open position with an unrealized loss exceeding 5%:
  1. Check holding period (>30 days to avoid churn)
  2. Find a replacement ticker in the same sector (different ticker)
  3. Estimate tax savings from harvesting the loss
  4. Check wash sale risk before recommending

Integrates with kairos_thesis_review.py's daily 9:35 AM review cycle.

Usage:
    from kairos_tax_harvest import evaluate_harvest_candidates
    harvests = evaluate_harvest_candidates(holdings_with_prices, ib, dry_run)
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
W = 72

# Minimum unrealized loss to consider harvesting
MIN_LOSS_PCT = 5.0
# Minimum holding days (avoid short-term churn)
MIN_HOLDING_DAYS = 30

SCHEMA_TAX_HARVEST_LOG = """
CREATE TABLE IF NOT EXISTS tax_harvest_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker               TEXT NOT NULL,
    eval_date            TEXT NOT NULL,
    unrealized_loss_pct  REAL NOT NULL,
    unrealized_loss_usd  REAL,
    holding_days         INTEGER,
    harvest_recommended  INTEGER NOT NULL DEFAULT 0,
    reason               TEXT NOT NULL,
    replacement_ticker   TEXT,
    estimated_tax_saving REAL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_HARVEST_REPLACEMENTS = """
CREATE TABLE IF NOT EXISTS harvest_replacements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker_sold       TEXT NOT NULL,
    ticker_replacement TEXT NOT NULL,
    harvest_date      TEXT NOT NULL,
    loss_realized     REAL NOT NULL,
    blocked_until     TEXT NOT NULL,
    injected_tier_c   INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _get_connection() -> sqlite3.Connection:
    from kairos_log_db import get_connection
    return get_connection()


def init_harvest_tables() -> None:
    """Create tax_harvest_log and harvest_replacements tables."""
    conn = _get_connection()
    conn.executescript(SCHEMA_TAX_HARVEST_LOG + SCHEMA_HARVEST_REPLACEMENTS)
    conn.commit()
    conn.close()


def _log_harvest_eval(ticker: str, loss_pct: float, loss_usd: float,
                      holding_days: int, recommended: bool, reason: str,
                      replacement: str | None = None,
                      tax_saving: float | None = None) -> int:
    """Write one row to tax_harvest_log. Returns row ID."""
    init_harvest_tables()
    conn = _get_connection()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = conn.execute(
        """INSERT INTO tax_harvest_log
           (ticker, eval_date, unrealized_loss_pct, unrealized_loss_usd,
            holding_days, harvest_recommended, reason,
            replacement_ticker, estimated_tax_saving)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, today, round(loss_pct, 2), round(loss_usd, 2),
         holding_days, 1 if recommended else 0, reason,
         replacement, round(tax_saving, 2) if tax_saving else None),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def _log_replacement(ticker_sold: str, ticker_replacement: str,
                     loss_realized: float) -> int:
    """Record a harvest replacement pairing. Returns row ID."""
    init_harvest_tables()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    blocked_until = (datetime.now(timezone.utc) + timedelta(days=31)).strftime("%Y-%m-%d")
    conn = _get_connection()
    cur = conn.execute(
        """INSERT INTO harvest_replacements
           (ticker_sold, ticker_replacement, harvest_date,
            loss_realized, blocked_until, injected_tier_c)
           VALUES (?, ?, ?, ?, ?, 1)""",
        (ticker_sold, ticker_replacement, today, round(loss_realized, 2),
         blocked_until),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def _find_replacement(ticker: str, sector: str) -> str | None:
    """Find a replacement ticker in the same sector from Tier A/B.

    Must be a different ticker in the same sector category.
    Returns the first match, or None if no suitable replacement exists.
    """
    from kairos_confluence import _build_sector_map

    sector_map = _build_sector_map()
    # Find all tickers in the same sector, excluding the one being sold
    candidates = [t for t, s in sector_map.items() if s == sector and t != ticker]

    if not candidates:
        return None

    # Prefer tickers we don't already hold
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        held = set(
            r["ticker"] for r in conn.execute(
                "SELECT DISTINCT ticker FROM holdings WHERE sold_date IS NULL"
            ).fetchall()
        )
        conn.close()
        # Filter to non-held candidates first
        non_held = [t for t in candidates if t not in held]
        if non_held:
            candidates = non_held
    except Exception:
        pass

    # Return the first candidate alphabetically for determinism
    candidates.sort()
    return candidates[0]


def _estimate_projected_gains() -> float:
    """Estimate projected short-term gains from recent fills.

    Looks at open positions with unrealized gains to estimate
    how much tax savings a harvested loss could offset.
    """
    try:
        conn = _get_connection()
        rows = conn.execute("""
            SELECT ticker,
                   SUM(quantity) AS total_qty,
                   ROUND(SUM(entry_price * quantity) / SUM(quantity), 2) AS avg_cost
            FROM holdings
            WHERE sold_date IS NULL
            GROUP BY ticker
        """).fetchall()
        conn.close()

        # Sum up unrealized gains (we only care about positive side)
        total_gains = 0.0
        for r in rows:
            # We don't have current prices here, so use recent realized gains
            pass

        # Fallback: look at realized gains this year
        conn = _get_connection()
        year_start = datetime.now(timezone.utc).strftime("%Y-01-01")
        row = conn.execute("""
            SELECT COALESCE(SUM(
                (sold_price - entry_price) * quantity
            ), 0) AS realized_gains
            FROM holdings
            WHERE sold_date IS NOT NULL
              AND sold_price > entry_price
              AND date(sold_date) >= date(?)
        """, (year_start,)).fetchone()
        conn.close()
        return float(row["realized_gains"]) if row else 0.0
    except Exception:
        return 0.0


def evaluate_harvest_candidates(holdings_data: list[dict],
                                dry_run: bool = False) -> list[dict]:
    """Evaluate all open positions for tax loss harvesting.

    Args:
        holdings_data: list of dicts with keys:
            ticker, total_qty, avg_cost, current_price, holding_days
        dry_run: if True, log evaluations but don't recommend execution

    Returns list of harvest recommendations (dicts with full context).
    """
    init_harvest_tables()

    from kairos_confluence import lookup_sector
    from kairos_tax_efficiency import load_tax_config

    tax_cfg = load_tax_config()
    st_rate = tax_cfg["short_term_rate"]

    recommendations = []
    projected_gains = _estimate_projected_gains()

    print(f"\n  {'─' * W}")
    print(f"  TAX LOSS HARVEST EVALUATION")
    print(f"  YTD realized gains: ${projected_gains:,.0f}")
    print(f"  Min loss threshold: {MIN_LOSS_PCT}%  |  Min holding: {MIN_HOLDING_DAYS}d")
    print(f"  {'─' * W}")

    for h in holdings_data:
        ticker = h["ticker"]
        avg_cost = h["avg_cost"]
        current_price = h["current_price"]
        total_qty = h["total_qty"]
        holding_days = h["holding_days"]

        if current_price is None or avg_cost <= 0:
            continue

        return_pct = (current_price - avg_cost) / avg_cost * 100
        unrealized_loss_usd = (avg_cost - current_price) * total_qty

        # Only evaluate positions with losses
        if return_pct >= 0:
            continue

        loss_pct = abs(return_pct)

        # Check 1: Loss threshold
        if loss_pct < MIN_LOSS_PCT:
            _log_harvest_eval(
                ticker, -loss_pct, unrealized_loss_usd, holding_days,
                recommended=False,
                reason=f"Loss {loss_pct:.1f}% below {MIN_LOSS_PCT}% threshold",
            )
            continue

        # Check 2: Holding period
        if holding_days < MIN_HOLDING_DAYS:
            reason = f"Held only {holding_days}d (min {MIN_HOLDING_DAYS}d to avoid churn)"
            print(f"    {ticker}: {return_pct:+.1f}% — skip: {reason}")
            _log_harvest_eval(
                ticker, -loss_pct, unrealized_loss_usd, holding_days,
                recommended=False, reason=reason,
            )
            continue

        # Check 3: Wash sale risk — was this ticker recently repurchased?
        try:
            from kairos_wash_sale import check_wash_sale_violation
            ws = check_wash_sale_violation(ticker, current_price, avg_cost)
            if ws["violation"]:
                reason = (f"Wash sale conflict: repurchased on {ws['repurchase_date']}, "
                          f"loss would be disallowed")
                print(f"    {ticker}: {return_pct:+.1f}% — skip: {reason}")
                _log_harvest_eval(
                    ticker, -loss_pct, unrealized_loss_usd, holding_days,
                    recommended=False, reason=reason,
                )
                continue
        except Exception as exc:
            print(f"    {ticker}: WARNING wash sale check failed: {exc}")

        # Check 4: Find replacement in same sector
        sector = lookup_sector(ticker)
        replacement = _find_replacement(ticker, sector)

        if replacement is None and sector != "unknown":
            reason = f"No replacement available in sector {sector}"
            print(f"    {ticker}: {return_pct:+.1f}% — skip: {reason}")
            _log_harvest_eval(
                ticker, -loss_pct, unrealized_loss_usd, holding_days,
                recommended=False, reason=reason,
            )
            continue

        # Check 5: Would the loss meaningfully offset gains?
        estimated_tax_saving = unrealized_loss_usd * st_rate
        if projected_gains <= 0 and estimated_tax_saving < 100:
            reason = (f"No projected gains to offset and tax saving "
                      f"${estimated_tax_saving:.0f} < $100 threshold")
            print(f"    {ticker}: {return_pct:+.1f}% — skip: {reason}")
            _log_harvest_eval(
                ticker, -loss_pct, unrealized_loss_usd, holding_days,
                recommended=False, reason=reason,
                replacement=replacement, tax_saving=estimated_tax_saving,
            )
            continue

        # All checks passed — recommend harvest
        tax_class = "long-term" if holding_days >= 365 else "short-term"
        reason = (f"HARVEST-LOSS: {loss_pct:.1f}% unrealized loss "
                  f"(${unrealized_loss_usd:,.0f}), "
                  f"est. tax saving ${estimated_tax_saving:,.0f} at {st_rate*100:.0f}%, "
                  f"replacement: {replacement or 'none'}")

        print(f"    {ticker}: {return_pct:+.1f}% ({holding_days}d, {tax_class}) "
              f"→ HARVEST recommended")
        print(f"      Loss: ${unrealized_loss_usd:,.0f}  "
              f"Tax saving: ${estimated_tax_saving:,.0f}  "
              f"Replacement: {replacement or 'N/A'}")

        _log_harvest_eval(
            ticker, -loss_pct, unrealized_loss_usd, holding_days,
            recommended=True, reason=reason,
            replacement=replacement, tax_saving=estimated_tax_saving,
        )

        recommendations.append({
            "ticker": ticker,
            "total_qty": total_qty,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "holding_days": holding_days,
            "loss_pct": round(loss_pct, 2),
            "loss_usd": round(unrealized_loss_usd, 2),
            "tax_saving": round(estimated_tax_saving, 2),
            "replacement": replacement,
            "sector": sector,
            "reason": reason,
        })

    if not recommendations:
        print(f"\n  No harvest candidates this cycle")
    else:
        print(f"\n  {len(recommendations)} harvest candidate(s) identified")

    return recommendations


def execute_harvest(rec: dict, ib, dry_run: bool = False) -> dict | None:
    """Execute a single tax loss harvest sell.

    Calls the existing sell infrastructure, then injects the replacement
    into Tier C for future purchase.

    Returns the execution result dict, or None if skipped.
    """
    ticker = rec["ticker"]
    total_qty = rec["total_qty"]
    avg_cost = rec["avg_cost"]
    current_price = rec["current_price"]
    holding_days = rec["holding_days"]
    replacement = rec["replacement"]
    loss_usd = rec["loss_usd"]

    sell_reason = (f"TAX-HARVEST: {rec['loss_pct']:.1f}% loss "
                   f"(${loss_usd:,.0f}), "
                   f"est. tax saving ${rec['tax_saving']:,.0f}")

    if dry_run:
        print(f"    DRY RUN — would harvest {total_qty} {ticker}")
        return None

    if ib is None:
        print(f"    IBKR unavailable — harvest logged but not executed")
        return None

    # Final wash sale check before execution
    try:
        from kairos_wash_sale import check_wash_sale_violation
        ws = check_wash_sale_violation(ticker, current_price, avg_cost)
        if ws["violation"]:
            print(f"    BLOCKED: Wash sale — {ticker} repurchased on {ws['repurchase_date']}")
            return None
    except Exception as exc:
        print(f"    WARNING: Final wash sale check failed: {exc}")

    # Execute the sell
    from kairos_stoploss import _place_market_sell

    print(f"    SELLING {total_qty} {ticker} (TAX-HARVEST)")
    execution = _place_market_sell(ib, ticker, total_qty)
    if execution.get("oversell_blocked"):
        # No order was sent — do not log a close.
        print(f"    {ticker}: harvest NOT sent — {execution.get('reason')}")
        return None
    sell_price = execution.get("fill_price", current_price)

    # Log via the shared sell path
    from kairos_thesis_review import _log_and_alert
    _log_and_alert(ticker, total_qty, avg_cost, sell_price,
                   holding_days, sell_reason, execution)

    # Record the replacement pairing
    if replacement:
        _log_replacement(ticker, replacement, loss_usd)

        # Inject replacement into Tier C watchlist
        try:
            from kairos_tier_c import add as tier_c_add
            from kairos_confluence import lookup_sector
            sector = lookup_sector(replacement)
            blocked_until = (datetime.now(timezone.utc) + timedelta(days=31)).strftime("%Y-%m-%d")
            tier_c_add(
                ticker=replacement,
                name=f"Tax harvest replacement for {ticker}",
                sector=sector,
                reason=(f"TAX-HARVEST replacement candidate, "
                        f"eligible after {blocked_until}"),
            )
            print(f"    Replacement {replacement} → Tier C "
                  f"(eligible after {blocked_until})")
        except Exception as exc:
            print(f"    WARNING: Tier C injection failed: {exc}")

    return execution
