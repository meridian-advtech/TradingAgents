"""
Kairos Crypto Executor — Simulated Trading

Reads the latest crypto decision from kairos_crypto_decisions.log,
applies risk guards (from kairos_crypto_risk.py) + confluence sizing,
and records a simulated fill using CoinGecko prices.  No orders are
submitted to IBKR.

Simulated positions are tracked in the crypto_holdings table and
displayed on the dashboard with a (SIM) label.

Usage:
    python3 kairos_crypto_execute.py          # Execute latest decision
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CRYPTO_LOG = os.path.join(SCRIPT_DIR, "kairos_crypto_decisions.log")
CRYPTO_CONFIG = os.path.join(SCRIPT_DIR, "kairos_crypto_config.json")
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
W = 72

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
    "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "matic-network",
    "LINK": "chainlink",
}


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


def _load_crypto_config() -> dict:
    if os.path.exists(CRYPTO_CONFIG):
        with open(CRYPTO_CONFIG) as f:
            return json.load(f)
    return {}


def _clean_symbol(raw: str) -> str:
    """Strip whitespace and trailing non-alpha chars from a crypto symbol.

    Fixes 'ETH s', 'ETH b', 'BTC  ' etc. that slip through from decision parsing.
    """
    import re
    cleaned = raw.strip()
    # Remove trailing single characters that aren't part of the symbol
    # e.g. "ETH s" → "ETH", "BTC b" → "BTC"
    cleaned = re.sub(r'\s+[a-zA-Z]$', '', cleaned)
    return cleaned.strip()


def _has_pending_order(asset: str) -> bool:
    """Check kairos.db for an open/pending order for this asset in the last 24h.

    Returns True if a 'Submitted' row exists within the last 24 hours.
    """
    import sqlite3

    if not os.path.exists(DB_PATH):
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now(timezone.utc)
                  - __import__('datetime').timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        row = conn.execute(
            """SELECT COUNT(*) FROM crypto_decisions
               WHERE asset = ? AND execution_status = 'Submitted'
               AND timestamp >= ?""",
            (asset, cutoff),
        ).fetchone()
        conn.close()
        return (row[0] or 0) > 0
    except Exception as exc:
        print(f"  WARNING: Pending order check failed: {exc}")
        return False


# ── Step 1: Read latest decision ────────────────────────────────────

def read_latest_crypto_decision() -> dict:
    """Parse the most recent raw (non-EXECUTION) decision from the crypto log."""
    if not os.path.exists(CRYPTO_LOG):
        raise RuntimeError(f"No crypto decisions log: {CRYPTO_LOG}")

    with open(CRYPTO_LOG, "r") as f:
        content = f.read()

    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(content[start : i + 1]))
                except json.JSONDecodeError:
                    pass

    if not objects:
        raise RuntimeError("No decision found in kairos_crypto_decisions.log")

    # Prefer raw decisions (not EXECUTION wrappers)
    for obj in reversed(objects):
        if obj.get("type") != "EXECUTION" and obj.get("action"):
            return obj

    # Fallback: unwrap EXECUTION
    for obj in reversed(objects):
        if obj.get("type") == "EXECUTION" and "decision" in obj:
            return obj["decision"]

    return objects[-1]


# ── Step 2: Risk guards + confluence ────────────────────────────────

def apply_crypto_guards(decision: dict, current_price: float) -> dict:
    """Apply crypto risk guards and confluence-aware sizing.

    Loads the asset's signal tags, computes confluence, and caps
    trade_usd based on the tier multiplier and risk guard limits.
    """
    action = decision.get("action", "HOLD").upper()
    trade_usd = decision.get("trade_usd", 0)
    asset = decision.get("asset", "?")

    if action == "HOLD" or trade_usd == 0:
        return decision

    cfg = _load_crypto_config()
    max_trade = cfg.get("trade_limits", {}).get("max_trade_usd", 500)

    # ── Confluence scoring ────────────────────────────────────────
    try:
        from kairos_crypto_signals import get_crypto_ticker_signals
        signal_tags = get_crypto_ticker_signals(asset)

        conf_cfg = cfg.get("confluence", {})
        points_table = conf_cfg.get("signal_points", {
            "HOT-RSI": 1, "HOT-REVERSION": 1,
            "HOT-CRYPTO-MOMENTUM": 2, "HOT-CRYPTO-MACRO": 1, "HOT-CRYPTO-BASELINE": 1,
        })
        tiers = conf_cfg.get("tiers", [
            {"min_score": 1, "name": "Single Signal", "nlv_pct": 0.005},
            {"min_score": 2, "name": "Medium",        "nlv_pct": 0.0075},
            {"min_score": 3, "name": "High",          "nlv_pct": 0.010},
            {"min_score": 5, "name": "Very High",     "nlv_pct": 0.015},
        ])
        tiers.sort(key=lambda t: t["min_score"])
        base_usd = conf_cfg.get("base_trade_usd", max_trade)

        score = sum(points_table.get(t, 1) for t in signal_tags)
        tier_name = "None"
        nlv_pct = 0.0
        if score > 0:
            for tier in tiers:
                if score >= tier["min_score"]:
                    tier_name = tier["name"]
                    nlv_pct = tier.get("nlv_pct", 0)

        # NLV-based sizing if we have a price, fallback to base_usd cap
        if nlv_pct > 0 and current_price > 0:
            # Use base_usd as the NLV proxy for crypto (not full portfolio NLV)
            max_usd = int(base_usd * (nlv_pct / 0.005))  # scale relative to lowest tier
        elif nlv_pct > 0:
            max_usd = base_usd
        else:
            max_usd = 0

        print(f"  Signals for {asset}: {signal_tags or '(none)'}")
        print(f"  Confluence: {tier_name} (score {score}, {nlv_pct:.2%}) → ${max_usd} max")

        decision["confluence"] = {
            "score": score,
            "tier": tier_name,
            "nlv_pct": nlv_pct,
            "signals": signal_tags,
        }

        if max_usd == 0 and score == 0:
            # No signals — still allow base trade (Claude decided to trade)
            max_usd = max_trade
            print(f"  No signals fired — using base limit ${max_usd}")

        if trade_usd > max_usd:
            print(f"  CRYPTO GUARD: Capped trade_usd from ${trade_usd} to ${max_usd}")
            trade_usd = max_usd

    except Exception as exc:
        print(f"  WARNING: Confluence failed ({exc}) — using static limit ${max_trade}")
        if trade_usd > max_trade:
            trade_usd = max_trade

    # ── Static guard: absolute max ────────────────────────────────
    if trade_usd > max_trade:
        print(f"  CRYPTO GUARD: Capped to absolute max ${max_trade}")
        trade_usd = max_trade

    decision["trade_usd"] = trade_usd

    # Compute quantity from trade_usd and price
    if current_price > 0:
        decision["quantity"] = round(trade_usd / current_price, 8)
    else:
        decision["quantity"] = 0

    return decision


# ── Step 3: Simulate execution (no IBKR) ──────────────────────────────

def _fetch_coingecko_price(asset: str) -> float:
    """Fetch the current USD price for a crypto asset from CoinGecko."""
    import requests
    coin_id = COINGECKO_IDS.get(asset, "")
    if not coin_id:
        return 0.0
    resp = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": coin_id, "vs_currencies": "usd"},
        timeout=10,
    )
    resp.raise_for_status()
    return float(resp.json().get(coin_id, {}).get("usd", 0))


def execute_crypto_order(decision: dict) -> dict:
    """Simulate a crypto order using CoinGecko price.

    No orders are submitted to IBKR.  On BUY, a new lot is recorded in
    crypto_holdings.  On SELL, open lots are closed FIFO and realized
    P&L is computed.
    """
    action = decision.get("action", "HOLD").upper()
    asset = _clean_symbol(decision.get("asset", "?"))
    decision["asset"] = asset
    trade_usd = decision.get("trade_usd", 0)
    qty = decision.get("quantity", 0)

    if action == "HOLD" or trade_usd == 0 or qty == 0:
        return {"status": "Skipped", "reason": "HOLD decision"}

    try:
        sim_price = _fetch_coingecko_price(asset)
        if sim_price <= 0:
            return {"status": "Cancelled", "reason": f"Could not fetch CoinGecko price for {asset}"}

        # Recalculate qty at the live CoinGecko price
        qty = round(trade_usd / sim_price, 8)
        decision["quantity"] = qty

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        print(f"  CoinGecko price: ${sim_price:,.2f}")
        print(f"  Simulating {action} {qty:.8f} {asset} (~${trade_usd:.0f}) @ ${sim_price:,.2f}")

        from kairos_log_db import init_db, insert_crypto_holding, sell_crypto_holdings
        init_db()

        result = {
            "status": "Filled (Simulated)",
            "fill_price": sim_price,
            "fill_time": ts,
            "commission": 0.0,
            "quantity_filled": qty,
            "simulated": True,
        }

        if action == "BUY":
            insert_crypto_holding(asset, ts, sim_price, qty)
            print(f"  Simulated BUY recorded: {qty:.8f} {asset} @ ${sim_price:,.2f}")

        elif action == "SELL":
            closed_lots = sell_crypto_holdings(asset, qty, ts, sim_price)
            if closed_lots:
                total_cost = sum(lot["entry_price"] * lot["quantity"] for lot in closed_lots)
                total_proceeds = sim_price * sum(lot["quantity"] for lot in closed_lots)
                realized_pnl = total_proceeds - total_cost
                result["closed_lots"] = closed_lots
                result["realized_pnl"] = round(realized_pnl, 2)
                pnl_sign = "+" if realized_pnl >= 0 else ""
                print(f"  Simulated SELL: closed {len(closed_lots)} lot(s), "
                      f"realized P&L: {pnl_sign}${realized_pnl:,.2f}")
            else:
                print(f"  WARNING: No open lots found for {asset} — SELL recorded but no P&L")

        return result

    except Exception as exc:
        return {"status": "Cancelled", "reason": str(exc)}


# ── Step 4: Logging ─────────────────────────────────────────────────

def log_crypto_execution(decision: dict, execution: dict):
    """Log the execution to flat file and SQLite (no Slack here)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    clean_asset = _clean_symbol(decision.get("asset", "?"))
    entry = {
        "type": "EXECUTION",
        "timestamp": ts,
        "decision": {
            "action": decision.get("action", "HOLD"),
            "asset": clean_asset,
            "trade_usd": decision.get("trade_usd", 0),
            "quantity": decision.get("quantity", 0),
            "rationale": decision.get("rationale", ""),
        },
        "execution": execution,
    }

    # Flat file log
    with open(CRYPTO_LOG, "a") as f:
        f.write("\n" + json.dumps(entry, indent=2))
        f.write("\n" + "=" * W + "\n")
    print(f"  Logged to {CRYPTO_LOG}")

    # SQLite
    try:
        from kairos_log_db import init_db, insert_crypto_decision
        init_db()

        conf = decision.get("confluence", {})
        insert_crypto_decision(
            timestamp=ts,
            asset=clean_asset,
            action=decision.get("action", "HOLD"),
            trade_usd=decision.get("trade_usd", 0),
            quantity=decision.get("quantity", 0),
            rationale=decision.get("rationale", ""),
            execution_price=execution.get("fill_price"),
            execution_status=execution.get("status"),
            commission=execution.get("commission"),
            confluence_score=conf.get("score"),
            confluence_tier=conf.get("tier"),
            simulated=bool(execution.get("simulated")),
        )
        print(f"  Logged to kairos.db (crypto_decisions)")
    except Exception as exc:
        print(f"  WARNING: DB logging failed: {exc}")

    # Slack is sent ONCE from main() after the full crypto cycle completes.


# ── main ────────────────────────────────────────────────────────────

def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS CRYPTO EXECUTOR — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    # Read decision
    print(banner("Reading Latest Crypto Decision"))
    decision = read_latest_crypto_decision()
    action = decision.get("action", "HOLD").upper()
    asset = _clean_symbol(decision.get("asset", "?"))
    decision["asset"] = asset  # write cleaned value back
    trade_usd = decision.get("trade_usd", 0)
    assets_evaluated = [_clean_symbol(a) for a in decision.get("assets_evaluated", [asset])]
    runner_up = _clean_symbol(decision.get("runner_up", ""))
    print(f"  Action:    {action}")
    print(f"  Asset:     {asset}")
    print(f"  Trade USD: ${trade_usd:.0f}")
    print(f"  Evaluated: {', '.join(assets_evaluated)}")
    print(f"  Rationale: {(decision.get('rationale') or 'N/A')[:100]}...")

    # ── Collect per-asset results for the consolidated Slack report ──
    # Claude evaluates all assets and picks one (or HOLD).
    # Build a line for each evaluated asset.
    asset_lines: list[str] = []
    execution = None
    filled_count = 0
    skipped_count = 0

    if action == "HOLD" or trade_usd == 0:
        print(banner("HOLD — No Action"))
        execution = {"status": "Skipped", "reason": "HOLD decision"}
        log_crypto_execution(decision, execution)
        # All assets were evaluated, all skipped
        for a in assets_evaluated:
            if a == asset:
                asset_lines.append(f"  {a:<5}  HOLD — {decision.get('rationale', 'no opportunity')[:80]}")
            else:
                asset_lines.append(f"  {a:<5}  Skipped — not selected")
            skipped_count += 1
    else:
        # Get current price from CoinGecko
        print(banner("Fetching Current Price"))
        current_price = 0.0
        try:
            current_price = _fetch_coingecko_price(asset)
        except Exception as exc:
            print(f"  WARNING: CoinGecko price fetch failed: {exc}")

        if current_price > 0:
            print(f"  {asset}: ${current_price:,.2f}")
        else:
            print(f"  WARNING: No price for {asset}")

        # Apply risk guards + confluence
        print(banner("Applying Crypto Risk Guards"))
        decision = apply_crypto_guards(decision, current_price)
        action = decision.get("action", "HOLD").upper()
        trade_usd = decision.get("trade_usd", 0)
        print(f"  Final: {action} ~${trade_usd:.0f} {asset}")

        if action == "HOLD" or trade_usd == 0:
            print(banner("SKIPPED"))
            execution = {"status": "Skipped", "reason": decision.get("risk_guard", "Guard triggered")}
            log_crypto_execution(decision, execution)
            asset_lines.append(f"  {asset:<5}  Skipped — {execution['reason'][:80]}")
            skipped_count += 1
        elif _has_pending_order(asset):
            # Duplicate order guard: open order already pending from a previous cycle
            print(banner("SKIPPED — Pending Order"))
            reason = "Skipped — open order already pending"
            print(f"  {asset}: {reason}")
            execution = {"status": "Skipped", "reason": reason}
            log_crypto_execution(decision, execution)
            asset_lines.append(f"  {asset:<5}  {reason}")
            skipped_count += 1
        else:
            # Execute
            print(banner(f"Executing: {action} ~${trade_usd:.0f} {asset}"))
            execution = execute_crypto_order(decision)

            # Log
            print(banner("Logging Execution"))
            log_crypto_execution(decision, execution)

            status = execution.get("status", "UNKNOWN")
            if execution.get("fill_price"):
                qty = decision.get("quantity", 0)
                asset_lines.append(
                    f"  {asset:<5}  {action} — {status} "
                    f"qty={qty} @ ${execution['fill_price']:,.2f} "
                    f"(~${trade_usd:,.0f})"
                )
                filled_count += 1
            else:
                reason = execution.get("reason", status)
                asset_lines.append(f"  {asset:<5}  {action} — {status}: {reason[:60]}")
                if status in ("Filled", "Submitted"):
                    filled_count += 1
                else:
                    skipped_count += 1  # Cancelled / unknown = not actioned

        # Non-selected assets
        for a in assets_evaluated:
            if a != asset:
                asset_lines.append(f"  {a:<5}  Skipped — not selected")
                skipped_count += 1

    # ── Terminal summary ──────────────────────────────────────────
    print("\n" + "━" * W)
    if execution and execution.get("fill_price"):
        sim_tag = " (SIM)" if execution.get("simulated") else ""
        print(f"  Crypto executor complete. {action} {asset} filled{sim_tag} @ ${execution['fill_price']:,.2f}")
    else:
        print(f"  Crypto executor complete. Status: {(execution or {}).get('status', 'N/A')}")
        reason = (execution or {}).get("reason")
        if reason:
            print(f"  Reason: {reason}")
    print("━" * W)

    # ── Slack: post simulated fills to #kairos-reports ──────────────
    try:
        from kairos_alerts import post_message

        if filled_count > 0:
            assets_block = "\n".join(asset_lines) if asset_lines else "  (no assets evaluated)"
            runner_str = f"\nRunner-up: {runner_up}" if runner_up else ""
            fill_lines = ""
            if execution and execution.get("fill_price"):
                qty = decision.get("quantity", 0)
                pnl_str = ""
                if execution.get("realized_pnl") is not None:
                    rpnl = execution["realized_pnl"]
                    pnl_sign = "+" if rpnl >= 0 else ""
                    pnl_str = f"  |  Realized P&L: {pnl_sign}${rpnl:,.2f}"
                fill_lines = (
                    f"\n*Simulated Fill:* {action} {qty} {asset} "
                    f"@ ${execution['fill_price']:,.2f} (~${trade_usd:,.0f}){pnl_str}"
                )
            report_text = (
                f":test_tube: *Kairos Crypto Cycle (SIM) — {ts}*\n"
                f"Evaluated: {len(assets_evaluated)} asset(s)  |  "
                f"Actioned: {filled_count}  |  Skipped: {skipped_count}\n"
                f"\n```\n{assets_block}\n```"
                f"{fill_lines}{runner_str}"
            )
            post_message("reports", report_text)

    except Exception as exc:
        print(f"  WARNING: Slack reporting failed: {exc}")


if __name__ == "__main__":
    main()
