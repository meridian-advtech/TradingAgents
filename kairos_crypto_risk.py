"""
Kairos Crypto Risk Guards — Phase B

Volatility-aware risk controls for crypto trading.
All thresholds are loaded from kairos_crypto_config.json — never hardcoded.

Guards enforced before ANY crypto trade:
  1. Position size   — no single crypto asset > N% of total portfolio
  2. Trade size      — no single trade > $X USD
  3. Circuit breaker — halt ALL crypto if BTC/ETH moves >N% in a 4h window
  4. Daily loss      — stop crypto trading if down >N% on crypto for the day

Usage:
    from kairos_crypto_risk import CryptoRiskGuard

    guard = CryptoRiskGuard()

    # Update price history on every pipeline run (feeds the circuit breaker)
    guard.update_price_history("BTC", 65432.00)
    guard.update_price_history("ETH", 3456.00)

    # Run all guards before a proposed trade
    result = guard.run_all_guards(
        symbol="BTC",
        current_price=65432.00,
        trade_usd=300.0,
        current_position_value=1500.0,
        portfolio_total=50000.0,
        crypto_portfolio_value=2000.0,
    )
    if result["all_passed"]:
        print("Trade approved by risk guards")
    else:
        for name, g in result["guards"].items():
            if not g["allowed"]:
                print(f"  BLOCKED by {name}: {g['reason']}")

Files written:
    kairos_crypto_prices.json  — rolling price history for circuit breaker
    kairos_crypto_daily.json   — daily P&L tracking
    kairos_crypto_alerts.log   — timestamped alert log
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_crypto_config.json")
PRICE_HISTORY_FILE = os.path.join(SCRIPT_DIR, "kairos_crypto_prices.json")
DAILY_TRACKING_FILE = os.path.join(SCRIPT_DIR, "kairos_crypto_daily.json")
ALERTS_LOG = os.path.join(SCRIPT_DIR, "kairos_crypto_alerts.log")


# ── Config ───────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load risk configuration from kairos_crypto_config.json."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Crypto config not found: {CONFIG_FILE}\n"
            "Create kairos_crypto_config.json before running the crypto pipeline."
        )
    with open(CONFIG_FILE) as f:
        return json.load(f)


# ── Result type ──────────────────────────────────────────────────────

@dataclass
class GuardResult:
    allowed: bool
    guard_name: str
    reason: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.allowed else "FAIL"
        return f"[{status}] {self.guard_name}: {self.reason}"


# ── Main risk guard class ────────────────────────────────────────────

class CryptoRiskGuard:
    """Central risk guard for crypto trading decisions.

    Instantiate once per pipeline run. Call update_price_history() to feed
    the circuit breaker, then run_all_guards() before any trade decision.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self._session_alerts: list[str] = []

    # ── Guard 1: Position size ────────────────────────────────────────

    def check_position_size(
        self,
        symbol: str,
        current_position_value: float,
        proposed_trade_usd: float,
        portfolio_total: float,
    ) -> GuardResult:
        """Ensure adding to a position won't exceed max % of total portfolio.

        Args:
            symbol: Asset symbol (e.g. "BTC")
            current_position_value: Current market value of existing holdings
            proposed_trade_usd: USD value of the proposed new trade
            portfolio_total: Total portfolio value (all assets)
        """
        max_pct = self.config["position_limits"]["max_position_pct"]
        new_value = current_position_value + proposed_trade_usd
        new_pct = new_value / portfolio_total if portfolio_total > 0 else 1.0

        allowed = new_pct <= max_pct
        reason = (
            f"{symbol} would be {new_pct:.1%} of portfolio "
            f"(limit: {max_pct:.0%}, current: ${current_position_value:,.0f}, "
            f"adding: ${proposed_trade_usd:,.0f})"
        )
        if not allowed:
            self._log_alert("POSITION_SIZE_BREACH", reason)

        return GuardResult(
            allowed=allowed,
            guard_name="position_size",
            reason=reason,
            details={
                "symbol": symbol,
                "current_value": current_position_value,
                "proposed_addition": proposed_trade_usd,
                "new_value": new_value,
                "new_pct": round(new_pct, 4),
                "max_pct": max_pct,
                "portfolio_total": portfolio_total,
            },
        )

    # ── Guard 2: Trade size ───────────────────────────────────────────

    def check_trade_size(self, trade_usd: float) -> GuardResult:
        """Ensure a single crypto trade does not exceed the max dollar limit."""
        max_usd = self.config["trade_limits"]["max_trade_usd"]
        allowed = trade_usd <= max_usd
        reason = f"Trade ${trade_usd:,.2f} vs limit ${max_usd:,.2f}"
        if not allowed:
            self._log_alert("TRADE_SIZE_BREACH", reason)

        return GuardResult(
            allowed=allowed,
            guard_name="trade_size",
            reason=reason,
            details={"trade_usd": trade_usd, "max_usd": max_usd},
        )

    # ── Guard 3: Circuit breaker ──────────────────────────────────────

    def update_price_history(self, symbol: str, price: float) -> None:
        """Record a price observation for the circuit breaker window.

        Call this on EVERY pipeline run (not just before trades) so the
        rolling window stays populated.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        history = self._load_price_history()

        if symbol not in history:
            history[symbol] = []

        history[symbol].append({"ts": now_iso, "price": price})

        # Prune entries older than window_hours + 1h buffer
        max_hours = self.config.get("price_history_max_hours", 5)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_hours)
        history[symbol] = [
            e for e in history[symbol]
            if datetime.fromisoformat(e["ts"]) > cutoff
        ]

        self._save_price_history(history)

    def check_circuit_breaker(self, symbol: str, current_price: float) -> GuardResult:
        """Check if price has moved beyond the threshold in the rolling window.

        Returns allowed=False (halt trading) if the move exceeds the configured
        percentage within the configured time window.
        """
        cfg = self.config["circuit_breaker"]
        threshold_pct = cfg["price_move_pct"]
        window_h = cfg["window_hours"]

        history = self._load_price_history()
        all_entries = history.get(symbol, [])

        if len(all_entries) < 2:
            return GuardResult(
                allowed=True,
                guard_name="circuit_breaker",
                reason=(
                    f"Insufficient price history for {symbol} "
                    f"({len(all_entries)} point(s), need ≥2) — guard skipped"
                ),
                details={"symbol": symbol, "data_points": len(all_entries)},
            )

        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_h)
        window_entries = [
            e for e in all_entries
            if datetime.fromisoformat(e["ts"]) > cutoff
        ]

        if not window_entries:
            return GuardResult(
                allowed=True,
                guard_name="circuit_breaker",
                reason=f"No price data within {window_h}h window for {symbol} — guard skipped",
                details={"symbol": symbol, "window_h": window_h},
            )

        oldest_price = window_entries[0]["price"]
        move_pct = abs((current_price - oldest_price) / oldest_price * 100)
        allowed = move_pct <= threshold_pct

        reason = (
            f"{symbol} moved {move_pct:.1f}% over last {window_h}h "
            f"(${oldest_price:,.2f} → ${current_price:,.2f}, limit: {threshold_pct:.1f}%)"
        )
        if not allowed:
            self._log_alert("CIRCUIT_BREAKER_TRIGGERED", reason)

        return GuardResult(
            allowed=allowed,
            guard_name="circuit_breaker",
            reason=reason,
            details={
                "symbol": symbol,
                "current_price": current_price,
                "oldest_window_price": oldest_price,
                "move_pct": round(move_pct, 2),
                "threshold_pct": threshold_pct,
                "window_h": window_h,
                "window_data_points": len(window_entries),
            },
        )

    # ── Guard 4: Daily loss limit ─────────────────────────────────────

    def update_daily_pnl(self, symbol: str, pnl_usd: float) -> None:
        """Record realized P&L for a completed crypto trade today.

        Call this after a trade fills, passing the realized gain/loss in USD.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tracking = self._load_daily_tracking()

        if tracking.get("date") != today:
            # New day — reset
            tracking = {"date": today, "pnl_by_symbol": {}, "total_pnl_usd": 0.0}

        prev = tracking["pnl_by_symbol"].get(symbol, 0.0)
        tracking["pnl_by_symbol"][symbol] = prev + pnl_usd
        tracking["total_pnl_usd"] = sum(tracking["pnl_by_symbol"].values())
        self._save_daily_tracking(tracking)

    def check_daily_loss(self, crypto_portfolio_value: float) -> GuardResult:
        """Check whether today's crypto P&L has breached the daily loss limit.

        Args:
            crypto_portfolio_value: Total USD value of all current crypto holdings
                                    (used as the denominator for loss %).
        """
        max_loss_pct = self.config["daily_loss_limit"]["max_loss_pct"]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tracking = self._load_daily_tracking()

        if tracking.get("date") != today:
            return GuardResult(
                allowed=True,
                guard_name="daily_loss",
                reason="No crypto P&L recorded today — guard passes",
                details={"date": today, "total_pnl_usd": 0.0},
            )

        total_pnl = tracking.get("total_pnl_usd", 0.0)

        if crypto_portfolio_value <= 0:
            # No crypto portfolio — nothing to protect
            return GuardResult(
                allowed=True,
                guard_name="daily_loss",
                reason="No crypto portfolio value — guard skipped",
                details={"total_pnl_usd": total_pnl},
            )

        loss_pct = (-total_pnl / crypto_portfolio_value) * 100
        allowed = loss_pct < max_loss_pct

        sign = "-" if total_pnl < 0 else "+"
        reason = (
            f"Daily crypto P&L: {sign}${abs(total_pnl):,.2f} "
            f"({loss_pct:+.2f}% loss vs limit -{max_loss_pct:.1f}%)"
        )
        if not allowed:
            self._log_alert("DAILY_LOSS_LIMIT_HIT", reason)

        return GuardResult(
            allowed=allowed,
            guard_name="daily_loss",
            reason=reason,
            details={
                "date": today,
                "total_pnl_usd": total_pnl,
                "loss_pct": round(loss_pct, 3),
                "max_loss_pct": max_loss_pct,
                "crypto_portfolio_value": crypto_portfolio_value,
            },
        )

    # ── Combined gate ────────────────────────────────────────────────

    def run_all_guards(
        self,
        symbol: str,
        current_price: float,
        trade_usd: float,
        current_position_value: float,
        portfolio_total: float,
        crypto_portfolio_value: float,
    ) -> dict[str, Any]:
        """Run all four risk guards and return a combined result.

        Returns:
            {
              "symbol": "BTC",
              "all_passed": True/False,
              "halted": True/False,   # True means circuit breaker or daily loss triggered
              "guards": {
                "trade_size":     {"allowed": True,  "reason": "..."},
                "position_size":  {"allowed": True,  "reason": "..."},
                "circuit_breaker":{"allowed": False, "reason": "..."},
                "daily_loss":     {"allowed": True,  "reason": "..."},
              }
            }
        """
        results: dict[str, GuardResult] = {
            "trade_size": self.check_trade_size(trade_usd),
            "position_size": self.check_position_size(
                symbol, current_position_value, trade_usd, portfolio_total
            ),
            "circuit_breaker": self.check_circuit_breaker(symbol, current_price),
            "daily_loss": self.check_daily_loss(crypto_portfolio_value),
        }

        all_passed = all(r.allowed for r in results.values())
        # "Halted" = hard stop from circuit breaker or daily loss (not just size limits)
        halted = (
            not results["circuit_breaker"].allowed
            or not results["daily_loss"].allowed
        )

        return {
            "symbol": symbol,
            "all_passed": all_passed,
            "halted": halted,
            "guards": {
                name: {"allowed": r.allowed, "reason": r.reason, "details": r.details}
                for name, r in results.items()
            },
        }

    # ── Convenience: print guard report ─────────────────────────────

    def print_guard_report(self, guard_result: dict) -> None:
        """Pretty-print the output of run_all_guards()."""
        symbol = guard_result.get("symbol", "?")
        all_passed = guard_result.get("all_passed", False)
        halted = guard_result.get("halted", False)

        status = "ALL PASSED" if all_passed else ("HALTED" if halted else "BLOCKED")
        print(f"\n  Risk Guards — {symbol} — {status}")
        print(f"  {'─' * 60}")
        for name, g in guard_result.get("guards", {}).items():
            icon = "✓" if g["allowed"] else "✗"
            print(f"  {icon} {name:<18} {g['reason']}")
        print()

    @property
    def session_alerts(self) -> list[str]:
        """Alerts raised during this guard session."""
        return list(self._session_alerts)

    # ── Private helpers ──────────────────────────────────────────────

    def _log_alert(self, alert_type: str, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}] {alert_type}: {message}"
        self._session_alerts.append(line)
        with open(ALERTS_LOG, "a") as f:
            f.write(line + "\n")
        print(f"  ⚠  RISK ALERT [{alert_type}]: {message}")

    def _load_price_history(self) -> dict:
        if os.path.exists(PRICE_HISTORY_FILE):
            try:
                with open(PRICE_HISTORY_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_price_history(self, history: dict) -> None:
        with open(PRICE_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

    def _load_daily_tracking(self) -> dict:
        if os.path.exists(DAILY_TRACKING_FILE):
            try:
                with open(DAILY_TRACKING_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_daily_tracking(self, tracking: dict) -> None:
        with open(DAILY_TRACKING_FILE, "w") as f:
            json.dump(tracking, f, indent=2)


# ── Module-level convenience ─────────────────────────────────────────

def get_config_summary() -> str:
    """Return a human-readable summary of current risk thresholds."""
    cfg = load_config()
    lines = [
        "  Crypto Risk Configuration:",
        f"    Max position:      {cfg['position_limits']['max_position_pct']:.0%} of portfolio",
        f"    Max trade size:    ${cfg['trade_limits']['max_trade_usd']:,} USD",
        f"    Circuit breaker:   >{cfg['circuit_breaker']['price_move_pct']:.0f}% move in {cfg['circuit_breaker']['window_hours']}h halts trading",
        f"    Daily loss limit:  >{cfg['daily_loss_limit']['max_loss_pct']:.0f}% crypto portfolio loss stops trading",
        f"    Tracked assets:    {', '.join(cfg.get('asset_symbols', ['BTC', 'ETH']))}",
    ]
    return "\n".join(lines)


# ── CLI smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Kairos Crypto Risk Guards — Smoke Test")
    print("=" * 60)

    print("\n[1] Config summary:")
    print(get_config_summary())

    guard = CryptoRiskGuard()

    print("\n[2] Populating price history (simulating past readings)...")
    from datetime import timedelta
    import time as _time

    # Inject a historical price directly to test circuit breaker
    history_data = {"BTC": [], "ETH": []}
    now_utc = datetime.now(timezone.utc)
    # BTC: price 4h ago was 60000, now 65000 — 8.3% move — should TRIP the breaker
    old_ts = (now_utc - timedelta(hours=4, minutes=5)).isoformat()
    history_data["BTC"].append({"ts": old_ts, "price": 60000.0})
    # Current reading
    history_data["BTC"].append({"ts": now_utc.isoformat(), "price": 65000.0})
    # ETH: normal 2% move — should PASS
    history_data["ETH"].append({"ts": old_ts, "price": 3400.0})
    history_data["ETH"].append({"ts": now_utc.isoformat(), "price": 3468.0})

    guard._save_price_history(history_data)
    print("  Price history injected.")

    print("\n[3] Circuit breaker — BTC (expect FAIL: >8% move):")
    cb_btc = guard.check_circuit_breaker("BTC", 65000.0)
    print(f"  {cb_btc}")

    print("\n[4] Circuit breaker — ETH (expect PASS: ~2% move):")
    cb_eth = guard.check_circuit_breaker("ETH", 3468.0)
    print(f"  {cb_eth}")

    print("\n[5] Trade size guard (expect FAIL: $600 > $500 limit):")
    ts = guard.check_trade_size(600.0)
    print(f"  {ts}")

    print("\n[6] Trade size guard (expect PASS: $400 <= $500 limit):")
    ts2 = guard.check_trade_size(400.0)
    print(f"  {ts2}")

    print("\n[7] Position size guard (expect FAIL: would be 7% of $50k portfolio):")
    ps = guard.check_position_size("BTC", 2000.0, 1500.0, 50000.0)
    print(f"  {ps}")

    print("\n[8] Position size guard (expect PASS: 4% of $50k portfolio):")
    ps2 = guard.check_position_size("ETH", 500.0, 1500.0, 50000.0)
    print(f"  {ps2}")

    print("\n[9] Daily loss guard (no trades today — expect PASS):")
    dl = guard.check_daily_loss(5000.0)
    print(f"  {dl}")

    print("\n[10] Full guard run — ETH, $400 trade, modest position (expect mixed):")
    result = guard.run_all_guards(
        symbol="ETH",
        current_price=3468.0,
        trade_usd=400.0,
        current_position_value=500.0,
        portfolio_total=50000.0,
        crypto_portfolio_value=2000.0,
    )
    guard.print_guard_report(result)
    print(f"  all_passed: {result['all_passed']}  halted: {result['halted']}")

    print("\n[11] Session alerts logged this run:")
    for alert in guard.session_alerts:
        print(f"  {alert}")

    # Cleanup injected test data
    guard._save_price_history({})

    print("\n" + "=" * 60)
    print("  Smoke test complete.")
    print("=" * 60)
