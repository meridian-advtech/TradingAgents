"""
Kairos Tier C — Opportunistic Intake System

Manages temporary tickers with a 14-day TTL in a sidecar file
(kairos_tier_c.json). Tickers can be added, removed, promoted to
Tier B, or audited for expiry.

Usage:
  python kairos_tier_c.py add TICKER "Name" "Sector" "reason"
  python kairos_tier_c.py remove TICKER
  python kairos_tier_c.py promote TICKER
  python kairos_tier_c.py audit
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

TIER_C_FILE = os.path.join(SCRIPT_DIR, "kairos_tier_c.json")
UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "kairos_universe.json")
TTL_DAYS = 14


# ── Helpers ────────────────────────────────────────────���─────────────

def _load_tier_c() -> list[dict]:
    if not os.path.exists(TIER_C_FILE):
        return []
    try:
        with open(TIER_C_FILE) as f:
            data = json.load(f)
        # Ensure all entries have expiry_alerted field, defaulting to False
        for e in data:
            if "expiry_alerted" not in e:
                e["expiry_alerted"] = False
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_tier_c(entries: list[dict]) -> None:
    with open(TIER_C_FILE, "w") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")


def _load_universe() -> dict:
    with open(UNIVERSE_FILE) as f:
        return json.load(f)


def _save_universe(data: dict) -> None:
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _existing_tickers() -> set[str]:
    """Return all tickers already in Tier A, Tier B, or Tier C."""
    universe = _load_universe()
    tickers = set()

    # Tier A equities
    for _cat, syms in universe.get("tier_a", {}).get("equities", {}).items():
        tickers.update(syms)
    # Tier A ETFs
    for _cat, syms in universe.get("tier_a", {}).get("etfs", {}).items():
        tickers.update(syms)
    # Tier B
    for entry in universe.get("tier_b", {}).get("tickers", []):
        sym = entry["symbol"] if isinstance(entry, dict) else entry
        tickers.add(sym)
    # Tier C
    for entry in _load_tier_c():
        tickers.add(entry["ticker"])

    return tickers


def _slack_alert(text: str, channel: str = "alerts") -> bool:
    """Send a message to a Slack channel. Fault-tolerant."""
    try:
        from kairos_alerts import post_message
        return post_message(channel, text)
    except Exception as exc:
        print(f"  WARNING: Slack alert failed: {exc}")
        return False


def load_tier_c_tickers() -> list[str]:
    """Return list of Tier C ticker symbols (for screener integration)."""
    return [e["ticker"] for e in _load_tier_c()]


# ── Commands ──────────────────────────────────────────��──────────────

def add(ticker: str, name: str, sector: str, reason: str) -> dict:
    """Add a ticker to Tier C with 14-day TTL.

    Returns {"ok": True, "entry": {...}} or {"ok": False, "error": "..."}.
    """
    ticker = ticker.upper().strip()

    # Check for duplicates across all tiers
    existing = _existing_tickers()
    if ticker in existing:
        msg = f"ERROR: {ticker} already exists in the universe (Tier A, B, or C)"
        print(f"  {msg}")
        return {"ok": False, "error": msg}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    expires = (datetime.now(timezone.utc) + timedelta(days=TTL_DAYS)).strftime("%Y-%m-%d")

    entry = {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "tier": "C",
        "added_date": today,
        "expires_date": expires,
        "add_reason": reason,
        "expiry_alerted": False,
    }

    entries = _load_tier_c()
    entries.append(entry)
    _save_tier_c(entries)

    print(f"  Added {ticker} to Tier C (expires {expires})")

    _slack_alert(
        f"\U0001f195 TIER C ADD: ${ticker} added to opportunistic watchlist.\n"
        f"Reason: {reason} | Expires: {expires}",
        channel="watchlist"
    )

    return {"ok": True, "entry": entry}


def remove(ticker: str) -> dict:
    """Remove a ticker from Tier C.

    Returns {"ok": True} or {"ok": False, "error": "..."}.
    """
    ticker = ticker.upper().strip()
    entries = _load_tier_c()
    new_entries = [e for e in entries if e["ticker"] != ticker]

    if len(new_entries) == len(entries):
        msg = f"ERROR: {ticker} not found in Tier C"
        print(f"  {msg}")
        return {"ok": False, "error": msg}

    _save_tier_c(new_entries)
    print(f"  Removed {ticker} from Tier C")

    _slack_alert(
        f"\U0001f5d1\ufe0f TIER C REMOVE: ${ticker} removed from opportunistic watchlist."
    )

    return {"ok": True}


def promote(ticker: str) -> dict:
    """Promote a ticker from Tier C to Tier B (permanent universe).

    Returns {"ok": True} or {"ok": False, "error": "..."}.
    """
    ticker = ticker.upper().strip()
    entries = _load_tier_c()
    target = None
    new_entries = []
    for e in entries:
        if e["ticker"] == ticker:
            target = e
        else:
            new_entries.append(e)

    if target is None:
        msg = f"ERROR: {ticker} not found in Tier C"
        print(f"  {msg}")
        return {"ok": False, "error": msg}

    # Add to kairos_universe.json as Tier B
    universe = _load_universe()
    tier_b = universe.setdefault("tier_b", {"added_date": "", "tickers": []})
    tier_b_tickers = tier_b.setdefault("tickers", [])

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tier_b_tickers.append({
        "symbol": ticker,
        "name": target.get("name", ticker),
        "sector": target.get("sector", "Unknown"),
        "added_date": today,
        "promoted_from": "C",
    })
    universe["metadata"]["last_updated"] = today
    _save_universe(universe)

    # Remove from Tier C
    _save_tier_c(new_entries)

    print(f"  Promoted {ticker} from Tier C → Tier B")

    _slack_alert(
        f"\u2b06\ufe0f TIER C PROMOTE: ${ticker} promoted to Tier B permanent universe.",
        channel="watchlist"
    )

    return {"ok": True}


def audit() -> list[dict]:
    """Check all Tier C entries for expiry. Returns list of expired entries.

    Sends Slack alerts for expired tickers but does NOT auto-remove them.
    """
    entries = _load_tier_c()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    expired = []
    modified = False

    for e in entries:
        if e.get("expires_date", "9999-12-31") <= today:
            expired.append(e)
            if not e.get("expiry_alerted", False):
                _slack_alert(
                    f"\u23f0 TIER C EXPIRY: ${e['ticker']} (added {e['added_date']}) has expired.\n"
                    f"Promote or drop? Run: python kairos_tier_c.py promote/remove {e['ticker']}",
                    channel="watchlist"
                )
                e["expiry_alerted"] = True
                modified = True

    if modified:
        _save_tier_c(entries)

    if expired:
        print(f"  Tier C audit: {len(expired)} expired — {', '.join(e['ticker'] for e in expired)}")
    else:
        print(f"  Tier C audit: {len(entries)} active, 0 expired")

    return expired


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kairos Tier C Manager")
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add", help="Add ticker to Tier C")
    p_add.add_argument("ticker")
    p_add.add_argument("name")
    p_add.add_argument("sector")
    p_add.add_argument("reason")

    p_rm = sub.add_parser("remove", help="Remove ticker from Tier C")
    p_rm.add_argument("ticker")

    p_promo = sub.add_parser("promote", help="Promote ticker to Tier B")
    p_promo.add_argument("ticker")

    sub.add_parser("audit", help="Check for expired entries")

    args = parser.parse_args()

    if args.command == "add":
        result = add(args.ticker, args.name, args.sector, args.reason)
        if not result["ok"]:
            sys.exit(1)
    elif args.command == "remove":
        result = remove(args.ticker)
        if not result["ok"]:
            sys.exit(1)
    elif args.command == "promote":
        result = promote(args.ticker)
        if not result["ok"]:
            sys.exit(1)
    elif args.command == "audit":
        audit()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
