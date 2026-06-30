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
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

TIER_C_FILE = os.path.join(SCRIPT_DIR, "kairos_tier_c.json")
UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "kairos_universe.json")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
IPO_CACHE_FILE = os.path.join(SCRIPT_DIR, "kairos_ipo_cache.json")
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
SCHEDULER_LOG = os.path.join(SCRIPT_DIR, "kairos_scheduler.log")
TTL_DAYS = 14

# ── Autonomous expiry-handling thresholds ────────────────────────────
EXPIRY_EXTEND_DAYS = 14            # how long an EXTEND grants
PROMOTE_PRICE_GAIN_PCT = 10.0      # >10% above entry price → promote
HOT_SIGNAL_LOOKBACK_DAYS = 30      # HOT-CONGRESS / HOT-INSIDER window
CHAIN_LOOKBACK_DAYS = 14           # HOT-CHAIN drop-guard window
FRESH_SIGNAL_LOOKBACK_DAYS = 7     # HOT-CHAIN / PRE-EVENT-DIP extend window
PROMOTE_SIGNALS = ("HOT-CONGRESS", "HOT-INSIDER")
FRESH_SIGNALS = ("HOT-CHAIN", "PRE-EVENT-DIP")


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


def _log_scheduler(msg: str) -> None:
    """Append a timestamped line to kairos_scheduler.log AND echo to stdout.

    The audit runs inside the scheduler-driven pipeline, but we write the
    log line directly so autonomous expiry decisions are durably recorded
    regardless of how stdout is captured.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] TIER-C-AUTO {msg}"
    print(f"  {msg}")
    try:
        with open(SCHEDULER_LOG, "a") as f:
            f.write(line + "\n")
    except IOError as exc:
        print(f"  WARNING: scheduler.log append failed: {exc}")


# ── Autonomous expiry decision inputs ────────────────────────────────

def _parse_decision_ts(row_ts: str, created_at: str = "") -> Optional[datetime]:
    """Parse a decisions-table timestamp into an aware UTC datetime."""
    for raw, fmt in (
        (row_ts, "%Y-%m-%d %H:%M:%S UTC"),
        (row_ts, "%Y-%m-%d %H:%M:%S"),
        (created_at, "%Y-%m-%d %H:%M:%S"),
    ):
        if not raw:
            continue
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _signals_in_data_inputs(data_inputs: str) -> list[str]:
    """Extract fired signal names from a decisions.data_inputs blob.

    Handles the JSON shape ({"confluence": {"signals": [...]}}, possibly
    double-encoded as a string) and falls back to a substring scan so a
    schema change can't silently drop signal detection.
    """
    if not data_inputs:
        return []
    out: list[str] = []
    blob = data_inputs
    for _ in range(2):  # data_inputs is occasionally a JSON-encoded string
        if isinstance(blob, str):
            try:
                blob = json.loads(blob)
            except (ValueError, TypeError):
                break
        else:
            break
    if isinstance(blob, dict):
        sigs = (blob.get("confluence") or {}).get("signals")
        if isinstance(sigs, list):
            out = [str(s) for s in sigs]
    if not out:  # substring fallback over the raw text
        text = data_inputs if isinstance(data_inputs, str) else json.dumps(blob)
        for sig in set(PROMOTE_SIGNALS + FRESH_SIGNALS):
            if sig in text:
                out.append(sig)
    return out


def _ticker_has_filled_buy(ticker: str) -> bool:
    """True iff a filled BUY for this ticker exists in the decisions table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM decisions "
                "WHERE ticker = ? AND action = 'BUY' "
                "AND execution_status = 'Filled'",
                (ticker,),
            ).fetchone()
        finally:
            conn.close()
        return bool(row and row[0] > 0)
    except sqlite3.Error as exc:
        print(f"  WARNING: decisions DB query failed for {ticker}: {exc}")
        return False


def _signal_fired_within(ticker: str, signal_names: tuple, days: int) -> bool:
    """True iff any of `signal_names` appears in this ticker's decisions
    data_inputs within the trailing `days` window."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    wanted = set(signal_names)
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            rows = conn.execute(
                "SELECT timestamp, created_at, data_inputs FROM decisions "
                "WHERE ticker = ? AND data_inputs IS NOT NULL",
                (ticker,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"  WARNING: decisions DB scan failed for {ticker}: {exc}")
        return False

    for row_ts, created_at, data_inputs in rows:
        dt = _parse_decision_ts(row_ts or "", created_at or "")
        if dt is None or dt < cutoff:
            continue
        fired = set(_signals_in_data_inputs(data_inputs))
        if fired & wanted:
            return True
    return False


def _in_ai_value_chain(ticker: str) -> bool:
    """True iff ticker appears anywhere in kairos_config.json ai_value_chain."""
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except (IOError, json.JSONDecodeError):
        return False
    chain = cfg.get("ai_value_chain")
    if not isinstance(chain, list):
        return False
    t = ticker.upper()
    for entry in chain:
        if isinstance(entry, dict) and (entry.get("ticker") or "").upper() == t:
            return True
        if isinstance(entry, str) and entry.upper() == t:
            return True
    return False


def _chain_signal_within(ticker: str, days: int) -> bool:
    """True iff a HOT-CHAIN fire for this ticker is recorded in the
    kairos_ipo_cache.json chain_signals snapshot within `days`."""
    try:
        with open(IPO_CACHE_FILE) as f:
            cache = json.load(f)
    except (IOError, json.JSONDecodeError):
        return False
    snap = cache.get("chain_signals")
    if not isinstance(snap, dict):
        return False
    snap_date = snap.get("date") or ""
    try:
        d = datetime.strptime(snap_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False
    if d < datetime.now(timezone.utc) - timedelta(days=days):
        return False
    t = ticker.upper()
    for sig in snap.get("signals") or []:
        if isinstance(sig, dict) and (sig.get("ticker") or "").upper() == t:
            return True
    return False


def _current_price(ticker: str) -> Optional[float]:
    """Latest close via yfinance. None on any failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if hist is None or len(hist) == 0:
            return None
        px = float(hist["Close"].iloc[-1])
        return px if px > 0 else None
    except Exception:
        return None


def _entry_price(entry: dict) -> Optional[float]:
    """Recorded Tier C entry price, if any. Checks the common key names
    plus the IPO offer-price midpoint as a fallback."""
    for key in ("entry_price", "added_price", "price_at_add", "add_price"):
        v = entry.get(key)
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            continue
    lo, hi = entry.get("offer_price_low"), entry.get("offer_price_high")
    try:
        if lo and hi:
            return (float(lo) + float(hi)) / 2.0
    except (TypeError, ValueError):
        pass
    return None


def _price_above_entry(entry: dict, pct: float) -> Optional[bool]:
    """True/False if entry price is recorded and we can fetch current price;
    None if the criterion can't be evaluated (no entry price / no quote)."""
    base = _entry_price(entry)
    if base is None:
        return None
    cur = _current_price(entry["ticker"])
    if cur is None:
        return None
    return cur >= base * (1.0 + pct / 100.0)


def decide_expiry(entry: dict) -> tuple[str, str]:
    """Autonomously decide what to do with an expired Tier C ticker.

    Returns (action, reason) where action is "PROMOTE" / "EXTEND" / "DROP".
    Precedence: PROMOTE > EXTEND > DROP (a fresh signal extends rather than
    drops; a recorded conviction signal promotes).
    """
    ticker = entry["ticker"].upper()

    # 1. PROMOTE if ANY conviction signal is present.
    promote_reasons: list[str] = []
    if _ticker_has_filled_buy(ticker):
        promote_reasons.append("filled BUY on record")
    if _in_ai_value_chain(ticker):
        promote_reasons.append("in AI value chain")
    for sig in PROMOTE_SIGNALS:
        if _signal_fired_within(ticker, (sig,), HOT_SIGNAL_LOOKBACK_DAYS):
            promote_reasons.append(f"{sig} in last {HOT_SIGNAL_LOOKBACK_DAYS}d")
    gain = _price_above_entry(entry, PROMOTE_PRICE_GAIN_PCT)
    if gain is True:
        promote_reasons.append(f"price >{PROMOTE_PRICE_GAIN_PCT:.0f}% above entry")
    if promote_reasons:
        return "PROMOTE", "; ".join(promote_reasons)

    # 2. EXTEND if a fresh signal is still working.
    #    - HOT-CHAIN / PRE-EVENT-DIP within 7d (decisions data_inputs), OR
    #    - HOT-CHAIN within 14d (chain_signals snapshot) — too fresh to drop.
    if _signal_fired_within(ticker, FRESH_SIGNALS, FRESH_SIGNAL_LOOKBACK_DAYS):
        return "EXTEND", f"active signal in last {FRESH_SIGNAL_LOOKBACK_DAYS}d"
    if (_chain_signal_within(ticker, FRESH_SIGNAL_LOOKBACK_DAYS)
            or _chain_signal_within(ticker, CHAIN_LOOKBACK_DAYS)):
        return "EXTEND", f"HOT-CHAIN within {CHAIN_LOOKBACK_DAYS}d"

    # 3. DROP — no conviction, no fresh signal.
    return "DROP", "no filled BUY, not in AI chain, no recent HOT signals"


def _promote_entry_to_tier_b(target: dict, reason: str) -> None:
    """Move a Tier C entry into Tier B in the universe file (no Tier C save —
    the caller persists the trimmed Tier C list in one pass)."""
    universe = _load_universe()
    tier_b = universe.setdefault("tier_b", {"added_date": "", "tickers": []})
    tier_b_tickers = tier_b.setdefault("tickers", [])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tier_b_tickers.append({
        "symbol": target["ticker"],
        "name": target.get("name", target["ticker"]),
        "sector": target.get("sector", "Unknown"),
        "added_date": today,
        "promoted_from": "C",
        "promotion_reason": reason,
    })
    universe.setdefault("metadata", {})["last_updated"] = today
    _save_universe(universe)


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
        f"\U0001f5d1\ufe0f TIER C REMOVE: ${ticker} removed from opportunistic watchlist.",
        channel="watchlist"
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
    """Check all Tier C entries for expiry and handle each AUTONOMOUSLY.

    Every expired ticker is promoted, extended, or dropped per
    decide_expiry() -- no manual prompt is ever posted. Tier C and the
    universe are persisted in a single pass. Returns the list of expired
    entries, each annotated with the action taken.
    """
    entries = _load_tier_c()
    today_dt = datetime.now(timezone.utc)
    today = today_dt.strftime("%Y-%m-%d")

    kept: list[dict] = []         # entries that remain in Tier C after audit
    expired: list[dict] = []      # expired entries (annotated with action)
    promoted = dropped = extended = 0
    universe_dirty = False

    for e in entries:
        if e.get("expires_date", "9999-12-31") > today:
            kept.append(e)         # not expired -- untouched
            continue

        ticker = e["ticker"]
        try:
            action, reason = decide_expiry(e)
        except Exception as exc:
            # Never let one bad ticker abort the whole audit; keep it and
            # try again next cycle.
            print(f"  WARNING: expiry decision failed for {ticker}: {exc}")
            kept.append(e)
            continue

        annotated = dict(e)
        annotated["expiry_action"] = action
        annotated["expiry_reason"] = reason
        expired.append(annotated)

        if action == "PROMOTE":
            _promote_entry_to_tier_b(e, reason)
            universe_dirty = True
            promoted += 1
            _log_scheduler(f"PROMOTE {ticker} -> Tier B :: {reason}")
            _slack_alert(
                f"⬆️ TIER C PROMOTED: ${ticker} → Tier B. Reason: {reason}",
                channel="watchlist",
            )
            # dropped from Tier C (not added to `kept`)

        elif action == "EXTEND":
            e["expires_date"] = (today_dt + timedelta(days=EXPIRY_EXTEND_DAYS)).strftime("%Y-%m-%d")
            e["expiry_alerted"] = False
            kept.append(e)
            extended += 1
            _log_scheduler(
                f"EXTEND {ticker} +{EXPIRY_EXTEND_DAYS}d "
                f"(new expiry {e['expires_date']}) :: {reason}"
            )
            _slack_alert(
                f"⏳ TIER C EXTENDED: ${ticker} — active signal, "
                f"{EXPIRY_EXTEND_DAYS} more days",
                channel="log",
            )

        else:  # DROP -- silent (log channel only)
            dropped += 1
            _log_scheduler(f"DROP {ticker} (silent) :: {reason}")
            _slack_alert(
                f"\U0001f5d1️ TIER C DROPPED: ${ticker} expired and removed "
                f"(no activity).",
                channel="log",
            )
            # dropped from Tier C (not added to `kept`)

    # Persist Tier C once (always -- EXTEND mutates dates in place).
    _save_tier_c(kept)
    if universe_dirty:
        print(f"  Tier C audit: universe updated with {promoted} promotion(s)")

    if expired:
        _log_scheduler(
            f"audit complete: {len(expired)} expired - "
            f"{promoted} promoted, {extended} extended, {dropped} dropped"
        )
    else:
        print(f"  Tier C audit: {len(kept)} active, 0 expired")

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
