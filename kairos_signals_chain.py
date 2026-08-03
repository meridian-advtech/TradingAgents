"""
Kairos AI Value Chain Signals — Leading-Indicator Propagation

Models the AI value chain as a three-tier dependency graph (defined in
kairos_config.json under `ai_value_chain`):

  Tier 1  — Linchpins / demand drivers (NVDA, MSFT, GOOGL, …). Their moves
            lead the chain.
  Tier 2  — Direct suppliers (MU, AMAT, EQIX, ANET, …) that depend on Tier 1.
  Tier 3  — Deep suppliers (VRT, ENTG, APD, NEE, …) that depend on Tier 2/1.

Core idea: when a Tier 1 linchpin makes a large move, downstream tickers
typically reprice with a lag of days-to-weeks. run_chain_signal_scan() finds
downstream names that have NOT yet repriced proportionally and flags them
HOT-CHAIN — a leading-indicator catch-up trade.

Public API:
    load_chain()                       Raw ai_value_chain list
    get_chain_tickers()                {TICKER: entry}
    get_chain_context(ticker)          Single entry or None
    get_chain_tier(ticker)             1 | 2 | 3 | None
    get_upstream_tickers(ticker)       Parsed upstream CSV → list
    get_downstream_tickers(ticker)     All tickers depending on `ticker`
    format_chain_prompt_block(tickers) Reasoning-prompt context block
    run_chain_signal_scan()            HOT-CHAIN candidate list (day-cached)
    get_chain_status_summary()         !chain Slack-command text

All functions degrade gracefully: missing config returns [], yfinance
failures return empty/None, and nothing here raises to its caller.
"""

import json
import os
from datetime import datetime
from typing import Optional

SCRIPT_DIR = "/Users/jelmore/Kairos"

CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
CACHE_FILE = os.path.join(SCRIPT_DIR, "kairos_ipo_cache.json")

CHAIN_SIGNAL_TAG = "HOT-CHAIN"
TIER1_TRIGGER_PCT = 3.0      # |%| move on a Tier 1 ticker that triggers a scan
REPRICE_RATIO = 0.5          # downstream "repriced" if it moved >=50% of trigger
CHAIN_CACHE_KEY = "chain_signals"

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = None


# ── Config / cache I/O ───────────────────────────────────────────────

def _load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (IOError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
            f.write("\n")
    except IOError as exc:
        print(f"  WARNING: chain cache write failed: {exc}")


def _today_et() -> str:
    if _ET is not None:
        return datetime.now(_ET).strftime("%Y-%m-%d")
    return datetime.utcnow().strftime("%Y-%m-%d")


# ── Chain catalog ────────────────────────────────────────────────────

def load_chain() -> list[dict]:
    """Read ai_value_chain from kairos_config.json. [] on any failure."""
    chain = _load_config().get("ai_value_chain")
    if not isinstance(chain, list):
        return []
    return [e for e in chain if isinstance(e, dict) and e.get("ticker")]


def get_chain_tickers() -> dict[str, dict]:
    """Return {TICKER: full_entry} for every chain ticker."""
    out: dict[str, dict] = {}
    for entry in load_chain():
        t = str(entry.get("ticker", "")).upper()
        if t:
            out[t] = entry
    return out


def get_chain_context(ticker: str) -> Optional[dict]:
    """Return the chain entry for a single ticker, or None if not in chain."""
    if not ticker:
        return None
    return get_chain_tickers().get(ticker.upper())


def get_chain_tier(ticker: str) -> Optional[int]:
    """Return 1, 2, or 3 — or None if the ticker is not in the chain."""
    ctx = get_chain_context(ticker)
    if ctx is None:
        return None
    try:
        return int(ctx.get("chain_tier"))
    except (TypeError, ValueError):
        return None


def get_upstream_tickers(ticker: str) -> list[str]:
    """Parse the upstream_dependency CSV string into a list of tickers."""
    ctx = get_chain_context(ticker)
    if ctx is None:
        return []
    return _parse_csv(ctx.get("upstream_dependency"))


def get_downstream_tickers(upstream_ticker: str) -> list[str]:
    """Return all chain tickers that list `upstream_ticker` as a dependency."""
    if not upstream_ticker:
        return []
    up = upstream_ticker.upper()
    out: list[str] = []
    for entry in load_chain():
        t = str(entry.get("ticker", "")).upper()
        if t and up in _parse_csv(entry.get("upstream_dependency")):
            out.append(t)
    return out


def _parse_csv(raw) -> list[str]:
    if not raw:
        return []
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


# ── Reasoning-prompt context block ───────────────────────────────────

def format_chain_prompt_block(tickers: list[str]) -> str:
    """Build the AI-value-chain context block for the reasoning prompt.

    One line per shortlisted ticker that is in the chain. Returns an empty
    string when none of `tickers` are chain members.
    """
    chain = get_chain_tickers()
    lines: list[str] = []
    for raw in tickers:
        t = str(raw).upper()
        entry = chain.get(t)
        if entry is None:
            continue
        tier = entry.get("chain_tier")
        role = entry.get("role", "")
        thesis = entry.get("thesis", "")
        upstream = entry.get("upstream_dependency")
        if upstream:
            leading = (f"Leading indicator: {upstream} moves precede {t} "
                       f"repricing by days to weeks.")
        else:
            leading = (f"Leading indicator: {t} is a Tier 1 linchpin — its "
                       f"moves lead downstream repricing by days to weeks.")
        lines.append(
            f"CHAIN CONTEXT: {t} is a Chain Tier {tier} AI play. "
            f"Role: {role}. Thesis: {thesis}. {leading}"
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


# ── HOT-CHAIN signal scan ────────────────────────────────────────────

def _prev_session_pct(ticker: str) -> Optional[float]:
    """Previous completed session's % change for `ticker` via yfinance.

    Returns None if yfinance is unavailable or the ticker has no data.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    try:
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        prev = float(closes.iloc[-2])
        last = float(closes.iloc[-1])
        if prev <= 0:
            return None
        return (last - prev) / prev * 100.0
    except Exception:
        return None


def run_chain_signal_scan() -> list[dict]:
    """Scan Tier 1 movers and flag downstream tickers that haven't repriced.

    For each Tier 1 ticker whose previous-session move exceeds
    TIER1_TRIGGER_PCT, every downstream Tier 2/3 ticker that has NOT moved at
    least REPRICE_RATIO of the trigger (same direction) is flagged HOT-CHAIN.

    Results are cached for the day under `chain_signals` in
    kairos_ipo_cache.json so the same set doesn't re-fire intra-day.
    """
    today = _today_et()
    cache = _load_cache()
    cached = cache.get(CHAIN_CACHE_KEY)
    if isinstance(cached, dict) and cached.get("date") == today:
        sigs = cached.get("signals")
        if isinstance(sigs, list):
            return sigs

    chain = get_chain_tickers()
    if not chain:
        return []

    tier1 = [t for t, e in chain.items() if _entry_tier(e) == 1]

    tier1_moves: dict[str, float] = {}
    for t in tier1:
        pc = _prev_session_pct(t)
        if pc is not None:
            tier1_moves[t] = pc

    signals: list[dict] = []
    flagged: set[str] = set()
    for trig, trig_pct in tier1_moves.items():
        if abs(trig_pct) <= TIER1_TRIGGER_PCT or trig_pct == 0:
            continue
        for d in get_downstream_tickers(trig):
            if d in flagged:
                continue
            d_pct = _prev_session_pct(d)
            if d_pct is None:
                continue
            # Repriced if downstream captured >=50% of the trigger move
            # (same sign). Negative ratio = moved the wrong way = not repriced.
            if d_pct / trig_pct >= REPRICE_RATIO:
                continue
            signals.append({
                "ticker": d,
                "chain_tier": get_chain_tier(d),
                "trigger_ticker": trig,
                "trigger_pct": round(trig_pct, 2),
                "ticker_pct": round(d_pct, 2),
                "gap_pct": round(trig_pct - d_pct, 2),
                "signal": CHAIN_SIGNAL_TAG,
            })
            flagged.add(d)

    movers = {t: round(p, 2) for t, p in tier1_moves.items() if abs(p) > 1.0}
    cache[CHAIN_CACHE_KEY] = {
        "date": today,
        "signals": signals,
        "tier1_movers": movers,
    }
    _save_cache(cache)

    # HOT-CHAIN Slack alert to #kairos-reports, deduped once per ET day.
    try:
        if signals:
            sent = cache.get("chain_alerts_sent")
            if not isinstance(sent, list):
                sent = []
            if today not in sent:
                n = len(signals)
                suffix = "y" if n == 1 else "ies"
                lines = [
                    f"\U0001f517 *HOT-CHAIN Signal — {n} opportunit{suffix} detected*"
                ]
                groups: dict[str, list[dict]] = {}
                order: list[str] = []
                for s in signals:
                    trig = s["trigger_ticker"]
                    if trig not in groups:
                        groups[trig] = []
                        order.append(trig)
                    groups[trig].append(s)
                for trig in order:
                    for s in groups[trig]:
                        tk = s["ticker"]
                        role = (get_chain_context(tk) or {}).get("role", "")
                        lines.append(
                            f"  {tk:6} (Tier {s.get('chain_tier')} — {role})   "
                            f"{s['trigger_ticker']} {s['trigger_pct']:+.1f}% but "
                            f"{tk} only {s['ticker_pct']:+.1f}%  "
                            f"gap: {s['gap_pct']:.1f}%"
                        )
                lines.append(
                    "_These tickers have been force-included in today's shortlist._"
                )
                from kairos_alerts import alert_pipeline_event
                alert_pipeline_event("\n".join(lines), channel="reports")
                sent.append(today)
                cache["chain_alerts_sent"] = sent
                _save_cache(cache)
    except Exception as exc:
        print(f"  WARNING: chain Slack alert failed: {exc}")

    return signals


def _entry_tier(entry: dict) -> Optional[int]:
    try:
        return int(entry.get("chain_tier"))
    except (TypeError, ValueError):
        return None


# ── !chain Slack summary ─────────────────────────────────────────────

def get_chain_status_summary() -> str:
    """Formatted text for the !chain Slack command.

    Shows today's Tier 1 movers (>1%) and any active HOT-CHAIN signals with
    the gap explanation. Reports "No active chain signals" when none.
    """
    try:
        signals = run_chain_signal_scan()
    except Exception as exc:
        return f":warning: Chain scan unavailable: `{exc}`"

    entry = _load_cache().get(CHAIN_CACHE_KEY) or {}
    movers = entry.get("tier1_movers") or {}
    big = {t: float(p) for t, p in movers.items() if abs(float(p)) > 1.0}

    lines = [":link: *AI Value Chain Status*", "```"]

    if big:
        lines.append("Tier 1 movers (>1%):")
        for t, p in sorted(big.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"  {t:<6}{p:+.2f}%")
    else:
        lines.append("Tier 1 movers (>1%): none")

    lines.append("")
    if signals:
        lines.append(f"Active HOT-CHAIN signals ({len(signals)}):")
        for s in signals:
            lines.append(
                f"  {s['ticker']:<6}Tier {s.get('chain_tier')} — "
                f"{s['trigger_ticker']} moved {s['trigger_pct']:+.2f}% but "
                f"{s['ticker']} only {s['ticker_pct']:+.2f}% "
                f"(gap {s['gap_pct']:+.2f}%)"
            )
    else:
        lines.append("No active chain signals")

    lines.append("```")
    return "\n".join(lines)
