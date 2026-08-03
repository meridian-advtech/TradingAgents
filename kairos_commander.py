"""
Kairos Commander — Two-Way Slack Interface (curl long-poll daemon)

Listens on #kairos-commands and responds to bang-prefixed commands:

  !status        portfolio value, cash, P&L, open positions, cycle state
  !positions     table of open positions with current price + P&L
  !performance   per-signal aggregate stats (signal_performance view)
  !why TICKER    most-recent rationale + thesis prediction for a ticker
  !run           trigger a full pipeline cycle (10-min cooldown)
  !dry-run       trigger a screener dry-run cycle (10-min cooldown)
  !pause         set paused flag in kairos_config.json (blocks new BUYs)
  !resume        clear paused flag
  !regime        current macro regime + classification reason
  !help          this list

The bot token is read from the SLACK_BOT_TOKEN env var, falling back to
kairos_config.json["slack"]["bot_token"].

Transport: all Slack I/O routes through kairos_alerts._slack_api_call (curl
subprocess), because this host's endpoint filter blocks the Python socket
layer for Slack's IP range — slack_sdk / slack_bolt / raw WebSockets all
fail with EBADF here, while Apple-signed curl is permitted. We therefore
long-poll conversations.history and reply via chat.postMessage, both over
curl. No app-level (xapp-) token or Socket Mode connection is used.

Designed to run under launchd. Failures are caught at the handler level
so a bad command never kills the daemon.
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

# ── Absolute paths (per spec) ────────────────────────────────────────
SCRIPT_DIR = "/Users/jelmore/TradingAgents"
sys.path.insert(0, SCRIPT_DIR)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
KAIROS_DB = os.path.join(SCRIPT_DIR, "kairos.db")
ML_DB = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
STATE_FILE = os.path.join(SCRIPT_DIR, "kairos_state.json")
PERFORMANCE_FILE = os.path.join(SCRIPT_DIR, "kairos_performance.json")
COMMANDER_LOG = os.path.join(SCRIPT_DIR, "kairos_commander.log")
PID_FILE = "/tmp/kairos_commander.pid"
# Dedicated IBKR clientId for the commander. Lives in the 7–14 gap between
# regime (15) and stoploss (20–29); every other module sits in 1–6, 15, 99,
# or the 20–99 random bands. Keeps !regime / snapshot off shared slots.
IBKR_CLIENT_ID = 10
COMMAND_CHANNEL_KEY = "commands"
COMMAND_CHANNEL_NAME = "kairos-commands"
CYCLE_COOLDOWN_SECONDS = 10 * 60  # !run / !dry-run guard

# ── Logging ──────────────────────────────────────────────────────────
# Only a FileHandler: under launchd the plist already redirects stdout to
# COMMANDER_LOG, so a StreamHandler(sys.stdout) would write every line twice
# into the same file. The FileHandler covers both manual and launchd runs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(COMMANDER_LOG, mode="a")],
)
log = logging.getLogger("kairos-commander")


# ── Config / tokens ──────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", CONFIG_FILE, exc)
        return {}


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def get_bot_token() -> str:
    """Return the Slack bot token, preferring the env var."""
    cfg = load_config()
    slack_cfg = cfg.get("slack", {}) if isinstance(cfg.get("slack"), dict) else {}
    bot = os.environ.get("SLACK_BOT_TOKEN") or slack_cfg.get("bot_token", "")
    return bot.strip()


def get_commands_channel_id() -> Optional[str]:
    cfg = load_config()
    channels = cfg.get("slack", {}).get("channels", {}) if isinstance(cfg.get("slack"), dict) else {}
    cid = channels.get(COMMAND_CHANNEL_KEY)
    return cid or None


# ── Pause flag persistence ───────────────────────────────────────────

def get_paused() -> bool:
    cfg = load_config()
    return bool(cfg.get("paused", False))


def set_paused(value: bool) -> None:
    cfg = load_config()
    cfg["paused"] = bool(value)
    save_config(cfg)


# ── Cooldown for !run / !dry-run ─────────────────────────────────────
_last_cycle_run: dict[str, float] = {}
_cycle_lock = threading.Lock()


def cooldown_remaining(key: str) -> float:
    with _cycle_lock:
        last = _last_cycle_run.get(key, 0.0)
    elapsed = time.time() - last
    return max(0.0, CYCLE_COOLDOWN_SECONDS - elapsed)


def mark_cycle_started(key: str) -> None:
    with _cycle_lock:
        _last_cycle_run[key] = time.time()


# ── DB helpers ───────────────────────────────────────────────────────

def _open_db(path: str) -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        log.warning("DB open failed (%s): %s", path, exc)
        return None


def get_open_holdings_from_db() -> list[dict]:
    """All currently-open lots (sold_date IS NULL), aggregated by ticker."""
    conn = _open_db(KAIROS_DB)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """SELECT ticker,
                      SUM(quantity)                       AS quantity,
                      SUM(entry_price * quantity) / SUM(quantity) AS avg_cost,
                      MIN(entry_date)                     AS first_entry
               FROM holdings
               WHERE sold_date IS NULL
               GROUP BY ticker
               ORDER BY ticker"""
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_signal_tags_for_ticker(ticker: str) -> list[str]:
    """Look up signals_fired from the most recent filled BUY in trade_outcomes."""
    conn = _open_db(ML_DB)
    if conn is None:
        return []
    try:
        row = conn.execute(
            "SELECT signals_fired FROM trade_outcomes "
            "WHERE ticker = ? AND action = 'BUY' AND outcome_label IS NULL "
            "ORDER BY timestamp_entry DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row["signals_fired"]:
        return []
    try:
        tags = json.loads(row["signals_fired"])
        if isinstance(tags, list):
            return tags
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ── IBKR snapshot (best-effort) ──────────────────────────────────────

def fetch_ibkr_snapshot(timeout_s: int = 8) -> dict:
    """Return live account + position prices, or {} on failure.

    Caller falls back to last-known DB values if this is empty.
    """
    snapshot: dict = {"ok": False}
    ib = None
    try:
        from ib_insync import IB, Stock  # noqa: F401
        ib = IB()
        ib.connect("127.0.0.1", 7497,
                   clientId=IBKR_CLIENT_ID,
                   timeout=timeout_s)
        wanted = {"NetLiquidation", "TotalCashValue", "UnrealizedPnL",
                  "BuyingPower", "RealizedPnL"}
        acct = {v.tag: v.value for v in ib.accountSummary() if v.tag in wanted}
        positions = []
        for p in ib.positions():
            positions.append({
                "symbol": p.contract.symbol,
                "quantity": float(p.position),
                "avg_cost": float(p.avgCost),
            })

        # Try to grab a snapshot price for each position
        ib.reqMarketDataType(4)
        prices: dict[str, float] = {}
        for pos in positions:
            try:
                contract = Stock(pos["symbol"], "SMART", "USD")
                ib.qualifyContracts(contract)
                mkt = ib.reqMktData(contract)
                ib.sleep(1)
                for attr in ("last", "close", "bid", "ask"):
                    val = getattr(mkt, attr, None)
                    if val is not None and val == val and val > 0:
                        prices[pos["symbol"]] = float(val)
                        break
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass
            except Exception:
                continue

        snapshot = {
            "ok": True,
            "account": acct,
            "positions": positions,
            "prices": prices,
        }
    except Exception as exc:
        log.info("IBKR snapshot unavailable: %s", exc)
    finally:
        if ib is not None:
            try:
                ib.disconnect()
            except Exception:
                pass
    return snapshot


# ── Command implementations ──────────────────────────────────────────

def cmd_help() -> str:
    return (
        "*Kairos Commander — available commands*\n"
        "```"
        "  !status         Portfolio value, cash, P&L, cycle state\n"
        "  !positions      Open positions w/ price + P&L\n"
        "  !performance    Per-signal win rate / avg return\n"
        "  !why TICKER     Latest rationale + thesis for a ticker\n"
        "  !run            Trigger a full pipeline cycle (10-min cooldown)\n"
        "  !dry-run        Trigger a screener dry-run (10-min cooldown)\n"
        "  !pause          Block new BUY orders\n"
        "  !resume         Clear the pause flag\n"
        "  !regime         Current macro regime + reason\n"
        "  !ipo            IPO watchlist status + recent S-1 matches\n"
        "  !chain          AI value-chain Tier 1 movers + HOT-CHAIN signals\n"
        "  !watchlist ...  add TICKER | remove TICKER | list (Tier C)\n"
        "  !help           This list\n"
        "```"
    )


def _read_cycle_state() -> tuple[Optional[int], Optional[str]]:
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return data.get("cycle_count"), data.get("last_run")
    except (IOError, json.JSONDecodeError):
        return None, None


def _latest_performance_snapshot() -> Optional[dict]:
    """Return the most-recent row from kairos_performance.json, or None."""
    try:
        with open(PERFORMANCE_FILE) as f:
            data = json.load(f)
        snaps = data.get("snapshots") or []
        if not snaps:
            return None
        # Snapshots are appended chronologically; use the last entry.
        last = snaps[-1]
        if not isinstance(last, dict):
            return None
        return last
    except (IOError, json.JSONDecodeError):
        return None


def _bulk_yfinance_prices(tickers: list[str]) -> dict[str, float]:
    """One yfinance.download() call for all tickers → {ticker: last_close}.

    Returns {} on failure. Tickers absent from the result are simply missing
    from the dict so callers can fall back to '—' for them.
    """
    if not tickers:
        return {}
    try:
        import yfinance as yf
    except ImportError:
        return {}
    try:
        # period=2d gives us yesterday + today even before today closes.
        # group_by="ticker" produces a column-multiindex when len>1.
        df = yf.download(
            tickers=" ".join(tickers),
            period="2d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        log.warning("yfinance bulk download failed: %s", exc)
        return {}

    out: dict[str, float] = {}
    if df is None or getattr(df, "empty", True):
        return out

    if len(tickers) == 1:
        # Single ticker → flat columns
        try:
            close = df["Close"].dropna()
            if len(close):
                out[tickers[0].upper()] = float(close.iloc[-1])
        except Exception:
            pass
        return out

    for t in tickers:
        try:
            sub = df[t]
        except (KeyError, TypeError):
            continue
        try:
            close = sub["Close"].dropna()
            if len(close):
                out[t.upper()] = float(close.iloc[-1])
        except Exception:
            continue
    return out


def cmd_status() -> str:
    holdings = get_open_holdings_from_db()
    snap = _latest_performance_snapshot()
    cycle, last_run = _read_cycle_state()
    paused = get_paused()

    open_count = len(holdings)
    # Cost basis from holdings: sum(quantity * avg_cost). Used either to
    # derive unrealized P&L (when we have a recent snapshot) or as the
    # full fallback when no snapshot exists at all.
    cost_basis = sum(
        float(h.get("quantity") or 0) * float(h.get("avg_cost") or 0)
        for h in holdings
    )

    if snap is not None:
        try:
            nlv = float(snap.get("value") or 0)
            cash = float(snap.get("cash") or 0)
            equity_value = float(snap.get("equity_value") or 0)
        except (TypeError, ValueError):
            nlv = cash = equity_value = 0.0
        # Unrealized P&L = current market value of open equity − cost basis.
        # If equity_value is missing, derive it as NLV − cash.
        market_value = equity_value if equity_value > 0 else max(nlv - cash, 0.0)
        upnl = market_value - cost_basis
        source = f"kairos_performance.json (as of {snap.get('date', '—')})"
    else:
        # No snapshot available — show cost basis only.
        nlv = cost_basis
        cash = 0.0
        upnl = 0.0
        source = "kairos.db cost basis (no snapshot file)"

    upnl_sign = "+" if upnl >= 0 else ""
    paused_line = " :pause_button: *PAUSED*" if paused else ""

    return (
        f":bar_chart: *Kairos Status* — {source}{paused_line}\n"
        f"```"
        f"  Net Liquidation : ${nlv:>14,.2f}\n"
        f"  Cash            : ${cash:>14,.2f}\n"
        f"  Unrealized P&L  : {upnl_sign}${upnl:>13,.2f}\n"
        f"  Open positions  : {open_count}\n"
        f"  Cycle count     : {cycle if cycle is not None else '—'}\n"
        f"  Last cycle      : {last_run or '—'}\n"
        f"```"
    )


def cmd_positions() -> str:
    holdings = get_open_holdings_from_db()
    if not holdings:
        return ":file_folder: No open positions in kairos.db."

    tickers = [h["ticker"] for h in holdings if h.get("ticker")]
    prices = _bulk_yfinance_prices(tickers)
    missing = [t for t in tickers if t.upper() not in prices]

    lines = [
        f"{'TICKER':<6}{'SHARES':>9}{'AVG COST':>11}"
        f"{'CURRENT':>11}{'P&L $':>11}{'P&L %':>8}  SIGNALS"
    ]
    for h in holdings:
        ticker = h["ticker"]
        qty = float(h.get("quantity") or 0)
        avg = float(h.get("avg_cost") or 0)
        cur = prices.get(ticker.upper())
        if cur:
            pnl_dol = (cur - avg) * qty
            pnl_pct = ((cur - avg) / avg * 100.0) if avg else 0.0
            cur_str = f"${cur:>9.2f}"
            pnl_dol_str = f"{'+' if pnl_dol >= 0 else ''}${pnl_dol:>8,.0f}"
            pnl_pct_str = f"{pnl_pct:+6.2f}%"
        else:
            cur_str = "       —"
            pnl_dol_str = "        —"
            pnl_pct_str = "      —"
        sig_tags = get_signal_tags_for_ticker(ticker)
        sig_str = ",".join(sig_tags) if sig_tags else "—"
        lines.append(
            f"{ticker:<6}{qty:>9.0f}${avg:>9.2f}{cur_str:>11}"
            f"{pnl_dol_str:>11}{pnl_pct_str:>8}  {sig_str}"
        )

    note = ""
    if missing:
        note = (f"\n_Prices unavailable from yfinance for "
                f"{len(missing)} ticker(s): {', '.join(missing[:10])}"
                f"{'…' if len(missing) > 10 else ''}_")
    return ":file_folder: *Open Positions*\n```\n" + "\n".join(lines) + "\n```" + note


def cmd_performance() -> str:
    try:
        from kairos_ml_thesis import format_signal_performance_section
        section = format_signal_performance_section()
        return section or ":chart_with_upwards_trend: No closed trades yet — performance view is empty."
    except Exception as exc:
        log.exception("performance lookup failed")
        return f":x: performance lookup failed: `{exc}`"


def cmd_why(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if not ticker or not re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", ticker):
        return f":x: `{ticker}` doesn't look like a ticker symbol."

    # 1. Most recent BUY rationale from kairos.db
    rationale = None
    bought_at = None
    conn = _open_db(KAIROS_DB)
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT rationale, timestamp, execution_price, quantity "
                "FROM decisions "
                "WHERE ticker = ? AND action = 'BUY' "
                "  AND execution_status IN ('Filled','Submitted') "
                "ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if row:
                rationale = row["rationale"]
                bought_at = (row["timestamp"], row["execution_price"], row["quantity"])
        finally:
            conn.close()

    # 2. Most recent thesis prediction from kairos_ml_outcomes.db
    thesis = None
    conn = _open_db(ML_DB)
    if conn is not None:
        try:
            row = conn.execute(
                """SELECT tp.predicted_direction, tp.predicted_timeframe_days,
                          tp.predicted_return_pct, tp.key_conditions,
                          tp.invalidation_conditions, tp.signal_type,
                          tp.conviction_score, tp.timestamp_entry
                   FROM thesis_predictions tp
                   JOIN trade_outcomes to_ ON to_.trade_id = tp.decision_id
                   WHERE tp.ticker = ? AND to_.outcome_label IS NULL
                   ORDER BY tp.id DESC LIMIT 1""",
                (ticker,),
            ).fetchone()
            if row:
                thesis = dict(row)
        finally:
            conn.close()

    if not rationale and not thesis:
        return f":mag: No recent BUY rationale or thesis found for `{ticker}`."

    parts = [f":mag: *Why {ticker}?*"]
    if bought_at:
        ts, px, qty = bought_at
        parts.append(f"_Bought {qty} @ ${px} on {ts}_")
    if rationale:
        parts.append("\n*Rationale*\n>" + rationale.replace("\n", "\n>"))
    if thesis:
        parts.append("\n*Thesis prediction*")
        parts.append(
            f"```"
            f"  Direction     : {thesis.get('predicted_direction') or '—'}\n"
            f"  Timeframe     : {thesis.get('predicted_timeframe_days') or '—'} days\n"
            f"  Expected move : "
            f"{thesis.get('predicted_return_pct') if thesis.get('predicted_return_pct') is not None else '—'}%\n"
            f"  Signal type   : {thesis.get('signal_type') or '—'}\n"
            f"  Conviction    : {thesis.get('conviction_score') or '—'}\n"
            f"```"
        )
        if thesis.get("key_conditions"):
            parts.append("*Key conditions (must hold)*\n>" +
                         thesis["key_conditions"].replace("\n", "\n>"))
        if thesis.get("invalidation_conditions"):
            parts.append("*Invalidation conditions*\n>" +
                         thesis["invalidation_conditions"].replace("\n", "\n>"))
    return "\n".join(parts)


def cmd_regime() -> str:
    try:
        from kairos_regime import detect_regime
        result = detect_regime(verbose=False, client_id=IBKR_CLIENT_ID)
    except Exception as exc:
        log.exception("regime detection failed")
        return f":x: Could not detect regime: `{exc}`"

    regime = result.get("regime", "?")
    reason = result.get("reason", "")
    data = result.get("data", {}) or {}
    changed = result.get("changed")
    prev = result.get("previous_regime")

    icon = {
        "NORMAL": ":white_check_mark:",
        "CAUTION": ":warning:",
        "RISK-OFF": ":no_entry:",
        "EXTREME-FEAR": ":rotating_light:",
    }.get(regime, ":grey_question:")

    vix = data.get("vix")
    spy = data.get("spy_price")
    spread = data.get("treasury_spread")
    kalshi = data.get("kalshi_recession_pct")

    delta = f"   (was {prev})" if changed and prev else ""
    return (
        f"{icon} *Macro regime: {regime}*{delta}\n"
        f"_{reason}_\n"
        f"```"
        f"  VIX                : {vix if vix is not None else '—'}\n"
        f"  SPY                : {f'${spy}' if spy is not None else '—'}\n"
        f"  10Y-2Y spread      : "
        f"{f'{spread:+.3f}' if spread is not None else '—'}\n"
        f"  Kalshi recession   : "
        f"{f'{kalshi}%' if kalshi is not None else '—'}\n"
        f"```"
    )


def cmd_ipo() -> str:
    """Render the IPO watchlist + recent S-1 matches.

    Falls back to a clear error message if kairos_ipo_intake can't be
    imported (e.g. missing yfinance).
    """
    try:
        from kairos_ipo_intake import (
            get_watchlist_status, fetch_edgar_s1_effectiveness,
        )
    except Exception as exc:
        return f":x: IPO intake module unavailable: `{exc}`"

    try:
        status = get_watchlist_status()
    except Exception as exc:
        log.exception("ipo status failed")
        status = []
        status_err = str(exc)
    else:
        status_err = None

    try:
        s1_hits = fetch_edgar_s1_effectiveness()
    except Exception as exc:
        log.exception("ipo S-1 fetch failed")
        s1_hits = []
        s1_err = str(exc)
    else:
        s1_err = None

    lines = [":rocket: *IPO Watchlist Status*"]
    if status_err:
        lines.append(f"_watchlist probe error:_ `{status_err}`")
    elif not status:
        lines.append("_(watchlist is empty)_")
    else:
        lines.append("```")
        lines.append(
            f"  {'COMPANY':<22}{'EXPECTED':<10}{'STATUS':<22}NOTES"
        )
        for s in status:
            tag = (
                f"LIVE @ {s['live_ticker']}" if s["is_live"]
                else "pending"
            )
            notes = (s.get("notes") or "")[:32]
            lines.append(
                f"  {s['name'][:22]:<22}{s['expected_ticker']:<10}"
                f"{tag:<22}{notes}"
            )
        lines.append("```")

    lines.append("\n*Recent EDGAR S-1 matches (last 3 days)*")
    if s1_err:
        lines.append(f"_EDGAR fetch error:_ `{s1_err}`")
    elif not s1_hits:
        lines.append("_(no fuzzy matches)_")
    else:
        for m in s1_hits[:5]:
            lines.append(
                f"  • _{m['match_name']}_ ← `{m['company']}` "
                f"({m['filing_type']}, {m['filing_date']}, score={m['score']})\n"
                f"    <{m['edgar_url']}|EDGAR>"
            )

    return "\n".join(lines)


def cmd_chain() -> str:
    """Render AI value-chain status — Tier 1 movers + active HOT-CHAIN signals."""
    try:
        from kairos_signals_chain import get_chain_status_summary
    except Exception as exc:
        return f":x: Chain signals module unavailable: `{exc}`"
    try:
        return get_chain_status_summary()
    except Exception as exc:
        log.exception("chain status failed")
        return f":x: Chain status failed: `{exc}`"


def cmd_watchlist(args: str) -> str:
    """Manage the Tier C opportunistic watchlist: add / remove / list.

    Usage:
      !watchlist add TICKER       (name=TICKER, sector=Unknown, 14-day TTL)
      !watchlist remove TICKER
      !watchlist list
    """
    try:
        import kairos_tier_c as tier_c
    except Exception as exc:
        return f":x: Tier C module unavailable: `{exc}`"

    parts = args.split()
    sub = parts[0].lower() if parts else ""

    if sub == "list":
        entries = tier_c._load_tier_c()
        if not entries:
            return ":clipboard: *Tier C watchlist* is empty."
        lines = [f"  {'TICKER':<8}{'EXPIRES':<12}REASON"]
        for e in entries:
            ticker = e.get("ticker", "?")
            expires = e.get("expires_date", "—")
            reason = (e.get("add_reason") or "")[:40]
            lines.append(f"  {ticker:<8}{expires:<12}{reason}")
        return (":clipboard: *Tier C watchlist*\n```\n"
                + "\n".join(lines) + "\n```")

    if sub in ("add", "remove"):
        if len(parts) < 2:
            return f":grey_question: Usage: `!watchlist {sub} TICKER`"
        ticker = parts[1].upper().strip()
        if not re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", ticker):
            return f":x: `{ticker}` doesn't look like a ticker symbol."

        if sub == "add":
            result = tier_c.add(
                ticker,
                name=ticker,
                sector="Unknown",
                reason="Added via Slack commander",
            )
            if result.get("ok"):
                entry = result.get("entry", {})
                return (f":white_check_mark: Added `{ticker}` to Tier C "
                        f"watchlist (expires {entry.get('expires_date', '—')}).")
            return f":x: {result.get('error', 'add failed')}"

        result = tier_c.remove(ticker)
        if result.get("ok"):
            return f":wastebasket: Removed `{ticker}` from Tier C watchlist."
        return f":x: {result.get('error', 'remove failed')}"

    return (":grey_question: Usage: `!watchlist add TICKER` | "
            "`!watchlist remove TICKER` | `!watchlist list`")


def cmd_pause() -> str:
    set_paused(True)
    return (
        ":pause_button: *Pipeline paused.* New BUY orders will be skipped "
        "until `!resume`. Stop-loss, thesis checkpoints, and SELLs continue."
    )


def cmd_resume() -> str:
    set_paused(False)
    return ":arrow_forward: *Pipeline resumed.* New BUY orders re-enabled."


def _run_cycle_subprocess(extra_args: list[str]) -> tuple[int, str]:
    """Spawn `python kairos_run.py --cycle ...` and capture tail of output."""
    python_exe = sys.executable
    cmd = [python_exe, os.path.join(SCRIPT_DIR, "kairos_run.py"),
           "--cycle"] + extra_args
    log.info("Spawning cycle: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min — cycles normally finish in a couple
        )
        tail = (proc.stdout[-1500:] if proc.stdout else "") \
            + ("\n[stderr]\n" + proc.stderr[-500:] if proc.stderr else "")
        return proc.returncode, tail
    except subprocess.TimeoutExpired:
        return 124, "(cycle timed out after 10 minutes)"
    except Exception as exc:
        return 1, f"(failed to spawn cycle: {exc})"


def _format_cooldown(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def trigger_cycle(say, thread_ts: Optional[str], dry_run: bool) -> None:
    """Run !run or !dry-run with cooldown guard. Threaded — never blocks
    the Bolt dispatcher.
    """
    key = "dry_run" if dry_run else "run"
    remaining = cooldown_remaining(key)
    if remaining > 0:
        say(text=(f":hourglass: `!{'dry-run' if dry_run else 'run'}` is on cooldown — "
                  f"try again in {_format_cooldown(remaining)}."),
            thread_ts=thread_ts)
        return

    mark_cycle_started(key)
    label = "dry-run" if dry_run else "full cycle"
    say(text=f":gear: Triggered {label}. Working on it…", thread_ts=thread_ts)

    def worker():
        extra = ["--screen-dry-run"] if dry_run else []
        rc, tail = _run_cycle_subprocess(extra)
        icon = ":white_check_mark:" if rc == 0 else ":x:"
        head = (f"{icon} `!{'dry-run' if dry_run else 'run'}` finished "
                f"with exit {rc}.\n")
        # Slack message length limit ~40k; keep it well under
        body = tail[-1800:] if tail else "(no output)"
        try:
            say(text=head + f"```{body}```", thread_ts=thread_ts)
        except Exception:
            log.exception("failed to post cycle result")

    threading.Thread(target=worker, name=f"kairos-{key}", daemon=True).start()


# ── Command parsing ──────────────────────────────────────────────────

_BANG = re.compile(r"^\s*!\s*([a-z\-]+)\s*(.*?)\s*$", re.IGNORECASE)


def parse_command(text: str) -> Optional[tuple[str, str]]:
    """Return (command, args) for a bang-prefixed message. None otherwise.

    Recognizes leading `!` or natural-language forms like
    "@kairos status" / "status please".
    """
    if not text:
        return None

    # Strip leading Slack user/channel mentions so "<@UXXX> !status" works.
    cleaned = re.sub(r"<[@#][A-Z0-9]+(?:\|[^>]*)?>", "", text).strip()
    if not cleaned:
        return None

    m = _BANG.match(cleaned)
    if m:
        return m.group(1).lower(), m.group(2).strip()

    # Bare keyword fallback (case-insensitive).
    stripped = cleaned.lower()
    first = stripped.split()[0]
    rest = stripped[len(first):].strip()
    if first in {
        "status", "positions", "performance", "why", "run",
        "dry-run", "dryrun", "pause", "resume", "regime", "ipo", "chain",
        "watchlist", "help",
    }:
        return first, rest
    return None


def handle_message(text: str, say, thread_ts: Optional[str]) -> None:
    parsed = parse_command(text)
    if parsed is None:
        say(text=":grey_question: I didn't understand that — try `!help`.",
            thread_ts=thread_ts)
        return

    cmd, args = parsed
    cmd_norm = cmd.replace("dryrun", "dry-run")
    log.info("dispatching command: %s (args=%r)", cmd_norm, args)

    try:
        if cmd_norm == "help":
            say(text=cmd_help(), thread_ts=thread_ts)
        elif cmd_norm == "status":
            say(text=cmd_status(), thread_ts=thread_ts)
        elif cmd_norm == "positions":
            say(text=":hourglass: Working on it…", thread_ts=thread_ts)
            say(text=cmd_positions(), thread_ts=thread_ts)
        elif cmd_norm == "performance":
            say(text=":hourglass: Working on it…", thread_ts=thread_ts)
            say(text=cmd_performance(), thread_ts=thread_ts)
        elif cmd_norm == "why":
            if not args:
                say(text=":grey_question: Usage: `!why TICKER`",
                    thread_ts=thread_ts)
            else:
                # First whitespace-separated token is the ticker
                ticker = args.split()[0]
                say(text=":hourglass: Working on it…", thread_ts=thread_ts)
                say(text=cmd_why(ticker), thread_ts=thread_ts)
        elif cmd_norm == "regime":
            say(text=":hourglass: Fetching macro indicators…",
                thread_ts=thread_ts)
            say(text=cmd_regime(), thread_ts=thread_ts)
        elif cmd_norm == "ipo":
            say(text=":hourglass: Probing IPO watchlist + EDGAR…",
                thread_ts=thread_ts)
            say(text=cmd_ipo(), thread_ts=thread_ts)
        elif cmd_norm == "chain":
            say(text=":hourglass: Scanning AI value chain…",
                thread_ts=thread_ts)
            say(text=cmd_chain(), thread_ts=thread_ts)
        elif cmd_norm == "watchlist":
            say(text=cmd_watchlist(args), thread_ts=thread_ts)
        elif cmd_norm == "pause":
            say(text=cmd_pause(), thread_ts=thread_ts)
        elif cmd_norm == "resume":
            say(text=cmd_resume(), thread_ts=thread_ts)
        elif cmd_norm == "run":
            trigger_cycle(say, thread_ts, dry_run=False)
        elif cmd_norm == "dry-run":
            trigger_cycle(say, thread_ts, dry_run=True)
        else:
            say(text=":grey_question: I didn't understand that — try `!help`.",
                thread_ts=thread_ts)
    except Exception as exc:
        log.exception("handler crashed for command %s", cmd_norm)
        try:
            say(text=f":x: `{cmd_norm}` failed: `{exc}`", thread_ts=thread_ts)
        except Exception:
            pass


# ── curl transport ───────────────────────────────────────────────────

POLL_INTERVAL_SECONDS = 3


def _make_say(token: str, channel_id: str):
    """Build a say(text, thread_ts=None) closure that posts via curl.

    Matches the signature handle_message / trigger_cycle already expect, so
    no command handler needs to change.
    """
    from kairos_alerts import _slack_api_call

    def say(text: str, thread_ts: Optional[str] = None) -> None:
        payload = {"channel": channel_id, "text": text}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        resp = _slack_api_call("chat.postMessage", token, payload)
        if not resp.get("ok"):
            log.warning("chat.postMessage failed: %s", resp.get("error"))

    return say


# ── Single-instance PID lock ─────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    """True if a process with this PID exists (signal 0 probes without killing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def acquire_lock() -> bool:
    """Take the single-instance lock. Returns False if another live instance holds it.

    Uses an atomic O_CREAT|O_EXCL create so two simultaneous starts can't both
    win the race. If the file already exists, the holding PID is probed: a live
    holder means we exit; a dead holder (stale lock) is reclaimed and retried
    once. On success the PID is written and an atexit hook removes the file.
    """
    for attempt in range(2):
        try:
            fd = os.open(PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                with open(PID_FILE) as f:
                    existing = int(f.read().strip())
            except (IOError, ValueError):
                existing = None
            if existing and existing != os.getpid() and _pid_alive(existing):
                log.error("Another commander is already running (pid %s) — "
                          "exiting.", existing)
                return False
            # Stale lock — reclaim and retry the atomic create once.
            log.warning("Reclaiming stale lock file (pid %s not running).",
                        existing)
            try:
                os.remove(PID_FILE)
            except OSError:
                pass
            continue
        else:
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            import atexit
            atexit.register(release_lock)
            return True

    log.error("Could not acquire lock after reclaiming stale file — exiting.")
    return False


def release_lock() -> None:
    """Remove the PID file if it still belongs to this process."""
    try:
        with open(PID_FILE) as f:
            if int(f.read().strip()) != os.getpid():
                return  # someone else owns it now; leave it alone
    except (IOError, ValueError):
        return
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def main():
    """curl long-poll commander — checks #kairos-commands every few seconds.

    Inbound and outbound Slack I/O both route through curl
    (kairos_alerts._slack_api_call); the Python socket layer is blocked for
    Slack on this host.
    """
    from kairos_alerts import _slack_api_call

    if not acquire_lock():
        sys.exit(0)  # clean exit — another instance owns the lock

    token = get_bot_token()
    if not token:
        log.error("Missing bot token — set SLACK_BOT_TOKEN in the "
                  "environment (~/.zshrc or the launchd plist). Never put it "
                  "in kairos_config.json; that file is tracked in git.")
        sys.exit(1)

    channel_id = get_commands_channel_id()
    if not channel_id:
        log.error("No channel ID configured for #%s in kairos_config.json — "
                  "nothing to poll.", COMMAND_CHANNEL_NAME)
        sys.exit(1)

    say = _make_say(token, channel_id)

    log.info("Kairos Commander starting (curl long-poll, paused=%s)…",
             get_paused())

    # Seed cursor to now so we only process NEW messages. Slack timestamps
    # use exactly 6 fractional digits; str(time.time()) emits 7, which Slack's
    # conversations.history `oldest` filter rejects (returns nothing), leaving
    # the cursor permanently stuck. Format to 6 decimals to match Slack.
    last_ts = f"{time.time():.6f}"
    log.info("Polling channel %s, seed last_ts=%s", channel_id, last_ts)

    poll_count = 0
    while True:
        try:
            resp = _slack_api_call(
                "conversations.history", token,
                {"channel": channel_id, "oldest": last_ts, "limit": 10},
            )
            poll_count += 1
            if poll_count % 10 == 0:
                log.info("Heartbeat: poll #%d ok=%s n_msgs=%d last_ts=%s",
                         poll_count, resp.get("ok"),
                         len(resp.get("messages", [])), last_ts)
            if not resp.get("ok"):
                log.warning("conversations.history failed: %s",
                            resp.get("error"))
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Slack returns newest-first; process oldest-first.
            for msg in reversed(resp.get("messages", [])):
                ts = msg.get("ts", "0")
                if float(ts) <= float(last_ts):
                    continue
                last_ts = ts  # advance regardless of message type
                if msg.get("bot_id") or msg.get("subtype"):
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                log.info("Received: %s", text)
                try:
                    handle_message(text, say=say, thread_ts=ts)
                except Exception:
                    log.exception("handle_message crashed")
        except Exception as exc:
            log.warning("Poll error: %s", exc)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Commander interrupted — exiting.")
    except Exception:
        log.exception("Fatal error in commander")
        # Allow launchd to restart us
        sys.exit(1)
