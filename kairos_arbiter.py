"""
Kairos Arbiter — Retrospective Trade-Outcome Analysis (Mistral)

A weekly + daily retrospective engine that reads closed trade outcomes and
open positions from the Kairos databases, hands them to the Mistral API, and
posts a structured review to Slack (#kairos-arbiter).  The Arbiter's job is to
judge whether conviction scores are calibrated, whether signals are firing on
appropriate setups, and what patterns explain wins and losses.

Data sources:
  kairos_ml_outcomes.db
    trade_outcomes      — closed/open trades w/ pnl_pct, signals_fired,
                          confluence_score, sector, outcome_label
    thesis_predictions  — conviction_score (1–10), signal_type, predicted move
                          (joined on thesis_predictions.decision_id = trade_id)

Modes:
  --mode daily   pull trades closed *today*; concise per-trade verdicts + flags
  --mode weekly  pull ALL closed trades (full timeline — intentional first run);
                 structured review w/ five named sections

Usage:
  export MISTRAL_API_KEY=...
  python3 kairos_arbiter.py --mode daily
  python3 kairos_arbiter.py --mode weekly
  python3 kairos_arbiter.py --mode weekly --dry-run   # print prompt, no API/Slack

────────────────────────────────────────────────────────────────────────────
launchd scheduling (do NOT create these yet — wire up after testing)

  Daily, 20:00 ET on weekdays (Mon–Fri)
  ~/Library/LaunchAgents/com.kairos.arbiter.daily.plist
    <key>ProgramArguments</key>
    <array>
      <string>/usr/bin/python3</string>
      <string>/Users/jelmore/TradingAgents/kairos_arbiter.py</string>
      <string>--mode</string>
      <string>daily</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
      <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
      <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
      <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
      <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
      <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
    </array>

  Weekly, 07:00 ET on Saturdays (launchd: Sunday=0 … Saturday=6)
  ~/Library/LaunchAgents/com.kairos.arbiter.weekly.plist
    <key>ProgramArguments</key>
    <array>
      <string>/usr/bin/python3</string>
      <string>/Users/jelmore/TradingAgents/kairos_arbiter.py</string>
      <string>--mode</string>
      <string>weekly</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
      <key>Weekday</key><integer>6</integer>
      <key>Hour</key><integer>7</integer>
      <key>Minute</key><integer>0</integer>
    </dict>

  NOTE: launchd fires on the machine's local wall-clock time. These entries
  assume the host is set to America/New_York. Add EnvironmentVariables with
  MISTRAL_API_KEY to each plist, or source it from a wrapper script.
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DB_PATH = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
KAIROS_DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "arbiter_reports")

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-medium-latest"
MISTRAL_TIMEOUT = 120

ARBITER_CHANNEL = "#kairos-arbiter"   # not yet in kairos_config.json channel map
ALERTS_CHANNEL = "alerts"             # logical key → C0AQA4ZH0P2

SYSTEM_PROMPT = (
    "You are the Arbiter for Kairos, an autonomous AI trading system built on "
    "information synthesis arbitrage. Kairos fires signals based on non-price "
    "data: insider filings (HOT-INSIDER, 21-day expected hold), congressional "
    "trades (HOT-CONGRESS, 30d), unusual options flow (HOT-OPTIONS, 7d), mean "
    "reversion (HOT-REVERSION, 5d), earnings catalysts (HOT-EARNINGS, 10d), "
    "event-driven catalysts (HOT-CATALYST, 14d), and AI value chain propagation "
    "(HOT-CHAIN, 7d). Each trade carries a conviction score on a 1–10 scale "
    "(higher = stronger conviction). ALWAYS render conviction as N/10 — never "
    "out of 15 or any other denominator.\n\n"
    "Your job is COUNTERFACTUAL, not bookkeeping. A trade making money is not the "
    "same as a trade being run well. You are given, for each closed trade, the "
    "full price path around it: the peak it reached WHILE WE HELD it (max gain "
    "available), what we actually realized, and where the price went AFTER we "
    "exited. Reason about EDGE CAPTURED versus EDGE AVAILABLE. Center every "
    "judgement on four questions:\n"
    "  1. Did we exit too EARLY? (price kept climbing after we sold — forgone gain)\n"
    "  2. Did we exit too LATE? (we let a large in-hold peak give back before exiting)\n"
    "  3. Should we NEVER HAVE BOUGHT this? (it never worked / never approached thesis)\n"
    "  4. (Missed entries are out of scope for now.)\n"
    "Be specific, cite the numbers you were given, and be willing to call a "
    "profitable trade poorly managed when the data supports it."
)

# ── Structured findings appendix (Arbiter → council feedback, Phase A) ─
# Sentinels delimit a machine-readable findings block the model appends AFTER its
# normal prose. parse_findings extracts the JSON between them; format_slack_report
# strips everything from the opening sentinel onward so Slack is unchanged.
FINDINGS_START = "===KAIROS_FINDINGS_V1==="
FINDINGS_END = "===END_KAIROS_FINDINGS==="

# The three axes and their sign conventions, handed to the model verbatim so the
# sign of each score is interpretable. (Mirrors kairos_log_db.AXIS_POSITIVE_MEANS.)
_AXIS_DEFS = (
    "  - exit_timing: Positive = we have been exiting too LATE (giving back "
    "in-hold peaks); correction is to exit earlier.\n"
    "  - reallocation_aggressiveness: Positive = we have been reallocating too "
    "EAGERLY (rotating before theses mature); correction is to hold longer before "
    "rotating.\n"
    "  - conviction_calibration: Positive = conviction scores have been "
    "OVER-confident (high-conviction trades underperformed); correction is to "
    "discount conviction.\n"
)

FINDINGS_APPENDIX = (
    "\n----------------------------------------------------------------------\n"
    "STRUCTURED FINDINGS APPENDIX (required)\n"
    "AFTER the prose review above, append a machine-readable findings block, "
    "delimited EXACTLY by these two sentinels each on their own line:\n"
    f"{FINDINGS_START}\n"
    "{ ...json... }\n"
    f"{FINDINGS_END}\n\n"
    "Score every trade you reviewed onto these three axes. The SIGN of each score "
    "follows its positive_means definition; the MAGNITUDE |score| (0..1) is the "
    "strength of the evidence:\n"
    f"{_AXIS_DEFS}\n"
    "Emit valid JSON with this exact shape:\n"
    "{\n"
    '  "axes": [\n'
    '    {"axis":"exit_timing","score":<float -1..1>,"sample_size":<int>,'
    '"trade_ids":[...],"observation":"<one sentence>","confidence":<float 0..1>},\n'
    '    {"axis":"reallocation_aggressiveness","score":<float -1..1>,'
    '"sample_size":<int>,"trade_ids":[...],"observation":"<one sentence>",'
    '"confidence":<float 0..1>},\n'
    '    {"axis":"conviction_calibration","score":<float -1..1>,'
    '"sample_size":<int>,"trade_ids":[...],"observation":"<one sentence>",'
    '"confidence":<float 0..1>}\n'
    "  ],\n"
    '  "qualitative": [\n'
    '    {"category":"<short tag>","observation":"<one sentence>",'
    '"trade_ids":[...],"confidence":<float 0..1>}\n'
    "  ]\n"
    "}\n\n"
    "Rules: include ALL THREE axes every time — if evidence for an axis is thin, "
    "return score 0 with a low confidence, do NOT omit the axis. trade_ids cite the "
    "trade_id values you based the score on. qualitative captures any other "
    "recurring pattern worth tracking and MAY be an empty list. Output nothing "
    "after the closing sentinel.\n"
)


# ── Database access ──────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(ML_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _dedup_by_trade(rows: list[sqlite3.Row]) -> list[dict]:
    """Collapse the LEFT JOIN to one row per trade_id (first prediction wins)."""
    seen: dict[str, dict] = {}
    for r in rows:
        d = dict(r)
        tid = d.get("trade_id")
        if tid not in seen:
            seen[tid] = d
    return list(seen.values())


def fetch_closed_trades(mode: str, today: str) -> list[dict]:
    """Closed trades for the requested window.

    daily  → trades whose exit date is today (UTC).
    weekly → every closed trade in the database (full timeline).
    """
    sql = """
        SELECT t.trade_id, t.ticker, t.action,
               t.timestamp_entry, t.timestamp_exit,
               t.price_entry, t.price_exit, t.quantity,
               t.pnl_pct, t.pnl_dollar, t.hold_duration_mins,
               t.signals_fired, t.confluence_score, t.sector, t.outcome_label,
               tp.conviction_score, tp.signal_type,
               tp.predicted_direction, tp.predicted_return_pct
        FROM trade_outcomes t
        LEFT JOIN thesis_predictions tp ON tp.decision_id = t.trade_id
        WHERE t.timestamp_exit IS NOT NULL
    """
    params: tuple = ()
    if mode == "daily":
        sql += " AND substr(t.timestamp_exit, 1, 10) = ?"
        params = (today,)
    sql += " ORDER BY t.timestamp_exit"

    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return _dedup_by_trade(rows)


def fetch_open_positions() -> list[dict]:
    """Current open positions with entry date, signals, and conviction."""
    sql = """
        SELECT t.trade_id, t.ticker, t.action, t.timestamp_entry,
               t.price_entry, t.quantity,
               t.signals_fired, t.confluence_score, t.sector,
               tp.conviction_score, tp.signal_type,
               tp.predicted_direction, tp.predicted_return_pct,
               tp.predicted_timeframe_days
        FROM trade_outcomes t
        LEFT JOIN thesis_predictions tp ON tp.decision_id = t.trade_id
        WHERE t.timestamp_exit IS NULL
        ORDER BY t.timestamp_entry
    """
    conn = _connect()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return _dedup_by_trade(rows)


# ── Live enrichment (yfinance — no IBKR dependency) ──────────────────

# Expected hold window (days) per signal, used to judge thesis timing.
HOLD_WINDOWS = {
    "HOT-EARNINGS": 10,
    "HOT-INSIDER": 21,
    "HOT-CONGRESS": 30,
    "HOT-OPTIONS": 7,
    "HOT-REVERSION": 5,
    "HOT-CATALYST": 14,
    "HOT-CHAIN": 7,
    "DEFAULT": 7,
}


def _fetch_current_price(ticker: str) -> float | None:
    """Latest price for a ticker via yfinance. None on any failure.

    Prefers the most recent daily close (history) and falls back to the
    live quote fields, mirroring the rest of the codebase.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", auto_adjust=False)
        if hist is not None and len(hist) > 0:
            close = float(hist["Close"].iloc[-1])
            if close > 0:
                return round(close, 2)
    except Exception:
        pass

    try:
        info = yf.Ticker(ticker).info or {}
        for key in ("regularMarketPrice", "currentPrice", "previousClose"):
            v = info.get(key)
            if v is None:
                continue
            f = float(v)
            if f > 0:
                return round(f, 2)
    except Exception:
        pass

    return None


def _hold_window_days(position: dict) -> int:
    """Expected hold window for a position's signal type (falls back to DEFAULT)."""
    sig = position.get("signal_type")
    if not sig:
        sigs = _signals_list(position.get("signals_fired"))
        sig = sigs[0] if sigs else None
    return HOLD_WINDOWS.get(sig, HOLD_WINDOWS["DEFAULT"])


def _days_since_entry(timestamp_entry) -> int | None:
    """Whole days between the entry timestamp and now (UTC). None if unparseable.

    Entry timestamps look like '2026-05-18 19:33:37 UTC'.
    """
    if not timestamp_entry:
        return None
    raw = str(timestamp_entry).replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - dt).days
        except ValueError:
            continue
    return None


def enrich_open_positions(positions: list[dict]) -> list[dict]:
    """Add live price, unrealized P&L, and thesis-window status to each position.

    Uses yfinance only (no IBKR), so the Arbiter runs whether or not IB Gateway
    is connected. A failed lookup for one ticker leaves its price fields None and
    logs a warning — it never aborts the run.
    """
    for p in positions:
        ticker = p.get("ticker")
        action = (p.get("action") or "BUY").upper()

        # Entry price from DB, with a pnl-based estimate as a last resort.
        entry_price = p.get("price_entry")
        try:
            entry_price = float(entry_price) if entry_price is not None else None
        except (TypeError, ValueError):
            entry_price = None

        current_price = _fetch_current_price(ticker) if ticker else None
        if current_price is None:
            print(f"  WARNING: price lookup failed for {ticker} — fields set to None")

        if entry_price is None and current_price is not None and p.get("pnl_pct") is not None:
            try:
                entry_price = round(current_price / (1 + float(p["pnl_pct"]) / 100.0), 2)
            except (TypeError, ValueError, ZeroDivisionError):
                entry_price = None

        # Unrealized P&L (long vs short aware).
        unrealized_pct = None
        unrealized_dollar = None
        if current_price is not None and entry_price not in (None, 0):
            if action == "SELL":
                unrealized_pct = (entry_price - current_price) / entry_price * 100.0
            else:
                unrealized_pct = (current_price - entry_price) / entry_price * 100.0
            unrealized_pct = round(unrealized_pct, 2)

            qty = p.get("quantity")
            try:
                qty = float(qty) if qty is not None else None
            except (TypeError, ValueError):
                qty = None
            if qty is not None:
                direction = -1.0 if action == "SELL" else 1.0
                unrealized_dollar = round((current_price - entry_price) * qty * direction, 2)

        days_held = _days_since_entry(p.get("timestamp_entry"))
        if days_held is None:
            days_remaining = None
            thesis_status = "UNKNOWN"
        else:
            days_remaining = _hold_window_days(p) - days_held
            thesis_status = "IN_WINDOW" if days_remaining >= 0 else "OVERDUE"

        p["current_price"] = current_price
        p["entry_price"] = entry_price
        p["unrealized_pnl_pct"] = unrealized_pct
        if unrealized_dollar is not None:
            p["unrealized_pnl_dollar"] = unrealized_dollar
        p["days_held"] = days_held
        p["days_remaining"] = days_remaining
        p["thesis_status"] = thesis_status

    return positions


# ── Price-path enrichment (counterfactual: edge captured vs available) ─

def _parse_utc(ts):
    """Parse a Kairos timestamp to an aware UTC datetime. None if unparseable.

    Handles both ledger formats: '2026-06-15 19:21:11 UTC' (decisions) and
    '2026-06-17T15:23:11Z' (trade_outcomes). The trailing ' UTC'/'Z' breaks
    naive parsing, so strip it before strptime.
    """
    if not ts:
        return None
    raw = (str(ts).strip()
           .replace(" UTC", "")
           .replace("Z", "")
           .replace("T", " ")
           .strip())
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _load_authoritative_sells() -> dict:
    """Filled SELL records from kairos.db keyed by ticker.

    The decisions table's data_inputs blob on a Filled SELL is the authoritative
    record of a realized trade: blended entry_price, sell_price, pnl_pct, and the
    sell_trigger. The ml_outcomes ledger can split a multi-fill position into
    per-lot rows and recompute pnl per-lot (e.g. SPCX showed +3.57% on one lot vs
    the true blended +6.6%), so we prefer this record. READ-ONLY.
    """
    sells: dict = defaultdict(list)
    if not os.path.exists(KAIROS_DB_PATH):
        print(f"  WARNING: {KAIROS_DB_PATH} not found — using ledger pnl only")
        return sells
    conn = sqlite3.connect(KAIROS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ticker, timestamp, data_inputs FROM decisions "
            "WHERE action = 'SELL' AND execution_status = 'Filled'"
        ).fetchall()
    finally:
        conn.close()

    for r in rows:
        di = r["data_inputs"]
        try:
            di = json.loads(di) if di else {}
            if isinstance(di, str):
                di = json.loads(di)
        except (TypeError, json.JSONDecodeError):
            di = {}
        if not isinstance(di, dict):
            continue
        sells[r["ticker"]].append({
            "dt": _parse_utc(r["timestamp"]),
            "entry_price": di.get("entry_price"),
            "exit_price": di.get("sell_price"),
            "pnl_pct": di.get("pnl_pct"),
            "pnl_dollars": di.get("pnl_dollars"),
            "sell_trigger": di.get("sell_trigger"),
            "holding_days": di.get("holding_days"),
        })
    return sells


def _match_sell(sells: dict, ticker, exit_dt):
    """Authoritative SELL for a closed trade — nearest by exit timestamp."""
    cands = sells.get(ticker) or []
    if not cands:
        return None
    if exit_dt is None:
        return cands[0]
    best, best_diff = None, None
    for c in cands:
        if c["dt"] is None:
            continue
        diff = abs((c["dt"] - exit_dt).total_seconds())
        if best_diff is None or diff < best_diff:
            best, best_diff = c, diff
    return best


def _download_daily(tickers: list, start: str, end: str) -> dict:
    """One BATCHED yfinance download → {ticker: DataFrame of daily OHLC}.

    Batches all tickers into a single yf.download call (40+ overdue positions
    must not become 40+ sequential requests). Returns {} if yfinance is missing
    or the call fails; per-ticker extraction failures are skipped, not fatal.
    """
    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return {}
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        print("  WARNING: yfinance/pandas unavailable — price-path enrichment skipped")
        return {}

    try:
        df = yf.download(tickers, start=start, end=end, auto_adjust=False,
                         progress=False, group_by="ticker", threads=True)
    except Exception as exc:
        print(f"  WARNING: yf.download failed for {len(tickers)} ticker(s): {exc}")
        return {}

    if df is None or len(df) == 0:
        print(f"  WARNING: yf.download returned no rows for {len(tickers)} ticker(s)")
        return {}

    out: dict = {}
    multi = isinstance(df.columns, pd.MultiIndex)
    for t in tickers:
        try:
            if multi:
                lvl0 = df.columns.get_level_values(0)
                sub = df[t] if t in lvl0 else df.xs(t, axis=1, level=-1)
            else:
                sub = df  # single ticker, flat columns
            if sub is None or "High" not in sub.columns:
                continue
            sub = sub.dropna(how="all")
            if len(sub) == 0:
                continue
            out[t] = sub
        except Exception:
            continue

    missing = [t for t in tickers if t not in out]
    if missing:
        print(f"  WARNING: no usable price history for: {', '.join(missing)}")
    return out


def enrich_closed_trades(closed: list[dict]) -> list[dict]:
    """Attach authoritative realized P&L + the full price path to each closed trade.

    For each trade:
      • realized_pnl_pct / entry_price / exit_price — from the authoritative SELL
        record (kairos.db) when matched, else the ledger recompute (logged).
      • max_gain_during_hold_pct + give_back_pct — MFE during the hold (exposes
        "exited too late").
      • forgone_gain_pct + current_vs_exit_pct — post-exit path (exposes
        "exited too early").
    Any ticker whose history fails to load is skipped gracefully with a note.
    """
    if not closed:
        return closed

    sells = _load_authoritative_sells()
    now = datetime.now(timezone.utc)
    min_start = None

    for t in closed:
        t["_entry_dt"] = _parse_utc(t.get("timestamp_entry"))
        t["_exit_dt"] = _parse_utc(t.get("timestamp_exit"))

        ledger_pnl = None
        try:
            ledger_pnl = round(float(t["pnl_pct"]), 2) if t.get("pnl_pct") is not None else None
        except (TypeError, ValueError):
            ledger_pnl = None

        m = _match_sell(sells, t.get("ticker"), t["_exit_dt"])
        if m:
            if m["entry_price"] is not None:
                t["entry_price"] = round(float(m["entry_price"]), 2)
            if m["exit_price"] is not None:
                t["exit_price"] = round(float(m["exit_price"]), 2)
            if m["pnl_pct"] is not None:
                t["realized_pnl_pct"] = round(float(m["pnl_pct"]), 2)
            t["sell_trigger"] = m.get("sell_trigger")
            t["pnl_source"] = "kairos.db decisions SELL (authoritative)"
            print(f"  {t.get('ticker')}: authoritative pnl "
                  f"{t.get('realized_pnl_pct')}% (ledger {ledger_pnl}%), "
                  f"trigger {m.get('sell_trigger')}")
        else:
            # No authoritative match — fall back to the ledger recompute.
            t["realized_pnl_pct"] = ledger_pnl
            try:
                if t.get("price_entry") is not None:
                    t["entry_price"] = round(float(t["price_entry"]), 2)
                if t.get("price_exit") is not None:
                    t["exit_price"] = round(float(t["price_exit"]), 2)
            except (TypeError, ValueError):
                pass
            t["pnl_source"] = "trade_outcomes ledger (no SELL match)"
            print(f"  {t.get('ticker')}: no authoritative SELL match — "
                  f"using ledger pnl {ledger_pnl}%")

        if t["_entry_dt"] and (min_start is None or t["_entry_dt"] < min_start):
            min_start = t["_entry_dt"]

    tickers = [t.get("ticker") for t in closed if t.get("ticker")]
    if not tickers or min_start is None:
        print("  No parseable entry dates — skipping closed-trade price paths")
        for t in closed:
            t.pop("_entry_dt", None)
            t.pop("_exit_dt", None)
        return closed

    start = (min_start - timedelta(days=2)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    hist = _download_daily(tickers, start, end)

    for t in closed:
        ticker = t.get("ticker")
        df = hist.get(ticker)
        if df is None:
            print(f"  WARNING: no price history for {ticker} — path metrics skipped")
            continue
        edt, xdt = t.get("_entry_dt"), t.get("_exit_dt")
        entry_p, exit_p = t.get("entry_price"), t.get("exit_price")
        realized = t.get("realized_pnl_pct")
        try:
            highs, closes = df["High"], df["Close"]

            # MFE during the hold: peak High between entry and exit (inclusive).
            if edt and xdt:
                hold_highs = highs.loc[edt.strftime("%Y-%m-%d"):xdt.strftime("%Y-%m-%d")]
                if len(hold_highs) > 0 and entry_p:
                    peak = float(hold_highs.max())
                    t["in_hold_peak_price"] = round(peak, 2)
                    t["max_gain_during_hold_pct"] = round((peak - entry_p) / entry_p * 100.0, 2)
                    if realized is not None:
                        t["give_back_pct"] = round(t["max_gain_during_hold_pct"] - realized, 2)

            # Post-exit path: peak + current from the day AFTER exit to now.
            if xdt:
                post_start = (xdt + timedelta(days=1)).strftime("%Y-%m-%d")
                post_highs = highs.loc[post_start:]
                post_closes = closes.loc[post_start:]
                if len(post_closes) > 0:
                    cur = float(post_closes.iloc[-1])
                    t["current_price"] = round(cur, 2)
                    if exit_p:
                        t["current_vs_exit_pct"] = round((cur - exit_p) / exit_p * 100.0, 2)
                if len(post_highs) > 0 and exit_p:
                    pk = float(post_highs.max())
                    t["peak_since_exit_price"] = round(pk, 2)
                    t["forgone_gain_pct"] = round((pk - exit_p) / exit_p * 100.0, 2)

            print(f"  {ticker}: in-hold peak {t.get('in_hold_peak_price')} "
                  f"(+{t.get('max_gain_during_hold_pct')}%, give-back {t.get('give_back_pct')}), "
                  f"post-exit peak {t.get('peak_since_exit_price')} "
                  f"(forgone {t.get('forgone_gain_pct')}%), now {t.get('current_price')} "
                  f"({t.get('current_vs_exit_pct')}% vs exit)")
        except Exception as exc:
            print(f"  WARNING: path metrics failed for {ticker}: {exc}")

    for t in closed:
        t.pop("_entry_dt", None)
        t.pop("_exit_dt", None)
    return closed


def enrich_overdue_positions(positions: list[dict]) -> list[dict]:
    """For OVERDUE open positions, add peak-during-hold and distance from it.

    Reuses the existing overdue flag (thesis_status == 'OVERDUE' set by
    enrich_open_positions). Adds peak_price_during_hold, peak_gain_during_hold_pct,
    and pct_off_peak (current vs the hold's high-water mark; negative = below it).
    Batched download; missing history is skipped, not fatal.
    """
    overdue = [p for p in positions if p.get("thesis_status") == "OVERDUE"]
    if not overdue:
        print("  No overdue positions to enrich with peak metrics")
        return positions

    now = datetime.now(timezone.utc)
    starts = [_parse_utc(p.get("timestamp_entry")) for p in overdue]
    starts = [s for s in starts if s]
    if not starts:
        print("  Overdue positions have no parseable entry dates — peak metrics skipped")
        return positions
    min_start = min(starts)

    tickers = [p.get("ticker") for p in overdue if p.get("ticker")]
    start = (min_start - timedelta(days=2)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    print(f"  Enriching {len(overdue)} overdue position(s) with peak metrics "
          f"({len(set(tickers))} tickers)")
    hist = _download_daily(tickers, start, end)

    for p in overdue:
        ticker = p.get("ticker")
        df = hist.get(ticker)
        if df is None:
            print(f"  WARNING: no price history for overdue {ticker} — peak metrics skipped")
            continue
        edt = _parse_utc(p.get("timestamp_entry"))
        entry_p, cur = p.get("entry_price"), p.get("current_price")
        try:
            hold_highs = df["High"].loc[edt.strftime("%Y-%m-%d"):] if edt else df["High"]
            if len(hold_highs) > 0:
                peak = float(hold_highs.max())
                p["peak_price_during_hold"] = round(peak, 2)
                if entry_p:
                    p["peak_gain_during_hold_pct"] = round((peak - entry_p) / entry_p * 100.0, 2)
                if cur and peak:
                    p["pct_off_peak"] = round((cur - peak) / peak * 100.0, 2)
            print(f"  {ticker} (overdue): peak {p.get('peak_price_during_hold')} "
                  f"(+{p.get('peak_gain_during_hold_pct')}% from entry), "
                  f"now {p.get('pct_off_peak')}% off peak")
        except Exception as exc:
            print(f"  WARNING: peak metrics failed for overdue {ticker}: {exc}")

    return positions


# ── Aggregates (concrete numbers for the model to reason over) ────────

def _signals_list(raw) -> list[str]:
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _conviction_bucket(score) -> str:
    if score is None:
        return "unknown"
    try:
        s = float(score)
    except (ValueError, TypeError):
        return "unknown"
    if s <= 4:
        return "low (1–4)"
    if s <= 7:
        return "mid (5–7)"
    return "high (8–10)"


def compute_aggregates(closed: list[dict]) -> dict:
    """Win rate / avg return broken out by signal and by conviction bucket."""
    by_signal: dict[str, list[float]] = defaultdict(list)
    by_bucket: dict[str, list[float]] = defaultdict(list)
    wins = losses = 0
    returns: list[float] = []

    for t in closed:
        # Prefer the authoritative realized P&L attached by enrich_closed_trades;
        # fall back to the raw ledger value when no SELL record was matched.
        pnl = t.get("realized_pnl_pct")
        if pnl is None:
            pnl = t.get("pnl_pct")
        if pnl is None:
            continue
        pnl = float(pnl)
        returns.append(pnl)
        if pnl > 0:
            wins += 1
        else:
            losses += 1

        sigs = _signals_list(t.get("signals_fired")) or ["(none)"]
        for s in sigs:
            by_signal[s].append(pnl)

        by_bucket[_conviction_bucket(t.get("conviction_score"))].append(pnl)

    def summarize(d: dict[str, list[float]]) -> dict[str, dict]:
        out = {}
        for k, vals in d.items():
            n = len(vals)
            win_n = sum(1 for v in vals if v > 0)
            out[k] = {
                "n": n,
                "win_rate": round(win_n / n, 3) if n else 0.0,
                "avg_return_pct": round(sum(vals) / n, 2) if n else 0.0,
            }
        return out

    total = wins + losses
    return {
        "total_closed": total,
        "overall_win_rate": round(wins / total, 3) if total else 0.0,
        "avg_return_pct": round(sum(returns) / len(returns), 2) if returns else 0.0,
        "by_signal": summarize(by_signal),
        "by_conviction_bucket": summarize(by_bucket),
    }


# ── Trade serialization for the prompt ───────────────────────────────

def _slim_closed(t: dict) -> dict:
    ledger_pnl = round(float(t["pnl_pct"]), 2) if t.get("pnl_pct") is not None else None
    realized = t.get("realized_pnl_pct")
    return {
        "ticker": t.get("ticker"),
        "action": t.get("action"),
        "entry": t.get("timestamp_entry"),
        "exit": t.get("timestamp_exit"),
        "entry_price": t.get("entry_price"),
        "exit_price": t.get("exit_price"),
        # realized_pnl_pct is the authoritative number; pnl_pct mirrors it so any
        # downstream reader keying on pnl_pct sees the corrected value.
        "realized_pnl_pct": realized if realized is not None else ledger_pnl,
        "pnl_pct": realized if realized is not None else ledger_pnl,
        "ledger_pnl_pct": ledger_pnl,
        "pnl_source": t.get("pnl_source"),
        "sell_trigger": t.get("sell_trigger"),
        # Price path — edge captured vs edge available.
        "in_hold_peak_price": t.get("in_hold_peak_price"),
        "max_gain_during_hold_pct": t.get("max_gain_during_hold_pct"),
        "give_back_pct": t.get("give_back_pct"),
        "peak_since_exit_price": t.get("peak_since_exit_price"),
        "forgone_gain_pct": t.get("forgone_gain_pct"),
        "current_price": t.get("current_price"),
        "current_vs_exit_pct": t.get("current_vs_exit_pct"),
        "outcome": t.get("outcome_label"),
        "signals": _signals_list(t.get("signals_fired")),
        "confluence_score": t.get("confluence_score"),
        "conviction_score": t.get("conviction_score"),
        "signal_type": t.get("signal_type"),
        "predicted_direction": t.get("predicted_direction"),
        "predicted_return_pct": t.get("predicted_return_pct"),
        "sector": t.get("sector"),
        "hold_days": round(t["hold_duration_mins"] / 1440.0, 1)
        if t.get("hold_duration_mins") else None,
    }


def _slim_open(t: dict) -> dict:
    return {
        "ticker": t.get("ticker"),
        "entry": t.get("timestamp_entry"),
        "signals": _signals_list(t.get("signals_fired")),
        "confluence_score": t.get("confluence_score"),
        "conviction_score": t.get("conviction_score"),
        "signal_type": t.get("signal_type"),
        "predicted_direction": t.get("predicted_direction"),
        "predicted_return_pct": t.get("predicted_return_pct"),
        "predicted_timeframe_days": t.get("predicted_timeframe_days"),
        "sector": t.get("sector"),
        "current_price": t.get("current_price"),
        "entry_price": t.get("entry_price"),
        "unrealized_pnl_pct": t.get("unrealized_pnl_pct"),
        "unrealized_pnl_dollar": t.get("unrealized_pnl_dollar"),
        "days_held": t.get("days_held"),
        "days_remaining": t.get("days_remaining"),
        "thesis_status": t.get("thesis_status"),
        # Peak-during-hold metrics (populated for OVERDUE positions).
        "peak_price_during_hold": t.get("peak_price_during_hold"),
        "peak_gain_during_hold_pct": t.get("peak_gain_during_hold_pct"),
        "pct_off_peak": t.get("pct_off_peak"),
    }


def build_user_prompt(mode: str, today: str, closed: list[dict],
                      open_pos: list[dict], stats: dict) -> str:
    payload = {
        "review_date": today,
        "mode": mode,
        "aggregates": stats,
        "closed_trades": [_slim_closed(t) for t in closed],
        "open_positions": [_slim_open(t) for t in open_pos],
    }
    data_blob = json.dumps(payload, indent=2, default=str)

    # ── Shared field glossary + counterfactual framing ──────────────
    field_guide = (
        "FIELDS per closed trade (all percentages are already computed for you):\n"
        "  entry_price / exit_price — authoritative blended fill prices.\n"
        "  realized_pnl_pct — what we ACTUALLY captured (authoritative; use this, "
        "not ledger_pnl_pct).\n"
        "  max_gain_during_hold_pct — the BEST gain available while we held "
        "(price peaked here before we exited).\n"
        "  give_back_pct = max_gain_during_hold_pct - realized_pnl_pct — points "
        "we let evaporate from the in-hold peak before exiting. LARGE give-back = "
        "candidate for EXITED TOO LATE.\n"
        "  forgone_gain_pct = (peak_since_exit - exit_price)/exit_price — how much "
        "MORE the price made AFTER we sold. LARGE positive = candidate for EXITED "
        "TOO EARLY.\n"
        "  current_vs_exit_pct — where the price sits now relative to our exit.\n"
        "  conviction_score — on a 1–10 scale; render as N/10.\n\n"
        "Required per CLOSED trade, two explicit verdicts citing the numbers:\n"
        "  EXIT TIMING: PREMATURE (cite forgone_gain_pct left on the table after "
        "exit) / LATE (cite give_back_pct surrendered from the in-hold peak) / "
        "WELL-TIMED. Judge edge captured vs edge available — NOT merely whether "
        "the trade made money. A +6% win that peaked +27% mid-hold was exited "
        "LATE, not won cleanly.\n"
        "  ENTRY QUALITY: JUSTIFIED (the trade worked / approached its thesis) vs "
        "SHOULD-NOT-HAVE-BOUGHT (underwater most of the hold, never approached its "
        "predicted target). Use the price path to decide.\n"
    )
    overdue_guide = (
        "Each open position includes live data: entry_price, current_price, "
        "unrealized_pnl_pct, days_held, days_remaining (expected hold window minus "
        "days held; negative = OVERDUE by that many days), thesis_status, and for "
        "OVERDUE positions also peak_gain_during_hold_pct and pct_off_peak "
        "(how far current price sits below its in-hold high-water mark).\n"
        "Required per OVERDUE position, one verdict citing unrealized P&L and "
        "days overdue:\n"
        "  HOLD (still has upside — justify from signal/thesis and how close to "
        "peak it is) vs EXIT (thesis has decayed, capital better redeployed — "
        "justify). Cite unrealized_pnl_pct and how many days past the window.\n"
    )

    if mode == "weekly":
        instructions = (
            "Review the FULL trade history below through a COUNTERFACTUAL lens: "
            "did we exit too early, exit too late, or buy something we never "
            "should have? (Missed entries are out of scope for now.)\n\n"
            + field_guide + "\n" + overdue_guide + "\n"
            "Produce a structured report with exactly these sections, each as a "
            "markdown header:\n"
            "  *Exit & Entry Verdicts* — per closed trade, the EXIT TIMING and "
            "ENTRY QUALITY verdicts above, with the numbers cited.\n"
            "  *Signal Performance* — per-signal win rate / avg return AND edge "
            "capture (are certain signals systematically exited too early/late?).\n"
            "  *Conviction Calibration* — do higher conviction buckets (1–10 "
            "scale) win more than lower ones? Quantify and call out inversions.\n"
            "  *Patterns & Anomalies* — recurring setups behind wins, losses, and "
            "mistimed exits; sector clustering; hold-time effects.\n"
            "  *Recommendations* — specific, actionable changes to exit rules, "
            "signal thresholds, scoring, or position sizing.\n"
            "  *Open Positions Watch* — the HOLD vs EXIT verdict per OVERDUE "
            "position; flag anything deep underwater.\n"
            "  *Flags* — systemic issues only (signal vacuum, overdue cluster, "
            "conviction/outcome mismatch). Omit if nothing is notable.\n"
        )
    else:  # daily
        instructions = (
            "Review the trades closed TODAY below through a COUNTERFACTUAL lens: "
            "did we exit too early, exit too late, or buy something we never "
            "should have? (Missed entries are out of scope for now.) Reason about "
            "edge captured vs edge available — NOT just whether each trade made "
            "money. Keep it tight.\n\n"
            + field_guide + "\n" + overdue_guide + "\n"
            "Output:\n"
            "  - Per closed trade: ticker, realized_pnl_pct, then the EXIT TIMING "
            "verdict (PREMATURE/LATE/WELL-TIMED with give_back_pct or "
            "forgone_gain_pct cited) and the ENTRY QUALITY verdict "
            "(JUSTIFIED / SHOULD-NOT-HAVE-BOUGHT). One or two tight lines each.\n"
            "  - A brief *Overdue Watch* — only OVERDUE positions, each with a "
            "HOLD vs EXIT verdict (cite unrealized P&L and days overdue). Omit if "
            "none.\n"
            "  - A short *Flags* section: systemic items only (conviction/outcome "
            "mismatches, overdue clusters, signal vacuum). Omit if nothing notable.\n"
            "Do not produce the full weekly section structure.\n"
        )

    return (
        f"{instructions}\n"
        f"{FINDINGS_APPENDIX}\n"
        f"Review date: {today}  |  Mode: {mode}\n"
        f"Closed trades in window: {len(closed)}  |  "
        f"Open positions: {len(open_pos)}\n\n"
        f"DATA (JSON):\n{data_blob}\n"
    )


# ── Mistral ──────────────────────────────────────────────────────────

def call_mistral(system_prompt: str, user_prompt: str, api_key: str) -> str:
    resp = requests.post(
        MISTRAL_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=MISTRAL_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Slack ────────────────────────────────────────────────────────────

def post_to_slack(channel: str, text: str) -> bool:
    """Post via the shared kairos_alerts plumbing (curl + bot token)."""
    sys.path.insert(0, SCRIPT_DIR)
    from kairos_alerts import post_message
    return post_message(channel, text)


def format_slack_report(mode: str, today: str, body: str,
                        closed_n: int, open_n: int) -> str:
    label = "Weekly Review" if mode == "weekly" else "Daily Review"
    header = (
        f":balance_scale: *Kairos Arbiter — {label}*  |  {today}\n"
        f"Closed in window: {closed_n}  |  Open positions: {open_n}\n"
        f"{'─' * 40}\n"
    )
    # Strip the structured findings appendix (everything from the opening
    # sentinel onward) so the Slack post is exactly the prose review — the
    # machine-readable block must never reach Slack.
    if body:
        cut = body.find(FINDINGS_START)
        if cut != -1:
            body = body[:cut].rstrip()
    return header + body


# ── Report persistence ───────────────────────────────────────────────

def write_report(mode: str, today: str, system_prompt: str, user_prompt: str,
                 closed: list[dict], open_pos: list[dict], stats: dict,
                 mistral_response: str | None, error: str | None) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{today}_{mode}.json")
    record = {
        "mode": mode,
        "review_date": today,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model": MISTRAL_MODEL,
        "input": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "aggregates": stats,
            "closed_trades": [_slim_closed(t) for t in closed],
            "open_positions": [_slim_open(t) for t in open_pos],
        },
        "mistral_response": mistral_response,
        "error": error,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path


# ── Structured findings: parse + persist (Phase A) ───────────────────

def parse_findings(text: str) -> dict | None:
    """Extract and parse the findings JSON block delimited by the sentinels.

    Returns the parsed dict, or None on missing sentinels / JSON error (logs a
    clear warning to stderr; never raises, never posts).
    """
    if not text:
        print("  parse_findings: empty response — no findings block", file=sys.stderr)
        return None
    start = text.find(FINDINGS_START)
    end = text.find(FINDINGS_END)
    if start == -1 or end == -1 or end < start:
        print("  parse_findings: WARNING — findings sentinels not found "
              f"({FINDINGS_START} / {FINDINGS_END}); skipping capture", file=sys.stderr)
        return None
    blob = text[start + len(FINDINGS_START):end].strip()
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  parse_findings: WARNING — could not parse findings JSON: {exc}",
              file=sys.stderr)
        return None


def persist_findings(run_id: str, mode: str, findings: dict) -> int:
    """Write one row per axis entry and one row per qualitative entry to
    arbiter_findings. Returns the number of rows inserted. Tolerant of missing
    keys; never raises on malformed individual entries.
    """
    if not findings:
        return 0
    sys.path.insert(0, SCRIPT_DIR)
    from kairos_log_db import get_connection, init_db
    init_db()  # ensure the tables exist (idempotent)

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = 0
    conn = get_connection()
    try:
        for ax in (findings.get("axes") or []):
            if not isinstance(ax, dict):
                continue
            conn.execute(
                "INSERT INTO arbiter_findings "
                "(run_id, mode, axis, score, sample_size, trade_ids, category, "
                " observation_text, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                (
                    run_id, mode, ax.get("axis"), ax.get("score"),
                    ax.get("sample_size"),
                    json.dumps(ax.get("trade_ids") or []),
                    ax.get("observation"), ax.get("confidence"), created_at,
                ),
            )
            rows += 1
        for q in (findings.get("qualitative") or []):
            if not isinstance(q, dict):
                continue
            conn.execute(
                "INSERT INTO arbiter_findings "
                "(run_id, mode, axis, score, sample_size, trade_ids, category, "
                " observation_text, confidence, created_at) "
                "VALUES (?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?)",
                (
                    run_id, mode,
                    json.dumps(q.get("trade_ids") or []),
                    q.get("category"), q.get("observation"),
                    q.get("confidence"), created_at,
                ),
            )
            rows += 1
        conn.commit()
    finally:
        conn.close()
    return rows


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Kairos Arbiter — retrospective trade analysis")
    parser.add_argument("--mode", choices=["daily", "weekly"], required=True,
                        help="daily: trades closed today; weekly: all closed trades")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full analysis input; do not call Mistral or post to Slack")
    parser.add_argument("--date", metavar="YYYY-MM-DD",
                        help="Override the review date (daily mode window). For testing/backfill.")
    args = parser.parse_args()

    mode = args.mode
    # Use ET timezone so the daily review correctly covers the trading day
    # (Arbiter runs at 8PM ET = midnight UTC, which would otherwise roll to next day)
    try:
        from zoneinfo import ZoneInfo
        et_now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        from datetime import timedelta
        et_now = datetime.now(timezone.utc) - timedelta(hours=4)
    today = args.date or et_now.strftime("%Y-%m-%d")
    if args.date:
        print(f"  (review date overridden to {today})")

    print(f"  Kairos Arbiter — mode={mode}  date={today} (ET)")

    closed = fetch_closed_trades(mode, today)

    # Fallback: if daily mode finds 0 closed trades, check previous trading day
    # (catches cases where yesterday's Arbiter run failed or date rolled over)
    if mode == "daily" and len(closed) == 0:
        from datetime import timedelta
        prev_et = et_now - timedelta(days=1)
        # Walk back to find last weekday
        while prev_et.weekday() >= 5:
            prev_et -= timedelta(days=1)
        prev_date = prev_et.strftime("%Y-%m-%d")
        prev_report = os.path.join(REPORTS_DIR, f"{prev_date}_daily.json")
        # Only fall back if yesterday's report doesn't exist (meaning it failed)
        if not os.path.exists(prev_report):
            print(f"  No trades today and no {prev_date} report found — reviewing {prev_date} as fallback")
            closed = fetch_closed_trades(mode, prev_date)
            if closed:
                today = prev_date  # Use the fallback date for the report

    print("  Enriching closed trades with authoritative P&L + price paths...")
    closed = enrich_closed_trades(closed)

    open_pos = enrich_open_positions(fetch_open_positions())
    print("  Enriching overdue positions with peak metrics...")
    open_pos = enrich_overdue_positions(open_pos)

    stats = compute_aggregates(closed)
    user_prompt = build_user_prompt(mode, today, closed, open_pos, stats)

    print(f"  Closed trades in window: {len(closed)}  |  Open positions: {len(open_pos)}")

    # ── Dry run: print and exit ─────────────────────────────────────
    if args.dry_run:
        print("\n" + "=" * 72)
        print("  SYSTEM PROMPT")
        print("=" * 72)
        print(SYSTEM_PROMPT)
        print("\n" + "=" * 72)
        print("  USER PROMPT")
        print("=" * 72)
        print(user_prompt)
        print("=" * 72)
        print("  DRY RUN — no Mistral call, no Slack post, no report written.")
        return 0

    # ── Daily fast-path: nothing closed today ───────────────────────
    if mode == "daily" and not closed:
        msg = (
            f":balance_scale: *Kairos Arbiter — Daily Review*  |  {today}\n"
            f"No trades closed today. {len(open_pos)} open position(s) being monitored."
        )
        post_to_slack(ARBITER_CHANNEL, msg)
        write_report(mode, today, SYSTEM_PROMPT, user_prompt,
                     closed, open_pos, stats, mistral_response=None, error=None)
        print("  No closed trades today — posted brief note.")
        return 0

    # ── API key ─────────────────────────────────────────────────────
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        err = "MISTRAL_API_KEY not set in environment"
        print(f"  ERROR: {err}")
        write_report(mode, today, SYSTEM_PROMPT, user_prompt,
                     closed, open_pos, stats, mistral_response=None, error=err)
        post_to_slack(ALERTS_CHANNEL,
                      f":warning: *Kairos Arbiter failed* ({mode}, {today})\n{err}")
        return 1

    # ── Mistral call ────────────────────────────────────────────────
    try:
        print(f"  Calling Mistral ({MISTRAL_MODEL})...")
        response = call_mistral(SYSTEM_PROMPT, user_prompt, api_key)
    except Exception as exc:
        err = f"Mistral API call failed: {exc}"
        print(f"  ERROR: {err}")
        write_report(mode, today, SYSTEM_PROMPT, user_prompt,
                     closed, open_pos, stats, mistral_response=None, error=err)
        post_to_slack(ALERTS_CHANNEL,
                      f":warning: *Kairos Arbiter failed* ({mode}, {today})\n{err}")
        return 1

    # ── Capture structured findings (Phase A — does not affect trading) ─
    # Only on the real path, after a successful Mistral call. Failures here are
    # logged but never block the report or Slack post.
    run_id = f"{today}_{mode}"
    findings = parse_findings(response)
    if findings is not None:
        try:
            n_findings = persist_findings(run_id, mode, findings)
            print(f"  Arbiter findings captured: {n_findings} row(s) "
                  f"(run_id={run_id})")
        except Exception as exc:
            print(f"  WARNING: persist_findings failed: {exc}", file=sys.stderr)
    else:
        print("  No structured findings captured (see warning above).")

    # ── Persist + post ──────────────────────────────────────────────
    report_path = write_report(mode, today, SYSTEM_PROMPT, user_prompt,
                               closed, open_pos, stats,
                               mistral_response=response, error=None)
    print(f"  Report written: {report_path}")

    slack_text = format_slack_report(mode, today, response, len(closed), len(open_pos))
    posted = post_to_slack(ARBITER_CHANNEL, slack_text)
    print(f"  Slack post to {ARBITER_CHANNEL}: {'OK' if posted else 'FAILED'}")

    # ── Weekly only: auto-propose fresh axis weights ────────────────
    # After the weekly analysis post, refresh the axis-weight proposals so the
    # learning loop keeps pace with closing trades. This writes 'proposed' rows
    # and posts a combined Slack summary ONLY — it never approves or changes a
    # live weight (the human gate is untouched). Fully guarded: a propose failure
    # must NEVER affect the Arbiter run. The daily path does not touch this.
    if mode == "weekly":
        try:
            from kairos_axis_weights import propose_all
            ap = propose_all()
            print(f"  Auto-proposed axis weights: {len(ap['proposals'])} proposal(s), "
                  f"{len(ap['errors'])} error(s) (run_id={ap['run_id']})")
        except Exception as exc:
            print(f"  WARNING: weekly axis-weight auto-propose failed: {exc}",
                  file=sys.stderr)

        # Exit-engine PARAM proposals (param:<dotted.config.path>), same human
        # gate. Writes 'proposed' rows + one combined Slack note only — never
        # approves or edits kairos_config.json. Separately guarded so a param
        # failure cannot affect the weight proposals or the Arbiter run.
        try:
            from kairos_axis_weights import propose_all_params
            pp = propose_all_params()
            print(f"  Auto-proposed exit params: {len(pp['proposals'])} proposal(s), "
                  f"{len(pp['errors'])} error(s) (run_id={pp['run_id']})")
        except Exception as exc:
            print(f"  WARNING: weekly exit-param auto-propose failed: {exc}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
