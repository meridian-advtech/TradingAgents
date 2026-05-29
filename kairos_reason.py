"""
Kairos Reasoning Engine — Milestone 4

Gathers all market data and portfolio state, then writes a structured
prompt to kairos_prompt.txt for Claude Code (the reasoning engine) to
read, reason over, and produce a trading decision.

Steps performed by this script:
  1. Run kairos_snapshot to collect data from all five sources
  2. Connect to IBKR paper account (port 7497) and pull portfolio
  3. Write everything into kairos_prompt.txt as a reasoning prompt

Step 4 (reasoning + decision logging) is performed by Claude Code itself.
"""

import json
import os
import sys
from datetime import datetime, timezone

from ib_insync import IB, Stock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_snapshot import run_snapshot
from kairos_log_db import init_db, get_connection, get_tax_context
from kairos_tax_efficiency import format_tax_efficiency_section

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(SCRIPT_DIR, "kairos_prompt.txt")
BIAS_HISTORY_FILE = os.path.join(SCRIPT_DIR, "kairos_bias_history.json")
TICKER = "AAPL"
W = 72
BIAS_REPEAT_THRESHOLD = 2  # flag if same ticker chosen N+ cycles in a row


def _get_position_rationale(ticker: str) -> str | None:
    """Fetch the entry rationale from the most recent filled BUY for a ticker."""
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT rationale FROM decisions "
            "WHERE ticker = ? AND action = 'BUY' AND execution_status = 'Filled' "
            "ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        conn.close()
        return row["rationale"] if row and row["rationale"] else None
    except Exception:
        return None


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Step 1: Market snapshot ──────────────────────────────────────────

def gather_market_data() -> str:
    print(banner("Step 1 — Gathering Market Snapshot (5 sources)"))
    report, _ = run_snapshot(print_report=False, save_file=False)
    print("  Done.")
    return report


# ── Step 2: IBKR portfolio ──────────────────────────────────────────

def gather_portfolio(shortlist: list[str] | None = None) -> dict:
    print(banner("Step 2 — Connecting to IBKR (port 7497)"))
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=2, timeout=10)

    # Account summary
    summary_raw = ib.accountSummary()
    keep = {"NetLiquidation", "TotalCashValue", "GrossPositionValue",
            "BuyingPower", "AvailableFunds", "UnrealizedPnL", "RealizedPnL"}
    account = {v.tag: v.value for v in summary_raw if v.tag in keep}

    # Positions
    positions = []
    for p in ib.positions():
        positions.append({
            "symbol": p.contract.symbol,
            "quantity": float(p.position),
            "avg_cost": round(p.avgCost, 2),
        })

    # Fetch prices for all shortlisted tickers (not just AAPL)
    tickers_to_price = list(shortlist or [TICKER])
    # Also include any held positions not already in the list
    held_symbols = {p["symbol"] for p in positions}
    for sym in held_symbols:
        if sym not in tickers_to_price:
            tickers_to_price.append(sym)

    ib.reqMarketDataType(4)
    ticker_prices: dict[str, float] = {}
    print(f"  Fetching prices for {len(tickers_to_price)} ticker(s)...")

    for sym in tickers_to_price:
        try:
            contract = Stock(sym, "SMART", "USD")
            ib.qualifyContracts(contract)
            mkt = ib.reqMktData(contract)
            ib.sleep(2)
            price = None
            for attr in ("last", "close", "bid", "ask"):
                val = getattr(mkt, attr, None)
                if val is not None and val == val:
                    price = round(val, 2)
                    break
            ib.cancelMktData(contract)
            if price:
                ticker_prices[sym] = price
        except Exception:
            pass

    ib.disconnect()

    portfolio = {
        "account": account,
        "positions": positions,
        "ticker_prices": ticker_prices,
        # Backwards compat
        "aapl_last_price": ticker_prices.get(TICKER),
    }

    print(f"  Net Liquidation: ${float(account.get('NetLiquidation', 0)):,.2f}")
    print(f"  Cash:            ${float(account.get('TotalCashValue', 0)):,.2f}")
    print(f"  Buying Power:    ${float(account.get('BuyingPower', 0)):,.2f}")
    if ticker_prices:
        print(f"  Prices fetched:")
        for sym, price in ticker_prices.items():
            held = next((p for p in positions if p["symbol"] == sym), None)
            if held:
                print(f"    {sym}: ${price} (held: {held['quantity']} @ ${held['avg_cost']})")
            else:
                print(f"    {sym}: ${price}")
    for pos in positions:
        if pos["symbol"] not in ticker_prices:
            print(f"  Position: {pos['symbol']} — {pos['quantity']} shares @ ${pos['avg_cost']} (no price)")

    return portfolio


# ── Step 3: Tax context ──────────────────────────────────────────────

def gather_tax_context() -> dict:
    """Pull tax lot data from kairos.db for the reasoning prompt."""
    print(banner("Step 3 — Gathering Tax Context"))
    init_db()  # ensure tables exist
    ctx = get_tax_context(TICKER)

    if not ctx["has_position"]:
        print(f"  No open {TICKER} holdings in DB.")
    else:
        print(f"  Open lots: {len(ctx['lots'])}, Total shares: {ctx['total_quantity']}")
        for lot in ctx["lots"]:
            print(f"    {lot['quantity']} shares from {lot['entry_date'][:10]} "
                  f"({lot['holding_days']}d, {lot['tax_rate']})")
        if ctx.get("any_short_term"):
            print(f"  WARNING: Some lots are short-term (taxed as ordinary income)")

    if ctx["wash_sale_risk"]:
        print(f"  WASH SALE ALERT: {TICKER} sold within last 30 days")
        for s in ctx["recent_sells"]:
            print(f"    Sold {s['quantity']} shares on {s['sold_date'][:10]}")

    return ctx


def gather_tax_efficiency(portfolio: dict,
                          shortlist: list[str] | None = None) -> str:
    """Step 3b — Build tax efficiency analysis for shortlisted positions.

    Only positions whose ticker appears in the shortlist are included,
    because those are the only tickers the model might recommend selling.
    Non-shortlisted positions are irrelevant to the current reasoning
    cycle and were inflating the prompt by ~20 KB.
    """
    print(banner("Step 3b — Tax Efficiency Analysis"))
    positions = portfolio.get("positions", [])
    ticker_prices = portfolio.get("ticker_prices", {})
    shortlist_set = set(shortlist) if shortlist else None

    tickers_prices: dict[str, float] = {}
    skipped = 0
    for pos in positions:
        sym = pos["symbol"]
        if shortlist_set and sym not in shortlist_set:
            skipped += 1
            continue
        price = ticker_prices.get(sym) or pos.get("avg_cost", 0)
        tickers_prices[sym] = price

    if not tickers_prices:
        print("  No shortlisted positions to analyze.")
        return ""

    section = format_tax_efficiency_section(tickers_prices)
    hold_count = section.count("HOLD-FOR-TAX")
    sell_count = section.count("SELL-OK")
    print(f"  Analyzed {len(tickers_prices)} shortlisted position(s): "
          f"{hold_count} HOLD-FOR-TAX, {sell_count} SELL-OK"
          + (f" ({skipped} non-shortlisted omitted)" if skipped else ""))
    return section


def format_tax_section(tax_ctx: dict) -> str:
    """Format the tax context into a prompt section string."""
    lines = []

    if not tax_ctx["has_position"]:
        lines.append("No open holdings tracked. A new BUY creates a fresh tax lot.")
        if tax_ctx["wash_sale_risk"]:
            lines.append("")
            lines.append("** WASH SALE WARNING **")
            lines.append(f"  {TICKER} was sold within the last 30 days.")
            lines.append("  Repurchasing now would trigger wash-sale rules: the loss")
            lines.append("  on the prior sale would be DISALLOWED and added to the")
            lines.append("  cost basis of the new shares.")
            for s in tax_ctx["recent_sells"]:
                lines.append(f"  - Sold {s['quantity']} shares on {s['sold_date'][:10]} "
                             f"@ ${s['sold_price']:.2f} (bought @ ${s['entry_price']:.2f})")
        return "\n".join(lines)

    lines.append(f"Open {TICKER} lots (FIFO order):")
    lines.append(f"  {'Lot':>3}  {'Entry Date':<12} {'Shares':>7} {'Entry $':>9} "
                 f"{'Days Held':>10} {'Tax Rate':<12} {'Days to LT':>11}")
    lines.append(f"  {'─'*3}  {'─'*12} {'─'*7} {'─'*9} {'─'*10} {'─'*12} {'─'*11}")

    for i, lot in enumerate(tax_ctx["lots"], 1):
        dtl = f"{lot['days_to_long_term']}d" if lot["tax_rate"] == "short-term" else "—"
        lines.append(
            f"  {i:>3}  {lot['entry_date'][:10]:<12} {lot['quantity']:>7.0f} "
            f"${lot['entry_price']:>8.2f} {lot['holding_days']:>9}d "
            f"{lot['tax_rate']:<12} {dtl:>11}"
        )

    lines.append(f"\n  Total: {tax_ctx['total_quantity']:.0f} shares")

    if tax_ctx.get("any_short_term"):
        short_lots = [l for l in tax_ctx["lots"] if l["tax_rate"] == "short-term"]
        short_qty = sum(l["quantity"] for l in short_lots)
        min_days_to_lt = min(l["days_to_long_term"] for l in short_lots)
        lines.append(f"\n  TAX IMPACT IF SELLING NOW:")
        lines.append(f"    {short_qty:.0f} shares would be taxed at SHORT-TERM rates")
        from kairos_tax_efficiency import load_tax_config
        tcfg = load_tax_config()
        lines.append(f"    (ordinary income at {tcfg['short_term_rate']*100:.0f}% federal rate)")
        lines.append(f"    Earliest lot reaches long-term status in {min_days_to_lt} days")
        lines.append(f"    Consider waiting if no urgent sell signal.")

    if tax_ctx["wash_sale_risk"]:
        lines.append(f"\n  ** WASH SALE WARNING **")
        lines.append(f"  {TICKER} was sold within the last 30 days:")
        for s in tax_ctx["recent_sells"]:
            lines.append(f"    - Sold {s['quantity']} shares on {s['sold_date'][:10]} "
                         f"@ ${s['sold_price']:.2f} (bought @ ${s['entry_price']:.2f})")
        lines.append(f"  A BUY now would trigger IRS wash-sale rules (IRC 1091).")
        lines.append(f"  The loss from the recent sale would be DISALLOWED and added")
        lines.append(f"  to the cost basis of newly purchased shares.")

    return "\n".join(lines)


# ── Step 4: Trading Ledger ───────────────────────────────────────────
#
# kairos_ledger.txt is a compact, append-only outcome log.  Each closed
# trade gets one pipe-delimited line.  The file also carries an auto-
# updated pattern summary header that the prompt reads directly.
#
# Format per line:
#   date | ticker | action | signals | pnl% | PASS/FAIL
#
# The header block (between the --- markers) is rewritten on every
# call to refresh_ledger_header().

LEDGER_FILE = os.path.join(SCRIPT_DIR, "kairos_ledger.txt")
HEADER_START = "--- PATTERN SUMMARY ---"
HEADER_END = "--- TRADE LOG ---"


def classify_signals(data_inputs: dict | None) -> dict:
    """Extract simplified signal tags from a decision's data_inputs JSON.

    Returns e.g. {"news": "bullish", "macro": "easing", "crypto": "risk-on",
                   "legislative": "neutral"}
    """
    if not data_inputs:
        return {}

    chain = data_inputs if isinstance(data_inputs, dict) else {}
    if "reasoning_chain" in chain:
        chain = chain["reasoning_chain"]

    def _pick(text: str, pos_words: tuple, neg_words: tuple,
              pos_label: str, neg_label: str, neutral: str) -> str:
        t = text.lower()
        if any(w in t for w in pos_words):
            return pos_label
        if any(w in t for w in neg_words):
            return neg_label
        return neutral

    return {
        "news": _pick(chain.get("news_sentiment", ""),
                       ("bullish", "positive"), ("bearish", "negative"),
                       "bullish", "bearish", "neutral"),
        "macro": _pick(chain.get("macro_environment", ""),
                        ("easing", "accommodative", "supportive", "lower", "cutting"),
                        ("tightening", "hawkish", "rising", "hiking"),
                        "easing", "tightening", "stable"),
        "crypto": _pick(chain.get("risk_appetite", ""),
                         ("positive", "risk-on", "up", "green", "bullish"),
                         ("negative", "risk-off", "down", "red", "bearish"),
                         "risk-on", "risk-off", "mixed"),
        "legislative": _pick(chain.get("legislative_risk", ""),
                              ("threat", "risk", "negative", "targeting"), (),
                              "risky", "neutral", "neutral"),
    }


def signals_to_tags(signals: dict) -> str:
    """Compact tag string: 'news=bullish macro=easing crypto=risk-on legis=neutral'"""
    short = {"legislative": "legis"}
    return " ".join(f"{short.get(k,k)}={v}" for k, v in signals.items() if v)


def record_ledger_entry(
    date: str,
    ticker: str,
    action: str,
    signals: dict,
    pnl_pct: float,
    reversion: bool = False,
    buy_signals: list[str] | None = None,
) -> None:
    """Append one closed-trade line to kairos_ledger.txt, then refresh header.

    Args:
        buy_signals: list of signal tags like ["HOT-EARNINGS", "HOT-RSI"]
                     that triggered this trade (from kairos_signals.py)
    """
    verdict = "PASS" if pnl_pct > 0 else "FAIL"
    tags = signals_to_tags(signals)
    if reversion:
        tags += " REVERSION"
    if buy_signals:
        for sig in buy_signals:
            if sig not in tags:
                tags += f" {sig}"
    # Add timestamp to distinguish multiple lots of same ticker on same day
    import datetime
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    line = f"{date} | {ticker:<5} | {action:<4} | {tags} | {pnl_pct:+.2f}% | {verdict} | {timestamp}"

    # Read existing lines (skip header block)
    trade_lines = _read_trade_lines()
    trade_lines.append(line)

    # Rebuild the whole file: header + trades
    _write_ledger(trade_lines)


def _read_trade_lines() -> list[str]:
    """Return just the trade lines from the ledger (no header)."""
    if not os.path.exists(LEDGER_FILE):
        return []
    lines = []
    in_trades = False
    with open(LEDGER_FILE, "r") as f:
        for raw in f:
            stripped = raw.rstrip("\n")
            if stripped == HEADER_END:
                in_trades = True
                continue
            if in_trades and stripped:
                lines.append(stripped)
    # If the file has no header markers, treat everything as trade lines
    if not in_trades:
        with open(LEDGER_FILE, "r") as f:
            for raw in f:
                stripped = raw.rstrip("\n")
                if stripped and not stripped.startswith("---"):
                    lines.append(stripped)
    return lines


def _build_pattern_summary(trade_lines: list[str]) -> str:
    """Analyze trade lines and produce a compact pattern summary."""
    from collections import defaultdict

    combo_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0})
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0

    for line in trade_lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        tags_str = parts[3]
        try:
            pnl_pct = float(parts[4].replace("%", ""))
        except ValueError:
            continue
        verdict = parts[5].strip()
        # Handle both old format (6 parts) and new format (7 parts with timestamp)
        if len(parts) >= 7:
            # New format has timestamp in parts[6], ignore it for pattern analysis
            pass

        if verdict == "PASS":
            total_wins += 1
            combo_stats[tags_str]["wins"] += 1
        else:
            total_losses += 1
            combo_stats[tags_str]["losses"] += 1
        combo_stats[tags_str]["total_pnl"] += pnl_pct
        total_pnl += pnl_pct

    total = total_wins + total_losses
    if total == 0:
        return "  No closed trades yet."

    lines = []
    win_rate = total_wins / total * 100
    lines.append(f"  Closed trades: {total}  Win rate: {win_rate:.0f}%  "
                 f"Cumulative P&L: {total_pnl:+.1f}%")
    lines.append("")

    # Sort combos: worst first (most actionable)
    ranked = sorted(combo_stats.items(),
                    key=lambda kv: kv[1]["total_pnl"])

    # Show losing combos
    losers = [(k, v) for k, v in ranked if v["losses"] >= 2]
    winners = [(k, v) for k, v in ranked if v["wins"] >= 2 and v["losses"] == 0]

    if losers:
        lines.append(f"  LOSING PATTERNS ({len(losers)} occurrences):")
        for tags, stats in losers:
            n = stats["wins"] + stats["losses"]
            lines.append(f"    [{tags}]  {stats['wins']}W / {stats['losses']}L / {n} total  "
                         f"cumulative {stats['total_pnl']:+.1f}%  ← AVOID")
        lines.append("")

    if winners:
        lines.append(f"  WINNING PATTERNS ({len(winners)} occurrences):")
        for tags, stats in winners:
            n = stats["wins"] + stats["losses"]
            lines.append(f"    [{tags}]  {stats['wins']}W / {stats['losses']}L / {n} total  "
                         f"cumulative {stats['total_pnl']:+.1f}%  ← REPEAT")
        lines.append("")

    if not losers and not winners:
        lines.append("  No dominant patterns yet (need 2+ occurrences).")

    # Per-signal-type stats
    signal_types = ["REVERSION", "HOT-EARNINGS", "HOT-RSI",
                    "HOT-INSIDER", "HOT-CONGRESS"]
    signal_stats = []

    for sig in signal_types:
        wins = losses = 0
        total_pnl = 0.0
        for line in trade_lines:
            if sig not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                continue
            try:
                pnl = float(parts[4].replace("%", ""))
            except ValueError:
                continue
            if parts[5].strip() == "PASS":
                wins += 1
            else:
                losses += 1
            total_pnl += pnl

        total = wins + losses
        if total > 0:
            wr = wins / total * 100
            signal_stats.append(
                f"  {sig + ':':18s}{total} trades, {wins}W/{losses}L "
                f"({wr:.0f}% WR), cumulative {total_pnl:+.1f}%")

    if signal_stats:
        lines.append("")
        lines.append("  SIGNAL PERFORMANCE:")
        lines.extend(signal_stats)

    return "\n".join(lines)


def _write_ledger(trade_lines: list[str]) -> None:
    """Write the full ledger file: header + trade lines."""
    summary = _build_pattern_summary(trade_lines)
    with open(LEDGER_FILE, "w") as f:
        f.write(HEADER_START + "\n")
        f.write(summary + "\n")
        f.write(HEADER_END + "\n")
        for line in trade_lines:
            f.write(line + "\n")


def refresh_ledger_header() -> None:
    """Re-read trades and rewrite the pattern summary header."""
    trade_lines = _read_trade_lines()
    _write_ledger(trade_lines)


MAX_LEDGER_LINES = 10  # keep prompt compact; pattern summary aggregates the full history


def gather_ledger() -> str:
    """Step 4 — Load the compact ledger for the reasoning prompt.

    Returns the pattern summary header (computed from ALL trades) plus
    at most the last MAX_LEDGER_LINES trade lines. This keeps the
    prompt bounded while still exposing the full statistical picture.
    """
    print(banner("Step 4 — Loading Trading Ledger"))
    init_db()

    if not os.path.exists(LEDGER_FILE):
        _write_ledger([])
        print("  No ledger found — created empty kairos_ledger.txt")
        return ""

    refresh_ledger_header()

    all_trades = _read_trade_lines()
    capped = all_trades[-MAX_LEDGER_LINES:] if len(all_trades) > MAX_LEDGER_LINES else all_trades

    # Rebuild content: full-history summary header + capped trade lines
    summary = _build_pattern_summary(all_trades)
    parts = [HEADER_START, summary, HEADER_END]
    parts.extend(capped)
    if len(all_trades) > MAX_LEDGER_LINES:
        parts.append(f"\n  (showing last {MAX_LEDGER_LINES} of {len(all_trades)} trades)")

    print(f"  Loaded {len(all_trades)} closed trade(s), "
          f"showing last {len(capped)} in prompt")

    return "\n".join(parts)


# ── Holding-bias detection + hard blacklist ──────────────────────────

def _load_bias_history() -> list[str]:
    """Load the list of recently chosen tickers (most recent last)."""
    if not os.path.exists(BIAS_HISTORY_FILE):
        return []
    try:
        with open(BIAS_HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def record_chosen_ticker(ticker: str) -> None:
    """Append a ticker to the bias history (called after each decision)."""
    history = _load_bias_history()
    history.append(ticker)
    # Keep last 10
    history = history[-10:]
    with open(BIAS_HISTORY_FILE, "w") as f:
        json.dump(history, f)


def get_blacklisted_tickers(held_symbols: set[str]) -> list[tuple[str, int]]:
    """Return list of (ticker, consecutive_count) for tickers that should
    be blacklisted this cycle.

    A ticker is blacklisted if:
      - It is an existing holding AND
      - It has been chosen BIAS_REPEAT_THRESHOLD+ consecutive times
    """
    history = _load_bias_history()
    if len(history) < BIAS_REPEAT_THRESHOLD:
        return []

    blacklisted = []
    # Count consecutive picks from the end of history
    if history:
        last = history[-1]
        count = 0
        for t in reversed(history):
            if t == last:
                count += 1
            else:
                break
        if count >= BIAS_REPEAT_THRESHOLD and last in held_symbols:
            blacklisted.append((last, count))

    return blacklisted


def build_blacklist_block(blacklisted: list[tuple[str, int]]) -> str:
    """Build the hard blacklist instruction for the prompt header.

    Returns empty string if nothing is blacklisted.
    """
    if not blacklisted:
        return ""

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║  MANDATORY BLACKLIST — DO NOT CHOOSE THESE TICKERS         ║",
        "╠══════════════════════════════════════════════════════════════╣",
    ]
    for ticker, count in blacklisted:
        lines.append(
            f"║  {ticker:<6} — BLACKLISTED (chosen {count} consecutive times)      ║"
        )
    lines.append("║                                                              ║")
    lines.append("║  These tickers MUST NOT be selected as the winner under      ║")
    lines.append("║  any circumstances this cycle. Choose from the remaining     ║")
    lines.append("║  candidates ONLY. If you select a blacklisted ticker,        ║")
    lines.append("║  the decision will be rejected and re-run.                   ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")

    return "\n".join(lines)


# ── Step 5: Write reasoning prompt ──────────────────────────────────

def write_prompt(market_data: str, portfolio: dict, tax_ctx: dict | None = None,
                 ledger_text: str | None = None,
                 shortlist: list[str] | None = None,
                 tax_efficiency_text: str | None = None,
                 regime_section: str | None = None) -> str:
    print(banner("Step 6 — Writing Reasoning Prompt"))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    tax_section = ""
    if tax_ctx:
        tax_text = format_tax_section(tax_ctx)
        tax_section = f"""
{'=' * W}
SECTION 3: TAX CONTEXT
{'=' * W}

{tax_text}
"""

    # Tax efficiency analysis (detailed $ impact)
    tax_eff_section = ""
    if tax_efficiency_text:
        tax_eff_section = f"""
{'=' * W}
SECTION 3b: TAX EFFICIENCY ANALYSIS
{'=' * W}

{tax_efficiency_text}
"""

    ledger_section = ""
    if ledger_text:
        ledger_section = f"""
{'=' * W}
SECTION 4: TRADING LESSONS (kairos_ledger.txt)
{'=' * W}

{ledger_text}
"""
    else:
        ledger_section = f"""
{'=' * W}
SECTION 4: TRADING LESSONS
{'=' * W}

No closed trades yet. The ledger will populate as positions are closed.
"""

    # Build screening section if shortlist provided
    screen_section = ""
    if shortlist:
        # Load screening result + signal summary for per-ticker context
        screen_result_file = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")
        signal_summary_file = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")

        screen_result = {}
        if os.path.exists(screen_result_file):
            try:
                with open(screen_result_file) as f:
                    screen_result = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

        signal_summary = {}
        if os.path.exists(signal_summary_file):
            try:
                with open(signal_summary_file) as f:
                    signal_summary = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

        signal_tags = signal_summary.get("signal_tags", {})
        earnings_hits = signal_summary.get("earnings_hits", {})
        rsi_hits = signal_summary.get("rsi_hits", {})
        insider_hits = signal_summary.get("insider_hits", {})
        congress_hits = signal_summary.get("congress_hits", {})
        options_hits = signal_summary.get("options_hits", {})
        kalshi_shifts = signal_summary.get("kalshi_shifts", [])
        reversion_tickers = screen_result.get("hot_reversion", [])

        # Also load standalone options activity file if not in summary
        if not options_hits:
            options_activity_file = os.path.join(SCRIPT_DIR, "kairos_options_activity.json")
            if os.path.exists(options_activity_file):
                try:
                    with open(options_activity_file) as f:
                        options_hits = json.load(f).get("hits", {})
                except (json.JSONDecodeError, IOError):
                    pass

        # HOT-CATALYST long-option setups (surfaced for awareness/dedup only —
        # the options engine self-selects and executes these in its own phase).
        catalyst_hits = {}
        catalyst_file = os.path.join(SCRIPT_DIR, "kairos_catalyst_signals.json")
        if os.path.exists(catalyst_file):
            try:
                with open(catalyst_file) as f:
                    catalyst_hits = json.load(f).get("by_ticker", {})
            except (json.JSONDecodeError, IOError):
                pass

        # Build per-ticker signal detail lines
        ticker_lines = []
        for t in shortlist:
            tags = signal_tags.get(t, [])
            # Also check if it's in reversion list
            if t in reversion_tickers and "HOT-REVERSION" not in tags:
                tags = ["HOT-REVERSION"] + tags
            tag_str = ", ".join(tags) if tags else "HOT/WARM (momentum)"
            ticker_lines.append(f"  {t:<6} [{tag_str}]")

            # Add signal-specific detail
            if t in earnings_hits:
                e = earnings_hits[t]
                ticker_lines.append(
                    f"         Earnings: beat by {e.get('surprise_pct', '?')}% "
                    f"on {e.get('date', '?')} "
                    f"(actual={e.get('eps_actual', '?')} vs est={e.get('eps_estimate', '?')})")
            if t in rsi_hits:
                ticker_lines.append(f"         RSI: {rsi_hits[t]} (oversold, below 30)")
            if t in insider_hits:
                for f in insider_hits[t][:2]:
                    ticker_lines.append(f"         Insider buy: {f.get('filer', '?')} on {f.get('date', '?')}")
            if t in congress_hits:
                for f in congress_hits[t][:2]:
                    ticker_lines.append(f"         Congress buy: {f.get('member', '?')} on {f.get('date', '?')}")
            if t in reversion_tickers:
                ticker_lines.append(
                    f"         REVERSION: sharp drop >3% — evaluate fundamental vs. sentiment")
            if t in options_hits:
                o = options_hits[t]
                triggers = ", ".join(o.get("triggered_by", []))
                ticker_lines.append(
                    f"         OPTIONS: vol/OI {o.get('vol_oi_ratio', '?')}x, "
                    f"IV rank {o.get('iv_rank', 0):.0%}, "
                    f"C/P {o.get('call_put_ratio', '?')}x "
                    f"({o.get('direction', '?')}) [{triggers}]")
            if t in catalyst_hits:
                for c in catalyst_hits[t]:
                    rt = "calls" if c.get("right") == "C" else "puts"
                    ivr = c.get("iv_rank")
                    ivr_str = f", IVrank {ivr:.0%}" if isinstance(ivr, (int, float)) else ""
                    ticker_lines.append(
                        f"         CATALYST: {c.get('setup', '?')} "
                        f"({c.get('direction', '?')}→{rt}{ivr_str}) — "
                        f"options engine handles this separately")

        ticker_detail = "\n".join(ticker_lines)

        # Kalshi macro context
        kalshi_note = ""
        if kalshi_shifts:
            shift_lines = []
            for s in kalshi_shifts:
                shift_lines.append(
                    f"  {s['title'][:60]}: {s['old_prob']:.0f}% → {s['new_prob']:.0f}% "
                    f"({s['direction']} {abs(s['shift']):.0f}pp)")
            kalshi_note = "\nKalshi Probability Shifts (macro context):\n" + "\n".join(shift_lines) + "\n"

        screen_section = f"""
{'=' * W}
SECTION 5: TIER 1 SCREENING + SIGNAL DETAIL
{'=' * W}

{len(shortlist)} tickers passed Tier 1 screening. Signal tags show WHY each
was flagged — use these to prioritize and calibrate confidence:

{ticker_detail}
{kalshi_note}
Signal legend:
  HOT-EARNINGS   = beat earnings estimates by >5% in last 30 days
  HOT-RSI        = 14-period RSI below 30 (oversold)
  HOT-INSIDER    = SEC Form 4 insider purchase in last 7 days
  HOT-CONGRESS   = congressional stock purchase in last 30 days
  HOT-REVERSION  = dropped >3% — needs fundamental vs. sentiment check
  HOT-KALSHI     = Kalshi prediction market shifted >10%
  HOT-OPTIONS    = unusual options activity (vol/OI spike, high IV rank, or skewed call/put)
  HOT-CATALYST   = long-option setup (pre-catalyst vol / post-crush reversion / unusual flow);
                   the options engine selects + executes the contract in its own phase —
                   do NOT size an equity position on this tag, it is for awareness only
  HOT/WARM       = price momentum + volume (Ollama screened)

CONFLUENCE SCORING (position sizing is automatic — do NOT set quantities):
  Multiple signals = higher confluence = larger position as % of NLV.
  Point weights: EARNINGS=2, INSIDER=2, OPTIONS=2, RSI=1, CONGRESS=1,
                 REVERSION=1, KALSHI=1.
    Score 1     → 1.0% of NLV  (~$10K at $1M)
    Score 2–3   → 1.5% of NLV  (~$15K)
    Score 4–5   → 2.0% of NLV  (~$20K)
    Score 6+    → 2.5% of NLV  (~$25K)

  Portfolio guardrails (enforced automatically before each order):
    • Cash reserve:  10% of NLV must remain as cash
    • Sector max:    25% of NLV in any single sector
    • Position max:  10% of NLV in any single stock

  You may recommend ALL tickers that merit a trade — quantity and guardrail
  enforcement happen downstream. Focus on conviction, not sizing.
"""
    # ── Build portfolio sections: separate held vs candidates ──────
    held_positions = portfolio.get("positions", [])
    held_symbols = {p["symbol"] for p in held_positions}
    ticker_prices = portfolio.get("ticker_prices", {})
    shortlist_set = set(shortlist or [])

    # Existing positions block — full detail for shortlisted holdings
    # (these are the ones the model might act on), one-line summary for rest.
    if held_positions:
        shortlisted_held = [p for p in held_positions if p["symbol"] in shortlist_set]
        other_held = [p for p in held_positions if p["symbol"] not in shortlist_set]

        pos_lines = []
        for p in shortlisted_held:
            sym = p["symbol"]
            price = ticker_prices.get(sym)
            price_str = f"  current: ${price:.2f}" if price else ""
            pos_lines.append(f"  {sym:<6} {p['quantity']:.0f} shares @ ${p['avg_cost']:.2f}{price_str}")
            rationale = _get_position_rationale(sym)
            if rationale:
                pos_lines.append(f"         Entry thesis: {rationale[:150]}")
        if other_held:
            pos_lines.append(
                f"\n  + {len(other_held)} other position(s) not on today's shortlist "
                f"(omitted for brevity)")
        positions_block = "\n".join(pos_lines) if pos_lines else "  (no open positions)"
    else:
        positions_block = "  (no open positions)"

    # Candidate tickers block
    candidate_tickers = shortlist or [TICKER]
    candidates_on_shortlist = [t for t in candidate_tickers if t not in held_symbols]
    held_on_shortlist = [t for t in candidate_tickers if t in held_symbols]

    cand_lines = []
    for t in candidate_tickers:
        price = ticker_prices.get(t)
        price_str = f"${price:.2f}" if price else "N/A"
        held_tag = " (EXISTING HOLDING)" if t in held_symbols else ""
        cand_lines.append(f"  {t:<6} {price_str:>10}{held_tag}")
    candidates_block = "\n".join(cand_lines)

    # Account summary (compact)
    acct = portfolio.get("account", {})
    acct_block = (
        f"  Net Liquidation: ${float(acct.get('NetLiquidation', 0)):>14,.2f}\n"
        f"  Cash Available:  ${float(acct.get('TotalCashValue', 0)):>14,.2f}\n"
        f"  Buying Power:    ${float(acct.get('BuyingPower', 0)):>14,.2f}"
    )

    # Hard blacklist for repeat-bias tickers
    blacklisted = get_blacklisted_tickers(held_symbols)
    blacklist_block = build_blacklist_block(blacklisted)
    blacklisted_symbols = {t for t, _ in blacklisted}

    # Mark blacklisted tickers in the candidates list
    if blacklisted_symbols:
        new_cand_lines = []
        for t in candidate_tickers:
            price = ticker_prices.get(t)
            price_str = f"${price:.2f}" if price else "N/A"
            if t in blacklisted_symbols:
                new_cand_lines.append(f"  {t:<6} {price_str:>10} *** BLACKLISTED — DO NOT CHOOSE ***")
            elif t in held_symbols:
                new_cand_lines.append(f"  {t:<6} {price_str:>10} (EXISTING HOLDING)")
            else:
                new_cand_lines.append(f"  {t:<6} {price_str:>10}")
        candidates_block = "\n".join(new_cand_lines)

        # Filter blacklisted from the tickers_evaluated list in the output
        eligible_tickers = [t for t in candidate_tickers if t not in blacklisted_symbols]
        print(f"  BLACKLIST active: {', '.join(blacklisted_symbols)} excluded")
        print(f"  Eligible candidates: {len(eligible_tickers)}")
    else:
        eligible_tickers = candidate_tickers

    regime_block = ""
    if regime_section:
        regime_block = regime_section + "\n"

    # Section 0b: After-tax NPV context for held positions approaching LT
    # Only positions held 180-364 days — the window where the tax decision
    # is most relevant.  Too early and there's no urgency; past 365 and
    # it's already long-term.
    tax_npv_block = ""
    try:
        from kairos_tax_efficiency import load_tax_config, calculate_aftertax_npv
        from kairos_log_db import get_open_holdings
        tcfg = load_tax_config()
        lt_days = tcfg["long_term_days"]
        shortlist_set = set(shortlist) if shortlist else set()
        positions = portfolio.get("positions", [])
        ticker_prices = portfolio.get("ticker_prices", {})
        npv_lines = []

        for pos in positions:
            sym = pos["symbol"]
            if shortlist_set and sym not in shortlist_set:
                continue
            price = ticker_prices.get(sym) or pos.get("avg_cost", 0)
            if price <= 0:
                continue
            lots = get_open_holdings(sym)
            if not lots:
                continue
            # Aggregate lots per ticker: use weighted avg entry and total qty
            total_qty = sum(l["quantity"] for l in lots)
            total_cost = sum(l["entry_price"] * l["quantity"] for l in lots)
            if total_qty <= 0:
                continue
            avg_entry = total_cost / total_qty
            # Use the earliest lot's holding days for the ticker
            min_hd = min((l.get("holding_days") or 0) for l in lots)
            # Only show the 180-364 day window
            if min_hd < 180 or min_hd >= lt_days:
                continue
            npv = calculate_aftertax_npv(avg_entry, price, total_qty, min_hd, tcfg)
            if npv["tax_savings"] <= 0:
                continue
            rec = "HOLD-FOR-TAX" if npv["hold_recommended"] else "SELL-OK"
            npv_lines.append(
                f"  {sym}: held {min_hd} days | {npv['days_to_lt']} days to LT threshold | "
                f"Tax savings from waiting: ${npv['tax_savings']:,.0f} | "
                f"Opp cost of waiting: ${npv['opportunity_cost']:,.0f} | "
                f"Recommendation: {rec}")

        if npv_lines:
            header = (
                f"{'=' * W}\n"
                f"SECTION 0b: TAX CONTEXT\n"
                f"{'=' * W}\n\n"
                f"TAX CONTEXT — positions approaching long-term threshold:\n\n"
            )
            tax_npv_block = header + "\n".join(npv_lines) + "\n\n"
            print(f"  Section 0b: {len(npv_lines)} position(s) in 180-364d tax window")
    except Exception as _npv_err:
        print(f"  WARNING: Section 0b (tax NPV) failed: {_npv_err}")

    # Sector concentration warnings — injected before reasoning framework
    sector_warning_block = ""
    try:
        from kairos_sector_monitor import build_sector_warnings
        sector_warning_block = build_sector_warnings(shortlist or [], portfolio)
        if sector_warning_block:
            print("  Sector concentration warnings injected into prompt")
    except Exception as _sec_err:
        print(f"  WARNING: sector monitor failed: {_sec_err}")

    # ML Section — load ML scoring results
    ml_section = ""
    try:
        ml_result_file = os.path.join(SCRIPT_DIR, "kairos_ml_result.json")
        if os.path.exists(ml_result_file):
            with open(ml_result_file) as f:
                ml_data = json.load(f)
            scored_candidates = ml_data.get("candidates", [])
            model_trained_on = ml_data.get("model_info", {}).get("trained_on", 0)
            
            if scored_candidates and model_trained_on > 0:
                ml_lines = []
                ml_lines.append(f"{'=' * W}")
                ml_lines.append("SECTION 5b: ML PATTERN RECOGNITION (Track 1)")
                ml_lines.append(f"{'=' * W}")
                ml_lines.append("")
                ml_lines.append("Machine-learning model trained on historical trade outcomes.")
                ml_lines.append(f"Trained on: {model_trained_on} closed trades")
                ml_lines.append("")
                ml_lines.append("ML confidence scores (0-1) indicate probability of profitable trade.")
                ml_lines.append("ML signal strength: STRONG (>=0.7), NEUTRAL (0.4-0.7), WEAK (<=0.4)")
                ml_lines.append("")
                
                # Create a mapping of ticker to ML score
                ml_scores = {c["ticker"]: c for c in scored_candidates}
                
                # Sort by confidence descending
                sorted_candidates = sorted(scored_candidates, key=lambda x: x.get("ml_confidence", 0), reverse=True)
                
                for c in sorted_candidates:
                    ticker = c.get("ticker", "?")
                    confidence = c.get("ml_confidence", 0.5)
                    signal = c.get("ml_signal", "NEUTRAL")
                    ml_lines.append(f"  {ticker:<6} ML Confidence: {confidence:.3f}  Signal: {signal}")
                
                ml_lines.append("")
                ml_lines.append("Use ML scores as an additional signal factor. Higher confidence")
                ml_lines.append("suggests historical patterns favor this trade. STRONG signals merit")
                ml_lines.append("higher conviction; WEAK signals warrant caution or smaller size.")
                ml_lines.append("")
                
                ml_section = "\n".join(ml_lines)
                print(f"  ML Section added: {len(scored_candidates)} candidates scored")
            else:
                print("  ML Section: No ML scores available (model not yet trained)")
        else:
            print("  ML Section: No ML result file found")
    except Exception as ml_err:
        print(f"  WARNING: ML section failed: {ml_err}")

    prompt = f"""KAIROS REASONING PROMPT — {timestamp}
{'=' * W}
{blacklist_block}
You are evaluating the following {len(candidate_tickers)} shortlisted tickers
for the BEST single trade opportunity.

Your existing portfolio positions are listed separately below — do NOT
default to adding to existing positions unless they are genuinely the
best opportunity on the shortlist. Every candidate must compete equally.

{regime_block}{tax_npv_block}{sector_warning_block}{'=' * W}
SECTION 1: MARKET SNAPSHOT (5 SOURCES)
{'=' * W}

{market_data}

{'=' * W}
SECTION 2a: EXISTING POSITIONS (for context only)
{'=' * W}

{positions_block}

Account:
{acct_block}

{'=' * W}
SECTION 2b: CANDIDATE TICKERS FOR EVALUATION (evaluate ALL of these)
{'=' * W}

{candidates_block}

{tax_section}{tax_eff_section}{ledger_section}{screen_section}{ml_section}
{'=' * W}
REASONING FRAMEWORK
{'=' * W}

Analyze each factor for EVERY candidate ticker:

1. NEWS SENTIMENT — For each candidate, are recent headlines bullish,
   bearish, or neutral? What stories could move the stock?

2. MACRO ENVIRONMENT — What does the Fed funds rate tell us about
   monetary policy? How does this affect each sector represented?

3. RISK APPETITE — What do crypto trends (BTC/ETH direction + volume)
   and prediction markets signal about broader risk sentiment?

4. LEGISLATIVE RISK — Are any bills in Congress a threat or tailwind
   for any candidate on the shortlist?

5. PORTFOLIO FIT — How does each candidate fit the existing portfolio?
   Would adding a new position improve diversification? Is cash
   available? Does an existing holding already give this exposure?

6. TAX CONSIDERATIONS — Kairos optimizes for after-tax net profit, not
   gross return. Review Sections 0b, 3, and 3b. For positions flagged
   HOLD-FOR-TAX, do not recommend SELL unless stop-loss or thesis-invalid
   conditions are met. The after-tax NPV calculation already accounts for
   the opportunity cost of waiting — a HOLD-FOR-TAX flag means the math
   favors patience even after accounting for redeployment returns.

7. TRADING LESSONS — Review the pattern summary in Section 4. If
   today's signal combo matches a LOSING pattern, reduce size or skip.
   If it matches a WINNING pattern, trade with higher confidence.

8. BUY SIGNALS — Review Section 5 signal tags for each candidate:
   - HOT-EARNINGS: strong post-earnings momentum; check sustainability
   - HOT-RSI: oversold bounce candidate; confirm no fundamental cause
   - HOT-INSIDER: insiders buying; weigh filer role (CEO > director)
   - HOT-CONGRESS: congressional purchase; supporting signal
   - HOT-REVERSION: sharp drop >3%; fundamental (AVOID) or sentiment (BUY)?
   - HOT-KALSHI: prediction market shift; factor direction into thesis
   Multiple signals on one ticker compound confidence.

9. ML PATTERN RECOGNITION — Review Section 5b ML confidence scores and signals:
   - STRONG (>=0.7): Historical patterns strongly favor this trade
   - NEUTRAL (0.4-0.7): Mixed or limited historical data
   - WEAK (<=0.4): Historical patterns suggest caution
   Use ML as an additional signal factor alongside the human-readable signals.
   Higher ML confidence should increase your conviction; lower confidence
   warrants smaller position sizes or skipping the trade entirely.

{'=' * W}
MULTI-TRADE RANKING (REQUIRED)
{'=' * W}

Rank ALL eligible candidates by conviction. Trade ALL tickers that have
genuine HOT signals and merit a position — do not limit to a single winner.
{"(Blacklisted tickers excluded: " + ", ".join(blacklisted_symbols) + ")" if blacklisted_symbols else ""}

For each ticker, determine:
  - Action: BUY, SELL, or SKIP (no position warranted)
  - Conviction: score 1–10 (10 = highest)
  - Sector: the universe category (e.g. mega_cap, financials, healthcare_biotech, energy_materials, mid_cap_growth, etf_sector, etc.)
  - Rationale: 1–2 sentences

Position sizing is handled automatically downstream based on signal
confluence. You do NOT need to specify quantities. Focus on which
tickers to trade and why.

{'=' * W}
REQUIRED OUTPUT
{'=' * W}

Produce a JSON object with a "trades" array containing ALL recommended
trades, ranked by conviction (highest first). Include every ticker that
warrants a BUY or SELL. Skip tickers with no edge.

For every BUY, include a thesis block so we can track prediction accuracy:
  - predicted_direction:  "UP" | "DOWN" | "NEUTRAL"
  - predicted_timeframe_days:  integer (typical horizon: 3–21 days)
  - predicted_return_pct:  expected move in %, signed (e.g. +6.5 or -3.0)
  - key_conditions:  2–3 short bullets (one string, newline-separated) describing
                     what must remain true for the thesis to hold
  - invalidation_conditions:  2–3 short bullets describing what would PROVE the
                              thesis wrong (e.g. "earnings miss", "breaks $X support")

{{
  "trades": [
    {{
      "action": "BUY",
      "ticker": "<ticker>",
      "sector": "<universe category>",
      "conviction": <1-10>,
      "rationale": "<1-2 sentences>",
      "predicted_direction": "UP",
      "predicted_timeframe_days": 7,
      "predicted_return_pct": 5.0,
      "key_conditions": "- macro stays risk-on\\n- sector flows positive\\n- no negative earnings revision",
      "invalidation_conditions": "- closes below 50-day MA\\n- earnings miss\\n- sector rotation against"
    }},
    ...more trades if warranted...
  ],
  "tickers_evaluated": {json.dumps(eligible_tickers)},
  "skipped": "<brief note on why remaining tickers were not traded>"
}}

Rules:
  - Include ALL tickers that merit a trade, not just the top one.
  - If NO ticker warrants a trade, return an empty trades array.
  - Tag REVERSION trades in the rationale for ledger tracking.
  - Sector MUST match one of the universe categories.
  - Thesis fields are REQUIRED for every BUY (omit them on SELL/HOLD).
"""

    with open(PROMPT_FILE, "w") as f:
        f.write(prompt)

    print(f"  Written to {PROMPT_FILE}")
    return prompt


# ── main ────────────────────────────────────────────────────────────

def load_shortlist() -> list[str]:
    """Load the Tier 1 screening shortlist, falling back to [AAPL]."""
    screen_file = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")
    if os.path.exists(screen_file):
        try:
            with open(screen_file) as f:
                data = json.load(f)
            shortlist = data.get("shortlist", [])
            if shortlist:
                return shortlist
        except (json.JSONDecodeError, IOError):
            pass
    return [TICKER]


def main():
    print("╔" + "═" * W + "╗")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"║  KAIROS REASONING ENGINE — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    shortlist = load_shortlist()
    print(f"  Evaluating {len(shortlist)} ticker(s): {', '.join(shortlist)}")

    market_data = gather_market_data()
    portfolio = gather_portfolio(shortlist)
    tax_ctx = gather_tax_context()
    tax_eff = gather_tax_efficiency(portfolio, shortlist=shortlist)
    ledger_text = gather_ledger()
    write_prompt(market_data, portfolio, tax_ctx, ledger_text,
                 shortlist=shortlist, tax_efficiency_text=tax_eff)

    print(banner("Ready for Reasoning"))
    print("  kairos_prompt.txt is ready.")
    print(f"  Tickers to evaluate: {', '.join(shortlist)}")
    print("  Claude Code will now read it, reason, and write the decision.")
    print("━" * W)


if __name__ == "__main__":
    main()
