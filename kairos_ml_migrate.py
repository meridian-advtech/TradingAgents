"""
Kairos ML Migration — Backfill trade_outcomes from kairos_ledger.txt + kairos_decisions.log

Reads existing data sources and populates the ML outcomes table with as
many fields as can be inferred.  Nullable fields that can't be determined
are left null.

Sources:
  - kairos_ledger.txt   → ticker, action, signals_fired, pnl_pct, outcome_label
  - kairos_decisions.log → timestamp, ticker, action, quantity, price_entry,
                           price_exit (from fill_price on SELL executions)
  - kairos.db holdings   → entry_price, sold_price, hold duration

Usage:
    python kairos_ml_migrate.py            # Migrate existing data
    python kairos_ml_migrate.py --reset    # Drop trade_outcomes and re-migrate
"""

import argparse
import json
import os
import sqlite3
import uuid
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER_FILE = os.path.join(SCRIPT_DIR, "kairos_ledger.txt")
LOG_FILE = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
KAIROS_DB = os.path.join(SCRIPT_DIR, "kairos.db")
W = 72


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Parse kairos_decisions.log ────────────────────────────────────────

def parse_log_objects(path: str) -> list[dict]:
    """Extract all top-level JSON objects from the log file."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
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
    return objects


# ── Parse kairos_ledger.txt ───────────────────────────────────────────

def parse_ledger(path: str) -> list[dict]:
    """Parse pipe-delimited trade lines from the ledger.

    Format: date | ticker | action | signal_tags | pnl_pct% | PASS/FAIL
    """
    if not os.path.exists(path):
        return []

    trades = []
    in_trades = False
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line == "--- TRADE LOG ---":
                in_trades = True
                continue
            if not in_trades or not line or line.startswith("---"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                continue
            try:
                pnl = float(parts[4].replace("%", "").strip())
            except ValueError:
                continue
            verdict = parts[5].strip()
            signal_tags = [s.strip() for s in parts[3].split() if s.strip()]

            trades.append({
                "date": parts[0].strip(),
                "ticker": parts[1].strip(),
                "action": parts[2].strip(),
                "signals_fired": signal_tags,
                "pnl_pct": pnl,
                "outcome_label": "WIN" if verdict == "PASS" else "LOSS",
            })
    return trades


# ── Extract filled executions from decisions log ──────────────────────

def extract_filled_trades(log_objects: list[dict]) -> list[dict]:
    """Extract BUY and SELL executions that were filled."""
    filled = []
    for obj in log_objects:
        if obj.get("type") != "EXECUTION":
            continue
        exec_data = obj.get("execution", {})
        if exec_data.get("status") != "Filled":
            continue
        dec = obj.get("decision", {})
        action = dec.get("action", "").upper()
        if action not in ("BUY", "SELL"):
            continue

        filled.append({
            "timestamp": obj.get("timestamp", ""),
            "ticker": dec.get("ticker", ""),
            "action": action,
            "quantity": dec.get("quantity", 0),
            "fill_price": exec_data.get("fill_price"),
            "fill_time": exec_data.get("fill_time", ""),
            "tickers_evaluated": dec.get("tickers_evaluated", []),
            "runner_up": dec.get("runner_up", ""),
            "rationale": dec.get("rationale", ""),
        })
    return filled


# ── Load holdings from kairos.db ──────────────────────────────────────

def load_holdings_from_db() -> list[dict]:
    """Load closed holdings (sold_date IS NOT NULL) from kairos.db."""
    if not os.path.exists(KAIROS_DB):
        return []
    conn = sqlite3.connect(KAIROS_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT *, CAST(julianday(sold_date) - julianday(entry_date) AS INTEGER) AS holding_days
               FROM holdings WHERE sold_date IS NOT NULL ORDER BY entry_date"""
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


# ── Load signal summary for confluence data ───────────────────────────

def load_signal_summary() -> dict:
    """Load signal tags from kairos_signal_summary.json."""
    path = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("signal_tags", {})
    except (json.JSONDecodeError, IOError):
        return {}


# ── Infer confluence score from signal tags ───────────────────────────

SIGNAL_POINTS = {
    "HOT-EARNINGS":  2,
    "HOT-RSI":       1,
    "HOT-INSIDER":   2,
    "HOT-CONGRESS":  1,
    "HOT-REVERSION": 1,
    "HOT-KALSHI":    1,
    "HOT-OPTIONS":   2,
}


def compute_confluence_from_tags(tags: list[str]) -> int:
    """Compute a confluence score from signal tag names."""
    score = 0
    for tag in tags:
        tag_upper = tag.upper()
        for key, pts in SIGNAL_POINTS.items():
            if key in tag_upper:
                score += pts
                break
    return score


# ── GICS sector lookup (basic mapping) ────────────────────────────────

TICKER_SECTOR = {
    "AAPL": "Information Technology",
    "S": "Information Technology",
    "CI": "Health Care",
    "CSCO": "Information Technology",
    "JPM": "Financials",
    "FDX": "Industrials",
    "DASH": "Consumer Discretionary",
    "RBLX": "Communication Services",
    "PATH": "Information Technology",
    "DD": "Materials",
    "REGN": "Health Care",
    "ROKU": "Communication Services",
    "SE": "Communication Services",
    "LCID": "Consumer Discretionary",
    "D": "Utilities",
    "F": "Consumer Discretionary",
    "ICE": "Financials",
    "MO": "Consumer Staples",
    "SO": "Utilities",
    "T": "Communication Services",
    "ABNB": "Consumer Discretionary",
    "AFRM": "Financials",
    "EOG": "Energy",
    "EPD": "Energy",
}


# ── Main migration ────────────────────────────────────────────────────

def migrate(reset: bool = False):
    from kairos_ml_outcomes import init_db, get_connection, DB_PATH

    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS ML MIGRATION".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    if reset:
        print(banner("Resetting trade_outcomes table"))
        conn = get_connection()
        conn.execute("DROP TABLE IF EXISTS trade_outcomes")
        conn.commit()
        conn.close()

    init_db()
    print(f"  Database: {DB_PATH}")

    # Check existing count
    conn = get_connection()
    existing = conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
    conn.close()
    if existing > 0 and not reset:
        print(f"  trade_outcomes already has {existing} rows — skipping migration.")
        print("  Use --reset to drop and re-migrate.")
        return

    # ── Load all sources ──────────────────────────────────────────────
    print(banner("Loading Sources"))

    log_objects = parse_log_objects(LOG_FILE)
    print(f"  kairos_decisions.log: {len(log_objects)} JSON entries")

    filled_trades = extract_filled_trades(log_objects)
    print(f"  Filled executions: {len(filled_trades)}")

    ledger_trades = parse_ledger(LEDGER_FILE)
    print(f"  kairos_ledger.txt: {len(ledger_trades)} closed trades")

    closed_holdings = load_holdings_from_db()
    print(f"  kairos.db holdings (closed): {len(closed_holdings)}")

    signal_summary = load_signal_summary()
    print(f"  Signal summary tickers: {len(signal_summary)}")

    # ── Build trade_outcomes rows from filled executions ──────────────
    print(banner("Migrating Filled Executions"))

    conn = get_connection()
    migrated = 0

    for trade in filled_trades:
        ticker = trade["ticker"]
        action = trade["action"]
        ts = trade["timestamp"]

        # Convert timestamp to ISO
        ts_iso = ts.replace(" UTC", "").strip()
        try:
            dt = datetime.strptime(ts_iso, "%Y-%m-%d %H:%M:%S")
            ts_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

        # Signals from summary
        signals = signal_summary.get(ticker, [])
        confluence = compute_confluence_from_tags(signals) if signals else None

        # Sector
        sector = TICKER_SECTOR.get(ticker)

        trade_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO trade_outcomes
               (trade_id, timestamp_entry, ticker, action, quantity,
                price_entry, signals_fired, confluence_score, sector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                ts_iso,
                ticker,
                action,
                trade["quantity"],
                trade["fill_price"],
                json.dumps(signals) if signals else None,
                confluence,
                sector,
            ),
        )
        migrated += 1

    conn.commit()
    conn.close()

    print(f"  Migrated {migrated} execution(s) as open trades")

    # ── Cross-reference ledger closed trades ──────────────────────────
    if ledger_trades:
        print(banner("Backfilling Ledger Closed Trades"))
        conn = get_connection()
        backfilled = 0
        for lt in ledger_trades:
            # Try to find matching open trade and close it
            row = conn.execute(
                """SELECT trade_id, price_entry FROM trade_outcomes
                   WHERE ticker = ? AND outcome_label IS NULL
                   ORDER BY timestamp_entry ASC LIMIT 1""",
                (lt["ticker"],),
            ).fetchone()
            if row:
                price_entry = row["price_entry"]
                pnl_pct = lt["pnl_pct"]
                # Back-compute price_exit from pnl_pct
                price_exit = price_entry * (1 + pnl_pct / 100) if price_entry else None
                pnl_dollar = None

                conn.execute(
                    """UPDATE trade_outcomes
                       SET pnl_pct = ?, price_exit = ?, pnl_dollar = ?,
                           outcome_label = ?, signals_fired = ?
                       WHERE trade_id = ?""",
                    (
                        round(pnl_pct, 4),
                        round(price_exit, 4) if price_exit else None,
                        pnl_dollar,
                        lt["outcome_label"],
                        json.dumps(lt["signals_fired"]) if lt["signals_fired"] else None,
                        row["trade_id"],
                    ),
                )
                backfilled += 1
        conn.commit()
        conn.close()
        print(f"  Backfilled {backfilled} closed trade(s) from ledger")

    # ── Cross-reference closed holdings for duration ──────────────────
    if closed_holdings:
        print(banner("Backfilling Hold Duration from Holdings"))
        conn = get_connection()
        duration_filled = 0
        for h in closed_holdings:
            row = conn.execute(
                """SELECT trade_id FROM trade_outcomes
                   WHERE ticker = ? AND price_entry = ?
                   AND hold_duration_mins IS NULL
                   LIMIT 1""",
                (h["ticker"], h["entry_price"]),
            ).fetchone()
            if row and h.get("holding_days") is not None:
                duration_mins = h["holding_days"] * 24 * 60
                conn.execute(
                    "UPDATE trade_outcomes SET hold_duration_mins = ? WHERE trade_id = ?",
                    (duration_mins, row["trade_id"]),
                )
                duration_filled += 1
        conn.commit()
        conn.close()
        print(f"  Backfilled {duration_filled} hold duration(s)")

    # ── Summary ───────────────────────────────────────────────────────
    print(banner("Verification"))
    from kairos_ml_outcomes import get_connection as gc
    conn = gc()
    total = conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
    open_n = conn.execute(
        "SELECT COUNT(*) FROM trade_outcomes WHERE outcome_label IS NULL"
    ).fetchone()[0]
    closed_n = conn.execute(
        "SELECT COUNT(*) FROM trade_outcomes WHERE outcome_label IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    print(f"  Total trades:  {total}")
    print(f"  Open:          {open_n}")
    print(f"  Closed:        {closed_n}")

    null_fields = {}
    conn = gc()
    for col in ("signals_fired", "confluence_score", "council_member_1_rec",
                "council_member_2_rec", "market_regime", "sector"):
        n = conn.execute(
            f"SELECT COUNT(*) FROM trade_outcomes WHERE {col} IS NULL"
        ).fetchone()[0]
        if n > 0:
            null_fields[col] = n
    conn.close()
    if null_fields:
        print("  Null fields (expected for historical data):")
        for col, n in null_fields.items():
            print(f"    {col}: {n}/{total} null")

    print("\n" + "━" * W)
    print("  Migration complete.")
    print("━" * W)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos ML Migration")
    parser.add_argument("--reset", action="store_true",
                        help="Drop trade_outcomes and re-migrate")
    args = parser.parse_args()
    migrate(reset=args.reset)
