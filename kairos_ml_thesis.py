"""
Kairos ML Thesis — Thesis Validation + Checkpoint Tracking

Captures Claude's structured prediction at trade entry, walks open
positions on a 3 / 7 / 14 / 21-day cadence to score how the prediction
is holding up, and feeds per-signal aggregate stats into the EOD report.

Tables (created by kairos_ml_outcomes.init_db):
    thesis_predictions  — one row per BUY at entry
    thesis_checkpoints  — one row per (decision_id, checkpoint_day)
    signal_performance  — view aggregating across trade_outcomes

Public surface:
    write_thesis_prediction(decision_id, ticker, ...)
    run_thesis_checkpoints(ib=None)
    get_signal_performance() -> dict
    format_signal_performance_section() -> str
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")

# Fallback cadence when a signal type has no configured hold window.
DEFAULT_CHECKPOINT_DAYS = (3, 7, 14, 21)


# ── DB connection ─────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_schema() -> None:
    """Idempotent: make sure the thesis tables / view exist."""
    try:
        from kairos_ml_outcomes import init_db
        init_db()
    except Exception:
        pass


# ── Config loaders ────────────────────────────────────────────────────

def _load_signal_hold_windows() -> dict[str, int]:
    """Read signal_hold_windows from kairos_config.json (reallocation)."""
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        return cfg.get("reallocation", {}).get("signal_hold_windows", {}) or {}
    except (IOError, json.JSONDecodeError):
        return {}


def _checkpoint_days_for(signal_type: Optional[str]) -> tuple[int, ...]:
    """Return the (day) checkpoint cadence for a signal.

    Strategy: always evaluate at the standard 3 / 7 / 14 / 21 cadence, but
    cap at the configured hold window so we don't keep checkpointing a
    signal whose horizon has expired. For example HOT-EARNINGS has a 10-day
    hold window → checkpoints fire at day 3 and 7 only.
    """
    windows = _load_signal_hold_windows()
    horizon = None
    if signal_type:
        horizon = windows.get(signal_type)
    if horizon is None:
        horizon = windows.get("DEFAULT")
    if horizon is None:
        return DEFAULT_CHECKPOINT_DAYS
    return tuple(d for d in DEFAULT_CHECKPOINT_DAYS if d <= int(horizon))


# ── Primary signal extraction ─────────────────────────────────────────

# Used when caller doesn't know which signal "led" the trade. We prefer
# higher-weighted / earnings-driven tags first.
_SIGNAL_PRIORITY = (
    "HOT-EARNINGS", "HOT-INSIDER", "HOT-OPTIONS", "HOT-CONGRESS",
    "HOT-RSI", "HOT-REVERSION", "HOT-KALSHI", "HOT-CATALYST",
)


def pick_primary_signal(signals: Optional[list[str]]) -> Optional[str]:
    """Return the strongest signal tag from a list, or None."""
    if not signals:
        return None
    tagset = {s.upper() for s in signals if s}
    for tag in _SIGNAL_PRIORITY:
        if tag in tagset:
            return tag
    # Fall back to the first non-empty tag
    return signals[0] if signals else None


# ── Write prediction at entry ─────────────────────────────────────────

def write_thesis_prediction(
    *,
    decision_id: str,
    ticker: str,
    timestamp_entry: Optional[str] = None,
    predicted_direction: Optional[str] = None,
    predicted_timeframe_days: Optional[int] = None,
    predicted_return_pct: Optional[float] = None,
    key_conditions: Optional[str] = None,
    signal_type: Optional[str] = None,
    conviction_score: Optional[float] = None,
    invalidation_conditions: Optional[str] = None,
) -> int:
    """Insert a thesis prediction row. Returns the new id.

    decision_id links to trade_outcomes.trade_id (the UUID minted by
    write_trade_open at BUY time).
    """
    _ensure_schema()
    if timestamp_entry is None:
        timestamp_entry = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    direction = (predicted_direction or "").strip().upper() or None
    if direction not in (None, "UP", "DOWN", "NEUTRAL"):
        direction = None

    conn = _get_connection()
    cur = conn.execute(
        """INSERT INTO thesis_predictions
           (ticker, decision_id, timestamp_entry,
            predicted_direction, predicted_timeframe_days, predicted_return_pct,
            key_conditions, signal_type, conviction_score, invalidation_conditions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker, decision_id, timestamp_entry,
            direction, predicted_timeframe_days, predicted_return_pct,
            key_conditions, signal_type, conviction_score, invalidation_conditions,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_thesis_prediction(decision_id: str) -> Optional[dict]:
    """Return the latest thesis_prediction row for a decision_id, or None."""
    _ensure_schema()
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM thesis_predictions WHERE decision_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (decision_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Checkpoint mechanics ──────────────────────────────────────────────

def _parse_ts(ts: str) -> Optional[datetime]:
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _days_since(ts_entry: str, now: Optional[datetime] = None) -> float:
    dt_entry = _parse_ts(ts_entry)
    if dt_entry is None:
        return 0.0
    now = now or datetime.utcnow()
    return max(0.0, (now - dt_entry).total_seconds() / 86400.0)


def _score_checkpoint(
    predicted_direction: Optional[str],
    predicted_return_pct: Optional[float],
    pct_move_actual: float,
) -> tuple[float, bool]:
    """Return (checkpoint_score, direction_correct)."""
    direction_actual = (
        "UP" if pct_move_actual > 0.0
        else "DOWN" if pct_move_actual < 0.0
        else "NEUTRAL"
    )
    pred_dir = (predicted_direction or "").strip().upper()

    if pred_dir == "NEUTRAL":
        # Neutral thesis: |move| < 1% counts as correct
        direction_correct = abs(pct_move_actual) < 1.0
    else:
        direction_correct = (pred_dir == direction_actual) and pred_dir in ("UP", "DOWN")

    if not direction_correct:
        return 0.0, False

    if predicted_return_pct is not None and predicted_return_pct != 0.0:
        target = float(predicted_return_pct)
        # >50% of predicted return achieved (in the predicted direction)
        if pred_dir == "DOWN":
            if pct_move_actual <= 0.5 * target:
                return 1.0, True
        else:  # UP or default
            if pct_move_actual >= 0.5 * target:
                return 1.0, True

    return 0.5, True


def _get_open_predictions() -> list[dict]:
    """Return predictions whose trade is still open in trade_outcomes.

    Joins so we have the entry price too. Predictions with no matching
    trade_outcomes row are skipped (graceful degradation).
    """
    _ensure_schema()
    conn = _get_connection()
    rows = conn.execute(
        """SELECT tp.id           AS pred_id,
                  tp.decision_id  AS decision_id,
                  tp.ticker       AS ticker,
                  tp.timestamp_entry AS timestamp_entry,
                  tp.predicted_direction AS predicted_direction,
                  tp.predicted_timeframe_days AS predicted_timeframe_days,
                  tp.predicted_return_pct AS predicted_return_pct,
                  tp.signal_type  AS signal_type,
                  trade_outcomes.price_entry AS price_entry,
                  trade_outcomes.signals_fired AS signals_fired
           FROM thesis_predictions AS tp
           JOIN trade_outcomes
             ON trade_outcomes.trade_id = tp.decision_id
           WHERE trade_outcomes.outcome_label IS NULL"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _is_ipo_trade(pred: dict) -> bool:
    """True if this prediction is associated with an IPO_MOMENTUM trade.

    Checks (a) the prediction's signal_type, and (b) the signals_fired
    JSON list captured on the trade. Falls back to the live IPO catalog
    via is_recent_ipo() so positions opened just before the 5-day window
    are still treated as IPOs through their hold.
    """
    sig_type = (pred.get("signal_type") or "").upper()
    if sig_type == "IPO_MOMENTUM":
        return True
    raw = pred.get("signals_fired")
    if raw:
        try:
            tags = json.loads(raw) if isinstance(raw, str) else list(raw)
            if any((str(t).upper() == "IPO_MOMENTUM") for t in tags):
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        from kairos_signals_ipo import is_recent_ipo
        return is_recent_ipo(pred.get("ticker") or "")
    except Exception:
        return False


def _build_ipo_metrics(ticker: str, current_price: float) -> Optional[dict]:
    """Pull IPO context and return the per-checkpoint snapshot dict."""
    try:
        from kairos_signals_ipo import get_ipo_context
    except Exception:
        return None
    try:
        ctx = get_ipo_context(ticker) or {}
    except Exception:
        return None
    ipo_price = ctx.get("ipo_price")
    day1_close = ctx.get("day1_close")
    above_ipo_price = None
    above_day1_close = None
    pct_from_ipo = None
    if ipo_price is not None and current_price is not None:
        try:
            above_ipo_price = bool(current_price > float(ipo_price))
            if float(ipo_price) > 0:
                pct_from_ipo = (current_price - float(ipo_price)) \
                    / float(ipo_price) * 100.0
        except (TypeError, ValueError):
            pass
    if day1_close is not None and current_price is not None:
        try:
            above_day1_close = bool(current_price > float(day1_close))
        except (TypeError, ValueError):
            pass
    return {
        "above_ipo_price": above_ipo_price,
        "above_day1_close": above_day1_close,
        "pct_from_ipo": (
            round(pct_from_ipo, 4) if pct_from_ipo is not None else None
        ),
        "ipo_price": ipo_price,
        "day1_close": day1_close,
        "days_since_ipo": ctx.get("days_since_ipo"),
    }


def _existing_checkpoint_days(decision_id: str) -> set[int]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT checkpoint_day FROM thesis_checkpoints WHERE decision_id = ?",
        (decision_id,),
    ).fetchall()
    conn.close()
    return {int(r["checkpoint_day"]) for r in rows}


def _record_checkpoint(
    *,
    decision_id: str,
    ticker: str,
    checkpoint_day: int,
    price_at_entry: float,
    price_at_checkpoint: float,
    pct_move_actual: float,
    direction_correct: bool,
    checkpoint_score: float,
    notes: str = "",
    ipo_metrics: Optional[dict] = None,
) -> None:
    conn = _get_connection()
    try:
        ipo_json = json.dumps(ipo_metrics) if ipo_metrics else None
        conn.execute(
            """INSERT OR IGNORE INTO thesis_checkpoints
               (ticker, decision_id, checkpoint_day, timestamp_checked,
                price_at_entry, price_at_checkpoint, pct_move_actual,
                direction_correct, thesis_conditions_intact, notes,
                checkpoint_score, ipo_metrics)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker, decision_id, checkpoint_day,
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                price_at_entry, price_at_checkpoint,
                round(pct_move_actual, 4),
                1 if direction_correct else 0,
                1 if direction_correct else 0,  # default conditions_intact = direction_correct
                notes,
                round(checkpoint_score, 4),
                ipo_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ── Price fetcher ─────────────────────────────────────────────────────

def _get_current_price(ticker: str, ib=None) -> Optional[float]:
    """Best-effort current price via an existing or fresh IBKR connection."""
    from ib_insync import IB, Stock

    own_conn = False
    if ib is None:
        try:
            import random
            ib = IB()
            ib.connect("127.0.0.1", 7497, clientId=random.randint(11, 14), timeout=10)
            own_conn = True
        except Exception:
            return None

    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqMarketDataType(4)
        mkt = ib.reqMktData(contract)
        ib.sleep(2)
        price: Optional[float] = None
        for attr in ("last", "close", "bid", "ask"):
            val = getattr(mkt, attr, None)
            if val is not None and val == val and val > 0:
                price = float(val)
                break
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass
        return round(price, 4) if price else None
    except Exception:
        return None
    finally:
        if own_conn:
            try:
                ib.disconnect()
            except Exception:
                pass


# ── Cycle entry point ─────────────────────────────────────────────────

def run_thesis_checkpoints(ib=None) -> dict:
    """Walk open positions; record any newly-due checkpoints.

    For each open prediction:
      - compute days_since(entry)
      - determine which checkpoint days are due (3 / 7 / 14 / 21, capped by
        signal hold window) and not yet recorded
      - if any is due, fetch current price and write one row per due day
      - if a written checkpoint scores 0.0 on day >= 7, alert #kairos-alerts

    Returns a small summary dict for the cycle log.
    """
    summary = {
        "evaluated": 0,
        "checkpoints_written": 0,
        "warnings_emitted": 0,
        "skipped_missing_price": 0,
    }

    try:
        predictions = _get_open_predictions()
    except Exception as exc:
        print(f"  [thesis-checkpoints] WARNING: failed to load predictions: {exc}")
        return summary

    if not predictions:
        return summary

    summary["evaluated"] = len(predictions)
    now = datetime.utcnow()

    for pred in predictions:
        decision_id = pred["decision_id"]
        ticker = pred["ticker"]
        signal_type = pred.get("signal_type")
        try:
            cadence = _checkpoint_days_for(signal_type)
            if not cadence:
                continue
            days_elapsed = _days_since(pred["timestamp_entry"], now=now)
            due = [d for d in cadence if days_elapsed >= d]
            if not due:
                continue
            already = _existing_checkpoint_days(decision_id)
            outstanding = [d for d in due if d not in already]
            if not outstanding:
                continue

            current_price = _get_current_price(ticker, ib=ib)
            if current_price is None:
                summary["skipped_missing_price"] += 1
                continue

            price_entry = pred.get("price_entry")
            if not price_entry or price_entry <= 0:
                continue
            pct_move = (current_price - price_entry) / price_entry * 100.0

            predicted_direction = pred.get("predicted_direction")
            predicted_return = pred.get("predicted_return_pct")

            score, direction_correct = _score_checkpoint(
                predicted_direction=predicted_direction,
                predicted_return_pct=predicted_return,
                pct_move_actual=pct_move,
            )

            # IPO-specific checkpoint snapshot (only for IPO_MOMENTUM trades)
            ipo_metrics: Optional[dict] = None
            if _is_ipo_trade(pred):
                ipo_metrics = _build_ipo_metrics(ticker, current_price)

            for day in outstanding:
                _record_checkpoint(
                    decision_id=decision_id,
                    ticker=ticker,
                    checkpoint_day=day,
                    price_at_entry=price_entry,
                    price_at_checkpoint=current_price,
                    pct_move_actual=pct_move,
                    direction_correct=direction_correct,
                    checkpoint_score=score,
                    notes=(f"signal={signal_type or 'n/a'} "
                           f"predicted={predicted_direction or '?'}@{predicted_return}%"),
                    ipo_metrics=ipo_metrics,
                )
                summary["checkpoints_written"] += 1

                if score == 0.0 and day >= 7:
                    _emit_thesis_warning(
                        ticker=ticker,
                        day=day,
                        pct_move=pct_move,
                        predicted_direction=predicted_direction,
                        predicted_return=predicted_return,
                        signal_type=signal_type,
                    )
                    summary["warnings_emitted"] += 1

        except Exception as exc:
            print(f"  [thesis-checkpoints] WARNING: {ticker} ({decision_id[:8]}): {exc}")
            continue

    if summary["checkpoints_written"] or summary["warnings_emitted"]:
        print(
            f"  [thesis-checkpoints] evaluated={summary['evaluated']} "
            f"written={summary['checkpoints_written']} "
            f"warnings={summary['warnings_emitted']} "
            f"skipped={summary['skipped_missing_price']}"
        )
    return summary


def _emit_thesis_warning(
    *,
    ticker: str,
    day: int,
    pct_move: float,
    predicted_direction: Optional[str],
    predicted_return: Optional[float],
    signal_type: Optional[str],
) -> None:
    """Post a #kairos-alerts warning when a checkpoint scores 0.0 on day 7+."""
    try:
        from kairos_alerts import alert_pipeline_event
        msg = (
            f":warning: *Thesis off-track — {ticker}* (day {day} checkpoint)\n"
            f"  Predicted: {predicted_direction or '?'} "
            f"{predicted_return if predicted_return is not None else '?'}% "
            f"(signal: {signal_type or 'n/a'})\n"
            f"  Actual move so far: {pct_move:+.2f}%\n"
            f"  Direction is wrong — consider reviewing the thesis."
        )
        alert_pipeline_event(msg, channel="alerts")
    except Exception as exc:
        print(f"  [thesis-checkpoints] WARNING: failed to alert for {ticker}: {exc}")


# ── Aggregate signal performance ──────────────────────────────────────

def get_signal_performance() -> dict:
    """Return per-signal aggregate stats from the signal_performance view.

    Shape:
        {
          "HOT-EARNINGS": {
              "total_trades": 12,
              "win_rate": 0.58,
              "avg_return_pct": 4.2,
              "avg_hold_days": 6.1,
              "avg_prediction_accuracy": 0.62,
          },
          ...
        }
    """
    _ensure_schema()
    out: dict[str, dict] = {}
    try:
        conn = _get_connection()
        rows = conn.execute(
            "SELECT signal_type, total_trades, win_rate, avg_return_pct, "
            "       avg_hold_days, avg_prediction_accuracy "
            "FROM signal_performance "
            "ORDER BY total_trades DESC"
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"  [signal-performance] WARNING: query failed: {exc}")
        return out

    for r in rows:
        sig = r["signal_type"] or "UNKNOWN"
        out[sig] = {
            "total_trades": int(r["total_trades"] or 0),
            "win_rate": float(r["win_rate"]) if r["win_rate"] is not None else None,
            "avg_return_pct": float(r["avg_return_pct"]) if r["avg_return_pct"] is not None else None,
            "avg_hold_days": float(r["avg_hold_days"]) if r["avg_hold_days"] is not None else None,
            "avg_prediction_accuracy": (
                float(r["avg_prediction_accuracy"])
                if r["avg_prediction_accuracy"] is not None else None
            ),
        }
    return out


def format_signal_performance_section() -> str:
    """Markdown-ish Slack section for the EOD report. Empty string if no data."""
    perf = get_signal_performance()
    if not perf:
        return ""

    lines = ["*Signal Performance (closed trades)*", "```"]
    lines.append(
        f"  {'signal':<18}{'trades':>7}  {'win%':>6}  {'avg ret':>8}  "
        f"{'avg hold':>9}  {'pred acc':>9}"
    )
    for sig, stats in perf.items():
        trades = stats["total_trades"]
        wr = stats["win_rate"]
        ret = stats["avg_return_pct"]
        hold = stats["avg_hold_days"]
        acc = stats["avg_prediction_accuracy"]
        wr_str = f"{wr*100:5.0f}%" if wr is not None else "    —"
        ret_str = f"{ret:+7.2f}%" if ret is not None else "      —"
        hold_str = f"{hold:7.1f}d" if hold is not None else "      —"
        acc_str = f"{acc:8.2f}" if acc is not None else "       —"
        lines.append(
            f"  {sig:<18}{trades:>7}  {wr_str:>6}  {ret_str:>8}  "
            f"{hold_str:>9}  {acc_str:>9}"
        )
    lines.append("```")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kairos thesis validation")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Run one checkpoint sweep over open predictions")
    parser.add_argument("--performance", action="store_true",
                        help="Print signal performance summary")
    args = parser.parse_args()

    if args.checkpoints:
        result = run_thesis_checkpoints()
        print(json.dumps(result, indent=2))
    elif args.performance:
        print(format_signal_performance_section() or "(no closed trades yet)")
    else:
        parser.print_help()
