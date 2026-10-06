"""
Kairos Outcome Features — the nightly trade-record job.

Three steps, in order (the fills-ledger addendum, 2026-09-28):
  (a) Flex Web Service pull of the configured query → record_fills
      (source 'flex_nightly') + record_position_events. Catches every fill the
      live path and the broker check missed, and late commissions. An HTTP or
      token error posts an alert naming the token — never fails silently.
  (b) rebuild_trades() — deterministic FIFO over fills + position_events.
  (c) Outcome features for CLOSED long trades, written to trade_features (the
      trade_outcomes view reads them from there):

    mfe_pct                — (max High in [entry, exit] − entry)/entry × 100
    give_back_pct          — mfe_pct − pnl_pct  (points of the in-hold peak surrendered)
    post_exit_peak_pct     — (max High in (exit, exit+window] − exit)/exit × 100
                             ONLY once the full window has elapsed; else NULL
    post_exit_window_days  — the window used
    forgone_gain_{5,14,30,60}d_pct — see kairos_ml_outcomes.FORGONE_HORIZONS
    prediction_accuracy / thesis_score — the entry thesis scored against the
                             realized close (was stamped at close time by the
                             old ledger; this job is now its only writer)
    features_filled_at     — set when mfe/give-back are computed

exit_reason is no longer written here: the view takes it from the trade's last
exit_annotations row (or 'CORPORATE-ACTION' for a cash merger).

Price fetch REUSES the Arbiter's batched yfinance helpers (kairos_arbiter.
_download_daily / _parse_utc). The MFE / give-back / post-exit MATH below is a
near-duplicate of kairos_arbiter.enrich_closed_trades for now.
    TODO(dedupe): once this module is the single source of truth for outcome
    features, refactor kairos_arbiter.enrich_closed_trades to call
    compute_features_for_trade() here instead of re-deriving the same metrics.

CLI:
    python3 kairos_outcome_features.py --backfill            # (a)+(b)+(c)
    python3 kairos_outcome_features.py --backfill --no-flex  # skip the Flex pull
    python3 kairos_outcome_features.py --backfill --dry-run  # print, write nothing
    python3 kairos_outcome_features.py --backfill --window 7
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Reuse the Arbiter's batched price-fetch + timestamp parsing verbatim (no edits
# to kairos_arbiter.py — these are imported, not copied).
from kairos_arbiter import _download_daily, _parse_utc


# ── Per-trade feature computation ────────────────────────────────────

def compute_features_for_trade(ticker, ts_entry, ts_exit, price_entry, price_exit,
                               action, window_days, pnl_pct=None,
                               _hist=None, now=None) -> dict:
    """Compute outcome features for ONE closed long trade.

    LONG only: returns {"skipped": reason} for any non-BUY action. Fields that
    cannot yet be computed are None (e.g. post_exit_peak_pct before the window
    matures). _hist optionally injects a pre-fetched {ticker: DataFrame} map so a
    caller can batch the yfinance download across many trades.
    """
    if (action or "").upper() != "BUY":
        return {"skipped": f"non-long action {action!r}"}

    entry_dt = _parse_utc(ts_entry)
    exit_dt = _parse_utc(ts_exit)
    if entry_dt is None or exit_dt is None:
        return {"skipped": "unparseable entry/exit timestamp"}

    try:
        price_entry = float(price_entry)
        price_exit = float(price_exit)
    except (TypeError, ValueError):
        return {"skipped": "missing entry/exit price"}
    if price_entry <= 0 or price_exit <= 0:
        return {"skipped": "non-positive entry/exit price"}

    if pnl_pct is None:
        pnl_pct = (price_exit - price_entry) / price_entry * 100.0
    else:
        pnl_pct = float(pnl_pct)

    now = now or datetime.now(timezone.utc)
    window_elapsed = now >= (exit_dt + timedelta(days=window_days))

    result = {
        "ticker": ticker,
        "pnl_pct": round(pnl_pct, 4),
        "mfe_pct": None,
        "give_back_pct": None,
        "post_exit_peak_pct": None,
        "post_exit_window_days": window_days,
        "window_elapsed": window_elapsed,
    }

    # Price history: injected (batched) or fetched per-trade.
    if _hist is not None:
        df = _hist.get(ticker)
    else:
        start = (entry_dt - timedelta(days=2)).strftime("%Y-%m-%d")
        end = (exit_dt + timedelta(days=window_days + 3)).strftime("%Y-%m-%d")
        df = _download_daily([ticker], start, end).get(ticker)

    if df is None or "High" not in getattr(df, "columns", []):
        result["skipped"] = "no price history"
        return result

    highs = df["High"]

    # MFE during the hold: peak High between entry and exit (inclusive).
    hold_highs = highs.loc[entry_dt.strftime("%Y-%m-%d"):exit_dt.strftime("%Y-%m-%d")]
    if len(hold_highs) > 0:
        peak = float(hold_highs.max())
        mfe = (peak - price_entry) / price_entry * 100.0
        result["mfe_pct"] = round(mfe, 4)
        result["give_back_pct"] = round(mfe - pnl_pct, 4)

    # Post-exit peak within the window — only once the window has fully elapsed.
    if window_elapsed:
        post_start = (exit_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        post_end = (exit_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")
        post_highs = highs.loc[post_start:post_end]
        if len(post_highs) > 0:
            pk = float(post_highs.max())
            result["post_exit_peak_pct"] = round((pk - price_exit) / price_exit * 100.0, 4)

    # ── Forgone gain across every horizon ────────────────────────────
    # Same canonical measure as post_exit_peak_pct, evaluated at each horizon in
    # FORGONE_HORIZONS: the best exit still available within N days of the one
    # taken. Each horizon matures independently, so a trade closed 20 days ago
    # yields 5d and 14d while 30d and 60d stay NULL rather than being computed
    # from a truncated window (which would understate forgone gain — exactly the
    # bias the longer horizons exist to remove).
    from kairos_ml_outcomes import FORGONE_HORIZONS, forgone_column
    post_start = (exit_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    for h in FORGONE_HORIZONS:
        col = forgone_column(h)
        result[col] = None
        if now < (exit_dt + timedelta(days=h)):
            continue  # window not yet elapsed
        hi = highs.loc[post_start:(exit_dt + timedelta(days=h)).strftime("%Y-%m-%d")]
        if len(hi) > 0:
            result[col] = round((float(hi.max()) - price_exit) / price_exit * 100.0, 4)

    return result


# ── Backfill driver ──────────────────────────────────────────────────

def _select_rows(conn, window_days, only_missing, now):
    """Closed trades needing feature work.

    only_missing: features_filled_at IS NULL, OR post_exit_peak_pct IS NULL and
    the post-exit window has now elapsed (so a previously-immature row matures).
    """
    from kairos_ml_outcomes import FORGONE_HORIZONS, forgone_column
    fg_cols = [forgone_column(h) for h in FORGONE_HORIZONS]
    rows = conn.execute(
        "SELECT trade_id, ticker, timestamp_entry, timestamp_exit, "
        "       price_entry, price_exit, pnl_pct, action, "
        "       mfe_pct, post_exit_peak_pct, features_filled_at, "
        + ", ".join(fg_cols) +
        " FROM trade_outcomes WHERE timestamp_exit IS NOT NULL "
        "ORDER BY timestamp_exit"
    ).fetchall()
    if not only_missing:
        return list(rows)

    todo = []
    for r in rows:
        if r["features_filled_at"] is None:
            todo.append(r)
            continue
        xdt = _parse_utc(r["timestamp_exit"])
        if r["post_exit_peak_pct"] is None:
            if xdt and now >= (xdt + timedelta(days=window_days)):
                todo.append(r)
                continue
        # A longer horizon that has newly matured also makes the row due — this
        # is what lets 30d/60d forgone gain fill in over time without a manual
        # re-run per horizon.
        if xdt and any(r[forgone_column(h)] is None
                       and now >= (xdt + timedelta(days=h))
                       for h in FORGONE_HORIZONS):
            todo.append(r)
    return todo


def _thesis_scores(conn, trade_id: str, pnl_pct, hold_mins) -> tuple:
    """(prediction_accuracy, thesis_score) for a closed trade, or (None, None)
    when no thesis prediction was logged for it."""
    from kairos_ml_outcomes import _score_prediction_accuracy
    pred = conn.execute(
        "SELECT predicted_direction, predicted_timeframe_days, predicted_return_pct "
        "FROM thesis_predictions WHERE decision_id = ? ORDER BY id DESC LIMIT 1",
        (trade_id,)).fetchone()
    if pred is None or pnl_pct is None:
        return None, None
    acc = _score_prediction_accuracy(
        predicted_direction=pred["predicted_direction"],
        predicted_timeframe_days=pred["predicted_timeframe_days"],
        predicted_return_pct=pred["predicted_return_pct"],
        actual_pnl_pct=float(pnl_pct), hold_duration_mins=int(hold_mins or 0))
    scores = [r[0] for r in conn.execute(
        "SELECT checkpoint_score FROM thesis_checkpoints WHERE decision_id = ? "
        "AND checkpoint_score IS NOT NULL", (trade_id,))]
    return acc, (round(sum(scores) / len(scores), 4) if scores else None)


def fill_thesis_scores(dry_run: bool = False) -> int:
    """Score the entry thesis of every closed trade that has none yet."""
    from kairos_log_db import get_connection
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT t.trade_id, t.pnl_pct, t.hold_duration_mins FROM trades t "
            "LEFT JOIN trade_features tf ON tf.trade_id = t.trade_id "
            "WHERE t.timestamp_exit IS NOT NULL AND t.pnl_pct IS NOT NULL "
            "AND tf.prediction_accuracy IS NULL "
            "AND EXISTS (SELECT 1 FROM thesis_predictions tp WHERE tp.decision_id = t.trade_id)"
        ).fetchall()
        n = 0
        for r in rows:
            acc, ts = _thesis_scores(conn, r["trade_id"], r["pnl_pct"], r["hold_duration_mins"])
            if acc is None:
                continue
            n += 1
            if not dry_run:
                conn.execute(
                    "INSERT INTO trade_features (trade_id, prediction_accuracy, thesis_score, source) "
                    "VALUES (?, ?, ?, 'nightly') ON CONFLICT(trade_id) DO UPDATE SET "
                    "prediction_accuracy = excluded.prediction_accuracy, "
                    "thesis_score = excluded.thesis_score", (r["trade_id"], acc, ts))
        if not dry_run:
            conn.commit()
        return n
    finally:
        conn.close()


def fill_features(window_days: int = 5, only_missing: bool = True,
                  dry_run: bool = False) -> dict:
    """Compute + persist outcome features for closed trades.

    Returns a summary dict. With dry_run=True, prints each trade's computed
    features and writes nothing.
    """
    from kairos_ml_outcomes import FORGONE_HORIZONS, forgone_column
    from kairos_log_db import get_connection

    now = datetime.now(timezone.utc)
    conn = get_connection()
    try:
        todo = _select_rows(conn, window_days, only_missing, now)
    finally:
        conn.close()

    summary = {
        "candidates": len(todo), "written": 0, "skipped": 0,
        "matured": 0, "immature": 0,
    }
    if not todo:
        print("  No closed trades need feature computation.")
        return summary

    # Batch the yfinance download across every candidate (one request).
    entry_dts = [d for d in (_parse_utc(r["timestamp_entry"]) for r in todo) if d]
    exit_dts = [d for d in (_parse_utc(r["timestamp_exit"]) for r in todo) if d]
    tickers = sorted({r["ticker"] for r in todo
                      if r["ticker"] and (r["action"] or "").upper() == "BUY"})
    hist = {}
    if tickers and entry_dts and exit_dts:
        # The download must reach past the LONGEST forgone horizon, not just
        # window_days — otherwise the 60d peak is silently computed from a
        # truncated series and understates forgone gain.
        _span = max(window_days, max(FORGONE_HORIZONS))
        start = (min(entry_dts) - timedelta(days=2)).strftime("%Y-%m-%d")
        end = (max(exit_dts) + timedelta(days=_span + 3)).strftime("%Y-%m-%d")
        print(f"  Fetching daily OHLC for {len(tickers)} ticker(s) {start} → {end} ...")
        hist = _download_daily(tickers, start, end)

    filled_at = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    write_conn = None if dry_run else get_connection()
    try:
        for r in todo:
            feats = compute_features_for_trade(
                r["ticker"], r["timestamp_entry"], r["timestamp_exit"],
                r["price_entry"], r["price_exit"], r["action"], window_days,
                pnl_pct=r["pnl_pct"], _hist=hist, now=now,
            )
            if feats.get("skipped"):
                summary["skipped"] += 1
                print(f"  SKIP {r['ticker']} ({r['trade_id'][:8]}): {feats['skipped']}")
                continue
            if feats["mfe_pct"] is None:
                # No usable in-hold history — leave for a later run.
                summary["skipped"] += 1
                print(f"  SKIP {r['ticker']} ({r['trade_id'][:8]}): no in-hold price rows")
                continue

            if feats["post_exit_peak_pct"] is not None:
                summary["matured"] += 1
            else:
                summary["immature"] += 1

            print(
                f"  {r['ticker']:<6} {r['trade_id'][:8]}  "
                f"mfe={feats['mfe_pct']:+.2f}%  give_back={feats['give_back_pct']:+.2f}%  "
                f"post_exit_peak="
                f"{('%.2f%%' % feats['post_exit_peak_pct']) if feats['post_exit_peak_pct'] is not None else 'pending'}"
                f"  window={'matured' if feats['window_elapsed'] else 'open'}"
            )

            if dry_run:
                continue

            # Forgone horizons are written with COALESCE semantics in reverse:
            # a horizon that has not yet matured is None and must NOT overwrite
            # a value already stored.
            fg = {forgone_column(h): feats.get(forgone_column(h)) for h in FORGONE_HORIZONS}
            any_fg = any(v is not None for v in fg.values())
            cols = ["mfe_pct", "give_back_pct", "post_exit_peak_pct",
                    "post_exit_window_days", "features_filled_at", *fg.keys(),
                    "forgone_filled_at"]
            vals = [feats["mfe_pct"], feats["give_back_pct"], feats["post_exit_peak_pct"],
                    feats["post_exit_window_days"], filled_at, *fg.values(),
                    filled_at if any_fg else None]
            updates = ", ".join(
                f"{c} = COALESCE(excluded.{c}, trade_features.{c})"
                if c in fg or c == "forgone_filled_at" else f"{c} = excluded.{c}"
                for c in cols)
            write_conn.execute(
                f"INSERT INTO trade_features (trade_id, {', '.join(cols)}, source) "
                f"VALUES (?, {', '.join('?' * len(cols))}, 'nightly') "
                f"ON CONFLICT(trade_id) DO UPDATE SET {updates}",
                (r["trade_id"], *vals))
            summary["written"] += 1
        if not dry_run:
            write_conn.commit()
    finally:
        if write_conn is not None:
            write_conn.close()

    verb = "would write" if dry_run else "wrote"
    print(f"\n  {verb} {summary['written'] if not dry_run else summary['candidates']-summary['skipped']} "
          f"row(s); skipped {summary['skipped']}; "
          f"post-exit window matured {summary['matured']} / pending {summary['immature']}.")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kairos nightly trade-record job — Flex pull, rebuild, outcome features")
    parser.add_argument("--backfill", action="store_true",
                        help="Run the nightly job: Flex pull, rebuild_trades, features")
    parser.add_argument("--no-flex", action="store_true",
                        help="Skip step (a), the Flex Web Service pull")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print computed features per trade; write nothing "
                             "(also skips the Flex pull and the rebuild)")
    parser.add_argument("--window", type=int, default=5,
                        help="Post-exit window in days (default: 5)")
    args = parser.parse_args()

    if not args.backfill:
        parser.error("nothing to do — pass --backfill")

    import kairos_ledger
    if not args.dry_run:
        # (a0) Same-day broker check. reqExecutions() only returns TODAY, and
        # Flex only has a day after the next overnight run, so a fill that
        # lands after the last cycle's broker check (LEN 287, placed 16:37 ET
        # 2026-10-05, filled 16:41-16:53 after hours) was invisible to the
        # ledger — and to the exit engine — until the following night.
        # Catching it here at 19:30 closes that gap to a few hours.
        try:
            bc = kairos_ledger.broker_check(client_id=12)
            print(f"  broker_check: new fills={bc.get('new_fills')} "
                  f"mismatches={len(bc.get('mismatches') or [])}")
            from kairos_execute import true_up_order_status_from_fills
            true_up_order_status_from_fills()
        except Exception as exc:
            print(f"  broker_check failed: {exc}")
        # (a) Flex pull — alerts (naming the token) on failure, never raises.
        if not args.no_flex:
            kairos_ledger.nightly_flex_pull()
        # (b) rebuild — runs even if the pull failed: live + broker-check fills
        # still need matching, and the rebuild is deterministic.
        try:
            print(f"  rebuild_trades: {kairos_ledger.rebuild_trades()}")
        except Exception as exc:
            kairos_ledger.alert(f":rotating_light: *Nightly rebuild_trades FAILED*: {exc}")

    # (c) features
    fill_features(window_days=args.window, only_missing=True, dry_run=args.dry_run)
    try:
        n = fill_thesis_scores(dry_run=args.dry_run)
        if n:
            print(f"  thesis scores: {n} closed trade(s) "
                  f"{'would be ' if args.dry_run else ''}scored")
    except Exception as exc:
        print(f"  thesis scoring failed: {exc}")

    # (d) ML retrain. train_model() otherwise loads the pickle from disk
    # forever, so the model never learned from new closes unless someone
    # retrained it by hand. Retrain nightly on the rebuilt ledger; the
    # council prompt (kairos_reason SECTION 5b) reads the fresh CV accuracy
    # and tells the council to discount the scores while it is below 55%,
    # so influence arrives automatically once measured skill does.
    if not args.dry_run:
        try:
            import kairos_ml
            info = kairos_ml.train_model(force_retrain=True) or {}
            print(f"  ml retrain: n={info.get('trade_count')} "
                  f"cv_accuracy={info.get('accuracy')}")
        except Exception as exc:
            print(f"  ml retrain failed: {exc}")
            try:
                kairos_ledger.alert(f":warning: *Nightly ML retrain FAILED*: {exc}")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
