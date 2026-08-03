"""
Kairos Outcome Features — B2a: persist closed-trade outcome features.

Behaviour-neutral, post-hoc enrichment. Computes max-favorable-excursion (MFE),
give-back, and post-exit run-up for CLOSED long trades and writes them into the
trade_outcomes feature columns (added idempotently in kairos_ml_outcomes.py):

    mfe_pct                — (max High in [entry, exit] − entry)/entry × 100
    give_back_pct          — mfe_pct − pnl_pct  (points of the in-hold peak surrendered)
    post_exit_peak_pct     — (max High in (exit, exit+window] − exit)/exit × 100
                             ONLY once the full window has elapsed; else NULL
    post_exit_window_days  — the window used
    exit_reason            — trigger copied from kairos.db position_exits_history
    features_filled_at     — set when mfe/give-back are computed

This does NOT touch write_trade_close, the scheduler, kairos_reason.py, or any
live trading path. Run it on demand / on a cadence to backfill features.

Price fetch REUSES the Arbiter's batched yfinance helpers (kairos_arbiter.
_download_daily / _parse_utc). The MFE / give-back / post-exit MATH below is a
near-duplicate of kairos_arbiter.enrich_closed_trades for now.
    TODO(dedupe): once this module is the single source of truth for outcome
    features, refactor kairos_arbiter.enrich_closed_trades to call
    compute_features_for_trade() here instead of re-deriving the same metrics.

CLI:
    python3 kairos_outcome_features.py --backfill            # fill all missing
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


# ── exit_reason lookup (kairos.db position_exits_history) ────────────

def _load_exit_reasons() -> dict:
    """{ticker: [{exit_date(dt), exit_reason}, ...]} from position_exits_history.

    The append-only history gives multiple exits per ticker; _match_exit_reason
    picks the one nearest each trade's exit timestamp (Δt match).
    """
    from kairos_log_db import get_connection
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT ticker, exit_date, exit_reason FROM position_exits_history"
        ).fetchall()
    finally:
        conn.close()
    out: dict = defaultdict(list)
    for r in rows:
        out[r["ticker"]].append({
            "exit_date": _parse_utc(r["exit_date"]),
            "exit_reason": r["exit_reason"],
        })
    return out


def _match_exit_reason(reasons: dict, ticker: str, exit_dt) -> str | None:
    """The exit_reason for this ticker nearest the trade's exit timestamp.

    position_exits_history holds a row per exit, so a re-traded ticker has
    several candidates; the nearest-by-date match attributes each trade to its
    own close instead of the latest.
    """
    cands = reasons.get(ticker) or []
    if not cands:
        return None
    if exit_dt is None:
        return cands[0]["exit_reason"]
    best, best_diff = None, None
    for c in cands:
        if c["exit_date"] is None:
            continue
        diff = abs((c["exit_date"] - exit_dt).total_seconds())
        if best_diff is None or diff < best_diff:
            best, best_diff = c["exit_reason"], diff
    return best if best is not None else cands[0]["exit_reason"]


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


def fill_features(window_days: int = 5, only_missing: bool = True,
                  dry_run: bool = False) -> dict:
    """Compute + persist outcome features for closed trades.

    Returns a summary dict. With dry_run=True, prints each trade's computed
    features and writes nothing.
    """
    from kairos_ml_outcomes import init_db, get_connection
    init_db()  # ensure the feature columns exist (idempotent)

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
        from kairos_ml_outcomes import FORGONE_HORIZONS
        _span = max(window_days, max(FORGONE_HORIZONS))
        start = (min(entry_dts) - timedelta(days=2)).strftime("%Y-%m-%d")
        end = (max(exit_dts) + timedelta(days=_span + 3)).strftime("%Y-%m-%d")
        print(f"  Fetching daily OHLC for {len(tickers)} ticker(s) {start} → {end} ...")
        hist = _download_daily(tickers, start, end)

    reasons = _load_exit_reasons()
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

            exit_dt = _parse_utc(r["timestamp_exit"])
            exit_reason = _match_exit_reason(reasons, r["ticker"], exit_dt)
            if feats["post_exit_peak_pct"] is not None:
                summary["matured"] += 1
            else:
                summary["immature"] += 1

            print(
                f"  {r['ticker']:<6} {r['trade_id'][:8]}  "
                f"mfe={feats['mfe_pct']:+.2f}%  give_back={feats['give_back_pct']:+.2f}%  "
                f"post_exit_peak="
                f"{('%.2f%%' % feats['post_exit_peak_pct']) if feats['post_exit_peak_pct'] is not None else 'pending'}"
                f"  exit_reason={(exit_reason or 'n/a')[:32]}"
                f"  window={'matured' if feats['window_elapsed'] else 'open'}"
            )

            if dry_run:
                continue

            # Forgone horizons are written with COALESCE semantics in reverse:
            # a horizon that has not yet matured is None and must NOT overwrite
            # a value already stored, so only non-None horizons are updated.
            from kairos_ml_outcomes import FORGONE_HORIZONS, forgone_column
            fg_cols, fg_vals = [], []
            for h in FORGONE_HORIZONS:
                col = forgone_column(h)
                if feats.get(col) is not None:
                    fg_cols.append(f"  {col} = ?")
                    fg_vals.append(feats[col])
            fg_sql = ("," + ",".join(fg_cols) + ", forgone_filled_at = ?") if fg_cols else ""
            if fg_cols:
                fg_vals.append(filled_at)

            write_conn.execute(
                "UPDATE trade_outcomes SET "
                "  mfe_pct = ?, give_back_pct = ?, post_exit_peak_pct = ?, "
                "  post_exit_window_days = ?, exit_reason = ?, features_filled_at = ?"
                + fg_sql +
                " WHERE trade_id = ?",
                (
                    feats["mfe_pct"], feats["give_back_pct"],
                    feats["post_exit_peak_pct"], feats["post_exit_window_days"],
                    exit_reason, filled_at, *fg_vals, r["trade_id"],
                ),
            )
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
        description="Kairos outcome features — backfill MFE/give-back/post-exit (B2a)")
    parser.add_argument("--backfill", action="store_true",
                        help="Compute + persist features for all closed trades missing them")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print computed features per trade; write nothing")
    parser.add_argument("--window", type=int, default=5,
                        help="Post-exit window in days (default: 5)")
    args = parser.parse_args()

    if not args.backfill:
        parser.error("nothing to do — pass --backfill")

    fill_features(window_days=args.window, only_missing=True, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
