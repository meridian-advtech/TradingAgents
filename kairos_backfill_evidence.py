"""
Kairos evidence backfill — 2026-07 ratchet-postmortem data layer.

Backfills the two-sided-objective evidence the redesigned learning loop needs.
DRY-RUN BY DEFAULT: prints per-row computed values + counts and writes nothing.
Pass --apply to write (a timestamped DB copy is taken first).

  --features            1b/1c: mfe_pct + forgone_gain_5d_pct via the batched
                        yfinance enricher (kairos_outcome_features.fill_features).
  --snapshots           1d: exit_params_snapshot (regime tag) reconstructed from
                        axis_weight_history for historical closed rows.

    python3 kairos_backfill_evidence.py --features            # dry-run
    python3 kairos_backfill_evidence.py --features --apply
    python3 kairos_backfill_evidence.py --snapshots           # dry-run
    python3 kairos_backfill_evidence.py --snapshots --apply
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kairos_ml_outcomes import (
    DB_PATH, KAIROS_DB_PATH, SNAPSHOT_PARAM_PATHS,
    build_exit_params_snapshot, _get_dotted,
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _backup_db(path: str) -> str:
    dst = f"{path}.bak_{_ts()}"
    shutil.copy2(path, dst)
    print(f"  DB backup: {path} -> {dst}")
    return dst


# ── 1b/1c: features (mfe + forgone_gain_5d) ──────────────────────────

def backfill_features(apply: bool) -> None:
    from kairos_outcome_features import fill_features
    print("\n=== 1b/1c  mfe_pct + forgone_gain_5d_pct backfill "
          f"({'APPLY' if apply else 'DRY-RUN'}) ===")
    if apply:
        _backup_db(DB_PATH)
    summary = fill_features(window_days=5, only_missing=True, dry_run=not apply)
    print(f"  summary: {summary}")


# ── 1d: exit_params_snapshot reconstruction ──────────────────────────

def _parse_dt(s):
    if not s:
        return None
    s = s.strip()
    # Strip zone suffixes BEFORE touching the ISO 'T' separator — otherwise
    # replace("T"," ") corrupts the "T" inside "UTC".
    if s.endswith(" UTC"):
        s = s[:-4]
    s = s.rstrip("Z").replace("T", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _current_param_values() -> dict:
    """Live whitelisted param values from kairos_config.json."""
    cfg = {}
    try:
        with open(os.path.join(SCRIPT_DIR, "kairos_config.json")) as f:
            cfg = json.load(f) or {}
    except (json.JSONDecodeError, IOError):
        pass
    out = {}
    for path in SNAPSHOT_PARAM_PATHS:
        val, found = _get_dotted(cfg, path)
        out[path] = val if found else None
    return out


def _load_approved_history() -> list:
    """Approved axis_weight_history rows, oldest→newest by decided_at.

    Covers both param:* rows (exit-engine params) and plain axis rows
    (exit_timing etc.), so a trade's exit date can be mapped to the value in
    force for every regime dimension.
    """
    conn = sqlite3.connect(f"file:{KAIROS_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT axis, new_weight, prior_weight, decided_at, status "
            "FROM axis_weight_history WHERE status = 'approved'"
        ).fetchall()
    finally:
        conn.close()
    hist = []
    for r in rows:
        dt = _parse_dt(r["decided_at"])
        if dt is None:
            continue
        hist.append({"axis": r["axis"], "new_weight": r["new_weight"],
                     "prior_weight": r["prior_weight"], "dt": dt})
    hist.sort(key=lambda h: h["dt"])
    return hist


def _value_in_force(axis: str, exit_dt, hist: list, fallback):
    """Value of `axis` in force at exit_dt = most recent approved change
    decided at/before exit_dt; else the earliest known prior_weight; else fallback.
    """
    applicable = [h for h in hist if h["axis"] == axis and exit_dt and h["dt"] <= exit_dt]
    if applicable:
        return applicable[-1]["new_weight"]
    earliest = next((h for h in hist if h["axis"] == axis), None)
    if earliest is not None and earliest["prior_weight"] is not None:
        return earliest["prior_weight"]
    return fallback


def _current_axis_weights() -> dict:
    conn = sqlite3.connect(f"file:{KAIROS_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {r["axis"]: r["weight"] for r in conn.execute(
            "SELECT axis, weight FROM axis_weights WHERE status = 'active'")}
    finally:
        conn.close()


def backfill_snapshots(apply: bool) -> None:
    print("\n=== 1d  exit_params_snapshot reconstruction "
          f"({'APPLY' if apply else 'DRY-RUN'}) ===")
    cur_params = _current_param_values()
    cur_weights = _current_axis_weights()
    hist = _load_approved_history()
    print(f"  current live params: {cur_params}")
    print(f"  current axis weights: { {k: round(v,4) for k,v in cur_weights.items()} }")
    print(f"  approved history rows: {len(hist)} "
          f"(param:* = {sum(1 for h in hist if h['axis'].startswith('param:'))}, "
          f"weights = {sum(1 for h in hist if not h['axis'].startswith('param:'))})")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT trade_id, ticker, timestamp_exit, exit_params_snapshot "
        "FROM trade_outcomes WHERE timestamp_exit IS NOT NULL "
        "AND exit_params_snapshot IS NULL ORDER BY timestamp_exit"
    ).fetchall()
    print(f"  candidates (closed, snapshot NULL): {len(rows)}")

    if apply:
        conn.close()
        _backup_db(DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

    written = 0
    sample_shown = 0
    for r in rows:
        exit_dt = _parse_dt(r["timestamp_exit"])
        param_overrides = {
            path: _value_in_force("param:" + path, exit_dt, hist, cur_params.get(path))
            for path in SNAPSHOT_PARAM_PATHS
        }
        weight_overrides = {
            axis: _value_in_force(axis, exit_dt, hist, w)
            for axis, w in cur_weights.items()
        }
        snap = build_exit_params_snapshot(
            reconstructed=True, param_overrides=param_overrides,
            weight_overrides=weight_overrides,
            as_of=(r["timestamp_exit"] or None),
        )
        if sample_shown < 8:
            print(f"    {r['ticker']:<6} {r['trade_id'][:8]}  exit={r['timestamp_exit']}  "
                  f"params={param_overrides}  exit_timing="
                  f"{round(weight_overrides.get('exit_timing', 0), 4)}")
            sample_shown += 1
        if apply:
            conn.execute("UPDATE trade_outcomes SET exit_params_snapshot = ? WHERE trade_id = ?",
                         (json.dumps(snap), r["trade_id"]))
            written += 1
    if apply:
        conn.commit()
    conn.close()
    verb = "wrote" if apply else "would write"
    print(f"  {verb} {written if apply else len(rows)} reconstructed snapshot(s).")


def main() -> int:
    p = argparse.ArgumentParser(description="Kairos evidence backfill (dry-run by default)")
    p.add_argument("--features", action="store_true", help="1b/1c mfe + forgone_gain_5d")
    p.add_argument("--snapshots", action="store_true", help="1d exit_params_snapshot reconstruction")
    p.add_argument("--apply", action="store_true", help="write to DB (default: dry-run)")
    args = p.parse_args()
    if not (args.features or args.snapshots):
        p.error("nothing to do — pass --features and/or --snapshots")
    if args.features:
        backfill_features(args.apply)
    if args.snapshots:
        backfill_snapshots(args.apply)
    if not args.apply:
        print("\n  DRY-RUN complete — nothing written. Re-run with --apply to persist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
