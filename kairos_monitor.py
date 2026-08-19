"""
Kairos Health Monitor — run wrapper with structured logging.

Wraps kairos_run.py --gather, writes a JSON record to kairos_monitor.log
for every pipeline run: timestamp, phases attempted, success/failure, duration,
and asset_class (equity / crypto / both).

Usage (called by the scheduler):
  python kairos_monitor.py [--gather] [--mode equity|crypto|both] [--no-ollama]

Standalone health check (prints last N runs):
  python kairos_monitor.py --status [--last N]
  python kairos_monitor.py --status --asset-class crypto
"""

import argparse
import json
import os
import subprocess
import kairos_spawn
import sys
import traceback
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# macOS fork-safety belt-and-braces. Every subprocess call in this process now
# goes through kairos_spawn (posix_spawn, which runs no atfork handlers), so
# this is not the primary defense — but any third-party library that forks
# internally (joblib/loky, multiprocessing, ProcessPoolExecutor) bypasses our
# code entirely, and that is exactly how kairos_ml's n_jobs=-1 crashed
# kairos_run.py 16 times on 2026-08-19. Set before any networking library
# imports, and set HERE rather than inherited, because this is its own process
# with its own environment. See kairos_spawn for the full crash signature.
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

MONITOR_LOG = os.path.join(SCRIPT_DIR, "kairos_monitor.log")
IBKR_STATE_FILE = os.path.join(SCRIPT_DIR, ".ibkr_last_ok")
MAX_LOG_LINES = 5000  # rotate after this many lines


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_record(record: dict):
    """Append a JSON record to the monitor log (one record per line)."""
    line = json.dumps(record) + "\n"

    # Simple rotation: if log exceeds MAX_LOG_LINES, keep the last half
    if os.path.exists(MONITOR_LOG):
        with open(MONITOR_LOG, "r") as f:
            lines = f.readlines()
        if len(lines) >= MAX_LOG_LINES:
            keep = lines[MAX_LOG_LINES // 2:]
            with open(MONITOR_LOG, "w") as f:
                f.writelines(keep)

    with open(MONITOR_LOG, "a") as f:
        f.write(line)


def check_ibkr_connection() -> dict:
    """Probe IB Gateway and log connection status.

    Writes an IBKR_HEALTH record to the monitor log every call.
    Maintains .ibkr_last_ok with the timestamp of the last successful
    connection so dropped-connection records can report the gap.
    """
    ts = _now_iso()
    record = {
        "timestamp": ts,
        "event": "IBKR_HEALTH",
        "connected": False,
        "last_ok": None,
        "downtime_s": None,
        "error": None,
    }

    # Read last-known-good timestamp
    last_ok = None
    if os.path.exists(IBKR_STATE_FILE):
        try:
            last_ok = open(IBKR_STATE_FILE).read().strip()
            record["last_ok"] = last_ok
        except IOError:
            pass

    try:
        sys.path.insert(0, SCRIPT_DIR)
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=4, timeout=8)

        # Quick sanity: pull one account value to confirm the session is live
        tags = {v.tag: v.value for v in ib.accountSummary()
                if v.tag == "NetLiquidation"}
        ib.disconnect()

        if not tags:
            record["error"] = "Connected but accountSummary returned no data"
        else:
            record["connected"] = True
            record["nlv"] = float(tags["NetLiquidation"])
            # Update last-ok file
            with open(IBKR_STATE_FILE, "w") as f:
                f.write(ts)

    except Exception as exc:
        record["error"] = str(exc)[:200]
        # Compute downtime gap
        if last_ok:
            try:
                last_dt = datetime.strptime(last_ok, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc)
                gap = (datetime.now(timezone.utc) - last_dt).total_seconds()
                record["downtime_s"] = round(gap)
            except ValueError:
                pass

    _append_record(record)
    return record


def print_ibkr_history(last_n: int = 30):
    """Print recent IBKR_HEALTH records to diagnose connection patterns."""
    if not os.path.exists(MONITOR_LOG):
        print("  No monitor log found.")
        return

    records = []
    with open(MONITOR_LOG) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("event") == "IBKR_HEALTH":
                    records.append(r)
            except json.JSONDecodeError:
                pass

    if not records:
        print("  No IBKR_HEALTH records found. Run with --ibkr-check first.")
        return

    recent = records[-last_n:]
    ok = sum(1 for r in recent if r.get("connected"))
    fail = len(recent) - ok

    W = 72
    print("╔" + "═" * W + "╗")
    print(f"║  IB GATEWAY HEALTH — Last {len(recent)} checks".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")
    print(f"  Connected: {ok}  Dropped: {fail}")

    # Find drop patterns — group by hour-of-day
    drops_by_hour: dict[int, int] = {}
    for r in records:
        if not r.get("connected"):
            try:
                h = int(r["timestamp"][11:13])
                drops_by_hour[h] = drops_by_hour.get(h, 0) + 1
            except (KeyError, ValueError):
                pass
    if drops_by_hour:
        peak_hour = max(drops_by_hour, key=drops_by_hour.get)
        print(f"\n  Drop frequency by hour (UTC):")
        for h in sorted(drops_by_hour):
            bar = "█" * drops_by_hour[h]
            marker = " ◄ peak" if h == peak_hour else ""
            print(f"    {h:02d}:00  {drops_by_hour[h]:>3}  {bar}{marker}")

    print(f"\n  {'Timestamp':<22} {'Status':<8} {'NLV':>14} {'Downtime':>10} Error")
    print("  " + "─" * 70)

    for r in recent:
        ts = r.get("timestamp", "?")[:19].replace("T", " ")
        status = "✓ UP" if r.get("connected") else "✗ DOWN"
        nlv = f"${r['nlv']:>12,.0f}" if r.get("nlv") else f"{'—':>13}"
        down = ""
        if r.get("downtime_s") is not None:
            mins = r["downtime_s"] // 60
            if mins >= 60:
                down = f"{mins // 60}h{mins % 60:02d}m"
            else:
                down = f"{mins}m"
        err = (r.get("error") or "")[:25]
        print(f"  {ts:<22} {status:<8} {nlv} {down:>10} {err}")

    print()


def run_pipeline(
    gather_only: bool = True,
    no_ollama: bool = False,
    mode: str = "equity",
) -> dict:
    """Run the pipeline as a subprocess, return a status record.

    Args:
        gather_only: If True, run only Phase 1 (and 1C for crypto modes).
        no_ollama:   If True, skip Ollama phase.
        mode:        "equity", "crypto", or "both" — determines asset classes.
    """
    import time

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "kairos_run.py")]
    if gather_only:
        cmd.append("--gather")
    if no_ollama:
        cmd.append("--no-ollama")
    cmd.extend(["--mode", mode])

    # Build phases label for the record
    if gather_only:
        base_phases = ["gather"] if mode in ("equity", "both") else []
        crypto_phases = ["gather_crypto"] if mode in ("crypto", "both") else []
        phases = base_phases + crypto_phases
    else:
        phases = ["orchestrate", "gather", "reason", "execute"]
        if mode in ("crypto", "both"):
            phases.insert(2, "gather_crypto")
        if no_ollama:
            phases = [p for p in phases if p != "orchestrate"]

    # Map mode to asset_class label stored in the record
    asset_class_map = {"equity": "equity", "crypto": "crypto", "both": "equity+crypto"}
    asset_class = asset_class_map.get(mode, "equity")

    record: dict = {
        "timestamp": _now_iso(),
        "phases": phases,
        "asset_class": asset_class,
        "mode": mode,
        "success": False,
        "exit_code": None,
        "duration_s": None,
        "error": None,
    }

    t0 = time.time()
    try:
        # Load API keys from .zshrc environment via login shell
        env = dict(os.environ)
        # posix_spawn, not subprocess.run — see kairos_spawn. This module is
        # not wired into anything as of 2026-08-19, but it is a cycle
        # supervisor that wraps kairos_run.py, so whenever it IS wired in it
        # would be forking exactly the kind of process that crashes. Hardened
        # now so re-enabling it does not silently reintroduce the bug.
        result = kairos_spawn.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute hard limit (both mode: equity screen ~7min + crypto ~2min)
            env=env,
            cwd=SCRIPT_DIR,
        )
        elapsed = round(time.time() - t0, 1)
        record["exit_code"] = result.returncode
        record["duration_s"] = elapsed
        record["success"] = result.returncode == 0

        if result.returncode != 0:
            # Capture last 10 lines of stderr/stdout for diagnosis
            combined = (result.stdout + result.stderr).strip()
            tail = "\n".join(combined.splitlines()[-10:])
            record["error"] = tail or "non-zero exit code"
        else:
            record["stdout_tail"] = "\n".join(
                result.stdout.strip().splitlines()[-5:]
            )

    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - t0, 1)
        record["duration_s"] = elapsed
        record["error"] = f"TIMEOUT after {round(time.time() - t0)}s"
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        record["duration_s"] = elapsed
        record["error"] = traceback.format_exc(limit=3)

    return record


def print_status(last_n: int = 20, asset_class_filter: str | None = None):
    """Print the last N run records in a human-readable table.

    Args:
        last_n:              How many recent records to show.
        asset_class_filter:  If set, only show records matching this asset class
                             ("equity", "crypto", "equity+crypto").
    """
    if not os.path.exists(MONITOR_LOG):
        print("  No monitor log found. Run the pipeline first.")
        return

    with open(MONITOR_LOG, "r") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    if not records:
        print("  Monitor log exists but contains no valid records.")
        return

    # Apply optional asset class filter
    if asset_class_filter:
        filtered = [r for r in records if r.get("asset_class") == asset_class_filter]
        filter_label = f" [{asset_class_filter}]"
    else:
        filtered = records
        filter_label = ""

    recent = filtered[-last_n:]
    ok_count = sum(1 for r in recent if r.get("success"))
    fail_count = len(recent) - ok_count

    W = 72
    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS MONITOR — Last {len(recent)} runs{filter_label}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")
    print(f"  OK: {ok_count}  FAIL: {fail_count}")

    # Per-class breakdown (only when showing all)
    if not asset_class_filter and records:
        classes = {}
        for r in records[-50:]:  # last 50 for the breakdown
            ac = r.get("asset_class", "equity")
            if ac not in classes:
                classes[ac] = {"ok": 0, "fail": 0}
            if r.get("success"):
                classes[ac]["ok"] += 1
            else:
                classes[ac]["fail"] += 1
        print()
        print("  Breakdown by asset class (last 50 runs):")
        for ac, counts in sorted(classes.items()):
            total = counts["ok"] + counts["fail"]
            rate = counts["ok"] / total * 100 if total else 0
            print(f"    {ac:<16} {total:>4} runs  {counts['ok']:>3} OK  {counts['fail']:>3} FAIL  ({rate:.0f}% success)")

    print()
    print(f"  {'Timestamp':<22} {'Asset Class':<14} {'Phases':<18} {'OK':<5} {'Dur(s)':<8} Notes")
    print("  " + "─" * 75)

    for r in recent:
        ts = r.get("timestamp", "?")[:19].replace("T", " ")
        ac = r.get("asset_class", "equity")[:12]
        phases = ",".join(r.get("phases", []))[:16]
        ok = "✓" if r.get("success") else "✗"
        dur = str(r.get("duration_s", "?"))
        err = r.get("error", "")
        note = err.splitlines()[0][:20] if err else r.get("stdout_tail", "")[:20].replace("\n", " ")
        print(f"  {ts:<22} {ac:<14} {phases:<18} {ok:<5} {dur:<8} {note}")

    print()

    # Flag if last relevant run failed
    if recent:
        last = recent[-1]
        if not last.get("success"):
            print(f"  ⚠ LAST{filter_label} RUN FAILED:")
            print(f"     {last.get('error', 'unknown error')[:200]}")
        else:
            print(f"  ✓ Last{filter_label} run succeeded.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Kairos Health Monitor")
    parser.add_argument("--gather",       action="store_true", help="Run gather phase only (default for scheduler)")
    parser.add_argument("--full",         action="store_true", help="Run full pipeline (gather+reason+execute)")
    parser.add_argument("--no-ollama",    action="store_true", help="Skip Ollama phase")
    parser.add_argument("--status",       action="store_true", help="Print recent run history")
    parser.add_argument("--last",         type=int, default=20, help="With --status: how many runs to show")
    parser.add_argument("--ibkr-check",   action="store_true", help="Probe IB Gateway and log connection status")
    parser.add_argument("--ibkr-history", action="store_true", help="Show IB Gateway connection history and drop patterns")
    parser.add_argument(
        "--mode",
        choices=["equity", "crypto", "both"],
        default="equity",
        help="Asset mode passed to kairos_run.py (equity/crypto/both)",
    )
    parser.add_argument(
        "--asset-class",
        dest="asset_class",
        default=None,
        help="With --status: filter display to equity / crypto / equity+crypto",
    )
    args = parser.parse_args()

    if args.status:
        print_status(args.last, asset_class_filter=args.asset_class)
        return

    if args.ibkr_history:
        print_ibkr_history(args.last)
        return

    if args.ibkr_check:
        r = check_ibkr_connection()
        status = "✓ Connected" if r["connected"] else "✗ Disconnected"
        print(f"[{r['timestamp']}] IBKR: {status}")
        if r.get("nlv"):
            print(f"  NLV: ${r['nlv']:,.2f}")
        if r.get("error"):
            print(f"  Error: {r['error']}")
        if r.get("downtime_s") is not None:
            print(f"  Last OK: {r['last_ok']}  (down ~{r['downtime_s'] // 60}m)")
        return

    # Run IBKR health check before every pipeline cycle
    ibkr_health = check_ibkr_connection()
    if not ibkr_health["connected"]:
        print(f"  ⚠ IBKR connection down: {ibkr_health.get('error', 'unknown')}")
        if ibkr_health.get("last_ok"):
            print(f"  Last successful connection: {ibkr_health['last_ok']}")

    gather_only = not args.full
    record = run_pipeline(gather_only=gather_only, no_ollama=args.no_ollama, mode=args.mode)
    record["ibkr_connected"] = ibkr_health["connected"]
    _append_record(record)

    # Print one-line summary to stdout (captured by scheduler logs)
    status_str = "OK" if record["success"] else "FAIL"
    dur = record.get("duration_s", "?")
    asset_class = record.get("asset_class", "equity")
    err = ""
    if not record["success"]:
        err = " — " + str(record.get("error", ""))[:80].replace("\n", " ")
    print(f"[{record['timestamp']}] {status_str} ({dur}s) [{asset_class}]{err}")

    if not record["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
