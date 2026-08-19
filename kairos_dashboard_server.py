#!/usr/bin/env python3
"""
Kairos Dashboard Server — Flask app on port 5001

Serves the dashboard HTML and provides a /refresh endpoint that
regenerates kairos_dashboard.html by running kairos_dashboard.py,
then redirects back to the dashboard.

Endpoints:
    GET  /          — Serve kairos_dashboard.html
    POST /refresh   — Regenerate dashboard, redirect to /
    GET  /api/status — JSON with last_updated timestamp and refresh state

Usage:
    python3 kairos_dashboard_server.py              # Start on port 5001
    python3 kairos_dashboard_server.py --port 5002  # Custom port
"""

import argparse
import json
import os
import subprocess
import kairos_spawn
import sys
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, redirect, send_file

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

DASHBOARD_HTML = os.path.join(SCRIPT_DIR, "kairos_dashboard.html")
DASHBOARD_PY = os.path.join(SCRIPT_DIR, "kairos_dashboard.py")
VENV_PYTHON = os.path.join(os.path.expanduser("~"), "Kairos-env", "bin", "python3")

app = Flask(__name__)

# Shared state for tracking refresh status
_refresh_lock = threading.Lock()
_refresh_state = {
    "in_progress": False,
    "last_updated": None,
    "last_error": None,
}


def _get_last_updated() -> str | None:
    """Return the mtime of kairos_dashboard.html as an ISO timestamp."""
    if os.path.exists(DASHBOARD_HTML):
        mtime = os.path.getmtime(DASHBOARD_HTML)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    return None


def _run_dashboard_refresh():
    """Run kairos_dashboard.py as a subprocess to regenerate the HTML."""
    with _refresh_lock:
        if _refresh_state["in_progress"]:
            return
        _refresh_state["in_progress"] = True
        _refresh_state["last_error"] = None

    python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

    try:
        # posix_spawn, not subprocess.run — this is a threaded Flask server
        # with live sockets; forking it can SIGSEGV in Network.framework's
        # atfork child handler pre-exec. See kairos_spawn.
        result = kairos_spawn.run(
            [python, DASHBOARD_PY],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=SCRIPT_DIR,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()[-300:]
            _refresh_state["last_error"] = err or f"Exit code {result.returncode}"
            print(f"[refresh] ERROR: exit code {result.returncode}")
            if err:
                print(f"[refresh] {err}")
        else:
            _refresh_state["last_updated"] = _get_last_updated()
            print(f"[refresh] Dashboard regenerated at {_refresh_state['last_updated']}")
    except subprocess.TimeoutExpired:
        _refresh_state["last_error"] = "Timed out after 120s"
        print("[refresh] ERROR: timed out")
    except Exception as exc:
        _refresh_state["last_error"] = str(exc)
        print(f"[refresh] ERROR: {exc}")
    finally:
        _refresh_state["in_progress"] = False


@app.route("/")
def index():
    """Serve the dashboard HTML file."""
    if not os.path.exists(DASHBOARD_HTML):
        return (
            "<h1>Dashboard not generated yet</h1>"
            '<p>Click <a href="/refresh">here</a> to generate it, '
            "or run <code>python3 kairos_dashboard.py</code></p>"
        ), 404
    return send_file(DASHBOARD_HTML, mimetype="text/html")


@app.route("/refresh", methods=["GET", "POST"])
def refresh():
    """Regenerate the dashboard and redirect to /."""
    _run_dashboard_refresh()
    return redirect("/")


@app.route("/api/status")
def api_status():
    """JSON endpoint for the refresh button's JS polling."""
    return jsonify({
        "in_progress": _refresh_state["in_progress"],
        "last_updated": _refresh_state["last_updated"] or _get_last_updated(),
        "last_error": _refresh_state["last_error"],
    })


def main():
    parser = argparse.ArgumentParser(description="Kairos Dashboard Server")
    parser.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    # Initialize last_updated from existing file
    _refresh_state["last_updated"] = _get_last_updated()

    print(f"Kairos Dashboard Server starting on http://{args.host}:{args.port}")
    print(f"  Dashboard file: {DASHBOARD_HTML}")
    print(f"  Python: {VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
