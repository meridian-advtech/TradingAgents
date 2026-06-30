#!/bin/bash
DAY=$(TZ=America/New_York date +%u)
HOUR=$(TZ=America/New_York date +%H)
MIN=$(TZ=America/New_York date +%M)
TIME=$((10#$HOUR * 100 + 10#$MIN))
[ "$DAY" -ge 6 ] && exit 0
[ "$TIME" -lt 925 ] && exit 0
[ "$TIME" -gt 1630 ] && exit 0
# ─────────────────────────────────────────────────────────────────────────────
# Kairos Multi-Asset Scheduler — Phase B (Crypto + Equities)
#
# Trading windows (Eastern Time):
#
#   00:00 – 00:05  closed            (IB Gateway nightly restart grace period)
#   00:05 – 09:30  crypto-only       (24/7 crypto, pre-market)
#   09:30 – 16:00  equity+crypto     (US market hours, weekdays only)
#   16:00 – 23:59  crypto-only       (24/7 crypto, post-market)
#   Weekends       crypto-only       (24/7 crypto, no equities)
#
# Returns one of three modes to the pipeline:
#   both         — equities AND crypto
#   crypto       — crypto only
#   closed       — nothing runs (script exits 0)
#
# Designed to be invoked by launchd every 30 minutes.
# Runs the FULL pipeline: screen → gather → reason → execute.
# All output is logged to kairos_scheduler.log.
#
# Usage (manual test):
#   bash /Users/jelmore/TradingAgents/kairos_scheduler.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

KAIROS_DIR="/Users/jelmore/TradingAgents"
VENV_ACTIVATE="/Users/jelmore/Kairos-env/bin/activate"
SCHEDULER_LOG="${KAIROS_DIR}/kairos_scheduler.log"

# ── Logging ──────────────────────────────────────────────────────────────────

log() {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$SCHEDULER_LOG"
}

# Rotate log if it exceeds 5MB
rotate_log() {
    if [ -f "$SCHEDULER_LOG" ] && [ "$(wc -c < "$SCHEDULER_LOG")" -gt 5242880 ]; then
        mv "$SCHEDULER_LOG" "${SCHEDULER_LOG}.old"
        log "Log rotated (exceeded 5MB)"
    fi
}

# ── Trading window detection ──────────────────────────────────────────────────
# Returns one of: "equity" | "closed"
# Crypto disabled — managed separately via Robinhood.
# All times in Eastern Time (TZ=America/New_York handles DST automatically).

get_trading_mode() {
    local dow hhmm
    dow=$(TZ="America/New_York" date '+%u')    # 1=Mon … 5=Fri, 6=Sat, 7=Sun
    hhmm=$(TZ="America/New_York" date '+%H%M') # 24h, e.g. 0930 or 2145

    # Equity window: 09:30–16:00 ET, weekdays only
    if [ "$dow" -lt 6 ] && [ "$hhmm" -ge 930 ] && [ "$hhmm" -lt 1600 ]; then
        echo "equity"
    else
        echo "closed"
    fi
}

# ── Dashboard refresh (once daily, at or after 16:15 ET on weekdays) ──────────
# Uses a date-stamped lock file so it fires exactly once per day regardless of
# how many 30-minute scheduler slots fall after 4:15pm.

run_dashboard_if_needed() {
    local et_hhmm et_date lock_file last_run
    et_hhmm=$(TZ="America/New_York" date '+%H%M')
    et_date=$(TZ="America/New_York" date '+%Y-%m-%d')
    lock_file="${KAIROS_DIR}/.dashboard_last_run"
    last_run=""

    # Only run at 16:15 ET or later (post market-close)
    if [ "$et_hhmm" -lt 1615 ]; then
        return 0
    fi

    # Read last-run date; skip if already refreshed today
    [ -f "$lock_file" ] && last_run=$(cat "$lock_file")
    if [ "$last_run" = "$et_date" ]; then
        log "DASH — dashboard already refreshed today (${et_date}), skipping"
        return 0
    fi

    log "DASH — refreshing performance dashboard (${et_date} ${et_hhmm} ET)"
    if python3 "${KAIROS_DIR}/kairos_dashboard.py"; then
        echo "$et_date" > "$lock_file"
        log "DASH — dashboard written → kairos_dashboard.html"
    else
        log "DASH — dashboard generation failed (exit $?)"
    fi
}

# ── End-of-day Slack summary (once per window per day) ────────────────────────
# Equity close: 16:15 ET on weekdays
# Crypto close: 22:05 ET daily (after crypto window ends)

run_eod_summary_if_needed() {
    local et_hhmm et_date dow lock_file last_run window
    et_hhmm=$(TZ="America/New_York" date '+%H%M')
    et_date=$(TZ="America/New_York" date '+%Y-%m-%d')
    dow=$(TZ="America/New_York" date '+%u')  # 1=Mon..5=Fri, 6=Sat, 7=Sun

    # Equity close: 16:15+ ET, weekdays only
    if [ "$dow" -lt 6 ] && [ "$et_hhmm" -ge 1615 ]; then
        lock_file="${KAIROS_DIR}/.eod_equity_last_run"
        last_run=""
        [ -f "$lock_file" ] && last_run=$(cat "$lock_file")
        if [ "$last_run" != "$et_date" ]; then
            log "EOD  — sending equity close summary to #kairos-reports"
            if python3 -c "from kairos_alerts import send_end_of_day_summary; send_end_of_day_summary('equity')"; then
                echo "$et_date" > "$lock_file"
                log "EOD  — equity summary sent"
            else
                log "EOD  — equity summary failed (exit $?)"
            fi
        fi
    fi

    # Crypto EOD summary disabled — crypto managed separately via Robinhood
}

# ── Main ──────────────────────────────────────────────────────────────────────

rotate_log

ET_TIME=$(TZ="America/New_York" date '+%Y-%m-%d %H:%M %Z')
DOW=$(TZ="America/New_York" date '+%A')
MODE=$(get_trading_mode)

case "$MODE" in
    "closed")
        log "SKIP — outside equity trading window (${DOW} ${ET_TIME})"
        exit 0
        ;;
    "equity")
        log "RUN  — equity window (${DOW} ${ET_TIME})"
        ;;
esac

# Activate virtual environment
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

# Load API keys from zshrc (handles keys not in launchd environment)
if [ -f "$HOME/.zshrc" ]; then
    # Extract only export lines to avoid running interactive shell code
    eval "$(grep -E '^export (ANTHROPIC|FINNHUB|FRED|KALSHI|CONGRESS|RENAISSANCE_CAPITAL)_API_KEY=' "$HOME/.zshrc" 2>/dev/null || true)"
fi

cd "$KAIROS_DIR"

# Ollama memory management: unload models from VRAM after 5 min of inactivity.
# This frees ~15 GB (mistral-small3.2) between 30-min scheduler cycles.
# Per-request keep_alive is also set in kairos_ollama.py; this env var acts as
# the server-wide fallback default for any requests that don't specify it.
export OLLAMA_KEEP_ALIVE="5m"

# Run full pipeline: orchestrate → screen → gather → reason → execute
# Monitor wrapper logs timing, exit code, and errors to kairos_monitor.log
if python3 kairos_run.py --cycle --mode equity; then
    log "PASS — full pipeline completed (mode: ${MODE})"
else
    EXIT_CODE=$?
    log "FAIL — pipeline exited with code ${EXIT_CODE} (mode: ${MODE})"
fi

# Ensure dashboard Flask server is running (port 5001)
ensure_dashboard_server() {
    if ! curl -s --max-time 2 http://127.0.0.1:5001/api/status >/dev/null 2>&1; then
        log "DASH — dashboard server not responding, reloading launchd agent"
        launchctl bootout "gui/$(id -u)/com.kairos.dashboard" 2>/dev/null || true
        launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.kairos.dashboard.plist" 2>/dev/null || true
        sleep 2
        if curl -s --max-time 2 http://127.0.0.1:5001/api/status >/dev/null 2>&1; then
            log "DASH — dashboard server restarted successfully"
        else
            log "DASH — WARNING: dashboard server failed to start"
        fi
    fi
}

ensure_dashboard_server

# Refresh performance dashboard once daily at/after 16:15 ET (weekdays only)
run_dashboard_if_needed

# Send end-of-day Slack summary (equity close 16:15 ET, crypto close 22:05 ET)
run_eod_summary_if_needed
