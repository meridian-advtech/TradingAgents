#!/usr/bin/env python3
"""Kairos boot/health monitor — detect silent infrastructure failure.

Two incidents in the week of 2026-08-10 motivated this, both undetected for days
because nothing in the stack alerts on its own absence:

  1. Ollama's background service died ~2026-08-13. Every scheduler cycle then
     failed `_check_ollama_models()` and exited 1. kairos_scheduler.log shows an
     unbroken RUN/FAIL ladder — the pipeline was running perfectly and trading
     nothing. Nobody was told.
  2. After a reboot, com.kairos.scheduler (and separately com.kairos.ibcgateway)
     did not survive and were never reloaded. No log lines at all, which reads
     identically to "market is closed".

Both were caught by chance. The gap they share: the components that would have
raised the alarm were the components that were down. This monitor runs in its
own LaunchAgent (com.kairos.healthcheck, StartInterval 1800) so it is not
downstream of anything it watches.

Four checks:
  1. Kairos LaunchAgents are loaded (`launchctl list` vs EXPECTED_AGENTS)
  2. Ollama answers /api/tags AND has the screen + reason models installed
  3. IB Gateway's port 7497 is listening
  4. The scheduler is making forward progress, evaluated ONLY inside the equity
     window — outside it the scheduler exits before logging, so a multi-hour gap
     is correct behaviour and must never alert.

READ-ONLY BY CONSTRUCTION
-------------------------
This script never fixes, restarts, reloads, relaunches, or kills anything. It
runs `launchctl list` (a read), opens and immediately closes a TCP socket, GETs
an HTTP endpoint, and reads a log file. Remediation is a human decision, the
same principle that governs the Arbiter's proposal gate and the sell guard's
refusal to act unilaterally. If you are tempted to add a `launchctl bootstrap`
here, add it to a separate tool that a human invokes.

Alerting is deliberately quiet: a fully healthy run posts NOTHING. An unresolved
failure is re-posted at most every REALERT_AFTER_HOURS, and immediately if the
set of failing checks changes. Recovery always posts exactly once.

Exit status: 0 = healthy (or checks skipped), 1 = at least one finding. The exit
code is for humans running it by hand; launchd ignores it.

Usage
-----
    python3 kairos_healthcheck.py              # normal run (may post to Slack)
    python3 kairos_healthcheck.py --dry-run    # run real checks, post nothing,
                                               # leave the de-dup state alone
    python3 kairos_healthcheck.py --verbose    # print every check's detail
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

ET = ZoneInfo("America/New_York")

# ── Configuration ────────────────────────────────────────────────────────────

# Every Kairos LaunchAgent that must be loaded for the system to function.
EXPECTED_AGENTS = (
    "com.kairos.scheduler",
    "com.kairos.commander",
    "com.kairos.arbiter.commander",
    "com.kairos.arbiter.daily",
    "com.kairos.arbiter.weekly",
    "com.kairos.outcome_features",
    "com.kairos.dashboard",
    "com.kairos.ibcgateway",
)

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_TIMEOUT_S = 10.0

# Model names are read from kairos_config.json at runtime so this monitor can
# never disagree with the gate it is watching. These are the fallbacks used only
# if the config is unreadable.
FALLBACK_SCREEN_MODEL = "phi4-mini"
FALLBACK_REASON_MODEL = "qwen3.6:35b-a3b"

IBKR_HOST = "localhost"
IBKR_PORT = 7497
IBKR_TIMEOUT_S = 10.0

# IB Gateway is restarted nightly at 23:59 ET with a 5-minute grace period
# (com.kairos.ibcgateway). Port 7497 is legitimately down across that boundary,
# so check 3 is suppressed between these ET times.
IBKR_RESTART_GRACE_START_HHMM = 2355
IBKR_RESTART_GRACE_END_HHMM = 15  # 00:15 ET

SCHEDULER_LOG = os.path.join(SCRIPT_DIR, "kairos_scheduler.log")
SCHEDULER_LOG_TAIL_BYTES = 256 * 1024

# The scheduler emits RUN/PASS/FAIL only in the "equity" mode window of
# kairos_scheduler.sh's get_trading_mode(): weekdays 09:30–16:00 ET. Between
# 09:25–09:30 and 16:00–16:30 the outer guard admits the script but
# get_trading_mode() returns "closed", so it logs SKIP and exits. Evaluating
# forward progress outside 09:30–16:00 would therefore false-alarm.
EQUITY_RUN_WINDOW_OPEN_HHMM = 930
EQUITY_RUN_WINDOW_CLOSE_HHMM = 1600

# launchd's StartInterval is not phase-aligned to the window, so the first cycle
# of the day can land up to 30 minutes after 09:30. Forward progress is only
# evaluated once the window has been open at least this long — before that, the
# newest RUN/PASS legitimately belongs to yesterday's session.
MAX_LOG_GAP_MIN = 45  # 30-min cycle + pipeline runtime + slack

# "Repeated FAIL with no intervening PASS for more than ~2 cycles" — a streak
# strictly greater than this is flagged as running-but-failing.
FAIL_STREAK_LIMIT = 2

# After a reboot every agent, Ollama, and IBC's gateway login are racing to come
# up at once. RunAtLoad fires this monitor into the middle of that, so a run
# within this many minutes of boot checks nothing and stays silent; the next
# StartInterval fire 30 minutes later does the real work. A genuine
# failed-to-survive-reboot is still caught within the hour instead of within days.
BOOT_GRACE_MINUTES = 10

STATE_FILE = os.path.join(SCRIPT_DIR, ".kairos_healthcheck_state.json")
REALERT_AFTER_HOURS = 4

SLACK_CHANNEL_KEY = "alerts"

_FOOTER = (
    "_Read-only monitor — nothing was restarted or reloaded. "
    "Remediation is yours. (`kairos_healthcheck.py`)_"
)


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """One specific, actionable failure.

    `key` is the de-duplication identity. It embeds the specifics (which agents
    are missing, which models are absent) so that a *changed* failure re-alerts
    immediately rather than hiding behind the 4-hour reminder interval.
    """

    key: str
    headline: str
    impact: str
    fix: str


@dataclass
class CheckResult:
    name: str
    detail: str
    findings: list[Finding] = field(default_factory=list)
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return not self.findings


# ── Probes ───────────────────────────────────────────────────────────────────
# Every syscall/network touch lives here, isolated from the decision logic, so
# kairos_selftest_healthcheck.py can drive real check + formatting code against
# simulated failures instead of stubbing out the logic under test.


def _launchctl_labels() -> tuple[set[str] | None, str]:
    """Loaded LaunchAgent labels in this user's domain. (None, err) if unknown."""
    try:
        proc = subprocess.run(
            ["/bin/launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return None, "/bin/launchctl not found"
    except subprocess.TimeoutExpired:
        return None, "`launchctl list` timed out after 15s"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"`launchctl list` failed: {exc}"

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        return None, f"`launchctl list` failed: {err}"

    labels: set[str] = set()
    for line in proc.stdout.splitlines()[1:]:  # skip the PID/Status/Label header
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].strip():
            labels.add(parts[2].strip())
    return labels, ""


def _fetch_ollama_tags() -> tuple[list[str] | None, tuple[str, str] | None]:
    """Installed Ollama model names, or (None, (kind, detail)) on failure.

    `kind` distinguishes the failure modes that need different fixes:
    "timeout" (hung), "refused" (not running), "http", "malformed".
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a hard dep
        return None, ("malformed", f"requests unavailable: {exc}")

    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=OLLAMA_TIMEOUT_S)
    except requests.Timeout:
        return None, ("timeout", f"GET {OLLAMA_TAGS_URL} timed out after {OLLAMA_TIMEOUT_S:.0f}s")
    except requests.ConnectionError as exc:
        return None, ("refused", f"cannot connect to {OLLAMA_TAGS_URL} ({type(exc).__name__})")
    except Exception as exc:
        return None, ("malformed", f"GET {OLLAMA_TAGS_URL} raised {type(exc).__name__}: {exc}")

    if resp.status_code != 200:
        return None, ("http", f"{OLLAMA_TAGS_URL} returned HTTP {resp.status_code}")

    try:
        models = resp.json().get("models", [])
        return [m["name"] for m in models], None
    except Exception as exc:
        return None, ("malformed", f"could not parse /api/tags response: {exc}")


def _probe_ibkr_port() -> tuple[bool, str]:
    """True if 127.0.0.1:7497 accepts a TCP connection.

    Prefers kairos_health.check_ib_gateway_connection — the existing shared
    port probe — so this monitor and the pipeline's own gate agree. Falls back
    to the same connect_ex idiom if that import breaks.

    Caveat worth knowing: a listening socket proves the gateway process is up,
    not that IBC completed its login. A full ib.connect() would prove more but
    would consume a clientId and could collide with a live cycle, which a
    read-only monitor has no business doing.
    """
    try:
        from kairos_health import check_ib_gateway_connection

        status, detail = check_ib_gateway_connection()
        return status == "PASS", detail
    except Exception:
        pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(IBKR_TIMEOUT_S)
    try:
        result = sock.connect_ex((IBKR_HOST, IBKR_PORT))
        if result == 0:
            return True, f"{IBKR_HOST}:{IBKR_PORT} accepting connections"
        return False, f"connection refused (errno {result})"
    except socket.timeout:
        return False, f"connect timed out after {IBKR_TIMEOUT_S:.0f}s"
    except Exception as exc:
        return False, f"connect error: {exc}"
    finally:
        sock.close()


def _read_scheduler_log_tail() -> tuple[str | None, str]:
    """Tail of kairos_scheduler.log, or (None, err)."""
    try:
        size = os.path.getsize(SCHEDULER_LOG)
        with open(SCHEDULER_LOG, "rb") as fh:
            if size > SCHEDULER_LOG_TAIL_BYTES:
                fh.seek(size - SCHEDULER_LOG_TAIL_BYTES)
            return fh.read().decode("utf-8", errors="replace"), ""
    except FileNotFoundError:
        return None, f"{SCHEDULER_LOG} does not exist"
    except Exception as exc:
        return None, f"cannot read {SCHEDULER_LOG}: {exc}"


def _boot_age_minutes() -> float | None:
    """Minutes since the machine booted, or None if undeterminable."""
    try:
        import psutil

        return (datetime.now(timezone.utc).timestamp() - psutil.boot_time()) / 60.0
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.boottime"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        match = re.search(r"sec\s*=\s*(\d+)", out)
        if match:
            boot = int(match.group(1))
            return (datetime.now(timezone.utc).timestamp() - boot) / 60.0
    except Exception:
        pass
    return None


# ── Check 1 — LaunchAgents loaded ────────────────────────────────────────────


def check_launch_agents() -> CheckResult:
    labels, err = _launchctl_labels()

    if labels is None:
        return CheckResult(
            "LaunchAgents loaded",
            err,
            [
                Finding(
                    key="launchagents:unknown",
                    headline=f"Cannot determine which LaunchAgents are loaded — {err}.",
                    impact=(
                        "The monitor is blind to the failure mode that took the "
                        "scheduler down after the last reboot."
                    ),
                    fix="Run `launchctl list | grep kairos` by hand and check all "
                    f"{len(EXPECTED_AGENTS)} agents are present.",
                )
            ],
        )

    missing = [label for label in EXPECTED_AGENTS if label not in labels]
    if not missing:
        return CheckResult(
            "LaunchAgents loaded", f"all {len(EXPECTED_AGENTS)} agents loaded"
        )

    uid = os.getuid()
    cmds = "\n".join(
        f"    launchctl bootstrap gui/{uid} ~/Library/LaunchAgents/{label}.plist"
        for label in missing
    )
    return CheckResult(
        "LaunchAgents loaded",
        f"missing: {', '.join(missing)}",
        [
            Finding(
                key="launchagents:" + ",".join(sorted(missing)),
                headline=(
                    f"{len(missing)} of {len(EXPECTED_AGENTS)} Kairos LaunchAgents "
                    f"are NOT loaded: {', '.join(missing)}."
                ),
                impact=_agent_impact(missing),
                fix="Reload each one (verify the plist first):\n" + cmds,
            )
        ],
    )


def _agent_impact(missing: list[str]) -> str:
    """Name the concrete consequence per agent — a bare label is not actionable."""
    consequences = {
        "com.kairos.scheduler": "the trading pipeline is not running at all — no screening, no reasoning, no execution, no stop-losses",
        "com.kairos.ibcgateway": "IB Gateway will not be relaunched after its nightly restart, so every IBKR call fails",
        "com.kairos.dashboard": "the dashboard server on port 5001 is down (no remote view over Tailscale)",
        "com.kairos.commander": "Slack commands in #kairos-commands are unanswered",
        "com.kairos.arbiter.commander": "Arbiter Slack interaction and Approve/Reject cards are dead",
        "com.kairos.arbiter.daily": "no nightly retrospective in #kairos-arbiter",
        "com.kairos.arbiter.weekly": "no weekly pattern review in #kairos-arbiter",
        "com.kairos.outcome_features": "closed trades are never enriched with mfe/give-back/forgone gain, so every new close is non-contributing evidence and the learning loop reports sample=0 indefinitely while trades keep closing",
    }
    lines = [
        f"{label} — {consequences.get(label, 'unknown role')}" for label in missing
    ]
    return "; ".join(lines)


# ── Check 2 — Ollama responsiveness ──────────────────────────────────────────


def _required_ollama_models() -> tuple[str, str]:
    """(screen_model, reason_model) as the pipeline itself resolves them."""
    try:
        from kairos_ollama import get_reason_model, get_screen_model

        return get_screen_model(), get_reason_model()
    except Exception:
        return FALLBACK_SCREEN_MODEL, FALLBACK_REASON_MODEL


def _model_present(wanted: str, installed: list[str]) -> bool:
    """Tag-insensitive match, identical to kairos_ollama.check_required_models.

    /api/tags reports "phi4-mini:latest", so an exact string compare against
    the configured "phi4-mini" would report a false absence.
    """
    base = wanted.split(":")[0]
    return any(name.startswith(base) for name in installed)


def check_ollama() -> CheckResult:
    screen, reason = _required_ollama_models()
    installed, error = _fetch_ollama_tags()

    if error is not None:
        kind, detail = error
        if kind == "timeout":
            headline = f"Ollama is not responding — {detail}."
        elif kind == "refused":
            headline = f"Ollama is not running — {detail}."
        elif kind == "http":
            headline = f"Ollama is up but unhealthy — {detail}."
        else:
            headline = f"Ollama returned an unusable response — {detail}."
        return CheckResult(
            "Ollama responsive",
            detail,
            [
                Finding(
                    key=f"ollama:{kind}",
                    headline=headline,
                    impact=(
                        "Every scheduler cycle will fail its health gate before "
                        "screening — this is the 2026-08-13 outage signature: the "
                        "pipeline runs, exits 1, and trades nothing."
                    ),
                    fix=(
                        "Reopen Ollama.app (or restart its background service), then "
                        f"confirm `ollama list` shows {screen} and {reason}. "
                        "Little Snitch blocks Ollama's outbound traffic by default — "
                        "that does not affect localhost, so a refused connection here "
                        "means the service itself is down."
                    ),
                )
            ],
        )

    installed = installed or []
    missing = [m for m in (screen, reason) if not _model_present(m, installed)]
    if not missing:
        return CheckResult(
            "Ollama responsive",
            f"responsive, {screen} + {reason} present ({len(installed)} models installed)",
        )

    roles = {screen: "Tier 1 screener", reason: "router/analyst"}
    listing = ", ".join(f"{m} ({roles[m]})" for m in missing)
    have = ", ".join(installed) if installed else "(none)"
    return CheckResult(
        "Ollama responsive",
        f"responsive but missing: {', '.join(missing)}",
        [
            Finding(
                key="ollama:models:" + ",".join(sorted(missing)),
                headline=(
                    f"Ollama is responding but {len(missing)} required model(s) are "
                    f"missing: {listing}."
                ),
                impact=(
                    "kairos_run._check_ollama_models() fails the cycle before "
                    "screening, so no trading happens."
                ),
                fix=(
                    "Pull the missing model(s): "
                    + "; ".join(f"`ollama pull {m}`" for m in missing)
                    + f". Currently installed: {have}."
                ),
            )
        ],
    )


# ── Check 3 — IB Gateway connectivity ────────────────────────────────────────


def _in_ibkr_restart_grace(now_et: datetime) -> bool:
    hhmm = now_et.hour * 100 + now_et.minute
    return hhmm >= IBKR_RESTART_GRACE_START_HHMM or hhmm < IBKR_RESTART_GRACE_END_HHMM


def check_ibkr_gateway(now_et: datetime) -> CheckResult:
    if _in_ibkr_restart_grace(now_et):
        return CheckResult(
            "IB Gateway port 7497",
            "skipped — inside the nightly IB Gateway restart grace period",
            skipped=True,
        )

    ok, detail = _probe_ibkr_port()
    if ok:
        return CheckResult("IB Gateway port 7497", detail)

    return CheckResult(
        "IB Gateway port 7497",
        detail,
        [
            Finding(
                key="ibkr:port_down",
                headline=(
                    f"IB Gateway is not reachable on {IBKR_HOST}:{IBKR_PORT} — {detail}."
                ),
                impact=(
                    "Nothing can reach the broker: no portfolio reads, no stop-loss "
                    "sells, no order execution. Positions are unmanaged."
                ),
                fix=(
                    "Check IB Gateway is running and logged in (IBC drives the login). "
                    f"If the process is gone, `launchctl kickstart -k gui/{os.getuid()}"
                    "/com.kairos.ibcgateway`. If it is running but refusing, re-check "
                    "Configure > API > 'Enable ActiveX and Socket Clients' — a Gateway "
                    "update has silently reset that checkbox before."
                ),
            )
        ],
    )


# ── Check 4 — Scheduler forward progress ─────────────────────────────────────

# [2026-08-12T19:53:40Z] RUN  — equity window (Wednesday 2026-08-12 15:53 EDT)
_LOG_LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]\s+([A-Z]+)\b")


def _parse_scheduler_events(log_text: str) -> list[tuple[datetime, str, str]]:
    """(utc_dt, event, line) in file order, consecutive duplicates collapsed.

    kairos_scheduler.sh's log() pipes through `tee -a`, and the LaunchAgent also
    redirects stdout to the same file — so every line is written twice. Counting
    a FAIL streak without collapsing those would double it and trip the
    running-but-failing threshold at one real failure instead of three.
    """
    events: list[tuple[datetime, str, str]] = []
    previous_line: str | None = None
    for raw in log_text.splitlines():
        line = raw.strip()
        match = _LOG_LINE_RE.match(line)
        if not match:
            continue
        if line == previous_line:  # tee duplicate
            continue
        previous_line = line
        stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        events.append((stamp, match.group(2), line))
    return events


def _trailing_fail_streak(events: list[tuple[datetime, str, str]]) -> int:
    """Consecutive FAILs at the tail with no intervening PASS."""
    streak = 0
    for _stamp, event, _line in reversed(events):
        if event == "FAIL":
            streak += 1
        elif event == "PASS":
            break
    return streak


def check_scheduler_progress(now_et: datetime, now_utc: datetime) -> CheckResult:
    name = "Scheduler forward progress"
    hhmm = now_et.hour * 100 + now_et.minute

    if now_et.isoweekday() >= 6:
        return CheckResult(name, "skipped — weekend, no equity cycles run", skipped=True)
    if hhmm < EQUITY_RUN_WINDOW_OPEN_HHMM or hhmm >= EQUITY_RUN_WINDOW_CLOSE_HHMM:
        return CheckResult(
            name,
            "skipped — outside the 09:30–16:00 ET equity window, where the "
            "scheduler logs SKIP and exits (long gaps are correct)",
            skipped=True,
        )

    window_open = now_et.replace(
        hour=EQUITY_RUN_WINDOW_OPEN_HHMM // 100,
        minute=EQUITY_RUN_WINDOW_OPEN_HHMM % 100,
        second=0,
        microsecond=0,
    )
    open_for_min = (now_et - window_open).total_seconds() / 60.0
    if open_for_min < MAX_LOG_GAP_MIN:
        return CheckResult(
            name,
            f"skipped — equity window has only been open {open_for_min:.0f}m; "
            f"launchd is not phase-aligned so the first cycle may not have fired yet",
            skipped=True,
        )

    log_text, err = _read_scheduler_log_tail()
    if log_text is None:
        return CheckResult(
            name,
            err,
            [
                Finding(
                    key="scheduler:log_unreadable",
                    headline=f"Cannot read the scheduler log — {err}.",
                    impact="Forward progress cannot be verified; the pipeline may be "
                    "stalled without anything noticing.",
                    fix=f"Check {SCHEDULER_LOG} exists and is readable, and that "
                    "com.kairos.scheduler's StandardOutPath still points at it.",
                )
            ],
        )

    events = _parse_scheduler_events(log_text)
    cutoff = now_utc - timedelta(minutes=MAX_LOG_GAP_MIN)
    recent = [e for e in events if e[0] >= cutoff]
    progressing = any(event in ("RUN", "PASS") for _s, event, _l in recent)

    if not progressing:
        activity = [e for e in events if e[1] in ("RUN", "PASS", "FAIL")]
        if activity:
            last_stamp, last_event, _ = activity[-1]
            age_min = (now_utc - last_stamp).total_seconds() / 60.0
            age = (
                f"{age_min / 60.0:.1f} hours" if age_min >= 90 else f"{age_min:.0f} minutes"
            )
            where = (
                f"last cycle activity was {last_event} at "
                f"{last_stamp.strftime('%Y-%m-%d %H:%M')}Z — {age} ago"
            )
        else:
            where = "the log tail contains no RUN/PASS/FAIL lines at all"
        return CheckResult(
            name,
            f"NOT RUNNING — {where}",
            [
                Finding(
                    key="scheduler:not_running",
                    headline=(
                        "Scheduler is NOT RUNNING — no RUN or PASS logged in the last "
                        f"{MAX_LOG_GAP_MIN} minutes, mid-session at "
                        f"{now_et.strftime('%H:%M')} ET ({where})."
                    ),
                    impact=(
                        "Zero trading activity: no entries, no thesis exits, and no "
                        "stop-loss evaluation. This is the failed-to-survive-reboot "
                        "signature."
                    ),
                    fix=(
                        "This is a LAUNCH problem, not a pipeline problem — check "
                        "com.kairos.scheduler is loaded (`launchctl list | grep "
                        "kairos.scheduler`) and reload it if absent. If it IS loaded, "
                        "run `bash kairos_scheduler.sh` by hand and watch where it "
                        "stops."
                    ),
                )
            ],
        )

    streak = _trailing_fail_streak(events)
    if streak > FAIL_STREAK_LIMIT:
        last_fail = [e for e in events if e[1] == "FAIL"][-1]
        return CheckResult(
            name,
            f"RUNNING BUT FAILING — {streak} consecutive FAIL cycles",
            [
                Finding(
                    key="scheduler:failing_streak",
                    headline=(
                        f"Scheduler is RUNNING BUT FAILING — {streak} consecutive "
                        "cycles have logged FAIL with no PASS in between. Most recent: "
                        f"`{last_fail[2]}`."
                    ),
                    impact=(
                        "launchd is firing correctly and the pipeline is exiting "
                        "non-zero every cycle, so nothing trades. This is the "
                        "2026-08-13 signature and is invisible from the outside — the "
                        "log looks busy."
                    ),
                    fix=(
                        "This is a PIPELINE problem, not a launch problem — do NOT "
                        "reload the agent. Read the FAIL cause in "
                        "kairos_scheduler.log (an Ollama/model gate failure is the "
                        "usual culprit; check the Ollama finding above if present), or "
                        "reproduce with `python3 kairos_run.py --cycle --mode equity`."
                    ),
                )
            ],
        )

    return CheckResult(
        name,
        f"progressing — {len(recent)} log events in the last {MAX_LOG_GAP_MIN}m, "
        f"trailing FAIL streak {streak}",
    )


# ── Alert formatting ─────────────────────────────────────────────────────────


def format_alert(
    findings: list[Finding],
    results: list[CheckResult],
    now_et: datetime,
    *,
    unresolved_since: datetime | None = None,
) -> str:
    evaluated = [r for r in results if not r.skipped]
    stamp = now_et.strftime("%Y-%m-%d %H:%M %Z")
    header = (
        f":warning: *Kairos health check* — {len(findings)} of {len(evaluated)} "
        f"evaluated checks FAILING · {stamp}"
    )
    if unresolved_since is not None:
        hours = (now_et - unresolved_since).total_seconds() / 3600.0
        header += f"\n_Still unresolved after {hours:.1f} hours (reminder)._"

    parts = [header, ""]
    for finding in findings:
        parts.append(f"*{finding.headline}*")
        parts.append(f"› *Impact:* {finding.impact}")
        parts.append(f"› *Likely fix:* {finding.fix}")
        parts.append("")

    skipped = [r for r in results if r.skipped]
    if skipped:
        parts.append(
            "_Not evaluated this run: " + "; ".join(f"{r.name} ({r.detail})" for r in skipped) + "._"
        )
    parts.append(_FOOTER)
    return "\n".join(parts)


def format_recovery(resolved: list[dict], now_et: datetime) -> str:
    stamp = now_et.strftime("%Y-%m-%d %H:%M %Z")
    parts = [f":white_check_mark: *Kairos health check — RECOVERED* · {stamp}", ""]
    parts.append("These failures have cleared:")
    for item in resolved:
        parts.append(f"• {item.get('headline', item.get('key', 'unknown'))}")
    parts.append("")
    parts.append("All four checks now pass. No further alerts until something breaks.")
    return "\n".join(parts)


# ── De-duplication state ─────────────────────────────────────────────────────


def load_state(path: str = STATE_FILE) -> dict:
    try:
        with open(path) as fh:
            state = json.load(fh)
        return state if isinstance(state, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        # A corrupt state file must not silence the monitor; start clean and say so.
        print(f"  state file unreadable ({exc}) — treating as empty")
        return {}


def save_state(state: dict, path: str = STATE_FILE) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"  WARNING: could not write state file: {exc}")


def _signature(findings: list[Finding]) -> str:
    return "|".join(sorted(f.key for f in findings))


def decide_alert(
    findings: list[Finding], state: dict, now_utc: datetime
) -> tuple[str, dict]:
    """Return (action, new_state).

    action is one of:
      "silent"    — nothing to say (healthy and was healthy, or duplicate)
      "new"       — first time this exact failure set has been seen
      "reminder"  — same failure set, but REALERT_AFTER_HOURS have elapsed
      "recovery"  — previously failing, now clear
    """
    previous_sig = state.get("signature", "")
    signature = _signature(findings)

    if not findings:
        if previous_sig:
            return "recovery", {}
        return "silent", {}

    record = {
        "signature": signature,
        "findings": [{"key": f.key, "headline": f.headline} for f in findings],
        "first_seen": state.get("first_seen") or now_utc.isoformat(),
        "last_alerted": now_utc.isoformat(),
    }

    if signature != previous_sig:
        # New or changed failure — alert immediately and restart the clock.
        record["first_seen"] = now_utc.isoformat()
        return "new", record

    last_alerted = state.get("last_alerted")
    if last_alerted:
        try:
            elapsed = now_utc - datetime.fromisoformat(last_alerted)
            if elapsed < timedelta(hours=REALERT_AFTER_HOURS):
                unchanged = dict(state)
                unchanged["findings"] = record["findings"]
                return "silent", unchanged
        except ValueError:
            pass  # unparseable timestamp — re-alert rather than stay quiet
    return "reminder", record


# ── Slack ────────────────────────────────────────────────────────────────────


def _load_slack_token_from_zshrc() -> None:
    """Lift SLACK_BOT_TOKEN out of ~/.zshrc when launchd did not provide it.

    launchd agents get a minimal environment, so the token that a login shell
    exports is not present. kairos_scheduler.sh solves this by eval-ing the
    export lines out of ~/.zshrc; this is the same idea without handing any of
    that file to a shell — the value is parsed, not executed. The token stays
    out of the plist, which is the whole point (a plist is a file that gets
    backed up, diffed, and pasted into chats).
    """
    if os.environ.get("SLACK_BOT_TOKEN"):
        return
    zshrc = os.path.join(os.path.expanduser("~"), ".zshrc")
    try:
        with open(zshrc) as fh:
            for line in fh:
                match = re.match(r"^\s*export\s+SLACK_BOT_TOKEN=(.*)$", line)
                if match:
                    value = match.group(1).strip().strip('"').strip("'")
                    if value:
                        os.environ["SLACK_BOT_TOKEN"] = value
                    return
    except Exception as exc:
        print(f"  could not read SLACK_BOT_TOKEN from ~/.zshrc: {exc}")


def _post_slack(text: str) -> bool:
    try:
        from kairos_alerts import post_message

        return post_message(SLACK_CHANNEL_KEY, text)
    except Exception as exc:
        print(f"  ERROR: Slack post failed: {exc}")
        return False


# ── Orchestration ────────────────────────────────────────────────────────────


# ── Check 5 — recent segfault/crash reports ──────────────────────────────────

CRASH_REPORT_DIR = os.path.expanduser("~/Library/Logs/DiagnosticReports")
CRASH_LOOKBACK_MINUTES = 40  # 30m cycle + slack, so consecutive runs don't gap


def check_recent_crashes(now_utc: datetime) -> CheckResult:
    """Catches the class of failure that motivated this whole file: a Kairos
    process (usually a fork()-in-a-multithreaded-process crash — see
    kairos_alerts.py's posix_spawn rewrite, 2026-08-18) segfaults, and until
    now nothing noticed except a macOS crash dialog popping up in front of J.
    A crashed cycle can still exit with output that LOOKS like a clean run
    from the outside (2026-08-19: 1 crash, trading recovered and filled fine)
    — this check exists so 'nothing looked wrong' stops being the only signal.
    """
    try:
        if not os.path.isdir(CRASH_REPORT_DIR):
            return CheckResult("Recent crashes", "no crash report directory found (nothing to check)")
        cutoff = now_utc.timestamp() - CRASH_LOOKBACK_MINUTES * 60
        hits = []
        for fname in os.listdir(CRASH_REPORT_DIR):
            if not (fname.startswith("Python-") and fname.endswith(".ips")):
                continue
            path = os.path.join(CRASH_REPORT_DIR, fname)
            try:
                if os.path.getmtime(path) >= cutoff:
                    hits.append((fname, path))
            except OSError:
                continue
    except Exception as exc:
        return CheckResult("Recent crashes", f"could not scan crash reports: {exc}", skipped=True)

    if not hits:
        return CheckResult("Recent crashes", f"none in the last {CRASH_LOOKBACK_MINUTES}m")

    # Pull a bit of context from the most recent one — coalition + whether it's
    # the known fork/atfork signature, so the alert is specific, not just a count.
    coalition, signature = "unknown", "unknown"
    try:
        with open(sorted(hits, key=lambda h: os.path.getmtime(h[1]))[-1][1]) as fh:
            head = fh.read(4000)
        m = re.search(r'"coalitionName"\s*:\s*"([^"]+)"', head)
        if m:
            coalition = m.group(1)
        if "multi-threaded process forked" in head or "subprocess_fork_exec" in head:
            signature = "fork() in a multi-threaded process (known class, see kairos_alerts.py)"
        elif "EXC_BAD_ACCESS" in head:
            signature = "EXC_BAD_ACCESS (segfault, cause not yet classified)"
    except Exception:
        pass

    names = sorted(set(f for f, _ in hits))
    return CheckResult(
        "Recent crashes",
        f"{len(hits)} in the last {CRASH_LOOKBACK_MINUTES}m",
        [
            Finding(
                key="crash:" + ",".join(names),
                headline=(
                    f"{len(hits)} Python crash report(s) in the last "
                    f"{CRASH_LOOKBACK_MINUTES} minutes, most recent from "
                    f"coalition '{coalition}'."
                ),
                impact=(
                    f"Signature: {signature}. A crashed process can still let the "
                    "cycle recover and trade normally (as on 2026-08-19), so this "
                    "can be quiet-but-real degradation rather than an outage — "
                    "worth a look even if nothing else looks broken."
                ),
                fix=(
                    "Check ~/Library/Logs/DiagnosticReports/ for the file(s): "
                    f"{', '.join(names)}. If the signature above is the known "
                    "fork class, find which subprocess.run() call site is still "
                    "unconverted to posix_spawn (kairos_commander.py, "
                    "kairos_dashboard_server.py, and this file's own launchctl/"
                    "sysctl calls are the known remaining candidates as of "
                    "2026-08-19)."
                ),
            )
        ],
    )


def run_checks(now_et: datetime, now_utc: datetime) -> list[CheckResult]:
    return [
        check_launch_agents(),
        check_ollama(),
        check_ibkr_gateway(now_et),
        check_scheduler_progress(now_et, now_utc),
        check_recent_crashes(now_utc),
    ]


def run_healthcheck(
    *,
    now_utc: datetime | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    state_path: str = STATE_FILE,
) -> dict:
    now_utc = now_utc or datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)

    boot_age = _boot_age_minutes()
    if boot_age is not None and boot_age < BOOT_GRACE_MINUTES:
        print(
            f"  boot grace — up {boot_age:.1f}m (< {BOOT_GRACE_MINUTES}m); agents, "
            "Ollama and the IBC gateway login are still racing. Checking nothing; "
            "next run in 30m."
        )
        return {"action": "silent", "findings": [], "results": [], "posted": False}

    results = run_checks(now_et, now_utc)
    findings = [f for r in results for f in r.findings]

    for result in results:
        if verbose or not result.ok:
            mark = "⊘" if result.skipped else ("✓" if result.ok else "✗")
            print(f"  {mark} {result.name}: {result.detail}")

    state = load_state(state_path)
    action, new_state = decide_alert(findings, state, now_utc)

    message = ""
    if action in ("new", "reminder"):
        unresolved_since = None
        if action == "reminder":
            try:
                unresolved_since = datetime.fromisoformat(
                    new_state["first_seen"]
                ).astimezone(ET)
            except Exception:
                unresolved_since = None
        message = format_alert(
            findings, results, now_et, unresolved_since=unresolved_since
        )
    elif action == "recovery":
        message = format_recovery(state.get("findings", []), now_et)

    posted = False
    if message and not dry_run:
        _load_slack_token_from_zshrc()
        posted = _post_slack(message)
    elif message and dry_run:
        print("\n--- would post to #kairos-alerts ---")
        print(message)
        print("--- end ---\n")

    if not dry_run:
        if action == "silent" and not findings:
            if os.path.exists(state_path) and not new_state:
                save_state({}, state_path)
        else:
            save_state(new_state, state_path)

    if action == "silent" and not findings:
        print(f"  all checks pass ({now_et.strftime('%Y-%m-%d %H:%M %Z')}) — no Slack post")
    elif action == "silent":
        print(
            f"  {len(findings)} finding(s) unchanged and already alerted within "
            f"{REALERT_AFTER_HOURS}h — suppressed (no Slack post)"
        )

    return {
        "action": action,
        "findings": findings,
        "results": results,
        "message": message,
        "posted": posted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kairos infrastructure health monitor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the real checks, print any alert, post nothing, leave state untouched",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print every check, not just failures"
    )
    args = parser.parse_args(argv)

    print(
        f"Kairos health check — {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        + (" (dry run)" if args.dry_run else "")
    )
    outcome = run_healthcheck(dry_run=args.dry_run, verbose=args.verbose)
    return 1 if outcome["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
