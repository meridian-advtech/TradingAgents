#!/usr/bin/env python3
"""Health-monitor tests. Every probe is stubbed; all check + alert logic is real.

The four probe functions in kairos_healthcheck (_launchctl_labels,
_fetch_ollama_tags, _probe_ibkr_port, _read_scheduler_log_tail) are the only
places that touch the OS or network. Replacing just those means these tests
exercise the real decision logic, the real de-duplication state machine, and the
real Slack message text — not stubs of them.

Run: python3 kairos_selftest_healthcheck.py [--show-messages]
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/Users/jelmore/Kairos")

import kairos_healthcheck as H  # noqa: E402

SHOW = "--show-messages" in sys.argv

# Monday 2026-08-17, 14:00 ET — mid-session, well past the window-open warmup.
NOW = datetime(2026, 8, 17, 18, 0, 0, tzinfo=timezone.utc)
assert NOW.astimezone(H.ET).isoweekday() <= 5, "fixture time must be a weekday"
assert NOW.astimezone(H.ET).strftime("%H:%M") == "14:00"

_results = []
_posts = []


def ck(label, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"\n         {detail}" if detail else ""))


def _fake_post(text):
    _posts.append(text)
    return True


H._post_slack = _fake_post
# Never let the boot-grace short-circuit swallow a test run.
H._boot_age_minutes = lambda: 999.0


def sched_log(cycles, now):
    """Build a scheduler log tail.

    `cycles` is [(minutes_ago, "PASS"|"FAIL"), ...] relative to `now`. Each cycle
    emits a RUN line and an outcome line, and every line is written TWICE —
    kairos_scheduler.sh pipes log() through `tee -a` while the LaunchAgent also
    redirects stdout to the same file, so the real log is genuinely duplicated.
    """
    lines = []
    for minutes_ago, outcome in sorted(cycles, key=lambda c: -c[0]):
        stamp = (now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        et = (now - timedelta(minutes=minutes_ago)).astimezone(H.ET)
        run = f"[{stamp}] RUN  — equity window ({et.strftime('%A %Y-%m-%d %H:%M %Z')})"
        if outcome == "PASS":
            end = f"[{stamp}] PASS — full pipeline completed (mode: equity)"
        else:
            end = f"[{stamp}] FAIL — pipeline exited with code 1 (mode: equity)"
        lines += [run, run, end, end]
    return "\n".join(lines) + "\n"


HEALTHY = [(10, "PASS"), (40, "PASS"), (70, "PASS"), (100, "PASS")]
FAILING = [(5, "FAIL"), (35, "FAIL"), (65, "FAIL"), (95, "FAIL")]
STALLED = [(180, "PASS"), (210, "PASS")]

# The log stub must be generated against the *simulated* clock, not a fixed
# timestamp — otherwise advancing time in the de-dup tests silently ages a
# healthy log into a stalled one and the fixture, not the code, decides the test.
_sim_now = NOW


def set_probes(agents=None, ollama=None, ibkr=None, cycles=None):
    """Install probe stubs; each argument defaults to the healthy answer."""
    H._launchctl_labels = lambda: (agents if agents is not None else (set(H.EXPECTED_AGENTS), ""))
    H._fetch_ollama_tags = lambda: (
        ollama if ollama is not None else (["phi4-mini:latest", "qwen3.6:35b-a3b"], None)
    )
    H._probe_ibkr_port = lambda: (
        ibkr if ibkr is not None else (True, "localhost:7497 accepting connections")
    )
    spec = HEALTHY if cycles is None else cycles
    H._read_scheduler_log_tail = lambda: (sched_log(spec, _sim_now), "")


def run(state_path, now=NOW):
    global _sim_now
    _sim_now = now
    _posts.clear()
    return H.run_healthcheck(now_utc=now, state_path=state_path, verbose=False)


def fresh_state():
    return os.path.join(tempfile.mkdtemp(prefix="kairos_hc_"), "state.json")


def show(title, text):
    print(f"\n  ── Slack message: {title} " + "─" * max(0, 50 - len(title)))
    for line in text.splitlines():
        print(f"  | {line}")
    print("  " + "─" * 60)


# ═════════════════════════════════════════════════════════════════════════════
print("\n=== A. Fully healthy system posts NOTHING ===")
set_probes()
out = run(fresh_state())
ck("no findings", out["findings"] == [], f"findings={out['findings']}")
ck("action is silent", out["action"] == "silent")
ck("zero Slack calls", len(_posts) == 0, f"{len(_posts)} post(s)")


# ═════════════════════════════════════════════════════════════════════════════
print("\n=== B. Check 1 — LaunchAgents missing ===")
loaded = set(H.EXPECTED_AGENTS) - {"com.kairos.scheduler", "com.kairos.ibcgateway"}
set_probes(agents=(loaded, ""))
out = run(fresh_state())
msg_b = out["message"]
ck("exactly one finding", len(out["findings"]) == 1)
ck("posted once", len(_posts) == 1)
ck("names com.kairos.scheduler", "com.kairos.scheduler" in msg_b)
ck("names com.kairos.ibcgateway", "com.kairos.ibcgateway" in msg_b)
ck("gives the bootstrap command", "launchctl bootstrap gui/" in msg_b)
ck("not generic", "something" not in msg_b.lower())
if SHOW:
    show("LaunchAgents missing", msg_b)


# ═════════════════════════════════════════════════════════════════════════════
print("\n=== C. Check 2 — Ollama timed out ===")
set_probes(ollama=(None, ("timeout", "GET http://localhost:11434/api/tags timed out after 10s")))
out = run(fresh_state())
msg_c = out["message"]
ck("one finding", len(out["findings"]) == 1)
ck("posted once", len(_posts) == 1)
ck("says not responding", "not responding" in msg_c)
ck("names the 10s timeout", "timed out after 10s" in msg_c)
ck("explains the health-gate consequence", "health gate" in msg_c)
ck("suggests reopening Ollama.app", "Ollama.app" in msg_c)
if SHOW:
    show("Ollama timeout", msg_c)


print("\n=== C2. Check 2 — Ollama refusing connections (distinct from timeout) ===")
set_probes(ollama=(None, ("refused", "cannot connect to http://localhost:11434/api/tags (ConnectionError)")))
out = run(fresh_state())
ck("distinct de-dup key from timeout", out["findings"][0].key == "ollama:refused")
ck("says not running, not 'not responding'", "is not running" in out["message"])
if SHOW:
    show("Ollama refused", out["message"])


# ═════════════════════════════════════════════════════════════════════════════
print("\n=== D. Check 2 — Ollama up, required models absent (the 2026-08-13 log) ===")
set_probes(ollama=([], None))
out = run(fresh_state())
msg_d = out["message"]
ck("one finding", len(out["findings"]) == 1)
ck("names phi4-mini", "phi4-mini" in msg_d)
ck("names qwen3.6:35b-a3b", "qwen3.6:35b-a3b" in msg_d)
ck("gives the pull commands", "ollama pull phi4-mini" in msg_d)
ck("distinguishes 'responding but missing' from 'down'", "is responding but" in msg_d)
if SHOW:
    show("Ollama models missing", msg_d)


print("\n=== D2. A :latest tag still counts as present ===")
set_probes(ollama=(["phi4-mini:latest", "qwen3.6:35b-a3b"], None))
out = run(fresh_state())
ck("no false 'missing model' on :latest", out["findings"] == [])


# ═════════════════════════════════════════════════════════════════════════════
print("\n=== E. Check 3 — IB Gateway port 7497 down ===")
set_probes(ibkr=(False, "connection refused (errno 61)"))
out = run(fresh_state())
msg_e = out["message"]
ck("one finding", len(out["findings"]) == 1)
ck("names the port", "7497" in msg_e)
ck("names the errno", "errno 61" in msg_e)
ck("states positions are unmanaged", "unmanaged" in msg_e)
ck("mentions the ActiveX/Socket checkbox", "Enable ActiveX" in msg_e)
if SHOW:
    show("IBKR gateway down", msg_e)


print("\n=== E2. Check 3 is suppressed during the nightly gateway restart ===")
midnight = datetime(2026, 8, 18, 4, 0, 0, tzinfo=timezone.utc)  # 00:00 ET
ck("00:00 ET is inside the grace window", H._in_ibkr_restart_grace(midnight.astimezone(H.ET)))
ck("14:00 ET is not", not H._in_ibkr_restart_grace(NOW.astimezone(H.ET)))
set_probes(ibkr=(False, "connection refused (errno 61)"))
out = run(fresh_state(), now=midnight)
ck("no IBKR alert during the restart grace period", out["findings"] == [])
ck("zero Slack calls", len(_posts) == 0)


# ═════════════════════════════════════════════════════════════════════════════
print("\n=== F. Check 4 — scheduler NOT RUNNING (failed-to-survive-reboot) ===")
set_probes(cycles=STALLED)
out = run(fresh_state())
msg_f = out["message"]
ck("one finding", len(out["findings"]) == 1)
ck("key is not_running", out["findings"][0].key == "scheduler:not_running")
ck("says NOT RUNNING", "NOT RUNNING" in msg_f)
ck("gives the stale age", "3.0 hours ago" in msg_f)
ck("calls it a LAUNCH problem", "LAUNCH problem" in msg_f)
ck("mentions stop-loss exposure", "stop-loss" in msg_f)
if SHOW:
    show("Scheduler not running", msg_f)


print("\n=== G. Check 4 — scheduler RUNNING BUT FAILING (the 2026-08-13 ladder) ===")
set_probes(cycles=FAILING)
out = run(fresh_state())
msg_g = out["message"]
ck("one finding", len(out["findings"]) == 1)
ck("key is failing_streak", out["findings"][0].key == "scheduler:failing_streak")
ck("says RUNNING BUT FAILING", "RUNNING BUT FAILING" in msg_g)
ck("counts 4 consecutive failures", "4 consecutive" in msg_g)
ck("calls it a PIPELINE problem", "PIPELINE problem" in msg_g)
ck("explicitly says do NOT reload the agent", "do NOT " in msg_g and "reload" in msg_g)
ck("distinct message from the not-running case", "NOT RUNNING" not in msg_g)
if SHOW:
    show("Scheduler running but failing", msg_g)


print("\n=== G2. tee-duplicated lines do not inflate the FAIL streak ===")
two_fails = sched_log([(5, "FAIL"), (35, "FAIL")], NOW)
events = H._parse_scheduler_events(two_fails)
ck("duplicate lines collapsed", len(events) == 4, f"got {len(events)} events (raw lines: 8)")
ck("streak counts 2 real cycles, not 4", H._trailing_fail_streak(events) == 2)
set_probes(cycles=[(5, "FAIL"), (35, "FAIL")])
out = run(fresh_state())
ck("2 failing cycles stay under the threshold (no alert)", out["findings"] == [])

print("\n=== G3. a PASS resets the streak ===")
mixed = sched_log([(5, "FAIL"), (35, "FAIL"), (65, "FAIL"), (95, "PASS"), (125, "FAIL")], NOW)
ck("streak stops at the PASS", H._trailing_fail_streak(H._parse_scheduler_events(mixed)) == 3)


print("\n=== G4. Check 4 is skipped outside the equity window ===")
for label, when in [
    ("Saturday 14:00 ET", datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)),
    ("weekday 21:00 ET", datetime(2026, 8, 18, 1, 0, tzinfo=timezone.utc)),
    ("weekday 16:20 ET", datetime(2026, 8, 17, 20, 20, tzinfo=timezone.utc)),
    ("weekday 09:40 ET (warmup)", datetime(2026, 8, 17, 13, 40, tzinfo=timezone.utc)),
]:
    set_probes(cycles=STALLED)
    out = run(fresh_state(), now=when)
    sched = [r for r in out["results"] if r.name == "Scheduler forward progress"][0]
    ck(f"{label}: skipped, no false alarm", sched.skipped and out["findings"] == [],
       sched.detail)


# ═════════════════════════════════════════════════════════════════════════════
print("\n=== H. De-duplication — same failure across consecutive runs ===")
state = fresh_state()
set_probes(ollama=(None, ("timeout", "GET http://localhost:11434/api/tags timed out after 10s")))

out1 = run(state)
first_posts = len(_posts)
out2 = run(state, now=NOW + timedelta(minutes=30))
second_posts = len(_posts)
out3 = run(state, now=NOW + timedelta(minutes=60))

ck("run 1 posts", out1["action"] == "new" and first_posts == 1)
ck("run 2 (30m later, same failure) posts NOTHING", out2["action"] == "silent" and second_posts == 0)
ck("run 3 (60m later) still silent", out3["action"] == "silent" and len(_posts) == 0)
ck("finding is still detected while suppressed", len(out2["findings"]) == 1)

out4 = run(state, now=NOW + timedelta(hours=H.REALERT_AFTER_HOURS, minutes=1))
ck(f"re-alerts once past {H.REALERT_AFTER_HOURS}h", out4["action"] == "reminder" and len(_posts) == 1)
ck("reminder says how long it has been unresolved", "Still unresolved after" in out4["message"])
if SHOW:
    show("4-hour reminder", out4["message"])


print("\n=== I. A CHANGED failure re-alerts immediately ===")
state = fresh_state()
set_probes(ollama=(None, ("timeout", "timed out after 10s")))
run(state)
set_probes(
    ollama=(None, ("timeout", "timed out after 10s")),
    ibkr=(False, "connection refused (errno 61)"),
)
out = run(state, now=NOW + timedelta(minutes=30))
ck("new failure set posts despite being inside the 4h window", out["action"] == "new")
ck("posted once", len(_posts) == 1)
ck("both failures named", "Ollama" in out["message"] and "7497" in out["message"])
ck("header counts 2 of 4", "2 of 4 evaluated checks FAILING" in out["message"])
if SHOW:
    show("Two simultaneous failures", out["message"])


print("\n=== J. Recovery fires exactly once, then silence ===")
state = fresh_state()
set_probes(ibkr=(False, "connection refused (errno 61)"))
run(state)
set_probes()  # everything healthy again
out_rec = run(state, now=NOW + timedelta(minutes=30))
ck("recovery posted", out_rec["action"] == "recovery" and len(_posts) == 1)
ck("recovery names what cleared", "7497" in out_rec["message"])
ck("recovery is visibly a recovery", "RECOVERED" in out_rec["message"])
if SHOW:
    show("Recovery", out_rec["message"])

out_after = run(state, now=NOW + timedelta(minutes=60))
ck("no repeat recovery on the next healthy run", out_after["action"] == "silent" and len(_posts) == 0)


print("\n=== K. Probe failures are surfaced, not swallowed ===")
set_probes(agents=(None, "`launchctl list` timed out after 15s"))
out = run(fresh_state())
ck("unknown agent state is a finding", len(out["findings"]) == 1)
ck("says it cannot determine", "Cannot determine" in out["message"])

H._read_scheduler_log_tail = lambda: (None, "/Users/jelmore/Kairos/kairos_scheduler.log does not exist")
out = run(fresh_state())
ck("unreadable log is a finding", any(f.key == "scheduler:log_unreadable" for f in out["findings"]))


print("\n=== L. Boot grace suppresses everything shortly after reboot ===")
H._boot_age_minutes = lambda: 2.0
set_probes(agents=(set(), ""), ibkr=(False, "refused"), cycles=STALLED)
out = run(fresh_state())
ck("nothing checked within the boot grace", out["findings"] == [] and out["results"] == [])
ck("zero Slack calls", len(_posts) == 0)
H._boot_age_minutes = lambda: 999.0


# ═════════════════════════════════════════════════════════════════════════════
total, passed = len(_results), sum(_results)
print(f"\n{'=' * 62}\n  {passed}/{total} assertions passed\n{'=' * 62}")
sys.exit(0 if passed == total else 1)
