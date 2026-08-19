"""
Kairos Spawn — launch child processes without fork()ing this one.

WHY THIS EXISTS
────────────────────────────────────────────────────────────────────────────
subprocess.run() launches children via fork()+exec(), and fork() runs every
pthread_atfork child handler in the child before exec() replaces it. Kairos
processes are the worst case for that: large, multi-threaded, already
networked (numpy/pandas/scipy/sklearn loaded, live IBKR and HTTPS sockets),
and on this host Little Snitch and Tailscale both install NetworkExtension
filters, so Network.framework registers an atfork child handler that walks its
own global state. That handler segfaults:

    EXC_BAD_ACCESS in os_log_preferences_refresh
      <- NEFlowDirectorDestroy <- nw_settings_child_has_forked
      <- _pthread_atfork_child_handlers <- fork
      <- _posixsubprocess.subprocess_fork_exec

27 crashes on 2026-08-18, and still escalating on 2026-08-19 (1 at 09:37, 4 at
09:47, 11 at 11:30). The child dies pre-exec, so the program never runs and
the caller just sees exit -11 — a silently skipped launchctl probe, or a
silently skipped Slack post.

posix_spawn(2) is not fork(): it does not duplicate the parent's threads or
address space and it does not run atfork handlers, so the crash class is
unreachable through it rather than merely less likely.

Note that subprocess cannot simply be configured onto its own posix_spawn fast
path. That path additionally requires close_fds=False (the default is True)
and an executable containing a directory separator, so ordinary
subprocess.run(["launchctl", ...]) always takes fork_exec. Verified by reading
CPython 3.11.15's subprocess.py gating conditions.

This module is the generalization of the pattern first written inline in
kairos_alerts.py's _slack_api_call (which stays as-is: it is verified working
in production and was not worth re-touching during a live incident).

DELIBERATELY subprocess-COMPATIBLE
────────────────────────────────────────────────────────────────────────────
run() returns a real subprocess.CompletedProcess and raises a real
subprocess.TimeoutExpired. Importing subprocess for those two types forks
nothing — they are a data holder and an exception class. Call sites therefore
convert by changing `subprocess.run(` to `kairos_spawn.run(` and keep their
existing `except subprocess.TimeoutExpired:` handlers, argument parsing and
timeout values untouched. Changing HOW a child launches should not become an
excuse to change WHAT it does.

DIFFERENCES FROM subprocess.run, and why they are safe here
────────────────────────────────────────────────────────────────────────────
  * Output goes to short-lived 0600 tempfiles, not pipes. posix_spawn has no
    pipe plumbing, and files also remove any chance of a pipe-buffer deadlock
    on a large stdout — `launchctl list` is hundreds of lines.
  * The timeout is hand-rolled (poll waitpid with WNOHANG, then SIGKILL),
    because posix_spawn hands back a bare pid with no waiter attached.
  * cwd is implemented by prefixing /usr/bin/env -C <dir>, since os.posix_spawn
    exposes no chdir file-action. env execs the target, so the pid, exit code
    and signal semantics the caller sees are unchanged; os.chdir() in the
    parent was rejected because these processes are multi-threaded.
  * shell=True is not supported at all. If a call site ever needs a shell, it
    should be given the env-var mitigation instead of being forced through
    here.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # for CompletedProcess / TimeoutExpired ONLY — never .run()
import tempfile
import time
from typing import Optional, Sequence

_ENV_BIN = "/usr/bin/env"

# Reset in the child, mirroring subprocess's restore_signals=True default
# (see _Py_RestoreSignals in Python/pylifecycle.c).
_SETSIGDEF = [s for s in (getattr(signal, n, None)
                          for n in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"))
              if s is not None]


def _resolve(exe: str) -> str:
    """Absolute path for exe, or raise FileNotFoundError like subprocess does."""
    if os.path.sep in exe:
        if not os.path.isfile(exe):
            raise FileNotFoundError(2, "No such file or directory", exe)
        return exe
    found = shutil.which(exe)
    if not found:
        raise FileNotFoundError(2, "No such file or directory", exe)
    return found


def _read(path: str, text: bool):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        raw = b""
    return raw.decode("utf-8", "replace") if text else raw


def _reap(pid: int, timeout: Optional[float]) -> Optional[int]:
    """Wait for pid. Returns the exit code, or None if it had to be killed.

    Negative return means terminated by that signal, matching subprocess.
    """
    if timeout is None:
        try:
            _, status = os.waitpid(pid, 0)
            return os.waitstatus_to_exitcode(status)
        except (ChildProcessError, OSError):
            return None

    deadline = time.monotonic() + timeout
    delay = 0.002
    while True:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            return None
        if done == pid:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() >= deadline:
            break
        time.sleep(delay)
        delay = min(delay * 1.5, 0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)      # reap, so we never leave a zombie behind
    except (ChildProcessError, OSError):
        pass
    return None


def run(argv: Sequence[str], *, capture_output: bool = True, text: bool = True,
        timeout: Optional[float] = None, cwd: Optional[str] = None,
        env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """posix_spawn a child and wait for it. Drop-in for the subprocess.run
    calls Kairos actually makes.

    Raises FileNotFoundError if argv[0] cannot be resolved, and
    subprocess.TimeoutExpired if the child outlives `timeout` — the same
    exceptions, with the same meanings, that the callers already handle.
    """
    argv = [str(a) for a in argv]
    if not argv:
        raise ValueError("argv must be non-empty")

    target = _resolve(argv[0])
    spawn_argv = [target] + list(argv[1:])
    if cwd is not None:
        # See DIFFERENCES above: env -C gives us cwd without a chdir race.
        _resolve(_ENV_BIN)
        spawn_argv = [_ENV_BIN, "-C", cwd] + spawn_argv
        exe = _ENV_BIN
    else:
        exe = target

    out_path = err_path = None
    file_actions = []
    if capture_output:
        out_fd, out_path = tempfile.mkstemp(prefix="kairos-spawn-out-")
        err_fd, err_path = tempfile.mkstemp(prefix="kairos-spawn-err-")
        os.close(out_fd)
        os.close(err_fd)
        file_actions = [
            (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0o666),
            (os.POSIX_SPAWN_OPEN, 1, out_path,
             os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
            (os.POSIX_SPAWN_OPEN, 2, err_path,
             os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
        ]

    try:
        pid = os.posix_spawn(
            exe, spawn_argv, os.environ if env is None else env,
            file_actions=tuple(file_actions) or None,
            setsigdef=_SETSIGDEF,
        )
        rc = _reap(pid, timeout)

        stdout = _read(out_path, text) if out_path else None
        stderr = _read(err_path, text) if err_path else None

        if rc is None:
            raise subprocess.TimeoutExpired(argv, timeout, output=stdout,
                                            stderr=stderr)
        return subprocess.CompletedProcess(argv, rc, stdout, stderr)
    finally:
        for p in (out_path, err_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
