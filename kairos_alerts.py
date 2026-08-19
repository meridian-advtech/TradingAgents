"""
Kairos Alerts — Shared Slack (Bot API) + Monitor Log Helpers

Centralizes alert emission so all pipeline components use the same
plumbing: kairos_run.py (timeout alerts), kairos_execute.py (trade
alerts), and any future alerting needs.

Uses the Slack Bot API via slack_sdk.

The bot token comes from the SLACK_BOT_TOKEN environment variable (set it in
~/.zshrc or the launchd plist). It must NEVER be written to kairos_config.json:
that file is tracked in git, and a live token committed there on 2026-08-02 had
to be scrubbed from history. slack.bot_token remains only as a legacy fallback
and is expected to stay "".

Channel IDs (not secrets) stay in kairos_config.json:

    slack.channels.alerts    — channel ID for trade alerts, timeouts
    slack.channels.reports   — channel ID for daily performance
    slack.channels.commands  — channel ID for inbound commands
    slack.channels.log       — channel ID for verbose pipeline logs

Every public function is fault-tolerant: prints a warning on failure
but never raises, so a Slack outage can never block a trade.

Usage:
    python3 kairos_alerts.py --test     # send a test message
    python3 kairos_alerts.py --setup    # create kairos-* channels (needs scope)
"""

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MONITOR_LOG = os.path.join(SCRIPT_DIR, "kairos_monitor.log")

# ── Slack transport ───────────────────────────────────────────────────
# Slack is reached by spawning curl (see _slack_api_call for why curl, and
# why posix_spawn rather than subprocess). /usr/bin/curl is preferred
# explicitly: it is the Apple-signed binary the host's network filter
# permits, and posix_spawn needs an absolute path anyway.
_CURL = "/usr/bin/curl" if os.path.exists("/usr/bin/curl") else (
    shutil.which("curl") or "")
_CURL_TIMEOUT_S = 15      # curl's own -m, the network-level timeout
_OUTER_TIMEOUT_S = 20     # our backstop if curl itself wedges
_SLACK_TRANSPORT = "os.posix_spawn"   # diagnostics only; see _slack_api_call


# ── Config ────────────────────────────────────────────────────────────

def _load_slack_config() -> dict:
    """Slack config: token from the ENVIRONMENT, channel IDs from config.

    Resolution order for the token is env-first and deliberate:
      1. SLACK_BOT_TOKEN — the only supported home for a live token.
      2. kairos_config.json slack.bot_token — legacy fallback, expected "".

    kairos_config.json is tracked in git, so any live value placed in it is one
    `git commit` away from being published; that happened on 2026-08-02 and cost
    a history rewrite. Keeping the field blank and the secret in the environment
    makes the accident structurally impossible rather than merely discouraged.

    This is the single chokepoint every Slack path resolves through
    (kairos_alerts, kairos_axis_weights, kairos_slack_cards).
    """
    config_file = os.path.join(SCRIPT_DIR, "kairos_config.json")
    defaults = {
        "bot_token": "",
        "channels": {
            "alerts": "",
            "reports": "",
            "commands": "",
            "log": "",
            "trades": "",
            "watchlist": "",
        },
    }
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            slack = cfg.get("slack", {})
            if slack.get("bot_token"):
                defaults["bot_token"] = slack["bot_token"]
                print("  WARNING: kairos_config.json holds a slack.bot_token — "
                      "that file is tracked in git. Move it to SLACK_BOT_TOKEN "
                      "and reset the field to \"\".")
            if slack.get("channels"):
                defaults["channels"].update(slack["channels"])
        except (json.JSONDecodeError, IOError):
            pass
    # Env wins outright: a token in the environment supersedes anything the
    # tracked config happens to carry.
    env_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if env_token:
        defaults["bot_token"] = env_token
    return defaults


def _token_fingerprint(token: str) -> str:
    """Non-reconstructable hint that a token is present — for operator output.

    Never returns enough of the secret to be useful if it lands in a log: the
    scheme prefix, a length, and the last 4 characters only.
    """
    if not token:
        return "(none)"
    scheme = token.split("-", 1)[0] if "-" in token else "token"
    return f"{scheme}-…{token[-4:]} (len {len(token)})"


_NO_TOKEN_MSG = ("Set SLACK_BOT_TOKEN in the environment (~/.zshrc or the "
                 "launchd plist). Do NOT put it in kairos_config.json; that "
                 "file is tracked in git.")


# ── Low-level helpers ─────────────────────────────────────────────────

def post_message(
    channel_key: str,
    text: str,
    blocks: list | None = None,
    cfg: dict | None = None,
) -> bool:
    """Post a message to a Slack channel by logical name.

    Args:
        channel_key: one of "alerts", "reports", "commands", "log",
                     OR a literal channel ID (e.g. "C0A0XJYNM29").
        text:        plain-text fallback / notification text.
        blocks:      optional Block Kit blocks for rich formatting.
        cfg:         override for Slack config.

    Returns True on success, False on failure (never raises).
    """
    if cfg is None:
        cfg = _load_slack_config()

    token = cfg.get("bot_token", "")
    if not token:
        print(f"  Slack skipped — no bot token. {_NO_TOKEN_MSG}")
        return False

    channels_map = cfg.get("channels", {})
    channel_id = channels_map.get(channel_key, channel_key)

    if not channel_id:
        print(f"  Slack skipped — no channel ID for '{channel_key}'")
        return False

    payload = {"channel": channel_id, "text": text}
    if blocks:
        payload["blocks"] = blocks

    resp = _slack_api_call("chat.postMessage", token, payload)
    if resp.get("ok"):
        print(f"  Slack → {channel_key} ({channel_id})")
        return True

    print(f"  WARNING: Slack post failed: {resp.get('error', 'unknown')}")
    return False


def _slack_api_call(method: str, token: str, payload: dict) -> dict:
    """Call a Slack Web API method over HTTPS via curl. Never raises.

    Why curl instead of slack_sdk / urllib: on this host an endpoint network
    filter blocks the Python interpreter's socket connections to Slack's IP
    range (every connect() returns EBADF, "Bad file descriptor"), while
    Apple-signed curl is permitted. slack_sdk, requests, and raw sockets all
    fail identically; curl is the only transport that reaches Slack here.

    Why os.posix_spawn instead of subprocess.run: subprocess launches children
    via fork()+exec(), and fork() runs every pthread_atfork child handler in
    the child before exec() replaces it. This process is large, multi-threaded
    and heavily networked (numpy/pandas/scipy/sklearn loaded, live sockets
    open), and on this host Little Snitch and Tailscale both install
    NetworkExtension filters -- so Network.framework registers an atfork child
    handler that walks its own global state. On 2026-08-18 that handler
    segfaulted 27 times in a 20-minute window:

        EXC_BAD_ACCESS in os_log_preferences_refresh
          <- NEFlowDirectorDestroy <- nw_settings_child_has_forked
          <- _pthread_atfork_child_handlers <- fork
          <- _posixsubprocess.subprocess_fork_exec

    Each crash is a silently dropped Slack message: the child dies pre-exec,
    so curl never runs and the parent just sees exit -11.

    posix_spawn(2) is not fork() -- it does not duplicate the parent's threads
    or address space and it does not run atfork handlers, so this crash class
    is unreachable through it rather than merely less likely. Note that
    subprocess could not have been coaxed onto its own posix_spawn fast path
    here: that path requires close_fds=False (the default is True) and an
    executable containing a directory separator, so subprocess.run(["curl",
    ...]) always took fork_exec.

    stdin/stdout/stderr go through short-lived 0600 tempfiles rather than
    pipes: posix_spawn has no built-in pipe plumbing, and files also remove
    any chance of a pipe-buffer deadlock on a large response body. The extra
    filesystem round-trip is irrelevant at this call rate.

    Returns the parsed JSON response, or {"ok": False, "error": ...} on any
    transport failure.
    """
    if not _CURL:
        return {"ok": False, "error": "curl not found on this host"}

    url = f"https://slack.com/api/{method}"
    body = json.dumps(payload)

    # Mirror subprocess's restore_signals=True, which resets these to default
    # in the child (see _Py_RestoreSignals in Python/pylifecycle.c).
    setsigdef = [s for s in (getattr(signal, n, None)
                             for n in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"))
                 if s is not None]

    in_fd, in_path = tempfile.mkstemp(prefix="kairos-slack-req-")
    out_fd, out_path = tempfile.mkstemp(prefix="kairos-slack-out-")
    err_fd, err_path = tempfile.mkstemp(prefix="kairos-slack-err-")
    for fd in (in_fd, out_fd, err_fd):
        os.close(fd)

    try:
        with open(in_path, "w") as f:
            f.write(body)

        argv = [
            "curl", "-sS", "-m", str(_CURL_TIMEOUT_S), "-X", "POST", url,
            "-H", f"Authorization: Bearer {token}",
            "-H", "Content-Type: application/json; charset=utf-8",
            "--data-binary", f"@{in_path}",
        ]
        file_actions = (
            (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0o666),
            (os.POSIX_SPAWN_OPEN, 1, out_path,
             os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
            (os.POSIX_SPAWN_OPEN, 2, err_path,
             os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
        )

        try:
            pid = os.posix_spawn(_CURL, argv, os.environ,
                                 file_actions=file_actions,
                                 setsigdef=setsigdef)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": f"curl spawn failed: {exc}"}

        rc, reap_err = _reap(pid, _OUTER_TIMEOUT_S)
        if reap_err:
            return {"ok": False, "error": reap_err}

        stdout = _read_text(out_path)
        if rc != 0:
            stderr = _read_text(err_path).strip()
            return {"ok": False, "error": f"curl exit {rc}: {stderr}"}
        try:
            return json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return {"ok": False, "error": f"non-JSON response: {stdout[:200]}"}
    except OSError as exc:
        return {"ok": False, "error": f"curl transport failed: {exc}"}
    finally:
        for p in (in_path, out_path, err_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _reap(pid: int, timeout_s: float) -> tuple:
    """Wait for pid, SIGKILLing it past timeout_s.

    Returns (exit_code, None) on a normal exit -- negative for a terminating
    signal, matching subprocess -- or (None, reason) if the child had to be
    killed or its status could not be collected.

    posix_spawn gives back a bare pid, so the timeout that subprocess.run()
    provided has to be done by hand: poll with WNOHANG rather than block, so
    a wedged curl can still be killed.
    """
    deadline = time.monotonic() + timeout_s
    delay = 0.002
    while True:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            # Something else already reaped it (a stray SIGCHLD handler, or
            # SIGCHLD set to SIG_IGN somewhere in this process). curl may well
            # have succeeded, but its status is gone -- report a transport
            # failure rather than trusting an unverified stdout.
            return None, "curl status unavailable (child already reaped)"
        if done == pid:
            return os.waitstatus_to_exitcode(status), None
        if time.monotonic() >= deadline:
            break
        time.sleep(delay)
        delay = min(delay * 1.5, 0.05)

    # Timed out: kill, then reap so we never leave a zombie behind.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except (ChildProcessError, OSError):
        pass
    return None, f"curl timed out after {timeout_s}s"


def _read_text(path: str) -> str:
    """Read a spawned child's output file. Never raises."""
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_monitor_event(event: str, **fields) -> None:
    """Append a JSON record to kairos_monitor.log."""
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": event,
        **fields,
    }
    try:
        with open(MONITOR_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except IOError as exc:
        print(f"  WARNING: Could not write to monitor log: {exc}")


# ── High-level alert functions ────────────────────────────────────────

def alert_reasoning_timeout(timeout_s: int) -> None:
    """Alert: Phase 2 reasoning timed out.  → alerts channel."""
    log_monitor_event(
        "PHASE2_TIMEOUT",
        phases=["reason"],
        asset_class="equity",
        success=False,
        error=f"Claude Code reasoning timed out after {timeout_s}s",
        duration_s=timeout_s,
    )
    print(f"  Timeout logged to {MONITOR_LOG}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    post_message(
        "alerts",
        f":warning: *Kairos Phase 2 Timeout*\n"
        f"Claude Code reasoning did not produce a decision "
        f"within {timeout_s}s.\n"
        f"No trade will be executed this cycle.\n"
        f"Timestamp: `{ts}`\n"
        f"Action required: check Claude CLI availability and "
        f"kairos_prompt.txt contents.",
    )


def alert_trade_executed(
    decision: dict,
    execution: dict,
    confluence: dict | None = None,
    runner_up_signals: list[str] | None = None,
    conviction_trade: bool = False,
    closed_lots: list[dict] | None = None,
    source_tier: str | None = None,
) -> None:
    """Alert: Trade was filled — lists all co-firing signals.  → alerts channel.

    Only call this for Submitted/Filled trades — never for Skipped/Cancelled.

    Args:
        decision:          the decision dict (action, ticker, quantity, rationale…)
        execution:         execution result (status, fill_price…)
        confluence:        confluence scoring result (from compute_confluence)
        runner_up_signals: signal tags for the runner-up ticker (optional)
        conviction_trade:  True if this is a Mode C conviction-only trade
        closed_lots:       for SELL fills — list of closed holding lots from DB
                           [{entry_price, quantity, holding_days}, ...]
        source_tier:       "A", "B", or "C" — prepend badge for Tier C hits
    """
    # Defense in depth: only post for actual fills, never for HOLD/Skipped/etc.
    status = execution.get("status", "?")
    if status not in ("Filled", "Submitted"):
        return

    action = decision.get("action", "?")
    ticker = decision.get("ticker", "?")
    qty = decision.get("quantity", 0)
    fill_price = execution.get("fill_price")
    rationale = decision.get("rationale", "")
    runner_up = decision.get("runner_up", "")

    if conviction_trade:
        emoji = ":brain:"
        title = "Kairos Conviction-Only Trade"
    elif action == "BUY":
        emoji = ":chart_with_upwards_trend:"
        title = "Kairos Trade Executed"
    elif action == "SELL":
        emoji = ":chart_with_downwards_trend:"
        title = "Kairos Trade Executed"
    else:
        emoji = ":pause_button:"
        title = "Kairos Trade Executed"

    price_str = f" @ ${fill_price:.2f}" if fill_price else ""

    # ── SELL P&L breakdown ────────────────────────────────────────
    sell_str = ""
    if action.upper() == "SELL" and fill_price and closed_lots:
        total_cost = sum(lot["entry_price"] * lot["quantity"] for lot in closed_lots)
        total_proceeds = fill_price * sum(lot["quantity"] for lot in closed_lots)
        gross_pnl = total_proceeds - total_cost
        avg_entry = total_cost / sum(lot["quantity"] for lot in closed_lots) if closed_lots else 0

        pnl_pct = (gross_pnl / total_cost * 100) if total_cost > 0 else 0.0
        pnl_sign = "+" if gross_pnl >= 0 else ""
        pnl_emoji = ":white_check_mark:" if gross_pnl >= 0 else ":x:"

        sell_str = (
            f"\n{pnl_emoji} *P&L: {pnl_sign}${gross_pnl:,.2f} ({pnl_sign}{pnl_pct:.1f}%)*\n"
            f"Entry: ${avg_entry:,.2f} \u2192 Exit: ${fill_price:,.2f}  |  {qty} shares\n"
        )
        for lot in closed_lots:
            days = lot["holding_days"]
            tax_rate = "long-term" if days >= 365 else "short-term"
            lot_pnl = (fill_price - lot["entry_price"]) * lot["quantity"]
            sell_str += (
                f"  \u2022 {lot['quantity']} shares held {days}d ({tax_rate}) "
                f"\u2192 {'+' if lot_pnl >= 0 else ''}${lot_pnl:,.2f}\n"
            )

    # Confluence section
    conf_str = ""
    if conviction_trade:
        conv_score = decision.get("conviction", "?")
        conf_str = f"\n*Mode C — Conviction {conv_score}/10, no HOT signals (1% NLV cap)*\n"
    elif confluence and confluence.get("score", 0) > 0:
        signals = confluence.get("signals",
                                 list(confluence.get("signals_detail", {}).keys()))
        # multiplier for equity, nlv_pct for crypto
        sizing_str = (
            f"{confluence['multiplier']}x sizing"
            if confluence.get("multiplier")
            else f"{confluence.get('nlv_pct', 0):.2%} NLV"
        )
        conf_str = (
            f"\n*Confluence: {confluence['tier']} "
            f"(score {confluence['score']}, {sizing_str})*\n"
            f"Signals fired:\n"
        )
        for sig in signals:
            conf_str += f"  \u2022 {sig}\n"
    elif confluence:
        conf_str = "\n*Confluence: None (no signals)*\n"

    runner_str = ""
    if runner_up:
        runner_str = f"\nRunner-up: {runner_up}"
        if runner_up_signals:
            runner_str += f" ({', '.join(runner_up_signals)})"

    rat_str = ""
    if rationale:
        rat_str = f"\n_{rationale[:200]}_"

    text = (
        f"{emoji} *{title}*\n"
        f"{action} {qty} {ticker}{price_str} ({status})"
        f"{sell_str}{conf_str}{runner_str}{rat_str}"
    )

    # Prepend Tier C badge if this ticker came from opportunistic watchlist
    if source_tier:
        text = format_tier_prefix(ticker, source_tier, text)

    log_monitor_event(
        "TRADE_EXECUTED",
        action=action,
        ticker=ticker,
        quantity=qty,
        fill_price=fill_price,
        status=status,
        confluence_score=confluence.get("score") if confluence else None,
        confluence_tier=confluence.get("tier") if confluence else None,
        conviction_trade=conviction_trade,
        source_tier=source_tier,
    )

    # Route BUY/SELL fills to trades channel
    post_message("trades", text)


def alert_trade_blocked(
    decision: dict,
    reason: str,
    confluence: dict | None = None,
) -> None:
    """Log a blocked trade to monitor — NO Slack alert for blocked/skipped trades."""
    action = decision.get("action", "?")
    ticker = decision.get("ticker", "?")
    qty = decision.get("quantity", 0)

    log_monitor_event(
        "TRADE_BLOCKED",
        action=action,
        ticker=ticker,
        quantity=qty,
        reason=reason,
        confluence_score=confluence.get("score") if confluence else None,
    )
    # Intentionally no post_message — blocked trades are logged only


def alert_pipeline_event(message: str, channel: str = "log") -> None:
    """Generic pipeline event.  Default → log channel."""
    log_monitor_event("PIPELINE_EVENT", message=message)
    post_message(channel, message)


def send_end_of_day_summary(window: str = "equity") -> None:
    """Send end-of-day summary to #kairos-reports.

    Args:
        window: "equity" (4:15 PM ET weekday) or "crypto" (10:05 PM ET daily)

    Pulls today's trades from kairos.db, current portfolio from IBKR,
    and formats a clean multi-section summary.
    """
    import sqlite3

    DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Query today's executed trades ─────────────────────────────
    buys, sells, realized_pnl, capital_deployed = [], [], 0.0, 0.0
    tickers_traded = set()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT ticker, action, quantity, execution_price, execution_status,
                   data_inputs, conviction_trade
            FROM decisions
            WHERE timestamp LIKE ? || '%'
              AND execution_status IN ('Filled', 'Submitted')
        """, (today,)).fetchall()

        for r in rows:
            r = dict(r)
            ticker = r["ticker"]
            action = r["action"].upper()
            qty = r.get("quantity", 0)
            price = r.get("execution_price") or 0
            tickers_traded.add(ticker)

            if action == "BUY":
                buys.append(r)
                capital_deployed += qty * price
            elif action == "SELL":
                sells.append(r)
                # Compute realized P&L from holdings table
                entry_lots = conn.execute("""
                    SELECT entry_price, quantity FROM holdings
                    WHERE ticker = ? AND sold_date LIKE ? || '%'
                """, (ticker, today)).fetchall()
                for lot in entry_lots:
                    realized_pnl += (price - lot["entry_price"]) * lot["quantity"]

        # ── Skipped count for today ─────────────────────────────
        skipped_count = conn.execute("""
            SELECT COUNT(*) FROM decisions
            WHERE timestamp LIKE ? || '%'
              AND execution_status = 'Skipped'
        """, (today,)).fetchone()[0]

        conn.close()
    except Exception as exc:
        print(f"  WARNING: EOD summary DB query failed: {exc}")
        buys, sells, skipped_count = [], [], 0

    # ── Query IBKR for current portfolio state ───────────────────
    nlv, cash, positions_text = 0.0, 0.0, ""
    unrealized_pnl = 0.0
    try:
        from ib_insync import IB
        import random
        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=random.randint(40, 49), timeout=10)

        wanted = {"NetLiquidation", "TotalCashValue", "UnrealizedPnL"}
        acct = {v.tag: float(v.value)
                for v in ib.accountSummary() if v.tag in wanted}
        nlv = acct.get("NetLiquidation", 0.0)
        cash = acct.get("TotalCashValue", 0.0)
        unrealized_pnl = acct.get("UnrealizedPnL", 0.0)

        pos_lines = []
        for p in ib.positions():
            sym = p.contract.symbol
            qty = float(p.position)
            avg = float(p.avgCost)
            # Request current price for unrealized P&L per position
            ib.reqMarketDataType(4)
            mkt = ib.reqMktData(p.contract)
            ib.sleep(1)
            cur_price = None
            for attr in ("last", "close", "bid", "ask"):
                val = getattr(mkt, attr, None)
                if val is not None and val == val and val > 0:
                    cur_price = float(val)
                    break
            ib.cancelMktData(p.contract)
            if cur_price and qty != 0:
                pos_pnl = (cur_price - avg) * qty
                pnl_sign = "+" if pos_pnl >= 0 else ""
                pos_lines.append(
                    f"  {sym:<6} {qty:>6.0f} sh  "
                    f"avg ${avg:>8.2f}  now ${cur_price:>8.2f}  "
                    f"P&L {pnl_sign}${pos_pnl:,.0f}"
                )
            elif qty != 0:
                pos_lines.append(f"  {sym:<6} {qty:>6.0f} sh  avg ${avg:>8.2f}")

        ib.disconnect()
        positions_text = "\n".join(pos_lines) if pos_lines else "  (no open positions)"
    except Exception as exc:
        print(f"  WARNING: EOD IBKR query failed: {exc}")
        positions_text = "  (IBKR unavailable)"

    # ── Format summary ───────────────────────────────────────────
    window_label = "Equity Close (4:15 PM ET)" if window == "equity" else "Crypto Close (10:05 PM ET)"
    rpnl_sign = "+" if realized_pnl >= 0 else ""
    upnl_sign = "+" if unrealized_pnl >= 0 else ""

    buy_tickers = ", ".join(r["ticker"] for r in buys) or "(none)"
    sell_tickers = ", ".join(r["ticker"] for r in sells) or "(none)"

    text = (
        f":bar_chart: *Kairos End-of-Day Summary — {window_label}*\n"
        f"Date: {today}\n"
        f"\n"
        f"*Trades Executed Today*\n"
        f"  Buys:  {len(buys)} ({buy_tickers})\n"
        f"  Sells: {len(sells)} ({sell_tickers})\n"
        f"  Skipped: {skipped_count}\n"
    )

    if capital_deployed > 0:
        text += f"  Capital deployed: ${capital_deployed:,.0f}\n"
    if sells:
        text += f"  Realized P&L: {rpnl_sign}${realized_pnl:,.2f}\n"

    text += (
        f"\n"
        f"*Portfolio*\n"
        f"  Total value: ${nlv:,.2f}\n"
        f"  Cash:        ${cash:,.2f}\n"
        f"  Unrealized:  {upnl_sign}${unrealized_pnl:,.2f}\n"
        f"\n"
        f"*Open Positions*\n"
        f"```\n{positions_text}\n```"
    )

    # Signal Performance — per-signal aggregate from thesis tracking.
    # Only appends if there are closed trades to report on.
    try:
        from kairos_ml_thesis import format_signal_performance_section
        sig_section = format_signal_performance_section()
        if sig_section:
            text += "\n\n" + sig_section
    except Exception as exc:
        print(f"  WARNING: signal performance section failed: {exc}")

    # IPO Watchlist — active recent-IPO positions + upcoming candidates.
    try:
        ipo_section = _format_ipo_watchlist_section()
        if ipo_section:
            text += "\n\n" + ipo_section
    except Exception as exc:
        print(f"  WARNING: IPO watchlist section failed: {exc}")

    log_monitor_event("EOD_SUMMARY", window=window, buys=len(buys),
                      sells=len(sells), skipped=skipped_count,
                      capital_deployed=capital_deployed,
                      realized_pnl=realized_pnl, nlv=nlv, cash=cash)

    post_message("reports", text)


def _format_ipo_watchlist_section() -> str:
    """Render IPO positions + upcoming watchlist for the EOD report.

    Empty string when nothing relevant — keeps the EOD message tight.
    """
    try:
        from kairos_signals_ipo import (
            get_ipo_tickers, get_ipo_context,
        )
    except Exception:
        return ""

    # --- Active IPO positions ---------------------------------------
    ipo_catalog = get_ipo_tickers() or {}
    active_lines: list[str] = []
    try:
        import sqlite3
        kdb = os.path.join(SCRIPT_DIR, "kairos.db")
        if os.path.exists(kdb) and ipo_catalog:
            conn = sqlite3.connect(kdb)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT ticker,
                          SUM(quantity) AS qty,
                          SUM(entry_price * quantity)/SUM(quantity) AS avg_cost,
                          MIN(entry_date) AS first_entry
                   FROM holdings
                   WHERE sold_date IS NULL
                   GROUP BY ticker"""
            ).fetchall()
            conn.close()
            holdings = {r["ticker"]: dict(r) for r in rows}
            for ticker in sorted(ipo_catalog.keys()):
                if ticker not in holdings:
                    continue
                h = holdings[ticker]
                ctx = get_ipo_context(ticker)
                ipo_price = ctx.get("ipo_price")
                cur_vs_ipo = ctx.get("current_vs_ipo_pct")
                days_held = ctx.get("days_since_ipo") or 0
                cur_str = (
                    f"{cur_vs_ipo:+.1f}%" if cur_vs_ipo is not None else "n/a"
                )
                ipo_str = f"${ipo_price:.2f}" if ipo_price else "n/a"
                active_lines.append(
                    f"  {ticker:<6} held {days_held}d  "
                    f"qty={float(h['qty']):.0f}  IPO {ipo_str}  "
                    f"vs IPO {cur_str}"
                )
    except Exception as exc:
        active_lines = [f"  (IPO holdings query failed: {exc})"]

    # --- Upcoming watchlist (still pre-ticker) ----------------------
    upcoming_lines: list[str] = []
    try:
        from kairos_ipo_intake import (
            get_watchlist_status, _load_cache as _ipo_cache,
        )
        news_cache = (_ipo_cache().get("pre_ipo_news") or {})
        for s in get_watchlist_status():
            if s.get("is_live"):
                continue  # surfaced in the active section instead
            news = news_cache.get(s["name"]) or {}
            arts = news.get("article_count") or 0
            vel = news.get("velocity_7d") or 0.0
            upcoming_lines.append(
                f"  {s['name'][:24]:<24}  "
                f"{s['expected_ticker']:<7}  "
                f"news 7d={arts}  velocity={vel:+.0f}"
            )
    except Exception as exc:
        upcoming_lines = [f"  (watchlist status failed: {exc})"]

    if not active_lines and not upcoming_lines:
        return ""

    parts = ["*IPO Watchlist*"]
    if active_lines:
        parts.append("Active recent-IPO positions:")
        parts.append("```\n" + "\n".join(active_lines) + "\n```")
    if upcoming_lines:
        parts.append("Upcoming (still pre-ticker):")
        parts.append("```\n" + "\n".join(upcoming_lines) + "\n```")
    return "\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────

def _cli_test():
    """Send a test message to verify Slack connectivity."""
    cfg = _load_slack_config()
    token = cfg.get("bot_token", "")

    print("=" * 60)
    print("  KAIROS ALERTS — Connectivity Test")
    print("=" * 60)

    if not token:
        print(f"  ERROR: no bot token. {_NO_TOKEN_MSG}")
        sys.exit(1)

    src = "SLACK_BOT_TOKEN (env)" if os.environ.get("SLACK_BOT_TOKEN", "").strip() \
          else "kairos_config.json (LEGACY — move it to the environment)"
    print(f"  Bot token: {_token_fingerprint(token)} from {src}")
    print(f"  Channels:")
    for key, cid in cfg.get("channels", {}).items():
        print(f"    {key:<10} → {cid}")

    # Auth test
    auth = _slack_api_call("auth.test", token, {})
    if auth.get("ok"):
        print(f"\n  Bot name:  {auth.get('user')}")
        print(f"  Workspace: {auth.get('team')}")
        print(f"  Bot ID:    {auth.get('user_id')}")
    else:
        print(f"\n  Auth FAILED: {auth.get('error', 'unknown')}")
        sys.exit(1)

    # Send test to each configured channel
    print()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for key, cid in cfg.get("channels", {}).items():
        if not cid or cid.startswith("_"):
            print(f"  {key}: skipped (no ID)")
            continue
        ok = post_message(
            key,
            f":white_check_mark: *Kairos Alerts Online*\n"
            f"Channel: `{key}` | Test at `{ts}`\n"
            f"slack_sdk integration verified.",
            cfg=cfg,
        )
        status = "OK" if ok else "FAILED"
        print(f"  {key}: {status}")

    print("\n" + "=" * 60)


def _cli_setup():
    """Attempt to create kairos-* channels (requires channels:manage scope)."""
    cfg = _load_slack_config()
    token = cfg.get("bot_token", "")

    if not token:
        print(f"  ERROR: no bot token. {_NO_TOKEN_MSG}")
        sys.exit(1)

    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=token)
    desired = {
        "alerts": "kairos-alerts",
        "reports": "kairos-reports",
        "commands": "kairos-commands",
        "log": "kairos-log",
    }

    created = {}
    for key, name in desired.items():
        try:
            resp = client.conversations_create(name=name, is_private=False)
            cid = resp["channel"]["id"]
            created[key] = cid
            print(f"  Created #{name} → {cid}")
        except SlackApiError as e:
            err = e.response["error"]
            if err == "name_taken":
                print(f"  #{name} already exists (look up ID manually)")
            elif err == "missing_scope":
                print(f"\n  Bot needs 'channels:manage' scope to create channels.")
                print(f"  Create them manually in Slack, then update kairos_config.json")
                print(f"  with the channel IDs (right-click channel → View channel details → ID at bottom).")
                print(f"\n  Channels needed:")
                for k, n in desired.items():
                    print(f"    {k:<10} → #{n}")
                sys.exit(1)
            else:
                print(f"  #{name}: {err}")

    if created:
        # Update config file
        config_file = os.path.join(SCRIPT_DIR, "kairos_config.json")
        with open(config_file) as f:
            full_cfg = json.load(f)
        for key, cid in created.items():
            full_cfg["slack"]["channels"][key] = cid
        with open(config_file, "w") as f:
            json.dump(full_cfg, f, indent=2)
        print(f"\n  Updated kairos_config.json with new channel IDs")


def format_tier_prefix(ticker: str, source_tier: str, text: str) -> str:
    """Prepend a Tier C badge to alert text when source_tier is 'C'."""
    if source_tier == "C":
        return f"\U0001f195 TIER C | {text}"
    return text


# ── !add-ticker command handler ──────────────────────────────────────

def handle_add_ticker(symbol: str, reason: str) -> dict:
    """Handle the !add-ticker Slack command.

    Uses yfinance to resolve ticker name/sector, then delegates to
    kairos_tier_c.add(). Returns a dict suitable for Slack reply.

    Args:
        symbol: ticker symbol (e.g. "PLTR")
        reason: user-provided reason string

    Returns:
        {"ok": True, "message": "..."} or {"ok": False, "error": "..."}
    """
    symbol = symbol.upper().strip()

    # Resolve name and sector via yfinance
    name = None
    sector = None
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName")
        sector = info.get("sector")
    except Exception as exc:
        print(f"  WARNING: yfinance lookup failed for {symbol}: {exc}")

    if not name:
        msg = (
            f"Could not resolve ticker `{symbol}` via yfinance. "
            f"Please add manually:\n"
            f"`python kairos_tier_c.py add {symbol} \"Company Name\" \"Sector\" \"{reason}\"`"
        )
        post_message("commands", msg)
        return {"ok": False, "error": msg}

    if not sector:
        sector = "Unknown"

    # Delegate to kairos_tier_c
    sys.path.insert(0, SCRIPT_DIR)
    from kairos_tier_c import add as tier_c_add
    result = tier_c_add(symbol, name, sector, reason)

    if result["ok"]:
        entry = result["entry"]
        reply = (
            f":white_check_mark: Added `${symbol}` ({name}) to Tier C.\n"
            f"Sector: {sector} | Expires: {entry['expires_date']}\n"
            f"Reason: {reason}"
        )
        post_message("commands", reply)
        return {"ok": True, "message": reply}
    else:
        post_message("commands", f":x: {result['error']}")
        return result


def main():
    parser = argparse.ArgumentParser(description="Kairos Alerts — Slack integration")
    parser.add_argument("--test", action="store_true", help="Send test message to all channels")
    parser.add_argument("--setup", action="store_true", help="Create kairos-* channels (needs scope)")
    args = parser.parse_args()

    if args.setup:
        _cli_setup()
    elif args.test:
        _cli_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
