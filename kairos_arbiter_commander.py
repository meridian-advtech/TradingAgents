"""
Kairos Arbiter Commander — Two-Way Conversational Slack Interface (curl long-poll)

Makes #kairos-arbiter a live conversation with the Mistral Arbiter. Polls the
channel for human messages, loads the full current portfolio context (all closed
trades + enriched open positions + aggregates — the same data a weekly run sees),
passes it plus a rolling conversation history to Mistral, and posts the reply
back to the channel.

Config-change approval workflow:
  When the Arbiter proposes a config change (pausing a signal, adjusting a
  threshold, changing a hold window), it ends its message with a machine-readable
  directive line. We detect it, strip it from the visible reply, and append:
    "Reply APPROVE to apply this change, or IGNORE to skip."
  The next human message decides:
    APPROVE → write the change to kairos_config.json and confirm.
    IGNORE / anything else → discard the pending change.
  Pending changes live in memory only (never persisted). Only one at a time.

Transport: all Slack I/O routes through kairos_alerts._slack_api_call (curl
subprocess), because this host's endpoint filter blocks the Python socket layer
for Slack's IP range. We long-poll conversations.history and reply via
chat.postMessage, both over curl. No Socket Mode / app-level token is used.

The bot never responds to its own messages — bot_id / subtype messages are
skipped, exactly like kairos_commander.py.

────────────────────────────────────────────────────────────────────────────
Launch Agent (persistent process, 10s poll)
~/Library/LaunchAgents/com.kairos.arbiter.commander.plist

  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.kairos.arbiter.commander</string>
    <key>ProgramArguments</key>
    <array>
      <string>/Users/jelmore/Kairos-env/bin/python3</string>
      <string>/Users/jelmore/Kairos/kairos_arbiter_commander.py</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
      <key>MISTRAL_API_KEY</key>
      <string>REPLACE_WITH_MISTRAL_API_KEY</string>
      <key>SLACK_BOT_TOKEN</key>
      <string>REPLACE_WITH_SLACK_BOT_TOKEN</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/jelmore/Kairos/kairos_arbiter_commander.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/jelmore/Kairos/kairos_arbiter_commander.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/jelmore/Kairos</string>
  </dict>
  </plist>

  Load:    launchctl load   ~/Library/LaunchAgents/com.kairos.arbiter.commander.plist
  Unload:  launchctl unload ~/Library/LaunchAgents/com.kairos.arbiter.commander.plist

  NOTE: interpreter is ~/Kairos-env/bin/python3 (the venv with yfinance/requests).
────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import requests

# ── Absolute paths (matches kairos_commander.py convention) ──────────
SCRIPT_DIR = "/Users/jelmore/Kairos"
sys.path.insert(0, SCRIPT_DIR)

from kairos_arbiter import (  # noqa: E402  (path inserted above)
    fetch_closed_trades,
    fetch_open_positions,
    enrich_open_positions,
    compute_aggregates,
    _slim_closed,
    _slim_open,
    MISTRAL_URL,
    MISTRAL_MODEL,
    MISTRAL_TIMEOUT,
)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
COMMANDER_LOG = os.path.join(SCRIPT_DIR, "kairos_arbiter_commander.log")
PID_FILE = "/tmp/kairos_arbiter_commander.pid"

ARBITER_CHANNEL_KEY = "arbiter"
ARBITER_CHANNEL_ID_FALLBACK = "C0BAQH32LTS"   # #kairos-arbiter

POLL_INTERVAL_SECONDS = 10
MAX_HISTORY_EXCHANGES = 10          # rolling window of user/assistant pairs
CONTEXT_TTL_SECONDS = 120           # reuse the heavy portfolio snapshot briefly

CONVO_SYSTEM_PROMPT = (
    "You are the Kairos Arbiter in conversational mode. You have full visibility "
    "into the current trade portfolio: closed trade history with outcomes, and "
    "open positions with live P&L, days held, and thesis window status. Answer "
    "questions about specific positions, signals, or patterns directly and "
    "concisely. When recommending a config change (pausing a signal, adjusting a "
    "threshold, changing a hold window), state the change clearly and explicitly "
    "so the approval workflow can detect it. Keep responses focused — this is a "
    "Slack channel, not a report."
)

# Appended to the per-turn context so the model emits a parseable directive when
# it proposes a change. The base system prompt above is kept verbatim.
DIRECTIVE_INSTRUCTIONS = (
    "If — and only if — you are proposing a concrete configuration change, end "
    "your message with exactly ONE line in this format (and nothing after it):\n"
    "  CONFIG-CHANGE | <type> | <target> | <value> | <one-line summary>\n"
    "where <type> is one of: pause_signal, resume_signal, set_threshold, "
    "set_hold_window.\n"
    "  - pause_signal / resume_signal: <target> is the signal name (e.g. "
    "HOT-EARNINGS); <value> is n/a.\n"
    "  - set_hold_window: <target> is the signal name; <value> is the new hold "
    "window in days (integer).\n"
    "  - set_threshold: <target> is a dotted config path (e.g. "
    "options_signal.iv_rank_threshold or confluence.signal_points.HOT-EARNINGS); "
    "<value> is the new number.\n"
    "Do not include the CONFIG-CHANGE line if you are only answering a question."
)

# ── Logging (FileHandler only — launchd redirects stdout to the same file) ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(COMMANDER_LOG, mode="a")],
)
log = logging.getLogger("kairos-arbiter-commander")


# ── Config / tokens ──────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", CONFIG_FILE, exc)
        return {}


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def get_bot_token() -> str:
    cfg = load_config()
    slack_cfg = cfg.get("slack", {}) if isinstance(cfg.get("slack"), dict) else {}
    bot = os.environ.get("SLACK_BOT_TOKEN") or slack_cfg.get("bot_token", "")
    return bot.strip()


def get_arbiter_channel_id() -> str:
    cfg = load_config()
    channels = cfg.get("slack", {}).get("channels", {}) \
        if isinstance(cfg.get("slack"), dict) else {}
    return channels.get(ARBITER_CHANNEL_KEY) or ARBITER_CHANNEL_ID_FALLBACK


def known_signals() -> set[str]:
    """All HOT-* signal names we recognise, drawn from the live config."""
    cfg = load_config()
    names: set[str] = set()
    sp = cfg.get("confluence", {}).get("signal_points", {})
    if isinstance(sp, dict):
        names.update(sp.keys())
    hw = cfg.get("reallocation", {}).get("signal_hold_windows", {})
    if isinstance(hw, dict):
        names.update(k for k in hw.keys() if k != "DEFAULT")
    # Signals the Arbiter knows about even if absent from the config tables.
    names.update({
        "HOT-EARNINGS", "HOT-INSIDER", "HOT-CONGRESS", "HOT-OPTIONS",
        "HOT-REVERSION", "HOT-CATALYST", "HOT-CHAIN",
    })
    return names


# ── Portfolio context (cached briefly) ───────────────────────────────

_context_cache = {"built_at": 0.0, "today": None, "blob": None,
                  "closed_n": 0, "open_n": 0}


def build_context(today: str) -> tuple[str, int, int]:
    """Full current portfolio snapshot, same data a weekly run assembles.

    Cached for CONTEXT_TTL_SECONDS so rapid follow-up questions don't re-run the
    yfinance enrichment for every position on every message.
    """
    now = time.monotonic()
    if (_context_cache["blob"] is not None
            and _context_cache["today"] == today
            and now - _context_cache["built_at"] < CONTEXT_TTL_SECONDS):
        return (_context_cache["blob"],
                _context_cache["closed_n"], _context_cache["open_n"])

    closed = fetch_closed_trades("weekly", today)
    open_pos = enrich_open_positions(fetch_open_positions())
    stats = compute_aggregates(closed)

    payload = {
        "review_date": today,
        "aggregates": stats,
        "closed_trades": [_slim_closed(t) for t in closed],
        "open_positions": [_slim_open(t) for t in open_pos],
    }
    blob = json.dumps(payload, indent=2, default=str)

    _context_cache.update({
        "built_at": now, "today": today, "blob": blob,
        "closed_n": len(closed), "open_n": len(open_pos),
    })
    log.info("Built portfolio context: %d closed, %d open positions",
             len(closed), len(open_pos))
    return blob, len(closed), len(open_pos)


def build_turn_user_message(today: str, question: str) -> str:
    blob, closed_n, open_n = build_context(today)
    return (
        "CURRENT PORTFOLIO CONTEXT (same data as a weekly Arbiter run). Use it to "
        f"answer the human's question.\nClosed trades: {closed_n}  |  Open "
        f"positions: {open_n}\n\n"
        f"{DIRECTIVE_INSTRUCTIONS}\n\n"
        f"DATA (JSON):\n{blob}\n\n"
        f"HUMAN QUESTION:\n{question}"
    )


# ── Mistral ──────────────────────────────────────────────────────────

def call_mistral_chat(messages: list[dict], api_key: str) -> str:
    resp = requests.post(
        MISTRAL_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "temperature": 0.3,
            "messages": messages,
        },
        timeout=MISTRAL_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Config-change directive parsing + application ────────────────────

_DIRECTIVE_RE = re.compile(
    r"CONFIG-CHANGE\s*\|\s*(?P<type>[a-z_]+)\s*\|\s*(?P<target>[^|]+?)\s*\|\s*"
    r"(?P<value>[^|]+?)\s*\|\s*(?P<summary>.+)",
    re.IGNORECASE,
)

_VALID_TYPES = {"pause_signal", "resume_signal", "set_threshold", "set_hold_window"}


def _coerce_number(raw: str):
    """'10' → 10, '0.55' → 0.55. Returns None if not numeric."""
    raw = raw.strip()
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() and "." not in raw else f


def parse_directive(response: str) -> tuple[str, Optional[dict]]:
    """Split a Mistral reply into (visible_body, pending_change | None).

    The pending change is validated against the live config; an unparseable or
    invalid directive is stripped and ignored (no approval prompt).
    """
    m = _DIRECTIVE_RE.search(response)
    if not m:
        return response.strip(), None

    body = (response[:m.start()] + response[m.end():]).strip()
    ctype = m.group("type").lower()
    target = m.group("target").strip()
    value_raw = m.group("value").strip()
    summary = m.group("summary").strip()

    pending = _validate_directive(ctype, target, value_raw, summary)
    if pending is None:
        log.warning("Ignoring invalid CONFIG-CHANGE directive: %s | %s | %s",
                    ctype, target, value_raw)
    return body, pending


def _validate_directive(ctype: str, target: str, value_raw: str,
                        summary: str) -> Optional[dict]:
    if ctype not in _VALID_TYPES:
        return None

    if ctype in ("pause_signal", "resume_signal"):
        sig = target.upper()
        if sig not in known_signals():
            return None
        return {"type": ctype, "target": sig, "value": None, "summary": summary}

    if ctype == "set_hold_window":
        sig = target.upper()
        days = _coerce_number(value_raw)
        if sig not in known_signals() or not isinstance(days, int) or days <= 0:
            return None
        return {"type": ctype, "target": sig, "value": days, "summary": summary}

    if ctype == "set_threshold":
        num = _coerce_number(value_raw)
        if num is None:
            return None
        # Only allow paths that already resolve to a numeric leaf in the config,
        # so the Arbiter can't invent arbitrary keys.
        path = [p for p in target.split(".") if p]
        if not path:
            return None
        node = load_config()
        for key in path[:-1]:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        if not isinstance(node, dict) or path[-1] not in node:
            return None
        if not isinstance(node[path[-1]], (int, float)) or isinstance(node[path[-1]], bool):
            return None
        return {"type": ctype, "target": target, "value": num, "summary": summary}

    return None


def apply_pending(pending: dict) -> str:
    """Write the approved change to kairos_config.json. Returns a status line."""
    cfg = load_config()
    ctype = pending["type"]
    target = pending["target"]
    value = pending["value"]

    if ctype == "pause_signal":
        lst = cfg.setdefault("paused_signals", [])
        if not isinstance(lst, list):
            lst = []
            cfg["paused_signals"] = lst
        if target in lst:
            return f"`{target}` was already paused — no change."
        lst.append(target)
        save_config(cfg)
        return f"Paused `{target}` (added to `paused_signals`)."

    if ctype == "resume_signal":
        lst = cfg.get("paused_signals", [])
        if isinstance(lst, list) and target in lst:
            lst.remove(target)
            cfg["paused_signals"] = lst
            save_config(cfg)
            return f"Resumed `{target}` (removed from `paused_signals`)."
        return f"`{target}` was not paused — no change."

    if ctype == "set_hold_window":
        realloc = cfg.setdefault("reallocation", {})
        windows = realloc.setdefault("signal_hold_windows", {})
        old = windows.get(target)
        windows[target] = value
        save_config(cfg)
        return (f"Hold window for `{target}`: {old if old is not None else '—'} "
                f"→ {value} days.")

    if ctype == "set_threshold":
        path = [p for p in target.split(".") if p]
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        old = node.get(path[-1])
        node[path[-1]] = value
        save_config(cfg)
        return f"`{target}`: {old} → {value}."

    return "Unknown change type — nothing applied."


def render_proposed_change(pending: dict) -> str:
    detail = {
        "pause_signal": f"pause {pending['target']}",
        "resume_signal": f"resume {pending['target']}",
        "set_hold_window": f"{pending['target']} hold window → {pending['value']} days",
        "set_threshold": f"{pending['target']} → {pending['value']}",
    }.get(pending["type"], pending["type"])
    return (
        f":gear: *Proposed config change*\n"
        f"> {pending['summary']}\n"
        f"> _({detail})_\n"
        f"Reply APPROVE to apply this change, or IGNORE to skip."
    )


# ── Conversation state (in memory only) ──────────────────────────────

_history: deque = deque(maxlen=MAX_HISTORY_EXCHANGES * 2)  # user+assistant msgs
_pending_change: Optional[dict] = None


def _append_exchange(question: str, answer: str) -> None:
    _history.append({"role": "user", "content": question})
    _history.append({"role": "assistant", "content": answer})


# ── Message handling ─────────────────────────────────────────────────

def handle_human_message(text: str, say) -> None:
    global _pending_change

    stripped = text.strip()
    if not stripped:
        return
    first = stripped.split()[0].upper()

    # ── Pending-change resolution takes priority ────────────────────
    if _pending_change is not None:
        if first in ("APPROVE", "APPROVED", "YES", "Y"):
            pending = _pending_change
            _pending_change = None
            try:
                status = apply_pending(pending)
                say(f":white_check_mark: *Change applied.* {status}")
                log.info("Applied config change: %s", pending)
            except Exception as exc:
                log.exception("apply_pending failed")
                say(f":x: Failed to apply change: `{exc}`")
            return
        if first in ("IGNORE", "NO", "N", "CANCEL", "SKIP"):
            discarded = _pending_change
            _pending_change = None
            say(f":wastebasket: Discarded pending change: _{discarded['summary']}_")
            return
        # Anything else discards the pending change, then we treat the message
        # as a fresh question so the human is never left hanging.
        discarded = _pending_change
        _pending_change = None
        say(f":wastebasket: Pending change discarded (no APPROVE): "
            f"_{discarded['summary']}_ — treating your message as a new question.")

    # ── Normal conversational turn ──────────────────────────────────
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        say(":x: MISTRAL_API_KEY is not set — I can't reach the Arbiter model.")
        log.error("MISTRAL_API_KEY missing")
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        user_msg = build_turn_user_message(today, stripped)
    except Exception as exc:
        log.exception("context build failed")
        say(f":x: Could not load portfolio context: `{exc}`")
        return

    messages = [{"role": "system", "content": CONVO_SYSTEM_PROMPT}]
    messages.extend(_history)
    messages.append({"role": "user", "content": user_msg})

    try:
        response = call_mistral_chat(messages, api_key)
    except Exception as exc:
        log.exception("Mistral call failed")
        say(f":x: Arbiter model call failed: `{exc}`")
        return

    body, pending = parse_directive(response)

    if pending is not None:
        _pending_change = pending
        say(body + "\n\n" + render_proposed_change(pending))
        log.info("Pending config change proposed: %s", pending)
    else:
        say(body)

    _append_exchange(stripped, body)


# ── curl say() closure ───────────────────────────────────────────────

def _make_say(token: str, channel_id: str):
    from kairos_alerts import _slack_api_call

    def say(text: str) -> None:
        resp = _slack_api_call(
            "chat.postMessage", token,
            {"channel": channel_id, "text": text},
        )
        if not resp.get("ok"):
            log.warning("chat.postMessage failed: %s", resp.get("error"))

    return say


# ── Single-instance PID lock ─────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock() -> bool:
    for _ in range(2):
        try:
            fd = os.open(PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                with open(PID_FILE) as f:
                    existing = int(f.read().strip())
            except (IOError, ValueError):
                existing = None
            if existing and existing != os.getpid() and _pid_alive(existing):
                log.error("Another arbiter commander is running (pid %s) — exiting.",
                          existing)
                return False
            log.warning("Reclaiming stale lock file (pid %s not running).", existing)
            try:
                os.remove(PID_FILE)
            except OSError:
                pass
            continue
        else:
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            import atexit
            atexit.register(release_lock)
            return True
    log.error("Could not acquire lock after reclaiming stale file — exiting.")
    return False


def release_lock() -> None:
    try:
        with open(PID_FILE) as f:
            if int(f.read().strip()) != os.getpid():
                return
    except (IOError, ValueError):
        return
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


# ── Main poll loop ───────────────────────────────────────────────────

def main() -> None:
    from kairos_alerts import _slack_api_call

    if not acquire_lock():
        sys.exit(0)

    token = get_bot_token()
    if not token:
        log.error("Missing bot token — set SLACK_BOT_TOKEN in the environment "
                  "(never kairos_config.json; it is tracked in git).")
        sys.exit(1)

    channel_id = get_arbiter_channel_id()
    say = _make_say(token, channel_id)

    log.info("Kairos Arbiter Commander starting (curl long-poll, channel=%s)…",
             channel_id)

    # Seed the cursor to now (6-decimal ts to match Slack's format) so we only
    # process NEW human messages, never history.
    last_ts = f"{time.time():.6f}"
    log.info("Polling channel %s, seed last_ts=%s", channel_id, last_ts)

    poll_count = 0
    while True:
        try:
            resp = _slack_api_call(
                "conversations.history", token,
                {"channel": channel_id, "oldest": last_ts, "limit": 10},
            )
            poll_count += 1
            if poll_count % 30 == 0:
                log.info("Heartbeat: poll #%d ok=%s n_msgs=%d last_ts=%s",
                         poll_count, resp.get("ok"),
                         len(resp.get("messages", [])), last_ts)
            if not resp.get("ok"):
                log.warning("conversations.history failed: %s", resp.get("error"))
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Slack returns newest-first; process oldest-first.
            for msg in reversed(resp.get("messages", [])):
                ts = msg.get("ts", "0")
                if float(ts) <= float(last_ts):
                    continue
                last_ts = ts  # advance regardless of message type
                # Never respond to bots (including ourself) or system subtypes.
                if msg.get("bot_id") or msg.get("subtype"):
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                log.info("Received: %s", text)
                try:
                    handle_human_message(text, say=say)
                except Exception:
                    log.exception("handle_human_message crashed")
        except Exception as exc:
            log.warning("Poll error: %s", exc)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Arbiter commander interrupted — exiting.")
    except Exception:
        log.exception("Fatal error in arbiter commander")
        sys.exit(1)
