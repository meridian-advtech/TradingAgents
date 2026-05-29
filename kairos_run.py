"""
Kairos Full Pipeline — Milestone 9 / Phase B (Crypto + Equities + ML)

Orchestrates the complete Kairos loop with split intelligence and multi-asset
awareness:

  Phase 0 (ORCHESTRATE)  — Ollama checks stale sources, filters news (local, fast)
  Phase 0.5 (SCREEN)     — Ollama Tier 1: screen 250+ tickers → max 15 shortlist
  Phase 0.7 (ML)         — ML Track 1: pattern recognition scoring on shortlist
  Phase 1 (GATHER)       — Run snapshot + pull IBKR portfolio → kairos_prompt.txt
  Phase 1C (GATHER CRYPTO) — CoinGecko data + risk guards + Ollama routing
  Phase 2 (REASON)       — Anthropic API reads the prompt, reasons, writes decision
  Phase 3 (EXECUTE)      — Read decision, apply risk guards, place order via IBKR

Asset modes (set by scheduler via --mode):
  equity  — equities only (original behavior, market hours 9:30–16:00 ET)
  crypto  — crypto only (outside market hours, 8:00–9:30 or 16:00–22:00 ET)
  both    — equities + crypto (overlap window 9:30–16:00 ET)

Division of labor:
  Ollama (local)   — stale detection, news filtering, crypto signal classification
  sklearn ML       — pattern recognition on trade outcomes (RandomForest)
  Anthropic API    — equity BUY/SELL/HOLD, crypto BUY/SELL/HOLD, tax analysis

Usage:
  python kairos_run.py                       # Run all phases (equity mode)
  python kairos_run.py --gather              # Phase 1 only
  python kairos_run.py --gather --mode both  # Equity + crypto gather
  python kairos_run.py --gather --mode crypto  # Crypto-only gather
  python kairos_run.py --execute             # Phase 3 only
  python kairos_run.py --orchestrate         # Phase 0 only
  python kairos_run.py --no-ollama           # Skip Phase 0
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

W = 72
DECISIONS_LOG = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
CRYPTO_PROMPT_FILE = os.path.join(SCRIPT_DIR, "kairos_crypto_prompt.txt")
CRYPTO_DECISIONS_LOG = os.path.join(SCRIPT_DIR, "kairos_crypto_decisions.log")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
STATE_FILE = os.path.join(SCRIPT_DIR, "kairos_state.json")


# ── IBKR error 10091 suppression ─────────────────────────────────────
# ib_insync logs "Error 10091, reqId …: Part of requested market data
# requires additional subscription" at ERROR level for every single
# options contract, flooding the terminal.  This filter suppresses
# those messages from console output while still writing them to the
# debug log file.

class _Suppress10091(logging.Filter):
    """Drop log records that contain IBKR error code 10091."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "10091" not in record.getMessage()


def _install_ibkr_error_filter():
    """Install the 10091 filter on the ib_insync logger hierarchy.

    - Console output: 10091 messages are suppressed (filter on root/ib_insync)
    - Debug file:     all IBKR messages including 10091 are captured
    """
    suppress = _Suppress10091()

    # Apply filter to all ib_insync loggers so 10091 never reaches console
    for name in ("ib_insync", "ib_insync.wrapper", "ib_insync.client", "ib_insync.ib"):
        logger = logging.getLogger(name)
        logger.addFilter(suppress)

    # Also filter the root logger's handlers (catches lastResort fallback)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(suppress)

    # Add a DEBUG-level file handler so 10091 is still captured for diagnostics
    debug_log = os.path.join(SCRIPT_DIR, "kairos_ibkr_debug.log")
    try:
        fh = logging.FileHandler(debug_log, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        # No filter on the file handler — it gets everything including 10091
        logging.getLogger("ib_insync").addHandler(fh)
        logging.getLogger("ib_insync").setLevel(logging.DEBUG)
    except IOError:
        pass  # non-critical if file handler fails


_install_ibkr_error_filter()


def phase_banner(phase: int | str, title: str):
    print()
    print("╔" + "═" * W + "╗")
    print(f"║  PHASE {phase}: {title}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")


def _parse_log_objects(path: str) -> list[dict]:
    """Extract all top-level JSON objects from a decision log file."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        content = f.read()
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(content[start : i + 1]))
                except json.JSONDecodeError:
                    pass
    return objects


def _clear_stale_raw_decisions() -> int:
    """Remove non-EXECUTION (raw) decision entries from the decision log.

    These are stale decisions from previous cycles that would be mistaken
    for fresh reasoning output.  EXECUTION entries are preserved.

    Returns the number of stale entries removed.
    """
    if not os.path.exists(DECISIONS_LOG):
        return 0

    objects = _parse_log_objects(DECISIONS_LOG)
    kept = [obj for obj in objects if obj.get("type") == "EXECUTION"]
    removed = len(objects) - len(kept)

    if removed > 0:
        with open(DECISIONS_LOG, "w") as f:
            for obj in kept:
                f.write(json.dumps(obj, indent=2))
                f.write("\n" + "=" * W + "\n\n")

    return removed


def _is_fresh_entry(obj: dict, exclude_types: tuple = ("EXECUTION",)) -> bool:
    """Check if a parsed log entry is a fresh decision (not an execution record).

    Recognises both formats:
      - Legacy single-trade: top-level "action" key
      - Multi-trade:         top-level "trades" key (list, possibly empty)
    """
    if obj.get("type") in exclude_types:
        return False
    # Legacy: has top-level action (BUY/SELL/HOLD)
    if obj.get("action"):
        return True
    # Multi-trade: has a trades array (including empty [] for HOLD)
    if "trades" in obj and isinstance(obj["trades"], list):
        return True
    return False


def has_fresh_decision(debug: bool = False) -> bool:
    """Check if a fresh raw (non-EXECUTION) decision exists in the log.

    A decision is 'fresh' only if there is a non-EXECUTION entry — these
    are written by Claude Code during Phase 2 reasoning.  Old EXECUTION
    entries (from the executor) do NOT count.
    """
    if not os.path.exists(DECISIONS_LOG) or os.path.getsize(DECISIONS_LOG) < 10:
        if debug:
            print(f"  [poll-debug] Log missing or <10 bytes")
        return False
    objects = _parse_log_objects(DECISIONS_LOG)
    if debug:
        print(f"  [poll-debug] Parsed {len(objects)} object(s) from kairos_decisions.log")
        for i, obj in enumerate(objects):
            obj_type = obj.get("type", "(none)")
            has_action = bool(obj.get("action"))
            has_trades = "trades" in obj
            keys = list(obj.keys())[:6]
            print(f"  [poll-debug]   [{i}] type={obj_type} action={has_action} "
                  f"trades={has_trades} keys={keys}")
    for obj in reversed(objects):
        if _is_fresh_entry(obj):
            return True
    return False


# ── Claude Code CLI invocation helpers ────────────────────────────────

def _load_claude_config() -> dict:
    """Load Claude + alerts config from kairos_config.json."""
    config_file = os.path.join(SCRIPT_DIR, "kairos_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            return {
                "model": cfg.get("claude", {}).get("model", "claude-sonnet-4-6"),
                "timeout": cfg.get("claude", {}).get("reasoning_timeout_s", 60),
                "slack_webhook_url": cfg.get("alerts", {}).get("slack_webhook_url", ""),
                "slack_channel": cfg.get("alerts", {}).get("slack_channel", "#kairos-alerts"),
            }
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "model": "claude-sonnet-4-6",
        "timeout": 60,
        "slack_webhook_url": "",
        "slack_channel": "#kairos-alerts",
    }


def _is_valid_decision(obj: object) -> bool:
    """Check if a parsed JSON object is a valid decision in any format.

    Accepts:
      - Legacy single-trade: {"action": "BUY", "ticker": "X", ...}
      - Multi-trade:         {"trades": [...], ...}  (including empty trades array)
    """
    if not isinstance(obj, dict):
        return False
    # Multi-trade format (new): has a "trades" key (list, possibly empty)
    if "trades" in obj and isinstance(obj["trades"], list):
        return True
    # Legacy single-trade format: has "action"
    if obj.get("action"):
        return True
    return False


def _normalize_decision(obj: dict) -> dict:
    """Normalize a parsed decision into a consistent format.

    - Multi-trade with trades: returned as-is (caller handles the list)
    - Multi-trade with empty trades []: converted to a HOLD decision
    - Legacy single-trade: returned as-is (has action/ticker at top level)
    """
    if "trades" in obj and isinstance(obj["trades"], list):
        if len(obj["trades"]) == 0:
            # Empty trades array → cycle-wide HOLD
            skipped = obj.get("skipped", "")
            rationale = skipped or "Claude returned empty trades array"
            obj["action"] = "HOLD"
            obj["ticker"] = "ALL"
            obj["rationale"] = rationale
            print(f"  [claude-thread] Empty trades array → HOLD (skipped: {rationale[:120]})")
    return obj


def _log_parse_failure(response: str) -> None:
    """Log the full raw response to kairos_monitor.log on parse failure."""
    monitor_log = os.path.join(SCRIPT_DIR, "kairos_monitor.log")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = {
        "type": "PARSE_FAILURE",
        "timestamp": ts,
        "raw_response_length": len(response),
        "raw_response": response[:5000],  # cap at 5KB to avoid log bloat
    }
    try:
        with open(monitor_log, "a") as f:
            f.write("\n" + json.dumps(entry, indent=2) + "\n")
        print(f"  [claude-thread] Full raw response logged to kairos_monitor.log")
    except IOError:
        pass


def _extract_json_decision(response: str) -> dict | None:
    """Extract a JSON decision block from Claude's raw response text.

    Handles:
      - Clean JSON output (both single-trade and multi-trade formats)
      - JSON inside markdown ```json ... ``` fences
      - JSON embedded in surrounding prose
      - Empty trades array ({"trades":[],...}) as a valid HOLD
      - Fallback: scan for any JSON block containing 'action' or 'trades'
    """
    # Try 1: direct parse of the entire response
    stripped = response.strip()
    try:
        obj = json.loads(stripped)
        if _is_valid_decision(obj):
            return _normalize_decision(obj)
    except json.JSONDecodeError:
        pass

    # Try 2: extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", response, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1).strip())
            if _is_valid_decision(obj):
                return _normalize_decision(obj)
        except json.JSONDecodeError:
            pass

    # Try 3: find balanced { ... } blocks containing "action" or "trades"
    depth = 0
    start = None
    for i, ch in enumerate(response):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = response[start : i + 1]
                if '"action"' in candidate or '"trades"' in candidate:
                    try:
                        obj = json.loads(candidate)
                        if _is_valid_decision(obj):
                            return _normalize_decision(obj)
                    except json.JSONDecodeError:
                        pass
                start = None

    # All attempts failed — log full response for debugging
    _log_parse_failure(response)
    return None


def _write_decision_to_log(decision: dict) -> None:
    """Append a raw decision entry to kairos_decisions.log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    decision.setdefault("timestamp", ts)
    with open(DECISIONS_LOG, "a") as f:
        f.write("\n" + json.dumps(decision, indent=2))
        f.write("\n" + "=" * W + "\n")


def _call_anthropic_api(prompt_text: str, system_instruction: str, cfg: dict) -> str | None:
    """Call the Anthropic Messages API and return the raw text response.

    Uses prompt caching: the system prompt and the static context
    (everything before the per-cycle data) are marked ephemeral so
    repeated calls within the TTL hit the cache.

    Returns the response text, or None on failure.
    """
    import anthropic
    import time

    MAX_RETRIES = 4
    RETRYABLE_ERRORS = (
        anthropic.InternalServerError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
    )
    NON_RETRYABLE_ERRORS = (
        anthropic.AuthenticationError,
        anthropic.BadRequestError,
    )

    model = cfg.get("model", "claude-sonnet-4-6")
    timeout = cfg.get("timeout", 60)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": system_instruction,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                timeout=timeout,
            )
            # Extract text from response
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            usage = response.usage
            cached = getattr(usage, "cache_read_input_tokens", 0) or 0
            print(f"  API usage: input={usage.input_tokens} "
                  f"(cached={cached}) output={usage.output_tokens}")
            return text

        except NON_RETRYABLE_ERRORS as exc:
            print(f"  ERROR: Anthropic API error: {exc}")
            return None
        except RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                backoff = [5, 15, 30][attempt - 1]
                print(f"  WARNING: Anthropic API transient error (attempt {attempt}/{MAX_RETRIES}): {exc}. Retrying in {backoff}s...")
                time.sleep(backoff)
            continue
        except anthropic.APIError as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            status_str = f" (status_code={status})" if status is not None else ""
            print(f"  ERROR: Anthropic API error{status_str}: {exc}")
            return None
        except Exception as exc:
            print(f"  ERROR: Unexpected error calling Anthropic API: {exc}")
            return None

    # All retries exhausted
    print(f"  ERROR: Anthropic API failed after {MAX_RETRIES} attempts. Last error: {last_error}")
    return None


def _ticker_is_recent_ipo(ticker: str) -> bool:
    """Thin wrapper used for log-output counting; never raises."""
    try:
        from kairos_signals_ipo import is_recent_ipo
        return is_recent_ipo(ticker)
    except Exception:
        return False


def _invoke_claude_reasoning(prompt_file: str, cfg: dict) -> dict | None:
    """Call the Anthropic API for equity reasoning.

    Reads the prompt from *prompt_file*, sends it to the Messages API,
    parses the JSON decision, writes it to kairos_decisions.log, and
    returns the decision dict (or None on failure).
    """
    try:
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
    except IOError as exc:
        print(f"  ERROR reading prompt: {exc}")
        return None

    # Prepend an IPO context block if any shortlisted ticker IPO'd in the
    # last IPO_HOLD_DAYS. The block instructs Claude to raise conviction
    # for the affected tickers (encoding the size_multiplier + boost from
    # score_ipo_momentum into a higher JSON conviction value).
    try:
        shortlist_for_ipo = get_screening_shortlist() or []
        if shortlist_for_ipo:
            from kairos_signals_ipo import format_ipo_prompt_block
            ipo_block = format_ipo_prompt_block(shortlist_for_ipo)
            if ipo_block:
                prompt_text = ipo_block + prompt_text
                print(f"  IPO context: injected for "
                      f"{sum(1 for t in shortlist_for_ipo if _ticker_is_recent_ipo(t))} "
                      f"shortlisted ticker(s)")
    except Exception as ipo_exc:
        print(f"  WARNING: IPO prompt injection failed: {ipo_exc}")

    # Prepend the event-driven seasonality block whenever a tracked
    # calendar event is in window. Block instructs Claude to raise
    # conviction on matching tickers per the per-pattern multiplier.
    try:
        from kairos_signals_events import get_active_events, format_event_prompt_block
        active_events = get_active_events()
        if active_events:
            event_block = format_event_prompt_block(active_events)
            if event_block:
                prompt_text = event_block + prompt_text
                print(f"  Event context: injected for "
                      f"{len(active_events)} active event(s) — "
                      f"{', '.join(sorted({e['ticker'] for e in active_events}))}")
    except Exception as evt_exc:
        print(f"  WARNING: Event prompt injection failed: {evt_exc}")

    system_instruction = (
        "Output ONLY the JSON object. No markdown fences, "
        "no explanation, no preamble. Raw JSON with a 'trades' array "
        "containing ALL recommended trades. Every BUY MUST include the "
        "thesis fields (predicted_direction, predicted_timeframe_days, "
        "predicted_return_pct, key_conditions, invalidation_conditions):\n"
        '{"trades":[{"action":"BUY","ticker":"...","sector":"...",'
        '"conviction":N,"rationale":"...",'
        '"predicted_direction":"UP|DOWN|NEUTRAL",'
        '"predicted_timeframe_days":N,"predicted_return_pct":F,'
        '"key_conditions":"...","invalidation_conditions":"..."},...],'
        '"tickers_evaluated":[...],"skipped":"..."}\n'
    )

    invoke_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"  API call started at: {invoke_ts}")
    print(f"  Model: {cfg.get('model', 'claude-sonnet-4-6')}  Timeout: {cfg['timeout']}s")

    raw_response = _call_anthropic_api(prompt_text, system_instruction, cfg)

    if raw_response is None:
        return None

    resp_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"  Response received at: {resp_ts}")
    print(f"  Response length: {len(raw_response)} chars")

    decision = _extract_json_decision(raw_response)
    if decision is None:
        print(f"  ERROR: Could not parse JSON from response.")
        preview = raw_response[:800].replace("\n", "\n    ")
        print(f"  Raw response preview:\n    {preview}")
        return None

    # Validate: multi-trade format has trades list, legacy has action/ticker
    trades = decision.get("trades")
    if isinstance(trades, list) and len(trades) > 0:
        for idx, t in enumerate(trades):
            if not t.get("action") or not t.get("ticker"):
                print(f"  WARNING: trade[{idx}] missing action/ticker, skipping")
    elif decision.get("action") == "HOLD":
        pass
    else:
        required = ("action", "ticker")
        missing = [f for f in required if not decision.get(f)]
        if missing:
            print(f"  ERROR: Missing required fields: {missing}")
            print(f"  Parsed object: {json.dumps(decision)[:500]}")
            return None

    _write_decision_to_log(decision)
    print(f"  Decision written to kairos_decisions.log")
    if isinstance(trades, list) and len(trades) > 0:
        print(f"  Multi-trade: {len(trades)} trade(s)")
        for t in trades:
            print(f"    {t.get('action','?')} {t.get('ticker','?')} "
                  f"conviction={t.get('conviction','?')}")
    else:
        print(f"  action={decision.get('action')} "
              f"ticker={decision.get('ticker')} quantity={decision.get('quantity', 0)}")

    return decision


def _log_reasoning_timeout(cfg: dict) -> None:
    """Log a Phase 2 timeout to kairos_monitor.log and send Slack alert."""
    from kairos_alerts import alert_reasoning_timeout
    alert_reasoning_timeout(cfg["timeout"])


def _log_phase_crash(phase: str, exc: Exception, tb: str) -> None:
    """Log a phase crash with full traceback to kairos_monitor.log."""
    monitor_log = os.path.join(SCRIPT_DIR, "kairos_monitor.log")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {
        "timestamp": ts,
        "event": "PHASE_CRASH",
        "phase": phase,
        "error": str(exc),
        "traceback": tb[-2000:],  # cap to avoid log bloat
    }
    try:
        with open(monitor_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except IOError:
        pass


# ── Phase 0: Ollama Orchestration ────────────────────────────────────

def run_orchestrate(verbose: bool = True) -> dict:
    """Phase 0 — Ollama handles routine pre-flight checks.

    Returns:
        dict with keys:
          - stale_sources: list of source names that need refreshing
          - ollama_available: bool
          - latency_ms: Ollama response time
    """
    phase_banner("0", "ORCHESTRATE — Ollama Local Agent Pre-flight")

    from kairos_router import KairosRouter, TaskType

    router = KairosRouter()
    result = {"stale_sources": [], "ollama_available": False, "latency_ms": 0}

    if not router.ollama_available:
        print("  WARNING: Ollama not available — skipping Phase 0.")
        print("  (Start Ollama with: ollama serve)")
        return result

    result["ollama_available"] = True
    print("  Ollama: available (llama3.2)")

    # Check which data sources are stale using snapshot file mtime as proxy
    snapshot_path = os.path.join(SCRIPT_DIR, "kairos_snapshot.txt")
    source_timestamps: dict[str, str | None] = {}

    if os.path.exists(snapshot_path):
        mtime = os.path.getmtime(snapshot_path)
        ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        for src in ("finnhub", "fred", "kalshi", "congress", "coingecko"):
            source_timestamps[src] = ts
    else:
        for src in ("finnhub", "fred", "kalshi", "congress", "coingecko"):
            source_timestamps[src] = None

    t0 = time.time()
    stale_result = router.route(
        TaskType.STALE_SOURCES,
        source_timestamps=source_timestamps,
        max_age_hours=4.0,
    )
    elapsed_ms = round((time.time() - t0) * 1000)
    result["latency_ms"] = elapsed_ms

    if stale_result.success:
        stale = stale_result.result or []
        result["stale_sources"] = stale
        if stale:
            print(f"  Stale sources detected: {stale}")
            print("  → These will be refreshed in Phase 1.")
        else:
            print("  All sources fresh (< 4h old). Snapshot still valid.")
        print(f"  Ollama response: {elapsed_ms}ms")
    else:
        print(f"  WARNING: Stale source check failed: {stale_result.error}")

    if verbose:
        router.print_stats()

    return result


# ── Phase 0R: Macro Regime Detection ─────────────────────────────────

# Module-level regime result — set by run_regime(), read by downstream phases
_current_regime: dict | None = None


def get_current_regime() -> dict | None:
    """Return the current regime result, or None if not yet detected."""
    return _current_regime


def run_regime() -> dict:
    """Phase 0R — Detect macro regime and set guardrails.

    Must run before the screener so guardrails can constrain downstream
    phases (position sizing, sector blocks, Mode C eligibility).
    """
    global _current_regime
    phase_banner("0R", "MACRO REGIME DETECTION — VIX / SPY / FRED / Kalshi")

    from kairos_regime import detect_regime
    result = detect_regime(verbose=True)
    _current_regime = result
    return result


# ── Phase 0.1: Stop-Loss Monitor ────────────────────────────────────

def run_stoploss(ib=None) -> dict:
    """Phase 0.1 — Check all open positions against regime-adjusted stop-loss.

    Must run after regime detection (Phase 0R) so thresholds are current.
    ib: optional shared IBKR connection to pass through to the stoploss module.
    """
    phase_banner("0.1", "STOP-LOSS MONITOR — Regime-Adjusted Drawdown Check")

    regime = _current_regime["regime"] if _current_regime else None
    from kairos_stoploss import run_stoploss as do_stoploss
    return do_stoploss(regime=regime, ib=ib)


# ── Phase 0.5: Tier 1 Screening ────────────────────────────────────

SCREEN_RESULT_FILE = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")


def run_screen(dry_run: bool = False, max_tier2: int = 15) -> dict:
    """Phase 0.5 — Ollama screens the broad universe, returns shortlist.

    Returns the screening result dict (see kairos_screener.run_screen).
    """
    phase_banner("0.5", "TIER 1 SCREEN — Ollama Broad Universe Scan")

    from kairos_screener import run_screen as do_screen
    import json as _json
    try:
        with open(os.path.join(SCRIPT_DIR, "kairos_config.json")) as _f:
            _cfg = _json.load(_f)
        _skip_t0 = _cfg.get("tier0_filter", {}).get("skip_tier0", False)
    except Exception:
        _skip_t0 = False
    result = do_screen(dry_run=dry_run, max_tier2=max_tier2, skip_tier0=_skip_t0)
    # Explicitly unload screen model before Qwen3 loads for Tier 2
    try:
        from kairos_ollama import unload_model, get_screen_model
        unload_model(get_screen_model())
    except Exception:
        pass

    # Persist shortlist so downstream phases can pick it up
    with open(SCREEN_RESULT_FILE, "w") as f:
        json.dump({
            "timestamp": result["timestamp"],
            "shortlist": result["shortlist"],
            "universe_size": result["universe_size"],
            "tier2_count": result["tier2_count"],
            "hot": result["hot"],
            "hot_reversion": result.get("hot_reversion", []),
            "hot_earnings": result.get("hot_earnings", []),
            "hot_rsi": result.get("hot_rsi", []),
            "hot_insider": result.get("hot_insider", []),
            "hot_congress": result.get("hot_congress", []),
            "signal_tags": result.get("signal_tags", {}),
            "source_tiers": result.get("source_tiers", {}),
            "warm": result["warm"],
            "cold_count": result["cold_count"],
            "elapsed_sec": result["elapsed_sec"],
        }, f, indent=2)

    print(f"\n  Shortlist saved → {SCREEN_RESULT_FILE}")
    print(f"  {result['tier2_count']}/{result['universe_size']} tickers → Tier 2 (Claude)")
    return result


def get_screening_shortlist() -> list[str] | None:
    """Load the most recent screening shortlist, if available."""
    if not os.path.exists(SCREEN_RESULT_FILE):
        return None
    try:
        with open(SCREEN_RESULT_FILE, "r") as f:
            data = json.load(f)
        return data.get("shortlist", [])
    except (json.JSONDecodeError, KeyError):
        return None


# ── Phase 0.7: ML Pattern Recognition ────────────────────────────────

ML_RESULT_FILE = os.path.join(SCRIPT_DIR, "kairos_ml_result.json")


def run_ml_phase(candidates: list[str] | None = None) -> list[dict]:
    """Phase 0.7 — ML pattern recognition scoring.
    
    Takes the Tier 1 screening shortlist and scores each candidate using
    the trained ML model. Results are saved and passed to downstream phases.
    
    Args:
        candidates: Optional list of ticker symbols. If None, loads from
                    kairos_screen_result.json.
    
    Returns:
        List of candidate dicts with ml_confidence, ml_signal, and 
        ml_trained_on added. Returns empty list if no candidates.
    """
    phase_banner("0.7", "ML PATTERN RECOGNITION — Track 1 Scoring")
    
    from kairos_ml import run_ml_phase as do_ml_phase
    
    # Load candidates if not provided
    if candidates is None:
        shortlist = get_screening_shortlist()
        if not shortlist:
            print("  No shortlist available — skipping ML phase")
            return []
        candidates = shortlist
    
    # Convert to candidate dicts (ticker only for now)
    candidate_dicts = [{"ticker": t} for t in candidates]
    
    # Score with ML
    scored = do_ml_phase(candidate_dicts)
    
    # Save results
    ml_result = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "candidates": scored,
        "model_info": {
            "trained_on": scored[0]["ml_trained_on"] if scored else 0
        }
    }
    with open(ML_RESULT_FILE, "w") as f:
        json.dump(ml_result, f, indent=2)
    
    print(f"  ML scoring complete → {ML_RESULT_FILE}")
    print(f"  Scored {len(scored)} candidates")
    for c in scored:
        print(f"    {c['ticker']}: {c['ml_signal']} (confidence: {c['ml_confidence']:.3f})")
    
    return scored


def load_ml_results() -> list[dict]:
    """Load ML scoring results from file."""
    if not os.path.exists(ML_RESULT_FILE):
        return []
    try:
        with open(ML_RESULT_FILE, "r") as f:
            data = json.load(f)
        return data.get("candidates", [])
    except (json.JSONDecodeError, IOError):
        return []


def get_ml_scored_shortlist() -> list[str]:
    """Get the ML-scored shortlist as a simple ticker list."""
    scored = load_ml_results()
    return [c["ticker"] for c in scored]


# ── Phase 1: Gather ─────────────────────────────────────────────────

def run_gather(use_ollama: bool = True, shortlist: list[str] | None = None):
    phase_banner(1, "GATHER — Snapshot + Portfolio + Tax + History")

    from kairos_reason import (
        gather_market_data, gather_portfolio,
        gather_tax_context, gather_tax_efficiency, gather_ledger, write_prompt,
    )

    market_data = gather_market_data()
    portfolio = gather_portfolio(shortlist)
    tax_ctx = gather_tax_context()
    tax_eff = gather_tax_efficiency(portfolio, shortlist=shortlist)
    ledger_text = gather_ledger()

    if use_ollama:
        _ollama_postprocess_snapshot(market_data, portfolio)

    # If a Tier 1 shortlist was provided, include it in the prompt
    if shortlist:
        print(f"\n  Tier 1 shortlist ({len(shortlist)} tickers): {', '.join(shortlist)}")

    # ── Options activity detection (IBKR, shortlist only) ─────────────
    if shortlist:
        _run_options_detection(shortlist)

    # ── HOT-CATALYST long-options detection (event calendar + crush scan) ──
    _run_catalyst_detection(shortlist)

    # Pass regime context into the prompt if available
    regime = get_current_regime()
    regime_section = regime["prompt_section"] if regime else None

    write_prompt(market_data, portfolio, tax_ctx, ledger_text,
                 shortlist=shortlist, tax_efficiency_text=tax_eff,
                 regime_section=regime_section)


def _run_options_detection(shortlist: list[str]) -> None:
    """Run HOT-OPTIONS detection on shortlisted tickers via IBKR.

    Results are saved to kairos_options_activity.json and merged into
    kairos_signal_summary.json for prompt assembly.
    """
    print(f"\n  ┌─ Options Activity Detection ({len(shortlist)} tickers) ────────┐")
    try:
        from kairos_signals_options import detect_options_activity, save_options_activity
    except ImportError as exc:
        print(f"  │  SKIP: {exc}")
        print(f"  └──────────────────────────────────────────────────────┘")
        return

    t0 = time.time()
    results = detect_options_activity(shortlist)
    elapsed = round(time.time() - t0, 1)

    if results:
        print(f"  │  HOT-OPTIONS hits: {len(results)} ({elapsed}s)")
        for sym, data in results.items():
            triggers = ", ".join(data["triggered_by"])
            print(f"  │    {sym}: vol/OI={data['vol_oi_ratio']:.1f}x  "
                  f"IV={data['iv_rank']:.0%}  C/P={data['call_put_ratio']:.1f}x  "
                  f"[{triggers}]")
        save_options_activity(results)

        # Merge HOT-OPTIONS into kairos_signal_summary.json
        summary_file = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                for ticker in results:
                    tags = summary.setdefault("signal_tags", {}).setdefault(ticker, [])
                    if "HOT-OPTIONS" not in tags:
                        tags.append("HOT-OPTIONS")
                summary["options_hits"] = results
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
            except (json.JSONDecodeError, IOError):
                pass
    else:
        print(f"  │  No unusual options activity detected ({elapsed}s)")

    # ── Second pass: yfinance sweep of EXECUTIVE_WATCHLIST tickers ───────
    # Catches unusual flow on closely-watched names that didn't make the
    # Tier-1 shortlist. Tagged HOT-OPTIONS-EXEC so downstream prompt
    # assembly can distinguish them from IBKR-quality main hits.
    try:
        from kairos_signals_options import (
            EXECUTIVE_WATCHLIST,
            _detect_options_with_yfinance,
            _load_options_config,
        )
    except ImportError as exc:
        print(f"  │  Watchlist sweep skipped: {exc}")
        print(f"  └──────────────────────────────────────────────────────┘")
        return

    shortlist_set = {t.upper() for t in shortlist}
    extra_tickers = [t for t in EXECUTIVE_WATCHLIST if t.upper() not in shortlist_set]
    exec_results: dict[str, dict] = {}

    if extra_tickers:
        print(f"  │  Watchlist sweep: {len(extra_tickers)} non-shortlist tickers via yfinance")
        cfg = _load_options_config()
        t1 = time.time()
        for sym in extra_tickers:
            yf_data = _detect_options_with_yfinance(sym, cfg)
            if yf_data:
                exec_results[sym] = yf_data
        sweep_elapsed = round(time.time() - t1, 1)

        if exec_results:
            print(f"  │  HOT-OPTIONS-EXEC hits: {len(exec_results)} ({sweep_elapsed}s)")
            for sym, data in exec_results.items():
                triggers = ", ".join(data.get("triggered_by", []))
                print(f"  │    {sym}: vol/OI={data.get('vol_oi_ratio', 0):.1f}x  "
                      f"C/P={data.get('call_put_ratio', 0):.1f}x  [{triggers}]")

            summary_file = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                    for ticker in exec_results:
                        tags = summary.setdefault("signal_tags", {}).setdefault(ticker, [])
                        if "HOT-OPTIONS-EXEC" not in tags:
                            tags.append("HOT-OPTIONS-EXEC")
                    merged = dict(summary.get("options_hits") or {})
                    merged.update(exec_results)
                    summary["options_hits"] = merged
                    with open(summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                except (json.JSONDecodeError, IOError):
                    pass
        else:
            print(f"  │  Watchlist sweep: no hits ({sweep_elapsed}s)")

    print(f"  └──────────────────────────────────────────────────────┘")


def _run_catalyst_detection(shortlist: list[str] | None) -> None:
    """Detect HOT-CATALYST long-option setups (event calendar + crush + flow).

    Self-contained, algorithmic detection — the options engine self-selects
    contracts at execute time. Here we only SURFACE the signals: persist them
    to kairos_catalyst_signals.json and register a HOT-CATALYST tag into
    kairos_signal_summary.json so the equity reasoning prompt is aware of them
    (awareness / dedup, exactly like HOT-OPTIONS). No contracts are chosen here.
    """
    print(f"\n  ┌─ HOT-CATALYST Detection ──────────────────────────────┐")
    try:
        from kairos_catalyst_signals import (
            detect_catalyst_signals, load_hot_catalyst_config,
        )
    except ImportError as exc:
        print(f"  │  SKIP: {exc}")
        print(f"  └──────────────────────────────────────────────────────┘")
        return

    t0 = time.time()
    try:
        cfg = load_hot_catalyst_config()
        signals = detect_catalyst_signals(tickers=shortlist, config=cfg)
    except Exception as exc:
        print(f"  │  Detection failed: {exc}")
        print(f"  └──────────────────────────────────────────────────────┘")
        return
    elapsed = round(time.time() - t0, 1)

    if not signals:
        print(f"  │  No catalyst setups detected ({elapsed}s)")
        print(f"  └──────────────────────────────────────────────────────┘")
        return

    by_ticker: dict[str, list] = {}
    for s in signals:
        by_ticker.setdefault(s["ticker"], []).append(s)

    print(f"  │  HOT-CATALYST setups: {len(signals)} ({elapsed}s)")
    for s in signals:
        rt = "calls" if s["right"] == "C" else "puts"
        ivr = s.get("iv_rank")
        ivr_str = f"{ivr:.0%}" if isinstance(ivr, (int, float)) else "n/a"
        print(f"  │    {s['ticker']}: {s['setup']} ({s['direction']}→{rt})  "
              f"IVrank={ivr_str}")

    catalyst_file = os.path.join(SCRIPT_DIR, "kairos_catalyst_signals.json")
    try:
        with open(catalyst_file, "w") as f:
            json.dump({"signals": signals, "by_ticker": by_ticker}, f,
                      indent=2, default=str)
    except IOError:
        pass

    # Register HOT-CATALYST tag into the equity prompt summary (awareness/dedup).
    summary_file = os.path.join(SCRIPT_DIR, "kairos_signal_summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            for ticker in by_ticker:
                tags = summary.setdefault("signal_tags", {}).setdefault(ticker, [])
                if "HOT-CATALYST" not in tags:
                    tags.append("HOT-CATALYST")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except (json.JSONDecodeError, IOError):
            pass

    print(f"  └──────────────────────────────────────────────────────┘")


def _ollama_postprocess_snapshot(market_data: str, portfolio: dict):
    """Ollama post-processes the raw snapshot: filter and summarize news."""
    from kairos_router import KairosRouter, TaskType

    router = KairosRouter()
    if not router.ollama_available:
        return

    print()
    print("  ┌─ Ollama Post-processing ──────────────────────────────┐")

    tickers = [p["symbol"] for p in portfolio.get("positions", [])]
    if not tickers:
        tickers = ["AAPL"]

    headlines = _extract_headlines(market_data)

    if headlines:
        print(f"  │  Screening {len(headlines)} headlines for {tickers}...")
        relevant_headlines = []
        filtered_count = 0
        for h in headlines:
            r = router.route(TaskType.NEWS_RELEVANCE, headline=h, tickers=tickers)
            if r.result is True or not r.success:
                relevant_headlines.append(h)
            else:
                filtered_count += 1

        print(f"  │  Filtered: {filtered_count} irrelevant, {len(relevant_headlines)} relevant")

        if relevant_headlines:
            print(f"  │  Summarizing {len(relevant_headlines)} headlines...")
            summary_result = router.route(
                TaskType.NEWS_SUMMARY,
                headlines=relevant_headlines,
                ticker=tickers[0],
            )
            if summary_result.success and summary_result.result:
                summary_path = os.path.join(SCRIPT_DIR, "kairos_news_summary.txt")
                with open(summary_path, "w") as f:
                    f.write(f"Ollama News Summary — {datetime.now(timezone.utc).isoformat()}\n")
                    f.write(f"Relevant headlines: {len(relevant_headlines)}/{len(headlines)}\n\n")
                    f.write(summary_result.result)
                print("  │  Summary saved to kairos_news_summary.txt")

    router.print_stats()
    print("  └──────────────────────────────────────────────────────┘")


def _extract_headlines(market_data: str) -> list[str]:
    """Best-effort extraction of news headlines from the snapshot text."""
    headlines = []
    in_news_section = False

    for line in market_data.splitlines():
        stripped = line.strip()

        if "finnhub" in stripped.lower() or "news" in stripped.lower():
            in_news_section = True
            continue

        if in_news_section and stripped.startswith("━") and len(stripped) > 10:
            in_news_section = False
            continue

        if in_news_section and stripped and len(stripped) > 20:
            if not stripped[0].isdigit() and "http" not in stripped:
                headlines.append(stripped.lstrip("•-→ "))

    return headlines[:20]


# ── Phase 1C: Crypto Gather ──────────────────────────────────────────

def _fetch_coingecko_structured() -> list[dict]:
    """Pull crypto market data from CoinGecko for configured assets.

    Loads asset list from kairos_crypto_config.json (defaults to BTC/ETH).

    Returns list of:
        {"id": str, "symbol": str, "price": float,
         "change_24h": float, "volume": float, "market_cap": float}
    Returns empty list on any network failure (caller handles gracefully).
    """
    import requests

    # Load asset list from config
    crypto_config_file = os.path.join(SCRIPT_DIR, "kairos_crypto_config.json")
    coin_ids = "bitcoin,ethereum"
    if os.path.exists(crypto_config_file):
        try:
            with open(crypto_config_file) as f:
                cc = json.load(f)
            assets = cc.get("assets", ["bitcoin", "ethereum"])
            coin_ids = ",".join(assets)
        except (json.JSONDecodeError, IOError):
            pass

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": coin_ids,
                "order": "market_cap_desc",
            },
            timeout=15,
        )
        resp.raise_for_status()
        coins = resp.json()
    except Exception as exc:
        print(f"  WARNING: CoinGecko fetch failed: {exc}")
        return []

    result = []
    for c in coins:
        result.append({
            "id": c.get("id", ""),
            "symbol": c.get("symbol", "").upper(),
            "price": float(c.get("current_price") or 0),
            "change_24h": float(c.get("price_change_percentage_24h") or 0),
            "volume": float(c.get("total_volume") or 0),
            "market_cap": float(c.get("market_cap") or 0),
        })
    return result


def run_gather_crypto(mode: str = "crypto", use_ollama: bool = True) -> dict:
    """Phase 1C — Gather crypto data, run risk guards, route signals.

    Args:
        mode:       "crypto" or "both"
        use_ollama: Whether to attempt Ollama signal classification

    Returns a status dict:
        {
          "halted": bool,          # True if circuit breaker / daily loss triggered
          "halt_reason": str,      # Human-readable halt cause (or "")
          "coins": list[dict],     # CoinGecko data for each asset
          "signals": dict,         # {symbol: "alert"|"normal"|"unknown"}
          "guards": dict,          # Per-symbol guard results
          "prompt_written": bool,  # Whether kairos_crypto_prompt.txt was written
        }
    """
    phase_banner("1C", "GATHER CRYPTO — CoinGecko + Risk Guards + Ollama Routing")

    from kairos_crypto_risk import CryptoRiskGuard, get_config_summary
    from kairos_router import KairosRouter, TaskType

    guard = CryptoRiskGuard()
    router = KairosRouter() if use_ollama else None

    print(get_config_summary())

    # ── Step 1: Fetch market data ─────────────────────────────────────
    print("\n  Fetching CoinGecko market data...")
    coins = _fetch_coingecko_structured()

    if not coins:
        print("  ERROR: Could not fetch crypto data. Skipping crypto phase.")
        return {"halted": False, "halt_reason": "data_fetch_failed",
                "coins": [], "signals": {}, "guards": {}, "prompt_written": False}

    for c in coins:
        direction = "▲" if c["change_24h"] >= 0 else "▼"
        print(f"  {c['symbol']:>5}  ${c['price']:>12,.2f}  {direction} {abs(c['change_24h']):.2f}%  "
              f"Vol: ${c['volume']/1e9:.2f}B")

    # ── Step 2: Update price history (feeds circuit breaker) ──────────
    for c in coins:
        guard.update_price_history(c["symbol"], c["price"])

    # ── Step 3: Run circuit breaker — flag HOT-REVERSION instead of halting
    print("\n  Running circuit breaker checks...")
    halted = False
    halt_reason = ""
    guard_results: dict[str, dict] = {}
    reversion_coins: list[str] = []

    for c in coins:
        cb = guard.check_circuit_breaker(c["symbol"], c["price"])
        guard_results[c["symbol"]] = {"circuit_breaker": cb}
        if not cb.allowed:
            # Instead of halting: flag as mean reversion candidate for Claude
            move_pct = cb.details.get("move_pct", 0)
            direction = "down" if c["change_24h"] < 0 else "up"
            if direction == "down":
                reversion_coins.append(c["symbol"])
                print(f"  [HOT-REVERSION] {c['symbol']} breaker triggered "
                      f"({move_pct:.1f}% {direction}) → flagging for Claude review")
            else:
                print(f"  [✓ PASS] {c['symbol']} breaker triggered "
                      f"({move_pct:.1f}% {direction}) → large up-move, not halting")
        else:
            print(f"  [✓ PASS] {c['symbol']} circuit breaker: {cb.reason}")

    # ── Step 4: Daily loss guard ────────────────────────────────────────
    crypto_portfolio_value = sum(c["price"] * 0 for c in coins)  # 0 held for now
    dl = guard.check_daily_loss(crypto_portfolio_value)
    print(f"\n  Daily loss guard: {dl.reason}")
    if not dl.allowed:
        halted = True
        halt_reason = f"Daily loss limit: {dl.reason}"

    if halted:
        print(f"\n  ⚠  CRYPTO TRADING HALTED — {halt_reason}")
        _write_crypto_halt_log(halt_reason)
        return {
            "halted": True,
            "halt_reason": halt_reason,
            "coins": coins,
            "signals": {},
            "guards": guard_results,
            "prompt_written": False,
        }

    # Pass reversion flags to signal classification
    if reversion_coins:
        print(f"\n  Mean reversion candidates: {', '.join(reversion_coins)}")
        print("  These will be included in the crypto prompt for Claude to evaluate.")

    # ── Step 5: Ollama signal classification ──────────────────────────
    signals: dict[str, str] = {}

    if use_ollama and router and router.ollama_available:
        print("\n  ┌─ Ollama Crypto Signal Classification ────────────────┐")
        for c in coins:
            # Override with HOT-REVERSION if circuit breaker flagged it
            if c["symbol"] in reversion_coins:
                signals[c["symbol"]] = "reversion"
                print(f"  │  ↩ {c['symbol']}: HOT-REVERSION ({c['change_24h']:+.2f}%)")
                continue
            r = router.route(
                TaskType.CRYPTO_SIGNAL_CLASSIFY,
                symbol=c["symbol"],
                change_pct=c["change_24h"],
                volume_change_pct=0.0,
            )
            sig = r.result if r.success else "unknown"
            signals[c["symbol"]] = sig
            icon = "⚡" if sig == "alert" else ("✓" if sig == "normal" else "?")
            print(f"  │  {icon} {c['symbol']}: {sig} ({c['change_24h']:+.2f}%)")
        print("  └──────────────────────────────────────────────────────┘")
    else:
        for c in coins:
            if c["symbol"] in reversion_coins:
                signals[c["symbol"]] = "reversion"
            elif abs(c["change_24h"]) >= 3.0:
                signals[c["symbol"]] = "alert"
            else:
                signals[c["symbol"]] = "normal"
        if not (use_ollama and router and router.ollama_available):
            print("  (Ollama unavailable — using threshold-based signal classification)")

    # ── Step 6: Run crypto signal detectors (HOT-RSI + HOT-REVERSION) ──
    asset_symbols = [c["symbol"] for c in coins]
    crypto_signal_tags: dict[str, list[str]] = {}
    try:
        from kairos_crypto_signals import run_crypto_signals
        crypto_signal_tags = run_crypto_signals(asset_symbols, coins)
    except Exception as exc:
        print(f"\n  WARNING: Crypto signal detection failed: {exc}")

    # Merge reversion coins into signal tags
    for sym in reversion_coins:
        tags = crypto_signal_tags.setdefault(sym, [])
        if "HOT-REVERSION" not in tags:
            tags.append("HOT-REVERSION")

    # ── Step 7: Always write crypto prompt (Claude decides HOLD if quiet) ─
    regime = get_current_regime()
    regime_section = regime["prompt_section"] if regime else None
    _write_crypto_prompt(coins, signals, guard_results,
                         signal_tags=crypto_signal_tags,
                         regime_section=regime_section)

    return {
        "halted": False,
        "halt_reason": "",
        "coins": coins,
        "signals": signals,
        "guards": guard_results,
        "prompt_written": True,
        "signal_tags": crypto_signal_tags,
    }


def _write_crypto_halt_log(reason: str) -> None:
    """Log a crypto trading halt event to kairos_crypto_decisions.log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = {
        "type": "CRYPTO_HALT",
        "timestamp": ts,
        "reason": reason,
    }
    with open(CRYPTO_DECISIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  Halt event logged to {CRYPTO_DECISIONS_LOG}")


def _write_crypto_prompt(
    coins: list[dict],
    signals: dict,
    guard_results: dict,
    signal_tags: dict[str, list[str]] | None = None,
    regime_section: str | None = None,
) -> None:
    """Write kairos_crypto_prompt.txt for Claude Code to reason about crypto."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    signal_tags = signal_tags or {}

    coin_lines = []
    for c in coins:
        direction = "▲" if c["change_24h"] >= 0 else "▼"
        sig = signals.get(c["symbol"], "unknown")
        coin_lines.append(
            f"  {c['symbol']:<5}  Price: ${c['price']:>12,.2f}  "
            f"24h: {direction}{abs(c['change_24h']):.2f}%  "
            f"Vol: ${c['volume']/1e9:.2f}B  Signal: {sig.upper()}"
        )

    guard_lines = []
    reversion_lines = []
    for symbol, guards in guard_results.items():
        cb = guards.get("circuit_breaker")
        if cb:
            sig = signals.get(symbol, "")
            if sig == "reversion":
                guard_lines.append(f"  {symbol} circuit_breaker: [HOT-REVERSION] {cb.reason}")
                reversion_lines.append(
                    f"  {symbol}: Sharp drop detected. Evaluate whether this is "
                    f"fundamental (bad news = AVOID) or sentiment-driven "
                    f"(overreaction = potential BUY opportunity)."
                )
            else:
                status = "PASS" if cb.allowed else "WARN"
                guard_lines.append(f"  {symbol} circuit_breaker: [{status}] {cb.reason}")

    reversion_section = ""
    if reversion_lines:
        reversion_section = f"""
{'=' * W}
SECTION 3: MEAN REVERSION CANDIDATES
{'=' * W}

The following assets triggered the circuit breaker with a sharp DROP.
Instead of halting, Kairos is flagging them for your review.

{chr(10).join(reversion_lines)}

For each HOT-REVERSION asset:
  - Check recent news for fundamental cause (hack, regulation, protocol failure)
  - If fundamental: AVOID — the drop is justified
  - If sentiment-driven (market panic, liquidation cascade): consider BUY
  - Use smaller position sizes for reversion trades (higher risk)
"""

    regime_block = ""
    if regime_section:
        regime_block = regime_section + "\n"

    prompt = f"""KAIROS CRYPTO REASONING PROMPT — {ts}
{'=' * W}

{regime_block}You are Kairos, an AI trading reasoning engine. You are now analyzing crypto
markets (BTC and ETH) to decide whether to BUY, SELL, or HOLD each asset.

This prompt is generated during the crypto trading window (8:00am–10:00pm ET).
Risk guards have run. Assets flagged HOT-REVERSION had sharp drops and need
your evaluation — determine if the drop is fundamental or an overreaction.

{'=' * W}
SECTION 1: CRYPTO MARKET DATA
{'=' * W}

{chr(10).join(coin_lines)}

{'=' * W}
SECTION 2: RISK GUARD STATUS
{'=' * W}

Guard status:
  - Trade size limit:    ${500:,} max per trade (configurable)
  - Position size limit: 5% of portfolio max per asset (configurable)
  - Circuit breaker:     Assets with >8% moves in 4h flagged as HOT-REVERSION
  - Daily loss limit:    Crypto portfolio not down >2% today

{chr(10).join(guard_lines) if guard_lines else "  (No specific guard details available)"}
{reversion_section}
{'=' * W}
REASONING FRAMEWORK
{'=' * W}

Analyze each factor:

1. PRICE ACTION — What does the 24h price move signal? Is the move a buying
   opportunity, a warning sign, or noise? Compare BTC and ETH moves.

2. VOLUME CONTEXT — High volume confirms price moves; low volume makes them
   suspect. Consider whether the volume supports the price action.

3. MACRO / RISK SENTIMENT — How does crypto price action align with broader
   macro conditions (Fed policy, equities direction, risk-on/off)?

4. CORRELATION — BTC and ETH often move together. If they diverge, that's
   notable. Which is leading?

5. POSITION SIZING — All trades must stay within risk guard limits:
   - Max ${500:,} per trade
   - No single asset > 5% of total portfolio

6. MEAN REVERSION — If Section 3 is present, evaluate each HOT-REVERSION
   asset. Is the drop fundamental (news-driven) or a sentiment overreaction?
   Use smaller position sizes for reversion trades.

{_crypto_signal_section(signal_tags, coins)}
{'=' * W}
REQUIRED OUTPUT
{'=' * W}

Produce a SINGLE JSON decision for the BEST opportunity across all assets.
Pick the asset with the strongest signal confluence. If no trade is warranted,
output HOLD.

{{
  "action": "BUY" | "SELL" | "HOLD",
  "asset": "<best asset symbol>",
  "trade_usd": <dollars, 0 for HOLD, max ${500:,}>,
  "rationale": "<2-3 sentence summary>",
  "assets_evaluated": {json.dumps([c["symbol"] for c in coins])},
  "runner_up": "<#2 asset: one sentence>"
}}

Do not exceed the risk guard limits.
"""

    with open(CRYPTO_PROMPT_FILE, "w") as f:
        f.write(prompt)

    print(f"\n  Crypto prompt written to {CRYPTO_PROMPT_FILE}")


def _crypto_signal_section(signal_tags: dict, coins: list[dict]) -> str:
    """Build signal detail + confluence hint for the crypto prompt."""
    lines = [
        f"{'=' * W}",
        "SIGNAL DETAIL + CONFLUENCE SCORING",
        f"{'=' * W}",
        "",
    ]

    if signal_tags:
        for sym, tags in sorted(signal_tags.items()):
            lines.append(f"  {sym}: [{', '.join(tags)}]")
        lines.append("")
    else:
        lines.append("  No signals fired this cycle.")
        lines.append("")

    # Load detailed signal data
    try:
        from kairos_crypto_signals import load_crypto_signal_summary
        summary = load_crypto_signal_summary()
        rsi_hits = summary.get("rsi_hits", {})
        reversion_hits = summary.get("reversion_hits", {})
        momentum_hits = summary.get("momentum_hits", {})
        macro = summary.get("macro", {})
        baseline = summary.get("baseline", {})

        # Per-asset detail
        all_syms = set(signal_tags.keys())
        for sym in sorted(all_syms):
            if sym in rsi_hits:
                lines.append(f"    {sym} RSI: {rsi_hits[sym]:.1f} (oversold < 30)")
            if sym in reversion_hits:
                info = reversion_hits[sym]
                lines.append(f"    {sym} REVERSION: {info.get('change_24h', 0):+.1f}% 24h drop")
            if sym in momentum_hits:
                info = momentum_hits[sym]
                lines.append(f"    {sym} MOMENTUM: +{info.get('gain_4h', 0):.1f}% in 4h "
                             f"(${info.get('price_4h_ago', 0):,.0f} → ${info.get('price_now', 0):,.0f})")

        # Market-wide signals
        if macro.get("fired"):
            lines.append(f"    MACRO: {macro.get('details', '?')}")
        if baseline.get("fired"):
            lines.append(f"    BASELINE: BTC ${baseline.get('btc_price', 0):,.0f} > "
                         f"7d SMA ${baseline.get('sma_7d', 0):,.0f} "
                         f"(+{baseline.get('pct_above', 0):.1f}%)")
    except Exception:
        pass

    lines.extend([
        "",
        "Signal legend:",
        "  HOT-RSI              = 14-period RSI < 30 (oversold, mean reversion)",
        "  HOT-REVERSION        = 24h drop >3% (evaluate fundamental vs sentiment)",
        "  HOT-CRYPTO-MOMENTUM  = 4h gain >3% + market cap growing (trend follow)",
        "  HOT-CRYPTO-MACRO     = fear/greed >50 (risk-on macro sentiment)",
        "  HOT-CRYPTO-BASELINE  = BTC above 7-day SMA (weak trend confirmation)",
        "",
        "CONFLUENCE SCORING (position sizing is automatic):",
        "  Point weights: MOMENTUM=2, RSI=1, REVERSION=1, MACRO=1, BASELINE=1",
        "    Score 1   → 0.5% NLV  (~$5K at $1M)",
        "    Score 2   → 0.75% NLV (~$7.5K)",
        "    Score 3-4 → 1.0% NLV  (~$10K)",
        "    Score 5+  → 1.5% NLV  (~$15K)",
        "  Multiple signals compound conviction. Focus on WHICH assets to",
        "  trade and WHY — sizing is enforced downstream.",
        "",
    ])

    return "\n".join(lines)


# ── Phase 2C: Crypto Reasoning ────────────────────────────────────────

def _clear_stale_crypto_decisions() -> int:
    """Remove non-EXECUTION entries from the crypto decision log."""
    if not os.path.exists(CRYPTO_DECISIONS_LOG):
        return 0
    objects = _parse_log_objects(CRYPTO_DECISIONS_LOG)
    kept = [o for o in objects if o.get("type") in ("EXECUTION", "CRYPTO_HALT")]
    removed = len(objects) - len(kept)
    if removed > 0:
        with open(CRYPTO_DECISIONS_LOG, "w") as f:
            for obj in kept:
                f.write(json.dumps(obj, indent=2))
                f.write("\n" + "=" * W + "\n\n")
    return removed


def _has_fresh_crypto_decision() -> bool:
    """Check if a fresh raw decision exists in the crypto log."""
    if not os.path.exists(CRYPTO_DECISIONS_LOG) or os.path.getsize(CRYPTO_DECISIONS_LOG) < 10:
        return False
    objects = _parse_log_objects(CRYPTO_DECISIONS_LOG)
    for obj in reversed(objects):
        if _is_fresh_entry(obj, exclude_types=("EXECUTION", "CRYPTO_HALT")):
            return True
    return False


def run_reason_crypto():
    """Phase 2C — Call Anthropic API for crypto reasoning."""
    phase_banner("2C", "REASON CRYPTO — Anthropic API Reasoning Engine")

    if not os.path.exists(CRYPTO_PROMPT_FILE):
        print("  No crypto prompt found. Skipping crypto reasoning.")
        return

    cfg = _load_claude_config()

    prompt_size = os.path.getsize(CRYPTO_PROMPT_FILE)
    print(f"  Crypto prompt size: {prompt_size:,} bytes")

    removed = _clear_stale_crypto_decisions()
    if removed:
        print(f"  Cleared {removed} stale crypto decision(s)")

    # Direct API call
    decision = _invoke_claude_crypto_reasoning(CRYPTO_PROMPT_FILE, cfg)

    if decision is not None:
        resp_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n  ✓ Crypto decision received at: {resp_ts}")
        print(f"  Parsed: {json.dumps(decision, indent=2)}")
    else:
        print(f"\n  FAILED: No crypto decision from API")
        from kairos_alerts import alert_reasoning_timeout
        alert_reasoning_timeout(cfg["timeout"])
        # Don't sys.exit — let equity pipeline continue if running in "both" mode


def _invoke_claude_crypto_reasoning(prompt_file: str, cfg: dict) -> dict | None:
    """Call the Anthropic API for crypto reasoning.

    Returns the decision dict, or None on failure.
    """
    try:
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
    except IOError as exc:
        print(f"  ERROR reading crypto prompt: {exc}")
        return None

    system_instruction = (
        "Output ONLY the JSON decision object. No markdown fences, "
        "no explanation. Raw JSON with: action, asset, trade_usd, "
        "rationale, assets_evaluated, runner_up.\n"
    )

    invoke_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"  Crypto API call started at: {invoke_ts}")

    raw = _call_anthropic_api(prompt_text, system_instruction, cfg)
    if raw is None:
        return None

    print(f"  Crypto response: {len(raw)} chars")

    decision = _extract_json_decision(raw)
    if decision is None:
        print(f"  ERROR: Could not parse crypto JSON")
        print(f"  Preview: {raw[:500]}")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    decision.setdefault("timestamp", ts)
    with open(CRYPTO_DECISIONS_LOG, "a") as f:
        f.write("\n" + json.dumps(decision, indent=2))
        f.write("\n" + "=" * W + "\n")
    print(f"  Crypto decision written: {decision.get('action')} {decision.get('asset')}")

    return decision


def run_execute_crypto():
    """Phase 3C — Execute the crypto decision."""
    phase_banner("3C", "EXECUTE CRYPTO — Order Placement")

    if not _has_fresh_crypto_decision():
        print("  No fresh crypto decision. Skipping.")
        return

    from kairos_crypto_execute import main as crypto_execute_main
    crypto_execute_main()


# ── Phase 2: Reason (Claude Code acts as the engine) ────────────────

def run_reason():
    phase_banner(2, "REASON — Claude Code Reasoning Engine")
    prompt_file = os.path.join(SCRIPT_DIR, "kairos_prompt.txt")

    if not os.path.exists(prompt_file):
        print("  ERROR: kairos_prompt.txt not found. Run --gather first.")
        sys.exit(1)

    cfg = _load_claude_config()

    # ── Diagnostic: show prompt metadata ──────────────────────────────
    prompt_mtime = os.path.getmtime(prompt_file)
    prompt_ts = datetime.fromtimestamp(prompt_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    prompt_size = os.path.getsize(prompt_file)
    print(f"  Prompt written at: {prompt_ts}")
    print(f"  Prompt size: {prompt_size:,} bytes")

    # ── Clear stale raw decisions so we don't reuse old reasoning ─────
    removed = _clear_stale_raw_decisions()
    if removed:
        print(f"  Cleared {removed} stale raw decision(s) from kairos_decisions.log")
        print(f"  (These were leftover from previous cycles — not fresh reasoning)")
    else:
        print(f"  No stale raw decisions to clear.")

    # ── Load shortlist for display ────────────────────────────────────
    shortlist = get_screening_shortlist() or ["(unknown)"]
    shortlist_str = ", ".join(shortlist[:5])
    if len(shortlist) > 5:
        shortlist_str += f" ... ({len(shortlist)} total)"

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Routing: BUY/SELL/HOLD decision → Anthropic API    │")
    print("  │  (direct API call, prompt caching enabled)          │")
    print("  │                                                     │")
    print(f"  │  Evaluating {len(shortlist):>2} shortlisted tickers              │")
    print(f"  │  Candidates: {shortlist_str:<39s} │")
    print(f"  │  Model: {cfg['model']:<24s} Timeout: {cfg['timeout']}s │")
    print("  └─────────────────────────────────────────────────────┘")

    # ── Call Anthropic API directly ──────────────────────────────────
    print(f"\n  Prompt: {prompt_file}")
    start_time = time.time()

    decision = _invoke_claude_reasoning(prompt_file, cfg)

    # ── Handle result ─────────────────────────────────────────────────
    if decision is not None:
        elapsed = int(time.time() - start_time)
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  ✓ FRESH DECISION RECEIVED                         │")
        print("  └─────────────────────────────────────────────────────┘")
        print(f"  Reasoning latency: ~{elapsed}s")
        print()
        trades = decision.get("trades", [])
        if trades:
            print(f"  Trades recommended: {len(trades)}")
            for t in trades:
                print(f"    {t.get('action','?'):<4} {t.get('ticker','?'):<6} "
                      f"sector={t.get('sector','?'):<20} "
                      f"conviction={t.get('conviction','?')}")
            print(f"  Evaluated: {decision.get('tickers_evaluated', [])}")
            skipped = decision.get("skipped", "")
            if skipped:
                print(f"  Skipped: {skipped[:120]}")
        else:
            print(f"  Parsed decision: {json.dumps(decision, indent=2)[:500]}")
    else:
        print()
        print("  ╔════════════════════════════════════════════════════╗")
        print(f"  ║  FAILED: No decision from API                     ║")
        print("  ║  Phase 3 (execute) will NOT run this cycle.       ║")
        print("  ╚════════════════════════════════════════════════════╝")
        _log_reasoning_timeout(cfg)
        sys.exit(1)


# ── Phase 3: Execute ────────────────────────────────────────────────

def run_execute():
    phase_banner(3, "EXECUTE — Order Placement + Confirmation")

    if not has_fresh_decision():
        print("  ERROR: No decision in kairos_decisions.log. Run reasoning first.")
        sys.exit(1)

    # Pre-execution tax filter: check if SELL should be overridden
    _apply_tax_filter()

    from kairos_execute import main as execute_main
    execute_main()


def run_options_execute():
    """Phase 3O: HOT-CATALYST long-options execute (parallel to equity execute).

    Manages open option positions (stop / TP / DTE) then enters new long-option
    setups. Self-contained engine — selects its own contracts. dry_run is taken
    from config (currently true → simulate + log); a live order needs BOTH
    config dry_run=false AND --no-dry-run, neither of which the scheduler passes,
    so the scheduled path always simulates. IBKR (paper, port 7497) is connected
    only to read NLV; sizing falls back to the sim default if it can't connect.
    """
    phase_banner("3O", "EXECUTE OPTIONS — HOT-CATALYST")

    try:
        from kairos_catalyst_signals import load_hot_catalyst_config
        from kairos_options_execute import (
            run_options_cycle, fetch_nlv, DEFAULT_SIM_NLV,
        )
        from kairos_log_db import init_db
    except ImportError as exc:
        print(f"  SKIP: HOT-CATALYST modules unavailable ({exc})")
        return

    try:
        init_db()
        cfg = load_hot_catalyst_config()
        dry_run = True  # scheduler never passes --no-dry-run; always simulate

        ib = None
        nlv = None
        try:
            import random
            from ib_insync import IB
            ib = IB()
            ib.connect("127.0.0.1", 7497, clientId=random.randint(60, 69), timeout=10)
            nlv = fetch_nlv(ib)
        except Exception as exc:
            print(f"  IBKR NLV fetch skipped ({exc}); using sim default.")
            ib = None

        if nlv is None:
            nlv = DEFAULT_SIM_NLV
            print(f"  NLV: ${nlv:,.0f} (simulation default)")
        else:
            print(f"  NLV: ${nlv:,.0f}")

        try:
            summary = run_options_cycle(cfg, nlv, dry_run, ib=ib)
            print(f"\n  HOT-CATALYST: closed {summary['closed']}, "
                  f"entered {summary['entered']}, skipped {summary['skipped']} "
                  f"({'simulated' if dry_run else 'LIVE'})")
        finally:
            if ib:
                ib.disconnect()
    except Exception as exc:
        print(f"  HOT-CATALYST execute error (non-fatal): {exc}")


def _apply_tax_filter():
    """Read the latest decision; if SELL, run tax efficiency check."""
    from kairos_execute import read_latest_decision

    try:
        decision = read_latest_decision()
    except Exception:
        return

    if decision.get("action", "").upper() != "SELL":
        return

    ticker = decision.get("ticker", "")
    if not ticker:
        return

    print(f"\n  Tax efficiency pre-check for SELL {ticker}...")

    # Get current price from IBKR
    try:
        from ib_insync import IB, Stock
        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=5, timeout=10)
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqMarketDataType(4)
        mkt = ib.reqMktData(contract)
        ib.sleep(3)
        current_price = None
        for attr in ("last", "close", "bid", "ask"):
            val = getattr(mkt, attr, None)
            if val is not None and val == val:
                current_price = round(val, 2)
                break
        ib.cancelMktData(contract)
        ib.disconnect()
    except Exception as e:
        print(f"  Could not get price for tax check: {e}")
        return

    if not current_price:
        return

    from kairos_tax_efficiency import check_tax_override

    # high_price: use entry price * 1.1 as rough proxy if we don't track highs
    # A real implementation would track high watermarks in the DB
    modified = check_tax_override(decision, current_price)

    if modified.get("tax_override"):
        # Rewrite the decision in the log so execute reads the updated version
        import json
        log_path = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
        with open(log_path, "a") as f:
            f.write("\n" + json.dumps(modified, indent=2))
            f.write("\n" + "=" * W + "\n")
        print(f"  Decision overridden to HOLD — written to kairos_decisions.log")


# ── Ollama model check ──────────────────────────────────────────────

def _check_ollama_models():
    """Verify both Ollama models are installed, then warm them up.

    After OLLAMA_KEEP_ALIVE (5 min), models are unloaded from VRAM.
    The warmup step sends a trivial prompt to each model so the ~30s
    cold-start reload happens here (with a generous 90s timeout)
    rather than inside the timed pipeline phases.
    """
    try:
        from kairos_ollama import check_required_models, warmup
    except ImportError:
        return

    status = check_required_models()

    screen_icon = "✓" if status["screen_model"] else "✗"
    reason_icon = "✓" if status["reason_model"] else "✗"
    print(f"  Ollama models:")
    print(f"    {screen_icon} Screen: {status['screen_name']}"
          f" (Tier 1, speed-optimized)")
    print(f"    {reason_icon} Reason: {status['reason_name']}"
          f" (Tier 2, quality-optimized)")

    if not status["all_ok"]:
        print()
        print("  ╔════════════════════════════════════════════════════════╗")
        print("  ║  MISSING OLLAMA MODELS — pipeline cannot start       ║")
        print("  ╠════════════════════════════════════════════════════════╣")
        for m in status["missing"]:
            print(f"  ║  Install with:  ollama pull {m:<25s}     ║")
        print("  ╚════════════════════════════════════════════════════════╝")
        print()
        print(f"  Installed models: {', '.join(status['available']) or '(none)'}")
        sys.exit(1)

    # Preload screen model only — reason model loads on-demand after screening
    # completes. keep_alive=0 ensures it unloads immediately after use,
    # so Qwen3 never coexists in memory with phi4-mini.
    warmup(status["screen_name"])


# ── Cycle-counter scheduler ──────────────────────────────────────────

def _load_scheduler_config() -> dict:
    """Load scheduler intervals from kairos_config.json."""
    defaults = {
        "cycle_interval_minutes": 30,
        "intervals": {"intake": 2, "audit": 1, "screener": 1, "reasoning": 1},
    }
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        sched = cfg.get("scheduler", {})
        defaults["cycle_interval_minutes"] = sched.get("cycle_interval_minutes", 30)
        if sched.get("intervals"):
            defaults["intervals"].update(sched["intervals"])
    except (IOError, json.JSONDecodeError):
        pass
    return defaults


def _load_state() -> dict:
    """Load cycle state from kairos_state.json, creating if missing."""
    defaults = {"cycle_count": 0, "last_run": None}
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        defaults["cycle_count"] = state.get("cycle_count", 0)
        defaults["last_run"] = state.get("last_run")
    except (IOError, json.JSONDecodeError):
        pass
    return defaults


def _save_state(state: dict) -> None:
    """Persist cycle state to kairos_state.json."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def _should_run(component: str, cycle_count: int, intervals: dict) -> bool:
    """Check if a component should run on this cycle.

    Cycle 0 → all components run (clean startup).
    Otherwise: run if cycle_count % interval == 0.
    """
    if cycle_count == 0:
        return True
    interval = intervals.get(component, 1)
    return (cycle_count % interval) == 0


def _cycles_until_next(component: str, cycle_count: int, intervals: dict) -> int:
    """How many cycles until the next run of this component."""
    interval = intervals.get(component, 1)
    if interval <= 1:
        return 0
    remainder = cycle_count % interval
    return interval - remainder if remainder != 0 else 0


def print_cycle_status():
    """Print the current scheduler state and next-run schedule."""
    sched = _load_scheduler_config()
    state = _load_state()
    cycle = state["cycle_count"]
    last_run = state["last_run"] or "(never)"
    intervals = sched["intervals"]
    interval_min = sched["cycle_interval_minutes"]

    print(f"  Current cycle: {cycle}")
    print(f"  Last run:      {last_run}")
    print(f"  Cycle length:  {interval_min} minutes")
    print()
    print("  Component schedule:")
    for comp in ("intake", "audit", "screener", "reasoning"):
        iv = intervals.get(comp, 1)
        until = _cycles_until_next(comp, cycle, intervals)
        runs_now = "NOW" if _should_run(comp, cycle, intervals) else f"in {until} cycle{'s' if until != 1 else ''}"
        print(f"    {comp + ':':13s}runs every {iv} cycle{'s' if iv != 1 else ' ':2s}| next run: {runs_now}")


def run_scheduled_cycle(args) -> None:
    """Execute one scheduler tick using the cycle-counter model.

    Loads config + state, determines which components should run,
    executes them in order, then increments cycle_count.
    """
    run_equity = False
    run_crypto = False
    sched = _load_scheduler_config()
    state = _load_state()
    cycle = state["cycle_count"]
    intervals = sched["intervals"]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS SCHEDULER — Cycle {cycle} — {ts}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    # Run comprehensive health check before anything else
    try:
        from kairos_health import run_health_check
        health_status = run_health_check(quick=args.cycle)
        
        # If there are critical failures, exit immediately
        if health_status.has_failures:
            print("\n❌ CRITICAL: System health check failed. Aborting cycle.")
            sys.exit(1)
            
    except Exception as health_error:
        print(f"\n⚠️  WARNING: Health check failed to run: {health_error}")
        # Continue execution but with warning

    # Show what will run this cycle
    components = ["intake", "audit", "screener", "reasoning"]
    will_run = [c for c in components if _should_run(c, cycle, intervals)]
    will_skip = [c for c in components if c not in will_run]
    print(f"  Will run:  {', '.join(will_run) if will_run else '(none)'}")
    if will_skip:
        print(f"  Skipping:  {', '.join(will_skip)}")

    try:
        from kairos_log_db import init_db
        init_db()
    except ImportError:
        pass

    # Create shared IBKR connection for this cycle
    shared_ib_connection = None
    if run_equity:
        try:
            from ib_insync import IB
            shared_ib_connection = IB()
            shared_ib_connection.connect("127.0.0.1", 7497, clientId=5, timeout=10)
            print(f"  IBKR connection established (clientId=5) for shared use")
        except Exception as exc:
            print(f"  WARNING: Shared IBKR connection failed: {exc}")
            shared_ib_connection = None

    use_ollama = not args.no_ollama
    if use_ollama:
        _check_ollama_models()

    run_equity = args.mode in ("equity", "both")
    run_crypto = args.mode in ("crypto", "both")
    do_screen = not args.no_screen and use_ollama and run_equity

    # Phase 0: Stale order cleanup — cancel any open IBKR orders older
    # than 8 hours so they don't tie up cash across cycles.
    if run_equity:
        phase_banner(0, "STALE ORDER CLEANUP")
        try:
            from kairos_execute import cancel_stale_orders
            n = cancel_stale_orders(max_age_hours=8)
            print(f"  Stale orders cancelled: {n}")
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  WARNING: Stale order cleanup failed: {exc}\n{tb}")
            _log_phase_crash("stale_order_cleanup", exc, tb)

    # a. Intake
    if _should_run("intake", cycle, intervals):
        phase_banner("I", "TIER C INTAKE")
        try:
            from kairos_intake import run_intake
            run_intake(dry_run=args.screen_dry_run)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  WARNING: Intake failed: {exc}\n{tb}")
            _log_phase_crash("intake", exc, tb)

    # b. Audit
    if _should_run("audit", cycle, intervals):
        phase_banner("A", "TIER C AUDIT")
        try:
            from kairos_tier_c import audit
            audit()
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  WARNING: Tier C audit failed: {exc}\n{tb}")
            _log_phase_crash("audit", exc, tb)

    # b1. IPO intake — daily at market open (9:30-9:35 ET, weekdays).
    # EDGAR + Finnhub don't need 30-min cadence; one scan/day is plenty.
    # The gate inside is_full_scan_due() also checks cache['last_full_scan']
    # so a single window crossing won't fire it twice.
    if run_equity:
        try:
            from kairos_ipo_intake import is_full_scan_due, run_ipo_intake
            if is_full_scan_due():
                phase_banner("IPO", "IPO INTAKE — Daily Watchlist + EDGAR + News")
                run_ipo_intake()
                # Event seasonality scan rides on the same daily cadence:
                # fetches yfinance price/high, scores active events, and
                # posts a Slack alert when a fresh PRE-EVENT-DIP fires.
                try:
                    from kairos_signals_events import run_event_scan
                    phase_banner("EVT", "EVENT SEASONALITY SCAN — Active Calendar Events")
                    run_event_scan()
                except Exception as evt_exc:
                    print(f"  WARNING: Event seasonality scan failed: {evt_exc}")
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  WARNING: IPO intake failed: {exc}\n{tb}")
            _log_phase_crash("ipo_intake", exc, tb)

    # b1b. HOT-IPO tracker — discovery, conviction scoring, capital
    # reservation. Runs immediately after intake on the same 9:30 ET
    # weekday cadence (gate widens to 9:30–9:40 so both fit).
    # Reservations default to dry_run=True; flip kairos_config["hot_ipo"]
    # ["dry_run"] = false to actually reserve & liberate.
    if run_equity:
        try:
            from kairos_ipo_tracker import is_tracker_due, run_ipo_tracker
            if is_tracker_due():
                phase_banner("IPOT", "HOT-IPO TRACKER — Pipeline + Conviction Scoring")
                _dry = True
                try:
                    import json as _json
                    with open("/Users/jelmore/TradingAgents/kairos_config.json") as _f:
                        _dry = bool(_json.load(_f).get("hot_ipo", {}).get("dry_run", True))
                except Exception:
                    pass
                run_ipo_tracker(dry_run=_dry)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  WARNING: IPO tracker failed: {exc}\n{tb}")
            _log_phase_crash("ipo_tracker", exc, tb)

    # b2. Thesis review — daily at 9:35am ET (weekdays only)
    if run_equity:
        try:
            from zoneinfo import ZoneInfo
            et_now = datetime.now(ZoneInfo("America/New_York"))
            thesis_time = sched.get("thesis_review_time_et", "09:35")
            thesis_h, thesis_m = int(thesis_time.split(":")[0]), int(thesis_time.split(":")[1])
            is_weekday = et_now.weekday() < 5
            # Run if we're within the 30-minute window starting at thesis_time
            in_window = (et_now.hour == thesis_h
                         and thesis_m <= et_now.minute < thesis_m + 30)
            if is_weekday and in_window:
                phase_banner("TR", "DAILY THESIS REVIEW — 9:35 ET")
                from kairos_thesis_review import run_thesis_review
                run_thesis_review(dry_run=args.screen_dry_run)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  WARNING: Thesis review failed: {exc}\n{tb}")
            _log_phase_crash("thesis_review", exc, tb)

    # c. Screener (includes regime + stop-loss)
    shortlist = None
    if _should_run("screener", cycle, intervals) and do_screen:
        if use_ollama:
            run_orchestrate(verbose=False)
        run_regime()
        # Phase 0.1: Stop-loss check after regime
        if run_equity:
            try:
                run_stoploss(ib=shared_ib_connection)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"  WARNING: Stop-loss phase failed: {exc}\n{tb}")
                _log_phase_crash("stoploss", exc, tb)
            # Thesis checkpoints — once per cycle, after stop-loss has
            # had a chance to close anything off. Failure here must NOT
            # block the rest of the cycle.
            try:
                from kairos_ml_thesis import run_thesis_checkpoints
                run_thesis_checkpoints(ib=shared_ib_connection)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"  WARNING: Thesis checkpoints failed: {exc}\n{tb}")
                _log_phase_crash("thesis_checkpoints", exc, tb)
        screen_result = run_screen(dry_run=args.screen_dry_run, max_tier2=args.max_tier2)
        shortlist = screen_result["shortlist"]
        
        # Phase 0.7: ML pattern recognition after screening
        if run_equity and shortlist:
            try:
                ml_scored = run_ml_phase(shortlist)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"  WARNING: ML phase failed: {exc}\n{tb}")
                _log_phase_crash("ml_phase", exc, tb)
                ml_scored = []

    # d. Reasoning + Execute
    if _should_run("reasoning", cycle, intervals):
        try:
            if run_equity:
                run_gather(use_ollama=use_ollama, shortlist=shortlist)
            if run_crypto:
                run_gather_crypto(mode=args.mode, use_ollama=use_ollama)
            if run_equity:
                run_reason()
                run_execute()
                run_options_execute()
            if run_crypto:
                run_reason_crypto()
                run_execute_crypto()
        except Exception as exc:
            print(f"  WARNING: Reasoning/Execute phase failed: {exc}")

    # Increment cycle and persist
    state["cycle_count"] = cycle + 1
    state["last_run"] = ts
    _save_state(state)

    print()
    print("━" * W)
    print(f"  CYCLE {cycle} COMPLETE — next cycle: {cycle + 1}")
    print(f"  State saved → {STATE_FILE}")
    print("━" * W)


# ── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kairos Full Pipeline (M9 / Phase B)")
    parser.add_argument("--gather",      action="store_true", help="Phase 1 only")
    parser.add_argument("--execute",     action="store_true", help="Phase 3 only")
    parser.add_argument("--orchestrate", action="store_true", help="Phase 0 only (Ollama checks)")
    parser.add_argument("--screen",      action="store_true", help="Phase 0.5 only (Tier 1 screen)")
    parser.add_argument("--cycle",       action="store_true", help="Run one scheduler cycle (cycle-counter model)")
    parser.add_argument("--cycle-status", action="store_true", help="Print scheduler state and exit")
    parser.add_argument("--no-ollama",   action="store_true", help="Skip Phase 0")
    parser.add_argument("--no-screen",   action="store_true", help="Skip Tier 1 screening")
    parser.add_argument("--screen-dry-run", action="store_true",
                        help="Tier 1 screen without Finnhub (fast test)")
    parser.add_argument("--max-tier2",   type=int, default=15,
                        help="Max tickers for Tier 2 (default: 15)")
    parser.add_argument(
        "--mode",
        choices=["equity", "crypto", "both"],
        default="equity",
        help="Asset mode: equity (default), crypto, or both",
    )
    args = parser.parse_args()

    if args.cycle_status:
        print_cycle_status()
        return

    if args.cycle:
        try:
            run_scheduled_cycle(args)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(f"\n  FATAL: unhandled exception in run_scheduled_cycle:\n{tb}")
            # Log structured event to kairos_monitor.log
            try:
                from kairos_alerts import log_monitor_event, post_message
                # Truncate traceback to last 500 chars for the JSON record
                tb_tail = tb.strip()[-500:]
                log_monitor_event(
                    "SCHEDULER_CRASH",
                    success=False,
                    error=tb_tail,
                )
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                post_message(
                    "alerts",
                    f":rotating_light: *Kairos Scheduler Crash*\n"
                    f"An unhandled exception killed the pipeline.\n"
                    f"```{tb_tail}```\n"
                    f"Timestamp: `{ts}`\n"
                    f"The scheduler will retry on the next 30-min tick.",
                )
            except Exception as alert_exc:
                print(f"  WARNING: crash alert failed: {alert_exc}")
            raise
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS FULL PIPELINE — {ts}".ljust(W + 1) + "║")
    print(f"║  Mode: {args.mode.upper():<10}  Ollama: {'ENABLED' if not args.no_ollama else 'DISABLED'}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    try:
        from kairos_log_db import init_db
        init_db()
        print("  Database: kairos.db ready")
    except ImportError:
        print("  WARNING: kairos_log_db not found — skipping DB init")

    use_ollama = not args.no_ollama

    # Verify both Ollama models are available before starting
    if use_ollama:
        _check_ollama_models()
    run_equity = args.mode in ("equity", "both")
    run_crypto = args.mode in ("crypto", "both")

    do_screen = not args.no_screen and use_ollama and run_equity

    if args.orchestrate:
        run_orchestrate()
        return

    if args.screen:
        run_screen(dry_run=args.screen_dry_run, max_tier2=args.max_tier2)
        return

    if args.gather:
        if use_ollama:
            run_orchestrate(verbose=False)
        run_regime()
        shortlist = None
        if do_screen:
            screen_result = run_screen(dry_run=args.screen_dry_run, max_tier2=args.max_tier2)
            shortlist = screen_result["shortlist"]
        if run_equity and do_screen and shortlist:
            try:
                ml_scored = run_ml_phase(shortlist)
            except Exception as exc:
                print(f"  WARNING: ML phase failed: {exc}")
                ml_scored = []
        if run_equity:
            run_gather(use_ollama=use_ollama, shortlist=shortlist)
        if run_crypto:
            run_gather_crypto(mode=args.mode, use_ollama=use_ollama)
        return

    if args.execute:
        run_execute()
        return

    # Full pipeline: 0 → 0R → 0.1 → [0.5] → [0.7] → 1 → [1C] → 2 → 3
    if use_ollama:
        run_orchestrate()

    # IPO intake — runs after orchestrate, before regime/screen, so any
    # IPO tickers promoted to Tier C are visible to the screener.
    if run_equity:
        try:
            from kairos_ipo_intake import run_ipo_intake
            run_ipo_intake()
        except Exception as exc:
            print(f"  WARNING: IPO intake failed: {exc}")

    regime_result = run_regime()

    # Phase 0.1: Stop-loss check (after regime, before screening)
    if run_equity:
        try:
            run_stoploss(ib=shared_ib_connection)
        except Exception as exc:
            print(f"  WARNING: Stop-loss phase failed: {exc}")
        try:
            from kairos_ml_thesis import run_thesis_checkpoints
            run_thesis_checkpoints(ib=shared_ib_connection)
        except Exception as exc:
            print(f"  WARNING: Thesis checkpoints failed: {exc}")

    shortlist = None
    if do_screen:
        screen_result = run_screen(dry_run=args.screen_dry_run, max_tier2=args.max_tier2)
        shortlist = screen_result["shortlist"]
        
        # Phase 0.7: ML pattern recognition after screening
        if run_equity and shortlist:
            try:
                ml_scored = run_ml_phase(shortlist)
            except Exception as exc:
                print(f"  WARNING: ML phase failed: {exc}")
                ml_scored = []

    if run_equity:
        run_gather(use_ollama=use_ollama, shortlist=shortlist)
    if run_crypto:
        run_gather_crypto(mode=args.mode, use_ollama=use_ollama)
    if run_equity:
        run_reason()
        run_execute()
        run_options_execute()
    if run_crypto:
        run_reason_crypto()
        run_execute_crypto()

    print()
    print("━" * W)
    print("  KAIROS PIPELINE COMPLETE")
    print(f"  Mode: {args.mode.upper()}")
    if do_screen and shortlist is not None:
        sl_len = len(shortlist)
        uni_size = screen_result.get("universe_size", "?")
        print(f"  Tier 1 (Ollama): screened {uni_size} tickers → {sl_len} passed to Tier 2")
    if use_ollama:
        print("  Ollama handled: screening, stale detection, news filtering")
    print("  Claude Code handled: BUY/SELL/HOLD decisions")
    if run_crypto:
        print("  Crypto: 5-asset universe, HOT-RSI + HOT-REVERSION signals")
    if regime_result:
        r = regime_result["regime"]
        changed = " (CHANGED)" if regime_result.get("changed") else ""
        print(f"  Macro regime: {r}{changed}")
    print("━" * W)


if __name__ == "__main__":
    main()
