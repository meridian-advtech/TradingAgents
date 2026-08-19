#!/usr/bin/env python3
"""
Kairos Model Watch — Weekly Tier 1 AI Model Release Scan

Runs weekly (Monday 07:30 ET, pre-market) via com.kairos.model_watch.plist.
Calls Claude (Sonnet 5, web search enabled) to check whether any new model
release in the past week meets the Tier 1 immediate-evaluation criteria
defined in the Olympus Hardware Discussion (2026-06-10):

    - Any new model from Meta, Alibaba/Qwen, DeepSeek, or Mistral above 70B
    - Any new MoE model that fits a 96-256GB memory envelope
    - Any model explicitly claiming superior reasoning, financial analysis,
      or structured-output capability
    - A new quantization method that materially changes what fits in the
      memory budget

Posts a Slack summary to #kairos-model-watch ONLY when something matches —
silence on a clean week is intentional, not a failure signal. A local state
file prevents re-alerting on a model already flagged in a prior run.

Usage:
    python3 kairos_model_watch.py            # normal run
    python3 kairos_model_watch.py --force    # ignore state file, re-check everything
    python3 kairos_model_watch.py --test     # dry run, print instead of posting to Slack
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "kairos_model_watch_state.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "kairos_model_watch.log")

TIER1_PROMPT = """You are screening AI model releases from the past 7 days for relevance to \
Kairos, an autonomous equity/crypto trading system. Kairos runs a local Ollama stack \
(screener + router, currently phi4-mini and qwen3.6:35b-a3b, on a 32GB MacBook Air) plus \
a trusted-API council (Claude Sonnet 5 / Opus 5, Mistral Medium 3.5). Chinese-origin models \
are acceptable ONLY as local open-weight downloads, never via their hosted API.

Search the web for AI model releases from the last 7 days and flag ONLY releases matching \
at least one of these Tier 1 criteria:

1. A new model from Meta, Alibaba/Qwen, DeepSeek, or Mistral above 70B parameters
2. A new MoE model that would fit a 96-256GB unified-memory envelope (i.e. NOT a \
   700B+/1T+ flagship that's unusable on consumer hardware)
3. A model explicitly claiming superior reasoning, financial analysis, or structured \
   JSON output capability
4. A new quantization method that materially changes what fits in a 32GB-256GB memory budget
5. A new Claude or Mistral model release (any size) via their official API

Respond with ONLY a JSON object, no other text, in this exact shape:

{
  "matches": [
    {
      "name": "model name",
      "source": "lab/company",
      "date": "approximate release date",
      "criteria": "which Tier 1 criterion it matches",
      "summary": "one to two sentence summary of why it's relevant",
      "size_note": "parameter count / memory footprint if known"
    }
  ]
}

If nothing from the past 7 days matches, return {"matches": []}. Do not include anything \
that fails all five criteria — routine minor version bumps, fine-tunes, or huge \
1T+ parameter closed-API-only flagships from Chinese labs do not qualify unless they meet \
criterion 5 (they don't, since 5 is Claude/Mistral only)."""


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"flagged_models": [], "last_run": None}


def _save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_scan() -> list[dict]:
    """Call Claude with web search to check for Tier 1 model releases."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        output_config={"effort": "medium"},
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 8,
        }],
        messages=[{"role": "user", "content": TIER1_PROMPT}],
    )

    # Web search responses interleave server_tool_use / web_search_tool_result
    # blocks with text blocks. We only want the final text block, which
    # should contain the JSON per the prompt's instructions.
    text_parts = [b.text for b in response.content if b.type == "text"]
    raw = "".join(text_parts).strip()

    # Strip markdown code fences if Claude wrapped the JSON despite instructions.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("matches", [])
    except json.JSONDecodeError:
        _log(f"WARNING: could not parse response as JSON. Raw: {raw[:500]}")
        return []


def _format_slack_message(matches: list[dict]) -> str:
    if not matches:
        return None

    lines = [f"*Kairos Model Watch — {len(matches)} Tier 1 release(s) this week*\n"]
    for m in matches:
        lines.append(
            f"• *{m.get('name', 'unknown')}* ({m.get('source', 'unknown')}, "
            f"{m.get('date', 'unknown date')})\n"
            f"  _{m.get('criteria', '')}_ — {m.get('size_note', 'size unknown')}\n"
            f"  {m.get('summary', '')}"
        )
    lines.append(
        "\n_Tier 1 = 30 min review to decide go/no-go on full benchmarking. "
        "See Olympus Hardware Discussion (2026-06-10) for the tiered framework._"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="ignore state file, re-check and re-flag everything")
    parser.add_argument("--test", action="store_true",
                         help="dry run — print instead of posting to Slack")
    args = parser.parse_args()

    _log("Starting weekly Tier 1 model scan...")
    state = _load_state()
    flagged_before = set(state.get("flagged_models", []))

    matches = run_scan()
    _log(f"Scan complete: {len(matches)} raw match(es) from Claude")

    if not args.force:
        new_matches = [m for m in matches if m.get("name") not in flagged_before]
    else:
        new_matches = matches

    if not new_matches:
        _log("No new Tier 1 releases this week. No Slack post (silence is expected).")
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        _save_state(state)
        return

    message = _format_slack_message(new_matches)
    _log(f"Found {len(new_matches)} new match(es): "
         f"{', '.join(m.get('name', '?') for m in new_matches)}")

    if args.test:
        print("\n--- DRY RUN, not posting to Slack ---")
        print(message)
    else:
        sys.path.insert(0, SCRIPT_DIR)
        from kairos_alerts import post_message
        ok = post_message("model_watch", message)
        if not ok:
            _log("WARNING: Slack post failed. See kairos_alerts output above.")

    state["flagged_models"] = list(flagged_before | {m.get("name") for m in new_matches})
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)
    _log("Done.")


if __name__ == "__main__":
    main()
