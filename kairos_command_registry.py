"""
Kairos Command Registry — the one place a structured command name is defined.

WHY THIS FILE EXISTS
────────────────────────────────────────────────────────────────────────────
Two independent Slack processes poll two different channels, and they have
fundamentally different interaction models:

  kairos_commander.py        — a rigid exact-match command parser. Only acts
                               on a recognized command; says so when it
                               doesn't recognize one.
  kairos_arbiter_commander.py — NOT a parser. A genuine conversation: every
                               human message in #kairos-arbiter gets full
                               portfolio context + rolling history + a live
                               Mistral reply.

Because the arbiter side answers *literally everything* in its channel, the
two had no shared awareness of each other's vocabulary. On 2026-08-18 J typed
`!pending` in #kairos-arbiter expecting the commander's tappable approval
cards; the arbiter answered instead, having read the words conversationally.
Nothing was broken — the two systems simply had no shared vocabulary.

This file is that shared vocabulary. It does NOT merge the two models: each
process keeps its own dispatch and its own personality. It only answers one
question for both of them: "is this text a structured command, and which?"

ROUTING CONTRACT
────────────────────────────────────────────────────────────────────────────
  #kairos-commands  polled by kairos_commander.py only.
                    Recognized command  -> handled.
                    Unrecognized        -> ":grey_question: I didn't
                                           understand that — try `!help`."
                    (unchanged, and deliberately so)

  #kairos-arbiter   polled by BOTH processes, which split the channel by
                    vocabulary rather than by message:
                      explicit `!cmd` in STRUCTURED_COMMANDS
                                          -> kairos_commander.py handles it;
                                             the arbiter skips it entirely (no
                                             Mistral call, no reply, nothing
                                             appended to conversation history).
                      anything else       -> kairos_commander.py stays SILENT
                                             (no error, not its channel to
                                             comment on); the arbiter handles
                                             it exactly as it always has.

    The two rules are complements, so every message is answered exactly once.
    Both sides get that decision from is_structured_command() below — one
    function, called by both — so they cannot drift into a double-reply or a
    silent drop.

MATCHING CONVENTION
────────────────────────────────────────────────────────────────────────────
A command is always matched on the FIRST WHITESPACE-DELIMITED TOKEN only;
everything after it is opaque argument text (`why` and `watchlist` use it).
Match is on the whole token, never a substring: "!whyever" is not `why`.

Leading Slack mention markup is stripped first, so "<@U123> !status" works.

There are two tiers, because the two channels can afford different amounts of
ambiguity:

  DEDICATED channel (#kairos-commands) — split_command().
    Accepts both `!status` and a bare `status`. A bare first token counts only
    if it is in ALL_COMMANDS, so "status please" is a command and "the status
    of things" is not. This is the historical kairos_commander.py behavior and
    is preserved exactly.

  SHARED channel (#kairos-arbiter) — is_structured_command().
    The leading `!` is REQUIRED. Bare words do not count, no matter how well
    they match.

WHY THE SHARED CHANNEL REQUIRES `!`
────────────────────────────────────────────────────────────────────────────
Because in a conversational channel the bare-word rule is actively harmful.
Command words like why / status / performance / regime / chain / pause are
ordinary English sentence-openers, and #kairos-arbiter is where J asks things
like:

    "why did you flag AMAT?"          <- the documented use case
    "status of the portfolio?"
    "why is the regime CAUTION"

Under bare-word matching every one of those parses as a command: the first
would run `!why` against the ticker "did". Worse, it would be swallowed in
BOTH directions — kairos_commander.py would claim it (dispatching nonsense),
and kairos_arbiter_commander.py would stay silent because it saw a command.
The human gets a wrong answer or none at all.

Requiring `!` makes the split unambiguous and puts it under the human's
control: punctuation, not vocabulary, decides which system answers. `!why
AMAT` is a command; "why did you flag AMAT?" is a conversation. Both are
reachable, which is the whole point of not merging the two bots.

CHANGING THIS FILE
────────────────────────────────────────────────────────────────────────────
Adding a command to STRUCTURED_COMMANDS makes it work in BOTH channels and
makes the arbiter stop replying to it. That is a two-process behavior change
from one edit — which is the point, but check both sides still make sense.

Deliberately NOT structured commands: `run` and `dry-run`. They live in
CYCLE_COMMANDS and stay exclusive to #kairos-commands, because triggering a
live trading cycle from the conversational channel — where the arbiter is
mid-discussion about hypothetical config changes — is not a thing that should
be one typo away.
"""

from __future__ import annotations

import re
from typing import Optional

# ── The vocabulary ────────────────────────────────────────────────────

# Commands that work in BOTH #kairos-commands and #kairos-arbiter, and which
# the arbiter must therefore leave alone. Frozen: importers read this, never
# mutate it.
#
# "proposals" was renamed from "pending" on 2026-08-19. It re-posts the
# tappable approval cards. The rename frees the word "pending" for the
# conversational side, which was already using it informally ("what's
# pending?") and which now keeps that phrasing.
STRUCTURED_COMMANDS: frozenset[str] = frozenset({
    "help",
    "status",
    "positions",
    "performance",
    "why",          # takes an argument: !why TICKER
    "regime",
    "ipo",
    "chain",
    "watchlist",    # takes an argument: !watchlist add|remove|list [TICKER]
    "pause",
    "resume",
    "proposals",
})

# Pipeline triggers. #kairos-commands ONLY — see the docstring's last section.
CYCLE_COMMANDS: frozenset[str] = frozenset({"run", "dry-run"})

# Spelling variants accepted for the bare (no-`!`) form. kairos_commander.py
# owns the canonicalization of these at dispatch time.
COMMAND_ALIASES: dict[str, str] = {"dryrun": "dry-run"}

# Every token kairos_commander.py will accept WITHOUT a leading `!`.
ALL_COMMANDS: frozenset[str] = (
    STRUCTURED_COMMANDS | CYCLE_COMMANDS | frozenset(COMMAND_ALIASES)
)

# ── Parsing ───────────────────────────────────────────────────────────

# "<@U12345>" / "<#C12345|kairos-commands>" — Slack mention markup.
_MENTION = re.compile(r"<[@#][A-Z0-9]+(?:\|[^>]*)?>")

# An explicit "!command rest-of-line".
_BANG = re.compile(r"^\s*!\s*([a-z\-]+)\s*(.*?)\s*$", re.IGNORECASE)


def split_command(text: str) -> Optional[tuple[str, str]]:
    """Return (command, args) for a command-shaped message, else None.

    Implements the MATCHING CONVENTION documented at the top of this file.
    `command` is lowercased; `args` is the remainder, stripped, and may be "".

    This is the single definition of "what counts as a command" in Kairos.
    kairos_commander.py parses with it to decide what to dispatch, and
    kairos_arbiter_commander.py checks it to decide what to stay out of; that
    shared source is what keeps a message from being answered twice or not at
    all.

    Note that a `!`-prefixed word is returned even when it is not a known
    command, so the caller can distinguish "you tried to issue a command I
    don't know" from "this is ordinary conversation".
    """
    if not text:
        return None

    cleaned = _MENTION.sub("", text).strip()
    if not cleaned:
        return None

    m = _BANG.match(cleaned)
    if m:
        return m.group(1).lower(), m.group(2).strip()

    # Bare keyword: only a known command counts, so prose stays prose.
    lowered = cleaned.lower()
    first = lowered.split()[0]
    if first in ALL_COMMANDS:
        return first, lowered[len(first):].strip()
    return None


def command_name(text: str) -> Optional[str]:
    """The command token in `text`, or None if it isn't command-shaped.

    Uses the DEDICATED-channel tier (bare words allowed).
    """
    parsed = split_command(text)
    return parsed[0] if parsed else None


def explicit_command(text: str) -> Optional[tuple[str, str]]:
    """Like split_command, but ONLY the `!`-prefixed form counts.

    Returns (command, args) or None. The command is returned even when it is
    not a known one, so a caller can tell "unknown command" from "prose".
    """
    if not text:
        return None
    cleaned = _MENTION.sub("", text).strip()
    if not cleaned:
        return None
    m = _BANG.match(cleaned)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).strip()


def is_structured_command(text: str) -> bool:
    """True if `text` is a command kairos_commander.py owns in a SHARED channel.

    Requires the explicit `!` form — see WHY THE SHARED CHANNEL REQUIRES `!`
    above; bare prose must stay conversational.

    This single function decides the split in #kairos-arbiter, and BOTH sides
    call it:

      kairos_arbiter_commander.py — True means return immediately: no Mistral
        call, no reply, nothing appended to conversation history.
      kairos_commander.py         — True means dispatch it; False means stay
        completely silent.

    They are exact complements because they read the same answer from here, so
    every message in the shared channel is answered exactly once.
    """
    parsed = explicit_command(text)
    return parsed is not None and parsed[0] in STRUCTURED_COMMANDS
