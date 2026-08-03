# Security — secret handling

## Rule

**Live secrets live in the environment. Never in a tracked file.**

`kairos_config.json` is tracked in git. Anything written into it is one
`git commit` away from being published, and unpublishing it means rewriting
history for every clone.

## What happened (2026-08-02)

A live Slack bot token was stored in `kairos_config.json` at `slack.bot_token`
and committed. Remediation required `git filter-repo` across the whole history
plus a force-push; every commit hash in the repository changed, and any
uncommitted work in the tree at that moment was lost.

Root cause was not carelessness — it was the shape of the config file. The
field `"bot_token": ""` sitting in a tracked JSON file *invites* someone to fill
it in. The fix is structural: the field stays blank forever, and the loader
reads the environment first.

## Where secrets belong

| secret | environment variable | set in |
|---|---|---|
| Slack bot token | `SLACK_BOT_TOKEN` | `~/.zshrc`, launchd plist |
| Slack app token (Socket Mode) | `SLACK_APP_TOKEN` | `~/.zshrc`, launchd plist |
| Finnhub API key | `FINNHUB_API_KEY` | `~/.zshrc` |
| Kalshi key id | `KALSHI_KEY_ID` | `~/.zshrc` |

Add one with:

```sh
echo 'export SLACK_BOT_TOKEN="xoxb-…"' >> ~/.zshrc && source ~/.zshrc
```

Long-running daemons started by launchd do not read `~/.zshrc` — they need the
value in the plist's `EnvironmentVariables` block as well. `rotate_secrets.sh`
updates the plist in place.

## Fields that must stay empty

- `kairos_config.json` → `slack.bot_token` — legacy fallback only. A non-empty
  value now prints a warning on every load.
- `kairos_config.json` → `renaissance_capital.api_key` — same rule.

Channel IDs, tickers, thresholds, and tuning parameters are **not** secrets and
belong in the config as normal.

## Token resolution order

`kairos_alerts._load_slack_config()` is the single chokepoint every Slack path
resolves through (`kairos_alerts`, `kairos_axis_weights`, `kairos_slack_cards`).
`kairos_commander` and `kairos_arbiter_commander` have their own
`get_bot_token()` with identical ordering:

1. `SLACK_BOT_TOKEN` from the environment — wins outright.
2. `kairos_config.json` `slack.bot_token` — legacy fallback, expected `""`,
   warns when non-empty.

## Before committing

```sh
git diff --cached | grep -nE 'xoxb-|xoxp-|xapp-|sk-|ghp_|AKIA|-----BEGIN'
```

Any hit means stop: remove the value, put it in the environment, and re-stage.
If a secret has already been pushed, treat it as compromised — **rotate it
first**, then scrub history. Scrubbing alone does not un-leak a token.
