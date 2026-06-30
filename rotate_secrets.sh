#!/bin/bash
# Kairos credential rotation — updates env locations only (NOT git-tracked config).
# Secrets are read via hidden prompts; nothing is echoed or written to shell history.
set -uo pipefail

LA="$HOME/Library/LaunchAgents"
ZSHRC="$HOME/.zshrc"
TS=$(date +%Y%m%d_%H%M%S)

SLACK_PLISTS=(
  "com.kairos.scheduler.plist"
  "com.kairos.commander.plist"
  "com.kairos.arbiter.commander.plist"
  "com.kairos.arbiter.daily.plist"
  "com.kairos.arbiter.weekly.plist"
)

echo "=== Kairos credential rotation ==="
echo

# ---- 1. Slack bot token ----
printf "Paste NEW Slack bot token (xoxb-...), then Enter: "
read -rs SLACK_TOK; echo
if [[ -z "${SLACK_TOK}" ]]; then
  echo "  (empty — skipping Slack)"
else
  if [[ "${SLACK_TOK}" != xoxb-* ]]; then
    echo "  WARNING: that doesn't start with xoxb- . Continue anyway? [y/N]"
    read -r ok; [[ "$ok" == "y" ]] || { echo "  aborted Slack update"; SLACK_TOK=""; }
  fi
fi

if [[ -n "${SLACK_TOK}" ]]; then
  for p in "${SLACK_PLISTS[@]}"; do
    f="$LA/$p"
    [[ -f "$f" ]] || { echo "  skip (missing): $p"; continue; }
    cp "$f" "$f.bak_$TS"
    # Replace the <string> on the line AFTER <key>SLACK_BOT_TOKEN</key>
    /usr/bin/python3 - "$f" "$SLACK_TOK" <<'PY'
import sys,re
f,tok=sys.argv[1],sys.argv[2]
s=open(f).read()
new=re.sub(r'(<key>SLACK_BOT_TOKEN</key>\s*<string>)(.*?)(</string>)',
           lambda m:m.group(1)+tok+m.group(3), s, count=1, flags=re.S)
open(f,'w').write(new)
print("  updated", f.split("/")[-1]) if new!=s else print("  NO CHANGE (pattern not found):", f.split("/")[-1])
PY
  done
  echo "  reloading launchd jobs..."
  for p in "${SLACK_PLISTS[@]}"; do
    f="$LA/$p"
    [[ -f "$f" ]] || continue
    launchctl unload "$f" 2>/dev/null
    launchctl load   "$f" 2>/dev/null && echo "    reloaded $p"
  done
fi

echo

# ---- 2. Renaissance Capital API key ----
printf "Paste NEW Renaissance Capital API key, then Enter: "
read -rs REN_KEY; echo
if [[ -z "${REN_KEY}" ]]; then
  echo "  (empty — skipping Renaissance)"
else
  cp "$ZSHRC" "$ZSHRC.bak_kairos_$TS"
  if grep -q '^export RENAISSANCE_CAPITAL_API_KEY=' "$ZSHRC"; then
    /usr/bin/python3 - "$ZSHRC" "$REN_KEY" <<'PY'
import sys,re
f,key=sys.argv[1],sys.argv[2]
s=open(f).read()
new=re.sub(r'^export RENAISSANCE_CAPITAL_API_KEY=.*$',
           'export RENAISSANCE_CAPITAL_API_KEY="'+key+'"', s, count=1, flags=re.M)
open(f,'w').write(new); print("  updated ~/.zshrc")
PY
  else
    printf 'export RENAISSANCE_CAPITAL_API_KEY="%s"\n' "$REN_KEY" >> "$ZSHRC"
    echo "  appended to ~/.zshrc"
  fi
fi

unset SLACK_TOK REN_KEY
echo
echo "=== done. backups saved with suffix .bak_$TS ==="
echo "Open a fresh shell (exec zsh) for the new Renaissance key to load."
