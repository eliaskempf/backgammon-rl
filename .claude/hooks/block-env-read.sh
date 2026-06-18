#!/usr/bin/env bash
# PreToolUse(Bash) guard: refuse any shell command that would read the repo's
# .env, so the OpenRouter API key stays hidden from Claude.
#
# This only inspects the bash command STRING. python-dotenv's load_dotenv()
# opens .env from inside the Python process (not a bash command), so live
# scripts still get the key at runtime — Claude just can't `cat`/`grep`/etc it.
#
# Exit 2 = block the tool call (stderr is shown to Claude as the reason).
set -euo pipefail

# Read the full hook payload (JSON on stdin). We deliberately avoid a jq
# dependency (jq isn't installed here) and scan the raw payload: over-matching
# only ever errs toward blocking, which is the safe direction.
payload="$(cat)"

# Match .env as a standalone path token: preceded by start/non-word AND followed by
# end/non-word. The leading boundary ignores module paths like "bgrl.env" (there .env
# follows a letter) while still catching "cat .env", "./.env", and ".env.local".
if printf '%s' "$payload" | grep -qE '(^|[^[:alnum:]_])\.env($|[^[:alnum:]_])'; then
  echo "Blocked: reading .env is not permitted — the OpenRouter API key must stay hidden from Claude." >&2
  exit 2
fi
exit 0
