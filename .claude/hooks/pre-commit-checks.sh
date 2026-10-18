#!/usr/bin/env bash
# PreToolUse(Bash) hook: runs only for real `git commit` commands.
#   1. Build + tests must pass (deny the commit otherwise).
#   2. If code/infra changes are staged without README.md, ask the user to
#      confirm instead of silently allowing or hard-blocking.
# Every other Bash command exits immediately with no output (allowed).
set -euo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null || true)"

# Match `git commit` / `git -C <dir> commit` at the start of the command or after ; && || |
if ! printf '%s' "$cmd" | grep -Eq '(^|[;&|][[:space:]]*)git([[:space:]]+-C[[:space:]]+[^[:space:]]+)?[[:space:]]+commit([[:space:]]|$)'; then
  exit 0
fi

repo="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

decide() { # $1 = allow|ask|deny, $2 = reason
  python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":sys.argv[1],"permissionDecisionReason":sys.argv[2]}}))' "$1" "$2"
  exit 0
}

# 1. Build + tests (the Go module lives in platform/)
if ! out="$(cd "$repo/platform" && go build ./... 2>&1 && go test -race -count=1 ./internal/... ./test/... 2>&1)"; then
  decide deny "Build or tests failed. Fix before committing.
$(printf '%s' "$out" | grep -E 'FAIL|error|cannot|undefined' | head -15)"
fi

# 2. README reminder for code/infra changes
staged="$(git -C "$repo" diff --cached --name-only)"
if ! printf '%s\n' "$staged" | grep -qx 'README.md'; then
  significant="$(printf '%s\n' "$staged" | grep -E '\.(go|proto|py|ts|tsx)$|/deploy/|Dockerfile|docker-compose|\.tf$' | grep -Ev '_test\.go$|/test_[^/]*\.py$|\.test\.ts$' | head -10 || true)"
  if [ -n "$significant" ]; then
    decide ask "Code/infra changes are staged without README.md. Check whether README needs updating:
$significant"
  fi
fi

exit 0
