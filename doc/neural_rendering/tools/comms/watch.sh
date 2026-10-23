#!/usr/bin/env bash
# Poll the coordination PR for new comments and new commits on the branch.
# Prints anything new since the last run and pulls the branch (fast-forward
# only). Run it in a loop or from cron; local agents read its output.
#
#   watch.sh            # one pass
#   watch.sh --loop 300 # every 5 minutes
set -euo pipefail
REPO="${NEURAL_PR_REPO:-matiasvivancob/blender}"
PR="${NEURAL_PR_NUMBER:-1}"
STATE_DIR="${NEURAL_WATCH_STATE:-$HOME/.cache/neural_pr_watch}"
mkdir -p "$STATE_DIR"
seen="$STATE_DIR/seen_comment_ids"
touch "$seen"

pass() {
  gh api "repos/$REPO/issues/$PR/comments?per_page=100" \
    --jq '.[] | "\(.id)\t\(.user.login)\t\(.created_at)\t\(.body)"' \
  | while IFS=$'\t' read -r id user created body; do
      if ! grep -qx "$id" "$seen"; then
        printf '\n=== PR comment %s by %s at %s ===\n%b\n' "$id" "$user" "$created" "$body"
        echo "$id" >> "$seen"
      fi
    done
  git fetch -q origin
  local behind
  behind="$(git rev-list --count HEAD..origin/"$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null || echo 0)"
  if [ "$behind" != "0" ]; then
    echo; echo "=== $behind new commit(s) on origin, pulling (ff-only) ==="
    git log --oneline HEAD..origin/"$(git rev-parse --abbrev-ref HEAD)"
    git pull -q --ff-only || echo "ff-only pull failed: local commits diverged, merge by hand"
  fi
}

if [ "${1:-}" = "--loop" ]; then
  while true; do pass; sleep "${2:-300}"; done
else
  pass
fi
