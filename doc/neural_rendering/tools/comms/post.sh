#!/usr/bin/env bash
# Post a message to the coordination PR so the cloud session wakes up.
#
#   post.sh result  "title" [body-file]     # a build/render/training outcome
#   post.sh question "title" [body-file]    # a design or codebase question
#   post.sh review   "title" [body-file]    # ask for a review of files or a commit range
#   post.sh blocked  "title" [body-file]    # something the cloud session can unblock
#
# Body is read from the file if given, else from stdin. Commit and push BEFORE
# posting so the cloud session can read the state you are describing.
set -euo pipefail
REPO="${NEURAL_PR_REPO:-matiasvivancob/blender}"
PR="${NEURAL_PR_NUMBER:-1}"
kind="${1:?kind: result|question|review|blocked}"
title="${2:?title}"
body="$(cat "${3:-/dev/stdin}")"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
printf '[%s] %s\n\nbranch: `%s` head: `%s`\n\n%s\n' "$kind" "$title" "$branch" "$sha" "$body" \
  | gh pr comment "$PR" --repo "$REPO" --body-file -
echo "posted [$kind] $title to $REPO#$PR"
