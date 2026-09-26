#!/usr/bin/env bash
# Intent Lock: keeps a PR's frozen goal sections from drifting silently.
#
# The frozen sections are the `**Goal:**` line under `## Problem / Motivation`,
# `## Why it matters` and `## Not a goal` (see .github/PULL_REQUEST_TEMPLATE.md).
# When a PR opens, github-actions[bot] posts one baseline comment holding a hash
# of them. On every later edit or push the hash is recomputed; a difference
# publishes a red `Intent Lock` check until a repository writer comments
# `/intent approve <head-sha>`, which moves the baseline to the current text.
#
# One script for same-repo and fork PRs alike: intent-lock.yml runs it on
# `pull_request_target` and `issue_comment`, both of which execute the BASE
# branch's copy. It reads the PR body through the API as data and never checks
# out or runs PR code.
#
# Inputs (environment): GH_TOKEN, REPO, PR, EVENT (pull_request_target |
# issue_comment), ACTION, EVENT_BODY (the PR body in the event payload), and
# for comments COMMENT_BODY + COMMENT_USER.
# `intent-lock.sh --hash` reads PR_BODY and prints only the hash.
#
# Trust: only a comment authored by github-actions[bot] counts as a baseline,
# so nobody else can forge one. A PR with no baseline (opened before this
# check existed) is skipped with a green note.
set -euo pipefail

BOT='github-actions[bot]'
MARKER_PREFIX='<!-- intent-lock baseline='
CHECK_NAME='Intent Lock'

# The frozen text, one tagged line per non-blank line, trailing whitespace
# stripped, so blank-line and CRLF churn is not a goal change. An HTML comment
# (the template's own guidance) is not goal text, so deleting one is no change.
# Fenced code is
# skipped for heading detection using the same CommonMark fence rule as
# pr-description-check.sh (same character, length >= opener).
frozen_text() {
  tr -d '\r' | awk '
    {
      line = $0
      # Outside a fence, cut every <!-- ... --> span (it may span lines) and
      # keep the visible text on either side of it.
      if (open == "") {
        out = ""
        while (line != "") {
          if (incomment) {
            k = index(line, "-->")
            if (!k) { line = ""; break }
            line = substr(line, k + 3); incomment = 0
          } else {
            k = index(line, "<!--")
            if (!k) { out = out line; line = ""; break }
            out = out substr(line, 1, k - 1); line = substr(line, k + 4); incomment = 1
          }
        }
        line = out
      }
      sub(/[ \t]+$/, "", line)
      s = line
      n = 0
      while (n < 3 && substr(s, 1, 1) == " ") { s = substr(s, 2); n++ }

      mch = ""
      if (substr(s, 1, 3) == "```") mch = "`"
      else if (substr(s, 1, 3) == "~~~") mch = "~"
      mlen = 0
      if (mch != "") {
        while (substr(s, mlen + 1, 1) == mch) mlen++
        rest = substr(s, mlen + 1)
      }
      if (open == "") {
        if (mch != "") { open = mch; olen = mlen }
      } else {
        if (mch == open && mlen >= olen && rest ~ /^[ \t]*$/) { open = ""; olen = 0 }
        if (sec == "why" || sec == "not") print sec "\t" line
        next
      }

      hn = 0
      while (hn < 7 && substr(s, hn + 1, 1) == "#") hn++
      c = substr(s, hn + 1, 1)
      if (hn >= 1 && hn <= 2 && (c == " " || c == "\t")) {
        h = tolower(s)
        sec = ""
        if (index(h, "## problem / motivation") == 1) sec = "problem"
        else if (index(h, "## why it matters") == 1) sec = "why"
        else if (index(h, "## not a goal") == 1) sec = "not"
        next
      }

      if (sec == "problem" && index(tolower(s), "**goal:**") == 1) {
        goal = substr(s, 10)
        sub(/^[ \t]+/, "", goal)
        print "goal\t" goal
      } else if ((sec == "why" || sec == "not") && line ~ /[^ \t]/) {
        print sec "\t" line
      }
    }
  '
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | cut -c1-64
  else
    shasum -a 256 | cut -c1-64
  fi
}

frozen_hash() { printf '%s' "$1" | frozen_text | sha256; }

if [ "${1:-}" = "--hash" ]; then
  frozen_hash "${PR_BODY:-}"
  exit 0
fi

case "$PR" in
  ''|*[!0-9]*) echo "::error::unexpected PR number '$PR'"; exit 1 ;;
esac

post_comment() {
  jq -n --arg body "$1" '{body: $body}' \
    | gh api --method POST "repos/$REPO/issues/$PR/comments" --input - >/dev/null
}

baseline_body() {
  printf '%s%s -->\n%s' "$MARKER_PREFIX" "$1" \
    "Intent Lock baseline for the frozen goal sections (Goal line, Why it matters, Not a goal). Changing them turns the \`$CHECK_NAME\` check red until a maintainer comments \`/intent approve <head-sha>\`."
}

pr_json="$(gh api "repos/$REPO/pulls/$PR")"
head="$(jq -r '.head.sha' <<<"$pr_json")"
body="$(jq -r '.body // ""' <<<"$pr_json")"
current="$(frozen_hash "$body")"
# The body the event carried. A baseline or an approval records THAT text, so
# an edit landing between the event and this run's API read is compared
# against it (red) instead of being recorded as approved.
seen="$(frozen_hash "${EVENT_BODY:-}")"

# The baseline: the FIRST marker comment by the bot. A failed read exits here,
# because reading it as "no baseline" would post a second one.
baseline_row="$(gh api --paginate "repos/$REPO/issues/$PR/comments" \
  --jq ".[] | select(.user.login == \"$BOT\") | select(.body | startswith(\"$MARKER_PREFIX\"))
        | \"\(.id) \((.body | capture(\"^<!-- intent-lock baseline=(?<h>[0-9a-f]{64}) -->\") | .h) // \"\")\"")"
baseline_row="${baseline_row%%$'\n'*}"
baseline_id="${baseline_row%% *}"
baseline="${baseline_row#* }"

# A PR that just opened records the body it opened with.
if [ "$EVENT" = "pull_request_target" ] && [ "$ACTION" = "opened" ] && [ -z "$baseline_id" ]; then
  post_comment "$(baseline_body "$seen")"
  baseline="$seen"
fi

# The one row for this PR on this head, keyed on external_id (the idiom
# fork-pr-description.yml documents). A failed lookup fails closed rather than
# POSTing a duplicate.
external_id="intent-lock:$PR"
if ! lookup="$(gh api --paginate "repos/$REPO/commits/$head/check-runs" \
  --jq ".check_runs[] | select(.name == \"$CHECK_NAME\") | \"\(.external_id)\t\(.id)\t\(.conclusion)\"")"; then
  echo "::error::could not list check-runs for $head; refusing to POST"
  exit 1
fi
existing=""
previous=""
while IFS=$'\t' read -r row_eid row_id row_conclusion; do
  [ "$row_eid" = "$external_id" ] || continue
  existing="$row_id"
  previous="$row_conclusion"
  break
done <<<"$lookup"

# Every run marks the row in_progress BEFORE evaluating, so a concurrent PR
# Readiness run reads "waiting", never the old green. A run that dies after
# this leaves the row in_progress, which also reads as waiting.
if [ -n "$existing" ]; then
  gh api --method PATCH "repos/$REPO/check-runs/$existing" -f status=in_progress >/dev/null
else
  existing="$(gh api --method POST "repos/$REPO/check-runs" \
    -f name="$CHECK_NAME" -f head_sha="$head" -f external_id="$external_id" \
    -f status=in_progress --jq '.id')"
fi

# An approval comment that is not accepted still re-evaluates and publishes
# below: its run shares the PR-event concurrency group, so it may have replaced
# a pending edit evaluation, and must not leave that edit's verdict unwritten.
approved=false
approve() {
  first_line="${COMMENT_BODY%%$'\n'*}"
  first_line="${first_line%$'\r'}"
  if [[ ! "$first_line" =~ ^/intent\ approve\ ([0-9a-fA-F]{7,40})[[:space:]]*$ ]]; then
    return
  fi
  requested="${BASH_REMATCH[1],,}"
  # A failed read is not "no": 404 means not a collaborator, anything else is
  # an unreadable answer the writer can retry. Both deny.
  perm_err="$(mktemp)"
  if ! permission="$(gh api "repos/$REPO/collaborators/$COMMENT_USER/permission" --jq '.permission' 2>"$perm_err")"; then
    if ! grep -qiE 'HTTP 404|Not Found' "$perm_err"; then
      post_comment "Intent Lock: \`/intent approve\` not recorded. GitHub did not answer the permission check; comment it again."
      return
    fi
    permission=""
  fi
  case "$permission" in
    admin|maintain|write) ;;
    *)
      post_comment "Intent Lock: \`/intent approve\` not accepted. Only a repository writer can approve a goal change."
      return
      ;;
  esac
  if [[ "$head" != "$requested"* ]]; then
    post_comment "Intent Lock: \`/intent approve\` not accepted. \`$requested\` is not the current head; re-run it with \`$head\`."
    return
  fi
  if [ "$seen" != "$current" ]; then
    post_comment "Intent Lock: \`/intent approve\` not accepted. The goal sections changed after the comment was posted; review them and approve again."
    return
  fi
  if [ -n "$baseline_id" ]; then
    jq -n --arg body "$(baseline_body "$current")" '{body: $body}' \
      | gh api --method PATCH "repos/$REPO/issues/comments/$baseline_id" --input - >/dev/null
  else
    post_comment "$(baseline_body "$current")"
  fi
  approved=true
}
if [ "$EVENT" = "issue_comment" ]; then
  approve
fi

if [ "$approved" = true ]; then
  conclusion=success
  title="Goal change approved by @$COMMENT_USER"
  summary="A maintainer approved the frozen goal sections at \`$head\`. The baseline now matches them."
elif [ -z "$baseline_id" ] && [ "$ACTION" != "opened" ]; then
  conclusion=success
  title="Skipped: no baseline"
  summary="This PR has no Intent Lock baseline (it predates the check), so the goal sections are not compared."
elif [ "$baseline" = "$current" ]; then
  conclusion=success
  title="Goal sections unchanged"
  summary="The Goal line, Why it matters and Not a goal match the baseline."
else
  conclusion=failure
  title="The PR goal changed"
  summary="The frozen goal sections (Goal line, Why it matters, Not a goal) differ from the baseline recorded when this PR opened. A maintainer must comment \`/intent approve $head\` to accept the new goal, or restore the original text. Agents never post this command."
fi

if [ -n "$existing" ]; then
  gh api --method PATCH "repos/$REPO/check-runs/$existing" \
    -f status=completed -f conclusion="$conclusion" \
    -f "output[title]=$title" -f "output[summary]=$summary" >/dev/null
else
  gh api --method POST "repos/$REPO/check-runs" \
    -f name="$CHECK_NAME" -f head_sha="$head" -f external_id="$external_id" \
    -f status=completed -f conclusion="$conclusion" \
    -f "output[title]=$title" -f "output[summary]=$summary" >/dev/null
fi

# PR Readiness reads this row. Its own pull_request_target run can finish
# before this one publishes, and a comment starts no readiness run at all, so
# recompute it whenever this verdict turned red or left red.
if [ "$previous" != "$conclusion" ] && { [ "$conclusion" = failure ] || [ "$previous" = failure ]; }; then
  if ! gh workflow run pr-readiness.yml --repo "$REPO" -f pr="$PR" -f sha="$head" >/dev/null; then
    echo "::error::could not dispatch pr-readiness.yml; re-run this job"
    exit 1
  fi
fi
