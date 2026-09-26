"""Behavioural tests for .github/scripts/intent-lock.sh (the Intent Lock check).

The script runs for real with ``gh`` replaced by a stub that serves canned API
answers and records every write, so each property below is observed rather
than read off the source:

* the hash covers exactly the frozen sections (Goal line, Why it matters, Not a
  goal) and ignores CRLF, trailing-space and blank-line churn;
* a PR that opens gets one bot baseline; a later change to the frozen text goes
  red and names the ``/intent approve <head>`` command;
* only a github-actions[bot] comment counts as a baseline;
* ``/intent approve`` is honoured only from a writer and only for the head;
* a PR with no baseline is skipped green.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "intent-lock.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None or shutil.which("jq") is None,
    reason="requires a POSIX bash and jq",
)

HEAD = "a686d96a83859a73eb93b322de04b21bdea5f093"
BOT = "github-actions[bot]"

BODY = """\
## Problem / Motivation

**Goal:** the thing works again.

Something is broken.

## Why it matters

Users hit it daily.

## Not a goal

- Rewriting the thing.

## What changed (motivation -> approach -> change)

A small fix.
"""

# The stub applies --jq itself (gh does), logs every write to $FIXTURES/calls,
# and keeps each write's --input body as $FIXTURES/input.<n>.
GH_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "workflow" ]; then
  echo "DISPATCH $*" >> "$FIXTURES/calls"
  [ -f "$FIXTURES/dispatch_fail" ] && exit 1
  exit 0
fi
shift
method=GET; url=""; jqf=""; fields=""
while [ $# -gt 0 ]; do
  case "$1" in
    --method) method="$2"; shift ;;
    --jq) jqf="$2"; shift ;;
    -f) fields="$fields $2"; shift ;;
    --input)
      n=$(ls "$FIXTURES" | grep -c '^input\.' || true)
      cat > "$FIXTURES/input.$n"; shift ;;
    repos/*) url="$1" ;;
  esac
  shift
done
if [ "$method" != GET ]; then
  echo "$method $url$fields" >> "$FIXTURES/calls"
  if [ -n "$jqf" ]; then echo 99; fi
  exit 0
fi
case "$url" in
  */pulls/*) file=pr.json ;;
  */issues/*/comments) file=comments.json ;;
  */check-runs) file=check_runs.json ;;
  */permission)
    if [ -f "$FIXTURES/perm_fail" ]; then echo "gh: Server Error (HTTP 502)" >&2; exit 1; fi
    file=permission.json ;;
  *) echo "gh stub: unhandled $url" >&2; exit 90 ;;
esac
[ -f "$FIXTURES/$file" ] || { echo "gh: Not Found (HTTP 404)" >&2; exit 1; }
if [ -n "$jqf" ]; then jq -r "$jqf" "$FIXTURES/$file"; else cat "$FIXTURES/$file"; fi
"""


def frozen_hash(body: str) -> str:
    out = subprocess.run(
        ["bash", str(SCRIPT), "--hash"],
        env={**os.environ, "PR_BODY": body},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return out.stdout.strip()


def baseline_comment(body: str, *, author: str = BOT, cid: int = 501) -> dict:
    return {
        "id": cid,
        "user": {"login": author},
        "body": f"<!-- intent-lock baseline={frozen_hash(body)} -->\nIntent Lock baseline",
    }


class Repo:
    def __init__(self, root: Path) -> None:
        self.fixtures = root / "fixtures"
        bindir = root / "bin"
        self.fixtures.mkdir()
        bindir.mkdir()
        stub = bindir / "gh"
        stub.write_text(GH_STUB)
        stub.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "FIXTURES": str(self.fixtures),
            "GH_TOKEN": "x",
            "REPO": "kirodotdev/KiroCrew",
            "PR": "7",
        }
        self.set(body=BODY, comments=[], check_runs=[])

    def set(self, *, body=None, comments=None, check_runs=None, permission=None) -> None:
        if body is not None:
            self.body = body
            (self.fixtures / "pr.json").write_text(
                json.dumps({"head": {"sha": HEAD}, "body": body})
            )
        if comments is not None:
            (self.fixtures / "comments.json").write_text(json.dumps(comments))
        if check_runs is not None:
            (self.fixtures / "check_runs.json").write_text(json.dumps({"check_runs": check_runs}))
        if permission is not None:
            (self.fixtures / "permission.json").write_text(json.dumps({"permission": permission}))

    def run(
        self,
        event: str = "pull_request_target",
        action: str = "edited",
        expect_rc: int = 0,
        **extra: str,
    ):
        env = {**self.env, "EVENT": event, "ACTION": action, "EVENT_BODY": self.body, **extra}
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=self.fixtures,
        )
        assert proc.returncode == expect_rc, proc.stderr
        calls_file = self.fixtures / "calls"
        return calls_file.read_text().splitlines() if calls_file.exists() else []

    def inputs(self) -> list[str]:
        return [
            json.loads((self.fixtures / f"input.{n}").read_text())["body"]
            for n in range(len(list(self.fixtures.glob("input.*"))))
        ]

    def approve(self, sha: str, user: str = "maint", **extra: str):
        return self.run(
            event="issue_comment",
            action="created",
            COMMENT_BODY=f"/intent approve {sha}",
            COMMENT_USER=user,
            **extra,
        )


@pytest.fixture()
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path)


def verdict(calls: list[str]) -> str:
    """The published check-run conclusion (the one completing write)."""
    [row] = [c for c in calls if "/check-runs" in c and "status=completed" in c]
    return row.split("conclusion=")[1].split()[0]


class TestFrozenHash:
    def test_formatting_churn_is_not_a_goal_change(self) -> None:
        churned = BODY.replace("\n", "  \r\n").replace(
            "Users hit it daily.", "\nUsers hit it daily.\n"
        )
        assert frozen_hash(churned) == frozen_hash(BODY)

    @pytest.mark.parametrize(
        "old,new",
        [
            ("the thing works again.", "the thing is rewritten."),
            ("Users hit it daily.", "Nobody hits it."),
            ("- Rewriting the thing.", "- Nothing."),
        ],
    )
    def test_each_frozen_section_is_covered(self, old: str, new: str) -> None:
        assert frozen_hash(BODY.replace(old, new)) != frozen_hash(BODY)

    def test_sections_outside_the_goal_are_not_covered(self) -> None:
        assert frozen_hash(BODY.replace("A small fix.", "A large fix.")) == frozen_hash(BODY)
        assert frozen_hash(BODY.replace("Something is broken.", "Other.")) == frozen_hash(BODY)

    def test_fenced_text_inside_a_frozen_section_is_covered(self) -> None:
        def body(word: str) -> str:
            return BODY.replace("Users hit it daily.", f"Users hit it daily.\n\n```\n{word}\n```")

        assert frozen_hash(body("one")) != frozen_hash(body("two"))

    def test_a_template_comment_is_not_goal_text(self) -> None:
        commented = BODY.replace(
            "Users hit it daily.",
            "<!-- Frozen. Impact if\n     left undone. -->\n\nUsers hit it daily.",
        ).replace("- Rewriting the thing.", "<!-- one line -->\n- Rewriting the thing.")
        assert frozen_hash(commented) == frozen_hash(BODY)

    def test_text_beside_a_comment_is_still_goal_text(self) -> None:
        def body(word: str) -> str:
            return BODY.replace("Users hit it daily.", f"<!-- note --> {word} <!-- a\nb --> tail")

        assert frozen_hash(body("Users hit it")) != frozen_hash(body("Nobody"))
        assert frozen_hash(body("Users hit it")) != frozen_hash(
            BODY.replace("Users hit it daily.", "")
        )

    def test_a_fenced_heading_does_not_open_a_frozen_section(self) -> None:
        fenced = BODY + "\n```\n## Not a goal\n- smuggled\n```\n"
        assert frozen_hash(fenced) == frozen_hash(BODY)


class TestBaselineAndDrift:
    def test_opening_posts_one_bot_baseline_and_goes_green(self, repo: Repo) -> None:
        calls = repo.run(action="opened")
        assert [c for c in calls if c.startswith("POST") and "/comments" in c]
        [posted] = repo.inputs()
        assert posted.startswith(f"<!-- intent-lock baseline={frozen_hash(BODY)} -->")
        assert verdict(calls) == "success"
        assert not [c for c in calls if c.startswith("DISPATCH")]

    def test_an_edit_before_the_opened_run_reads_is_red_not_the_baseline(self, repo: Repo) -> None:
        repo.set(body=BODY.replace("works again", "is rewritten"))
        calls = repo.run(action="opened", EVENT_BODY=BODY)
        [posted] = repo.inputs()
        assert posted.startswith(f"<!-- intent-lock baseline={frozen_hash(BODY)} -->")
        assert verdict(calls) == "failure"

    def test_unchanged_goal_is_green(self, repo: Repo) -> None:
        repo.set(comments=[baseline_comment(BODY)])
        assert verdict(repo.run()) == "success"

    def test_changed_goal_is_red_names_the_command_and_recomputes_readiness(
        self, repo: Repo
    ) -> None:
        repo.set(
            comments=[baseline_comment(BODY)], body=BODY.replace("works again", "is rewritten")
        )
        calls = repo.run(action="synchronize")
        assert verdict(calls) == "failure"
        [row] = [c for c in calls if "status=completed" in c]
        assert f"/intent approve {HEAD}" in row
        assert (
            f"DISPATCH workflow run pr-readiness.yml --repo kirodotdev/KiroCrew -f pr=7 -f sha={HEAD}"
            in calls
        )

    def test_only_a_bot_comment_counts_as_the_baseline(self, repo: Repo) -> None:
        changed = BODY.replace("works again", "is rewritten")
        # A forged marker matching the NEW text, posted before the real one.
        repo.set(
            body=changed,
            comments=[baseline_comment(changed, author="mallory", cid=400), baseline_comment(BODY)],
        )
        assert verdict(repo.run()) == "failure"

    def test_a_pr_without_a_baseline_is_skipped_green(self, repo: Repo) -> None:
        calls = repo.run(action="edited")
        assert verdict(calls) == "success"
        assert "Skipped" in [c for c in calls if "status=completed" in c][0]
        assert repo.inputs() == []

    def test_an_existing_row_is_updated_not_duplicated(self, repo: Repo) -> None:
        repo.set(
            comments=[baseline_comment(BODY)],
            check_runs=[
                {
                    "name": "Intent Lock",
                    "external_id": "intent-lock:8",
                    "id": 1,
                    "conclusion": "success",
                },
                {
                    "name": "Intent Lock",
                    "external_id": "intent-lock:7",
                    "id": 2,
                    "conclusion": "success",
                },
            ],
        )
        calls = repo.run()
        assert [c.split()[:2] for c in calls if "/check-runs" in c] == [
            ["PATCH", "repos/kirodotdev/KiroCrew/check-runs/2"]
        ] * 2

    def test_the_row_is_in_progress_before_the_verdict(self, repo: Repo) -> None:
        # A concurrent PR Readiness run must never read the old green while
        # this run evaluates, so the row goes in_progress first.
        repo.set(comments=[baseline_comment(BODY)], body=BODY.replace("works again", "is new"))
        calls = repo.run()
        rows = [c for c in calls if "/check-runs" in c]
        assert rows[0] == "POST repos/kirodotdev/KiroCrew/check-runs name=Intent Lock head_sha=" + (
            HEAD + " external_id=intent-lock:7 status=in_progress"
        )
        assert rows[1].startswith("PATCH repos/kirodotdev/KiroCrew/check-runs/99 status=completed")

    def test_a_failed_dispatch_fails_the_job(self, repo: Repo) -> None:
        repo.set(comments=[baseline_comment(BODY)], body=BODY.replace("works again", "is new"))
        (repo.fixtures / "dispatch_fail").write_text("")
        calls = repo.run(action="synchronize", expect_rc=1)
        assert verdict(calls) == "failure"


class TestApprove:
    def _red(self, repo: Repo) -> None:
        repo.set(
            body=BODY.replace("works again", "is rewritten"),
            comments=[baseline_comment(BODY)],
            check_runs=[
                {
                    "name": "Intent Lock",
                    "external_id": "intent-lock:7",
                    "id": 2,
                    "conclusion": "failure",
                }
            ],
        )

    def test_a_writer_approving_the_head_moves_the_baseline(self, repo: Repo) -> None:
        self._red(repo)
        repo.set(permission="write")
        calls = repo.approve(HEAD[:12])
        assert "PATCH repos/kirodotdev/KiroCrew/issues/comments/501" in calls
        [patched] = repo.inputs()
        new_hash = frozen_hash(BODY.replace("works again", "is rewritten"))
        assert patched.startswith(f"<!-- intent-lock baseline={new_hash} -->")
        assert verdict(calls) == "success"
        assert any(c.startswith("DISPATCH") for c in calls)

    @pytest.mark.parametrize("permission", ["read", "triage", None])
    def test_a_non_writer_is_refused(self, repo: Repo, permission: str | None) -> None:
        self._red(repo)
        if permission:
            repo.set(permission=permission)
        calls = repo.approve(HEAD)
        assert not [c for c in calls if "/issues/comments/" in c]  # baseline kept
        assert verdict(calls) == "failure"
        assert "Only a repository writer" in repo.inputs()[0]

    def test_a_goal_edited_after_the_approval_comment_is_refused(self, repo: Repo) -> None:
        self._red(repo)
        repo.set(permission="write")
        calls = repo.approve(HEAD, EVENT_BODY=BODY.replace("works again", "is approved text"))
        assert not [c for c in calls if "/issues/comments/" in c]  # baseline kept
        assert verdict(calls) == "failure"
        assert "changed after the comment" in repo.inputs()[0]

    def test_the_job_is_not_named_like_the_verdict_row(self) -> None:
        # pr_status.py reads a red "Intent Lock" check-run as a maintainer
        # wait; a crashed job must stay an ordinary failure.
        spec = yaml.safe_load((ROOT / ".github" / "workflows" / "intent-lock.yml").read_text())
        assert spec["jobs"]["lock"]["name"] != "Intent Lock"

    def test_an_unreadable_permission_is_not_called_a_non_writer(self, repo: Repo) -> None:
        self._red(repo)
        (repo.fixtures / "permission.json").unlink(missing_ok=True)
        (repo.fixtures / "perm_fail").write_text("")
        calls = repo.approve(HEAD)
        assert not [c for c in calls if "/issues/comments/" in c]  # baseline kept
        assert verdict(calls) == "failure"
        assert "did not answer" in repo.inputs()[0]

    def test_a_stale_sha_is_refused(self, repo: Repo) -> None:
        self._red(repo)
        repo.set(permission="admin")
        calls = repo.approve("b" * 40)
        assert not [c for c in calls if "/issues/comments/" in c]  # baseline kept
        assert verdict(calls) == "failure"
        assert HEAD in repo.inputs()[0]

    def test_a_malformed_command_only_re_evaluates(self, repo: Repo) -> None:
        self._red(repo)
        calls = repo.run(
            event="issue_comment",
            action="created",
            COMMENT_BODY="/intent approve please",
            COMMENT_USER="x",
        )
        assert repo.inputs() == []
        assert verdict(calls) == "failure"

    def test_a_refused_comment_that_replaced_an_edit_run_still_goes_red(self, repo: Repo) -> None:
        # The comment run shares the PR's concurrency group, so it can evict a
        # pending edit evaluation. It must publish that edit's red itself, not
        # leave the old green standing.
        repo.set(
            body=BODY.replace("works again", "is rewritten"),
            comments=[baseline_comment(BODY)],
            check_runs=[
                {
                    "name": "Intent Lock",
                    "external_id": "intent-lock:7",
                    "id": 2,
                    "conclusion": "success",
                }
            ],
            permission="read",
        )
        calls = repo.approve(HEAD)
        assert calls[0] == "PATCH repos/kirodotdev/KiroCrew/check-runs/2 status=in_progress"
        assert verdict(calls) == "failure"
