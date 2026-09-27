"""Ratchet: how many places can still end a runtime without saying who fired.

Every path that signals a process reaches one of a handful of primitives in
:mod:`kiro_crew.platform_compat`, and a caller that reaches them directly does so
unattributed: the log that follows says a process died, never who decided it
should. Routing those callers through one attributed place
(:func:`kiro_crew.runtime_ownership.note_runtime_kill`) is the work of several
changes, so this module measures the remainder and ratchets it:
:data:`BYPASS_BASELINE` may go DOWN in any change and may never go up. A new kill
site must either attribute itself or lower something else to pay for itself.

Three things are pinned, and the third is what makes the first two mean anything:

* the COUNT of direct primitive calls outside the primitive module
  (:func:`test_kill_primitive_bypass_count_does_not_grow`);
* that the count is not low merely because the scan matches nothing -- a real
  primitive call is found where one is known to be
  (:func:`test_the_needles_match_a_known_call_site`);
* that the two paths every other kill funnels into DO attribute themselves
  (:func:`test_attributed_kill_paths_note_the_kill`). Without this the count
  could fall to zero while nothing was ever logged.

Complements ``test_process_identity_structural.py`` rather than repeating it.
That module answers a different question -- whether the reaper's four modules
address a process by a verified handle instead of a bare pid, which is about
killing a STRANGER that inherited a recycled pid -- and it forbids outright inside
those modules. This one counts, repo-wide.

The scan is :mod:`ast`, so only real call expressions count: a primitive named in
a docstring or a comment is not a kill site, and a grep-based version of this test
reported 108 where 66 calls exist. The needles are assembled from fragments so
this file does not contain the names it searches for, and so a reader grepping the
tree for kill sites is not handed the detector as one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "kiro_crew"

#: The module that DEFINES the primitives. Its own calls are the primitives'
#: implementation -- ``kill_pid_pinned`` delegating to ``kill_pid``, which issues
#: the group signal -- not a caller deciding a process should die.
PRIMITIVE_HOME = "platform_compat.py"

#: Assembled, not written: this file must not contain the names it scans for.
_KILL = "kill"
PID_KILL_PRIMITIVES = frozenset(
    {
        _KILL + "_pid",
        _KILL + "_process_tree",
        _KILL + "_pid_pinned",
        _KILL + "_process_tree_pinned",
        _KILL + "pg",
    }
)

#: Direct primitive calls outside :data:`PRIMITIVE_HOME`, measured on the change
#: that introduced the attribution line. RATCHET: lower this when a change routes
#: a site through an attributed path; never raise it.
#:
#: What the remainder is, so the next change knows where to look: the session
#: teardown's own escalation (``session_pid``), app-backend and dev-preview
#: process management, the test harness, the cron script runner, and a spread of
#: single-site tools. The two biggest runtime kill paths already attribute
#: themselves -- see :func:`test_attributed_kill_paths_note_the_kill` -- and their
#: primitive calls remain counted here, because attribution brackets their
#: escalation rather than replacing it.
BYPASS_BASELINE = 66

#: Functions that must say who fired before they signal anything, as module path
#: -> function name. These are the paths every other kill funnels into: the
#: synchronous provider teardown (a dashboard reset-all, a pool teardown, a
#: failed start's cleanup) and the runtime's own kill.
ATTRIBUTED_KILL_PATHS = (
    ("session_pid.py", "_sync_kill_provider"),
    ("acp/runtime.py", "kill"),
)

ATTRIBUTION = "note_runtime_kill"

#: A call site the needles MUST find, so an empty scan cannot read as success.
#: ``kill_process_tree`` is what a POSIX group teardown ends in.
KNOWN_CALL_SITE = ("session_pid.py", _KILL + "_process_tree")


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _primitive_calls(tree: ast.AST) -> list[tuple[int, str]]:
    return [
        (node.lineno, name)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _callee_name(node)) in PID_KILL_PRIMITIVES
    ]


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _bypass_sites() -> dict[str, list[tuple[int, str]]]:
    sites: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(SRC.rglob("*.py")):
        if path.name == PRIMITIVE_HOME:
            continue
        tree = _parse(path)
        if tree is None:
            continue
        found = _primitive_calls(tree)
        if found:
            sites[str(path.relative_to(SRC))] = found
    return sites


def _function_named(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_kill_primitive_bypass_count_does_not_grow() -> None:
    """The number of sites that kill without saying who fired may only fall."""
    sites = _bypass_sites()
    total = sum(len(v) for v in sites.values())
    breakdown = "\n".join(
        f"  {count:3d}  {module}"
        for module, count in sorted(
            ((m, len(v)) for m, v in sites.items()), key=lambda kv: (-kv[1], kv[0])
        )
    )
    assert total <= BYPASS_BASELINE, (
        f"{total} direct kill-primitive call(s) now bypass the attribution point, above "
        f"the baseline of {BYPASS_BASELINE}. Route the new site through a path that calls "
        f"kiro_crew.runtime_ownership.{ATTRIBUTION} instead of raising this number.\n"
        f"Per module:\n{breakdown}"
    )


def test_bypass_baseline_is_not_stale() -> None:
    """A baseline left far above the real count stops ratcheting anything.

    Without this, a change that lowers 20 sites but leaves the number alone hands
    the next 20 regressions a free pass.
    """
    total = sum(len(v) for v in _bypass_sites().values())
    assert total == BYPASS_BASELINE, (
        f"the real count is {total} but BYPASS_BASELINE says {BYPASS_BASELINE}. "
        f"Lower the baseline to {total} in this change -- that is the ratchet clicking."
    )


def test_the_needles_match_a_known_call_site() -> None:
    """POSITIVE CONTROL: the scan finds a call that is known to be there.

    Without this, the counting tests pass trivially on a tree where the needles
    match nothing at all -- a renamed primitive, a typo in a fragment -- and that
    reads as total success.
    """
    module, needle = KNOWN_CALL_SITE
    tree = _parse(SRC / module)
    assert tree is not None, f"{module} must parse"
    found = [name for _, name in _primitive_calls(tree)]
    assert needle in found, (
        f"{module} calls no {needle}, so the needles in this file match nothing "
        f"and every count above is meaningless"
    )


@pytest.mark.parametrize(("module", "function"), ATTRIBUTED_KILL_PATHS)
def test_attributed_kill_paths_note_the_kill(module: str, function: str) -> None:
    """The two paths every other kill funnels into must say who fired.

    This is what a falling count has to mean. A count that reached zero because
    the primitives were renamed, with nothing ever logged, would satisfy the
    ceiling above and leave the field exactly as undiagnosable as before.
    """
    tree = _parse(SRC / module)
    assert tree is not None, f"{module} must parse"
    target = _function_named(tree, function)
    assert target is not None, f"{module} must define {function}"
    notes = [
        node
        for node in ast.walk(target)
        if isinstance(node, ast.Call) and _callee_name(node) == ATTRIBUTION
    ]
    assert notes, (
        f"{module}:{function} signals a process without calling {ATTRIBUTION}, so a "
        f"runtime it ends leaves nothing in the log naming the caller or the reason"
    )


@pytest.mark.parametrize(("module", "function"), ATTRIBUTED_KILL_PATHS)
def test_the_attribution_names_a_caller_and_a_reason(module: str, function: str) -> None:
    """A bare call that passes neither would log an empty attribution."""
    tree = _parse(SRC / module)
    assert tree is not None
    target = _function_named(tree, function)
    assert target is not None
    for node in ast.walk(target):
        if isinstance(node, ast.Call) and _callee_name(node) == ATTRIBUTION:
            passed = {kw.arg for kw in node.keywords}
            assert {"reason", "caller"} <= passed, (
                f"{module}:{function} calls {ATTRIBUTION} without both reason and "
                f"caller (passed: {sorted(p for p in passed if p)})"
            )
            return
    pytest.fail(f"{module}:{function} does not call {ATTRIBUTION}")
