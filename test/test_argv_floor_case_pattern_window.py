"""Every argv window on the floor survives a ``case`` compound in its argv.

The six consumers of the substitution-depth walker each bound a different
window (the mint-verb scan, the ``pkill`` pattern leg, the bare-``kill`` body
window, the product CLI's operand list, the rsync environment walk and the
ssh-family argument walk).  Before ``_SubstitutionDepth`` they all shared one
blind spot: a ``case`` PATTERN's ``)`` scored as a substitution closer, so the
window closed at the first separator after ``esac`` and the argument that named
the target was never read.  Only the bare-``kill`` window had a compensation
(its raw-text re-derivation).

Each consumer gets the two measurements the issue asks for: the shape that
bash runs and the floor must deny, and a false-positive control -- an allowed
command carrying ``case ... esac`` followed by an unrelated command that
mentions the protected word -- that must stay allowed.  Verdicts go through the
public gate so the tests survive a refactor of how the floors are wired.
"""

from __future__ import annotations

import pytest

from kiro_crew.security import argv_floor, is_denied

_K = "k" + "ill"
_PK = "p" + _K
_NAME = "kiro" + "crew"
_TOK = "to" + "ken"

_RULE_KILL = "self-protection-" + _K
_RULE_MINT = "credential-exfil-" + _NAME + "-" + _TOK
_RULE_SSH = "sandbox-escape-ssh-self"

# The esac-tail body the issue measured, parametrised on the trailing command.
_TAIL = "case x in x) :;; esac; {cmd}"


def _rule(cmd: str) -> "str | None":
    verdict = is_denied(cmd)
    if verdict is None:
        return None
    return verdict.rsplit("rule=", 1)[1].split()[0]


class TestRowOneIsDeniedByTheTokenWindowAlone:
    """The issue's acceptance: revert the raw window locally, row 1 still denies."""

    ROW_ONE = f"{_K} $(" + _TAIL.format(cmd=f"pgrep -f {_NAME}") + ")"

    def test_with_the_raw_window_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(argv_floor, "_bare_" + _K + "_raw_bodies", lambda source: [])
        assert argv_floor._is_self_kill(self.ROW_ONE) is True

    def test_through_the_gate(self) -> None:
        assert _rule(self.ROW_ONE) == _RULE_KILL


class TestEveryConsumerReadsPastThePattern:
    """One denied shape per consumer whose target sits after ``esac``."""

    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            # mint-verb scan: the verb after the substitution's esac-tail body
            (f"{_NAME} $(" + _TAIL.format(cmd="echo status") + f") {_TOK}", _RULE_MINT),
            (f"{_NAME} $(case x in x|y) :;; esac; echo status) {_TOK}", _RULE_MINT),
            # the case WORD is itself a substitution spanning tokens, and the
            # body's ``)`` rides on ``esac`` -- the verb is the next word
            (f"{_NAME} $(case $(echo x) in x) :;; esac) {_TOK}", _RULE_MINT),
            # a QUOTED ``)`` as the pattern: de-quoted it is ``))``, and bash
            # refuses that unquoted, so the run is pattern text
            (f"{_NAME} $(case x in ')') :;; esac) {_TOK}", _RULE_MINT),
            (f"{_NAME} $(case x in 'x)') :;; esac; echo status) {_TOK}", _RULE_MINT),
            # newlines where bash allows them: before ``in``, before the first
            # pattern, before ``;;`` and before ``esac`` (a frame renders each
            # as a standalone ``;``)
            (f"{_NAME} $(case x\nin x) :;; esac; :) {_TOK}", _RULE_MINT),
            (f"{_NAME} $(case x in\nx) :;; esac; :) {_TOK}", _RULE_MINT),
            (f"{_NAME} $(case x in x)\n:;;\nesac; :) {_TOK}", _RULE_MINT),
            # an option-shaped pattern (``-c)``) the operand scan would skip
            (f"{_NAME} $(case $1 in -c) echo status;; esac) {_TOK}", _RULE_MINT),
            # the empty pattern de-quotes to a bare ``)`` and terminates
            (f"{_K} $(case x in '') :;; esac; pgrep -f {_NAME})", _RULE_KILL),
            # a ``)`` inside a parameter expansion, as the pattern and as a body word
            (f"{_NAME} $(case x in ${{v:-x)}}) :;; esac; :) {_TOK}", _RULE_MINT),
            (f"{_PK} -f $(case x in ${{v:-x)}}) :;; esac; echo {_NAME})", _RULE_KILL),
            (f"{_NAME} $(: ${{v:-x)}}; echo status) {_TOK}", _RULE_MINT),
            # a nested ``case`` glued to the outer pattern
            (f"{_NAME} $(case x in x)case y in y) :;; esac;; esac; :) {_TOK}", _RULE_MINT),
            (f"{_NAME} $(case $(echo x) in x) :;; esac; echo status) {_TOK}", _RULE_MINT),
            # pkill pattern leg: the name is produced INSIDE the body, after esac
            (f"{_PK} -f $(" + _TAIL.format(cmd=f"echo {_NAME}") + ")", _RULE_KILL),
            (f"{_PK} -f $(case $(echo x) in x) :;; esac; echo {_NAME})", _RULE_KILL),
            # bare kill body window (row 1 of the issue, plus the paren-pattern spelling)
            (f"{_K} $(" + _TAIL.format(cmd=f"pgrep -f {_NAME}") + ")", _RULE_KILL),
            (f"{_K} $(case x in (x) :;; esac; pgrep -f {_NAME})", _RULE_KILL),
            # ssh-family argument walk: the self-host operand after the body
            ("rsync -e ssh $(" + _TAIL.format(cmd="echo src") + ") 127.0.0.1:/x", _RULE_SSH),
        ],
    )
    def test_denied(self, cmd: str, rule: str) -> None:
        assert _rule(cmd) == rule


class TestAnUnrelatedLaterCommandIsNotAttributed:
    """False-positive controls, one per consumer: the case ends, the argv ends."""

    @pytest.mark.parametrize(
        "cmd",
        [
            # mint-verb scan
            f"{_NAME} status $(case x in a) echo b;; esac); echo {_TOK}",
            f"{_NAME} status $(case x in a|b) echo c;; esac); echo {_TOK}",
            # pkill pattern leg
            f"{_PK} -f $(" + _TAIL.format(cmd="echo other") + f"); echo {_NAME}",
            f"{_PK} -f other; " + _TAIL.format(cmd=f"echo {_NAME}"),
            # bare kill body window
            f"{_K} $(" + _TAIL.format(cmd="pgrep -f other") + f"); echo {_NAME}",
            f"{_K} 123; " + _TAIL.format(cmd=f"echo {_NAME}"),
            f"{_K} $(case x in a) :;; esac>/dev/null); echo {_NAME}",
            # ``esac`` after a newline still ends the compound, so the window ends
            f"{_K} $(case x in x) :;;\nesac); echo {_NAME}",
            f"{_K} $(case x in x) :;;\nesac; pgrep -f other); echo {_NAME}",
            f"{_K} $(case x\nin x) :;; esac; pgrep -f other); echo {_NAME}",
            # the EMPTY pattern (``'')``) terminates: the window ends at the ``;`` after esac
            f"{_PK} -f $(case $x in '') echo foo;; esac); cd ~/{_NAME}",
            "rsync -e ssh $(case $x in '') echo src;; esac) remotebox:/x; echo 127.0.0.1",
            # an option-shaped pattern is fed to the walker even though the operand
            # scan skips it, so the pattern closes and the window ends at ``esac);``
            f"{_NAME} status $(case $1 in -c) echo hi;; esac); grep {_TOK} log",
            f"{_NAME} status $(case $1 in -c) echo hi;; esac) restart",
            # an expansion's own ``)`` does not keep the window open past the ``;``
            f"{_K} ${{v:-x)}}; echo {_NAME}",
            f"{_K} ${{PIDS:-$(pgrep x)}}; echo {_NAME}",
            f"{_K} ${{PIDS:-$(pgrep x; echo 1)}}; echo {_NAME}",
            # product CLI operands: the leading operand decides, a case body in
            # the argv does not make ``restart`` leading
            f"{_NAME} status $(case x in a) echo b;; esac) restart",
            # rsync environment walk: a pattern alternation is not a separator
            # and the selector never crosses the compound
            "case $x in a|b) rsync -e ssh x y;; esac; echo localhost",
            "case $x in a|b) rsync x remotebox:/y;; esac; echo 127.0.0.1",
            # ssh-family argument walk
            "ssh host $(" + _TAIL.format(cmd="echo hi") + "); echo localhost",
            "rsync -e ssh $(" + _TAIL.format(cmd="echo src") + ") remotebox:/x; echo 127.0.0.1",
            "scp $(" + _TAIL.format(cmd="echo f") + ") remotebox:/x; echo localhost",
        ],
    )
    def test_allowed(self, cmd: str) -> None:
        assert _rule(cmd) is None
