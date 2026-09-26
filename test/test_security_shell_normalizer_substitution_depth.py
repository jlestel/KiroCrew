"""``_SubstitutionDepth``: the case-aware successor to the paren counter.

Every argv window on the floor stops at the first token that ends the argv
while no command substitution is open.  Measured with the bare
``_substitution_depth_delta`` -- a per-token character count with no shell
grammar -- a ``case`` PATTERN's ``)`` (the token ``x)``) scores as a
substitution closer and a pattern's ``|`` (``x|y)``) as a separator, so such a
window closes at the first list operator after ``esac`` and every later clause
is lost.  These tests pin the walker on the token shapes the tokenizer
actually produces (``$(case``, ``x)``, ``(x)``, ``x|y)``, ``:;;``, ``esac;``,
``esac)``), each measured against what bash spans.
"""

from __future__ import annotations

import pytest

from kiro_crew.security.shell_normalizer import (
    _outside_expansions,
    _substitution_depth_delta,
    _SubstitutionDepth,
    normalize_shell_command,
)

_NAME = "kiro" + "crew"


def _window(cmd: str, *, command_position: bool = False, skip: int = 1) -> list[str]:
    """The tokens a consumer window anchored after the first word would scan."""
    depth = _SubstitutionDepth(command_position=command_position)
    scanned: list[str] = []
    for token in normalize_shell_command(cmd)[skip:]:
        scanned.append(token)
        if depth.feed(token):
            break
    return scanned


class TestCasePatternDoesNotCloseTheWindow:
    """A ``kill $( ... )`` whose body ends in a lookup: the lookup stays in the window."""

    @pytest.mark.parametrize(
        "body",
        [
            # the issue's measured spelling
            "case x in x) :;; esac; pgrep -f {n}",
            # a leading ``(`` on the pattern, and the ``$( case`` split spelling
            "case x in (x) :;; esac; pgrep -f {n}",
            " case x in (x) :;; esac; pgrep -f {n}",
            # a pattern alternation: its ``|`` is not a pipe
            "case x in x|y) :;; esac; pgrep -f {n}",
            "case x in (x|y) :;; esac; pgrep -f {n}",
            # several clauses: ``;;`` re-arms the pattern each time
            "case x in x) :;; y) :;; esac; pgrep -f {n}",
            # the other clause terminators
            "case x in x) :;& y) :;;& esac; pgrep -f {n}",
            # an empty case, and a body glued to its pattern
            "case x in esac; pgrep -f {n}",
            "case x in x)pgrep -f {n};; esac",
            # a substitution and an extglob group INSIDE the pattern balance first
            "case x in $(echo x)) :;; esac; pgrep -f {n}",
            "case x in @(a|b)) :;; esac; pgrep -f {n}",
            # nested: a case inside a substitution inside a case body, and a
            # plain nested case whose ``esac;;`` closes the inner and re-arms the outer
            "case x in x) $(case y in y) :;; esac);; esac; pgrep -f {n}",
            "case x in x) case y in y) :;; esac;; esac; pgrep -f {n}",
            # case in every command position bash grants it
            "if true; then case x in x) :;; esac; fi; pgrep -f {n}",
            "while :; do case x in x) :;; esac; break; done; pgrep -f {n}",
            "function f case x in x) :;; esac; pgrep -f {n}",
            "time -p case x in x) :;; esac; pgrep -f {n}",
            "coproc case x in x) :;; esac; pgrep -f {n}",
            # ``esac`` as an ARGUMENT does not disarm; ``esac`` in command position does
            "case x in x) echo esac;; esac; pgrep -f {n}",
            # a clause body glued to its pattern is fed through the grammar too:
            # a nested ``case``, an ``esac``, a substitution opener
            "case x in x)case y in y) :;; esac;; esac; pgrep -f {n}",
            "case x in x)esac; pgrep -f {n}",
            "case x in x)$(case y in y) :;; esac);; esac; pgrep -f {n}",
            # a ``)`` inside a parameter expansion is expansion text, in the
            # pattern and in an ordinary word of the body alike
            "case x in ${{v:-x)}}) :;; esac; pgrep -f {n}",
            "case x in ${{v:-$(echo x)}}) :;; esac; pgrep -f {n}",
            "case x in ${{a:-${{b)}}}}) :;; esac; pgrep -f {n}",
            ": ${{v:-x)}}; pgrep -f {n}",
            # a quoted ``)`` in a position bash refuses unquoted is pattern text
            "case x in ')') :;; esac; pgrep -f {n}",
            "case x in 'x)') :;; esac; pgrep -f {n}",
            # the EMPTY pattern de-quotes to a bare ``)``: it terminates
            "case x in '') :;; esac; pgrep -f {n}",
            'case x in "") :;; esac; pgrep -f {n}',
            "case x in '))') :;; esac; pgrep -f {n}",
            "case x in (')') :;; esac; pgrep -f {n}",
            # the WORD spans tokens: ``in`` follows once its substitution closes
            "case $(echo x) in x) :;; esac; pgrep -f {n}",
            "case $(echo x y) in x) :;; esac; pgrep -f {n}",
            "case $(echo $(echo x)) in x) :;; esac; pgrep -f {n}",
        ],
    )
    def test_the_whole_body_is_one_argument(self, body: str) -> None:
        cmd = "kill $(" + body.format(n=_NAME) + ")"
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_the_counter_alone_still_closes_early(self) -> None:
        # The regression this walker exists for, measured on the raw counter.
        depth = 0
        seen: list[str] = []
        for token in normalize_shell_command("kill $(case x in x) :;; esac; pgrep -f x)")[1:]:
            seen.append(token)
            depth += _substitution_depth_delta(token)
            if depth <= 0 and (";" in token or "|" in token):
                break
        assert seen == ["$(case", "x", "in", "x)", ":;;"]


class TestBoundariesOutsideACaseAreUnchanged:
    """Everything the counter got right, the walker gets right the same way."""

    @pytest.mark.parametrize(
        ("cmd", "expected"),
        [
            ("kill 123; echo $(cat /tmp/x)", ["123;"]),
            ("kill $(true; echo 1)", ["$(true;", "echo", "1)"]),
            ("kill $(pgrep -f other); case x in x) :;; esac; echo x", ["$(pgrep", "-f", "other);"]),
            # a case that STARTS after the window's own argv ended is never entered
            ("kill 123; case x in x) :;; esac; pgrep -f x", ["123;"]),
            # the window's first token is an ARGUMENT: ``case`` there is data
            ("kill case x in x) :;; esac; pgrep -f x", ["case", "x", "in", "x)", ":;;"]),
            # ``case`` after an ordinary verb, an assignment or a redirect is data
            ("kill $(echo case x in x); pgrep -f x)", ["$(echo", "case", "x", "in", "x);"]),
            ("kill $(v=1 case x in x); pgrep -f x)", ["$(v=1", "case", "x", "in", "x);"]),
            (
                "kill $(echo -n case x in x); pgrep -f x)",
                ["$(echo", "-n", "case", "x", "in", "x);"],
            ),
            # a redirect glued to ``esac`` still disarms, and the substitution's
            # own ``)`` behind it closes the window at the separator
            (
                "kill $(case x in a) :;; esac>/dev/null); echo x",
                ["$(case", "x", "in", "a)", ":;;", "esac>/dev/null);"],
            ),
            (
                "kill $(case x in a) :;; esac 2>&1); echo x",
                ["$(case", "x", "in", "a)", ":;;", "esac", "2>&1);"],
            ),
        ],
    )
    def test_window(self, cmd: str, expected: list[str]) -> None:
        assert _window(cmd) == expected

    def test_a_frame_walk_starts_in_command_position(self) -> None:
        # The rsync/ssh outer walk feeds a whole frame from its first token,
        # where ``case`` IS the reserved word: the pattern's ``|`` must not be
        # read as a pipe there either.
        cmd = "case $x in a|b) rsync -e ssh x y;; esac; echo z"
        depth = _SubstitutionDepth(command_position=True)
        boundaries = [tok for tok in normalize_shell_command(cmd) if depth.feed(tok)]
        assert boundaries == ["y;;", "esac;"]

    def test_depth_never_goes_negative(self) -> None:
        depth = _SubstitutionDepth()
        depth.feed(")")
        depth.feed(")")
        assert depth.top_level
        depth.feed("$(x")
        assert not depth.top_level
        depth.feed("y)")
        assert depth.top_level


class TestParensInsideAParameterExpansionAreText:
    """``_outside_expansions``: what the walker counts of a token with ``${ … }``."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("${v:-x)}", "v:-x"),
            ("${v:-x)})", "v:-x)"),
            # a ``$(`` opened inside the expansion is real, and so is its closer
            ("${v:-$(a)}b)", "v:-$(a)b)"),
            ("${v:-$(a;", "v:-$(a;"),
            ("b)}", "b)}"),
            ("${PIDS:-$(pgrep x)};", "PIDS:-$(pgrep x);"),
            ("${a:-${b)}}x)", "a:-bx)"),
            # no expansion: the token is returned as is
            ("$(pgrep", "$(pgrep"),
            ("plain)", "plain)"),
        ],
    )
    def test_reduction(self, token: str, expected: str) -> None:
        assert _outside_expansions(token) == expected

    def test_an_expansion_closer_does_not_end_the_window(self) -> None:
        # ``kill $(: ${v:-x)}; pgrep -f <name>)``: the body is one argument.
        cmd = "kill $(: ${v:-x)}; pgrep -f x)"
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_a_real_closer_after_an_expansion_still_ends_it(self) -> None:
        cmd = "kill ${PIDS:-$(pgrep x; echo 1)}; echo x"
        assert _window(cmd) == ["${PIDS:-$(pgrep", "x;", "echo", "1)};"]


class TestNewlinesInsideTheGrammarAreTransparent:
    """A frame renders a newline as a standalone ``;``; the case state holds across it."""

    @pytest.mark.parametrize(
        "frame",
        [
            # newline between the WORD and ``in``, one and two of them
            ["$(case", "x", ";", "in", "x)", ":;;", "esac;", "pgrep", "-f", "x)"],
            ["$(case", "x", ";", ";", "in", "x)", ":;;", "esac;", "pgrep", "-f", "x)"],
            # newline between ``in`` and the first pattern, and between ``;;`` and the next
            ["$(case", "x", "in", ";", "x)", ":;;", "esac;", "pgrep", "-f", "x)"],
            ["$(case", "x", "in", "x)", ":;;", ";", "y)", ":;;", "esac;", "pgrep", "-f", "x)"],
            # newline between ``;;`` and ``esac``: ``esac`` still reads as the word
            ["$(case", "x", "in", "x)", ":;;", ";", "esac;", "pgrep", "-f", "x)"],
        ],
    )
    def test_the_body_stays_one_argument(self, frame: list[str]) -> None:
        depth = _SubstitutionDepth()
        ended = [depth.feed(tok) for tok in frame]
        assert ended == [False] * len(frame)
        assert depth.top_level

    def test_esac_after_a_newline_closes_the_case_for_real(self) -> None:
        # ``kill $(case x in x) :;;<newline>esac); echo <name>``: the ``esac)``
        # must close the compound AND the substitution, so the window ends at ``;``.
        depth = _SubstitutionDepth()
        frame = ["$(case", "x", "in", "x)", ":;;", ";", "esac);", "echo", "x"]
        ended = [depth.feed(tok) for tok in frame]
        assert ended == [False, False, False, False, False, False, True, False, False]
        assert depth.top_level


class TestDataTokensStillAdvanceAPendingPattern:
    """``feed_data``: a caller's data token is inert unless bash reads it as the pattern."""

    def test_outside_a_case_nothing_changes(self) -> None:
        depth = _SubstitutionDepth()
        depth.feed("$(x")
        assert depth.feed_data("print(1); y)") is False
        assert depth.depth == 1  # the payload's parens and separator are data

    def test_an_option_shaped_pattern_is_still_the_pattern(self) -> None:
        # ``case $1 in -c) echo hi;; esac); grep x`` -- an operand scan reads
        # ``-c)`` as an option and would skip it; the pattern must close anyway.
        depth = _SubstitutionDepth()
        for tok in ["$(case", "$1", "in"]:
            depth.feed(tok)
        assert depth.feed_data("-c)") is False
        ended = [depth.feed(tok) for tok in ["echo", "hi;;", "esac);"]]
        assert ended == [False, False, True]
        assert depth.top_level


class TestEsacReArms:
    """After ``esac`` the walker is out of the case: a later ``)`` closes for real."""

    def test_a_closer_after_esac_counts(self) -> None:
        depth = _SubstitutionDepth()
        for token in normalize_shell_command("kill $(case x in x) :;; esac)")[1:]:
            ended = depth.feed(token)
        assert depth.top_level
        assert ended is False

    def test_a_pattern_after_esac_is_a_closer_again(self) -> None:
        # Once the compound is closed, ``x)`` is what it always was to the counter.
        depth = _SubstitutionDepth()
        for token in ["$(case", "x", "in", "x)", ":;;", "esac;", "$(echo", "x)"]:
            depth.feed(token)
        assert depth.depth == 1
        depth.feed("x)")
        assert depth.top_level
