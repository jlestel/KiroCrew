"""Behaviour of the counted pid shields and the kill attribution line.

Both rules under test are about a process having more than one interested party:
a shield must survive the first holder leaving, and a kill must say who ended it.
"""

from __future__ import annotations

import logging

from kiro_crew import runtime_ownership as ro


class _FakeRuntime:
    """Something that carries a pid, which is all the attribution reads."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid


# -- the counted shield --


def test_a_pid_stays_shielded_until_its_last_holder_leaves() -> None:
    """As a set, the first holder to leave tore the shield off a process the
    second was still using, and the sweep then reaped a live runtime."""
    shield = ro.PidRefcount()
    shield.add(4242)
    shield.add(4242)
    shield.discard(4242)
    assert 4242 in shield, "one holder is gone, the other still needs the shield"
    shield.discard(4242)
    assert 4242 not in shield and not shield


def test_a_zero_count_pid_is_removed_rather_than_kept_at_zero() -> None:
    """Iteration and truthiness must never report a pid nothing holds."""
    shield = ro.PidRefcount()
    shield.add(7)
    shield.discard(7)
    assert list(shield) == [] and len(shield) == 0 and not shield


def test_the_refcount_reads_like_the_set_it_replaces() -> None:
    shield = ro.PidRefcount([11, 22, 22])
    assert set(shield) == {11, 22}
    assert 11 in shield and 33 not in shield
    assert sorted(shield | {33}) == [11, 22, 33]
    assert len(shield) == 2 and bool(shield) is True
    assert shield.count(22) == 2 and shield.count(11) == 1 and shield.count(99) == 0
    shield.clear()
    assert not shield


def test_the_refcount_rejects_what_is_not_a_pid() -> None:
    shield = ro.PidRefcount()
    for value in (0, -1, True, False):
        shield.add(value)  # type: ignore[arg-type]
    assert not shield
    shield.discard(12345)  # unheld: a no-op, not an error


def test_the_protected_pid_shield_is_reference_counted() -> None:
    """Two pools shielding one process. ``session_pid``'s registry is the one
    every app worker pool and the knowledge LLM pool register with, and each
    pairs its own register with its own unregister."""
    from kiro_crew import session_pid

    pid = 999_001
    session_pid.register_protected_pid(pid)
    session_pid.register_protected_pid(pid)
    try:
        session_pid.unregister_protected_pid(pid)
        assert pid in session_pid._protected_pids(), "the second holder still needs it"
        session_pid.unregister_protected_pid(pid)
        assert pid not in session_pid._protected_pids()
    finally:
        # Only THIS pid. The registry is process-global and other tests in the
        # same worker shield their own pids in it, so clearing the whole thing
        # would tear their shields off and fail them instead of this one.
        while pid in session_pid._protected_pids():
            session_pid.unregister_protected_pid(pid)


def test_the_starting_pid_shield_is_reference_counted() -> None:
    """Two concurrent starts of one session legitimately shield one pid: the
    allocator carries a race budget for starting the same session twice."""
    from kiro_crew.session_allocation import SessionRegistryState

    state = SessionRegistryState()
    state.starting_pids.add(4242)
    state.starting_pids.add(4242)
    state.starting_pids.discard(4242)
    assert 4242 in state.starting_pids
    state.starting_pids.discard(4242)
    assert 4242 not in state.starting_pids and not state.starting_pids


def test_the_starting_pid_shield_still_reads_as_a_set_for_the_sweep() -> None:
    """The orphan sweep unions it with other pid sources, so it has to keep
    answering ``in``, iteration and ``set(...)``."""
    from kiro_crew.session_allocation import SessionRegistryState

    state = SessionRegistryState()
    state.starting_pids.add(101)
    assert set(state.starting_pids) == {101}
    assert 101 in state.starting_pids
    assert bool(state.starting_pids) is True


# -- the attribution line --


def test_the_kill_line_names_pid_caller_and_reason(caplog) -> None:
    """The line the field could not get: a death log written where the process
    is reaped says a process died, never who decided it should."""
    with caplog.at_level(logging.WARNING, logger="kiro_crew.runtime_ownership"):
        ro.note_runtime_kill(909, reason="dashboard reset all", caller="reset handler")
    line = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "909" in line and "reset handler" in line and "dashboard reset all" in line


def test_the_kill_line_is_warning_not_info(caplog) -> None:
    """The gateway runs at WARNING, so INFO would omit the one line naming who
    fired from the log of every deployment that has the problem."""
    with caplog.at_level(logging.DEBUG, logger="kiro_crew.runtime_ownership"):
        ro.note_runtime_kill(4242, reason="teardown", caller="test")
    levels = {r.levelno for r in caplog.records}
    assert levels == {logging.WARNING}, f"expected one WARNING record, got levels {levels}"


def test_the_kill_line_reads_a_pid_off_an_object(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="kiro_crew.runtime_ownership"):
        ro.note_runtime_kill(_FakeRuntime(pid=5150), reason="teardown", caller="test")
    assert "5150" in "\n".join(r.getMessage() for r in caplog.records)


def test_an_unsignalable_target_is_still_attributed_without_a_pid(caplog) -> None:
    """A stand-in coerces to 1 through ``__index__``, and pid<=1 selects the
    ``kill(0)`` / ``kill(-n)`` group semantics rather than one process -- so the
    pid is reported as absent rather than as 1, and the line is still written."""
    for target in (None, 0, 1, -1, True, "4242", object()):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="kiro_crew.runtime_ownership"):
            ro.note_runtime_kill(target, reason="teardown", caller="test")
        line = "\n".join(r.getMessage() for r in caplog.records)
        assert "pid=None" in line, f"{target!r} should report no pid, got: {line}"
