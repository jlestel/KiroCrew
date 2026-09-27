"""Behaviour of the counted pid shields and the kill attribution line.

Both rules under test are about a process having more than one interested party:
a shield must survive the first holder leaving, and a kill must say who ended it.
"""

from __future__ import annotations

import logging

import pytest

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


# -- the lease and the gate it feeds --


class _FakeLeaseHolder:
    """A provider shape that can hold a lease, recording what it was asked."""

    def __init__(self, runtime: _FakeRuntime, *, owns: bool = True) -> None:
        self._runtime = runtime
        self._owns_runtime = owns
        self._runtime_lease: str | None = None
        self.acquired = 0
        self.released = 0

    async def acquire_runtime_lease(self) -> None:
        self.acquired += 1
        if not self._owns_runtime or self._runtime_lease is not None:
            return
        runtime = self._runtime

        async def _already_spawned() -> object:
            return runtime

        acquisition = await ro.RUNTIME_OWNERSHIP.acquire(
            runtime, "sess", _already_spawned, cap=ro.CHAT_RUNTIME_CAP
        )
        self._runtime_lease = acquisition.lease

    async def release_runtime_lease(self) -> None:
        self.released += 1
        lease = self._runtime_lease
        if lease is None:
            return
        self._runtime_lease = None
        await ro.RUNTIME_OWNERSHIP.release(lease)


class _Wrapper:
    """The outer provider: it holds the leasing one at ``_client``."""

    def __init__(self, inner: object) -> None:
        self._client = inner


@pytest.mark.asyncio
async def test_a_leased_runtime_refuses_a_kill_and_an_unleased_one_allows_it() -> None:
    """The whole point of the table: the gate's answer must change with it."""
    ro._reset_for_tests()
    runtime = _FakeRuntime(pid=5150)
    holder = _FakeLeaseHolder(runtime)
    assert ro.authorize_runtime_kill(runtime, reason="before", caller="t") is True
    await holder.acquire_runtime_lease()
    assert (
        ro.authorize_runtime_kill(runtime, reason="leased", caller="t") is False
    ), "a live tenant holds this process; a kill must be refused"
    await holder.release_runtime_lease()
    assert ro.authorize_runtime_kill(runtime, reason="after", caller="t") is True
    ro._reset_for_tests()


@pytest.mark.asyncio
async def test_the_helpers_find_the_holder_through_the_outer_provider() -> None:
    """Both shapes reach the kill paths, so both must be leasable."""
    ro._reset_for_tests()
    runtime = _FakeRuntime(pid=5151)
    inner = _FakeLeaseHolder(runtime)
    outer = _Wrapper(inner)
    await ro.acquire_session_lease(outer)
    assert inner.acquired == 1, "the lease lives on _client, not on the wrapper"
    assert ro.authorize_runtime_kill(runtime, reason="leased", caller="t") is False
    await ro.release_session_lease(outer)
    assert inner.released == 1
    assert ro.authorize_runtime_kill(runtime, reason="free", caller="t") is True
    ro._reset_for_tests()


@pytest.mark.asyncio
async def test_a_shape_with_no_lease_methods_is_skipped_not_an_error() -> None:
    """Non-ACP backends and the pre-startup placeholder client reach these calls."""
    ro._reset_for_tests()
    await ro.acquire_session_lease(object())
    await ro.release_session_lease(object())
    await ro.acquire_session_lease(_Wrapper(object()))
    ro._reset_for_tests()


@pytest.mark.asyncio
async def test_a_subagent_never_takes_a_lease() -> None:
    """It is handed a runtime it did not spawn and must not kill, so a lease of
    its own would refuse the owner's legitimate teardown."""
    ro._reset_for_tests()
    runtime = _FakeRuntime(pid=5152)
    await ro.acquire_session_lease(_FakeLeaseHolder(runtime, owns=False))
    assert ro.authorize_runtime_kill(runtime, reason="owner", caller="t") is True
    ro._reset_for_tests()


@pytest.mark.asyncio
async def test_releasing_twice_cannot_drop_a_later_tenants_lease() -> None:
    """A teardown racing the dashboard's reset calls release more than once."""
    ro._reset_for_tests()
    runtime = _FakeRuntime(pid=5153)
    first = _FakeLeaseHolder(runtime)
    await first.acquire_runtime_lease()
    await first.release_runtime_lease()
    second = _FakeLeaseHolder(runtime)
    await second.acquire_runtime_lease()
    await first.release_runtime_lease()
    assert (
        ro.authorize_runtime_kill(runtime, reason="second", caller="t") is False
    ), "the second tenant's lease must survive the first one's repeated release"
    ro._reset_for_tests()


@pytest.mark.asyncio
async def test_the_refusal_names_the_pid_the_caller_and_the_reason() -> None:
    """The refusal log is the only record that a kill was stopped."""
    ro._reset_for_tests()
    runtime = _FakeRuntime(pid=5154)
    holder = _FakeLeaseHolder(runtime)
    await holder.acquire_runtime_lease()
    logger = logging.getLogger("kiro_crew.runtime_ownership")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        assert ro.authorize_runtime_kill(runtime, reason="sweep", caller="reaper") is False
    finally:
        logger.removeHandler(handler)
    assert records and records[0].levelno == logging.WARNING
    text = records[0].getMessage()
    assert "5154" in text and "reaper" in text and "sweep" in text
    ro._reset_for_tests()


@pytest.mark.asyncio
async def test_a_mock_shaped_provider_is_not_mistaken_for_a_lease_holder() -> None:
    """Mocked providers flow through these kill paths all over the suite.

    A ``MagicMock`` answers every ``hasattr`` and returns a ``MagicMock`` from the
    call, which is not awaitable -- so recognising a holder by its methods turns
    every mocked provider on a release path into a TypeError at the await.
    """
    from unittest.mock import MagicMock

    await ro.release_session_lease(MagicMock())
    await ro.acquire_session_lease(MagicMock())


@pytest.mark.asyncio
async def test_a_holder_that_currently_leases_nothing_is_still_a_holder() -> None:
    """An empty slot means "no lease yet", not "cannot hold one" -- otherwise the
    first acquire on every session would be skipped."""
    ro._reset_for_tests()
    runtime = _FakeRuntime(pid=5155)
    holder = _FakeLeaseHolder(runtime)
    assert holder._runtime_lease is None
    await ro.acquire_session_lease(holder)
    assert holder.acquired == 1 and isinstance(holder._runtime_lease, str)
    await ro.release_session_lease(holder)
    ro._reset_for_tests()
