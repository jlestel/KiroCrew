"""Runtime ownership: the pool owns the process, a session holds a LEASE.

A session must never be able to name a process. It asks for a runtime, gets a
LEASE, and gives the lease back; whether that release ends the process is the
pool's decision and nobody else's.

:class:`RuntimeOwnership`
    The lease table. A runtime lives while at least one lease is outstanding. The
    LAST release hands the runtime back for teardown; every earlier release only
    drops a reference. ``cap`` bounds how many leases one process serves, and so
    bounds the blast radius of that process dying.

:func:`authorize_runtime_kill`
    The gate. One place asks "does anyone still hold this?" before a runtime
    dies, and one place records WHO asked. Without the table the gate has nothing
    to consult; without the gate the table is advisory and any caller with a pid
    can still take a process out from under its tenants.

:class:`PidRefcount`
    The same rule one layer down, for the sweep shields: a pid stays shielded
    until its LAST holder drops it.

Why the gate and the release sites are one change
-------------------------------------------------
The gate can only be switched on in the same change that makes releasing
universal. A gate wired to a provider that holds a lease for its lifetime also
refuses the force-kill paths -- ``_sync_kill_provider`` is reached through
``_dispatch_hard_kill`` and through the dashboard's reset-all fallback -- so a
teardown that does not release first is declined, and the process it declines to
signal leaks. Every path that legitimately ends a runtime therefore releases
before it signals, and that is why they all move together.

Why the gate is separate from the killing
-----------------------------------------
Ownership and identity are different questions and both must be answered:

* ownership -- "does anyone still need this runtime?" -- is THIS module, and a
  wrong answer kills a co-tenant's live process;
* identity -- "is this pid still the process I recorded?" -- is
  :mod:`kiro_crew.process_identity`, and a wrong answer kills a stranger that
  inherited a recycled pid.

So :func:`authorize_runtime_kill` returns a verdict rather than doing the
killing, and a caller that owns a careful escalation -- a SIGTERM grace before
SIGKILL, a process group proved by a vouching member, a Windows tree pinned by
handle -- keeps it and asks the gate once, at the top. Folding those into a
general kill helper would mean replacing a group id captured while the leader was
alive with one re-resolved from the pid at signal time, which is the identity
defect above.

This module is a LEAF on purpose: it imports nothing from ``kiro_crew``.
``session_pid`` is the pid bookkeeping leaf and calls the gate, so an import from
here into the agent layer would close ``session_pid -> acp -> acp.runtime ->
session_pid`` and raise ``ImportError`` on the first import of either
(``test_agent_lifecycle_cycle.py`` pins the absence), and the agent-SDK boundary
gate refuses such an import outright, type-only ones included.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Hashable, Iterable, Iterator, MutableSet
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Leases one chat runtime may serve. ONE, so every chat session keeps its own
#: process: at this cap no acquisition can join an occupied runtime, so each
#: release is a last release and hands the process straight back. Raising it is a
#: behaviour change and needs the eligibility rules that decide WHICH sessions
#: may share a process, which do not live here.
CHAT_RUNTIME_CAP = 1


@runtime_checkable
class OwnedRuntime(Protocol):
    """What this table needs of a runtime: a pid, and whether it is still alive.

    A Protocol rather than an import of the concrete runtime class. The
    agent-SDK boundary gate refuses application code that reaches the ACP layer,
    type-only imports included -- and it is right to, because this module is
    reached FROM below. It is also the honest contract: the table reads a pid and
    asks whether the process is alive, and calls nothing else on what it holds.
    """

    @property
    def pid(self) -> int | None: ...

    def is_alive(self) -> bool: ...


class PidRefcount(MutableSet[int]):
    """A pid shield that counts its holders instead of merely listing them.

    A ``set`` of shielded pids is wrong as soon as two holders shield one pid:
    the first to leave calls ``discard`` and tears the shield off a process the
    second is still using. This counts, so a pid stays shielded until the LAST
    holder drops it, and holders stay independent -- which is the same rule the
    lease table applies to runtimes, one layer up.

    Reads like the set it replaces (``in``, iteration, ``len``, truthiness,
    ``set(...)``, set algebra), so a holder that pairs every ``add`` with one
    ``discard`` needs no change. A pid whose count reaches zero is REMOVED rather
    than left at zero, so iteration and truthiness never report a pid nothing
    holds.
    """

    def __init__(self, initial: Iterable[int] | None = None) -> None:
        self._counts: Counter[int] = Counter()
        for pid in initial or ():
            self.add(pid)

    def add(self, value: int) -> None:
        """Take a reference on *value*; the first one raises the shield."""
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            self._counts[value] += 1

    def discard(self, value: int) -> None:
        """Drop ONE reference; the shield falls only when the last one goes."""
        remaining = self._counts.get(value)
        if remaining is None:
            return
        if remaining <= 1:
            del self._counts[value]
        else:
            self._counts[value] = remaining - 1

    def count(self, value: int) -> int:
        """How many holders *value* currently has (0 when unshielded)."""
        return self._counts.get(value, 0)

    def clear(self) -> None:
        """Drop every reference on every pid."""
        self._counts.clear()

    def __contains__(self, value: object) -> bool:
        return value in self._counts

    def __iter__(self) -> Iterator[int]:
        return iter(list(self._counts))

    def __len__(self) -> int:
        return len(self._counts)

    def __repr__(self) -> str:
        return f"PidRefcount({dict(self._counts)!r})"


@dataclass
class _Entry:
    """One live runtime and the LEASES held on it.

    Keyed by lease rather than by session key, because a session key is not a
    count. The allocator legitimately starts the same session twice at once (it
    carries a race budget for exactly that), and both starts must be releasable
    on their own: if the two shared one reference, the loser's teardown would
    drop it and the winner's live process would be killed mid-turn.
    """

    runtime: OwnedRuntime
    key: Hashable
    leases: dict[str, str] = field(default_factory=dict)

    def has_room(self, cap: int) -> bool:
        return len(self.leases) < cap

    def session_keys(self) -> list[str]:
        return sorted(set(self.leases.values()))


@dataclass
class Acquisition:
    """What :meth:`RuntimeOwnership.acquire` handed back.

    ``lease`` is this acquisition's own handle and the ONLY thing that releases
    it. Two acquisitions -- even for one session key -- get two leases, and the
    runtime dies when the last of them is gone.

    ``joined`` is False for the acquisition that FOUNDED the runtime and True for
    every one that landed on an existing one. Callers use it for the two
    decisions that differ: whether the provider owns the process, and whether
    per-session start work that rewrites process-level state may run.
    """

    runtime: OwnedRuntime
    joined: bool
    leases_on_runtime: int
    lease: str


class RuntimeOwnership:
    """The lease table: which runtimes exist and who still needs them.

    One lock serializes the whole registry rather than one lock per key. The
    critical section is a dict lookup plus, for a miss, one spawn, and holding it
    across that spawn is a deliberate trade of concurrency for simplicity: the
    bookkeeping this lock protects -- entry list, lease index, last-runtime map --
    stays provably consistent because nothing else can observe it mid-spawn.
    """

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._by_lease: dict[str, _Entry] = {}
        self._last_for_session: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    # -- writes --

    async def acquire(
        self,
        key: Hashable,
        session_key: str,
        spawn: Callable[[], Awaitable[OwnedRuntime]],
        *,
        cap: int = CHAT_RUNTIME_CAP,
    ) -> Acquisition:
        """Take a LEASE on a runtime, spawning one only when needed.

        Placement: the runtime this session key last landed on, when it is still
        alive, compatible and has room -- which keeps a restart on the process
        its history is on -- then the first compatible runtime with room,
        otherwise ``spawn()``. Compatible means an equal *key*, and room means
        fewer than *cap* leases are held on it.

        Every call mints its OWN lease, including a second call for a session key
        that already holds one. That is the point rather than an oversight: the
        allocator legitimately starts one session twice at once, and each start
        has to be releasable without ending the other.

        At ``cap=1`` no entry with a lease has room, so every acquisition spawns
        and serves exactly one lease -- which is the unpooled behaviour, with the
        lease recorded.

        A dead runtime is never handed out and never counted: it is dropped on the
        way past, so the caller that finds none alive spawns, and the sessions
        that were on it rejoin through this same path on their own next turn.
        """
        async with self._lock:
            self._drop_dead_locked()
            entry = self._pick_locked(key, cap=cap, session_key=session_key)
            joined = entry is not None
            if entry is None:
                runtime = await spawn()
                entry = _Entry(runtime=runtime, key=key)
                self._entries.append(entry)
            lease = uuid.uuid4().hex
            entry.leases[lease] = session_key
            self._by_lease[lease] = entry
            self._last_for_session[session_key] = entry
            logger.info(
                "runtime_ownership outcome=%s pid=%s leases=%d cap=%d runtimes=%d",
                "joined" if joined else "spawned",
                _pid_of(entry.runtime),
                len(entry.leases),
                cap,
                len(self._entries),
            )
            return Acquisition(
                runtime=entry.runtime,
                joined=joined,
                leases_on_runtime=len(entry.leases),
                lease=lease,
            )

    async def release(self, lease: str) -> OwnedRuntime | None:
        """Drop ONE lease; return the runtime to kill only if it was the last.

        The runtime is returned rather than killed here so the caller performs
        the teardown it already performs, with its own logging and error
        handling. Returns None while any other lease is outstanding, which is
        what stops one session's close, reset or model switch -- or the teardown
        of a start that lost a same-key race -- from killing a process another
        acquisition is still using.

        At ``cap=1`` there is never another lease, so this always returns the
        runtime and the caller always kills it, exactly as it did before the
        table existed.

        Releasing is not optional for a caller that is about to signal. The gate
        refuses a runtime whose lease is still out, so a kill path that skips its
        release refuses its own teardown and leaks the process.
        """
        async with self._lock:
            entry = self._by_lease.pop(lease, None)
            if entry is None:
                return None
            entry.leases.pop(lease, None)
            if entry.leases:
                logger.info(
                    "runtime_ownership outcome=released pid=%s remaining_leases=%d",
                    _pid_of(entry.runtime),
                    len(entry.leases),
                )
                return None
            self._forget_entry_locked(entry)
            logger.info(
                "runtime_ownership outcome=last_release pid=%s",
                _pid_of(entry.runtime),
            )
            return entry.runtime

    # -- reads --

    def pid_has_outstanding_leases(self, pid: int) -> bool:
        """Whether a LIVE runtime on *pid* still has leases outstanding.

        The reset path asks this to decide whether a pid that outlived one
        session's shutdown is a survivor or a co-tenant's live process. It closes
        a window the session table cannot see: a joining session takes its lease
        inside ``provider.start``, but is registered as a live session only once
        start RETURNS, so for the whole cold start a co-tenant's reset would
        otherwise find no survivor and kill the shared process under it.

        Liveness is probed HERE rather than inferred from the registry. Dead
        entries are dropped by ``_drop_dead_locked``, which runs inside
        ``acquire`` -- so between a runtime's death and the next acquisition its
        entry is still registered with its leases intact. Counting one would
        report a dead process as a co-tenant and suppress both its reap and the
        sweep of the children that escaped it.

        Deliberately not locked. It only reads, there is no ``await`` in the
        scan, and the caller runs on the same event loop the registry is mutated
        from -- so it observes a consistent snapshot without being able to
        deadlock against a release in flight.
        """
        return self.leases_on_pid(pid) > 0

    def leases_on_pid(self, pid: int) -> int:
        """How many leases LIVE runtimes on *pid* hold, for the kill gate.

        A dead runtime's leases are not counted, and that is the difference
        between a gate and a lock. A process that has already exited cannot be
        harmed by the signal, while refusing to signal it suppresses the reap of
        its zombie and the sweep of the descendants that escaped its group --
        which is the leak the teardown exists to stop. So death releases
        ownership, and only a live process can be defended.
        """
        total = 0
        for entry in self._entries:
            if not entry.leases:
                continue
            if _pid_of(entry.runtime) != pid:
                continue
            if not self._alive(entry):
                continue
            total += len(entry.leases)
        return total

    def leases_on_runtime(self, runtime: object) -> int:
        """How many leases this exact runtime OBJECT holds, for the kill gate.

        Identity, not pid: it answers for the object the caller is about to kill
        even before that object has a pid, and it cannot be confused by a second
        runtime that later inherits the same number. Dead runtimes are excluded
        for the reason :meth:`leases_on_pid` gives.
        """
        for entry in self._entries:
            if entry.runtime is not runtime:
                continue
            if not entry.leases or not self._alive(entry):
                return 0
            return len(entry.leases)
        return 0

    # -- internals --

    def _pick_locked(self, key: Hashable, *, cap: int, session_key: str) -> _Entry | None:
        sticky = self._last_for_session.get(session_key)
        if (
            sticky is not None
            and any(e is sticky for e in self._entries)
            and sticky.key == key
            and sticky.has_room(cap)
            and self._alive(sticky)
        ):
            return sticky
        for entry in self._entries:
            if entry.key == key and entry.has_room(cap) and self._alive(entry):
                return entry
        return None

    @staticmethod
    def _alive(entry: _Entry) -> bool:
        probe = getattr(entry.runtime, "is_alive", None)
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception:
            return False

    def _drop_dead_locked(self) -> None:
        for entry in list(self._entries):
            if not self._alive(entry):
                orphaned = entry.session_keys()
                self._forget_entry_locked(entry)
                if orphaned:
                    logger.warning(
                        "runtime_ownership outcome=dead_runtime_dropped pid=%s sessions=%d",
                        _pid_of(entry.runtime),
                        len(orphaned),
                    )

    def _forget_entry_locked(self, entry: _Entry) -> None:
        for lease in list(entry.leases):
            if self._by_lease.get(lease) is entry:
                del self._by_lease[lease]
        for session_key in entry.session_keys():
            if self._last_for_session.get(session_key) is entry:
                del self._last_for_session[session_key]
        entry.leases.clear()
        self._entries = [e for e in self._entries if e is not entry]


#: The gateway's one runtime ownership table. A module-level singleton because
#: the ownership question is global: a kill gate can only refuse on behalf of a
#: lease that was taken in the same registry it consults.
RUNTIME_OWNERSHIP = RuntimeOwnership()


def _pid_of(target: object) -> int | None:
    """The pid *target* names, whether it IS one or merely carries one.

    Rejects everything that is not a real, positive, non-init pid. Test
    stand-ins are the sharp edge: a ``Mock`` attribute coerces to 1 through
    ``__index__``, and a pid of 1 or below also selects the ``kill(0)`` /
    ``kill(-n)`` process-group semantics rather than one process.
    """
    pid = target if isinstance(target, int) else getattr(target, "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return None
    return pid


def outstanding_leases(target: object) -> int:
    """Leases held on *target*, by object identity when it is a runtime.

    Identity first because it is the stronger answer and is available earlier: a
    runtime object is the thing the caller holds, while its pid can be absent
    before a spawn and can name a different process after an exit.
    """
    if not isinstance(target, int):
        held = RUNTIME_OWNERSHIP.leases_on_runtime(target)
        if held:
            return held
    pid = _pid_of(target)
    if pid is None:
        return 0
    return RUNTIME_OWNERSHIP.leases_on_pid(pid)


def note_runtime_kill(target: object, *, reason: str, caller: str) -> None:
    """Record that a runtime is about to be killed, and by whom.

    Called at the DECISION point, before anything is signalled, which is what
    makes it usable: the existing death log is written where the process is
    reaped, so it says a process died without saying who ended it.

    WARNING, and not debug chatter. The gateway runs at WARNING, so INFO here
    would mean the one line naming who fired is missing from the log of every
    deployment that has the problem.
    """
    logger.warning(
        "runtime kill pid=%s caller=%s reason=%s",
        _pid_of(target),
        caller,
        reason,
    )


def authorize_runtime_kill(target: object, *, reason: str, caller: str) -> bool:
    """The ONE ownership gate a runtime kill passes, and where the shot is logged.

    False means REFUSED: a live runtime still has leases outstanding, so the
    caller holds a pid it does not own and must not signal it. A caller that
    legitimately ends this runtime releases its lease first and is then
    authorized; one that signals without releasing is refusing its own teardown.

    Both outcomes are logged at WARNING -- the allow path through
    :func:`note_runtime_kill`, so there is one attribution implementation and the
    refusal is the only thing this adds.
    """
    held = outstanding_leases(target)
    if held:
        logger.warning(
            "runtime_ownership REFUSED kill pid=%s leases=%d caller=%s reason=%s: "
            "the process is still leased, so this caller must release before it signals",
            _pid_of(target),
            held,
            caller,
            reason,
        )
        return False
    note_runtime_kill(target, reason=reason, caller=caller)
    return True


#: Sentinel for "this object has no lease slot at all", distinct from a slot
#: holding None, which is a real holder that currently leases nothing.
_MISSING: object = object()


def _lease_holder(provider: object) -> object | None:
    """The object on *provider* that can hold a runtime lease, or None.

    Two shapes reach the kill paths: the runtime-backed session provider itself,
    and the outer provider that swaps one in as ``_client`` once startup
    completes. Duck-typed rather than imported by class, because this module sits
    below the ACP layer and the agent-SDK boundary check refuses knowledge of it,
    a type-only import included.

    A holder is recognised by its lease SLOT, not by having the two methods. Test
    stand-ins are the sharp edge here exactly as they are for pids: a ``MagicMock``
    answers every ``hasattr`` and returns a ``MagicMock`` from the call, which is
    not awaitable -- so a method-shaped check turns every mocked provider on these
    paths into a TypeError. A real holder's slot is ``None`` or the lease string.
    """
    for candidate in (provider, getattr(provider, "_client", None)):
        if candidate is None:
            continue
        slot = getattr(candidate, "_runtime_lease", _MISSING)
        if slot is not _MISSING and (slot is None or isinstance(slot, str)):
            return candidate
    return None


async def acquire_session_lease(provider: object) -> None:
    """Record a registered session's claim on its runtime. No-op for other shapes.

    Called where the session joins the registry, so the lease and registry
    membership mean the same thing. A provider with no runtime to lease -- a
    non-ACP backend, a placeholder client before startup swapped in the real one
    -- simply has no holder and is skipped.
    """
    holder = _lease_holder(provider)
    if holder is not None:
        await holder.acquire_runtime_lease()  # type: ignore[attr-defined]


async def release_session_lease(provider: object) -> None:
    """Give up a session's claim on its runtime. Idempotent; no-op for other shapes.

    For the paths that end a session WITHOUT a graceful ``shutdown`` -- a failure
    after registration, the dashboard's force-kill fallback. A path that shuts the
    provider down normally has already released inside ``shutdown``.
    """
    holder = _lease_holder(provider)
    if holder is not None:
        await holder.release_runtime_lease()  # type: ignore[attr-defined]


def _reset_for_tests() -> None:
    """Drop all ownership state. Tests only -- there is one registry per gateway."""
    RUNTIME_OWNERSHIP._entries.clear()
    RUNTIME_OWNERSHIP._by_lease.clear()
    RUNTIME_OWNERSHIP._last_for_session.clear()


__all__ = [
    "Acquisition",
    "CHAT_RUNTIME_CAP",
    "OwnedRuntime",
    "PidRefcount",
    "RUNTIME_OWNERSHIP",
    "RuntimeOwnership",
    "acquire_session_lease",
    "authorize_runtime_kill",
    "note_runtime_kill",
    "outstanding_leases",
    "release_session_lease",
]
