"""Who held a pid, and who ended a runtime.

Two things live here, and both are about a process having more than one
interested party:

:class:`PidRefcount`
    A pid shield that counts its holders. Several independent holders shield one
    pid, each pairing its own register with its own unregister, so a shield that
    merely LISTS pids is torn off by the first holder to leave while the others
    are still using the process.

:func:`note_runtime_kill`
    The one place a runtime kill is attributed. It records the pid, the call site
    and the intent before anything is signalled.

What is deliberately NOT here yet
---------------------------------
A lease table, and a gate that refuses a kill while a runtime is still leased.
Both are wanted -- a session should hold a lease rather than a pid, so one
session's teardown cannot end a process another session is running on -- and
neither can land on its own:

* a lease table with no one acquiring is inert machinery, and the refusal it
  exists for can never execute;
* a gate that DOES refuse, wired to a provider that holds a lease for its
  lifetime, refuses the force-kill paths as well -- ``_sync_kill_provider`` is
  reached through ``_dispatch_hard_kill`` and the dashboard's reset-all fallback,
  and neither releases anything -- so the process it declines to signal leaks
  instead.

So the table, the gate, the provider's acquire and release, and every kill path
releasing before it signals are ONE unit, and they land together. This change
lands the part that stands alone: the counted shield, and the attribution line
that says who fired.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator, MutableSet

logger = logging.getLogger(__name__)


class PidRefcount(MutableSet[int]):
    """A pid shield that counts its holders instead of merely listing them.

    A ``set`` of shielded pids is wrong as soon as two holders shield one pid:
    the first to leave calls ``discard`` and tears the shield off a process the
    second is still using. This counts, so a pid stays shielded until the LAST
    holder drops it, and holders stay independent.

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


def note_runtime_kill(target: object, *, reason: str, caller: str) -> None:
    """Record that a runtime is about to be killed, and by whom.

    Called at the DECISION point, before anything is signalled, which is what
    makes it usable: the existing death log is written where the process is
    reaped, so it says a process died without saying who ended it. A session
    whose process was killed under it reports only that the process died, and
    nothing correlates that with the teardown that caused it.

    WARNING, and not debug chatter. The gateway runs at WARNING, so INFO here
    would mean the one line naming who fired is missing from the log of every
    deployment that has the problem.

    *caller* names the call SITE and *reason* the intent. Both are for the human
    reading the log afterwards, so name the site rather than the module.
    """
    logger.warning(
        "runtime kill pid=%s caller=%s reason=%s",
        _pid_of(target),
        caller,
        reason,
    )


__all__ = ["PidRefcount", "note_runtime_kill"]
