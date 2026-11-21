"""Runtime checks for the protocol properties P1, P2 and P4.

* ``TaskLedger`` is the only place a task's state changes. It rejects illegal transitions,
  which is how "a task runs at most once" is enforced: DONE has no way out.
* ``InvariantChecker`` cross-checks the ledger against the actual buffers, queues and
  in-flight sets of the nodes at every SAMPLE instant.

In strict mode (tests) the first violation raises. Otherwise violations are counted and the
first few are printed, so a long experiment still finishes and reports them.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from core.model import Task, TaskState

if TYPE_CHECKING:
    from core.node import CloudServer, ServerNode

log = logging.getLogger(__name__)

S = TaskState
_LEGAL: dict[TaskState, set[TaskState]] = {
    S.NEW: {S.BUFFERED, S.IN_FLIGHT},  # IN_FLIGHT: oversized task sent straight to the cloud
    S.BUFFERED: {S.IN_FLIGHT, S.QUEUED, S.BLOCKED, S.LOST},
    S.IN_FLIGHT: {S.QUEUED, S.BUFFERED, S.LOST},  # BUFFERED: rejected, or taken back
    S.QUEUED: {S.RUNNING, S.LOST},
    S.RUNNING: {S.DONE, S.LOST},
    S.DONE: set(),
    S.BLOCKED: set(),
    S.LOST: {S.BUFFERED},  # sender takes it back from a server that crashed holding it
}
LEGAL = [[dst in _LEGAL[src] for dst in TaskState] for src in TaskState]

NON_TERMINAL = (S.BUFFERED, S.IN_FLIGHT, S.QUEUED, S.RUNNING)


class InvariantViolation(AssertionError):
    """Raised in strict mode when P1, P2 or P4 breaks."""


class TaskLedger:
    """Tracks how many tasks are in each state and validates every transition."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.counts = [0] * len(TaskState)
        self.created = 0
        self.violations: Counter[str] = Counter()

    def register(self, task: Task) -> None:
        """Start tracking a freshly created task (state NEW)."""
        self.counts[S.NEW] += 1
        self.created += 1

    def move(self, task: Task, new: TaskState) -> None:
        """Change ``task.state`` to ``new`` if that transition is legal."""
        old = task.state
        if not LEGAL[old][new]:
            self.violation("P1", f"illegal transition {old.name} -> {new.name} for {task!r}")
            return
        self.counts[old] -= 1
        self.counts[new] += 1
        task.state = new

    def violation(self, prop: str, detail: str) -> None:
        """Record (or, in strict mode, raise) a violation of property ``prop``."""
        self.violations[prop] += 1
        if self.strict:
            raise InvariantViolation(f"{prop}: {detail}")
        if self.violations[prop] <= 5:
            log.warning("%s violated: %s", prop, detail)

    def count(self, state: TaskState) -> int:
        """Number of tasks currently in ``state``."""
        return self.counts[state]

    @property
    def outstanding(self) -> int:
        """Tasks that are not finished, blocked or lost yet."""
        return sum(self.counts[s] for s in NON_TERMINAL)

    @property
    def total_violations(self) -> int:
        """All violations recorded so far."""
        return sum(self.violations.values())


class InvariantChecker:
    """Checks P1, P2 and P4 against the real data structures of the nodes."""

    def __init__(
        self,
        ledger: TaskLedger,
        queue_capacity: int,
        max_hold: float,
        decide_interval: float,
    ) -> None:
        self.ledger = ledger
        self.queue_capacity = queue_capacity
        # rule 5 masks HOLD once a task is max_hold old, and the next decision instant
        # comes at most one interval later
        self.buffer_age_bound = max_hold + decide_interval + 1e-9

    def check(self, now: float, nodes: Sequence[ServerNode], cloud: CloudServer) -> None:
        """Run all three checks; problems go through ``ledger.violation``."""
        self._check_p1(nodes, cloud)
        self._check_p2(now, nodes)
        self._check_p4(nodes)

    def _check_p1(self, nodes: Sequence[ServerNode], cloud: CloudServer) -> None:
        led = self.ledger
        if sum(led.counts) != led.created:
            led.violation("P1", f"state counts sum to {sum(led.counts)}, created {led.created}")
        live = [nd for nd in nodes if not nd.crashed]
        buffered = sum(len(nd.buffer) + nd.tasks_in_rounds for nd in live)
        queued = sum(len(nd.queue) for nd in live) + len(cloud.queue)
        running = sum(nd.running is not None for nd in live) + (cloud.running is not None)
        pending = sum(len(o.pending) for nd in live for o in nd.inflight.values())
        expected = {S.BUFFERED: buffered, S.QUEUED: queued, S.RUNNING: running}
        for state, actual in expected.items():
            if led.count(state) != actual:
                led.violation(
                    "P1", f"{state.name}: ledger says {led.count(state)}, nodes hold {actual}"
                )
        # every IN_FLIGHT task must still be waiting for an answer at its sender
        if led.count(S.IN_FLIGHT) > pending + cloud.in_transit:
            led.violation("P1", f"{led.count(S.IN_FLIGHT)} tasks in flight but only {pending}")

    def _check_p2(self, now: float, nodes: Sequence[ServerNode]) -> None:
        for nd in nodes:
            if nd.crashed:
                continue
            for task in nd.buffer:
                if now - task.t_buffer > self.buffer_age_bound:
                    self.ledger.violation(
                        "P2", f"{task!r} has been in the buffer of {nd.idx} since {task.t_buffer}"
                    )

    def _check_p4(self, nodes: Sequence[ServerNode]) -> None:
        for nd in nodes:
            if nd.crashed:
                continue
            in_queue = len(nd.queue) + (nd.running is not None)
            if nd.q_count != in_queue or nd.q_count > self.queue_capacity:
                self.ledger.violation(
                    "P4", f"server {nd.idx} queue holds {in_queue} (counter {nd.q_count})"
                )
