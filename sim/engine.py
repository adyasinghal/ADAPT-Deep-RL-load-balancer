"""Discrete-event core: a clock and a heap of timed callbacks.

Events are ``(time, seq, callback, args)`` tuples; ``seq`` keeps events at the same time in the
order they were scheduled, which makes runs deterministic. The event types of the brief map
onto callbacks: message delivery, service completion, end of a dispatch round and OFFER
timeouts are scheduled by the nodes and the transport, while the periodic DECIDE / SAMPLE
instants and crashes are driven by the env between ``run_until`` calls.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any


class Engine:
    """Simulation clock plus event queue. Also serves as the ``Clock`` the nodes use."""

    def __init__(self) -> None:
        self.now = 0.0
        self._heap: list[tuple[float, int, Callable[..., Any], tuple[Any, ...]]] = []
        self._seq = 0
        self.events_run = 0

    def call_at(self, t: float, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` at simulated time ``t`` (which must not be in the past)."""
        if t < self.now:
            raise ValueError(f"event scheduled in the past: {t} < {self.now}")
        heapq.heappush(self._heap, (t, self._seq, fn, args))
        self._seq += 1

    def call_later(self, delay: float, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` after ``delay`` seconds."""
        self.call_at(self.now + delay, fn, *args)

    def run_until(self, t_end: float) -> None:
        """Process every event with time <= ``t_end`` and leave the clock at ``t_end``."""
        heap = self._heap
        pop = heapq.heappop
        while heap and heap[0][0] <= t_end:
            t, _, fn, args = pop(heap)
            self.now = t
            fn(*args)
            self.events_run += 1
        self.now = t_end

    @property
    def pending(self) -> int:
        """Number of events still waiting."""
        return len(self._heap)
