"""EVBLB (Sohrabi et al., GLOBECOM 2021) and its variants B2, B3 and B6.

Each server collects the requests of its Voronoi cell for ``dt`` seconds and then sends the
whole batch to whichever of {itself} + its neighbours has the most room left (largest P_rem,
EVBLB Algorithm 1 line 10). It needs fresh neighbour loads to do that, so every dispatch first
pays a poll round. Batches go out on the global ``dt`` grid (t = dt, 2 dt, ...), as in the
paper's episodic intervals; a server dispatches early only when rule 5 forces it to.

* B2: ``EVBLB(dt=2.0)``
* B3: ``EVBLB(dt)`` for dt in {0.25, 0.5, 1, 4}
* B6: ``PerfectInfo()``: dispatch every 0.25 s with free, instant load information
"""

from __future__ import annotations

import numpy as np

from core.actions import FIRST_SEND, HOLD, LOCAL
from core.node import NodeView
from policies.base import BasePolicy


class EVBLB(BasePolicy):
    """Fixed-interval EVBLB; picks the target with the most room after polling."""

    needs_poll = True

    def __init__(self, dt: float = 2.0) -> None:
        super().__init__()
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.max_hold = self.dt
        self.name = f"evblb_dt{self.dt:g}"

    def due(self, now: float) -> bool:
        """True at the end of every ``dt`` interval."""
        k = now / self.dt
        return now > 0 and abs(k - round(k)) < 1e-9

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list[NodeView]) -> np.ndarray:
        """Dispatch on the grid (or when forced), otherwise HOLD."""
        actions = np.full(len(masks), HOLD, dtype=np.int64)
        if not ctx:
            return actions
        due = self.due(ctx[0].now)
        for i, view in enumerate(ctx):
            m = masks[i]
            if not m[LOCAL]:
                continue  # empty buffer (or crashed)
            if due or not m[HOLD]:
                actions[i] = self.best_target(view, m)
        return actions

    @staticmethod
    def best_target(view: NodeView, mask: np.ndarray) -> int:
        """LOCAL or SEND_j for the allowed candidate with the largest true P_rem.

        Ties go to the server itself, then to the nearer neighbour (lower slot).
        """
        best_action, best = LOCAL, view.p_rem()
        for slot, p in enumerate(view.true_neighbour_prem()):
            action = FIRST_SEND + slot
            if p is not None and mask[action] and p > best:
                best_action, best = action, p
        return best_action


class PerfectInfo(EVBLB):
    """B6: EVBLB that dispatches every 0.25 s and gets neighbour loads for free.

    Only the poll is free; the dispatch round itself still costs the usual amount.
    """

    needs_poll = False

    def __init__(self) -> None:
        super().__init__(dt=0.25)
        self.name = "perfect"
