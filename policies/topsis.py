"""B7: TOPSIS server selection, as in TOLB (Wu et al., Electronics 13, 2024).

Like EVBLB it collects tasks for 2 s and polls its neighbours, but instead of taking the
candidate with the most room left it scores every candidate on two criteria with TOPSIS:
remaining resources P_rem (more is better) and distance (less is better). Unlike EVBLB this
weighs distance, so it is a non-learned rule with the same trade-off ADAPT's distance term
targets.

TOLB derives the criterion weights with AHP; we use fixed weights (0.7, 0.3) instead.
"""

from __future__ import annotations

import numpy as np

from core.actions import FIRST_SEND, LOCAL
from core.node import NodeView
from policies.evblb import EVBLB


def topsis_scores(
    benefit: np.ndarray, cost: np.ndarray, weights: tuple[float, float] = (0.7, 0.3)
) -> np.ndarray:
    """TOPSIS closeness score S of each candidate (TOLB Eqs. 25-32); higher is better.

    ``benefit`` and ``cost`` hold one value per candidate. A criterion on which every
    candidate is equal cannot tell them apart and is set to 1 for all of them.
    """
    columns = []
    for values, larger_is_better in ((benefit, True), (cost, False)):
        values = np.asarray(values, dtype=float)
        span = values.max() - values.min()
        if span == 0:
            columns.append(np.ones_like(values))
        elif larger_is_better:
            columns.append((values - values.min()) / span)  # Eq. 25
        else:
            columns.append((values.max() - values) / span)  # Eq. 26
    weighted = np.column_stack(columns) * np.asarray(weights)  # Eq. 27
    best, worst = weighted.max(axis=0), weighted.min(axis=0)  # Eqs. 28-29
    to_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))  # Eq. 30
    to_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))  # Eq. 31
    total = to_best + to_worst
    return np.divide(to_worst, total, out=np.ones_like(total), where=total > 0)  # Eq. 32


class Topsis(EVBLB):
    """EVBLB's 2 s interval and poll round, with the target picked by TOPSIS."""

    def __init__(self, dt: float = 2.0, weights: tuple[float, float] = (0.7, 0.3)) -> None:
        super().__init__(dt=dt)
        self.weights = tuple(weights)
        self.name = "topsis"

    def best_target(self, view: NodeView, mask: np.ndarray) -> int:
        """Highest TOPSIS score among {self} and the allowed neighbours; ties go to self."""
        actions = [LOCAL]
        prem = [view.p_rem()]
        dist = [0.0]
        for slot, (p, d) in enumerate(
            zip(view.true_neighbour_prem(), view.neighbour_dist, strict=True)
        ):
            if p is not None and mask[FIRST_SEND + slot]:
                actions.append(FIRST_SEND + slot)
                prem.append(p)
                dist.append(d)
        if len(actions) == 1:
            return LOCAL
        scores = topsis_scores(np.array(prem), np.array(dist), self.weights)
        return actions[int(np.argmax(scores))]
