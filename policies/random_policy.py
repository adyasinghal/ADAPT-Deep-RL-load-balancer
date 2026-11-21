"""A policy that picks uniformly among the allowed actions.

Used by the property tests: the protocol's safety must not depend on the policy being sensible.
"""

from __future__ import annotations

import numpy as np

from policies.base import BasePolicy


class RandomPolicy(BasePolicy):
    """Uniformly random over each server's mask."""

    name = "random"

    def __init__(self, max_hold: float = 2.0) -> None:
        super().__init__()
        self.max_hold = max_hold

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list) -> np.ndarray:
        """Draw a random score per action, hide the masked ones and take the best."""
        scores = self.rng.random(masks.shape)
        scores[~masks] = -1.0
        return scores.argmax(axis=1)
