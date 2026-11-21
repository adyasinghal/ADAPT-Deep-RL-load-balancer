"""B1: nearest only. Every task runs on its Voronoi home server; no balancing at all."""

from __future__ import annotations

import numpy as np

from core.actions import HOLD, LOCAL
from policies.base import BasePolicy


class Nearest(BasePolicy):
    """LOCAL at every decision instant where the buffer is non-empty."""

    name = "nearest"
    max_hold = 0.25

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list) -> np.ndarray:
        """LOCAL wherever it's allowed (i.e. the buffer is non-empty), else HOLD."""
        return np.where(masks[:, LOCAL], LOCAL, HOLD)
