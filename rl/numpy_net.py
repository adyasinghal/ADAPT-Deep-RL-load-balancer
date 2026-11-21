"""Numpy-only forward pass of the ADAPT Q-network (no torch needed).

Used for evaluation in the experiment workers and by the demo. The weight file is written by
``rl.networks.export_npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MASKED_Q = -1e9  # value given to masked actions before any argmax / softmax


class NumpyQNet:
    """Forward pass of ``DuelingQNet`` in numpy, from an ``export_npz`` file."""

    def __init__(self, path: str | Path) -> None:
        with np.load(path) as data:
            w = {k: data[k].astype(np.float32) for k in data.files}
        self.w1, self.b1 = w["body_0_weight"].T, w["body_0_bias"]
        self.w2, self.b2 = w["body_2_weight"].T, w["body_2_bias"]
        self.wv, self.bv = w["value_weight"].T, w["value_bias"]
        self.wa, self.ba = w["advantage_weight"].T, w["advantage_bias"]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for a batch of observations, shape (batch, n_actions)."""
        h = np.maximum(obs @ self.w1 + self.b1, 0.0)
        h = np.maximum(h @ self.w2 + self.b2, 0.0)
        adv = h @ self.wa + self.ba
        return h @ self.wv + self.bv + adv - adv.mean(axis=1, keepdims=True)


def greedy(q: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Best allowed action per row."""
    return np.where(masks, q, MASKED_Q).argmax(axis=1)
