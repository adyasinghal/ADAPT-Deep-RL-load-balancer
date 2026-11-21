"""Experience replay shared by all servers (centralised training, one buffer).

Transitions are added a whole decision instant at a time: one row per server.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.actions import N_ACTIONS, N_FEATURES


@dataclass
class Batch:
    """A sampled mini-batch (numpy arrays)."""

    obs: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    next_mask: np.ndarray


class ReplayBuffer:
    """Fixed-size ring buffer of (s, mask, a, r, s', mask') rows.

    Episodes end on a time limit, never in a true terminal state, so there is no ``done``
    flag: every transition bootstraps from its next state.
    """

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, N_FEATURES), dtype=np.float32)
        self.next_obs = np.zeros((capacity, N_FEATURES), dtype=np.float32)
        self.mask = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.next_mask = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, mask, action, reward, next_obs, next_mask) -> None:
        """Append a block of rows (one per server), overwriting the oldest when full."""
        k = len(action)
        idx = (self._pos + np.arange(k)) % self.capacity
        self.obs[idx] = obs
        self.mask[idx] = mask
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_obs[idx] = next_obs
        self.next_mask[idx] = next_mask
        self._pos = int((self._pos + k) % self.capacity)
        self.size = min(self.size + k, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Uniform sample with replacement."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        return Batch(
            self.obs[idx],
            self.mask[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.next_mask[idx],
        )
