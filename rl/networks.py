"""The ADAPT Q-network: a small MLP with a dueling head.

    30 features -> 128 -> 128 -> V (1) and A (7);  Q = V + A - mean(A)

Training uses the torch module. For evaluation and the demo the weights are exported to a
plain ``.npz`` file and run by ``rl.numpy_net.NumpyQNet``, so neither needs torch installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.actions import N_ACTIONS, N_FEATURES
from rl.numpy_net import MASKED_Q, NumpyQNet, greedy

__all__ = ["DuelingQNet", "NumpyQNet", "export_npz", "greedy", "masked"]


class DuelingQNet(nn.Module):
    """Shared Q-network every server runs a copy of."""

    def __init__(self, n_features: int = N_FEATURES, n_actions: int = N_ACTIONS, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Q-values, shape (batch, n_actions)."""
        h = self.body(obs)
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(dim=1, keepdim=True)


def masked(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the Q-values of forbidden actions by a huge negative number."""
    return q.masked_fill(~mask, MASKED_Q)


def export_npz(net: DuelingQNet, path: str | Path) -> None:
    """Save the weights as plain numpy arrays for ``NumpyQNet``."""
    arrays = {
        name.replace(".", "_"): p.detach().cpu().numpy() for name, p in net.named_parameters()
    }
    np.savez(path, **arrays)
