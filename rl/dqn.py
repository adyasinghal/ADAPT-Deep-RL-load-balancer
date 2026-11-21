"""Double DQN with action masking, shared by every server.

* Huber loss on the TD error, Adam, soft target updates (tau).
* Double DQN target: the online net picks the best *allowed* next action, the target net
  scores it. Masking the next-state argmax matters: otherwise the target could be built on an
  action the protocol would never allow.
* Exploration is epsilon-greedy per server, uniformly over that server's allowed actions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from rl.networks import DuelingQNet, export_npz, greedy, masked
from rl.replay import Batch


@dataclass(frozen=True)
class DQNConfig:
    """Learner hyperparameters (the ``dqn`` section of ``config/train.yaml``)."""

    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 256
    tau: float = 0.005
    hidden: int = 128
    huber_delta: float = 1.0
    grad_clip: float = 10.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DQNConfig:
        """Pick the known keys out of a config dict.

        Values are cast to the default's type because YAML reads ``1e-3`` as a string.
        """
        fields = cls.__dataclass_fields__
        kwargs = {}
        for key, value in d.items():
            if key in fields:
                kwargs[key] = type(fields[key].default)(value)
        return cls(**kwargs)


def to_tensors(batch: Batch) -> tuple[torch.Tensor, ...]:
    """Numpy batch -> torch tensors (no copies where possible)."""
    return (
        torch.from_numpy(batch.obs),
        torch.from_numpy(batch.mask),
        torch.from_numpy(batch.action),
        torch.from_numpy(batch.reward),
        torch.from_numpy(batch.next_obs),
        torch.from_numpy(batch.next_mask),
    )


class DQNAgent:
    """Online and target networks plus the optimiser."""

    def __init__(self, cfg: DQNConfig, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.online = DuelingQNet(hidden=cfg.hidden)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.updates = 0

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values of the online network for a batch of observations."""
        return self.online(torch.from_numpy(obs)).numpy()

    def act(
        self, obs: np.ndarray, masks: np.ndarray, epsilon: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Epsilon-greedy over each server's allowed actions (one batched forward pass)."""
        actions = greedy(self.q_values(obs), masks)
        explore = rng.random(len(actions)) < epsilon
        if explore.any():
            scores = rng.random(masks.shape)
            scores[~masks] = -1.0
            actions[explore] = scores.argmax(axis=1)[explore]
        return actions

    # ------------------------------------------------------------------ learning

    def td_loss(self, batch: Batch) -> torch.Tensor:
        """Huber loss between Q(s, a) and the Double DQN target."""
        obs, _, action, reward, next_obs, next_mask = to_tensors(batch)
        q = self.online(obs).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            next_action = masked(self.online(next_obs), next_mask).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_action).squeeze(1)
            target = reward + self.cfg.gamma * next_q
        return F.smooth_l1_loss(q, target, beta=self.cfg.huber_delta)

    def step(self, loss: torch.Tensor) -> float:
        """One optimiser step on ``loss`` followed by a soft target update."""
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.soft_update()
        self.updates += 1
        return float(loss.item())

    def update(self, batch: Batch) -> float:
        """One DQN gradient step."""
        return self.step(self.td_loss(batch))

    @torch.no_grad()
    def soft_update(self) -> None:
        """target <- tau * online + (1 - tau) * target."""
        tau = self.cfg.tau
        for p, tp in zip(self.online.parameters(), self.target.parameters(), strict=True):
            tp.mul_(1.0 - tau).add_(p, alpha=tau)

    def sync_target(self) -> None:
        """Copy the online weights into the target network."""
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------ saving

    def save(self, stem: str | Path, meta: dict[str, Any] | None = None) -> None:
        """Write ``<stem>.pt`` (torch, for further training) and ``<stem>.npz`` (numpy)."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.online.state_dict(), "meta": meta or {}}, f"{stem}.pt")
        export_npz(self.online, f"{stem}.npz")

    def load(self, path: str | Path) -> None:
        """Load online weights from a ``.pt`` file and sync the target net."""
        state = torch.load(path, weights_only=False)
        self.online.load_state_dict(state["state_dict"])
        self.sync_target()


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Epsilon that falls linearly from ``start`` to ``end`` over ``decay_steps`` steps."""
    if decay_steps <= 0:
        return end
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)
