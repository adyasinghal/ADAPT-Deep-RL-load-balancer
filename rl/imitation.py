"""EVBLB-guided warm start.

1. Run an expert, EVBLB written in ADAPT's action space, inside ADAPT's own env (no polls;
   the expert reads the true loads through ``ctx``), and record its transitions.
2. Behaviour cloning: train the Q-network so that the expert's action has the highest
   (masked) Q-value, using the Q-values as logits of a cross-entropy loss.
3. The recorded transitions also go into the replay buffer that DQN fine-tuning starts from.

The BC loss also includes the normal TD loss on the same batch. Cross-entropy alone only fixes
the *ranking* of the Q-values and leaves their scale arbitrary, so the first DQN updates would
wipe the ranking out again; with the TD term the cloned network's values are already on the
right scale when fine-tuning starts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from core.actions import HOLD, LOCAL
from core.node import NodeView
from policies.base import BasePolicy
from policies.evblb import EVBLB
from rl.dqn import DQNAgent, to_tensors
from rl.networks import masked
from rl.replay import ReplayBuffer
from sim.env import EdgeEnv


class EvblbExpert(BasePolicy):
    """EVBLB in ADAPT's action space: HOLD until ``dt`` since the last dispatch, then send the
    whole buffer to the allowed candidate with the largest true P_rem."""

    name = "evblb_expert"

    def __init__(self, dt: float = 2.0) -> None:
        super().__init__()
        self.dt = dt

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list[NodeView]) -> np.ndarray:
        """Per-server timer; respects whatever masks it is given (ADAPT's, here)."""
        actions = np.full(len(masks), HOLD, dtype=np.int64)
        for i, view in enumerate(ctx):
            m = masks[i]
            if not m[LOCAL]:
                continue
            due = view.now - view.last_dispatch >= self.dt - 1e-9
            if due or not m[HOLD]:
                actions[i] = EVBLB.best_target(view, m)
        return actions


def collect_expert(
    make_env: Callable[[int], tuple[EdgeEnv, int]],
    adapt: Any,
    expert: BasePolicy,
    episodes: int,
    replay: ReplayBuffer,
) -> list[dict[str, float]]:
    """Run ``expert`` for ``episodes`` episodes and push every transition into ``replay``.

    ``make_env(episode)`` returns a fresh env and the seed to reset it with; ``adapt`` is the
    ADAPT policy object whose masks (e.g. the fixed-interval variant's) the expert must obey.
    """
    stats = []
    for ep in range(episodes):
        env, seed = make_env(ep)
        obs, info = env.reset(seed)
        expert.reset(env.topology, env.policy_rng)
        mask = adapt.effective_masks(info["masks"], env.views)
        rewards = []
        done = False
        while not done:
            action = expert.act(obs, mask, env.views)
            next_obs, reward, done, info = env.step(action)
            next_mask = adapt.effective_masks(info["masks"], env.views)
            replay.add(obs, mask, action, reward, next_obs, next_mask)
            rewards.append(float(reward.mean()))
            obs, mask = next_obs, next_mask
        summary = env.summary()
        stats.append({"reward": float(np.mean(rewards)), "resp_mean": summary["resp_mean"]})
    return stats


def behaviour_clone(
    agent: DQNAgent, replay: ReplayBuffer, steps: int, batch_size: int, td_weight: float = 1.0
) -> list[float]:
    """Fit the Q-network to the expert's actions; returns the loss of every step."""
    losses = []
    for _ in range(steps):
        batch = replay.sample(batch_size)
        obs, mask, action, *_ = to_tensors(batch)
        logits = masked(agent.online(obs), mask)
        loss = F.cross_entropy(logits, action) + td_weight * agent.td_loss(batch)
        losses.append(agent.step(loss))
    agent.sync_target()
    return losses


@torch.no_grad()
def agreement(agent: DQNAgent, replay: ReplayBuffer, n: int = 20_000) -> float:
    """Share of recorded states with a real choice where the greedy action equals the expert's."""
    batch = replay.sample(n)
    choice = batch.mask.sum(axis=1) > 1
    q = agent.q_values(batch.obs[choice])
    picked = np.where(batch.mask[choice], q, -np.inf).argmax(axis=1)
    return float((picked == batch.action[choice]).mean())
