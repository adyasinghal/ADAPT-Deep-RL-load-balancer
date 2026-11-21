"""B4: ML-enhanced EVBLB (Esmaeili et al., Comput. Commun. 222, 2024).

Two separate learners per server, as in the paper:

* **When** -- tabular Q-learning picks the dispatch interval (Algorithm 1). The state is the
  network-wide LBF rounded to 0.01 (101 states), the actions are the intervals
  {0.25, 0.5, 1, 2, 4} s and the reward is Eq. 9: theta (1 - LBF) + (1 - theta) dt / dt_max.
  Each time the chosen interval runs out the Q-table is updated with Eq. 8 and a new interval
  is picked epsilon-greedily.
* **Where** -- a gradient bandit over {itself} + its neighbours (Algorithm 3). It picks the arm
  with the highest softmax preference (Eq. 15) and learns from the outcome of the batch
  (Eq. 14): -1 if it was rejected, otherwise (thr - est_finish)^+ / thr with thr = 30 s, where
  est_finish comes back on the ACK.

It never polls; all it learns about neighbours is what its own batches experience.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.actions import FIRST_SEND, HOLD, LOCAL, N_SLOTS
from core.messages import MsgKind
from core.model import Topology
from core.node import NodeView
from policies.base import BasePolicy

INTERVALS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_LBF_STATES = 101
N_ARMS = 1 + N_SLOTS  # arm 0 = the server itself, arm 1 + s = neighbour slot s


class MLEvblb(BasePolicy):
    """Q-learning dispatch interval plus gradient-bandit neighbour choice, per server."""

    name = "ml_evblb"
    needs_poll = False
    max_hold = max(INTERVALS)

    def __init__(
        self,
        theta: float = 0.5,
        alpha: float = 0.2,
        gamma: float = 0.2,
        epsilon: float = 0.1,
        step: float = 0.1,
        threshold: float = 30.0,
    ) -> None:
        super().__init__()
        self.intervals = np.array(INTERVALS)
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.step = step
        self.threshold = threshold
        self.lbf = 0.0

    def reset(self, topology: Topology, rng: np.random.Generator) -> None:
        """Fresh Q-tables and bandits; every server starts with a random interval."""
        super().reset(topology, rng)
        n = topology.n
        self.q = np.zeros((n, N_LBF_STATES, len(self.intervals)))
        self.state = np.zeros(n, dtype=int)
        self.choice = rng.integers(len(self.intervals), size=n)
        self.started = np.zeros(n)
        self.pref = np.zeros((n, N_ARMS))
        self.mean_reward = np.zeros((n, N_ARMS))
        self.pulls = np.zeros((n, N_ARMS))
        self.exists = np.ones((n, N_ARMS), dtype=bool)
        self.exists[:, 1:] = topology.neighbours >= 0
        self.lbf = 0.0

    # ------------------------------------------------------------------ interval learner

    @staticmethod
    def lbf_state(lbf: float) -> int:
        """Nearest of the 101 LBF states 0.00, 0.01, ..., 1.00 (paper Eq. 10)."""
        return int(round(min(max(lbf, 0.0), 1.0) * 100))

    def interval_reward(self, lbf: float, dt: float) -> float:
        """Paper Eq. 9."""
        return self.theta * (1.0 - lbf) + (1.0 - self.theta) * dt / self.intervals.max()

    def q_update(self, i: int, s: int, a: int, r: float, s_next: int) -> None:
        """Paper Eq. 8: Q <- (1 - alpha) Q + alpha (r + gamma max_a' Q(s', a'))."""
        target = r + self.gamma * self.q[i, s_next].max()
        self.q[i, s, a] = (1.0 - self.alpha) * self.q[i, s, a] + self.alpha * target

    def pick_interval(self, i: int, s: int) -> int:
        """Epsilon-greedy over the intervals (random tie-break)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.intervals)))
        row = self.q[i, s]
        return int(self.rng.choice(np.flatnonzero(row == row.max())))

    # ------------------------------------------------------------------ neighbour bandit

    def probabilities(self, i: int) -> np.ndarray:
        """Softmax of the preferences over the arms that exist (paper Eq. 15)."""
        h = np.where(self.exists[i], self.pref[i], -np.inf)
        e = np.exp(h - h.max())
        return e / e.sum()

    def pick_arm(self, i: int, mask: np.ndarray) -> int:
        """Most probable allowed arm (Algorithm 3, line 9), ties broken at random."""
        allowed = np.zeros(N_ARMS, dtype=bool)
        allowed[0] = mask[LOCAL]
        allowed[1:] = mask[FIRST_SEND:]
        p = np.where(allowed, self.probabilities(i), -1.0)
        return int(self.rng.choice(np.flatnonzero(p == p.max())))

    def batch_reward(self, rejected: bool, est_finish: float) -> float:
        """Paper Eq. 14."""
        if rejected:
            return -1.0
        return max(self.threshold - est_finish, 0.0) / self.threshold

    def bandit_update(self, i: int, arm: int, reward: float) -> None:
        """Paper Eq. 16 for the chosen arm, then update its running mean reward.

        The other arms keep their preference: Algorithm 3 sets their reward to their own
        average (line 14), so their (R - mean) term in Eq. 16 is zero.
        """
        pi = self.probabilities(i)[arm]
        self.pref[i, arm] += self.step * (reward - self.mean_reward[i, arm]) * pi
        self.pulls[i, arm] += 1
        self.mean_reward[i, arm] += (reward - self.mean_reward[i, arm]) / self.pulls[i, arm]

    # ------------------------------------------------------------------ policy API

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list[NodeView]) -> np.ndarray:
        """Advance each server's interval clock and dispatch where an interval ended."""
        actions = np.full(len(masks), HOLD, dtype=np.int64)
        if not ctx:
            return actions
        now = ctx[0].now
        for i, view in enumerate(ctx):
            due = now - self.started[i] >= self.intervals[self.choice[i]] - 1e-9
            if due:
                s_next = self.lbf_state(self.lbf)
                dt = self.intervals[self.choice[i]]
                reward = self.interval_reward(self.lbf, dt)
                self.q_update(i, self.state[i], self.choice[i], reward, s_next)
                self.state[i] = s_next
                self.choice[i] = self.pick_interval(i, s_next)
                self.started[i] = now
            m = masks[i]
            if not m[LOCAL]:
                continue  # nothing buffered
            if m[HOLD] and not due:
                continue  # keep collecting until the interval ends (or rule 5 forces us)
            arm = self.pick_arm(i, m)
            if arm == 0:
                actions[i] = LOCAL
                # the server knows its own queue, so the local arm is rewarded right away
                self.bandit_update(i, 0, self.batch_reward(False, view.est_finish()))
            else:
                actions[i] = FIRST_SEND + arm - 1
        return actions

    def observe(
        self, rewards: np.ndarray, next_obs: np.ndarray, done: bool, info: dict[str, Any]
    ) -> None:
        """Pick up the global LBF and the ACK/REJECT outcomes of the last interval."""
        self.lbf = float(info.get("lbf", self.lbf))
        for node, slot, kind, _, est in info.get("replies", ()):
            rejected = kind == MsgKind.REJECT
            self.bandit_update(node, 1 + slot, self.batch_reward(rejected, est))
