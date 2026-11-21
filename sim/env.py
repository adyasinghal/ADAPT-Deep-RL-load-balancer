"""Multi-agent environment API used by the RL code and by the experiment runner.

Interface (agreed between the simulator and the agent workstreams)::

    env = EdgeEnv(cfg, policy)          # or StubEnv(cfg) while the simulator is not ready
    obs, info = env.reset(seed)         # obs: (n, 30) float32, info["masks"]: (n, 7) bool
    obs, rewards, done, info = env.step(actions)   # actions: (n,) int, rewards: (n,) float32

``info`` always has ``masks`` (what each server may do next) and ``time``. ``env.views`` is a
list of read-only node views passed to ``Policy.act`` as ``ctx``.

Run ``python -m sim.env --stub --steps 100`` to exercise the stub.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from core.actions import HOLD, LOCAL, N_ACTIONS, N_FEATURES
from core.config import load_config


class StubEnv:
    """Random stand-in for the real env with exactly the same shapes and API.

    Lets the agent code be written and smoke-tested before the simulator exists. Observations
    and rewards are noise; masks are random but HOLD and LOCAL are always allowed so every row
    has a legal action.
    """

    def __init__(self, cfg: dict[str, Any], horizon: int = 400) -> None:
        self.n = int(cfg["topology"]["n_servers"])
        self.dt = float(cfg["run"]["decide_interval"])
        self.horizon = horizon
        self.views: list = []
        self._rng = np.random.default_rng(0)
        self._step = 0
        self._masks = np.ones((self.n, N_ACTIONS), dtype=bool)

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode; the same seed gives the same sequence."""
        self._rng = np.random.default_rng(seed)
        self._step = 0
        return self._obs(), self._info()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        """Check the actions against the current masks and return random feedback."""
        actions = np.asarray(actions)
        if actions.shape != (self.n,):
            raise ValueError(f"expected {self.n} actions, got shape {actions.shape}")
        illegal = ~self._masks[np.arange(self.n), actions]
        if illegal.any():
            raise ValueError(f"masked actions chosen by servers {np.flatnonzero(illegal)}")
        self._step += 1
        rewards = -self._rng.random(self.n).astype(np.float32)
        done = self._step >= self.horizon
        return self._obs(), rewards, done, self._info()

    def _obs(self) -> np.ndarray:
        return self._rng.random((self.n, N_FEATURES), dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        self._masks = self._rng.random((self.n, N_ACTIONS)) < 0.6
        self._masks[:, HOLD] = True
        self._masks[:, LOCAL] = True
        return {"masks": self._masks.copy(), "time": self._step * self.dt}


def _run_stub(steps: int, seed: int) -> None:
    from policies.random_policy import RandomPolicy

    cfg = load_config()
    env = StubEnv(cfg, horizon=steps)
    policy = RandomPolicy()
    policy.reset(None, np.random.default_rng(seed))
    obs, info = env.reset(seed)
    total = 0.0
    done = False
    while not done:
        actions = policy.act(obs, info["masks"], env.views)
        obs, rewards, done, info = env.step(actions)
        total += float(rewards.mean())
    print(f"stub env: n={env.n} obs={obs.shape} masks={info['masks'].shape}")
    print(f"ran {steps} steps to t={info['time']:.2f}s, mean reward {total / steps:.3f}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stub", action="store_true", help="run the random stub env")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stub:
        _run_stub(args.steps, args.seed)
    else:
        parser.error("only --stub is available so far")


if __name__ == "__main__":
    main()
