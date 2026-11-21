"""Run one full episode of a policy in the env and collect its metrics."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from sim.env import EdgeEnv


def run_episode(
    cfg: dict[str, Any], policy: Any, seed: int, strict: bool | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate ``policy`` for one episode and return (time series, summary)."""
    env = EdgeEnv(cfg, policy, strict=strict)
    obs, info = env.reset(seed)
    policy.reset(env.topology, env.policy_rng)
    started = time.perf_counter()
    done = False
    while not done:
        actions = policy.act(obs, info["masks"], env.views)
        obs, rewards, done, info = env.step(actions)
        policy.observe(rewards, obs, done, info)
    summary = env.summary()
    summary["wall_clock"] = time.perf_counter() - started
    summary["sim_end"] = env.now
    return env.timeseries(), summary
