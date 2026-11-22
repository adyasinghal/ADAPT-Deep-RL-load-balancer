"""Tests for the env API."""

import numpy as np
import pytest

from core.actions import HOLD, LOCAL, N_ACTIONS, N_FEATURES
from core.config import load_config
from policies import make_policy
from sim.env import EdgeEnv
from sim.rollout import run_episode


def small_cfg(**dotted):
    base = {"topology.n_servers": 10, "run.duration": 30.0, "run.arrival_cutoff": 5.0}
    base.update(dotted)
    return load_config(dotted=base)


def test_real_env_shapes_and_first_observation():
    env = EdgeEnv(small_cfg(), make_policy({"name": "evblb"}), strict=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (10, N_FEATURES) and obs.dtype == np.float32
    assert info["masks"].shape == (10, N_ACTIONS)
    # nothing has arrived at t = 0, so HOLD is the only option everywhere
    assert info["masks"][:, HOLD].all() and not info["masks"][:, 1:].any()
    obs, rewards, done, info = env.step(np.zeros(10, dtype=int))
    assert rewards.shape == (10,) and info["reward_parts"].shape == (10, 5)
    assert info["time"] == 0.25


def test_masks_follow_the_protocol_rules():
    policy = make_policy({"name": "random"})
    env = EdgeEnv(small_cfg(), policy, strict=True)
    obs, info = env.reset(seed=1)
    policy.reset(env.topology, env.policy_rng)
    for _ in range(60):
        masks = info["masks"]
        for i, view in enumerate(env.views):
            nonempty = view.buffer_len > 0
            assert masks[i, LOCAL] == nonempty
            assert masks[i, HOLD] or nonempty
            nbrs = view.neighbours
            for s in range(5):
                if nbrs[s] < 0:
                    assert not masks[i, 2 + s]
        # features stay in their documented ranges
        assert (obs[:, :10] >= 0).all() and (obs[:, :10] <= 1.0001).all()
        obs, _, _, info = env.step(policy.act(obs, masks, env.views))


def test_forbidden_action_raises():
    env = EdgeEnv(small_cfg(), make_policy({"name": "evblb"}))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.full(10, LOCAL))  # buffers are empty at t = 0


def test_same_seed_same_rollout():
    cfg = small_cfg()
    ts_a, sum_a = run_episode(cfg, make_policy({"name": "evblb", "dt": 1.0}), seed=7)
    ts_b, sum_b = run_episode(cfg, make_policy({"name": "evblb", "dt": 1.0}), seed=7)
    assert ts_a.equals(ts_b)
    sum_a.pop("wall_clock"), sum_b.pop("wall_clock")
    assert sum_a == sum_b


def test_short_run_has_no_violations_and_drains():
    _, summary = run_episode(small_cfg(), make_policy({"name": "evblb"}), seed=3, strict=True)
    assert summary["violations"] == 0
    assert summary["unfinished"] == 0
    assert summary["completed"] + summary["blocked"] == summary["created"]


def test_tidal_workload_runs():
    cfg = small_cfg(**{"workload.name": "W2", "run.duration": 320.0})
    _, summary = run_episode(cfg, make_policy({"name": "nearest"}), seed=2, strict=True)
    assert summary["created"] > 0 and summary["violations"] == 0
