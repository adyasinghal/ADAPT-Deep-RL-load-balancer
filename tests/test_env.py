"""Tests for the env API (stub for now)."""

import numpy as np
import pytest

from core.actions import N_ACTIONS, N_FEATURES
from core.config import load_config
from policies.random_policy import RandomPolicy
from sim.env import StubEnv


def make_stub(n=8):
    cfg = load_config(dotted={"topology.n_servers": n})
    return StubEnv(cfg, horizon=20)


def test_stub_shapes():
    env = make_stub()
    obs, info = env.reset(seed=1)
    assert obs.shape == (8, N_FEATURES) and obs.dtype == np.float32
    assert info["masks"].shape == (8, N_ACTIONS) and info["masks"].dtype == bool
    policy = RandomPolicy()
    policy.reset(None, np.random.default_rng(0))
    obs, rewards, done, info = env.step(policy.act(obs, info["masks"], env.views))
    assert rewards.shape == (8,)
    assert not done


def test_stub_is_deterministic():
    a, b = make_stub(), make_stub()
    obs_a, _ = a.reset(seed=5)
    obs_b, _ = b.reset(seed=5)
    assert np.array_equal(obs_a, obs_b)


def test_stub_rejects_masked_actions():
    env = make_stub()
    _, info = env.reset(seed=3)
    masks = info["masks"]
    row, col = np.argwhere(~masks)[0]
    actions = np.zeros(8, dtype=int)
    actions[row] = col
    with pytest.raises(ValueError):
        env.step(actions)
