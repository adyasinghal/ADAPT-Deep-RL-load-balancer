"""Tests for the RL pieces: network, replay, DQN update and the ADAPT policy."""

import numpy as np
import pytest
import torch

from core.actions import HOLD, LOCAL, N_ACTIONS, N_FEATURES
from policies.adapt import AdaptPolicy
from rl.dqn import DQNAgent, DQNConfig, linear_epsilon
from rl.networks import DuelingQNet, NumpyQNet, export_npz, greedy
from rl.replay import ReplayBuffer
from sim.env import EdgeEnv
from tests.test_env import small_cfg


def random_masks(rng, n):
    masks = rng.random((n, N_ACTIONS)) < 0.4
    masks[:, HOLD] = True
    return masks


def test_numpy_forward_matches_torch(tmp_path):
    torch.manual_seed(0)
    net = DuelingQNet()
    export_npz(net, tmp_path / "w.npz")
    obs = np.random.default_rng(0).random((64, N_FEATURES), dtype=np.float32)
    with torch.no_grad():
        expected = net(torch.from_numpy(obs)).numpy()
    assert np.allclose(NumpyQNet(tmp_path / "w.npz")(obs), expected, atol=1e-5)


def test_greedy_never_picks_a_masked_action():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(500, N_ACTIONS))
    masks = random_masks(rng, 500)
    actions = greedy(q, masks)
    assert masks[np.arange(500), actions].all()


def test_replay_ring_buffer_overwrites_oldest():
    buf = ReplayBuffer(5, np.random.default_rng(0))
    for k in range(3):  # 3 blocks of 2 rows
        obs = np.full((2, N_FEATURES), k, dtype=np.float32)
        masks = np.ones((2, N_ACTIONS), dtype=bool)
        buf.add(obs, masks, np.array([k, k]), np.array([k, k], dtype=float), obs, masks)
    assert len(buf) == 5
    assert sorted(buf.action.tolist()) == [0, 1, 1, 2, 2]
    batch = buf.sample(32)
    assert batch.obs.shape == (32, N_FEATURES) and batch.next_mask.shape == (32, N_ACTIONS)


def test_epsilon_schedule():
    assert linear_epsilon(0, 1.0, 0.05, 100) == 1.0
    assert linear_epsilon(50, 1.0, 0.0, 100) == pytest.approx(0.5)
    assert linear_epsilon(10_000, 1.0, 0.05, 100) == pytest.approx(0.05)


def test_exploration_respects_masks():
    agent = DQNAgent(DQNConfig())
    rng = np.random.default_rng(2)
    masks = random_masks(rng, 400)
    obs = rng.random((400, N_FEATURES), dtype=np.float32)
    actions = agent.act(obs, masks, epsilon=1.0, rng=rng)
    assert masks[np.arange(400), actions].all()
    assert len(set(actions.tolist())) > 2


def test_double_dqn_target_ignores_masked_next_actions():
    agent = DQNAgent(DQNConfig(gamma=0.9))
    buf = ReplayBuffer(10, np.random.default_rng(0))
    obs = np.zeros((1, N_FEATURES), dtype=np.float32)
    only_hold = np.zeros((1, N_ACTIONS), dtype=bool)
    only_hold[0, HOLD] = True
    buf.add(obs, only_hold, np.array([HOLD]), np.array([1.0]), obs, only_hold)
    # make a masked action look hugely valuable in both networks
    with torch.no_grad():
        agent.online.advantage.bias[LOCAL] += 100.0
    agent.sync_target()
    with torch.no_grad():
        q = agent.online(torch.from_numpy(obs))[0]
    # target must use Q(s', HOLD), not the masked LOCAL
    expected = 1.0 + 0.9 * q[HOLD].item()
    loss = agent.td_loss(buf.sample(1)).item()
    delta = abs(q[HOLD].item() - expected)
    assert loss == pytest.approx(delta - 0.5 if delta > 1 else 0.5 * delta**2, rel=1e-4)


def test_updates_reduce_the_loss_on_a_fixed_batch():
    agent = DQNAgent(DQNConfig(lr=1e-2, gamma=0.0))
    rng = np.random.default_rng(3)
    buf = ReplayBuffer(256, rng)
    obs = rng.random((256, N_FEATURES), dtype=np.float32)
    masks = np.ones((256, N_ACTIONS), dtype=bool)
    actions = rng.integers(0, N_ACTIONS, 256)
    buf.add(obs, masks, actions, obs[:, 0] - obs[:, 1], obs, masks)
    batch = buf.sample(256)
    first = agent.update(batch)
    for _ in range(200):
        last = agent.update(batch)
    assert last < first / 5


def test_saved_weights_load_into_the_policy(tmp_path):
    agent = DQNAgent(DQNConfig())
    agent.save(tmp_path / "adapt")
    policy = AdaptPolicy(weights=str(tmp_path / "adapt.npz"))
    env = EdgeEnv(small_cfg(), policy, strict=True)
    obs, info = env.reset(seed=0)
    policy.reset(env.topology, env.policy_rng)
    done = False
    while not done:
        obs, _, done, info = env.step(policy.act(obs, info["masks"], env.views))
    assert env.summary()["violations"] == 0
    other = DQNAgent(DQNConfig(), seed=5)
    other.load(tmp_path / "adapt.pt")
    assert np.allclose(other.q_values(obs), agent.q_values(obs))


def test_adapt_traits_per_variant():
    full, poll, fixed = AdaptPolicy(), AdaptPolicy("poll"), AdaptPolicy("fixedDT")
    assert full.uses_cache and full.capacity_mask and not full.needs_poll
    assert poll.needs_poll and poll.fresh_obs and not poll.uses_cache
    assert fixed.fixed_dt == 2.0 and fixed.name == "adapt_fixedDT"
    with pytest.raises(ValueError):
        AdaptPolicy("nope")


def test_fixed_interval_variant_only_dispatches_every_two_seconds():
    policy = AdaptPolicy("fixedDT")
    env = EdgeEnv(small_cfg(), policy, strict=True)
    obs, info = env.reset(seed=1)
    last = np.zeros(env.n)
    for _ in range(60):
        masks = policy.effective_masks(info["masks"], env.views)
        free = masks[:, LOCAL] & masks[:, HOLD]
        too_soon = env.now - last < 2.0 - 1e-9
        assert not free[too_soon].any()
        actions = np.where(masks[:, LOCAL], LOCAL, HOLD)
        last[actions != HOLD] = env.now
        obs, _, _, info = env.step(actions)
