"""The per-node observation (used by the demo) must equal the env's batched one."""

import numpy as np

from core.features import LocalTraffic, node_observation
from policies.adapt import AdaptPolicy
from policies.random_policy import RandomPolicy
from sim.env import EdgeEnv
from tests.test_env import small_cfg


def traffic_from_env(env, i):
    """Rebuild server i's LocalTraffic from the env's arrays."""
    tr = LocalTraffic()
    tr.rate_ewma = env._rate_ewma[i]
    tr.last_cpu = env._last_cpu[i]
    tr.history.extend(env._cpu_hist[i])
    return tr


def test_node_observation_matches_the_env():
    adapt = AdaptPolicy()  # for its traits: heartbeats on, capacity mask on
    env = EdgeEnv(small_cfg(**{"workload.rho": 0.9}), adapt, strict=True)
    obs, info = env.reset(seed=3)
    driver = RandomPolicy()
    driver.reset(env.topology, np.random.default_rng(0))
    for step in range(200):
        if step % 10 == 0:
            for i, node in enumerate(env.world.nodes):
                own_obs, own_mask = node_observation(node, traffic_from_env(env, i))
                np.testing.assert_allclose(own_obs, obs[i], atol=1e-6)
                np.testing.assert_array_equal(own_mask, info["masks"][i])
        obs, _, _, info = env.step(driver.act(obs, info["protocol_masks"], env.views))


def test_local_traffic_updates_like_the_env():
    tr = LocalTraffic()
    tr.end_epoch(4, 2000.0, 0.25)
    assert tr.rate_ewma == 0.2 * 16 and tr.last_cpu == 500.0
    tr.end_epoch(0, 0.0, 0.25)
    assert tr.last_cpu == 0.0 and tr.cpu_variance() == 0.0
    tr.end_epoch(2, 1400.0, 0.25)
    assert tr.cpu_variance() == np.var([500.0, 700.0])
