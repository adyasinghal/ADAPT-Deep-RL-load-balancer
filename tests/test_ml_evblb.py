"""Tests for B4, the ML-enhanced EVBLB of the 2024 paper."""

import numpy as np
import pytest

from core.actions import HOLD
from core.config import load_config
from core.messages import MsgKind
from core.model import Topology
from policies.ml_evblb import MLEvblb
from sim.rollout import run_episode
from tests.helpers import add_tasks, make_world


def make_b4(n=3, seed=0):
    topo = Topology.from_arrays([[0, 0], [10, 0], [20, 0]][:n], [4] * n, [500] * n, [1] * n, 5, 40)
    policy = MLEvblb()
    policy.reset(topo, np.random.default_rng(seed))
    return policy


def test_b4_q_update_matches_eq_8():
    policy = make_b4()
    policy.q[0, 30] = [0.0, 0.5, 0.2, 0.0, 0.1]
    lbf = 0.12
    r = policy.interval_reward(lbf, 1.0)
    assert r == pytest.approx(0.5 * 0.88 + 0.5 * 1.0 / 4.0)
    policy.q[0, 10, 2] = 0.4
    policy.q_update(0, 10, 2, r, 30)
    assert policy.q[0, 10, 2] == pytest.approx(0.8 * 0.4 + 0.2 * (r + 0.2 * 0.5))
    assert MLEvblb.lbf_state(0.126) == 13 and MLEvblb.lbf_state(1.7) == 100


def test_b4_bandit_reward_and_update():
    policy = make_b4()
    assert policy.batch_reward(True, 1.0) == -1.0
    assert policy.batch_reward(False, 6.0) == pytest.approx(0.8)
    assert policy.batch_reward(False, 45.0) == 0.0
    before = policy.probabilities(0)
    assert before[3:] == pytest.approx([0, 0, 0])  # server 0 has only two neighbours
    policy.bandit_update(0, 1, 0.8)
    assert policy.pref[0, 1] == pytest.approx(0.1 * 0.8 * before[1])
    assert policy.mean_reward[0, 1] == pytest.approx(0.8)
    assert policy.pref[0, 0] == 0.0 and policy.pref[0, 2] == 0.0


def test_b4_learns_from_rejects_in_observe():
    policy = make_b4()
    policy.observe(
        None, None, False, {"lbf": 0.3, "replies": [(0, 1, int(MsgKind.REJECT), 2, 0.0)]}
    )
    assert policy.lbf == 0.3
    assert policy.pref[0, 2] < 0 and policy.mean_reward[0, 2] == -1.0


def test_b4_only_dispatches_when_its_interval_runs_out_or_rule_5_forces_it():
    policy = make_b4()
    policy.choice[:] = 3  # 2 s
    w = make_world()
    add_tasks(w, 0, [100])
    masks = np.array([nd.protocol_mask(4.0) for nd in w.nodes])
    w.engine.run_until(1.0)
    assert policy.act(None, masks, w.views)[0] == HOLD
    w.engine.run_until(2.0)
    assert policy.act(None, masks, w.views)[0] != HOLD
    assert policy.started[0] == 2.0


def test_b4_runs_cleanly_without_polling():
    dotted = {"topology.n_servers": 12, "run.duration": 120.0, "run.arrival_cutoff": 10.0}
    _, summary = run_episode(load_config(dotted=dotted), MLEvblb(), seed=4, strict=True)
    assert summary["violations"] == 0 and summary["unfinished"] == 0
    assert summary["msg_POLL"] == 0
