"""Tests for the baseline policies."""

import numpy as np
import pytest

from core.actions import HOLD, LOCAL, send_action
from core.config import load_config
from policies.evblb import EVBLB, PerfectInfo
from policies.nearest import Nearest
from policies.random_policy import RandomPolicy
from sim.rollout import run_episode
from tests.helpers import add_tasks, make_world


def test_random_policy_respects_masks():
    rng = np.random.default_rng(0)
    masks = rng.random((200, 7)) < 0.3
    masks[:, 0] = True
    policy = RandomPolicy()
    policy.reset(None, np.random.default_rng(1))
    actions = policy.act(np.zeros((200, 30), np.float32), masks, [])
    assert masks[np.arange(200), actions].all()


def test_nearest_goes_local_whenever_it_can():
    masks = np.zeros((3, 7), dtype=bool)
    masks[:, HOLD] = True
    masks[1, LOCAL] = True
    actions = Nearest().act(None, masks, [])
    assert actions.tolist() == [HOLD, LOCAL, HOLD]


def test_evblb_picks_true_argmax_of_p_rem():
    w = make_world()
    # load server 1 (slot 0 of server 0) heavily, leave server 2 (slot 1) empty
    add_tasks(w, 1, [1000] * 20)
    w.nodes[1].start_dispatch(LOCAL)
    add_tasks(w, 0, [500] * 5)
    w.nodes[0].start_dispatch(LOCAL)
    w.engine.run_until(0.1)
    add_tasks(w, 0, [100])
    mask = np.array(w.nodes[0].protocol_mask(2.0))
    assert EVBLB.best_target(w.views[0], mask) == send_action(1)
    # if the empty neighbour is masked, it falls back to itself (server 0 < server 1 load)
    mask[send_action(1)] = False
    assert EVBLB.best_target(w.views[0], mask) == LOCAL


def test_evblb_dispatches_only_on_its_grid():
    policy = EVBLB(dt=1.0)
    assert not policy.due(0.0) and not policy.due(0.75)
    assert policy.due(1.0) and policy.due(3.0)
    assert EVBLB(dt=0.25).due(0.5)


def test_evblb_pays_for_polls():
    dotted = {"topology.n_servers": 10, "run.duration": 20.0, "run.arrival_cutoff": 2.0}
    cfg = load_config(dotted=dotted)
    _, summary = run_episode(cfg, EVBLB(dt=2.0), seed=0)
    assert summary["msg_POLL"] == summary["msg_POLL_REPLY"] > 0
    assert summary["decision_latency"] > 0.020  # poll wait + the round's own cost
    _, nearest = run_episode(cfg, Nearest(), seed=0)
    assert nearest["msg_POLL"] == 0


def test_poll_round_refreshes_the_cache():
    w = make_world()
    node = w.nodes[0]
    add_tasks(w, 1, [1000] * 4)
    w.nodes[1].start_dispatch(LOCAL)
    w.engine.run_until(0.5)
    node.send_polls()
    w.engine.run_until(0.5 + w.costs.poll_wait[0])
    assert w.counter.counts[0] == 2  # one POLL per neighbour
    assert node.cache_prem[0] == pytest.approx(w.nodes[1].p_rem())
    assert node.cache_time[0] > 0.5


def test_perfect_info_dispatches_every_interval_without_polling():
    policy = PerfectInfo()
    assert policy.dt == 0.25 and not policy.needs_poll
    dotted = {"topology.n_servers": 10, "run.duration": 20.0, "run.arrival_cutoff": 2.0}
    _, summary = run_episode(load_config(dotted=dotted), policy, seed=0)
    assert summary["msg_POLL"] == 0
    assert summary["violations"] == 0 and summary["unfinished"] == 0
