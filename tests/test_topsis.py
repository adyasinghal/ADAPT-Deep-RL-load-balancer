"""Tests for B7, TOPSIS server selection from TOLB."""

import numpy as np
import pytest

from core.actions import FIRST_SEND, LOCAL
from core.config import load_config
from policies.topsis import Topsis, topsis_scores
from sim.rollout import run_episode
from tests.helpers import add_tasks, make_world


def test_topsis_matches_a_hand_worked_example():
    # P_rem (benefit) 0.5, 0.8, 0.2 and distance (cost) 0, 10, 20 with weights 0.7 / 0.3:
    # normalised benefit 0.5, 1, 0 and cost 1, 0.5, 0 -> weighted (0.35, 0.3), (0.7, 0.15),
    # (0, 0); ideal (0.7, 0.3), anti-ideal (0, 0)
    scores = topsis_scores(np.array([0.5, 0.8, 0.2]), np.array([0.0, 10.0, 20.0]))
    e_best = np.array([0.35, 0.15, np.hypot(0.7, 0.3)])
    e_worst = np.array([np.hypot(0.35, 0.3), np.hypot(0.7, 0.15), 0.0])
    assert scores == pytest.approx(e_worst / (e_best + e_worst))
    assert scores == pytest.approx([0.5684, 0.8268, 0.0], abs=1e-4)
    assert int(np.argmax(scores)) == 1


def test_topsis_with_a_constant_criterion():
    # equal distances: only P_rem decides
    scores = topsis_scores(np.array([0.1, 0.9]), np.array([5.0, 5.0]))
    assert scores[1] > scores[0]
    assert topsis_scores(np.array([0.3, 0.3]), np.array([1.0, 1.0])) == pytest.approx([1, 1])


def test_topsis_prefers_a_near_neighbour_over_a_slightly_emptier_far_one():
    w = make_world(positions=[[0, 0], [5, 0], [35, 0]])
    add_tasks(w, 0, [800] * 10)
    w.nodes[0].start_dispatch(LOCAL)
    add_tasks(w, 1, [100])
    w.nodes[1].start_dispatch(LOCAL)
    w.engine.run_until(0.1)
    add_tasks(w, 0, [100])
    mask = np.array(w.nodes[0].protocol_mask(2.0))
    # server 2 is (slightly) emptier, but six times further away
    assert w.views[0].true_neighbour_prem()[1] > w.views[0].true_neighbour_prem()[0]
    assert Topsis().best_target(w.views[0], mask) == FIRST_SEND + 0


def test_b7_runs_cleanly_and_polls():
    dotted = {"topology.n_servers": 12, "run.duration": 120.0, "run.arrival_cutoff": 10.0}
    _, summary = run_episode(load_config(dotted=dotted), Topsis(), seed=4, strict=True)
    assert summary["violations"] == 0 and summary["unfinished"] == 0
    assert summary["msg_POLL"] > 0
