"""Tests for the topology, the resource indicators and the cost model."""

import numpy as np
import pytest

from core.config import load_config
from core.model import CostModel, Indicators, Topology
from sim.workloads import WORKLOADS, rate_for_load


def random_topology(n=40, seed=0):
    cfg = load_config(dotted={"topology.n_servers": n})
    return Topology.generate(cfg, np.random.default_rng(seed))


def test_voronoi_home_matches_brute_force():
    topo = random_topology()
    points = np.random.default_rng(1).uniform(0, 200, size=(500, 2))
    brute = np.argmin(((points[:, None, :] - topo.positions[None]) ** 2).sum(-1), axis=1)
    assert np.array_equal(topo.home_of(points), brute)


def test_home_skips_crashed_servers():
    topo = random_topology(n=10)
    alive = np.ones(10, dtype=bool)
    alive[[2, 5]] = False
    homes = topo.home_of(np.random.default_rng(2).uniform(0, 200, (300, 2)), alive)
    assert not np.isin(homes, [2, 5]).any()


def test_neighbours_are_nearest_within_radius_and_sorted():
    topo = random_topology(n=60)
    d = topo.distances()
    for i in range(topo.n):
        slots = [j for j in topo.neighbours[i] if j >= 0]
        assert len(slots) <= 5
        assert i not in slots
        assert all(d[i, j] <= topo.psi for j in slots)
        assert list(d[i, slots]) == sorted(d[i, slots])
        # nobody closer was left out
        others = [j for j in range(topo.n) if j != i and j not in slots]
        if len(slots) < 5:
            assert all(d[i, j] > topo.psi for j in others)
        elif others:
            assert min(d[i, others]) >= max(d[i, slots])
        for j in slots:
            assert i in topo.reverse[j]


def test_empty_slots_when_isolated():
    topo = Topology.from_arrays(
        [[0, 0], [10, 0], [150, 150]], [4] * 3, [500] * 3, [1.0] * 3, nu=5, psi=40
    )
    assert list(topo.neighbours[0]) == [1, -1, -1, -1, -1]
    assert list(topo.neighbours[2]) == [-1] * 5


def test_indicators_by_hand():
    ind = Indicators(1 / 3, 1 / 3, 1 / 3, 2.0)
    # 16 GB, 1000 GB, 2 GHz -> (0.5 + 0.5 + 0.5) / 3
    assert ind.capacity(16, 1000, 2.0) == pytest.approx(0.5)
    # 320 MB, 20000 MB, 800 Mcycles -> (0.01 + 0.01 + 0.1) / 3
    assert ind.usage(320, 20_000, 800) == pytest.approx(0.04)
    assert ind.utilisation(2000.0, 2.0) == pytest.approx(0.5)
    assert ind.utilisation(9000.0, 2.0) == 1.0


def test_cost_model():
    cfg = load_config()
    topo = Topology.from_arrays([[0, 0], [30, 40]], [4, 4], [500, 500], [1.0, 1.0], 5, 60)
    costs = CostModel(cfg, topo)
    assert costs.delay[0][1] == pytest.approx(0.002 + 0.0001 * 50)
    assert costs.delay[0][0] == 0.0
    assert costs.dispatch_cost(10) == pytest.approx(0.020 + 0.005)
    assert costs.transfer_extra(3) == pytest.approx(0.024)
    assert costs.poll_wait[0] == pytest.approx(2 * 0.007)

    zero = CostModel(load_config(dotted={"costs.zero_costs": True}), topo)
    assert zero.delay[0][1] == 0.0 and zero.dispatch_cost(10) == 0.0 and zero.poll_wait[0] == 0


def test_rate_for_load():
    cfg = load_config()
    topo = Topology.from_arrays([[0, 0], [1, 1]], [4, 4], [500, 500], [1.0, 3.0], 5, 40)
    assert rate_for_load(cfg, topo, 0.55) == pytest.approx(0.55 * 4000 / 550)


def test_flash_crowd_and_diurnal_rates():
    cfg = load_config()
    topo = random_topology(n=10)
    rng = np.random.default_rng(0)
    w3 = WORKLOADS["W3"](cfg, topo, 0.7, rng)
    base = w3.base_rate
    assert w3.rate(100.0) == base and w3.rate(310.0) == 3 * base and w3.rate(370.0) == base
    assert w3.rate(910.0) == 3 * base
    w5 = WORKLOADS["W5"](cfg, topo, 0.7, rng)
    assert w5.rate(0.0) == pytest.approx(base)
    assert w5.rate(450.0) == pytest.approx(base / 0.7 * 1.0)
    assert w5.rate(1350.0) == pytest.approx(base / 0.7 * 0.4)


def test_moving_hotspot_concentrates_traffic():
    cfg = load_config()
    w4 = WORKLOADS["W4"](cfg, random_topology(n=10), 0.7, np.random.default_rng(1))
    pos = w4.positions(10.0, 20_000)
    near = np.linalg.norm(pos - w4.centre(10.0), axis=1) <= 20.0 + 1e-9
    # 40% from the disk plus the uniform arrivals that happen to land in it
    assert 0.40 < near.mean() < 0.45
    assert not np.array_equal(w4.centre(10.0), w4.centre(310.0))
