"""Small helpers to build hand-made worlds for the protocol tests."""

from __future__ import annotations

import numpy as np

from core.config import load_config
from core.model import Task, Topology
from sim.world import World, build_world

LINE = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]  # three servers, all neighbours of each other


def make_world(positions=LINE, cpu=None, mem=None, **dotted) -> World:
    """A strict-mode world over a fixed topology. Extra kwargs are dotted config overrides."""
    n = len(positions)
    cfg = load_config(dotted={k.replace("__", "."): v for k, v in dotted.items()})
    topo = Topology.from_arrays(
        positions,
        mem if mem is not None else [32] * n,
        [2000] * n,
        cpu if cpu is not None else [1.0] * n,
        nu=5,
        psi=40,
    )
    return build_world(cfg, topo, np.random.default_rng(0), strict=True)


_next_id = [10_000]


def add_tasks(world: World, home: int, cpus, mem=200.0, disk=300.0) -> list[Task]:
    """Put tasks with the given CPU demands straight into ``home``'s buffer, arriving now."""
    tasks = []
    for c in cpus:
        t = Task(_next_id[0], home, world.engine.now, mem, disk, float(c))
        _next_id[0] += 1
        world.ledger.register(t)
        tasks.append(t)
    world.nodes[home].accept_arrivals(tasks)
    return tasks
