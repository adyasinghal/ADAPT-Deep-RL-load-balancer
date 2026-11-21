"""Wiring: build the engine, transport, nodes and cloud of one run.

Kept separate from the env so tests can drive the nodes directly without the RL interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.invariants import TaskLedger
from core.messages import MessageCounter
from core.metrics import MetricsRecorder
from core.model import CostModel, Indicators, Topology
from core.node import CloudServer, NodeContext, NodeView, OracleDetector, ProtocolParams, ServerNode
from core.transport import SimTransport
from sim.engine import Engine


@dataclass
class World:
    """All live objects of one simulation run."""

    cfg: dict[str, Any]
    topology: Topology
    engine: Engine
    transport: SimTransport
    counter: MessageCounter
    ledger: TaskLedger
    recorder: MetricsRecorder
    costs: CostModel
    indicators: Indicators
    params: ProtocolParams
    nodes: list[ServerNode]
    cloud: CloudServer
    views: list[NodeView]

    def alive(self) -> np.ndarray:
        """Boolean mask of servers that have not crashed."""
        return np.array([not nd.crashed for nd in self.nodes])


def build_world(
    cfg: dict[str, Any],
    topology: Topology,
    loss_rng: np.random.Generator,
    strict: bool | None = None,
) -> World:
    """Create a fresh world for ``topology``. ``strict`` overrides ``invariants.strict``."""
    if strict is None:
        strict = bool(cfg["invariants"]["strict"])
    engine = Engine()
    costs = CostModel(cfg, topology)
    counter = MessageCounter()
    transport = SimTransport(engine, costs, counter, loss_rng, cfg["costs"]["p_loss"])
    ledger = TaskLedger(strict=strict)
    recorder = MetricsRecorder(
        horizon=float(cfg["run"]["duration"]),
        sample_interval=float(cfg["run"]["sample_interval"]),
    )
    indicators = Indicators.from_config(cfg)
    params = ProtocolParams.from_config(cfg)
    nodes: list[ServerNode] = []
    ctx = NodeContext(
        clock=engine,
        transport=transport,
        ledger=ledger,
        costs=costs,
        indicators=indicators,
        params=params,
        recorder=recorder,
        detector=OracleDetector(nodes),
    )
    nodes.extend(ServerNode(i, topology, ctx) for i in range(topology.n))
    transport.nodes = nodes
    cloud = CloudServer(float(cfg["cloud"]["cpu_ghz"]), costs.cloud_latency, ctx)
    views = [NodeView(nd, nodes) for nd in nodes]
    return World(
        cfg=cfg,
        topology=topology,
        engine=engine,
        transport=transport,
        counter=counter,
        ledger=ledger,
        recorder=recorder,
        costs=costs,
        indicators=indicators,
        params=params,
        nodes=nodes,
        cloud=cloud,
        views=views,
    )
