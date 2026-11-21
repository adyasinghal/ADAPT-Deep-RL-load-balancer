"""Static parts of the simulation model: tasks, topology, resource indicators and costs.

Units: task memory/disk in MB, server memory/disk in GB, CPU demand in Mcycles, CPU speed in
GHz, time in seconds, distance in area units. A task with ``cpu`` Mcycles on a server with
``R`` GHz runs for ``cpu / (1000 * R)`` seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from core.actions import N_SLOTS


class TaskState(IntEnum):
    """Where a task is. Every task is in exactly one of these at any time (property P1)."""

    NEW = 0
    BUFFERED = 1  # in its home server's buffer, or in a dispatch round that hasn't finished
    IN_FLIGHT = 2  # offered to another server (or on its way to the cloud), not yet admitted
    QUEUED = 3
    RUNNING = 4
    DONE = 5
    BLOCKED = 6  # dropped because a queue was full
    LOST = 7  # lost_in_crash


class Task:
    """One user request. Plain object with slots because a run creates ~300k of them."""

    __slots__ = (
        "id",
        "home",
        "h",
        "mem",
        "disk",
        "cpu",
        "state",
        "offer",
        "executor",
        "t_arrive",
        "t_buffer",
        "t_dispatch",
        "t_enqueue",
        "t_start",
        "t_finish",
    )

    def __init__(self, tid: int, home: int, t_arrive: float, mem: float, disk: float, cpu: float):
        self.id = tid
        self.home = home
        self.h = 0  # hop count: +1 on every REJECT
        self.mem = mem
        self.disk = disk
        self.cpu = cpu
        self.state = TaskState.NEW
        self.offer = -1  # id of the home server's latest OFFER that carried this task
        self.executor = -1
        self.t_arrive = t_arrive
        self.t_buffer = t_arrive  # when it (last) entered the buffer; used for P2
        self.t_dispatch: float | None = None
        self.t_enqueue: float | None = None
        self.t_start: float | None = None
        self.t_finish: float | None = None

    def __repr__(self) -> str:
        return f"Task({self.id}, home={self.home}, {self.state.name}, h={self.h})"


@dataclass
class Topology:
    """Server positions and capacities plus the fixed neighbour slots of every server."""

    positions: np.ndarray  # (n, 2)
    mem_gb: np.ndarray  # (n,)
    disk_gb: np.ndarray  # (n,)
    cpu_ghz: np.ndarray  # (n,)
    neighbours: np.ndarray  # (n, N_SLOTS) int, -1 marks an empty slot
    neighbour_dist: np.ndarray  # (n, N_SLOTS), 0 for empty slots
    psi: float
    area: float
    reverse: list[list[int]] = field(default_factory=list)  # servers that list i as a neighbour
    _trees: dict[tuple[int, ...], tuple[cKDTree, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    @property
    def n(self) -> int:
        """Number of edge servers."""
        return len(self.cpu_ghz)

    @classmethod
    def generate(cls, cfg: dict[str, Any], rng: np.random.Generator) -> Topology:
        """Random topology from the ``topology`` and ``servers`` config sections."""
        topo_cfg, srv = cfg["topology"], cfg["servers"]
        n = int(topo_cfg["n_servers"])
        area = float(topo_cfg["area"])
        positions = rng.uniform(0.0, area, size=(n, 2))
        mem = rng.choice(np.asarray(srv["memory_gb"], dtype=float), size=n)
        disk = rng.choice(np.asarray(srv["disk_gb"], dtype=float), size=n)
        steps = round((srv["cpu_ghz_max"] - srv["cpu_ghz_min"]) / srv["cpu_ghz_step"])
        speeds = np.round(srv["cpu_ghz_min"] + srv["cpu_ghz_step"] * np.arange(steps + 1), 1)
        cpu = rng.choice(speeds, size=n)
        return cls.from_arrays(positions, mem, disk, cpu, topo_cfg["nu"], topo_cfg["psi"], area)

    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray,
        mem_gb: np.ndarray,
        disk_gb: np.ndarray,
        cpu_ghz: np.ndarray,
        nu: int,
        psi: float,
        area: float = 200.0,
    ) -> Topology:
        """Build a topology from explicit arrays (tests use this for hand-made layouts).

        Neighbours are the (up to) ``nu`` nearest other servers within distance ``psi``,
        sorted by distance. This is what EVBLB's Algorithm 2 converges to with a fine radius
        step, without having to simulate the radius growth.
        """
        if nu > N_SLOTS:
            raise ValueError(f"nu={nu} but observations only have {N_SLOTS} neighbour slots")
        positions = np.asarray(positions, dtype=float)
        n = len(positions)
        neighbours = np.full((n, N_SLOTS), -1, dtype=int)
        dists = np.zeros((n, N_SLOTS))
        if n > 1 and nu > 0:
            tree = cKDTree(positions)
            k = min(nu + 1, n)
            d, idx = tree.query(positions, k=k, distance_upper_bound=psi)
            d, idx = d.reshape(n, k), idx.reshape(n, k)
            for i in range(n):
                pairs = zip(d[i], idx[i], strict=True)
                found = [(dd, j) for dd, j in pairs if j != i and j < n]
                for slot, (dd, j) in enumerate(found[:nu]):
                    neighbours[i, slot] = j
                    dists[i, slot] = dd
        reverse: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in neighbours[i]:
                if j >= 0:
                    reverse[j].append(i)
        return cls(
            positions=positions,
            mem_gb=np.asarray(mem_gb, dtype=float),
            disk_gb=np.asarray(disk_gb, dtype=float),
            cpu_ghz=np.asarray(cpu_ghz, dtype=float),
            neighbours=neighbours,
            neighbour_dist=dists,
            psi=float(psi),
            area=float(area),
            reverse=reverse,
        )

    def distances(self) -> np.ndarray:
        """(n, n) matrix of Euclidean distances between servers."""
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    def home_of(self, points: np.ndarray, alive: np.ndarray | None = None) -> np.ndarray:
        """Voronoi home server (nearest live server) of each point in ``points``."""
        if alive is None:
            alive = np.ones(self.n, dtype=bool)
        key = tuple(np.flatnonzero(~alive))
        if key not in self._trees:
            ids = np.flatnonzero(alive)
            self._trees[key] = (cKDTree(self.positions[ids]), ids)
        tree, ids = self._trees[key]
        _, nearest = tree.query(points)
        return ids[nearest]


@dataclass(frozen=True)
class Indicators:
    """EVBLB's capacity / remaining-resource indicators (Eqs. 3-5) and utilisation."""

    alpha: float
    beta: float
    gamma: float
    dt_ref: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Indicators:
        """Read the ``indicators`` section."""
        c = cfg["indicators"]
        return cls(float(c["alpha"]), float(c["beta"]), float(c["gamma"]), float(c["dt_ref"]))

    def capacity(self, mem_gb, disk_gb, cpu_ghz):
        """P_e: normalised capacity of a server (works on scalars or arrays)."""
        return self.alpha * mem_gb / 32 + self.beta * disk_gb / 2000 + self.gamma * cpu_ghz / 4

    def usage(self, mem_mb, disk_mb, cpu_mc):
        """How much of P_e a task (or a sum of tasks) uses up."""
        return (
            self.alpha * mem_mb / 32000
            + self.beta * disk_mb / 2_000_000
            + self.gamma * cpu_mc / (4000 * self.dt_ref)
        )

    def utilisation(self, cpu_mc, cpu_ghz):
        """u_i = min(1, queued Mcycles / (1000 R dt_ref)), with the same window for everyone."""
        return np.minimum(1.0, cpu_mc / (1000.0 * cpu_ghz * self.dt_ref))


class CostModel:
    """Message, transfer and dispatch-round costs. Identical for every method.

    With ``costs.zero_costs`` every delay and cost here is 0 (experiment E9).
    """

    def __init__(self, cfg: dict[str, Any], topology: Topology) -> None:
        c = cfg["costs"]
        zero = bool(c["zero_costs"])
        dist = topology.distances()
        delay = np.zeros_like(dist) if zero else c["msg_base"] + c["msg_per_unit"] * dist
        np.fill_diagonal(delay, 0.0)
        # nested lists: indexing a Python list is a lot faster than a numpy scalar lookup
        self.delay: list[list[float]] = delay.tolist()
        self.transfer_per_task = 0.0 if zero else float(c["transfer_per_task"])
        self.dispatch_base = 0.0 if zero else float(c["dispatch_base"])
        self.dispatch_per_task = 0.0 if zero else float(c["dispatch_per_task"])
        self.cloud_latency = float(cfg["cloud"]["latency"])
        self.poll_wait = np.zeros(topology.n)
        for i in range(topology.n):
            nbrs = [j for j in topology.neighbours[i] if j >= 0]
            if nbrs:
                self.poll_wait[i] = 2.0 * max(delay[i, j] for j in nbrs)
        self.max_poll_wait = float(self.poll_wait.max(initial=0.0))

    def transfer_extra(self, n_tasks: int) -> float:
        """Serialisation time of a batch on top of the normal message delay."""
        return self.transfer_per_task * n_tasks

    def dispatch_cost(self, batch_size: int) -> float:
        """Time one dispatch round of ``batch_size`` tasks keeps the dispatcher busy."""
        return self.dispatch_base + self.dispatch_per_task * batch_size
