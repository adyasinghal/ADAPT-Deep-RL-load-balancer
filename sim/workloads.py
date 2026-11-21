"""Workloads: arrival rate over time and how each task's size and position are drawn.

The arrival rate is piecewise constant and only changes on multiples of 0.25 s, so the env can
draw each decision epoch's arrivals in one go (a Poisson count plus uniform times).

Only W1 and W2 are implemented here; W3-W6 plug in the same way (subclass ``Workload`` and
add it to ``WORKLOADS``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.model import Topology


def rate_for_load(cfg: dict[str, Any], topology: Topology, rho: float) -> float:
    """Total arrival rate (tasks/s) giving load ``rho``: rho * sum(1000 R_i) / E[T_cpu]."""
    capacity = 1000.0 * float(topology.cpu_ghz.sum())  # Mcycles per second
    return rho * capacity / float(cfg["tasks"]["mean_cpu_mcycles"])


class Workload:
    """W1: stationary Poisson arrivals at uniform positions with uniform task sizes."""

    name = "W1"

    def __init__(
        self, cfg: dict[str, Any], topology: Topology, rho: float, rng: np.random.Generator
    ) -> None:
        self.base_rate = rate_for_load(cfg, topology, rho)
        self.area = topology.area
        self.rng = rng
        t = cfg["tasks"]
        self.mem_range = tuple(t["mem_mb"])
        self.disk_range = tuple(t["disk_mb"])
        self.cpu_range = tuple(t["cpu_mcycles"])

    def rate(self, t: float) -> float:
        """Arrival rate in tasks/s during the epoch starting at ``t``."""
        return self.base_rate

    def sample(self, t: float, k: int) -> tuple[np.ndarray, ...]:
        """Positions (k, 2), memory, disk and CPU demand of ``k`` tasks arriving around ``t``."""
        rng = self.rng
        positions = self.positions(t, k)
        mem = rng.uniform(*self.mem_range, size=k)
        disk = rng.uniform(*self.disk_range, size=k)
        cpu = self.cpu(t, k)
        return positions, mem, disk, cpu

    def positions(self, t: float, k: int) -> np.ndarray:
        """Uniform over the whole area."""
        return self.rng.uniform(0.0, self.area, size=(k, 2))

    def cpu(self, t: float, k: int) -> np.ndarray:
        """Uniform CPU demand in Mcycles."""
        return self.rng.uniform(*self.cpu_range, size=k)


class Tidal(Workload):
    """W2 (from ATOA): 150 s stable phases alternating with 150 s unstable phases.

    Stable: the base rate, T_cpu ~ N(550, 55). Unstable: the rate is re-drawn every 10 s as
    base * U(0.5, 1.5) and T_cpu ~ N(550, 275). T_cpu is clipped to the usual range.
    """

    name = "W2"
    phase = 150.0
    block = 10.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._multipliers: dict[int, float] = {}

    def unstable(self, t: float) -> bool:
        """Whether ``t`` falls in an unstable phase (the run starts stable)."""
        return int(t // self.phase) % 2 == 1

    def rate(self, t: float) -> float:
        """Base rate in stable phases, a fresh random multiple every 10 s otherwise."""
        if not self.unstable(t):
            return self.base_rate
        block = int(t // self.block)
        if block not in self._multipliers:
            self._multipliers[block] = self.rng.uniform(0.5, 1.5)
        return self.base_rate * self._multipliers[block]

    def cpu(self, t: float, k: int) -> np.ndarray:
        """Normal around 550 Mcycles, ten times wider in unstable phases."""
        sd = 275.0 if self.unstable(t) else 55.0
        return np.clip(self.rng.normal(550.0, sd, size=k), *self.cpu_range)


WORKLOADS: dict[str, type[Workload]] = {"W1": Workload, "W2": Tidal}


def make_workload(cfg: dict[str, Any], topology: Topology, rng: np.random.Generator) -> Workload:
    """Build the workload named in ``cfg["workload"]``."""
    name = cfg["workload"]["name"]
    if name not in WORKLOADS:
        raise ValueError(f"unknown workload {name!r}; have {sorted(WORKLOADS)}")
    return WORKLOADS[name](cfg, topology, float(cfg["workload"]["rho"]), rng)
