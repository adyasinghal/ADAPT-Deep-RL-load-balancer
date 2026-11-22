"""The interface every load-balancing policy implements.

The env calls ``act`` once per decision instant (every 0.25 s) for *all* servers at once and
gets back one action index per server (see ``core.actions``). Learners also get ``observe``.

Besides ``act`` and ``observe``, the env reads a few optional traits from the policy
(``BasePolicy`` gives the defaults):

* ``needs_poll``     -- every dispatch first pays a poll round (EVBLB, TOPSIS, ADAPT-poll).
* ``max_hold``       -- HOLD is masked once the oldest buffered task is this old (rule 5).
* ``uses_cache``     -- servers send heartbeats so neighbour load reports stay fresh (ADAPT, B4).
* ``fresh_obs``      -- neighbour features use true values instead of cached ones (ADAPT-poll).
* ``capacity_mask``  -- also mask SEND_j when the cached room at j is smaller than the batch
                        (ADAPT only; it is a heuristic, not a protocol rule).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from core.model import Topology
    from core.node import NodeView


class Policy(Protocol):
    """Structural type for policies; see the module docstring for the env-facing traits."""

    name: str
    needs_poll: bool
    max_hold: float

    def reset(self, topology: Topology, rng: np.random.Generator) -> None:
        """Called once per episode before the first ``act``."""

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list[NodeView]) -> np.ndarray:
        """Return one action per server.

        ``obs`` is (n, 30) float32, ``masks`` is (n, 7) bool (True = allowed) and ``ctx`` gives
        read-only access to the true node state for baselines that are allowed to use it.
        """

    def observe(
        self, rewards: np.ndarray, next_obs: np.ndarray, done: bool, info: dict[str, Any]
    ) -> None:
        """Feedback after each env step. Only learners need to do anything here."""


class BasePolicy:
    """Convenient base class with the default traits and no-op hooks."""

    name = "base"
    needs_poll = False
    max_hold = 2.0
    uses_cache = False
    fresh_obs = False
    capacity_mask = False

    def __init__(self) -> None:
        self.topology: Topology | None = None
        self.rng: np.random.Generator = np.random.default_rng(0)

    def reset(self, topology: Topology, rng: np.random.Generator) -> None:
        """Remember the topology and the policy's own random generator."""
        self.topology = topology
        self.rng = rng

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list[NodeView]) -> np.ndarray:
        """Subclasses decide here."""
        raise NotImplementedError

    def observe(
        self, rewards: np.ndarray, next_obs: np.ndarray, done: bool, info: dict[str, Any]
    ) -> None:
        """Ignore feedback by default."""
