"""One server's ADAPT observation, computed from that server alone.

The env builds all servers' observations at once with numpy (fast, for training and the
experiments). The demo runs each server on its own, so it needs the same 30 features for a
single node. ``tests/test_features.py`` checks that both give identical results.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from core.actions import FIRST_SEND, N_FEATURES, N_SLOTS
from core.node import ServerNode

CPU_VARIANCE_REF = 67500.0  # variance of U(100, 1000)
VARIANCE_EPOCHS = 10
EWMA_ALPHA = 0.2


class LocalTraffic:
    """Arrival statistics of one server's Voronoi cell (features 6-8)."""

    def __init__(self) -> None:
        self.rate_ewma = 0.0
        self.last_cpu = 0.0
        self.history: deque[float] = deque(maxlen=VARIANCE_EPOCHS)

    def end_epoch(self, count: int, cpu_sum: float, dt: float) -> None:
        """Fold in the arrivals of the decision interval that just ended."""
        self.rate_ewma = EWMA_ALPHA * count / dt + (1 - EWMA_ALPHA) * self.rate_ewma
        mean = cpu_sum / count if count else np.nan
        self.history.append(mean)
        self.last_cpu = mean if count else 0.0

    def cpu_variance(self) -> float:
        """Variance of the per-interval mean task size over the last 10 intervals with arrivals."""
        values = [v for v in self.history if not np.isnan(v)]
        return float(np.var(values)) if len(values) >= 2 else 0.0


def node_observation(
    node: ServerNode,
    traffic: LocalTraffic,
    *,
    max_hold: float = 2.0,
    mean_cpu: float = 550.0,
    psi: float = 40.0,
    capacity_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The 30 features and the action mask of ``node`` at the current time.

    Uses only what the node itself knows: its own queue and buffer, its cell's arrivals and
    the load reports it has cached from its neighbours.
    """
    now = node.now
    obs = np.zeros(N_FEATURES, dtype=np.float32)
    cpu = node.cpu_ghz
    buf = node.buffer
    obs[0] = node.utilisation()
    obs[1] = node.q_count / node.ctx.params.queue_capacity
    obs[2] = cpu / 4.0
    if buf:
        obs[3] = min(len(buf) / 100.0, 1.0)
        obs[4] = min(sum(t.cpu for t in buf) / (1000.0 * cpu * 2.0), 1.0)
        obs[5] = min((now - node.oldest_arrival()) / 2.0, 1.0)
    obs[6] = np.clip(traffic.rate_ewma * mean_cpu / (1000.0 * cpu), 0.0, 2.0) / 2.0
    obs[7] = traffic.last_cpu / 1000.0
    obs[8] = np.clip(traffic.cpu_variance() / CPU_VARIANCE_REF, 0.0, 1.0)

    exists = np.array(
        [j >= 0 and not node.suspected[s] for s, j in enumerate(node.neighbours)], dtype=bool
    )
    prem = np.where(exists, node.cache_prem, 0.0)
    util = np.where(exists, node.cache_util, 0.0)
    age = np.where(exists, np.clip((now - node.cache_time) / 2.0, 0.0, 1.0), 0.0)
    dist = np.array(node.neighbour_dist) / psi

    values = np.concatenate([[obs[0]], util[exists]])
    obs[9] = float(np.std(values))
    slots = np.stack([np.clip(prem, -1.0, 1.0), age, dist, exists], axis=1)
    obs[10:] = slots.reshape(4 * N_SLOTS)

    mask = np.array(node.protocol_mask(max_hold), dtype=bool)
    if capacity_mask and buf:
        mask[FIRST_SEND:] &= prem >= node.batch_usage()
    return obs, mask
