"""Metrics: per-second time series and the per-run summary (brief section 7)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.messages import MessageCounter, MsgKind
from core.model import Task


def load_balancing_factor(util: np.ndarray) -> float:
    """LBF (EVBLB Eq. 1): population standard deviation of the servers' utilisations."""
    if len(util) == 0:
        return 0.0
    return float(np.std(util))


def time_to_balance(t: np.ndarray, lbf: np.ndarray, threshold: float, hold: float) -> float:
    """First time from which LBF stays below ``threshold`` for ``hold`` seconds (NaN if never)."""
    start = None
    for ti, below in zip(t, lbf < threshold, strict=True):
        if not below:
            start = None
            continue
        if start is None:
            start = ti
        if ti - start >= hold:
            return float(start)
    return math.nan


class MetricsRecorder:
    """Collects everything we report. The nodes call the ``on_*`` hooks."""

    def __init__(self, horizon: float, sample_interval: float = 1.0) -> None:
        self.horizon = horizon
        self.sample_interval = sample_interval
        self.response_times: list[float] = []
        self.completed_in_horizon = 0
        self.last_finish = 0.0
        self.blocked = 0
        self.rounds = 0
        self.decision_latency_sum = 0.0
        self.replies: list[tuple[int, int, int, int, float]] = []  # emptied by the env
        self._rows: list[dict[str, float]] = []
        self._win_done = 0
        self._win_resp = 0.0

    # hooks called by the nodes -------------------------------------------------------

    def on_complete(self, task: Task) -> None:
        """Record a finished task's response time (arrival to result)."""
        resp = task.t_finish - task.t_arrive
        self.response_times.append(resp)
        self._win_done += 1
        self._win_resp += resp
        if task.t_finish <= self.horizon:
            self.completed_in_horizon += 1
        self.last_finish = max(self.last_finish, task.t_finish)

    def on_block(self, task: Task) -> None:
        """Count a dropped task."""
        self.blocked += 1

    def on_round(self, node: int, decided: float, departs: float, size: int) -> None:
        """Count a dispatch round and its decision latency (incl. poll wait and queueing)."""
        self.rounds += 1
        self.decision_latency_sum += departs - decided

    def on_reply(self, node: int, slot: int, kind: MsgKind, n_ids: int, est: float) -> None:
        """Keep ACK/REJECT outcomes for this epoch (B4's bandit needs them)."""
        self.replies.append((node, slot, int(kind), n_ids, est))

    # sampling and summary ------------------------------------------------------------

    def sample(
        self, now: float, lbf: float, counter: MessageCounter, util_mean: float = math.nan
    ) -> None:
        """Append one row to the time series and start a new 1 s window."""
        row: dict[str, float] = {
            "t": now,
            "lbf": lbf,
            "util_mean": util_mean,
            "throughput": self._win_done / self.sample_interval,
            "resp_mean": self._win_resp / self._win_done if self._win_done else math.nan,
            "blocked": self.blocked,
            "dispatch_rounds": self.rounds,
        }
        row.update(counter.as_dict())
        self._rows.append(row)
        self._win_done = 0
        self._win_resp = 0.0

    def timeseries(self) -> pd.DataFrame:
        """The sampled time series as a DataFrame."""
        return pd.DataFrame(self._rows)

    def summary(
        self,
        *,
        n_servers: int,
        created: int,
        outstanding: int,
        lost: int,
        counter: MessageCounter,
        violations: int,
        reruns_after_crash: int = 0,
    ) -> dict[str, Any]:
        """One row of ``summary.csv``."""
        ts = self.timeseries()
        resp = np.asarray(self.response_times)
        completed = len(resp)
        if len(ts):
            late = ts[ts["t"] >= self.horizon / 2]
            lbf_late = float(late["lbf"].mean())
            ttb = time_to_balance(ts["t"].to_numpy(), ts["lbf"].to_numpy(), 0.05, 60.0)
        else:
            lbf_late, ttb = math.nan, math.nan
        return {
            "resp_mean": float(resp.mean()) if completed else math.nan,
            "resp_p95": float(np.percentile(resp, 95)) if completed else math.nan,
            "throughput": self.completed_in_horizon / self.horizon,
            "blocking_rate": self.blocked / created if created else 0.0,
            "lbf_mean": lbf_late,
            "makespan": self.last_finish if outstanding == 0 else math.nan,
            "time_to_balance": ttb,
            "msgs_per_task": counter.total / completed if completed else math.nan,
            "rounds_per_server_s": self.rounds / (n_servers * self.horizon),
            "decision_latency": (
                self.decision_latency_sum / self.rounds if self.rounds else math.nan
            ),
            "lost_in_crash": lost,
            "created": created,
            "completed": completed,
            "blocked": self.blocked,
            "unfinished": outstanding,
            "violations": violations,
            "reruns_after_crash": reruns_after_crash,
            "msgs_total": counter.total,
            **counter.as_dict(),
        }
