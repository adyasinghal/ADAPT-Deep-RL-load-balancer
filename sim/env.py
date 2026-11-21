"""Multi-agent environment API used by the RL code and by the experiment runner.

Interface (agreed between the simulator and the agent workstreams)::

    env = EdgeEnv(cfg, policy)          # or StubEnv(cfg) while the simulator is not ready
    obs, info = env.reset(seed)         # obs: (n, 30) float32, info["masks"]: (n, 7) bool
    obs, rewards, done, info = env.step(actions)   # actions: (n,) int, rewards: (n,) float32

``info`` always has ``masks`` (what each server may do next) and ``time``; the real env also
gives ``protocol_masks``, ``lbf`` (global), ``reward_parts`` (n, 5) and ``replies`` (the
ACK/REJECT outcomes of the last interval). ``env.views`` is the list of read-only node views
passed to ``Policy.act`` as ``ctx``.

Each ``step`` applies the actions at the current decision instant, advances the simulation by
one decision interval (0.25 s), hands that interval's arrivals to the buffers and returns the
observation at the next instant together with every server's reward for the interval.

    python -m sim.env --stub --steps 100                # exercise the stub
    python -m sim.env --policy evblb --duration 300     # one real run, prints the summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.actions import FIRST_SEND, HOLD, LOCAL, N_ACTIONS, N_FEATURES, N_SLOTS
from core.config import load_config
from core.invariants import InvariantChecker
from core.model import Task, TaskState, Topology
from core.node import NodeView
from sim.workloads import Workload, make_workload
from sim.world import World, build_world

CPU_VARIANCE_REF = 67500.0  # variance of U(100, 1000); normalises the ATOA stability feature
VARIANCE_EPOCHS = 10
EWMA_ALPHA = 0.2


@dataclass(frozen=True)
class PolicyTraits:
    """The policy attributes the env needs (see ``policies.base`` for what each one means)."""

    max_hold: float = 2.0
    needs_poll: bool = False
    uses_cache: bool = False
    fresh_obs: bool = False
    capacity_mask: bool = False

    @classmethod
    def of(cls, policy: Any) -> PolicyTraits:
        """Read the traits off a policy object, falling back to the defaults."""
        if policy is None:
            return cls()
        return cls(
            max_hold=float(getattr(policy, "max_hold", cls.max_hold)),
            needs_poll=bool(getattr(policy, "needs_poll", False)),
            uses_cache=bool(getattr(policy, "uses_cache", False)),
            fresh_obs=bool(getattr(policy, "fresh_obs", False)),
            capacity_mask=bool(getattr(policy, "capacity_mask", False)),
        )


class EdgeEnv:
    """The simulator behind the multi-agent RL interface.

    ``topology`` pins a hand-made topology (tests); otherwise one is drawn from the seed.
    ``strict`` overrides ``invariants.strict``.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        policy: Any = None,
        topology: Topology | None = None,
        strict: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.traits = PolicyTraits.of(policy)
        self._fixed_topology = topology
        self._strict = strict
        run = cfg["run"]
        self.dt = float(run["decide_interval"])
        self.duration = float(run["duration"])
        self.cutoff = self.duration - float(run["arrival_cutoff"])
        self.drain = bool(run["drain"])
        self.drain_limit = float(run["drain_limit"])
        self.sample_every = max(1, round(float(run["sample_interval"]) / self.dt))
        r = cfg["reward"]
        self.reward_weights = np.array(
            [r["w_wait"], r["w_lbf"], r["w_dispatch"], r["w_distance"], r["w_blocked"]],
            dtype=np.float32,
        )
        self.wait_ref = float(r["wait_ref"])
        self.dispatch_ref = float(r["dispatch_ref"])
        self.world: World | None = None
        self.views: list[NodeView] = []
        self.policy_rng = np.random.default_rng(0)
        self.lbf = 0.0
        self.util_mean = 0.0

    # ------------------------------------------------------------------ properties

    @property
    def n(self) -> int:
        """Number of edge servers."""
        return self._w.topology.n

    @property
    def now(self) -> float:
        """Current simulated time."""
        return self._w.engine.now

    @property
    def topology(self) -> Topology:
        """Topology of the current episode."""
        return self._w.topology

    @property
    def _w(self) -> World:
        if self.world is None:
            raise RuntimeError("call reset() first")
        return self.world

    # ------------------------------------------------------------------ API

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Build a fresh world for ``seed`` and return the observation at t = 0."""
        streams = np.random.SeedSequence(seed).spawn(6)
        topo_rng, arr_rng, wl_rng, loss_rng, fault_rng, policy_rng = (
            np.random.default_rng(s) for s in streams
        )
        self.policy_rng = policy_rng
        topology = self._fixed_topology or Topology.generate(self.cfg, topo_rng)
        w = build_world(self.cfg, topology, loss_rng, self._strict)
        self.world = w
        self.views = w.views
        n = topology.n
        self._checker = InvariantChecker(
            w.ledger, w.params.queue_capacity, self.traits.max_hold, self.dt
        )
        self._workload: Workload = make_workload(self.cfg, topology, wl_rng)
        self._arr_rng = arr_rng
        self._k = 0
        self._next_id = 0

        # one (n, 5) array per cache field, shared with the nodes so we don't copy each epoch
        self._cache_prem = np.stack([nd.cache_prem for nd in w.nodes])
        self._cache_util = np.stack([nd.cache_util for nd in w.nodes])
        self._cache_time = np.stack([nd.cache_time for nd in w.nodes])
        for i, nd in enumerate(w.nodes):
            nd.cache_prem = self._cache_prem[i]
            nd.cache_util = self._cache_util[i]
            nd.cache_time = self._cache_time[i]
        self._nbr = topology.neighbours
        self._dist_norm = (topology.neighbour_dist / topology.psi).astype(np.float32)
        self._cpu = topology.cpu_ghz
        self._max_mem_mb = float(topology.mem_gb.max()) * 1000.0
        self._max_disk_mb = float(topology.disk_gb.max()) * 1000.0
        self._alive = np.ones(n, dtype=bool)

        # traffic features (6-8)
        self._rate_ewma = np.zeros(n)
        self._last_cpu = np.zeros(n)
        self._cpu_hist = np.full((n, VARIANCE_EPOCHS), np.nan)

        faults = self.cfg["faults"]
        if faults["crash_time"] is not None and faults["crash_fraction"] > 0:
            k = int(round(float(faults["crash_fraction"]) * n))
            victims = fault_rng.choice(n, size=k, replace=False)
            w.engine.call_at(float(faults["crash_time"]), self._crash, victims.tolist())

        if self.traits.uses_cache:
            for nd in w.nodes:
                nd.send_heartbeats()
        obs, masks = self._observe()
        return obs, self._info(masks)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        """Apply one action per server, advance one decision interval and observe."""
        w = self._w
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError(f"expected {self.n} actions, got shape {actions.shape}")
        self._apply(actions)
        t_prev = self.now
        self._k += 1
        t = self._k * self.dt
        w.engine.run_until(t)
        self._arrivals(t_prev, t)
        if self.traits.uses_cache:
            for nd in w.nodes:
                nd.send_heartbeats()
        obs, masks = self._observe()
        rewards, parts = self._rewards()
        done = self._is_done(t)
        if (self._k % self.sample_every == 0 and t <= self.duration + 1e-9) or done:
            if t <= self.duration + 1e-9:
                w.recorder.sample(t, self.lbf, w.counter, self.util_mean)
            self._checker.check(t, w.nodes, w.cloud)
        info = self._info(masks)
        info["reward_parts"] = parts
        return obs, rewards, done, info

    def timeseries(self) -> pd.DataFrame:
        """Per-second time series of the episode so far."""
        return self._w.recorder.timeseries()

    def summary(self) -> dict[str, Any]:
        """Summary metrics of the episode (call after ``done``)."""
        w = self._w
        return w.recorder.summary(
            n_servers=self.n,
            created=w.ledger.created,
            outstanding=w.ledger.outstanding,
            lost=w.ledger.count(TaskState.LOST),
            counter=w.counter,
            violations=w.ledger.total_violations,
            reruns_after_crash=w.ledger.reruns_after_crash,
        )

    # ------------------------------------------------------------------ internals

    def _apply(self, actions: np.ndarray) -> None:
        w = self._w
        legal = self._pmask[np.arange(self.n), actions]
        if not legal.all():
            bad = np.flatnonzero(~legal)
            raise ValueError(
                f"actions forbidden by the protocol for servers {bad.tolist()}: "
                f"{actions[bad].tolist()}"
            )
        poll = self.traits.needs_poll
        for i in np.flatnonzero(actions != HOLD).tolist():
            node = w.nodes[i]
            wait = 0.0
            if poll:
                node.send_polls()
                wait = float(w.costs.poll_wait[i])
            node.start_dispatch(int(actions[i]), wait)

    def _crash(self, victims: list[int]) -> None:
        for i in victims:
            self._w.nodes[i].crash()
            self._alive[i] = False

    def _arrivals(self, t0: float, t1: float) -> None:
        """Draw the arrivals of (t0, t1] and give them to their home servers' buffers."""
        w = self._w
        n = self.n
        counts = np.zeros(n)
        cpu_sum = np.zeros(n)
        hi = min(t1, self.cutoff)
        k = 0
        if hi > t0:
            k = int(self._arr_rng.poisson(self._workload.rate(t0) * (hi - t0)))
        if k:
            times = np.sort(t0 + self._arr_rng.random(k) * (hi - t0))
            pos, mem, disk, cpu = self._workload.sample(t0, k)
            alive = None if self._alive.all() else self._alive
            homes = w.topology.home_of(pos, alive)
            counts = np.bincount(homes, minlength=n).astype(float)
            cpu_sum = np.bincount(homes, weights=cpu, minlength=n)
            per_home: list[list[Task]] = [[] for _ in range(n)]
            register = w.ledger.register
            max_mem, max_disk = self._max_mem_mb, self._max_disk_mb
            for home, ta, m, d, c in zip(
                homes.tolist(),
                times.tolist(),
                mem.tolist(),
                disk.tolist(),
                cpu.tolist(),
                strict=True,
            ):
                task = Task(self._next_id, home, ta, m, d, c)
                self._next_id += 1
                register(task)
                if m > max_mem or d > max_disk:
                    w.cloud.submit(task)  # fits no edge server (EVBLB Alg. 1, lines 1-8)
                else:
                    per_home[home].append(task)
            for i, tasks in enumerate(per_home):
                if tasks:
                    w.nodes[i].accept_arrivals(tasks)

        self._rate_ewma = EWMA_ALPHA * counts / self.dt + (1 - EWMA_ALPHA) * self._rate_ewma
        has = counts > 0
        mean_cpu = np.where(has, cpu_sum / np.maximum(counts, 1), np.nan)
        self._cpu_hist[:, self._k % VARIANCE_EPOCHS] = mean_cpu
        self._last_cpu = np.where(has, mean_cpu, 0.0)

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the (n, 30) observation and the masks at the current instant."""
        w = self._w
        now = self.now
        n = self.n
        traits = self.traits
        pmask = np.zeros((n, N_ACTIONS), dtype=bool)
        util = np.zeros(n)
        prem = np.zeros(n)
        qlen = np.zeros(n)
        bsize = np.zeros(n)
        bcpu = np.zeros(n)
        age = np.zeros(n)
        usage = np.zeros(n)
        suspected = np.zeros((n, N_SLOTS), dtype=bool)
        for i, nd in enumerate(w.nodes):
            pmask[i] = nd.protocol_mask(traits.max_hold)
            util[i] = nd.utilisation()
            prem[i] = nd.p_rem()
            qlen[i] = nd.q_count
            suspected[i] = nd.suspected
            buf = nd.buffer
            if buf:
                bsize[i] = len(buf)
                bcpu[i] = sum(t.cpu for t in buf)
                age[i] = now - nd.oldest_arrival()
                if traits.capacity_mask:
                    usage[i] = nd.batch_usage()

        cpu = self._cpu
        obs = np.zeros((n, N_FEATURES), dtype=np.float32)
        obs[:, 0] = util
        obs[:, 1] = qlen / w.params.queue_capacity
        obs[:, 2] = cpu / 4.0
        obs[:, 3] = np.minimum(bsize / 100.0, 1.0)
        obs[:, 4] = np.minimum(bcpu / (1000.0 * cpu * 2.0), 1.0)
        obs[:, 5] = np.minimum(age / 2.0, 1.0)
        mean_cpu = float(self.cfg["tasks"]["mean_cpu_mcycles"])
        obs[:, 6] = np.clip(self._rate_ewma * mean_cpu / (1000.0 * cpu), 0.0, 2.0) / 2.0
        obs[:, 7] = self._last_cpu / 1000.0
        obs[:, 8] = np.clip(_nan_variance(self._cpu_hist) / CPU_VARIANCE_REF, 0.0, 1.0)

        exists = (self._nbr >= 0) & ~suspected
        safe_nbr = np.where(self._nbr >= 0, self._nbr, 0)
        if traits.fresh_obs:
            nb_prem, nb_util = prem[safe_nbr], util[safe_nbr]
            nb_age = np.zeros((n, N_SLOTS))
        else:
            nb_prem, nb_util = self._cache_prem, self._cache_util
            with np.errstate(invalid="ignore"):
                nb_age = np.clip((now - self._cache_time) / 2.0, 0.0, 1.0)
        nb_prem = np.where(exists, nb_prem, 0.0)
        nb_util = np.where(exists, nb_util, 0.0)
        nb_age = np.where(exists, nb_age, 0.0)

        # local LBF: std of our own utilisation and the (cached) ones of our neighbours
        vals = np.concatenate([util[:, None], nb_util], axis=1)
        valid = np.concatenate([np.ones((n, 1), dtype=bool), exists], axis=1)
        cnt = valid.sum(axis=1)
        mean = (vals * valid).sum(axis=1) / cnt
        lbf_loc = np.sqrt((((vals - mean[:, None]) ** 2) * valid).sum(axis=1) / cnt)
        obs[:, 9] = lbf_loc

        slots = np.stack([np.clip(nb_prem, -1.0, 1.0), nb_age, self._dist_norm, exists], axis=2)
        obs[:, 10:] = slots.reshape(n, 4 * N_SLOTS)

        masks = pmask.copy()
        if traits.capacity_mask:
            masks[:, FIRST_SEND:] &= nb_prem >= usage[:, None]
        self._pmask = pmask
        self._lbf_loc = lbf_loc
        if self._alive.any():
            self.lbf = float(np.std(util[self._alive]))
            self.util_mean = float(util[self._alive].mean())
        else:
            self.lbf = self.util_mean = 0.0
        return obs, masks

    def _rewards(self) -> tuple[np.ndarray, np.ndarray]:
        """Reward of every server for the interval that just ended (brief section 6)."""
        w = self._w
        psi = w.topology.psi
        parts = np.zeros((self.n, 5), dtype=np.float32)
        for i, nd in enumerate(w.nodes):
            wait, dispatched, cost, dist, blocked = nd.take_epoch_stats()
            parts[i, 0] = wait / self.wait_ref
            parts[i, 2] = cost / self.dispatch_ref if dispatched else 0.0
            parts[i, 3] = dist / psi
            parts[i, 4] = blocked
        parts[:, 1] = self._lbf_loc
        return -(parts @ self.reward_weights), parts

    def _is_done(self, t: float) -> bool:
        if t < self.duration - 1e-9:
            return False
        if not self.drain:
            return True
        return self._w.ledger.outstanding == 0 or t >= self.duration + self.drain_limit - 1e-9

    def _info(self, masks: np.ndarray) -> dict[str, Any]:
        w = self._w
        replies, w.recorder.replies = w.recorder.replies, []
        return {
            "masks": masks,
            "protocol_masks": self._pmask.copy(),
            "time": self.now,
            "lbf": self.lbf,
            "replies": replies,
        }


def _nan_variance(hist: np.ndarray) -> np.ndarray:
    """Row-wise variance ignoring NaNs; 0 for rows with fewer than two values."""
    valid = ~np.isnan(hist)
    cnt = valid.sum(axis=1)
    filled = np.where(valid, hist, 0.0)
    mean = filled.sum(axis=1) / np.maximum(cnt, 1)
    var = (np.where(valid, (hist - mean[:, None]) ** 2, 0.0)).sum(axis=1) / np.maximum(cnt, 1)
    return np.where(cnt >= 2, var, 0.0)


class StubEnv:
    """Random stand-in for the real env with exactly the same shapes and API.

    Lets the agent code be written and smoke-tested before the simulator exists. Observations
    and rewards are noise; masks are random but HOLD and LOCAL are always allowed so every row
    has a legal action.
    """

    def __init__(self, cfg: dict[str, Any], horizon: int = 400) -> None:
        self.n = int(cfg["topology"]["n_servers"])
        self.dt = float(cfg["run"]["decide_interval"])
        self.horizon = horizon
        self.views: list = []
        self._rng = np.random.default_rng(0)
        self._step = 0
        self._masks = np.ones((self.n, N_ACTIONS), dtype=bool)

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode; the same seed gives the same sequence."""
        self._rng = np.random.default_rng(seed)
        self._step = 0
        return self._obs(), self._info()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        """Check the actions against the current masks and return random feedback."""
        actions = np.asarray(actions)
        if actions.shape != (self.n,):
            raise ValueError(f"expected {self.n} actions, got shape {actions.shape}")
        illegal = ~self._masks[np.arange(self.n), actions]
        if illegal.any():
            raise ValueError(f"masked actions chosen by servers {np.flatnonzero(illegal)}")
        self._step += 1
        rewards = -self._rng.random(self.n).astype(np.float32)
        done = self._step >= self.horizon
        return self._obs(), rewards, done, self._info()

    def _obs(self) -> np.ndarray:
        return self._rng.random((self.n, N_FEATURES), dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        self._masks = self._rng.random((self.n, N_ACTIONS)) < 0.6
        self._masks[:, HOLD] = True
        self._masks[:, LOCAL] = True
        return {"masks": self._masks.copy(), "time": self._step * self.dt}


def _run_stub(steps: int, seed: int) -> None:
    from policies.random_policy import RandomPolicy

    cfg = load_config()
    env = StubEnv(cfg, horizon=steps)
    policy = RandomPolicy()
    policy.reset(None, np.random.default_rng(seed))
    obs, info = env.reset(seed)
    total = 0.0
    done = False
    while not done:
        actions = policy.act(obs, info["masks"], env.views)
        obs, rewards, done, info = env.step(actions)
        total += float(rewards.mean())
    print(f"stub env: n={env.n} obs={obs.shape} masks={info['masks'].shape}")
    print(f"ran {steps} steps to t={info['time']:.2f}s, mean reward {total / steps:.3f}")


def _run_real(args: argparse.Namespace) -> None:
    from policies import make_policy
    from sim.rollout import run_episode

    dotted = {
        "run.duration": args.duration,
        "workload.name": args.workload,
        "workload.rho": args.rho,
        "topology.n_servers": args.n,
    }
    cfg = load_config(dotted=dotted)
    spec: dict[str, Any] = {"name": args.policy}
    if args.dt is not None:
        spec["dt"] = args.dt
    _, summary = run_episode(cfg, make_policy(spec), args.seed)
    print(json.dumps(summary, indent=2, default=float))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the stub env or one simulation.")
    parser.add_argument("--stub", action="store_true", help="run the random stub env")
    parser.add_argument("--steps", type=int, default=100, help="stub: number of steps")
    parser.add_argument("--policy", default="evblb")
    parser.add_argument("--dt", type=float, default=None, help="EVBLB interval")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--workload", default="W1")
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if args.stub:
        _run_stub(args.steps, args.seed)
    else:
        _run_real(args)


if __name__ == "__main__":
    main()
