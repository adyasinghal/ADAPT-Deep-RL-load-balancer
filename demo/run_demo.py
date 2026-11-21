"""Live demo: eight edge servers running ADAPT as independent asyncio tasks.

    python -m demo.run_demo                      # 60 s, real time, live terminal view
    python -m demo.run_demo --speed 4            # four simulated seconds per real second
    python -m demo.run_demo --no-tty --crash-at 20 --crash-node 3   # scripted, plain log

Each server is a ``ServerNode`` (the same class the simulator uses) with its own inbox task and
its own decision task. Every 0.25 s a server builds its 30 local features, runs its own copy of
the trained network (numpy, from ``weights/adapt.npz``) and holds or dispatches. Messages go
through ``AsyncTransport`` with the simulator's delays; nothing is shared between servers except
the network they talk over.

In the live view press 0-7 to crash that server (rule 1 then takes its unanswered tasks back),
or q to quit. The invariant checker runs every second and its violation count is shown.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.actions import ACTION_NAMES, HOLD
from core.config import ROOT, load_config
from core.features import LocalTraffic, node_observation
from core.invariants import InvariantChecker, TaskLedger
from core.messages import MessageCounter
from core.metrics import MetricsRecorder, load_balancing_factor
from core.model import CostModel, Indicators, Task, TaskState, Topology
from core.node import CloudServer, NodeContext, OracleDetector, ProtocolParams, ServerNode
from core.transport import AsyncTransport
from rl.numpy_net import NumpyQNet, greedy
from sim.workloads import rate_for_load

DECIDE = 0.25


class AsyncClock:
    """Simulated time on top of the asyncio loop; ``speed`` simulated seconds per real second."""

    def __init__(self, loop: asyncio.AbstractEventLoop, speed: float) -> None:
        self.loop = loop
        self.speed = speed
        self._start = loop.time()

    @property
    def now(self) -> float:
        """Simulated seconds since the demo started."""
        return (self.loop.time() - self._start) * self.speed

    def call_at(self, t: float, fn, *args) -> None:
        """Run ``fn(*args)`` at simulated time ``t``."""
        self.loop.call_at(self._start + t / self.speed, fn, *args)

    async def sleep_until(self, t: float) -> None:
        """Wait until simulated time ``t``."""
        await asyncio.sleep(max(0.0, (t - self.now) / self.speed))


@dataclass
class DemoSettings:
    """Command-line settings of one demo run."""

    n: int = 8
    duration: float = 60.0
    speed: float = 1.0
    rho: float = 0.7
    seed: int = 0
    area: float = 90.0  # smaller than the 200 x 200 of the experiments so 8 servers are neighbours
    weights: str = str(ROOT / "weights" / "adapt.npz")
    crash_at: float | None = None
    crash_node: int = 0
    p_loss: float = 0.0
    tty: bool = True
    drain_limit: float = 120.0


class Demo:
    """All the pieces of one demo run."""

    def __init__(self, settings: DemoSettings, loop: asyncio.AbstractEventLoop) -> None:
        s = settings
        self.s = s
        cfg = load_config(dotted={"topology.n_servers": s.n, "topology.area": s.area})
        self.cfg = cfg
        rng = np.random.default_rng(s.seed)
        self.rng = rng
        self.topology = Topology.generate(cfg, rng)
        self.clock = AsyncClock(loop, s.speed)
        self.counter = MessageCounter()
        costs = CostModel(cfg, self.topology)
        self.transport = AsyncTransport(self.clock, costs, self.counter, rng, s.p_loss)
        self.ledger = TaskLedger(strict=False)
        self.recorder = MetricsRecorder(horizon=s.duration)
        params = ProtocolParams.from_config(cfg)
        self.nodes: list[ServerNode] = []
        ctx = NodeContext(
            clock=self.clock,
            transport=self.transport,
            ledger=self.ledger,
            costs=costs,
            indicators=Indicators.from_config(cfg),
            params=params,
            recorder=self.recorder,
            detector=OracleDetector(self.nodes),
        )
        self.nodes.extend(ServerNode(i, self.topology, ctx) for i in range(s.n))
        self.transport.attach(self.nodes)
        self.cloud = CloudServer(cfg["cloud"]["cpu_ghz"], costs.cloud_latency, ctx)
        # asyncio timers can fire a few ms late; allow for that in the P2 bound
        jitter = 0.02 * s.speed
        self.checker = InvariantChecker(self.ledger, params.queue_capacity, 2.0, DECIDE + jitter)
        self.traffic = [LocalTraffic() for _ in self.nodes]
        self.cell_arrivals = np.zeros((s.n, 2))  # (count, cpu) since the last decision
        self.last_action = ["-"] * s.n
        self.rate = rate_for_load(cfg, self.topology, s.rho)
        self.next_id = 0
        self.arrivals_open = True
        self.stop = asyncio.Event()

    # ------------------------------------------------------------------ server tasks

    async def decide_loop(self, i: int) -> None:
        """Server i on its own: every 0.25 s observe locally, run its network, act."""
        node, traffic = self.nodes[i], self.traffic[i]
        net = NumpyQNet(self.s.weights)  # every server loads its own copy
        k = 1
        while not self.stop.is_set():
            await self.clock.sleep_until(k * DECIDE)
            k += 1
            if node.crashed:
                self.last_action[i] = "crashed"
                continue
            count, cpu = self.cell_arrivals[i]
            self.cell_arrivals[i] = 0.0
            traffic.end_epoch(int(count), float(cpu), DECIDE)
            node.send_heartbeats()
            obs, mask = node_observation(node, traffic)
            action = int(greedy(net(obs[None, :]), mask[None, :])[0])
            if action != HOLD:
                node.start_dispatch(action)
            if node.buffer or action != HOLD:
                self.last_action[i] = ACTION_NAMES[action]

    async def arrivals(self) -> None:
        """Poisson arrivals at uniform positions, handed to their home (nearest live) server."""
        rng = self.rng
        t = 0.0
        while self.arrivals_open and not self.stop.is_set():
            t += rng.exponential(1.0 / self.rate)
            if t >= self.s.duration:
                break
            await self.clock.sleep_until(t)
            alive = np.array([not nd.crashed for nd in self.nodes])
            home = int(self.topology.home_of(rng.uniform(0, self.s.area, (1, 2)), alive)[0])
            task = Task(
                self.next_id,
                home,
                self.clock.now,
                rng.uniform(100, 500),
                rng.uniform(100, 1000),
                rng.uniform(100, 1000),
            )
            self.next_id += 1
            self.ledger.register(task)
            self.nodes[home].accept_arrivals([task])
            self.cell_arrivals[home] += (1, task.cpu)
        self.arrivals_open = False

    async def checks(self) -> None:
        """Invariant checks once a simulated second."""
        k = 1
        while not self.stop.is_set():
            await self.clock.sleep_until(k)
            k += 1
            self.checker.check(self.clock.now, self.nodes, self.cloud)

    def crash(self, i: int) -> None:
        """Crash server i (from a key press or --crash-at)."""
        if 0 <= i < len(self.nodes) and not self.nodes[i].crashed:
            self.nodes[i].crash()

    # ------------------------------------------------------------------ views

    def status(self) -> dict[str, Any]:
        """Numbers shown in the live view and the final summary."""
        alive = [nd for nd in self.nodes if not nd.crashed]
        util = np.array([nd.utilisation() for nd in alive])
        resp = self.recorder.response_times[-200:]
        done = len(self.recorder.response_times)
        return {
            "t": self.clock.now,
            "lbf": load_balancing_factor(util),
            "created": self.ledger.created,
            "completed": done,
            "blocked": self.recorder.blocked,
            "lost": self.ledger.count(TaskState.LOST),
            "outstanding": self.ledger.outstanding,
            "resp_recent": float(np.mean(resp)) if resp else math.nan,
            "msgs_per_task": self.counter.total / done if done else math.nan,
            "violations": self.ledger.total_violations,
            "reruns": self.ledger.reruns_after_crash,
        }

    def render(self) -> str:
        """Full-screen text view: one row per server plus the totals."""
        st = self.status()
        lines = [
            f"ADAPT demo   t = {st['t']:6.1f} s   speed x{self.s.speed:g}   rho {self.s.rho:g}"
            f"   {'arrivals on' if self.arrivals_open else 'draining'}",
            "",
            " srv  GHz   queue                            util  buffer  last action",
        ]
        for i, nd in enumerate(self.nodes):
            if nd.crashed:
                lines.append(f"  {i}   {nd.cpu_ghz:.1f}   CRASHED")
                continue
            bar = "#" * min(30, nd.q_count)
            lines.append(
                f"  {i}   {nd.cpu_ghz:.1f}   {nd.q_count:4d} {bar:<30s} {nd.utilisation():5.2f}"
                f"  {len(nd.buffer):5d}   {self.last_action[i]}"
            )
        lines += [
            "",
            f"LBF {st['lbf']:.3f}   completed {st['completed']}/{st['created']}"
            f"   recent response {st['resp_recent']:.2f} s   msgs/task {st['msgs_per_task']:.2f}",
            f"blocked {st['blocked']}   lost in crash {st['lost']}   re-runs {st['reruns']}"
            f"   invariant violations {st['violations']}",
            "",
            "keys: 0-7 crash that server, q quit",
        ]
        return "\n".join(lines)

    async def display(self) -> None:
        """Redraw the live view (tty) or print a status line every 5 s (plain)."""
        every = 0.5 if self.s.tty else 5.0
        k = 1
        while not self.stop.is_set():
            await self.clock.sleep_until(k * every * (1 if self.s.tty else self.s.speed))
            k += 1
            if self.s.tty:
                sys.stdout.write("\x1b[H\x1b[2J" + self.render() + "\n")
                sys.stdout.flush()
            else:
                st = self.status()
                print(
                    f"t={st['t']:6.1f}s  LBF {st['lbf']:.3f}  done {st['completed']}/"
                    f"{st['created']}  resp {st['resp_recent']:.2f}s  "
                    f"violations {st['violations']}",
                    flush=True,
                )


async def run_demo(settings: DemoSettings) -> dict[str, Any]:
    """Run the demo to the end (arrivals stop at ``duration``, then it drains)."""
    loop = asyncio.get_running_loop()
    demo = Demo(settings, loop)
    tasks = [asyncio.create_task(demo.transport.serve(i)) for i in range(settings.n)]
    tasks += [asyncio.create_task(demo.decide_loop(i)) for i in range(settings.n)]
    tasks += [asyncio.create_task(demo.checks()), asyncio.create_task(demo.display())]
    arrivals = asyncio.create_task(demo.arrivals())
    if settings.crash_at is not None:
        demo.clock.call_at(settings.crash_at, demo.crash, settings.crash_node)
    restore = _keyboard(loop, demo) if settings.tty else None
    try:
        await arrivals
        deadline = settings.duration + settings.drain_limit
        while demo.ledger.outstanding and demo.clock.now < deadline and not demo.stop.is_set():
            await asyncio.sleep(0.05)
    finally:
        demo.stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if restore:
            restore()
    demo.checker.check(demo.clock.now, demo.nodes, demo.cloud)
    return demo.status()


def _keyboard(loop: asyncio.AbstractEventLoop, demo: Demo):
    """Read single key presses without Enter; returns a function that restores the terminal."""
    if not sys.stdin.isatty():
        return None
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def on_key() -> None:
        key = os.read(fd, 1).decode(errors="ignore")
        if key.isdigit():
            demo.crash(int(key))
        elif key.lower() == "q":
            demo.arrivals_open = False
            demo.stop.set()

    loop.add_reader(fd, on_key)

    def restore() -> None:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    return restore


def main() -> None:
    """Command-line entry point; exits with status 1 if an invariant was violated."""
    parser = argparse.ArgumentParser(description="Live ADAPT demo over asyncio message passing.")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals")
    parser.add_argument("--speed", type=float, default=1.0, help="simulated s per real s")
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default=str(ROOT / "weights" / "adapt.npz"))
    parser.add_argument("--crash-at", type=float, help="crash a server at this time")
    parser.add_argument("--crash-node", type=int, default=0)
    parser.add_argument("--p-loss", type=float, default=0.0)
    parser.add_argument("--no-tty", action="store_true", help="plain log lines, no live view")
    args = parser.parse_args()
    settings = DemoSettings(
        duration=args.duration,
        speed=args.speed,
        rho=args.rho,
        seed=args.seed,
        weights=args.weights,
        crash_at=args.crash_at,
        crash_node=args.crash_node,
        p_loss=args.p_loss,
        tty=not args.no_tty and sys.stdout.isatty(),
    )
    summary = asyncio.run(run_demo(settings))
    print(
        f"\nfinished at t = {summary['t']:.1f} s: {summary['completed']} of "
        f"{summary['created']} tasks done, {summary['blocked']} blocked, "
        f"{summary['lost']} lost in the crash, {summary['outstanding']} unfinished, "
        f"{summary['violations']} invariant violations"
    )
    sys.exit(1 if summary["violations"] else 0)


if __name__ == "__main__":
    main()
