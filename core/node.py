"""ServerNode: one edge server and the dispatch protocol it runs.

The same class runs inside the simulator and (later) inside the asyncio demo. A node never
reads another node's state; all it knows about its neighbours comes from messages. Timers go
through a ``Clock`` and messages through a ``Transport``, which are the only two things that
differ between the simulator and the demo.

Protocol rules from the brief, and where they live:

1. OFFERed tasks stay in ``inflight`` until ACK/REJECT; on timeout the same OFFER is resent
   to the same server; after ``max_retries`` resends we ask the failure detector and, if the
   target really crashed, take the tasks back (``_on_timeout``, ``_give_up``).
2. The receiver de-duplicates by task id (``_on_offer``, ``seen``).
3. Admission uses the receiver's true queue length and capacities (``_on_offer``).
4. A REJECT sends the task back to the buffer with ``h + 1``; at ``h >= hop_limit`` only
   LOCAL is allowed (``_on_reply``, ``protocol_mask``).
5. HOLD is masked once the oldest buffered task has waited ``max_hold`` (``protocol_mask``).
6. Only buffered tasks are dispatched; queued tasks never move (nothing ever dequeues a task
   except to run it).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from core.actions import FIRST_SEND, HOLD, LOCAL, N_ACTIONS, N_SLOTS, action_slot
from core.invariants import TaskLedger
from core.messages import LOAD_REPORTS, Message, MsgKind
from core.model import CostModel, Indicators, Task, TaskState, Topology
from core.transport import Transport

S = TaskState


class Clock(Protocol):
    """Current time plus timers (the sim ``Engine`` implements this)."""

    now: float

    def call_at(self, t: float, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` at time ``t``."""


class Recorder(Protocol):
    """Hooks the metrics module implements; the node calls them as things happen."""

    def on_complete(self, task: Task) -> None:
        """A task finished."""

    def on_block(self, task: Task) -> None:
        """A task was dropped because a queue was full."""

    def on_round(self, node: int, decided: float, departs: float, size: int) -> None:
        """A dispatch round of ``size`` tasks was decided and will end at ``departs``."""

    def on_reply(self, node: int, slot: int, kind: MsgKind, n_ids: int, est: float) -> None:
        """An ACK or REJECT for ``n_ids`` tasks came back from neighbour ``slot``."""


class FailureDetector(Protocol):
    """Answers whether a server has crashed."""

    def is_crashed(self, idx: int) -> bool:
        """True if server ``idx`` is considered crashed."""


@dataclass(frozen=True)
class ProtocolParams:
    """Protocol constants from the ``protocol`` and ``servers`` config sections."""

    queue_capacity: int = 500
    hop_limit: int = 2
    max_retries: int = 3
    timeout_slack: float = 0.05
    heartbeat_period: float = 2.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ProtocolParams:
        """Read the relevant config keys."""
        p = cfg["protocol"]
        return cls(
            queue_capacity=int(cfg["servers"]["queue_capacity"]),
            hop_limit=int(p["hop_limit"]),
            max_retries=int(p["max_retries"]),
            timeout_slack=float(p["timeout_slack"]),
            heartbeat_period=float(p["heartbeat_period"]),
        )


@dataclass
class NodeContext:
    """Everything the nodes of one run share."""

    clock: Clock
    transport: Transport
    ledger: TaskLedger
    costs: CostModel
    indicators: Indicators
    params: ProtocolParams
    recorder: Recorder
    detector: FailureDetector


class DispatchRound:
    """A batch taken out of the buffer that leaves when the dispatch round ends."""

    __slots__ = ("tasks", "action", "decided_at", "departs_at")

    def __init__(self, tasks: list[Task], action: int, decided_at: float, departs_at: float):
        self.tasks = tasks
        self.action = action
        self.decided_at = decided_at
        self.departs_at = departs_at


class Offer:
    """Tasks sent to one neighbour and not yet acknowledged or rejected."""

    __slots__ = ("offer_id", "target", "slot", "pending", "attempt")

    def __init__(self, offer_id: int, target: int, slot: int, tasks: list[Task]):
        self.offer_id = offer_id
        self.target = target
        self.slot = slot
        self.pending: dict[int, Task] = {t.id: t for t in tasks}
        self.attempt = 0


class ServerNode:
    """One edge server: buffer, FIFO queue, in-flight offers and a neighbour load cache."""

    def __init__(self, idx: int, topology: Topology, ctx: NodeContext) -> None:
        self.idx = idx
        self.ctx = ctx
        self.cpu_ghz = float(topology.cpu_ghz[idx])
        self.mem_mb = float(topology.mem_gb[idx]) * 1000.0
        self.disk_mb = float(topology.disk_gb[idx]) * 1000.0
        ind = ctx.indicators
        self.capacity = float(
            ind.capacity(topology.mem_gb[idx], topology.disk_gb[idx], topology.cpu_ghz[idx])
        )
        self.neighbours = [int(j) for j in topology.neighbours[idx]]
        self.neighbour_dist = [float(d) for d in topology.neighbour_dist[idx]]
        self.slot_of = {j: s for s, j in enumerate(self.neighbours) if j >= 0}
        self.reverse = list(topology.reverse[idx])
        self.crashed = False

        self.buffer: list[Task] = []
        self.rounds: deque[DispatchRound] = deque()
        self.tasks_in_rounds = 0
        self.queue: deque[Task] = deque()
        self.running: Task | None = None
        self.q_count = 0  # queue + running, the number admission compares to Q_max
        self.q_mem = 0.0
        self.q_disk = 0.0
        self.q_cpu = 0.0
        self.inflight: dict[int, Offer] = {}
        self.seen: set[int] = set()  # ids admitted here, for de-duplication (rule 2)
        self._next_offer = 0
        self._round_free_at = 0.0
        self.last_dispatch = 0.0

        # Load cache: what we last heard from each neighbour slot, and when (our clock).
        # A neighbour we never heard from is assumed empty, i.e. P_rem = its capacity.
        self.cache_prem = np.zeros(N_SLOTS)
        self.cache_util = np.zeros(N_SLOTS)
        self.cache_time = np.full(N_SLOTS, -np.inf)
        for s, j in enumerate(self.neighbours):
            if j >= 0:
                self.cache_prem[s] = ind.capacity(
                    topology.mem_gb[j], topology.disk_gb[j], topology.cpu_ghz[j]
                )
        self.suspected = [False] * N_SLOTS
        self._last_sent: dict[int, float] = {}

        # per-epoch bookkeeping for the RL reward; read and reset by take_epoch_stats()
        self._wait_area = 0.0
        self._last_touch = 0.0
        self.epoch_blocked = 0
        self.epoch_dispatch_cost = 0.0
        self.epoch_dispatched = False
        self.epoch_sent_dist = 0.0

    # ------------------------------------------------------------------ state queries

    @property
    def now(self) -> float:
        """Current time on this node's clock."""
        return self.ctx.clock.now

    def p_rem(self) -> float:
        """Remaining-resource indicator P_rem = P_e - usage of everything queued (EVBLB Eq. 5)."""
        used = self.ctx.indicators.usage(self.q_mem, self.q_disk, self.q_cpu)
        return self.capacity - used

    def utilisation(self) -> float:
        """u_i over the fixed dt_ref window; the running task counts in full."""
        return float(self.ctx.indicators.utilisation(self.q_cpu, self.cpu_ghz))

    def oldest_arrival(self) -> float:
        """Arrival time of the oldest buffered task (inf if the buffer is empty)."""
        return min((t.t_arrive for t in self.buffer), default=np.inf)

    def batch_usage(self) -> float:
        """How much P_rem the current buffer would use up at its destination."""
        mem = sum(t.mem for t in self.buffer)
        disk = sum(t.disk for t in self.buffer)
        cpu = sum(t.cpu for t in self.buffer)
        return float(self.ctx.indicators.usage(mem, disk, cpu))

    def protocol_mask(self, max_hold: float) -> list[bool]:
        """Actions the protocol allows right now (rules 4 and 5, plus empty/crashed slots)."""
        mask = [False] * N_ACTIONS
        if self.crashed or not self.buffer:
            mask[HOLD] = True
            return mask
        mask[HOLD] = self.now - self.oldest_arrival() < max_hold
        mask[LOCAL] = True
        if max(t.h for t in self.buffer) < self.ctx.params.hop_limit:
            for s, j in enumerate(self.neighbours):
                mask[FIRST_SEND + s] = j >= 0 and not self.suspected[s]
        return mask

    # ------------------------------------------------------------------ arrivals and dispatch

    def accept_arrivals(self, tasks: Sequence[Task]) -> None:
        """New tasks from this server's Voronoi cell join the buffer.

        Arrivals are handed over at the next decision instant (nothing can act on a buffered
        task before then), but they keep their true arrival times and the time they waited
        before being handed over still counts towards the reward.
        """
        if not tasks:
            return
        now = self.now
        self._touch()
        move = self.ctx.ledger.move
        for t in tasks:
            move(t, S.BUFFERED)
            t.t_buffer = t.t_arrive
            self._wait_area += now - t.t_arrive
            self.buffer.append(t)

    def send_polls(self) -> None:
        """Ask every neighbour for its current load (EVBLB-style poll round)."""
        for j in self.neighbours:
            if j >= 0:
                self._send(Message(MsgKind.POLL, self.idx, j))

    def start_dispatch(self, action: int, poll_wait: float = 0.0) -> None:
        """Take the whole buffer into a dispatch round towards ``action``'s target.

        Rounds at one server run one at a time. The batch leaves when its round ends, after
        any poll wait and the round's own cost.
        """
        if action == HOLD or not self.buffer:
            raise ValueError(f"server {self.idx}: nothing to dispatch with action {action}")
        now = self.now
        costs = self.ctx.costs
        self._touch()
        batch, self.buffer = self.buffer, []
        cost = costs.dispatch_cost(len(batch))
        start = max(now + poll_wait, self._round_free_at)
        end = start + cost
        self._round_free_at = end
        self.tasks_in_rounds += len(batch)
        rnd = DispatchRound(batch, action, now, end)
        self.rounds.append(rnd)
        self.last_dispatch = now
        self.epoch_dispatched = True
        self.epoch_dispatch_cost += cost
        if action != LOCAL:
            self.epoch_sent_dist += self.neighbour_dist[action_slot(action)]
        self.ctx.recorder.on_round(self.idx, now, end, len(batch))
        self.ctx.clock.call_at(end, self._finish_round, rnd)

    def _finish_round(self, rnd: DispatchRound) -> None:
        if self.crashed:
            return
        self._touch()
        self.rounds.remove(rnd)
        self.tasks_in_rounds -= len(rnd.tasks)
        now = self.now
        for t in rnd.tasks:
            t.t_dispatch = now
        if rnd.action == LOCAL:
            for t in rnd.tasks:
                self._enqueue_local(t)
            return
        slot = action_slot(rnd.action)
        if self.suspected[slot]:
            # the target was found to be crashed while this round was running
            self._back_to_buffer(rnd.tasks, now)
            return
        self._make_offer(slot, rnd.tasks)

    def _enqueue_local(self, task: Task) -> None:
        if self.q_count < self.ctx.params.queue_capacity and self._fits(task):
            self._enqueue(task)
        else:
            self.ctx.ledger.move(task, S.BLOCKED)
            self.epoch_blocked += 1
            self.ctx.recorder.on_block(task)

    def _back_to_buffer(self, tasks: Sequence[Task], now: float) -> None:
        move = self.ctx.ledger.move
        for t in tasks:
            move(t, S.BUFFERED)
            t.t_buffer = now
            self.buffer.append(t)

    # ------------------------------------------------------------------ offers (rules 1-4)

    def _make_offer(self, slot: int, tasks: list[Task]) -> None:
        offer = Offer(self._next_offer, self.neighbours[slot], slot, tasks)
        self._next_offer += 1
        move = self.ctx.ledger.move
        for t in tasks:
            move(t, S.IN_FLIGHT)
            t.offer = offer.offer_id
        self.inflight[offer.offer_id] = offer
        self._send_offer(offer)

    def _send_offer(self, offer: Offer) -> None:
        """(Re)send the still-unanswered part of ``offer`` and arm its timeout.

        Only unanswered tasks are resent: a task already rejected may have been dispatched
        somewhere else since, and offering it again could run it twice.
        """
        tasks = list(offer.pending.values())
        costs = self.ctx.costs
        extra = costs.transfer_extra(len(tasks))
        self._send(
            Message(MsgKind.OFFER, self.idx, offer.target, offer_id=offer.offer_id, tasks=tasks),
            extra,
        )
        j = offer.target
        round_trip = costs.delay[self.idx][j] + extra + costs.delay[j][self.idx]
        offer.attempt += 1
        self.ctx.clock.call_at(
            self.now + round_trip + self.ctx.params.timeout_slack,
            self._on_timeout,
            offer.offer_id,
            offer.attempt,
        )

    def _on_timeout(self, offer_id: int, attempt: int) -> None:
        offer = self.inflight.get(offer_id)
        if self.crashed or offer is None or offer.attempt != attempt:
            return  # already answered, or a newer attempt owns the timer
        if offer.attempt > self.ctx.params.max_retries and self.ctx.detector.is_crashed(
            offer.target
        ):
            self._give_up(offer)
        else:
            self._send_offer(offer)

    def _give_up(self, offer: Offer) -> None:
        """The target crashed: take the unanswered tasks back into the buffer (rule 1)."""
        del self.inflight[offer.offer_id]
        self.suspected[offer.slot] = True
        self._touch()
        now = self.now
        led = self.ctx.ledger
        for t in offer.pending.values():
            if t.state in (S.IN_FLIGHT, S.LOST):
                led.move(t, S.BUFFERED)
                t.t_buffer = now
                self.buffer.append(t)
            elif t.state == S.DONE:
                led.rerun_after_crash(t)  # it finished there, but every ACK and the DONE got lost
            else:
                led.violation("P1", f"{t!r} taken back from crashed {offer.target}")

    def _on_offer(self, msg: Message) -> None:
        """Admission control at the receiving side (rules 2 and 3)."""
        accepted: list[int] = []
        rejected: list[int] = []
        self._touch()
        cap = self.ctx.params.queue_capacity
        for t in msg.tasks or ():
            if t.id in self.seen:
                accepted.append(t.id)  # resent OFFER: we already hold it, just ACK again
            elif t.state != S.IN_FLIGHT:
                # the sender no longer considers this task offered (e.g. it was rejected and
                # sent elsewhere); admitting it could run it twice
                self.ctx.ledger.violation("P1", f"{t!r} offered to {self.idx} but not in flight")
                rejected.append(t.id)
            elif self.q_count < cap and self._fits(t):
                self.seen.add(t.id)
                self._enqueue(t)
                accepted.append(t.id)
            else:
                rejected.append(t.id)
        prem, util = self.p_rem(), self.utilisation()
        est = self.q_cpu / (1000.0 * self.cpu_ghz)
        for kind, ids in ((MsgKind.ACK, accepted), (MsgKind.REJECT, rejected)):
            if ids:
                self._send(
                    Message(
                        kind,
                        self.idx,
                        msg.src,
                        offer_id=msg.offer_id,
                        ids=ids,
                        p_rem=prem,
                        util=util,
                        est_finish=est,
                    )
                )

    def _on_reply(self, msg: Message) -> None:
        offer = self.inflight.get(msg.offer_id)
        if offer is None:
            return
        hits = 0
        if msg.kind == MsgKind.ACK:
            for tid in msg.ids or ():
                hits += offer.pending.pop(tid, None) is not None
        else:
            self._touch()
            now = self.now
            led = self.ctx.ledger
            for tid in msg.ids or ():
                t = offer.pending.pop(tid, None)
                if t is None:
                    continue
                hits += 1
                led.move(t, S.BUFFERED)
                t.h += 1
                t.t_buffer = now
                self.buffer.append(t)
        if hits:
            self.ctx.recorder.on_reply(self.idx, offer.slot, msg.kind, hits, msg.est_finish)
        if not offer.pending:
            del self.inflight[offer.offer_id]

    def _on_done(self, msg: Message) -> None:
        """A DONE also settles the task if its ACK got lost."""
        offer = self.inflight.get(msg.offer_id)
        if offer is None:
            return
        for tid in msg.ids or ():
            offer.pending.pop(tid, None)
        if not offer.pending:
            del self.inflight[offer.offer_id]

    # ------------------------------------------------------------------ queue and service

    def _fits(self, task: Task) -> bool:
        return task.mem <= self.mem_mb and task.disk <= self.disk_mb

    def _enqueue(self, task: Task) -> None:
        """Admit ``task`` to the FIFO queue (caller has checked capacity and called _touch)."""
        self.ctx.ledger.move(task, S.QUEUED)
        task.t_enqueue = self.now
        task.executor = self.idx
        self.queue.append(task)
        self.q_count += 1
        self.q_mem += task.mem
        self.q_disk += task.disk
        self.q_cpu += task.cpu
        if self.running is None:
            self._start_next()

    def _start_next(self) -> None:
        task = self.queue.popleft()
        self.ctx.ledger.move(task, S.RUNNING)
        task.t_start = self.now
        self.running = task
        self.ctx.clock.call_at(
            self.now + task.cpu / (1000.0 * self.cpu_ghz), self._service_done, task
        )

    def _service_done(self, task: Task) -> None:
        if self.crashed or self.running is not task:
            return
        self._touch()
        self.ctx.ledger.move(task, S.DONE)
        task.t_finish = self.now
        self.running = None
        self.q_count -= 1
        self.q_mem -= task.mem
        self.q_disk -= task.disk
        self.q_cpu -= task.cpu
        if self.q_count == 0:  # avoid float drift building up in the running sums
            self.q_mem = self.q_disk = self.q_cpu = 0.0
        self.ctx.recorder.on_complete(task)
        if task.home != self.idx:
            self._send(
                Message(
                    MsgKind.DONE,
                    self.idx,
                    task.home,
                    offer_id=task.offer,
                    ids=[task.id],
                    p_rem=self.p_rem(),
                    util=self.utilisation(),
                )
            )
        if self.queue:
            self._start_next()

    # ------------------------------------------------------------------ messages

    def receive(self, msg: Message) -> None:
        """Handle one delivered message."""
        if self.crashed:
            return
        kind = msg.kind
        if kind in LOAD_REPORTS:
            self._update_cache(msg)
        if kind == MsgKind.OFFER:
            self._on_offer(msg)
        elif kind == MsgKind.ACK or kind == MsgKind.REJECT:
            self._on_reply(msg)
        elif kind == MsgKind.DONE:
            self._on_done(msg)
        elif kind == MsgKind.POLL:
            self._send(
                Message(
                    MsgKind.POLL_REPLY,
                    self.idx,
                    msg.src,
                    p_rem=self.p_rem(),
                    util=self.utilisation(),
                )
            )

    def _update_cache(self, msg: Message) -> None:
        slot = self.slot_of.get(msg.src)
        if slot is None:
            return
        self.cache_prem[slot] = msg.p_rem
        self.cache_util[slot] = msg.util
        self.cache_time[slot] = self.now  # age is measured on *our* clock, no sync needed

    def send_heartbeats(self) -> None:
        """Tell every server that lists us as a neighbour our load, if we've been quiet 2 s."""
        if self.crashed:
            return
        now = self.now
        period = self.ctx.params.heartbeat_period
        prem = util = None
        for k in self.reverse:
            if now - self._last_sent.get(k, -np.inf) >= period:
                if prem is None:
                    prem, util = self.p_rem(), self.utilisation()
                self._send(Message(MsgKind.HEARTBEAT, self.idx, k, p_rem=prem, util=util))

    def _send(self, msg: Message, extra_delay: float = 0.0) -> None:
        self._last_sent[msg.dst] = self.now
        self.ctx.transport.send(msg, extra_delay)

    # ------------------------------------------------------------------ crash and reward stats

    def crash(self) -> None:
        """Crash-stop: everything this server holds is lost and it never answers again."""
        if self.crashed:
            return
        self.crashed = True
        led = self.ctx.ledger
        held = list(self.buffer) + [t for r in self.rounds for t in r.tasks] + list(self.queue)
        if self.running is not None:
            held.append(self.running)
        for t in held:
            led.move(t, S.LOST)
        # tasks we offered but nobody admitted yet: the OFFERs die with us
        for offer in self.inflight.values():
            for t in offer.pending.values():
                if t.state == S.IN_FLIGHT:
                    led.move(t, S.LOST)
        self.buffer, self.queue, self.running = [], deque(), None
        self.rounds.clear()
        self.inflight.clear()
        self.tasks_in_rounds = self.q_count = 0

    def _touch(self) -> None:
        """Integrate the number of waiting tasks over time (for the reward's W term)."""
        now = self.now
        waiting = len(self.buffer) + self.tasks_in_rounds + len(self.queue)
        self._wait_area += waiting * (now - self._last_touch)
        self._last_touch = now

    def take_epoch_stats(self) -> tuple[float, bool, float, float, int]:
        """Return and reset (W, dispatched, c_d, sent distance, blocked) for the last epoch."""
        self._touch()
        stats = (
            self._wait_area,
            self.epoch_dispatched,
            self.epoch_dispatch_cost,
            self.epoch_sent_dist,
            self.epoch_blocked,
        )
        self._wait_area = 0.0
        self.epoch_dispatched = False
        self.epoch_dispatch_cost = 0.0
        self.epoch_sent_dist = 0.0
        self.epoch_blocked = 0
        return stats


class CloudServer:
    """The remote cloud: one fast server with an unbounded FIFO queue and a fixed latency.

    Only tasks that fit no edge server go here (EVBLB Algorithm 1, lines 1-8).
    """

    def __init__(self, cpu_ghz: float, latency: float, ctx: NodeContext) -> None:
        self.cpu_ghz = cpu_ghz
        self.latency = latency
        self.ctx = ctx
        self.queue: deque[Task] = deque()
        self.running: Task | None = None
        self.in_transit = 0
        self.completed = 0

    def submit(self, task: Task) -> None:
        """Send a brand-new task to the cloud."""
        self.ctx.ledger.move(task, S.IN_FLIGHT)
        task.t_dispatch = self.ctx.clock.now
        self.in_transit += 1
        self.ctx.clock.call_at(self.ctx.clock.now + self.latency, self._arrive, task)

    def _arrive(self, task: Task) -> None:
        self.in_transit -= 1
        self.ctx.ledger.move(task, S.QUEUED)
        task.t_enqueue = self.ctx.clock.now
        task.executor = -1
        self.queue.append(task)
        if self.running is None:
            self._start_next()

    def _start_next(self) -> None:
        task = self.queue.popleft()
        self.ctx.ledger.move(task, S.RUNNING)
        task.t_start = self.ctx.clock.now
        self.running = task
        done_at = self.ctx.clock.now + task.cpu / (1000.0 * self.cpu_ghz)
        self.ctx.clock.call_at(done_at, self._done, task)

    def _done(self, task: Task) -> None:
        self.ctx.ledger.move(task, S.DONE)
        task.t_finish = self.ctx.clock.now + self.latency  # the result still has to come back
        self.running = None
        self.completed += 1
        self.ctx.recorder.on_complete(task)
        if self.queue:
            self._start_next()


class OracleDetector:
    """Perfect failure detector: never suspects a live server.

    The protocol's exactly-once argument assumes a perfect detector. In the simulator we can
    provide one directly; a node still only asks it after ``max_retries`` unanswered resends.
    """

    def __init__(self, nodes: Sequence[ServerNode]) -> None:
        self.nodes = nodes

    def is_crashed(self, idx: int) -> bool:
        """True iff server ``idx`` has crashed."""
        return self.nodes[idx].crashed


class NodeView:
    """Read-only access to one node's *true* state, for baselines that may use it."""

    __slots__ = ("_node", "_nodes")

    def __init__(self, node: ServerNode, nodes: Sequence[ServerNode]) -> None:
        self._node = node
        self._nodes = nodes

    @property
    def idx(self) -> int:
        """Server index."""
        return self._node.idx

    @property
    def now(self) -> float:
        """Current simulated time."""
        return self._node.now

    @property
    def crashed(self) -> bool:
        """Whether this server has crashed."""
        return self._node.crashed

    @property
    def last_dispatch(self) -> float:
        """Time of this server's last dispatch decision."""
        return self._node.last_dispatch

    @property
    def neighbours(self) -> list[int]:
        """Neighbour server per slot, -1 for empty slots."""
        return list(self._node.neighbours)

    @property
    def neighbour_dist(self) -> list[float]:
        """Distance to the neighbour in each slot."""
        return list(self._node.neighbour_dist)

    @property
    def buffer_len(self) -> int:
        """Tasks waiting in the buffer."""
        return len(self._node.buffer)

    @property
    def queue_len(self) -> int:
        """Tasks in the queue including the running one."""
        return self._node.q_count

    def p_rem(self) -> float:
        """True remaining-resource indicator of this server."""
        return self._node.p_rem()

    def utilisation(self) -> float:
        """True utilisation of this server."""
        return self._node.utilisation()

    def true_neighbour_prem(self) -> list[float | None]:
        """True P_rem of each neighbour slot; None for empty slots and crashed servers."""
        out: list[float | None] = []
        for j in self._node.neighbours:
            if j < 0 or self._nodes[j].crashed:
                out.append(None)
            else:
                out.append(self._nodes[j].p_rem())
        return out

    def cached_neighbour_prem(self) -> list[float]:
        """P_rem of each neighbour slot as last reported to this server."""
        return [float(x) for x in self._node.cache_prem]
