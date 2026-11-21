"""How messages get from one node to another.

``SimTransport`` delivers on the simulation clock; ``AsyncTransport`` delivers between asyncio
tasks for the demo. Both have the same ``send`` method, so ``ServerNode`` doesn't change.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

from core.messages import Message, MessageCounter
from core.model import CostModel

if TYPE_CHECKING:
    from core.node import ServerNode
    from sim.engine import Engine


class Transport(Protocol):
    """Anything that can send a message on behalf of a node."""

    def send(self, msg: Message, extra_delay: float = 0.0) -> None:
        """Send ``msg``; ``extra_delay`` is added on top of the pair's message delay."""


class SimTransport:
    """Point-to-point delivery with per-pair delays and independent random loss.

    A message to or from a crashed server is dropped when it would be delivered (crash-stop:
    the crashed server's network interface is gone too). Every message counts as sent, whether
    or not it is later lost.
    """

    def __init__(
        self,
        engine: Engine,
        costs: CostModel,
        counter: MessageCounter,
        rng: np.random.Generator,
        p_loss: float = 0.0,
    ) -> None:
        self.engine = engine
        self.costs = costs
        self.counter = counter
        self.rng = rng
        self.p_loss = float(p_loss)
        self.nodes: Sequence[ServerNode] = []
        # tests can install a filter that decides exactly which messages are lost
        self.drop_filter: Callable[[Message], bool] | None = None
        self.lost = 0

    def send(self, msg: Message, extra_delay: float = 0.0) -> None:
        """Count the message, maybe lose it, otherwise schedule its delivery."""
        self.counter.add(msg.kind)
        if self.drop_filter is not None and self.drop_filter(msg):
            self.lost += 1
            return
        if self.p_loss > 0.0 and self.rng.random() < self.p_loss:
            self.lost += 1
            return
        delay = self.costs.delay[msg.src][msg.dst] + extra_delay
        self.engine.call_at(self.engine.now + delay, self._deliver, msg)

    def _deliver(self, msg: Message) -> None:
        src, dst = self.nodes[msg.src], self.nodes[msg.dst]
        if src.crashed or dst.crashed:
            return
        dst.receive(msg)


class AsyncTransport:
    """Message passing between asyncio tasks, for the demo.

    Every node has an ``asyncio.Queue`` inbox. ``send`` puts the message into the destination's
    inbox after the same per-pair delay as in the simulator (measured on ``clock``, which may
    run faster than real time); each node's own task reads its inbox and calls
    ``node.receive``. Loss and crash handling are the same as ``SimTransport``.
    """

    def __init__(
        self,
        clock,
        costs: CostModel,
        counter: MessageCounter,
        rng: np.random.Generator,
        p_loss: float = 0.0,
    ) -> None:
        self.clock = clock
        self.costs = costs
        self.counter = counter
        self.rng = rng
        self.p_loss = float(p_loss)
        self.nodes: Sequence[ServerNode] = []
        self.inboxes: list[asyncio.Queue[Message]] = []
        self.lost = 0

    def attach(self, nodes: Sequence[ServerNode]) -> None:
        """Create one inbox per node."""
        self.nodes = nodes
        self.inboxes = [asyncio.Queue() for _ in nodes]

    def send(self, msg: Message, extra_delay: float = 0.0) -> None:
        """Count the message, maybe lose it, otherwise deliver it into the inbox later."""
        self.counter.add(msg.kind)
        if self.p_loss > 0.0 and self.rng.random() < self.p_loss:
            self.lost += 1
            return
        delay = self.costs.delay[msg.src][msg.dst] + extra_delay
        self.clock.call_at(self.clock.now + delay, self.inboxes[msg.dst].put_nowait, msg)

    async def serve(self, idx: int) -> None:
        """The receiving loop of node ``idx``: one asyncio task per node."""
        node, inbox = self.nodes[idx], self.inboxes[idx]
        while True:
            msg = await inbox.get()
            if node.crashed or self.nodes[msg.src].crashed:
                continue
            node.receive(msg)
