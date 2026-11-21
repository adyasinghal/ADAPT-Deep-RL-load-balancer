"""Tests for the event engine and the simulated transport."""

import numpy as np
import pytest

from core.config import load_config
from core.messages import Message, MessageCounter, MsgKind
from core.model import CostModel, Topology
from core.transport import SimTransport
from sim.engine import Engine


def test_events_run_in_time_order_and_fifo_on_ties():
    eng = Engine()
    seen = []
    eng.call_at(2.0, seen.append, "c")
    eng.call_at(1.0, seen.append, "a")
    eng.call_at(1.0, seen.append, "b")
    eng.call_at(5.0, seen.append, "late")
    eng.run_until(3.0)
    assert seen == ["a", "b", "c"]
    assert eng.now == 3.0
    assert eng.pending == 1


def test_events_can_schedule_more_events():
    eng = Engine()
    seen = []

    def first():
        eng.call_later(0.5, seen.append, eng.now + 0.5)

    eng.call_at(1.0, first)
    eng.run_until(10.0)
    assert seen == [1.5]


def test_scheduling_in_the_past_fails():
    eng = Engine()
    eng.run_until(1.0)
    with pytest.raises(ValueError):
        eng.call_at(0.5, print)


class Sink:
    def __init__(self, eng):
        self.eng = eng
        self.crashed = False
        self.got = []

    def receive(self, msg):
        self.got.append((self.eng.now, msg.kind))


def make_transport(p_loss=0.0):
    cfg = load_config()
    topo = Topology.from_arrays([[0, 0], [30, 40]], [4, 4], [500, 500], [1.0, 1.0], 5, 60)
    eng = Engine()
    counter = MessageCounter()
    tr = SimTransport(eng, CostModel(cfg, topo), counter, np.random.default_rng(0), p_loss)
    tr.nodes = [Sink(eng), Sink(eng)]
    return eng, tr, counter


def test_message_delay_and_extra_delay():
    eng, tr, counter = make_transport()
    tr.send(Message(MsgKind.POLL, 0, 1))
    tr.send(Message(MsgKind.OFFER, 0, 1), extra_delay=0.016)
    eng.run_until(1.0)
    (t1, k1), (t2, k2) = tr.nodes[1].got
    assert t1 == pytest.approx(0.007) and k1 == MsgKind.POLL
    assert t2 == pytest.approx(0.023) and k2 == MsgKind.OFFER
    assert counter.total == 2


def test_lost_messages_still_count_and_crashed_nodes_get_nothing():
    eng, tr, counter = make_transport()
    tr.drop_filter = lambda m: m.kind == MsgKind.ACK
    tr.send(Message(MsgKind.ACK, 0, 1))
    tr.nodes[1].crashed = True
    tr.send(Message(MsgKind.POLL, 0, 1))
    eng.run_until(1.0)
    assert tr.nodes[1].got == []
    assert counter.counts[MsgKind.ACK] == 1 and tr.lost == 1


def test_random_loss_rate():
    eng, tr, _ = make_transport(p_loss=0.3)
    for _ in range(4000):
        tr.send(Message(MsgKind.HEARTBEAT, 0, 1))
    eng.run_until(1.0)
    assert 0.27 < tr.lost / 4000 < 0.33
