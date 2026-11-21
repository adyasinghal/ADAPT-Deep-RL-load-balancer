"""Protocol tests for ServerNode (rules 1-6) on hand-made three-server worlds."""

import pytest

from core.actions import HOLD, LOCAL, send_action
from core.messages import MsgKind
from core.model import Task, TaskState
from tests.helpers import add_tasks, make_world

S = TaskState


def test_fifo_service_times():
    w = make_world()
    tasks = add_tasks(w, 0, [500, 1000, 250])
    w.nodes[0].start_dispatch(LOCAL)
    w.engine.run_until(10.0)
    round_end = 0.020 + 3 * 0.0005
    finishes = [t.t_finish for t in tasks]
    assert finishes == pytest.approx([round_end + 0.5, round_end + 1.5, round_end + 1.75])
    assert all(t.state == S.DONE and t.executor == 0 for t in tasks)


def test_dispatch_rounds_are_serialised():
    w = make_world()
    node = w.nodes[0]
    add_tasks(w, 0, [100, 100])
    node.start_dispatch(LOCAL)
    add_tasks(w, 0, [100])
    node.start_dispatch(LOCAL)
    first, second = node.rounds
    assert first.departs_at == pytest.approx(0.021)
    assert second.departs_at == pytest.approx(0.021 + 0.0205)
    # the batch is still "buffered" while its round runs
    assert w.ledger.count(S.BUFFERED) == 3


def test_poll_wait_delays_the_round():
    w = make_world()
    add_tasks(w, 0, [100])
    w.nodes[0].start_dispatch(LOCAL, poll_wait=0.1)
    assert w.nodes[0].rounds[0].departs_at == pytest.approx(0.1 + 0.0205)


def test_offer_arrives_after_message_and_transfer_delay():
    w = make_world()
    tasks = add_tasks(w, 0, [500] * 4)
    w.nodes[0].start_dispatch(send_action(0))  # slot 0 is server 1, distance 10
    w.engine.run_until(0.2)
    expected = (0.020 + 4 * 0.0005) + (0.002 + 0.001) + 4 * 0.008
    assert tasks[0].t_enqueue == pytest.approx(expected)
    assert all(t.executor == 1 for t in tasks)
    assert w.nodes[0].inflight == {}  # the ACK came back


def test_lost_ack_resend_goes_to_same_server_without_duplicates():
    w = make_world()
    dropped = []

    def drop_first_ack(msg):
        if msg.kind == MsgKind.ACK and not dropped:
            dropped.append(msg)
            return True
        return False

    w.transport.drop_filter = drop_first_ack
    tasks = add_tasks(w, 0, [500, 500])
    w.nodes[0].start_dispatch(send_action(0))
    w.engine.run_until(0.3)
    counts = w.counter.counts
    assert counts[MsgKind.OFFER] == 2 and counts[MsgKind.ACK] == 2
    assert w.nodes[1].q_count == 2  # admitted once, not twice
    assert w.nodes[0].inflight == {}
    w.engine.run_until(5.0)
    assert all(t.state == S.DONE and t.executor == 1 for t in tasks)
    assert w.recorder.response_times and len(w.recorder.response_times) == 2


def test_done_settles_a_task_whose_ack_was_lost():
    w = make_world(protocol__timeout_slack=5.0)  # make sure DONE wins the race
    w.transport.drop_filter = lambda m: m.kind == MsgKind.ACK
    add_tasks(w, 0, [100])
    w.nodes[0].start_dispatch(send_action(0))
    w.engine.run_until(1.0)
    assert w.nodes[0].inflight == {}
    assert w.counter.counts[MsgKind.OFFER] == 1


def test_reject_increments_hops_and_hop_limit_leaves_only_local():
    w = make_world(servers__queue_capacity=1)
    add_tasks(w, 1, [1000])  # fills server 1 for about a second
    w.nodes[1].start_dispatch(LOCAL)
    w.engine.run_until(0.05)
    (task,) = add_tasks(w, 0, [100])
    node = w.nodes[0]
    for expected_h in (1, 2):
        node.start_dispatch(send_action(0))
        w.engine.run_until(w.engine.now + 0.1)
        assert task.state == S.BUFFERED and task.h == expected_h
    mask = node.protocol_mask(max_hold=2.0)
    assert mask[LOCAL] and not any(mask[send_action(s)] for s in range(5))
    node.start_dispatch(LOCAL)
    w.engine.run_until(1.0)
    assert task.state == S.DONE and task.executor == 0


def test_admission_never_exceeds_queue_capacity():
    w = make_world(servers__queue_capacity=3)
    tasks = add_tasks(w, 0, [500] * 5)
    w.nodes[0].start_dispatch(send_action(0))
    w.engine.run_until(0.2)
    assert w.nodes[1].q_count == 3
    back = [t for t in tasks if t.state == S.BUFFERED]
    assert len(back) == 2 and all(t.h == 1 for t in back)
    assert w.counter.counts[MsgKind.REJECT] == 1


def test_local_overflow_is_blocked():
    w = make_world(servers__queue_capacity=2)
    tasks = add_tasks(w, 0, [500] * 3)
    w.nodes[0].start_dispatch(LOCAL)
    w.engine.run_until(0.1)
    assert [t.state for t in tasks].count(S.BLOCKED) == 1
    assert w.recorder.blocked == 1


def test_crashed_target_tasks_taken_back_only_after_all_retries():
    w = make_world()
    w.nodes[1].crash()
    tasks = add_tasks(w, 0, [500, 500])
    node = w.nodes[0]
    node.start_dispatch(send_action(0))
    round_end = 0.021
    timeout = (0.003 + 2 * 0.008 + 0.003) + 0.05
    w.engine.run_until(round_end + 4 * timeout - 1e-6)
    assert all(t.state == S.IN_FLIGHT for t in tasks)
    w.engine.run_until(round_end + 4 * timeout + 1e-6)
    assert w.counter.counts[MsgKind.OFFER] == 1 + 3
    assert all(t.state == S.BUFFERED for t in tasks)
    assert node.suspected[0]
    mask = node.protocol_mask(max_hold=2.0)
    assert not mask[send_action(0)] and mask[send_action(1)]


def test_live_target_is_never_given_up_on():
    w = make_world()
    sent = []

    def drop_first_six_offers(msg):
        if msg.kind == MsgKind.OFFER:
            sent.append(msg)
            return len(sent) <= 6
        return False

    w.transport.drop_filter = drop_first_six_offers
    (task,) = add_tasks(w, 0, [200])
    w.nodes[0].start_dispatch(send_action(0))
    w.engine.run_until(5.0)
    assert len(sent) == 7
    assert task.state == S.DONE and task.executor == 1
    assert not w.nodes[0].suspected[0]


def test_tasks_queued_at_a_crashed_server_are_recovered_if_unacknowledged():
    w = make_world()
    w.transport.drop_filter = lambda m: m.kind == MsgKind.ACK
    tasks = add_tasks(w, 0, [1000, 1000])
    w.nodes[0].start_dispatch(send_action(0))
    w.engine.run_until(0.06)
    assert all(t.state in (S.QUEUED, S.RUNNING) for t in tasks)
    w.nodes[1].crash()
    assert all(t.state == S.LOST for t in tasks)
    w.engine.run_until(1.0)
    assert all(t.state == S.BUFFERED for t in tasks)  # taken back, will run elsewhere


def test_crash_loses_everything_the_server_holds():
    w = make_world()
    add_tasks(w, 0, [1000, 1000])
    w.nodes[0].start_dispatch(LOCAL)
    w.engine.run_until(0.1)
    add_tasks(w, 0, [100])
    w.nodes[0].crash()
    assert w.ledger.count(S.LOST) == 3
    assert w.nodes[0].protocol_mask(2.0) == [True] + [False] * 6


def test_hold_is_masked_once_the_oldest_task_is_old_enough():
    w = make_world()
    node = w.nodes[0]
    assert node.protocol_mask(2.0)[HOLD] and not node.protocol_mask(2.0)[LOCAL]
    add_tasks(w, 0, [100])
    w.engine.run_until(1.99)
    assert node.protocol_mask(2.0)[HOLD]
    w.engine.run_until(2.0)
    assert not node.protocol_mask(2.0)[HOLD]


def test_cloud_round_trip():
    w = make_world()
    task = Task(1, 0, 0.0, 100.0, 100.0, 1000.0)
    w.ledger.register(task)
    w.cloud.submit(task)
    w.engine.run_until(1.0)
    assert task.state == S.DONE
    assert task.t_finish == pytest.approx(0.1 + 1000 / 10_000 + 0.1)


def test_heartbeats_only_after_quiet_period():
    w = make_world()
    node = w.nodes[1]
    node.send_heartbeats()  # never sent anything -> one to each reverse neighbour
    assert w.counter.counts[MsgKind.HEARTBEAT] == 2
    w.engine.run_until(1.0)
    node.send_heartbeats()
    assert w.counter.counts[MsgKind.HEARTBEAT] == 2
    w.engine.run_until(2.0)
    node.send_heartbeats()
    assert w.counter.counts[MsgKind.HEARTBEAT] == 4
    # the receiver timestamps the report with its own clock, on arrival
    w.engine.run_until(2.1)
    assert w.nodes[0].cache_time[0] == pytest.approx(2.0 + 0.003)


def test_resend_only_carries_unanswered_tasks():
    # server 1 can take one task: of a two-task OFFER one is accepted and one rejected
    w = make_world(servers__queue_capacity=1)
    offers = []
    dropped = set()

    def watch(msg):
        if msg.kind == MsgKind.OFFER:
            offers.append([t.id for t in msg.tasks])
        if msg.kind in (MsgKind.ACK, MsgKind.DONE) and msg.kind not in dropped:
            dropped.add(msg.kind)  # lose the first ACK and the first DONE
            return True
        return False

    w.transport.drop_filter = watch
    first, second = add_tasks(w, 0, [10, 10])
    w.nodes[0].start_dispatch(send_action(0))
    w.engine.run_until(1.0)
    assert offers[0] == [first.id, second.id]
    assert len(offers) >= 2 and all(ids == [first.id] for ids in offers[1:])
    assert first.executor == 1 and second.state == S.BUFFERED and second.h == 1
