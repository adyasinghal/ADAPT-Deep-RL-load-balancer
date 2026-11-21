"""The invariant checks must actually catch broken states (otherwise the other tests mean
little), and the reward bookkeeping must be exact."""

import numpy as np
import pytest

from core.actions import LOCAL, send_action
from core.invariants import InvariantChecker, InvariantViolation
from core.model import Task, TaskState
from sim.env import EdgeEnv
from tests.helpers import add_tasks, make_world
from tests.test_env import small_cfg

S = TaskState


def checker_for(world, max_hold=2.0):
    return InvariantChecker(world.ledger, world.params.queue_capacity, max_hold, 0.25)


def test_illegal_transition_raises():
    w = make_world()
    (task,) = add_tasks(w, 0, [100])
    with pytest.raises(InvariantViolation):
        w.ledger.move(task, S.DONE)  # BUFFERED -> DONE skips running


def test_done_tasks_can_never_run_again():
    w = make_world()
    (task,) = add_tasks(w, 0, [100])
    w.nodes[0].start_dispatch(LOCAL)
    w.engine.run_until(1.0)
    assert task.state == S.DONE
    with pytest.raises(InvariantViolation):
        w.ledger.move(task, S.QUEUED)


def test_checker_catches_a_task_missing_from_the_buffers():
    w = make_world()
    add_tasks(w, 0, [100, 100])
    w.nodes[0].buffer.pop()  # the task silently disappears
    with pytest.raises(InvariantViolation, match="P1"):
        checker_for(w).check(w.engine.now, w.nodes, w.cloud)


def test_checker_catches_an_old_buffered_task():
    w = make_world()
    add_tasks(w, 0, [100])
    w.engine.run_until(2.5)
    with pytest.raises(InvariantViolation, match="P2"):
        checker_for(w).check(w.engine.now, w.nodes, w.cloud)


def test_checker_catches_an_overfull_queue():
    w = make_world(servers__queue_capacity=2)
    add_tasks(w, 0, [500, 500])
    w.nodes[0].start_dispatch(LOCAL)
    w.engine.run_until(0.1)
    w.nodes[0].queue.append(Task(-1, 0, 0.0, 1.0, 1.0, 1.0))  # bypassing admission control
    w.nodes[0].q_count += 1
    with pytest.raises(InvariantViolation):
        checker_for(w).check(w.engine.now, w.nodes, w.cloud)


def test_clean_world_passes_all_checks():
    w = make_world()
    add_tasks(w, 0, [300] * 4)
    w.nodes[0].start_dispatch(send_action(1))
    add_tasks(w, 1, [300] * 2)
    w.nodes[1].start_dispatch(LOCAL)
    for t in np.arange(0.01, 3.0, 0.01):
        w.engine.run_until(t)
        checker_for(w).check(t, w.nodes, w.cloud)


def test_waiting_time_integral_is_exact():
    w = make_world()
    node = w.nodes[0]
    add_tasks(w, 0, [1000, 1000])  # both arrive at t = 0
    w.engine.run_until(0.25)
    wait, dispatched, cost, dist, blocked = node.take_epoch_stats()
    assert wait == pytest.approx(0.5) and not dispatched
    node.start_dispatch(LOCAL)  # round ends at 0.271; first task runs, second waits
    w.engine.run_until(0.5)
    wait, dispatched, cost, dist, blocked = node.take_epoch_stats()
    assert wait == pytest.approx(2 * 0.021 + (0.5 - 0.271))
    assert dispatched and cost == pytest.approx(0.021)


def test_oversized_tasks_go_to_the_cloud():
    # every edge server has 300 MB of memory, so tasks needing more must go to the cloud
    cfg = small_cfg(**{"servers.memory_gb": [0.3]})
    env = EdgeEnv(cfg, strict=True)
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, _, done, info = env.step(np.zeros(env.n, dtype=int) + LOCAL * info["masks"][:, LOCAL])
    summary = env.summary()
    assert env.world.cloud.completed > 0
    assert summary["completed"] + summary["blocked"] == summary["created"]
