"""Property-based tests: the protocol stays safe whatever the policy does.

Hypothesis draws random topologies, loads, message loss, an optional crash and the protocol
traits of a *random* policy (polling or not, heartbeats or not, any hold cap). The env runs in
strict mode, so any P1/P2/P4 violation raises during the run. Afterwards we check that, once
arrivals stop, every task that wasn't blocked or lost in the crash has finished.

The test is derandomised so the suite gives the same result on every machine; change the
``seed`` range or drop ``derandomize`` to explore further.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.config import load_config
from policies.random_policy import RandomPolicy
from sim.rollout import run_episode


@settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n=st.integers(5, 20),
    rho=st.floats(0.3, 1.1),
    p_loss=st.one_of(st.just(0.0), st.floats(0.0, 0.2)),
    crash_time=st.one_of(st.none(), st.floats(5.0, 110.0)),
    max_hold=st.sampled_from([0.25, 0.5, 1.0, 2.0]),
    needs_poll=st.booleans(),
    uses_cache=st.booleans(),
    capacity_mask=st.booleans(),
    zero_costs=st.booleans(),
    workload=st.sampled_from(["W1", "W2"]),
    seed=st.integers(0, 2**16),
)
def test_protocol_properties_hold_for_a_random_policy(
    n,
    rho,
    p_loss,
    crash_time,
    max_hold,
    needs_poll,
    uses_cache,
    capacity_mask,
    zero_costs,
    workload,
    seed,
):
    dotted = {
        "topology.n_servers": n,
        "workload.name": workload,
        "workload.rho": rho,
        "run.duration": 120.0,
        "run.arrival_cutoff": 10.0,
        "run.drain_limit": 5000.0,
        "costs.p_loss": p_loss,
        "costs.zero_costs": zero_costs,
    }
    if crash_time is not None:
        dotted["faults.crash_time"] = crash_time
        dotted["faults.crash_fraction"] = 1.0 / n  # exactly one server
    policy = RandomPolicy(max_hold=max_hold)
    policy.needs_poll = needs_poll
    policy.uses_cache = uses_cache
    policy.capacity_mask = capacity_mask

    _, summary = run_episode(load_config(dotted=dotted), policy, seed, strict=True)

    assert summary["violations"] == 0
    assert summary["unfinished"] == 0, "tasks left over after arrivals stopped"
    accounted = summary["completed"] + summary["blocked"] + summary["lost_in_crash"]
    assert accounted == summary["created"]
    if crash_time is None:
        assert summary["lost_in_crash"] == 0
    if crash_time is None or p_loss == 0.0:
        # a task can only run twice when a crash and lost replies come together
        assert summary["reruns_after_crash"] == 0
