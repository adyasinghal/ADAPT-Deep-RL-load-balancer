# Design notes

One line per decision that is not spelled out in the coding brief, or where we deviate from it.
Items marked **(deviation)** change something the brief or the plan states; please flag them in
the report.

## Configuration and setup
- All constants are in `config/default.yaml`; experiment YAMLs only override keys. Unknown dotted keys raise.
- Developed and tested on Python 3.11 (`uv venv --python 3.11`).
- The topology is drawn from the run seed, so each seed is a different topology as well as different traffic.
- Random streams: one `SeedSequence(seed)` is split into topology / arrivals / workload / message loss / faults / policy, so adding a policy never shifts the traffic.

## Simulation model
- Arrivals of each 0.25 s interval are drawn in one go (Poisson count, uniform times) and handed to the buffers at the next decision instant; they keep their true arrival times and the time waited before hand-over counts in the reward's W term. Nothing can act on a buffered task between decision instants, so this changes no decision.
- Oversized tasks (fit no edge server) are submitted to the cloud at that same instant. With the default task ranges this never happens; the path is unit-tested.
- Utilisation u_i and P_rem count the running task's full T_cpu, not just what is left of it.
- Global LBF is taken over live servers only.
- Response time = finish - arrival. For cloud tasks the 100 ms return trip is included.
- Arrivals stop `arrival_cutoff` (60 s) before the end. After `duration` the run keeps going (policy still acting, no new arrivals) until every task is finished, capped at 600 s ("drain"). The time series stops at `duration`; response-time stats include tasks finished during the drain so that methods that leave long queues are not flattered; throughput counts completions up to `duration`; makespan is the last completion (NaN if the cap is hit).
- Only W1 and W2 are implemented in `sim/workloads.py` (W3-W6 belong to workstream 3). Crash injection and message loss are already supported through the `faults` and `costs.p_loss` config keys.
- `zero_costs` zeroes message delay, transfer, dispatch round and poll costs. It leaves `p_loss` and the cloud latency alone (they are not overhead costs).

## Protocol (`core/node.py`)
- **(deviation)** OFFER timeout = that OFFER's own expected round trip (message delay + 8 ms/task transfer + reply delay) + 50 ms, not 2 x max message delay + 50 ms. A batch transfer is often longer than two message delays, and a shorter timeout would resend while the first copy is still on the wire.
- Resends carry only the tasks still waiting for an answer: an already-rejected task may have been dispatched elsewhere, and offering it again could run it twice.
- **(deviation)** Failure detection: after `max_retries` (3) unanswered resends the sender asks the failure detector; the simulator provides a perfect (oracle) one. If the target is alive the sender just keeps resending. Declaring a server crashed after k timeouts, as the brief says, wrongly suspects live servers under message loss and then runs tasks twice, which breaks P1; the plan's proof assumes a perfect detector anyway.
- Taking tasks back from a crashed server also recovers tasks it had queued but never acknowledged (state LOST -> BUFFERED). If such a task had already finished there and its ACKs and DONE were all lost, it would run twice; the ledger reports that as a P1 violation. This needs a crash plus several lost replies in a row; with message loss AND crashes P1 is "exactly once except in that case" (worth one sentence in the proofs section).
- Messages still in transit to or from a server when it crashes are dropped (crash-stop includes its network interface). Its unadmitted in-flight tasks count as lost_in_crash.
- A DONE also settles an in-flight task whose ACK was lost.
- If an OFFER is partly accepted, the ACK and the REJECT are two separate messages (both counted).
- DONE is only sent when the executor is not the home server (a local completion needs no message).
- Rejected and taken-back tasks keep their arrival time, so the HOLD mask (rule 5) and feature 5 use the age since arrival; P2 is checked on the time a task last entered the buffer.
- If the target of a running dispatch round is found to have crashed before the round ends, the batch goes back into the buffer instead of being offered.
- Heartbeats are only sent when the policy uses the load cache (`uses_cache`: ADAPT, B4); polling baselines would never read them, so they don't pay for them. The first heartbeats go out at t = 0.

## Env and observations (`sim/env.py`)
- `reset(seed)` returns `(obs, info)` and masks are in `info["masks"]` (Gym style), since the brief's `reset -> obs` has nowhere to put the masks.
- Policies may set extra traits read by the env: `uses_cache`, `fresh_obs` (ADAPT-poll), `capacity_mask` (ADAPT). `info["masks"]` is what that policy may do; the env only enforces the protocol masks (`info["protocol_masks"]`) and raises on a forbidden action.
- The SEND_j "cached P_rem < batch usage" mask is applied only for policies with `capacity_mask`; it is an ADAPT heuristic, not a protocol rule.
- A neighbour never heard from has cached P_rem = its static P_e, u = 0 and age feature 1. Empty slots and slots of servers found to have crashed are all zeros with exists = 0.
- Feature 8 (task-size variance) uses only epochs that had arrivals in that cell; 0 when fewer than two such epochs are in the last 10.
- `info["replies"]` lists the ACK/REJECT outcomes of the last interval as (server, slot, kind, n_tasks, est_finish) for B4's bandit.

## Baselines
- B1 (nearest) dispatches LOCAL at every decision instant with a non-empty buffer and so pays a dispatch round each time, like every other method.
- EVBLB dispatches on the global dt grid (t = dt, 2dt, ...), matching the paper's episodic intervals; it only goes earlier when rule 5 forces it (returned tasks). Its candidate set includes the server itself (Algorithm 2 starts from N = {e_i}); ties go to the server itself.
- EVBLB picks its target from true P_rem at the decision instant, then pays the poll round (2 messages per neighbour, wait for the slowest reply) before the dispatch round. The values it uses can be up to one poll wait (~10 ms) older than the replies; under message loss it still "sees" every neighbour.
- Note for the report: with alpha = beta = gamma = 1/3, P_rem is dominated by the static memory/disk terms, so EVBLB equalises P_rem rather than CPU utilisation. Its CPU-based LBF therefore stays high (about 0.29 at 300 s and 0.42 at 1800 s for n = 50, rho = 0.7, against 0.42-0.45 for nearest-only).
