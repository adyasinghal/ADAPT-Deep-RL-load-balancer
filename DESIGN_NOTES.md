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
- Taking tasks back from a crashed server also recovers tasks it had queued but never acknowledged (state LOST -> BUFFERED).
- **(caveat for the P1 proof)** If a task finished at a server that then crashed, and every ACK for it plus its DONE were lost, the sender cannot tell it from a task that never ran and takes it back: a real system would run it twice. This needs a crash *and* lost replies, so P1 is exactly-once when there is either no message loss or no crash, and at-least-once for that one case otherwise. The simulator counts these as `reruns_after_crash` (summary column) instead of calling them violations; the property test asserts the count is 0 unless loss and a crash happen together. It was 0 in all 200 property examples and in 30 extra runs with 20% loss plus a crash.
- The receiver only admits tasks that are actually in flight; anything else is reported as a P1 violation and rejected (defence against a sender re-offering a task it has already moved).
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

## Experiments (`experiments/run.py`)
- Method labels default to the policy's `name` (`nearest`, `evblb_dt2`, `perfect`, ...); an entry can set `label` explicitly.
- Only the `summary.csv` of each experiment is committed; the per-run time series (about 14 MB for E1 + E2) are regenerated by the runner in about 90 s on 8 cores.
- E2 finding: with the brief's cost constants, EVBLB's response time *increases* monotonically with dt at both rho = 0.5 and 0.9 (no U-shape); dt = 0.25 s is best on latency and worst on messages per task (7.4 vs 1.8 at dt = 4). The 20 ms dispatch round is small next to 0.25 s, so short intervals are cheap in latency. A U-shape would need larger per-round costs.

## Tests
- `tests/test_properties.py` runs 200 derandomised hypothesis examples (n 5-20, rho 0.3-1.1, loss up to 0.2, optional crash, random protocol traits, W1/W2) in strict mode; about 20 s.
- Mutation check done by hand on the protocol: removing receiver de-duplication, resending already-rejected tasks, and giving up on a live server after k timeouts each make the suite fail.
- B4 (Q-learning update) and B7 (TOPSIS score) tests belong with those baselines (workstream 3).

## ADAPT agent (`rl/`, `policies/adapt.py`)
- The reward is exactly the brief's (W counts tasks waiting in this server's own buffer and queue). We checked for "dumping" (a server is not charged for tasks it sends away): the trained agent sends LOCAL about 60% of the time and spreads the rest over its neighbours, so it does not dump; the local-LBF term and the capacity mask seem to be enough.
- Reward weights are the brief's starting values (1, 0.5, 0.2, 0.1, 5); the one allowed tuning pass has not been done yet. The blocking weight is the obvious candidate (see the results below).
- Episodes end on a time limit, never in a true terminal state, so every transition bootstraps (no `done` flag in the replay buffer).
- Training episodes use `arrival_cutoff: 0` and no drain; evaluation runs use the normal 1800 s scenario with drain.
- Not in the brief: gradient-norm clipping at 10, 5000 transitions in the buffer before the first update, torch limited to one thread (faster for this tiny MLP).
- epsilon decays per decision instant (= gradient step): 1.0 -> 0.05 over 50k steps without the warm start, 0.2 -> 0.02 with it. 20 episodes are 48k steps, so epsilon ends near its final value.
- After every training episode a greedy, no-learning run on one fixed *training* scenario (W1, rho 0.7, seed 0) is logged as `eval_reward` / `eval_resp`. Raw episode rewards mostly reflect the topology and load drawn for that episode; this column is the learning curve to plot (F7).
- Inference runs the exported `.npz` weights in numpy (`rl/numpy_net.py`); a test checks it matches the torch forward pass. Experiment workers never import torch.
- ADAPT-fixedDT allows a dispatch only once 2 s have passed since that server's last dispatch (per-server timer, rule 5 still wins). ADAPT-poll sends no heartbeats (it would never read them).
- Weights are small (~90 kB each) and committed under `weights/`; `python -m rl.train --variant <v>` reproduces them bit for bit.

## Warm start (`rl/imitation.py`)
- **(deviation)** The behaviour-cloning loss is cross-entropy on the masked Q-values *plus* the normal TD loss on the same batch. Cross-entropy alone fixes only the ranking of the Q-values and leaves their scale arbitrary, so the first DQN updates would undo it.
- The expert is EVBLB in ADAPT's action space: per-server 2 s timer, then the allowed candidate with the largest *true* P_rem (read through `ctx`, no poll cost). It obeys ADAPT's masks, including the capacity mask, so every expert action is one the network may take.
- 3 expert episodes give 360k transitions; the 200k replay buffer keeps the most recent 200k, which are also what fine-tuning starts from.
- After cloning, the greedy network agrees with the expert on 89% of the states where there was a real choice (57% for fixedDT, whose masks hide the timing part).

## M4 results (`results/adapt_eval/summary.csv`, 1800 s, n = 50, test seeds 100-104)
- Paired per-seed differences (same topology), 95% CI. Trained ADAPT vs EVBLB (B2), mean response time: -4.1 +- 2.4 s (W1, rho 0.7), -8.3 +- 5.8 (W1, 0.9), -4.4 +- 2.4 (W2, 0.7), -9.4 +- 5.8 (W2, 0.9). It also beats B6 (perfect information) on the mean, but those intervals include 0.
- Messages per task: ADAPT 1.29-1.33 vs EVBLB 1.96-2.23 (intervals do not overlap).
- Blocking is worse: ADAPT blocks +0.2-0.3 percentage points more than EVBLB at rho 0.7 and +2.2-2.7 points more at rho 0.9 (paired CIs exclude 0). ADAPT trades a few blocked tasks for much lower latency; raising `w_blocked` is the natural thing to tune.
- Episode 0 (straight after cloning, `adapt_ep0`): response time equal to EVBLB within noise (paired differences -2.3 to +0.0 s, all CIs include 0) with fewer messages (1.25-1.56 per task), but it blocks more (+0.8 to +5.3 points). So the "as good as B2 at episode 0" check holds for latency and messages, not for blocking.
- Warm start vs no warm start: by episode 20 they are about equal (warm start -0.7 to -1.1 s better at rho 0.9, CIs include 0); the warm start mainly gives a better curve early on.
- ADAPT-fixedDT is no better than the cloned network, so most of the gain comes from learning *when* to dispatch, which fits E2 (short intervals are cheap in latency). ADAPT-poll matches ADAPT on latency but needs about 5 messages per task.
