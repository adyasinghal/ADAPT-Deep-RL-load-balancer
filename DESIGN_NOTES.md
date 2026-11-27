# Design notes

These notes record the modelling and implementation choices we made where the papers leave
details open, and the places where we changed our first design, with the reason for each. They
end with the results in more detail than the README. Items marked **(deviation)** are changes to
our first design or to a paper's method.

Labels (B1-B7 methods, W1-W6 workloads, E1-E9 experiments, F1-F9 figures), the protocol rules
1-6, the guarantees P1-P5 and the 30 observation features are defined in the README; [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358)-[[5]](https://doi.org/10.1109/ICASSP48485.2024.10445952)
are the reference papers listed there.

## Notation
- P_e: a server's normalised capacity, a weighted sum of its memory, disk and CPU (EVBLB [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358),
  Eqs. 3-5); alpha, beta, gamma are those three weights (1/3 each).
- P_rem: the capacity a server has left, P_e minus what its queued tasks use ("room left").
- R_i: CPU speed of server i (GHz). T_cpu: a task's CPU demand (Mcycles).
- u_i: utilisation of server i, its queued work divided by what it can process in 2 s, capped at 1.
- LBF: standard deviation of u_i over the live servers; local LBF: the same over a server and
  its neighbours.
- dt: the fixed dispatch interval of EVBLB-style methods. psi: the neighbour radius (40 units);
  each server has at most nu = 5 neighbours.
- W (reward term): task-seconds spent waiting in the server's own buffer and queue during the
  interval.
- AHP: the analytic hierarchy process, which TOLB [[4]](https://doi.org/10.3390/electronics13081511) uses to weight its TOPSIS criteria.
- Epoch / decision instant: the 0.25 s interval between two decisions of a server.

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
- W3 (flash crowd): the first burst starts at t = 300 s, then one every 600 s, so a run doesn't open with a burst.
- W4 (moving hotspot): the hotspot centre is drawn uniformly but kept at least 20 units from the edge, so the whole disk is inside the area.
- W5 (diurnal) is defined relative to the configured rho: rho(t) = rho + 0.3 sin(2 pi t / 1800), which gives 0.4 -> 1.0 -> 0.4 at rho = 0.7.
- W6 is W1's traffic; its crashes and message loss come from the `faults` / `costs.p_loss` keys that `config/experiments/e8.yaml` sets.
- `zero_costs` zeroes message delay, transfer, dispatch round and poll costs. It leaves `p_loss` and the cloud latency alone (they are not overhead costs).

## Protocol (`core/node.py`)
- **(deviation)** OFFER timeout = that OFFER's own expected round trip (message delay + 8 ms/task transfer + reply delay) + 50 ms, not 2 x max message delay + 50 ms. A batch transfer is often longer than two message delays, and a shorter timeout would resend while the first copy is still on the wire.
- Resends carry only the tasks still waiting for an answer: an already-rejected task may have been dispatched elsewhere, and offering it again could run it twice.
- **(deviation)** Failure detection: after `max_retries` (3) unanswered resends the sender asks the failure detector; the simulator provides a perfect (oracle) one. If the target is alive the sender just keeps resending. Declaring a server crashed after k timeouts, as our first design did, wrongly suspects live servers under message loss and then runs tasks twice, which breaks P1; the P1 proof assumes a perfect detector anyway.
- Taking tasks back from a crashed server also recovers tasks it had queued but never acknowledged (state LOST -> BUFFERED).
- **(caveat for the P1 proof)** If a task finished at a server that then crashed, and every ACK for it plus its DONE were lost, the sender cannot tell it from a task that never ran and takes it back: a real system would run it twice. This needs a crash *and* lost replies, so P1 is exactly-once when there is either no message loss or no crash, and at-least-once for that one case otherwise. The simulator counts these as `reruns_after_crash` (summary column) instead of calling them violations; the property test asserts the count is 0 unless loss and a crash happen together. It was 0 in all 200 property examples and in 30 extra runs with 20% loss plus a crash.
- The receiver only admits tasks that are actually in flight; anything else is reported as a P1 violation and rejected (defence against a sender re-offering a task it has already moved).
- Messages still in transit to or from a server when it crashes are dropped (crash-stop includes its network interface). Its unadmitted in-flight tasks count as lost_in_crash.
- A DONE also settles an in-flight task whose ACK was lost.
- If an OFFER is partly accepted, the ACK and the REJECT are two separate messages (both counted).
- DONE is only sent when the executor is not the home server (a local completion needs no message).
- Rejected and taken-back tasks keep their arrival time, so the HOLD mask (rule 5) and feature 5 use the age since arrival; P2 is checked on the time a task last entered the buffer.
- If the target of a running dispatch round is found to have crashed before the round ends, the batch goes back into the buffer instead of being offered.
- Heartbeats are only sent when the policy uses the load cache (`uses_cache`: ADAPT only); polling baselines and B4 never read it, so they don't pay for them. The first heartbeats go out at t = 0.

## Env and observations (`sim/env.py`)
- `reset(seed)` returns `(obs, info)` and masks are in `info["masks"]` (Gym style), since a plain `reset -> obs` has nowhere to put the masks.
- Policies may set extra traits read by the env: `uses_cache`, `fresh_obs` (ADAPT-poll), `capacity_mask` (ADAPT). `info["masks"]` is what that policy may do; the env only enforces the protocol masks (`info["protocol_masks"]`) and raises on a forbidden action.
- The SEND_j "cached P_rem < batch usage" mask is applied only for policies with `capacity_mask`; it is an ADAPT heuristic, not a protocol rule.
- A neighbour never heard from has cached P_rem = its static P_e, u = 0 and age feature 1. Empty slots and slots of servers found to have crashed are all zeros with exists = 0.
- Feature 8 (task-size variance) uses only epochs that had arrivals in that cell; 0 when fewer than two such epochs are in the last 10.
- `info["replies"]` lists the ACK/REJECT outcomes of the last interval as (server, slot, kind, n_tasks, est_finish) for B4's bandit.

## Baselines
- B1 (nearest) dispatches LOCAL at every decision instant with a non-empty buffer and so pays a dispatch round each time, like every other method.
- EVBLB dispatches on the global dt grid (t = dt, 2dt, ...), matching EVBLB's [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358) episodic intervals; it only goes earlier when rule 5 forces it (returned tasks). Its candidate set includes the server itself (Algorithm 2 of [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358) starts from N = {e_i}); ties go to the server itself.
- EVBLB picks its target from true P_rem at the decision instant, then pays the poll round (2 messages per neighbour, wait for the slowest reply) before the dispatch round. The values it uses can be up to one poll wait (~10 ms) older than the replies; under message loss it still "sees" every neighbour.
- With alpha = beta = gamma = 1/3, P_rem is dominated by the static memory/disk terms, so EVBLB equalises P_rem rather than CPU utilisation. Its CPU-based LBF therefore stays high (about 0.29 at 300 s and 0.42 at 1800 s for n = 50, rho = 0.7, against 0.42-0.45 for nearest-only).
- u_i is capped at 1 once a server holds 2 s of work, so at these loads many servers sit at 1 and the LBF saturates around 0.4-0.45 for every method (F1, F6). "Time to balance" (LBF below 0.05 for 60 s) is never reached by any method in E5; it is reported as "-".
- B3 in E3/E9 is EVBLB with dt = 0.25 s, the interval with the lowest response time at both loads in E2. B6 uses the same interval with free information, so B3 and B6 come out almost identical.
- B4 (ML-enhanced EVBLB), interval learner: per server, state = network-wide LBF rounded to 0.01 (as in [[2]](https://doi.org/10.1016/j.comcom.2024.04.009); it is the one global input any method gets, and it comes from `info["lbf"]`), initial interval random (Alg. 1 of [[2]](https://doi.org/10.1016/j.comcom.2024.04.009), line 2), Q-update each time the chosen interval runs out.
- B4, neighbour bandit: we read Eq. 16 of [[2]](https://doi.org/10.1016/j.comcom.2024.04.009) literally. Only the chosen arm's preference moves, by step * (R - its mean reward) * its probability; Alg. 3 of [[2]](https://doi.org/10.1016/j.comcom.2024.04.009), line 14, sets the other arms' reward to their own mean, so their term is zero. The mean is updated after the preference. The arm is the argmax of the softmax (line 9), ties at random, so B4 explores little once one arm has a positive mean.
- B4 rewards: every ACK or REJECT message is one bandit reward for its arm (a partly rejected batch gives one of each); the LOCAL arm is rewarded at once from the server's own queue (`NodeView.est_finish`). B4 never polls and sends no heartbeats.
- B7 (TOPSIS) uses fixed weights 0.7 (P_rem) / 0.3 (distance) instead of the AHP weighting used in [[4]](https://doi.org/10.3390/electronics13081511). A criterion on which all candidates are equal gets normalised value 1 for all of them; ties go to the server itself.

## Experiments (`experiments/run.py`)
- Method labels default to the policy's `name` (`nearest`, `evblb_dt2`, `perfect`, ...); an entry can set `label` explicitly.
- Only the `summary.csv` of each experiment is committed, with the tables and figures made from them. The per-run time series (about 18 MB for the ones F1, F6 and F8 use) are not; `make_plots` skips those three figures until E1, E5 and E3 have been rerun.
- `--methods/--workloads/--rhos/--n` on the runner run a single cell of a grid (used for the fresh-clone check in the README).
- E2 finding: with our cost constants, EVBLB's response time *increases* monotonically with dt at both rho = 0.5 and 0.9 (no U-shape); dt = 0.25 s is best on latency and worst on messages per task (7.4 vs 1.8 at dt = 4). The 20 ms dispatch round is small next to 0.25 s, so short intervals are cheap in latency. A U-shape would need larger per-round costs.

## Tests
- `tests/test_properties.py` runs 200 derandomised hypothesis examples (n 5-20, rho 0.3-1.1, loss up to 0.2, optional crash, random protocol traits, W1/W2) in strict mode; about 20 s.
- Mutation check done by hand on the protocol: removing receiver de-duplication, resending already-rejected tasks, and giving up on a live server after k timeouts each make the suite fail.
- B4's Q-update is tested against Eq. 8 of [[2]](https://doi.org/10.1016/j.comcom.2024.04.009) on a toy transition and B7's score against a hand-worked three-candidate example (`tests/test_ml_evblb.py`, `tests/test_topsis.py`).
- ruff also enforces a docstring on every module, class and public function outside `tests/`.

## ADAPT agent (`rl/`, `policies/adapt.py`)
- The reward is the one given in the README (W counts tasks waiting in this server's own buffer and queue). We checked for "dumping" (a server is not charged for tasks it sends away): the trained agent sends LOCAL about 60% of the time and spreads the rest over its neighbours, so it does not dump; the local-LBF term and the capacity mask seem to be enough.
- Reward weights are our starting values (1, 0.5, 0.2, 0.1, 5) and were not tuned. The blocking weight is the obvious one to tune (see the results below).
- Episodes end on a time limit, never in a true terminal state, so every transition bootstraps (no `done` flag in the replay buffer).
- Training episodes use `arrival_cutoff: 0` and no drain; evaluation runs use the normal 1800 s scenario with drain.
- Not in our first design: gradient-norm clipping at 10, 5000 transitions in the buffer before the first update, torch limited to one thread (faster for this tiny MLP).
- epsilon decays per decision instant (= gradient step): 1.0 -> 0.05 over 50k steps without the warm start, 0.2 -> 0.02 with it. 20 episodes are 48k steps, so epsilon ends near its final value.
- After every training episode a greedy, no-learning run on one fixed *training* scenario (W1, rho 0.7, seed 0) is logged as `eval_reward` / `eval_resp`. Raw episode rewards mostly reflect the topology and load drawn for that episode; this column is the learning curve to plot (F7).
- Inference runs the exported `.npz` weights in numpy (`rl/numpy_net.py`); a test checks it matches the torch forward pass. Experiment workers never import torch.
- ADAPT-fixedDT allows a dispatch only once 2 s have passed since that server's last dispatch (per-server timer, rule 5 still wins). ADAPT-poll sends no heartbeats (it would never read them).
- The `.npz` weights (~90 kB each) are committed under `weights/`; that is all evaluation, the experiments and the demo load. The `.pt` torch checkpoints (only needed to resume training) and the warm-start snapshots of fixedDT/poll are git-ignored; `python -m rl.train --variant <v>` rewrites all of them bit for bit.

## Warm start (`rl/imitation.py`)
- **(deviation)** The behaviour-cloning loss is cross-entropy on the masked Q-values *plus* the normal TD loss on the same batch. Cross-entropy alone fixes only the ranking of the Q-values and leaves their scale arbitrary, so the first DQN updates would undo it.
- The expert is EVBLB in ADAPT's action space: per-server 2 s timer, then the allowed candidate with the largest *true* P_rem (read through `ctx`, no poll cost). It obeys ADAPT's masks, including the capacity mask, so every expert action is one the network may take.
- 3 expert episodes give 360k transitions; the 200k replay buffer keeps the most recent 200k, which are also what fine-tuning starts from.
- After cloning, the greedy network agrees with the expert on 89% of the states where there was a real choice (57% for fixedDT, whose masks hide the timing part).

## ADAPT evaluation (`results/adapt_eval/summary.csv`, 1800 s, n = 50, test seeds 100-104)
- Paired per-seed differences (same topology), 95% CI. Trained ADAPT vs EVBLB (B2), mean response time: -4.1 +- 2.4 s (W1, rho 0.7), -8.3 +- 5.8 (W1, 0.9), -4.4 +- 2.4 (W2, 0.7), -9.4 +- 5.8 (W2, 0.9). It also beats B6 (perfect information) on the mean, but those intervals include 0.
- Messages per task: ADAPT 1.29-1.33 vs EVBLB 1.96-2.23 (intervals do not overlap).
- Blocking is worse: ADAPT blocks +0.2-0.3 percentage points more than EVBLB at rho 0.7 and +2.2-2.7 points more at rho 0.9 (paired CIs exclude 0). ADAPT trades a few blocked tasks for much lower latency; raising `w_blocked` is the natural thing to tune.
- Episode 0 (straight after cloning, `adapt_ep0`): response time equal to EVBLB within noise (paired differences -2.3 to +0.0 s, all CIs include 0) with fewer messages (1.25-1.56 per task), but it blocks more (+0.8 to +5.3 points). So straight after cloning it is as good as EVBLB on latency and messages, but not on blocking.
- Warm start vs no warm start: by episode 20 they are about equal (warm start -0.7 to -1.1 s better at rho 0.9, CIs include 0); the warm start mainly gives a better curve early on.
- ADAPT-fixedDT is no better than the cloned network, so most of the gain comes from learning *when* to dispatch, which fits E2 (short intervals are cheap in latency). ADAPT-poll matches ADAPT on latency but needs about 5 messages per task.

## Tables and figures (`experiments/tables.py`, `plots/make_plots.py`)
- Every table cell is mean +- the half-width of the 95% t-interval over runs (5 seeds, or all 40 E3 runs of a method in E4). `e3_paired` takes ADAPT minus each baseline seed by seed; requiring the CIs not to overlap would be too strict here because topologies differ a lot between seeds.
- **(deviation)** F5 (messages per task) uses a linear axis, not a log axis: the values span less than a factor of 10, and bars on a log axis exaggerate small differences. Nearest-only sends no messages and is left out.
- F3/F4 leave out nearest-only (about 40-70 s response time, 12-37% blocking), which would squash every other bar; its numbers are in the tables.
- F8 cuts each run into 10 s blocks (so EVBLB's 2 s grid averages out), drops the first and last minute, and bins blocks by mean utilisation; a bin needs 20 blocks. It uses W2 and W3, the workloads whose load changes during a run.
- Colours: one fixed colour per method in every figure, from a palette order checked for colour-blind separation; ADAPT's variants share ADAPT's orange and differ by line style. Three of the colours are low-contrast on white, which is why every number is also in a table.

## Experiment results (`results/`, test seeds 100-104)
- E3, paired by seed against EVBLB: ADAPT's mean response time is lower in all 8 scenarios, by 4-9 s on W1/W2 and 8-28 s on W3/W4 (workloads it never trained on), with 1.3 instead of 2.1 messages per task and no polls. It blocks more: +0.2-0.3 percentage points on W1/W2 at rho 0.7, up to +5.8 on W4. It blocks less than B4 everywhere.
- F8: ADAPT dispatches about every 0.7 s and holds a little longer (about 1 s) as utilisation approaches 1, i.e. it batches more when busy; B4's learned interval goes the other way (2.3 s -> 1.5 s).
- E9: with every added cost at zero ADAPT still beats EVBLB on W1, rho 0.7 (4.3 s vs 8.3 s), so its gain is not just the overhead we added. The costs are small enough that every method barely changes.
- E7: the network trained on n = 50 also beats EVBLB at n = 10, 25 and 100.
- E8: 10% of servers crashing at 1800 s plus 2% message loss: no invariant violations and no re-runs for either method; ADAPT keeps its latency lead.
- E5/E6 repeat the ADAPT evaluation: the warm start and no-warm-start networks end up equal; fixing the interval (fixedDT) loses most of the gain; polling (poll) keeps it but needs 5 messages per task.

## Demo (`demo/run_demo.py`)
- Eight servers in a 90 x 90 area (instead of 200 x 200) so that each has neighbours; the network trained on n = 50 in 200 x 200 is used as is, since every feature is local and normalised.
- Arrivals go straight into the home server's buffer as they happen (no batching per interval); each server counts its own cell's arrivals for features 6-8.
- The task ledger, metrics and failure detector are shared Python objects because the demo is one process; they are only used for checking and display, never by a server's decisions. The failure detector is the same oracle as in the simulator.
- asyncio timers can fire a few ms late, so the demo's P2 check allows 0.02 x speed extra seconds.
