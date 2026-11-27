# ADAPT: Adaptive Dispatch And Polling-free Targeting

Decentralised deep reinforcement learning (RL) for load balancing in edge networks.  
Course project for CSD438 Distributed Systems.

**Report:** [PDF](https://github.com/adyasinghal/ADAPT-Deep-RL-load-balancer/releases/download/v1.0-docs/Learning_When_and_Where_to_Offload__Decentralised_Deep_Reinforcement_Learning_for_Adaptive_Load_Balancing_in_Edge_Networks.pdf) · **Presentation:** [PPTX](https://github.com/adyasinghal/ADAPT-Deep-RL-load-balancer/releases/download/v1.0-docs/ADAPT.Learning.When.and.Where.to.Offload.pptx)

Every edge server runs a copy of one small shared neural network. Every 0.25 s it looks only at
what it knows locally (its own buffer and queue, plus the load its neighbours last reported on
replies they send anyway) and decides **when** to pass on the tasks waiting in its buffer and
**where** to send them: keep them, or hand them to one of up to five neighbours. It never asks
its neighbours how busy they are (no polling). A message protocol around the network keeps
every task safe whatever the network decides. We compare ADAPT with EVBLB [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358) and its learned
variants in an event-driven simulator, and run it as eight independent asyncio servers in a
demo.

Labels such as W1, B2, E3 or F5 and terms such as ρ or LBF are explained in the
[Glossary](#glossary). Numbers in square brackets refer to the [Reference papers](#reference-papers).

## Contents

- [The problem](#the-problem)
- [How ADAPT works](#how-adapt-works)
- [Results in short](#results-in-short)
- [Reference papers](#reference-papers)
- [Setup](#setup)
- [Quick check: reproduce the main result](#quick-check-reproduce-the-main-result)
- [Every command](#every-command)
- [Live demo](#live-demo)
- [Repository layout](#repository-layout)
- [Glossary](#glossary)
- [Using the env from other code](#using-the-env-from-other-code)

## The problem

Edge servers sit next to the users (at base stations), so they answer much faster than a
distant cloud, but each one is small. Users do not spread out evenly, so some servers get far
more tasks than others: their queues fill up, tasks wait, and new tasks are turned away
(**blocked**), while neighbouring servers sit idle. Moving work to where there is room fixes
this, but each server must decide on its own, with partial and slightly old information.

Every server keeps making two decisions:

- **When** to dispatch: keep collecting new tasks in its buffer, or send the batch on now?
  Dispatching often reacts quickly but costs a dispatch round (and its messages) each time.
- **Where** to dispatch: run the batch itself, or hand it to a neighbour? Choosing well needs
  each neighbour's current load, and asking for it (polling) costs a round trip every time.

Existing methods fix one of these decisions and pay for the other. EVBLB [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358) dispatches every
2 s and polls all neighbours before every dispatch; its learned follow-ups [[2]](https://doi.org/10.1016/j.comcom.2024.04.009), [[3]](https://doi.org/10.1016/j.comcom.2025.108337) learn only
one of the two decisions, or rely on network-wide information that no single server has.
ADAPT learns both decisions with one network, using only local information.

## How ADAPT works

**What a server observes** (30 numbers, all local; `sim/env.py` and `core/features.py`):

| Features | What they describe |
| --- | --- |
| 0-2 | the server itself: utilisation, queue length, CPU speed |
| 3-5 | its buffer: number of tasks, their total CPU demand, age of the oldest task |
| 6-8 | traffic in its area: arrival rate, last mean task size, task-size variance (from ATOA [[5]](https://doi.org/10.1109/ICASSP48485.2024.10445952)) |
| 9 | local imbalance: spread of utilisation across the server and its neighbours |
| 10-29 | 5 neighbour slots, 4 numbers each: last reported room left, age of that report, distance, whether the slot is filled |

**What it can do** (7 actions, `core/actions.py`): HOLD (keep collecting), LOCAL (move the
whole buffer into its own queue), or SEND_1 ... SEND_5 (offer the whole buffer to neighbour
1-5). Actions that the protocol forbids are masked out before the network chooses.

**What it is rewarded for**, every 0.25 s (weights in `config/default.yaml`):
reward = −(waiting time + 0.5 × local imbalance + 0.2 × dispatch cost + 0.1 × distance sent +
5 × blocked tasks), each term normalised.

**How it is trained** (`rl/`, `config/train.yaml`), with centralised training and
decentralised execution:

1. **Warm start:** the network first imitates EVBLB, so it starts from a sensible policy.
2. **Dueling Double DQN** [[6]](https://doi.org/10.1038/nature14236), [[7]](https://doi.org/10.1609/aaai.v30i1.10295), [[8]](https://arxiv.org/abs/1511.06581): one network (30 → 128 → 128 → 7) is trained on the
   experience of all 50 servers, for 20 episodes on workloads W1 and W2 at loads 0.5-0.9.
3. **Execution:** each server runs its own frozen copy of the trained network, on its own data
   only. The trained weights are in `weights/`.

**The protocol** (`core/node.py`). Its six rules:

1. A server that offers a batch keeps it until the receiver answers; if no answer comes, it
   resends the same offer to the same server. Only after repeated timeouts does it ask the
   failure detector and, if the receiver has really crashed, take the tasks back.
2. The receiver drops duplicate offers by task id.
3. The receiver admits only what fits its real queue and capacity.
4. A rejected task goes back to the buffer with its hop count increased; after two rejections
   it can only be kept locally.
5. A server cannot hold a task in its buffer for more than 2 s.
6. Only buffered tasks are dispatched; tasks already in a queue never move.

Together these give five guarantees that hold **whatever the network outputs**. They are
checked every simulated second in every run (`core/invariants.py`) and by property-based tests
(`tests/test_properties.py`):

| | Guarantee |
| --- | --- |
| P1 | No task runs twice (exception: if a server crashes *and* every reply about a task it finished is lost; counted separately, never observed) |
| P2 | A task leaves the buffer within 2.25 s (2 s hold limit + one 0.25 s decision interval) |
| P3 | A task is offered at most 3 times |
| P4 | No queue ever holds more than 500 tasks |
| P5 | Response time is bounded (from P2, P3 and first-come-first-served queues) |

`DESIGN_NOTES.md` records the modelling and implementation choices we made where the papers
leave details open, and where we changed our first design, with the reason for each. It also
has the results in more detail.

## Results in short

1800 s runs, 50 servers, 5 test seeds (100-104) that were never used in training.
`results/tables/` has every number with 95% confidence intervals and `results/figures/` the
plots; `results/README.md` explains what each file is.

| | EVBLB (B2) | ADAPT |
| --- | --- | --- |
| Mean response time, W1 (steady traffic), ρ = 0.7 | 8.3 s | 4.2 s |
| Mean response time, W4 (moving hotspot, never trained on), ρ = 0.9 | 39.8 s | 13.6 s |
| Messages per task (all E3 runs) | 2.1 | 1.3 (no polls) |
| Blocked tasks, W1, ρ = 0.9 | 2.0% | 4.3% |

Compared seed by seed, ADAPT's mean response time is lower than EVBLB's in all eight E3
scenarios (4 workloads × 2 loads); it does block more tasks. The gap remains when all overhead
costs are set to zero (E9), and the network trained on 50 servers also works for 10-100
servers (E7). No guarantee (P1-P5) was violated in any run.

## Reference papers

The five papers this project builds on and compares with:

| | Paper | What it does | How this repository uses it |
| --- | --- | --- | --- |
| [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358) | V. Sohrabi, M. E. Esmaeili, M. Dolati, A. Khonsari, A. Dadlani, "[EVBLB: Efficient Voronoi tessellation-based load balancing in edge computing networks](https://doi.org/10.1109/GLOBECOM46510.2021.9685358)," IEEE GLOBECOM, 2021. | Maps each user to its nearest server (Voronoi cells); every 2 s a server polls its neighbours and sends its whole batch to the one with the most room left. | Our system model (`core/model.py`), baselines B2, B3 and B6 (`policies/evblb.py`), and the expert ADAPT imitates in its warm start (`rl/imitation.py`). |
| [[2]](https://doi.org/10.1016/j.comcom.2024.04.009) | M. E. Esmaeili, A. Khonsari, V. Sohrabi, A. Dadlani, "[Reinforcement learning-based dynamic load balancing in edge computing networks](https://doi.org/10.1016/j.comcom.2024.04.009)," Computer Communications, vol. 222, 2024. | "ML-enhanced EVBLB": a Q-learner picks the dispatch interval and a bandit picks the neighbour, as two separate learners, with no polling. | Baseline B4 (`policies/ml_evblb.py`); the closest prior work. Our reward builds on its reward. |
| [[3]](https://doi.org/10.1016/j.comcom.2025.108337) | M. E. Esmaeili, A. Khonsari, M. Dolati, "[Dynamic distance-based load balancing in mobile edge computing with deep reinforcement learning](https://doi.org/10.1016/j.comcom.2025.108337)," Computer Communications, vol. 244, 2025. | "Deep Q radius": a deep Q-network picks the radius within which EVBLB looks for neighbours, using network-wide information. | Not re-implemented (that is why there is no B5). We take its DQN set-up and its traffic inputs (features 6-7), but keep every input local. |
| [[4]](https://doi.org/10.3390/electronics13081511) | Z. Wu, Z. Jia, X. Pang, S. Zhao, "[Deep reinforcement learning-based task offloading and load balancing for vehicular edge computing](https://doi.org/10.3390/electronics13081511)," Electronics, vol. 13, no. 8, 2024. | "TOLB": one central deep RL agent; an overloaded server moves tasks to the server with the best TOPSIS score (free resources and distance). | Baseline B7 (`policies/topsis.py`): its TOPSIS rule. |
| [[5]](https://doi.org/10.1109/ICASSP48485.2024.10445952) | L. Wang, J. Zhang, "[Adaptive multi-armed bandit learning for task offloading in mobile edge computing](https://doi.org/10.1109/ICASSP48485.2024.10445952)," IEEE ICASSP, 2024. | "ATOA": treats offloading as a bandit and switches between two bandit strategies depending on how variable the traffic is. | Its task-size variance is ADAPT's feature 8, and its tidal traffic is our workload W2 (`sim/workloads.py`). |

The learning methods ADAPT uses:

- [[6]](https://doi.org/10.1038/nature14236) V. Mnih et al., "[Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236)," Nature, 2015 (DQN).
- [[7]](https://doi.org/10.1609/aaai.v30i1.10295) H. van Hasselt, A. Guez, D. Silver, "[Deep reinforcement learning with double Q-learning](https://doi.org/10.1609/aaai.v30i1.10295)," AAAI, 2016 (Double DQN).
- [[8]](https://arxiv.org/abs/1511.06581) Z. Wang et al., "[Dueling network architectures for deep reinforcement learning](https://arxiv.org/abs/1511.06581)," ICML, 2016 (dueling network).

## Setup

Python 3.11. Torch is only needed to train; everything else runs on numpy.

```bash
python3.11 -m venv .venv            # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q                           # 85 tests, about 25 s (200 hypothesis examples)
```

Run every command below from the repository folder with the environment activated.

## Quick check: reproduce the main result

The full comparison (experiment E3) runs every method on four workloads at two loads and takes
about 6.5 minutes. To check our headline number in under a minute, run just one part of it:
ADAPT against EVBLB on the steady workload (W1) at load 0.7.

```bash
python -m experiments.run --config config/experiments/e3.yaml \
    --methods adapt evblb_dt2 --workloads W1 --rhos 0.7 --out results/check
```

- No training is needed: the trained ADAPT network is already in `weights/`.
- It runs 10 simulations (each method on the 5 test seeds) and takes under half a minute.
- When it finishes it prints a summary table. Its `resp_mean` column (mean response time)
  should read **4.211 for ADAPT** (`adapt`) and **8.288 for EVBLB** (`evblb_dt2`), the same
  as in the results table above and in `results/e3/summary.csv`.
- `--out results/check` writes this run's results to a separate folder, so the committed E3
  results are not overwritten.

## Every command

| Command | What it does | Time (8 cores) |
| --- | --- | --- |
| `pytest -q` | unit, protocol, property-based, env, RL and demo tests | 25 s |
| `python -m sim.env --policy evblb --duration 300` | one run, prints its summary as JSON | 1-3 s |
| `python -m rl.train --config config/train.yaml` | train ADAPT (warm start + 20 episodes) | 1-2 min |
| `python -m rl.train --config config/train.yaml --variant noWS` | also `fixedDT`, `poll` | 1-2 min each |
| `python -m experiments.run --config config/experiments/e1.yaml` | E1: nearest vs EVBLB, n = 10-100 | 30 s |
| `python -m experiments.run --config config/experiments/e2.yaml` | E2: EVBLB's interval trade-off | 1 min |
| `python -m experiments.run --config config/experiments/e3.yaml` | E3 (and E4): all methods, W1-W4, ρ 0.7/0.9 | 6.5 min |
| `python -m experiments.run --config config/experiments/e5.yaml` | E5: cold start, first 2000 s | 35 s |
| `python -m experiments.run --config config/experiments/e6.yaml` | E6: fixedDT / poll ablation | 50 s |
| `python -m experiments.run --config config/experiments/e7.yaml` | E7: n = 10, 25, 100 | 1 min |
| `python -m experiments.run --config config/experiments/e8.yaml` | E8: crashes + 2% message loss | 25 s |
| `python -m experiments.run --config config/experiments/e9.yaml` | E9: E3's W1 cell with zero costs | 1 min |
| `python -m experiments.run --config config/experiments/adapt_eval.yaml` | ADAPT variants vs EVBLB | 3.5 min |
| `python -m experiments.tables` | every table, markdown + LaTeX, into `results/tables/` | 1 s |
| `python -m plots.make_plots` | every figure, PDF + PNG, into `results/figures/` | 3 s |
| `python -m demo.run_demo` | live demo (below) | 60 s |

Runner options: `--quick` (300 s, one seed), `--duration`, `--seeds`, `--workers`, and the
`--methods / --workloads / --rhos / --n` filters to run part of a grid.

Only each experiment's `summary.csv` is committed. Tables and most figures are made from those;
F1, F6 and F8 also need the per-run time series, so rerun E1, E5 and E3 before
`make_plots` if you want those three redrawn.

## Live demo

```bash
python -m demo.run_demo                   # 8 servers, 60 s, real time, live terminal view
python -m demo.run_demo --speed 4         # 4 simulated seconds per real second
```
<!-- python -m demo.run_demo --no-tty --crash-at 20 --crash-node 3   # scripted, for recording -->

Eight `ServerNode`s (the class the simulator uses) run as independent asyncio tasks and talk
through `AsyncTransport`. Each server builds its own observation and runs its own numpy copy of
`weights/adapt.npz`. In the live view press **0-7** to crash that server: its neighbours retry,
the failure detector confirms the crash and they take their unanswered tasks back. Press **q**
to quit. The invariant checker runs every second and its violation count is on screen. The
command exits with status 1 if an invariant broke.

## Repository layout

| Path | What is in it |
| --- | --- |
| `README.md` | this overview |
| `DESIGN_NOTES.md` | modelling and implementation choices, with reasons, and detailed results |
| `config/default.yaml` | every simulation constant |
| `config/experiments/` | one YAML per experiment (E1-E3, E5-E9, the ADAPT evaluation); each starts with a comment saying what it tests |
| `config/train.yaml` | ADAPT training settings |
| `core/` | model, `ServerNode` and its protocol, messages, transports, metrics, invariant checks, per-node features |
| `sim/` | event engine, the RL env (`EdgeEnv`), workloads W1-W6, `run_episode` |
| `policies/` | B1 nearest, B2/B3 EVBLB, B4 ML-enhanced EVBLB, B6 perfect information, B7 TOPSIS, random, ADAPT inference |
| `rl/` | Q-network (torch and numpy), replay buffer, Double DQN, EVBLB imitation warm start, training |
| `weights/` | trained ADAPT networks as `.npz` (see the Glossary for the variants); training also writes `.pt` torch checkpoints, which are not committed |
| `experiments/` | experiment runner, results tables, method display names |
| `plots/` | figures F1-F9 |
| `demo/` | the asyncio demo |
| `tests/` | all tests |
| `results/` | `summary.csv` per experiment, training logs, `tables/`, `figures/`; see `results/README.md` |

## Glossary

**Compared methods** (B1-B7; B5 was reserved for Deep Q radius [[3]](https://doi.org/10.1016/j.comcom.2025.108337), which we did not
re-implement):

| Label | Method | Name in results | Code |
| --- | --- | --- | --- |
| B1 | Nearest server only: every task runs where it arrives, no balancing (worst case) | `nearest` | `policies/nearest.py` |
| B2 | EVBLB [[1]](https://doi.org/10.1109/GLOBECOM46510.2021.9685358) with its published 2 s dispatch interval | `evblb_dt2` | `policies/evblb.py` |
| B3 | EVBLB with its best interval from E2 (0.25 s) | `evblb_dt0.25` | `policies/evblb.py` |
| B4 | ML-enhanced EVBLB [[2]](https://doi.org/10.1016/j.comcom.2024.04.009) | `ml_evblb` | `policies/ml_evblb.py` |
| B6 | Perfect information: EVBLB with free, instant load reports (a reference, not realistic) | `perfect` | `policies/evblb.py` |
| B7 | TOPSIS target selection from TOLB [[4]](https://doi.org/10.3390/electronics13081511) | `topsis` | `policies/topsis.py` |
| | ADAPT (ours) | `adapt` | `policies/adapt.py` |

**ADAPT variants** (trained networks in `weights/`):

| Name | Weights | What it is |
| --- | --- | --- |
| `adapt` | `adapt.npz` | full ADAPT |
| `adapt_noWS` | `adapt_noWS.npz` | trained without the EVBLB warm start |
| `adapt_fixedDT` | `adapt_fixedDT.npz` | dispatch interval fixed at 2 s, learns only *where* |
| `adapt_poll` | `adapt_poll.npz` | polls neighbours for fresh load before every dispatch |
| `adapt_ep0` | `adapt_bc.npz` | the network straight after the warm start, before any RL training |

**Workloads** (`sim/workloads.py`):

| Label | Traffic |
| --- | --- |
| W1 | steady: constant arrival rate, users spread uniformly |
| W2 | tidal: calm and volatile phases alternate (from ATOA [[5]](https://doi.org/10.1109/ICASSP48485.2024.10445952)) |
| W3 | flash crowds: short bursts at 3× the normal rate |
| W4 | moving hotspot: a cluster of users that jumps to a new place every few minutes |
| W5 | diurnal: load rises and falls slowly over a run (implemented, not used in the experiments) |
| W6 | W1's traffic with server crashes and message loss (used in E8) |

ADAPT is trained only on W1 and W2, so W3 and W4 test how well it generalises.

**Experiments** (configs in `config/experiments/`, results in `results/`):

| Label | Question it answers |
| --- | --- |
| E1 | Does our simulator reproduce EVBLB's basic behaviour? (B1 vs B2, 10-100 servers) |
| E2 | How does EVBLB's dispatch interval trade response time against messages? |
| E3 | Main comparison: every method on W1-W4 at ρ = 0.7 and 0.9 |
| E4 | Overhead (messages, dispatch rounds) per method; computed from the E3 runs, so it has no config of its own |
| E5 | Cold start: how quickly does each method balance the load in the first 2000 s? |
| E6 | Ablation: how much of ADAPT's gain comes from learning *when* (fixedDT) and from fresh polled information (poll)? |
| E7 | Scale: does the network trained on 50 servers work for 10, 25 and 100? |
| E8 | Robustness: 10% of servers crash and 2% of messages are lost (W6) |
| E9 | Fairness check: E3's W1 cell with every added cost set to zero |
| `adapt_eval` | ADAPT and its variants against EVBLB, before the main experiments |

**Figures** (`results/figures/`):

| Label | Figure |
| --- | --- |
| F1 | LBF over time, nearest-only vs EVBLB (E1) |
| F2 | EVBLB's mean response time against its dispatch interval (E2) |
| F3 | Mean response time per scenario and method (E3) |
| F4 | Blocking rate per scenario and method (E3) |
| F5 | Messages per completed task (E4) |
| F6 | LBF over the first 2000 s (E5) |
| F7 | Training curve: evaluation reward after each episode, with and without warm start |
| F8 | Time between dispatches against server utilisation (E3, W2 and W3) |
| F9 | Response time with the added costs on (E3) and off (E9) |

**Terms**:

| Term | Meaning |
| --- | --- |
| task | one user request to be processed |
| buffer | where a server collects new tasks until it dispatches them |
| queue | the tasks a server has accepted to run itself, first come, first served (at most 500) |
| dispatch | moving the whole buffer: into the server's own queue (LOCAL) or offered to a neighbour (SEND) |
| polling | asking neighbours for their current load before deciding; costs a round trip |
| ρ (rho), load | arrival rate as a fraction of total server capacity; 0.7 = busy, 0.9 = near full |
| response time | time from a task's arrival to its completion |
| blocked | a task turned away because the queue it was meant for is full |
| LBF | load balancing factor: spread (standard deviation) of utilisation across servers; 0 = perfectly even |
| messages per task | protocol messages sent per completed task |
| invariant violation | any break of the guarantees P1-P5; must stay 0 |
| re-run | a task run a second time after a crash (the P1 exception); counted separately |
| test seeds | random seeds 100-104; each gives a different topology and traffic, never used in training (training used seeds 0-9) |
| compared seed by seed | ADAPT and a baseline run on the same seed (same topology and traffic) and their results subtracted, which removes the variation between seeds |
| n | number of edge servers |

## Using the env from other code

```python
from core.config import load_config
from policies import make_policy
from sim.env import EdgeEnv

policy = make_policy({"name": "adapt"})             # or {"name": "evblb", "dt": 1.0}, ...
env = EdgeEnv(load_config(dotted={"workload.rho": 0.7}), policy)   # the policy supplies its traits
obs, info = env.reset(seed=100)                     # obs (n, 30), info["masks"] (n, 7)
policy.reset(env.topology, env.policy_rng)
done = False
while not done:
    actions = policy.act(obs, info["masks"], env.views)
    obs, rewards, done, info = env.step(actions)    # info["reward_parts"]: (n, 5) components
print(env.summary())
```

`sim.rollout.run_episode(cfg, policy, seed)` does exactly this and returns the time series and
the summary.
