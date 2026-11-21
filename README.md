# ADAPT: Adaptive Dispatch And Polling-free Targeting

Decentralised deep-RL load balancing for edge networks (CSD438 course project).

Every edge server runs a copy of one small shared Q-network. Every 0.25 s it looks only at its
own buffer and queue plus the load reports its neighbours piggyback on normal replies, and
decides **when** to dispatch its buffered tasks (HOLD or now) and **where** (itself or one of
up to five neighbours). It never polls. A message protocol around the network (retries to the
same server, de-duplication, admission control, a hop limit and a hold cap) keeps every task
safe whatever the network outputs. The baseline is EVBLB (Sohrabi et al., GLOBECOM 2021).

`DESIGN_NOTES.md` lists every assumption and deviation from the coding brief, plus the results
in more detail.

## Results in short

1800 s runs, 50 servers, unseen test seeds 100-104 (`results/tables/` for every number with
95% confidence intervals, `results/figures/` for the plots):

| | EVBLB (B2) | ADAPT |
| --- | --- | --- |
| Mean response time, W1, rho = 0.7 | 8.3 s | 4.2 s |
| Mean response time, W4 (moving hotspot, never trained on), rho = 0.9 | 39.8 s | 13.6 s |
| Messages per task (all E3 runs) | 2.1 | 1.3 (no polls) |
| Blocked tasks, W1, rho = 0.9 | 2.0% | 4.3% |

Paired by seed, ADAPT's response time is lower than EVBLB's in all eight E3 scenarios; it does
block more tasks. The gap survives with all overhead costs set to zero (E9), and the same
network works for 10-100 servers (E7). No protocol invariant was violated in any run.

## Setup

Python 3.11. Torch is only needed to train; everything else runs on numpy.

```bash
python3.11 -m venv .venv            # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q                           # 88 tests, about 25 s (200 hypothesis examples)
```

## Reproducing one E3 cell from a fresh clone

The trained weights are in `weights/`, so this needs no training:

```bash
python -m experiments.run --config config/experiments/e3.yaml \
    --methods adapt evblb_dt2 --workloads W1 --rhos 0.7 --out results/check
```

10 runs, about 25 s on 8 cores. The overview it prints should show a mean response time of
4.21 s for `adapt` and 8.29 s for `evblb_dt2`, matching `results/e3/summary.csv`. Use `--out`
so the committed `results/e3/summary.csv` isn't overwritten by the one-cell run.

## Every command

| Command | What it does | Time (8 cores) |
| --- | --- | --- |
| `pytest -q` | unit, protocol, property-based, env, RL and demo tests | 25 s |
| `python -m sim.env --stub --steps 100` | random stub env with the real API | 1 s |
| `python -m sim.env --policy evblb --duration 300` | one run, prints its summary as JSON | 1-3 s |
| `python -m rl.train --config config/train.yaml` | train ADAPT (warm start + 20 episodes) | 1-2 min |
| `python -m rl.train --config config/train.yaml --variant noWS` | also `fixedDT`, `poll` | 1-2 min each |
| `python -m experiments.run --config config/experiments/e1.yaml` | E1: nearest vs EVBLB, n = 10-100 | 30 s |
| `python -m experiments.run --config config/experiments/e2.yaml` | E2: EVBLB's interval trade-off | 1 min |
| `python -m experiments.run --config config/experiments/e3.yaml` | E3 (and E4): all methods, W1-W4, rho 0.7/0.9 | 6.5 min |
| `python -m experiments.run --config config/experiments/e5.yaml` | E5: cold start, first 2000 s | 35 s |
| `python -m experiments.run --config config/experiments/e6.yaml` | E6: fixedDT / poll ablation | 50 s |
| `python -m experiments.run --config config/experiments/e7.yaml` | E7: n = 10, 25, 100 | 1 min |
| `python -m experiments.run --config config/experiments/e8.yaml` | E8: crashes + 2% message loss | 25 s |
| `python -m experiments.run --config config/experiments/e9.yaml` | E9: E3's W1 cell with zero costs | 1 min |
| `python -m experiments.run --config config/experiments/adapt_eval.yaml` | ADAPT variants vs EVBLB (M4 check) | 3.5 min |
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
python -m demo.run_demo --no-tty --crash-at 20 --crash-node 3   # scripted, for recording
```

Eight `ServerNode`s (the class the simulator uses) run as independent asyncio tasks and talk
through `AsyncTransport`. Each server builds its own observation and runs its own numpy copy of
`weights/adapt.npz`. In the live view press **0-7** to crash that server: its neighbours retry,
the failure detector confirms the crash and they take their unanswered tasks back. Press **q**
to quit. The invariant checker runs every second and its violation count is on screen. The
command exits with status 1 if an invariant broke.

## Layout

| Path | What is in it |
| --- | --- |
| `config/default.yaml` | every simulation constant |
| `config/experiments/` | one YAML per experiment (E1-E3, E5-E9, the ADAPT evaluation) |
| `config/train.yaml` | ADAPT training settings |
| `core/` | model, `ServerNode` and its protocol, messages, transports, metrics, invariant checks, per-node features |
| `sim/` | event engine, the RL env (`EdgeEnv`, `StubEnv`), workloads W1-W6, `run_episode` |
| `policies/` | B1 nearest, B2/B3 EVBLB, B4 ML-enhanced EVBLB, B6 perfect information, B7 TOPSIS, random, ADAPT inference |
| `rl/` | Q-network (torch and numpy), replay buffer, Double DQN, EVBLB imitation warm start, training |
| `weights/` | trained ADAPT weights as `.npz` (numpy inference); training also writes `.pt` torch checkpoints, which are not committed |
| `experiments/` | experiment runner, results tables, method display names |
| `plots/` | report figures |
| `demo/` | the asyncio demo |
| `tests/` | all tests |
| `results/` | `summary.csv` per experiment, training logs, `tables/`, `figures/` |

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
