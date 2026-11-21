# ADAPT: Adaptive Dispatch And Polling-free Targeting

Decentralised deep-RL load balancing for edge networks (CSD438 course project).
Every edge server runs a copy of one small shared DQN and decides every 0.25 s whether to
hold its buffered tasks or send them to itself or to one of up to five neighbours. The
baseline it is compared against is EVBLB (Sohrabi et al., GLOBECOM 2021).

See `DESIGN_NOTES.md` for every assumption we made on top of the brief.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

| Path | What is in it |
| --- | --- |
| `config/default.yaml` | every simulation constant |
| `config/experiments/` | one YAML per experiment (E1, E2, the ADAPT evaluation) |
| `core/` | model, `ServerNode` protocol, messages, transport, metrics, invariant checks |
| `sim/` | event engine, the RL env (`EdgeEnv`, `StubEnv`), workloads, `run_episode` |
| `policies/` | baselines: nearest (B1), EVBLB (B2/B3), perfect information (B6), random; ADAPT inference |
| `rl/` | Q-network, replay buffer, Double DQN, EVBLB imitation warm start, training script |
| `config/train.yaml` | training settings |
| `weights/` | trained ADAPT weights (`.pt` for torch, `.npz` for numpy inference) |
| `experiments/run.py` | runs an experiment grid and writes CSVs to `results/<exp>/` |
| `tests/` | unit, protocol and property-based tests |

## Running things

```bash
pytest -q                                              # all tests, ~20 s (200 hypothesis examples)
python -m sim.env --stub --steps 100                   # random stub env, same API as the real one
python -m sim.env --policy evblb --duration 300        # one run, prints the summary as JSON
python -m sim.env --policy nearest --rho 0.9 --workload W2
python -m experiments.run --config config/experiments/e1.yaml          # E1, ~30 s on 8 cores
python -m experiments.run --config config/experiments/e2.yaml          # E2, ~60 s on 8 cores
python -m experiments.run --config config/experiments/e1.yaml --quick  # smoke test
```

A single 1800 s run with n = 50 takes 3-14 s of wall clock depending on the policy and the load.

## Using the env from the agent code

```python
from core.config import load_config
from sim.env import EdgeEnv

env = EdgeEnv(load_config(dotted={"workload.rho": 0.7}), policy)   # policy supplies its traits
obs, info = env.reset(seed=0)                       # obs (n, 30), info["masks"] (n, 7)
while True:
    actions = policy.act(obs, info["masks"], env.views)
    obs, rewards, done, info = env.step(actions)    # info["reward_parts"]: (n, 5) components
    if done:
        break
print(env.summary())
```

## Training and evaluating ADAPT

```bash
python -m rl.train --config config/train.yaml                     # ADAPT, ~1 min on a laptop CPU
python -m rl.train --config config/train.yaml --variant noWS      # also fixedDT, poll
python -m experiments.run --config config/experiments/adapt_eval.yaml   # 140 runs, ~3.5 min on 8 cores
```

Training writes `weights/adapt[_variant].pt/.npz` (and `_bc` weights straight after the warm
start) and a per-episode log to `results/training/`. The `eval_reward` column of that log is
the learning curve. In experiment YAMLs ADAPT is `{name: adapt}` or
`{name: adapt, variant: noWS}`; `weights:` points at any other `.npz`.
