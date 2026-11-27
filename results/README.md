# Results

Everything here was produced by the commands in the main README; labels (E1-E9, B1-B7, W1-W6,
F1-F9) are explained in its Glossary. All runs use the test seeds 100-104, which were never used
in training.

## Folders

| Folder | What it holds | Made by |
| --- | --- | --- |
| `e1/`, `e2/`, `e3/`, `e5/` ... `e9/` | `summary.csv` of that experiment (E4 has no folder: it is computed from the E3 runs) | `python -m experiments.run --config config/experiments/<e>.yaml` |
| `adapt_eval/` | `summary.csv` of ADAPT and its variants against EVBLB | `python -m experiments.run --config config/experiments/adapt_eval.yaml` |
| `training/` | one training log per ADAPT variant | `python -m rl.train --config config/train.yaml [--variant ...]` |
| `tables/` | every results table, as Markdown (`.md`) and LaTeX (`.tex`) | `python -m experiments.tables` |
| `figures/` | figures F1-F9, as PDF and PNG | `python -m plots.make_plots` |

The per-second time series of each run are not committed (they are large and the runner
recreates them). A folder `check/` appears if you run the quick check from the main README; it
is not committed either.

## `summary.csv` (one row per run)

| Column | Meaning |
| --- | --- |
| `method`, `workload`, `rho`, `n`, `seed` | which run this is: method name, workload, load ρ, number of servers, seed |
| `resp_mean`, `resp_p95` | mean and 95th-percentile response time (s) |
| `throughput` | tasks completed per second |
| `blocking_rate`, `blocked` | share and number of tasks turned away because a queue was full |
| `lbf_mean` | mean load balancing factor over the run (0 = perfectly even) |
| `makespan` | time of the last completion (s) |
| `time_to_balance` | first time the LBF stayed below 0.05 for 60 s (empty if never) |
| `msgs_per_task`, `msgs_total`, `msg_<TYPE>` | protocol messages per completed task, in total, and by message type |
| `rounds_per_server_s` | dispatch rounds per server per second |
| `decision_latency` | mean time from a dispatch decision until the batch leaves, including any polling (s) |
| `created`, `completed`, `unfinished` | tasks created, completed, and still unfinished when the run stopped |
| `lost_in_crash` | tasks that were inside a server when it crashed (E8 only) |
| `violations` | breaks of the guarantees P1-P5 (0 in every run) |
| `reruns_after_crash` | tasks run twice after a crash, the one exception to P1 (0 in every run) |
| `wall_clock`, `sim_end` | real seconds the run took, and the simulated time it ended at |

## `training/*.csv` (one row per training episode)

`adapt.csv` is full ADAPT; `adapt_noWS.csv`, `adapt_fixedDT.csv` and `adapt_poll.csv` are the
variants. `phase` is `expert` for the EVBLB runs used by the warm start, then `train`. Each row
has the episode's workload, load and seed, its reward and the five reward terms (`r_wait` ...
`r_blocked`), response time, LBF, blocking, messages, the exploration rate `epsilon` and the
training `loss`. `eval_reward` and `eval_resp` come from a greedy run on one fixed training
scenario after each episode; they are the learning curve in F7.

## Tables

| File | Contents |
| --- | --- |
| `e1` | E1: response time, blocking and LBF of B1 and B2 for each network size |
| `e2` | E2: EVBLB at each dispatch interval and load |
| `e3_response`, `e3_p95`, `e3_blocking`, `e3_lbf`, `e3_throughput` | E3: one metric per table, methods as columns, scenarios as rows |
| `e3_paired` | E3: ADAPT minus each baseline, compared seed by seed, for response time and blocking |
| `e4` | E4: messages and dispatch overhead per method over all E3 runs |
| `e5` | E5: cold-start metrics per method |
| `e6` | E6: ablation (ADAPT, fixedDT, poll) per workload |
| `e7` | E7: response time per network size |
| `e8` | E8: crashes and message loss |
| `e9` | E9: response time and messages with the added costs on and off |

Every cell is the mean ± the half-width of the 95% confidence interval over the runs.

## Figures

| File | Figure |
| --- | --- |
| `f1_e1_lbf` | F1: LBF over time, nearest-only vs EVBLB (E1) |
| `f2_e2_interval` | F2: EVBLB's mean response time against its dispatch interval (E2) |
| `f3_e3_response` | F3: mean response time per scenario and method (E3) |
| `f4_e3_blocking` | F4: blocking rate per scenario and method (E3) |
| `f5_e4_messages` | F5: messages per completed task over all E3 runs (E4) |
| `f6_e5_cold_start` | F6: LBF over the first 2000 s (E5) |
| `f7_training` | F7: evaluation reward after each training episode, with and without warm start |
| `f8_effective_interval` | F8: time between dispatches against server utilisation (E3, W2 and W3) |
| `f9_e9_costs` | F9: response time with the added costs on (E3) and off (E9), W1, ρ 0.7 |
