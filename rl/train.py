"""Train ADAPT's shared Q-network.

    python -m rl.train --config config/train.yaml --variant noWS     # no warm start
    python -m rl.train --config config/train.yaml --variant fixedDT  # learns only "where"
    python -m rl.train --config config/train.yaml --variant poll     # fresh polled loads

Writes ``weights/adapt[_<variant>].pt`` and ``.npz`` plus one row per episode to
``results/training/adapt[_<variant>].csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.actions import HOLD, LOCAL
from core.config import ROOT, load_config, read_yaml
from policies.adapt import AdaptPolicy, weights_stem
from rl.dqn import DQNAgent, DQNConfig, linear_epsilon
from rl.replay import ReplayBuffer
from sim.env import EdgeEnv

REWARD_PARTS = ["r_wait", "r_lbf", "r_dispatch", "r_distance", "r_blocked"]


class Trainer:
    """Holds the agent, the replay buffer and the episode schedule of one training run."""

    def __init__(self, tcfg: dict[str, Any], variant: str) -> None:
        self.tcfg = tcfg
        self.variant = variant
        seed = int(tcfg["seed"])
        torch.set_num_threads(1)  # a 30-128-128-7 MLP is faster single-threaded
        self.rng = np.random.default_rng(seed)
        self._scenario_rng = np.random.default_rng(seed + 1)
        self.agent = DQNAgent(DQNConfig.from_dict(tcfg["dqn"]), seed=seed)
        self.replay = ReplayBuffer(int(tcfg["replay_capacity"]), np.random.default_rng(seed + 2))
        self.policy = AdaptPolicy(variant)  # only for its env traits and masks
        self.explore = tcfg["exploration"]
        self.steps = 0
        self._episode = 0  # scenarios drawn so far (expert episodes included)
        self.trained_episodes = 0

    # ------------------------------------------------------------------ scenarios

    def next_env(self) -> tuple[EdgeEnv, int, dict[str, Any]]:
        """Env for the next episode: alternate workloads, random rho, cycle training seeds."""
        t = self.tcfg
        k = self._episode
        self._episode += 1
        scenario = {
            "workload": t["workloads"][k % len(t["workloads"])],
            "rho": float(self._scenario_rng.choice(t["rhos"])),
            "seed": int(t["train_seeds"][k % len(t["train_seeds"])]),
        }
        return self.make_env(scenario), scenario["seed"], scenario

    def make_env(self, scenario: dict[str, Any]) -> EdgeEnv:
        """Training env for a {workload, rho} scenario."""
        t = self.tcfg
        dotted = {
            "run.duration": float(t["episode_duration"]),
            "topology.n_servers": int(t["n_servers"]),
            "workload.name": scenario["workload"],
            "workload.rho": scenario["rho"],
        }
        cfg = load_config(t.get("scenario") or {}, dotted)
        return EdgeEnv(cfg, self.policy)

    # ------------------------------------------------------------------ phases

    def train_episode(self) -> dict[str, Any]:
        """One epsilon-greedy episode with one gradient step per decision instant."""
        env, seed, scenario = self.next_env()
        explore, cfg = self.explore, self.agent.cfg
        obs, info = env.reset(seed)
        mask = self.policy.effective_masks(info["masks"], env.views)
        started = time.perf_counter()
        rewards, parts, losses = [], [], []
        choices = dispatches = 0
        done = False
        while not done:
            eps = linear_epsilon(
                self.steps, explore["eps_start"], explore["eps_end"], int(explore["decay_steps"])
            )
            action = self.agent.act(obs, mask, eps, self.rng)
            next_obs, reward, done, info = env.step(action)
            next_mask = self.policy.effective_masks(info["masks"], env.views)
            self.replay.add(obs, mask, action, reward, next_obs, next_mask)
            self.steps += 1
            if len(self.replay) >= int(self.tcfg["learning_starts"]):
                for _ in range(int(self.tcfg["updates_per_epoch"])):
                    losses.append(self.agent.update(self.replay.sample(cfg.batch_size)))
            # how often the agent dispatches when it actually has a choice
            free = mask[:, LOCAL] & mask[:, HOLD]
            choices += int(free.sum())
            dispatches += int((action[free] != HOLD).sum())
            rewards.append(float(reward.mean()))
            parts.append(info["reward_parts"].mean(axis=0))
            obs, mask = next_obs, next_mask
        summary = env.summary()
        self.trained_episodes += 1
        row = {
            "phase": "train",
            "episode": self.trained_episodes - 1,
            **scenario,
            "reward": float(np.mean(rewards)),
            **dict(zip(REWARD_PARTS, np.mean(parts, axis=0).tolist(), strict=True)),
            "resp_mean": summary["resp_mean"],
            "lbf_mean": summary["lbf_mean"],
            "blocking_rate": summary["blocking_rate"],
            "msgs_per_task": summary["msgs_per_task"],
            "dispatch_rate": dispatches / max(choices, 1),
            "epsilon": eps,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "steps": self.steps,
            "wall_clock": time.perf_counter() - started,
        }
        if self.tcfg.get("eval", {}).get("enabled"):
            row.update(self.evaluate())
        return row

    def evaluate(self) -> dict[str, float]:
        """Greedy episode (no exploration, no learning) on the fixed eval scenario."""
        scenario = {k: self.tcfg["eval"][k] for k in ("workload", "rho", "seed")}
        env = self.make_env(scenario)
        obs, info = env.reset(int(scenario["seed"]))
        rewards = []
        done = False
        while not done:
            mask = self.policy.effective_masks(info["masks"], env.views)
            obs, reward, done, info = env.step(self.agent.act(obs, mask, 0.0, self.rng))
            rewards.append(float(reward.mean()))
        summary = env.summary()
        return {
            "eval_reward": float(np.mean(rewards)),
            "eval_resp": summary["resp_mean"],
            "eval_lbf": summary["lbf_mean"],
        }


def train(tcfg: dict[str, Any], variant: str, stem: Path, log_path: Path) -> pd.DataFrame:
    """Full training run; saves weights after every episode and returns the episode log."""
    trainer = Trainer(tcfg, variant)
    rows: list[dict[str, Any]] = []
    meta = {"variant": variant, "config": tcfg}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(int(tcfg["episodes"])):
        row = trainer.train_episode()
        rows.append(row)
        trainer.agent.save(stem, {**meta, "episodes": row["episode"] + 1})
        pd.DataFrame(rows).to_csv(log_path, index=False)
        print(
            f"ep {row['episode']:3d} {row['workload']} rho={row['rho']:.1f} seed={row['seed']}: "
            f"reward {row['reward']:.3f}, resp {row['resp_mean']:.2f} s, "
            f"LBF {row['lbf_mean']:.3f}, dispatch {row['dispatch_rate']:.2f}, "
            f"eps {row['epsilon']:.2f}, loss {row['loss']:.4f} ({row['wall_clock']:.0f} s)"
            + (
                f" | eval reward {row['eval_reward']:.3f}, resp {row['eval_resp']:.2f} s"
                if "eval_reward" in row
                else ""
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train ADAPT's shared Q-network.")
    parser.add_argument("--config", default=str(ROOT / "config" / "train.yaml"))
    parser.add_argument("--variant", default="noWS", choices=["full", "noWS", "fixedDT", "poll"])
    parser.add_argument("--episodes", type=int, help="override the number of episodes")
    parser.add_argument("--out", help="weight file stem (default weights/adapt[_variant])")
    args = parser.parse_args()

    tcfg = read_yaml(args.config)
    if args.episodes is not None:
        tcfg["episodes"] = args.episodes
    stem = Path(args.out) if args.out else weights_stem(args.variant)
    log_path = ROOT / "results" / "training" / f"{stem.name}.csv"
    started = time.perf_counter()
    train(tcfg, args.variant, stem, log_path)
    print(f"saved {stem}.pt / .npz, log {log_path} ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
