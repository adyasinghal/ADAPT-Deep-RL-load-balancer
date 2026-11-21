"""Run an experiment grid from a YAML file and write its CSVs.

    python -m experiments.run --config config/experiments/e2.yaml
    python -m experiments.run --config config/experiments/e1.yaml --quick   # smoke test

An experiment file looks like::

    name: e2
    scenario: {run: {duration: 1800}}        # optional nested overrides of default.yaml
    methods:
      - {name: evblb, dt: 0.25}              # optional "label"; defaults to the policy's name
    grid:
      workload: [W1]
      rho: [0.5, 0.9]
      n: [50]
      seed: [100, 101, 102, 103, 104]

Every (method, workload, rho, n, seed) combination is one run. Each run writes its time series
to ``results/<exp>/<label>__<workload>__rho<rho>__n<n>__seed<seed>.csv`` and one row of
``results/<exp>/summary.csv``. Runs go to a process pool.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import ROOT, deep_merge, load_config, read_yaml
from policies import make_policy
from sim.rollout import run_episode

ID_COLUMNS = ["method", "workload", "rho", "n", "seed"]


def method_label(spec: dict[str, Any]) -> str:
    """Name used in file names and tables for a method entry."""
    return str(spec.get("label") or make_policy(spec).name)


def run_file_name(label: str, workload: str, rho: float, n: int, seed: int) -> str:
    """Time-series file name of one run."""
    return f"{label}__{workload}__rho{rho:g}__n{n}__seed{seed}.csv"


def expand(exp: dict[str, Any]) -> list[dict[str, Any]]:
    """All runs of an experiment, as picklable job dicts."""
    grid = exp["grid"]
    jobs = []
    for spec, workload, rho, n, seed in itertools.product(
        exp["methods"], grid["workload"], grid["rho"], grid["n"], grid["seed"]
    ):
        jobs.append(
            {
                "label": method_label(spec),
                "policy": dict(spec),
                "scenario": exp.get("scenario") or {},
                "workload": str(workload),
                "rho": float(rho),
                "n": int(n),
                "seed": int(seed),
            }
        )
    return jobs


def run_job(job: dict[str, Any], out_dir: str) -> dict[str, Any]:
    """Run one job, write its time series and return its summary row."""
    dotted = {
        "workload.name": job["workload"],
        "workload.rho": job["rho"],
        "topology.n_servers": job["n"],
    }
    cfg = load_config(job["scenario"], dotted)
    ts, summary = run_episode(cfg, make_policy(job["policy"]), job["seed"])
    name = run_file_name(job["label"], job["workload"], job["rho"], job["n"], job["seed"])
    ts.to_csv(Path(out_dir) / name, index=False)
    ids = {
        "method": job["label"],
        "workload": job["workload"],
        "rho": job["rho"],
        "n": job["n"],
        "seed": job["seed"],
    }
    return {**ids, **summary}


def print_overview(summary: pd.DataFrame) -> None:
    """Mean over seeds of the main metrics, one line per method and scenario."""
    cols = ["resp_mean", "resp_p95", "lbf_mean", "blocking_rate", "msgs_per_task", "violations"]
    table = summary.groupby(ID_COLUMNS[:-1])[cols].mean()
    with pd.option_context(
        "display.width", 160, "display.max_columns", None, "display.float_format", "{:.3f}".format
    ):
        print(table)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run an experiment grid and write CSVs.")
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("--out", default=str(ROOT / "results"), help="results root")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--duration", type=float, help="override run.duration")
    parser.add_argument("--seeds", type=int, nargs="+", help="override the seed list")
    parser.add_argument(
        "--quick", action="store_true", help="smoke test: 300 s runs, first seed only"
    )
    args = parser.parse_args()

    exp = read_yaml(args.config)
    if args.quick:
        args.duration = args.duration or 300.0
        args.seeds = args.seeds or exp["grid"]["seed"][:1]
    if args.duration is not None:
        exp["scenario"] = deep_merge(
            exp.get("scenario") or {}, {"run": {"duration": args.duration}}
        )
    if args.seeds:
        exp["grid"]["seed"] = args.seeds

    out_dir = Path(args.out) / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = expand(exp)
    print(f"{exp['name']}: {len(jobs)} runs on {args.workers} workers -> {out_dir}")
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, str(out_dir)): job for job in jobs}
        for done, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            print(
                f"[{done}/{len(jobs)}] {row['method']} {row['workload']} rho={row['rho']:g} "
                f"n={row['n']} seed={row['seed']}: resp {row['resp_mean']:.2f} s, "
                f"LBF {row['lbf_mean']:.3f} ({row['wall_clock']:.1f} s)",
                flush=True,
            )
    summary = pd.DataFrame(rows).sort_values(ID_COLUMNS).reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"wrote {out_dir / 'summary.csv'} in {time.perf_counter() - started:.0f} s")
    print_overview(summary)


if __name__ == "__main__":
    main()
