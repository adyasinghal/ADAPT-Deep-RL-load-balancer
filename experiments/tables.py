"""Results tables from the experiments' summary CSVs.

    python -m experiments.tables

Every cell is the mean over the test seeds with a 95% confidence interval (t-distribution),
written as ``mean ± half-width``. Each table is saved as markdown and as a LaTeX ``booktabs``
table under ``results/tables/``; tables whose experiment has not been run are skipped.

The ``e3_paired`` table compares ADAPT with each baseline seed by seed (same topology and
traffic), which removes the large between-topology spread from the comparison.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core.config import ROOT
from experiments.labels import name, ordered, scenario

RESULTS = ROOT / "results"


def ci95(values) -> tuple[float, float]:
    """Mean and half-width of the 95% t-interval (half-width 0 for a single value)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), 0.0
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def fmt(values, digits: int = 2, scale: float = 1.0) -> str:
    """'mean ± half' of ``values * scale``."""
    mean, half = ci95(np.asarray(values, dtype=float) * scale)
    if np.isnan(mean):
        return "-"
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def load(experiment: str) -> pd.DataFrame | None:
    """summary.csv of an experiment, or None if it hasn't been run."""
    path = RESULTS / experiment / "summary.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------- builders


def by_method(df: pd.DataFrame, columns: dict[str, tuple[str, int, float]]) -> pd.DataFrame:
    """One row per method; ``columns`` maps header -> (metric, digits, scale)."""
    rows = []
    for label in ordered(df["method"]):
        sub = df[df["method"] == label]
        row = {"Method": name(label)}
        for header, (metric, digits, scale) in columns.items():
            row[header] = fmt(sub[metric], digits, scale)
        rows.append(row)
    return pd.DataFrame(rows)


def method_by_scenario(
    df: pd.DataFrame, metric: str, digits: int = 2, scale: float = 1.0, key: str = "scenario"
) -> pd.DataFrame:
    """Methods as columns, one row per scenario (workload, rho) or per ``key``."""
    df = df.copy()
    if key == "scenario":
        df["scenario"] = [scenario(w, r) for w, r in zip(df["workload"], df["rho"], strict=True)]
    rows = []
    for value in dict.fromkeys(df.sort_values(["workload", "rho", "n"])[key]):
        sub = df[df[key] == value]
        row = {key.capitalize() if key != "n" else "n": value}
        for label in ordered(df["method"]):
            row[name(label)] = fmt(sub[sub["method"] == label][metric], digits, scale)
        rows.append(row)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, reference: str, metric: str, scale: float = 1.0) -> pd.DataFrame:
    """``reference`` minus every other method, matched by seed, per scenario."""
    wide = df.pivot_table(index=["workload", "rho", "seed"], columns="method", values=metric)
    rows = []
    for (workload, rho), block in wide.groupby(level=[0, 1]):
        row = {"Scenario": scenario(workload, rho)}
        for label in ordered(c for c in block.columns if c != reference):
            row[f"vs {name(label)}"] = fmt((block[reference] - block[label]) * scale)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- the tables


def table_e1() -> pd.DataFrame | None:
    """E1 per network size: response time, blocking and LBF of B1 and B2."""
    df = load("e1")
    if df is None:
        return None
    out = []
    for n_servers in sorted(df["n"].unique()):
        t = by_method(
            df[df["n"] == n_servers],
            {
                "Response (s)": ("resp_mean", 2, 1),
                "Blocking (%)": ("blocking_rate", 2, 100),
                "LBF": ("lbf_mean", 3, 1),
            },
        )
        t.insert(0, "n", n_servers)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def table_e2() -> pd.DataFrame | None:
    """E2 per load and dispatch interval."""
    df = load("e2")
    if df is None:
        return None
    df = df.assign(dt=df["method"].str.replace("evblb_dt", "").astype(float))
    rows = []
    for (rho, dt), sub in df.groupby(["rho", "dt"]):
        rows.append(
            {
                "ρ": rho,
                "Interval (s)": f"{dt:g}",
                "Response (s)": fmt(sub["resp_mean"]),
                "p95 (s)": fmt(sub["resp_p95"], 1),
                "Msgs / task": fmt(sub["msgs_per_task"]),
                "LBF": fmt(sub["lbf_mean"], 3),
            }
        )
    return pd.DataFrame(rows)


def e3_metric(metric: str, digits: int, scale: float = 1.0):
    """Builder for one E3 metric (methods as columns, scenarios as rows)."""

    def build() -> pd.DataFrame | None:
        df = load("e3")
        return None if df is None else method_by_scenario(df, metric, digits, scale)

    return build


def table_e3_paired() -> pd.DataFrame | None:
    """ADAPT minus each E3 baseline, per scenario, for response time and blocking."""
    df = load("e3")
    if df is None:
        return None
    resp = paired(df, "adapt", "resp_mean")
    resp.insert(1, "Metric", "Response (s)")
    block = paired(df, "adapt", "blocking_rate", scale=100)
    block.insert(1, "Metric", "Blocking (pp)")
    return pd.concat([resp, block], ignore_index=True)


def table_e4() -> pd.DataFrame | None:
    """E4 overhead per method over all E3 runs."""
    df = load("e3")
    if df is None:
        return None
    df = df.assign(
        polls_per_task=(df["msg_POLL"] + df["msg_POLL_REPLY"]) / df["completed"],
        latency_ms=df["decision_latency"] * 1000,
    )
    return by_method(
        df,
        {
            "Msgs / task": ("msgs_per_task", 2, 1),
            "Poll msgs / task": ("polls_per_task", 2, 1),
            "Rounds / server / s": ("rounds_per_server_s", 2, 1),
            "Decision latency (ms)": ("latency_ms", 1, 1),
        },
    )


def table_e5() -> pd.DataFrame | None:
    """E5 cold-start metrics per method."""
    df = load("e5")
    if df is None:
        return None
    return by_method(
        df,
        {
            "Time to balance (s)": ("time_to_balance", 0, 1),
            "LBF (2nd half)": ("lbf_mean", 3, 1),
            "Response (s)": ("resp_mean", 2, 1),
            "Blocking (%)": ("blocking_rate", 2, 100),
        },
    )


def table_e6() -> pd.DataFrame | None:
    """E6 ablation per workload."""
    df = load("e6")
    if df is None:
        return None
    out = []
    for workload in sorted(df["workload"].unique()):
        t = by_method(
            df[df["workload"] == workload],
            {
                "Response (s)": ("resp_mean", 2, 1),
                "Blocking (%)": ("blocking_rate", 2, 100),
                "Msgs / task": ("msgs_per_task", 2, 1),
            },
        )
        t.insert(0, "Workload", workload)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def table_e7() -> pd.DataFrame | None:
    """E7 response time per network size."""
    df = load("e7")
    return None if df is None else method_by_scenario(df, "resp_mean", 2, key="n")


def table_e8() -> pd.DataFrame | None:
    """E8 robustness to crashes and message loss."""
    df = load("e8")
    if df is None:
        return None
    return by_method(
        df,
        {
            "Response (s)": ("resp_mean", 2, 1),
            "Blocking (%)": ("blocking_rate", 2, 100),
            "Lost in crash": ("lost_in_crash", 0, 1),
            "Re-runs": ("reruns_after_crash", 1, 1),
            "Violations": ("violations", 0, 1),
        },
    )


def table_e9() -> pd.DataFrame | None:
    """E9 response time and messages with the added costs on and off."""
    on, off = load("e3"), load("e9")
    if on is None or off is None:
        return None
    on = on[(on["workload"] == "W1") & (on["rho"] == 0.7)]
    rows = []
    for label in ordered(off["method"]):
        a, b = on[on["method"] == label], off[off["method"] == label]
        rows.append(
            {
                "Method": name(label),
                "Response, costs on (s)": fmt(a["resp_mean"]),
                "Response, costs off (s)": fmt(b["resp_mean"]),
                "Msgs / task, on": fmt(a["msgs_per_task"]),
                "Msgs / task, off": fmt(b["msgs_per_task"]),
            }
        )
    return pd.DataFrame(rows)


TABLES: dict[str, tuple[Callable[[], pd.DataFrame | None], str]] = {
    "e1": (table_e1, "E1: nearest-only vs EVBLB, W1, ρ = 0.7"),
    "e2": (table_e2, "E2: EVBLB dispatch interval trade-off, W1, n = 50"),
    "e3_response": (e3_metric("resp_mean", 2), "E3: mean response time (s), n = 50"),
    "e3_p95": (e3_metric("resp_p95", 1), "E3: 95th-percentile response time (s), n = 50"),
    "e3_blocking": (e3_metric("blocking_rate", 2, 100), "E3: blocking rate (%), n = 50"),
    "e3_lbf": (e3_metric("lbf_mean", 3), "E3: LBF, second half of the run, n = 50"),
    "e3_throughput": (e3_metric("throughput", 1), "E3: throughput (tasks/s), n = 50"),
    "e3_paired": (table_e3_paired, "E3: ADAPT minus each baseline, paired by seed"),
    "e4": (table_e4, "E4: control overhead, all 40 E3 runs per method"),
    "e5": (table_e5, "E5: cold start, first 2000 s of W1, ρ = 0.7"),
    "e6": (table_e6, "E6: ablation, ρ = 0.7"),
    "e7": (table_e7, "E7: mean response time (s) for other network sizes, W1, ρ = 0.7"),
    "e8": (table_e8, "E8: 10% of servers crash at 1800 s, 2% message loss"),
    "e9": (table_e9, "E9: W1, ρ = 0.7 with the added costs on and off"),
}


def to_markdown(table: pd.DataFrame) -> str:
    """GitHub-style markdown table (avoids pandas' optional ``tabulate`` dependency)."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join(" --- " for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """booktabs LaTeX for ``table`` (wide tables should go in a ``table*``)."""

    def cell(value) -> str:
        text = str(value).replace("±", r"$\pm$").replace("ρ", r"$\rho$")
        return text.replace("%", r"\%").replace("_", r"\_")

    spec = "l" * len(table.columns)
    lines = [
        r"\begin{table*}[t]" if len(table.columns) > 5 else r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{{cell(caption)}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(cell(c) for c in table.columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in table.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines.append(r"\end{table*}" if len(table.columns) > 5 else r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write every table whose experiment has been run."""
    parser = argparse.ArgumentParser(description="Make the results tables.")
    parser.add_argument("--out", default=str(RESULTS / "tables"))
    parser.add_argument("--quiet", action="store_true", help="don't print the tables")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key, (build, caption) in TABLES.items():
        table = build()
        if table is None:
            print(f"skip {key}: experiment not run yet")
            continue
        markdown = f"**{caption}** (mean ± 95% CI over runs)\n\n" + to_markdown(table)
        (out / f"{key}.md").write_text(markdown + "\n")
        (out / f"{key}.tex").write_text(to_latex(table, caption, key))
        if not args.quiet:
            print(markdown, end="\n\n")
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
