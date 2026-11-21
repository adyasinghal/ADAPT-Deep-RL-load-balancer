"""Every figure of the report, made only from the CSVs under ``results/``.

    python -m plots.make_plots

Figures go to ``results/figures/`` as PDF (for the report) and PNG. A figure whose experiment
has not been run is skipped with a message.

Colours: every method keeps one colour in every figure (``COLORS``). The order of that palette
was checked for colour-blind separation; ADAPT's variants share ADAPT's orange and differ by
line style, so no figure needs more than seven hues. All numbers in the figures are also in the
tables (``python -m experiments.tables``), which is the accessible way to read them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import ROOT
from experiments.labels import name, ordered
from experiments.run import run_file_name
from experiments.tables import ci95, load

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# categorical slots in their validated order, then the methods they belong to
COLORS = {
    "evblb_dt2": "#2a78d6",
    "adapt": "#eb6834",
    "perfect": "#1baf7a",
    "evblb_dt0.25": "#eda100",
    "ml_evblb": "#e87ba4",
    "topsis": "#008300",
    "nearest": "#6250d6",
}
# ADAPT variants: same hue as ADAPT, told apart by line style
VARIANT_STYLE = {"adapt": "-", "adapt_noWS": "--", "adapt_ep0": ":", "adapt_fixedDT": "-."}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

COLUMN = 3.5  # IEEE column width, inches
PAGE = 7.16  # IEEE two-column width


def color(label: str) -> str:
    """Colour of a method (ADAPT variants use ADAPT's)."""
    return COLORS.get(label, COLORS["adapt"] if label.startswith("adapt") else MUTED)


def style() -> None:
    """Quiet chart chrome: hairline solid grid, light axes, ink-coloured text."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    """Write ``results/figures/<stem>.pdf`` and ``.png``."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    # no creation date in the PDF, so regenerating an unchanged figure gives the same file
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}")


def series(experiment: str, label: str, workload: str, rho: float, n: int, seeds) -> list:
    """Time-series DataFrames of one method / scenario (missing files are skipped)."""
    frames = []
    for seed in seeds:
        path = RESULTS / experiment / run_file_name(label, workload, rho, n, seed)
        if path.exists():
            frames.append(pd.read_csv(path))
    return frames


def mean_curve(
    frames: list, column: str, smooth: int = 30, until: float = 60.0
) -> tuple[np.ndarray, ...]:
    """Mean over seeds of a rolling-mean-smoothed column, plus its 95% CI band.

    The last ``until`` seconds are dropped: arrivals stop there (makespan measurement), so
    every curve collapses and the tail says nothing about the method.
    """
    keep = frames[0]["t"] <= frames[0]["t"].iloc[-1] - until
    stacked = np.vstack([f[column].rolling(smooth, min_periods=1).mean()[keep] for f in frames])
    mean = stacked.mean(axis=0)
    half = np.array([ci95(stacked[:, k])[1] for k in range(stacked.shape[1])])
    return frames[0]["t"][keep].to_numpy(), mean, half


def end_label(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """Direct label at the end of a line, in secondary ink (never the series colour)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=INK_2,
        fontsize=7,
    )


def bars_with_ci(ax: plt.Axes, df: pd.DataFrame, metric: str, methods, scale=1.0) -> None:
    """One bar per method (mean over seeds) with a 95% CI whisker."""
    x = np.arange(len(methods))
    for k, label in enumerate(methods):
        mean, half = ci95(df[df["method"] == label][metric] * scale)
        ax.bar(k, mean, width=0.62, color=color(label), edgecolor=SURFACE, linewidth=1)
        ax.errorbar(k, mean, yerr=half, color=INK_2, linewidth=0.8, capsize=2)
    ax.set_xticks(x, [name(m) for m in methods], rotation=35, ha="right")
    ax.set_xlim(-0.6, len(methods) - 0.4)


def grouped_bars(
    df: pd.DataFrame, metric: str, ylabel: str, scale: float, stem: str, skip=("nearest",)
) -> None:
    """E3: for each rho a panel, grouped by workload, one bar per method."""
    methods = [m for m in ordered(df["method"]) if m not in skip]
    rhos = sorted(df["rho"].unique())
    workloads = sorted(df["workload"].unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(PAGE, 2.4), sharey=False)
    width = 0.8 / len(methods)
    for ax, rho in zip(np.atleast_1d(axes), rhos, strict=True):
        sub = df[df["rho"] == rho]
        for j, label in enumerate(methods):
            means, halves = [], []
            for w in workloads:
                m, h = ci95(sub[(sub["workload"] == w) & (sub["method"] == label)][metric] * scale)
                means.append(m)
                halves.append(h)
            xs = np.arange(len(workloads)) - 0.4 + width * (j + 0.5)
            ax.bar(
                xs,
                means,
                width=width,
                color=color(label),
                edgecolor=SURFACE,
                linewidth=0.8,
                label=name(label),
            )
            ax.errorbar(
                xs, means, yerr=halves, fmt="none", ecolor=INK_2, elinewidth=0.6, capsize=1.2
            )
        ax.set_xticks(np.arange(len(workloads)), workloads)
        ax.set_title(f"ρ = {rho:g}")
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.08))
    save(fig, stem)


# ---------------------------------------------------------------------------- figures


def f1_e1_lbf() -> bool:
    """F1: LBF over time, nearest-only vs EVBLB (E1, n = 50)."""
    df = load("e1")
    if df is None:
        return False
    seeds = sorted(df["seed"].unique())
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    curves = {
        label: series("e1", label, "W1", 0.7, 50, seeds) for label in ("nearest", "evblb_dt2")
    }
    if not all(curves.values()):
        plt.close(fig)
        return False  # only summary.csv is in the repo; rerun E1 for the time series
    for label, frames in curves.items():
        t, mean, half = mean_curve(frames, "lbf")
        ax.fill_between(t, mean - half, mean + half, color=color(label), alpha=0.1, linewidth=0)
        ax.plot(t, mean, color=color(label), label=name(label))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LBF (30 s mean)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", ncol=2)
    save(fig, "f1_e1_lbf")
    return True


def f2_e2_interval() -> bool:
    """F2: EVBLB's mean response time against its dispatch interval (E2), one panel per rho."""
    df = load("e2")
    if df is None:
        return False
    df = df.assign(dt=df["method"].str.replace("evblb_dt", "").astype(float))
    rhos = sorted(df["rho"].unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(COLUMN, 1.9))
    for ax, rho in zip(axes, rhos, strict=True):
        dts = sorted(df["dt"].unique())
        stats_ = [ci95(df[(df["rho"] == rho) & (df["dt"] == d)]["resp_mean"]) for d in dts]
        means = [m for m, _ in stats_]
        ax.errorbar(
            dts,
            means,
            yerr=[h for _, h in stats_],
            color=color("evblb_dt2"),
            marker="o",
            markersize=4,
            markeredgecolor=SURFACE,
            capsize=2,
            ecolor=INK_2,
            elinewidth=0.6,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(dts, [f"{d:g}" for d in dts])
        ax.minorticks_off()
        ax.set_title(f"ρ = {rho:g}")
        ax.set_xlabel("Dispatch interval ΔT (s)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean response time (s)")
    fig.tight_layout()
    save(fig, "f2_e2_interval")
    return True


def f3_e3_response() -> bool:
    """F3: mean response time per scenario and method (E3)."""
    df = load("e3")
    if df is None:
        return False
    grouped_bars(df, "resp_mean", "Mean response time (s)", 1.0, "f3_e3_response")
    return True


def f4_e3_blocking() -> bool:
    """F4: blocking rate per scenario and method (E3)."""
    df = load("e3")
    if df is None:
        return False
    grouped_bars(df, "blocking_rate", "Blocked tasks (%)", 100.0, "f4_e3_blocking")
    return True


def f5_e4_messages() -> bool:
    """F5: control messages per completed task over all E3 runs (E4)."""
    df = load("e3")
    if df is None:
        return False
    methods = [m for m in ordered(df["method"]) if m != "nearest"]  # nearest sends none
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    bars_with_ci(ax, df, "msgs_per_task", methods)
    ax.set_ylabel("Messages per task")
    ax.set_ylim(bottom=0)
    save(fig, "f5_e4_messages")
    return True


def f6_e5_cold_start() -> bool:
    """F6: LBF over the first 2000 s (E5)."""
    df = load("e5")
    if df is None:
        return False
    seeds = sorted(df["seed"].unique())
    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    curves = {label: series("e5", label, "W1", 0.7, 50, seeds) for label in ordered(df["method"])}
    if not all(curves.values()):
        plt.close(fig)
        return False  # rerun E5 for the time series
    for label, frames in curves.items():
        t, mean, _ = mean_curve(frames, "lbf", smooth=60)
        ax.plot(
            t, mean, color=color(label), linestyle=VARIANT_STYLE.get(label, "-"), label=name(label)
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LBF (60 s mean)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.22))
    save(fig, "f6_e5_cold_start")
    return True


def f7_training() -> bool:
    """F7: greedy evaluation reward after each training episode, with and without warm start."""
    curves = {}
    for label, file in (("adapt", "adapt.csv"), ("adapt_noWS", "adapt_noWS.csv")):
        path = RESULTS / "training" / file
        if path.exists():
            log = pd.read_csv(path)
            curves[label] = log[log["phase"] == "train"]
    if not curves:
        return False
    fig, ax = plt.subplots(figsize=(COLUMN, 2.1))
    for label, log in curves.items():
        ax.plot(
            log["episode"] + 1,
            log["eval_reward"],
            color=color(label),
            linestyle=VARIANT_STYLE[label],
            label=name(label),
        )
        last = log.iloc[-1]
        end_label(ax, last["episode"] + 1, last["eval_reward"], name(label))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 25.5)
    ax.set_xlabel("Training episode (600 s each)")
    ax.set_ylabel("Eval reward per server-step")
    ax.legend(loc="lower right", handlelength=3)
    save(fig, "f7_training")
    return True


def dispatch_profile(frames: list, window: int = 10):
    """Mean time between dispatches per server, binned by mean utilisation.

    Each run is cut into ``window``-second blocks (so EVBLB's fixed 2 s grid averages out);
    the first and last minute are dropped (warm-up, and the end when arrivals stop). A bin
    needs at least 20 blocks to be shown.
    """
    util, rate = [], []
    for f in frames:
        f = f[(f["t"] > 60) & (f["t"] <= f["t"].iloc[-1] - 60)]
        blocks = np.arange(len(f)) // window
        grouped = f.assign(block=blocks).groupby("block")
        rounds = grouped["dispatch_rounds"].agg(lambda r: r.iloc[-1] - r.iloc[0])
        util.append(grouped["util_mean"].mean().to_numpy())
        rate.append(rounds.to_numpy() / (50.0 * (window - 1)))  # per server per second
    util, rate = np.concatenate(util), np.concatenate(rate)
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(util, bins) - 1
    centres, interval = [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() >= 20 and rate[sel].mean() > 0:
            centres.append((bins[b] + bins[b + 1]) / 2)
            interval.append(1.0 / rate[sel].mean())
    return np.array(centres), np.array(interval)


def f8_effective_interval() -> bool:
    """F8: mean time between dispatches against the load at that moment (E3, W2 and W3).

    W2 (tidal) and W3 (flash crowds) are the workloads whose load changes during a run.
    """
    df = load("e3")
    if df is None:
        return False
    seeds = sorted(df["seed"].unique())
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    drawn = False
    for label in ("evblb_dt2", "ml_evblb", "adapt"):
        frames = [
            f
            for workload in ("W2", "W3")
            for rho in sorted(df["rho"].unique())
            for f in series("e3", label, workload, rho, 50, seeds)
            if "util_mean" in f
        ]
        if not frames:
            continue
        centres, interval = dispatch_profile(frames)
        ax.plot(
            centres,
            interval,
            color=color(label),
            marker="o",
            markersize=3,
            markeredgecolor=SURFACE,
            label=name(label),
        )
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_xlabel("Mean server utilisation (10 s blocks)")
    ax.set_ylabel("Time between dispatches (s)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    save(fig, "f8_effective_interval")
    return True


def f9_e9_costs() -> bool:
    """F9: response time with the added costs on (E3) and off (E9), W1, rho 0.7."""
    on, off = load("e3"), load("e9")
    if on is None or off is None:
        return False
    on = on[(on["workload"] == "W1") & (on["rho"] == 0.7)]
    methods = [m for m in ordered(off["method"]) if m != "nearest"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE / 1.6, 2.3), sharey=True)
    for ax, (df, title) in zip(axes, ((on, "Costs on (E3)"), (off, "Costs off (E9)")), strict=True):
        bars_with_ci(ax, df, "resp_mean", methods)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean response time (s)")
    fig.tight_layout()
    save(fig, "f9_e9_costs")
    return True


FIGURES_ALL: dict[str, Callable[[], bool]] = {
    "F1": f1_e1_lbf,
    "F2": f2_e2_interval,
    "F3": f3_e3_response,
    "F4": f4_e3_blocking,
    "F5": f5_e4_messages,
    "F6": f6_e5_cold_start,
    "F7": f7_training,
    "F8": f8_effective_interval,
    "F9": f9_e9_costs,
}


def main() -> None:
    """Make every figure whose data exists."""
    parser = argparse.ArgumentParser(description="Make the report figures from results/.")
    parser.add_argument("--only", nargs="+", help="e.g. --only F3 F5")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    style()
    for key, make in FIGURES_ALL.items():
        if args.only and key not in args.only:
            continue
        if not make():
            print(f"skip {key}: its experiment (or its per-run time series) is missing")
    print(f"figures in {Path(FIGURES)}")


if __name__ == "__main__":
    main()
