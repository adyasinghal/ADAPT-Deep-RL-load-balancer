"""Tests for the statistics and formatting behind the results tables."""

import numpy as np
import pandas as pd
import pytest

from experiments.tables import ci95, fmt, paired, to_latex, to_markdown


def test_ci95_uses_the_t_distribution():
    mean, half = ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    assert half == pytest.approx(2.776445 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))
    assert ci95([7.0]) == (7.0, 0.0)
    assert np.isnan(ci95([np.nan])[0])


def test_fmt_scales_and_rounds():
    assert fmt([0.01, 0.03], digits=1, scale=100) == "2.0 ± 12.7"
    assert fmt([]) == "-"


def test_paired_differences_match_seeds():
    rows = []
    for seed, (a, b) in enumerate([(1.0, 3.0), (2.0, 5.0), (4.0, 5.0)]):
        rows += [
            {"workload": "W1", "rho": 0.7, "seed": seed, "method": "adapt", "x": a},
            {"workload": "W1", "rho": 0.7, "seed": seed, "method": "evblb_dt2", "x": b},
        ]
    table = paired(pd.DataFrame(rows), "adapt", "x")
    assert table.loc[0, "vs EVBLB (B2)"] == fmt([-2.0, -3.0, -1.0])


def test_markdown_and_latex_output():
    table = pd.DataFrame({"Method": ["ADAPT"], "Blocking (%)": ["1.0 ± 0.5"]})
    md = to_markdown(table)
    assert md.splitlines()[0] == "| Method | Blocking (%) |"
    tex = to_latex(table, "Caption with ρ", "x")
    assert r"\toprule" in tex and r"$\pm$" in tex and r"\%" in tex and r"$\rho$" in tex
