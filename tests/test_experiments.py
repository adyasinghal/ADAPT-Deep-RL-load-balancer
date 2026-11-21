"""Tests for the experiment runner."""

import pandas as pd

from experiments.run import expand, run_file_name, run_job


def test_grid_expansion_and_labels():
    exp = {
        "name": "t",
        "methods": [{"name": "nearest"}, {"name": "evblb", "dt": 0.5}, {"name": "perfect"}],
        "grid": {"workload": ["W1", "W2"], "rho": [0.5], "n": [10], "seed": [1, 2]},
    }
    jobs = expand(exp)
    assert len(jobs) == 3 * 2 * 2
    assert {j["label"] for j in jobs} == {"nearest", "evblb_dt0.5", "perfect"}


def test_run_job_writes_time_series(tmp_path):
    job = {
        "label": "evblb_dt1",
        "policy": {"name": "evblb", "dt": 1.0},
        "scenario": {"run": {"duration": 30.0, "arrival_cutoff": 5.0}},
        "workload": "W1",
        "rho": 0.5,
        "n": 10,
        "seed": 100,
    }
    row = run_job(job, str(tmp_path))
    assert row["method"] == "evblb_dt1" and row["violations"] == 0
    ts = pd.read_csv(tmp_path / run_file_name("evblb_dt1", "W1", 0.5, 10, 100))
    assert list(ts["t"]) == list(range(1, 31))
    assert {"lbf", "throughput", "resp_mean", "msg_POLL"} <= set(ts.columns)
