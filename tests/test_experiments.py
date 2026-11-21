"""Tests for the experiment runner."""

import argparse

import pandas as pd
import pytest

from experiments.run import expand, run_file_name, run_job, select


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
    assert {"lbf", "util_mean", "throughput", "resp_mean", "msg_POLL"} <= set(ts.columns)
    assert ts["util_mean"].between(0, 1).all()


def test_filters_pick_one_cell():
    exp = {
        "name": "t",
        "methods": [{"name": "nearest"}, {"name": "evblb", "dt": 2.0}],
        "grid": {"workload": ["W1", "W2"], "rho": [0.7, 0.9], "n": [50], "seed": [100, 101]},
    }
    args = argparse.Namespace(methods=["evblb_dt2"], workloads=["W2"], rhos=[0.9], n=None)
    jobs = select(expand(exp), args)
    assert len(jobs) == 2 and {j["seed"] for j in jobs} == {100, 101}
    with pytest.raises(SystemExit):
        select(expand(exp), argparse.Namespace(methods=["nope"], workloads=None, rhos=None, n=None))
