"""
Unit tests for the regression gate subsystem.

All tests are hermetic — they write temp CSVs and do not touch artifacts/.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gates.thresholds import GateThresholds
from gates.comparator import compare_metrics, RegressionResult
from gates.gate_runner import run_gate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _overall(tp=0, fp=27, fn=355, precision=0.0, recall=0.0, f1=float("nan"),
             n_scenes=100, map=0.001):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "n_scenes": n_scenes, "map": map}


def _cls_rows():
    return [
        {"class": "bottle",     "tp": 0, "fp": 0, "fn": 44, "precision": float("nan"),
         "recall": 0.0, "f1": float("nan"), "n_scenes": 40, "ap": 0.0, "tier": "easy"},
        {"class": "sports ball", "tp": 0, "fp": 2, "fn": 22, "precision": 0.0,
         "recall": 0.0, "f1": float("nan"), "n_scenes": 31, "ap": 0.009, "tier": "easy"},
    ]


def _profile_rows():
    return [
        {"profile": "baseline", "tp": 0, "fp": 4, "fn": 60, "recall": 0.0,
         "precision": float("nan"), "f1": float("nan"), "n_scenes": 17},
        {"profile": "dark",     "tp": 0, "fp": 5, "fn": 38, "recall": 0.0,
         "precision": float("nan"), "f1": float("nan"), "n_scenes": 17},
    ]


def _write_baseline(d: Path, overall: dict, cls_rows: list, profile_rows: list) -> None:
    pd.DataFrame([overall]).to_csv(d / "metrics_overall.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(d / "metrics_class.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(d / "metrics_profile.csv", index=False)


def _tmpdir():
    return tempfile.mkdtemp()


# ── GateThresholds ────────────────────────────────────────────────────────────

def test_thresholds_defaults():
    t = GateThresholds()
    assert t.recall_slack == pytest.approx(0.02)
    assert t.map_slack == pytest.approx(0.005)

def test_thresholds_floor():
    t = GateThresholds(map_slack=0.005)
    assert t.floor(0.009, t.map_slack) == pytest.approx(0.004)

def test_thresholds_from_yaml():
    t = GateThresholds.from_yaml()  # reads configs/gate_thresholds.yml
    assert 0.0 < t.map_slack < 0.1
    assert 0.0 < t.recall_slack < 0.5


# ── compare_metrics: identity (baseline == current) ───────────────────────────

def test_identity_all_pass():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(), _cls_rows(), _profile_rows())
        t = GateThresholds()
        results = compare_metrics(base, cur, t)
        failures = [r for r in results if not r.passed]
        assert failures == [], f"Expected no failures, got: {failures}"


# ── compare_metrics: sports-ball AP regression ────────────────────────────────

def test_map_regression_detected():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(), _cls_rows(), _profile_rows())

        # degrade sports ball AP to 0
        degraded_cls = [
            {**row, "ap": 0.0} if row["class"] == "sports ball" else row
            for row in _cls_rows()
        ]
        _write_baseline(cur, {**_overall(), "map": 0.0}, degraded_cls, _profile_rows())

        results = compare_metrics(base, cur, GateThresholds())
        failures = [r for r in results if not r.passed]
        names = {r.slice_name for r in failures}
        assert "class:sports ball" in names
        assert any(r.metric == "ap" for r in failures if r.slice_name == "class:sports ball")


def test_overall_map_regression_detected():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(map=0.01), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(map=0.0),  _cls_rows(), _profile_rows())

        results = compare_metrics(base, cur, GateThresholds())
        failures = [r for r in results if not r.passed]
        assert any(r.slice_name == "overall" and r.metric == "map" for r in failures)


# ── compare_metrics: FP count regression ─────────────────────────────────────

def test_fp_spike_detected():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(fp=10), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(fp=50), _cls_rows(), _profile_rows())  # +400%

        results = compare_metrics(base, cur, GateThresholds(fp_slack_frac=0.30))
        failures = [r for r in results if not r.passed]
        assert any(r.slice_name == "overall" and r.metric == "fp_count" for r in failures)


def test_fp_within_slack_passes():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(fp=10), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(fp=12), _cls_rows(), _profile_rows())  # +20%

        results = compare_metrics(base, cur, GateThresholds(fp_slack_frac=0.30))
        fp_results = [r for r in results
                      if r.slice_name == "overall" and r.metric == "fp_count"]
        assert all(r.passed for r in fp_results)


# ── compare_metrics: recall regression ───────────────────────────────────────

def test_recall_regression_detected():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(recall=0.5), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(recall=0.4), _cls_rows(), _profile_rows())

        t = GateThresholds(recall_slack=0.02)  # floor = 0.48
        results = compare_metrics(base, cur, t)
        failures = [r for r in results if not r.passed]
        assert any(r.slice_name == "overall" and r.metric == "recall" for r in failures)


def test_recall_within_slack_passes():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(recall=0.5), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(recall=0.49), _cls_rows(), _profile_rows())

        t = GateThresholds(recall_slack=0.02)  # floor = 0.48; 0.49 >= 0.48 → pass
        results = compare_metrics(base, cur, t)
        recall_results = [r for r in results
                          if r.slice_name == "overall" and r.metric == "recall"]
        assert all(r.passed for r in recall_results)


# ── gate_runner ───────────────────────────────────────────────────────────────

def test_gate_runner_returns_true_on_pass():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(), _cls_rows(), _profile_rows())
        assert run_gate(base, cur, GateThresholds()) is True


def test_gate_runner_returns_false_on_regression():
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        _write_baseline(base, _overall(map=0.01), _cls_rows(), _profile_rows())
        _write_baseline(cur,  _overall(map=0.0),  _cls_rows(), _profile_rows())
        assert run_gate(base, cur, GateThresholds()) is False


def test_gate_runner_names_regressed_slice(capsys):
    with tempfile.TemporaryDirectory() as base_d, \
         tempfile.TemporaryDirectory() as cur_d:
        base, cur = Path(base_d), Path(cur_d)
        degraded_cls = [
            {**row, "ap": 0.0} if row["class"] == "sports ball" else row
            for row in _cls_rows()
        ]
        _write_baseline(base, _overall(), _cls_rows(), _profile_rows())
        _write_baseline(cur, _overall(map=0.0), degraded_cls, _profile_rows())
        run_gate(base, cur, GateThresholds())
        out = capsys.readouterr().out
        assert "sports ball" in out
        assert "FAILED" in out


# ── RegressionResult helpers ──────────────────────────────────────────────────

def test_regression_result_delta():
    r = RegressionResult("overall", "map", baseline=0.01, current=0.005,
                         floor=0.005, passed=True)
    assert r.delta == pytest.approx(-0.005)

def test_regression_result_str_contains_metric():
    r = RegressionResult("class:cup", "ap", baseline=0.5, current=0.4,
                         floor=0.495, passed=False)
    s = str(r)
    assert "FAIL" in s
    assert "class:cup" in s
    assert "ap" in s
