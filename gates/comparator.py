"""
Compare current eval metrics against a stored baseline and return
a list of RegressionResult objects — one per (slice, metric) pair checked.

Design: every check is explicit and named so the gate report can say exactly
which slice regressed, by how much, and what the floor was.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gates.thresholds import GateThresholds


@dataclass
class RegressionResult:
    slice_name: str     # e.g. "class:sports_ball", "profile:dark", "overall"
    metric: str         # e.g. "ap", "recall", "mAP", "fp_count"
    baseline: float
    current: float
    floor: float
    passed: bool

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.slice_name} / {self.metric}: "
            f"baseline={self.baseline:.4f}  current={self.current:.4f}  "
            f"delta={self.delta:+.4f}  floor={self.floor:.4f}"
        )


def _safe(v) -> float:
    """Return 0.0 for NaN/None."""
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _check(
    slice_name: str,
    metric: str,
    baseline: float,
    current: float,
    floor: float,
) -> RegressionResult:
    return RegressionResult(
        slice_name=slice_name,
        metric=metric,
        baseline=baseline,
        current=current,
        floor=floor,
        passed=current >= floor,
    )


def compare_metrics(
    baseline_dir: Path,
    current_dir: Path,
    thresholds: GateThresholds,
) -> list[RegressionResult]:
    """
    Load CSVs from *baseline_dir* and *current_dir*, run all gate checks,
    return the full result list (failures and passes).
    """
    results: list[RegressionResult] = []

    # ── Overall ────────────────────────────────────────────────────────────────
    b_ov = pd.read_csv(baseline_dir / "metrics_overall.csv").iloc[0]
    c_ov = pd.read_csv(current_dir  / "metrics_overall.csv").iloc[0]

    for metric, slack, hard_floor in [
        ("map",       thresholds.map_slack,       thresholds.hard_floor_map),
        ("recall",    thresholds.recall_slack,     thresholds.hard_floor_recall),
        ("precision", thresholds.precision_slack,  -1.0),
    ]:
        b_val = _safe(b_ov.get(metric, 0))
        c_val = _safe(c_ov.get(metric, 0))
        floor = max(thresholds.floor(b_val, slack),
                    hard_floor if hard_floor >= 0 else -999.0)
        results.append(_check("overall", metric, b_val, c_val, floor))

    # FP count — directional: more FPs = regression
    b_fp = _safe(b_ov.get("fp", 0))
    c_fp = _safe(c_ov.get("fp", 0))
    fp_ceil = b_fp * (1 + thresholds.fp_slack_frac)
    results.append(RegressionResult(
        slice_name="overall", metric="fp_count",
        baseline=b_fp, current=c_fp,
        floor=fp_ceil,   # reuse "floor" field as ceiling for FP
        passed=c_fp <= fp_ceil,
    ))

    # ── Per-class AP ───────────────────────────────────────────────────────────
    b_cls = pd.read_csv(baseline_dir / "metrics_class.csv").set_index("class")
    c_cls_raw = pd.read_csv(current_dir / "metrics_class.csv")
    # guard against missing index column name differences
    if "class" in c_cls_raw.columns:
        c_cls = c_cls_raw.set_index("class")
    else:
        c_cls = c_cls_raw.set_index(c_cls_raw.columns[0])

    for cls in b_cls.index:
        if cls not in c_cls.index:
            continue
        b_ap = _safe(b_cls.loc[cls, "ap"])
        c_ap = _safe(c_cls.loc[cls, "ap"])
        floor = thresholds.floor(b_ap, thresholds.map_slack)
        results.append(_check(f"class:{cls}", "ap", b_ap, c_ap, floor))

        b_rec = _safe(b_cls.loc[cls, "recall"])
        c_rec = _safe(c_cls.loc[cls, "recall"])
        floor_r = thresholds.floor(b_rec, thresholds.recall_slack)
        results.append(_check(f"class:{cls}", "recall", b_rec, c_rec, floor_r))

    # ── Per-profile recall ─────────────────────────────────────────────────────
    _compare_slice_csv(
        baseline_dir / "metrics_profile.csv",
        current_dir  / "metrics_profile.csv",
        prefix="profile",
        thresholds=thresholds,
        results=results,
    )

    # ── Per-distance recall ────────────────────────────────────────────────────
    _compare_slice_csv(
        baseline_dir / "metrics_distance_bin.csv",
        current_dir  / "metrics_distance_bin.csv",
        prefix="distance",
        thresholds=thresholds,
        results=results,
    )

    # ── Per-lighting recall ────────────────────────────────────────────────────
    _compare_slice_csv(
        baseline_dir / "metrics_lighting_bin.csv",
        current_dir  / "metrics_lighting_bin.csv",
        prefix="lighting",
        thresholds=thresholds,
        results=results,
    )

    return results


def _compare_slice_csv(
    b_path: Path,
    c_path: Path,
    prefix: str,
    thresholds: GateThresholds,
    results: list[RegressionResult],
) -> None:
    if not b_path.exists() or not c_path.exists():
        return
    b_df = pd.read_csv(b_path).set_index(pd.read_csv(b_path).columns[0])
    c_df = pd.read_csv(c_path).set_index(pd.read_csv(c_path).columns[0])
    for idx in b_df.index:
        if idx not in c_df.index:
            continue
        b_rec = _safe(b_df.loc[idx, "recall"])
        c_rec = _safe(c_df.loc[idx, "recall"])
        floor = thresholds.floor(b_rec, thresholds.recall_slack)
        results.append(_check(f"{prefix}:{idx}", "recall", b_rec, c_rec, floor))

        b_fp = _safe(b_df.loc[idx, "fp"])
        c_fp = _safe(c_df.loc[idx, "fp"])
        fp_ceil = b_fp * (1 + thresholds.fp_slack_frac)
        results.append(RegressionResult(
            slice_name=f"{prefix}:{idx}", metric="fp_count",
            baseline=b_fp, current=c_fp,
            floor=fp_ceil,
            passed=c_fp <= fp_ceil,
        ))
