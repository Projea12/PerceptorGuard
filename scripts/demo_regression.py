#!/usr/bin/env python3
"""
Planted-regression demo: shows the gate going RED → GREEN.

This demo does NOT re-run YOLO inference — it synthesises a degraded metrics
directory by zeroing out the sports-ball AP (simulating what happens when the
IoU match threshold is cranked from 0.5 → 0.9, making the two sub-threshold
sports-ball TPs fall out).  This mirrors the real CI scenario where a commit
that changes EvalRunner defaults degrades measurable signal.

Usage:
  python scripts/demo_regression.py
  python scripts/demo_regression.py --baseline artifacts/baseline

To simulate the real CI scenario (re-runs inference at iou=0.9):
  python scripts/run_eval.py --iou 0.9 --out artifacts/eval_degraded
  python scripts/run_gate.py --current artifacts/eval_degraded
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from gates.gate_runner import run_gate
from gates.thresholds import GateThresholds

_W = 74


def _patch_class_csv(src_path: Path, dst_path: Path) -> None:
    """Zero sports-ball AP to simulate iou_threshold=0.9 regression."""
    df = pd.read_csv(src_path)
    if "ap" in df.columns and "class" in df.columns:
        df.loc[df["class"] == "sports ball", "ap"] = 0.0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)


def _patch_overall_csv(src_path: Path, dst_path: Path) -> None:
    """Zero overall mAP to match degraded class AP."""
    df = pd.read_csv(src_path)
    if "map" in df.columns:
        df["map"] = 0.0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Planted-regression gate demo")
    ap.add_argument("--baseline", type=Path, default=Path("artifacts/baseline"),
                    help="Baseline directory (default: artifacts/baseline)")
    args = ap.parse_args()

    if not args.baseline.exists():
        sys.exit(
            f"ERROR: {args.baseline} not found.\n"
            "Run:  python scripts/save_baseline.py"
        )

    thresholds = GateThresholds.from_yaml()

    # ── STEP 1: Gate on un-degraded current eval (should PASS) ────────────────
    current_good = Path("artifacts/eval")
    if not current_good.exists() or not (current_good / "metrics_overall.csv").exists():
        sys.exit("ERROR: artifacts/eval not found — run scripts/run_eval.py first")

    print("═" * _W)
    print("  DEMO STEP 1 — Gate on current (good) metrics")
    print("═" * _W)
    ok = run_gate(args.baseline, current_good, thresholds)
    assert ok, "Unexpected: gate should PASS on unmodified eval metrics"

    # ── STEP 2: Synthesise degraded metrics (sports-ball AP zeroed) ────────────
    print("\n")
    print("═" * _W)
    print("  DEMO STEP 2 — Plant regression: zero sports-ball AP")
    print("  (simulates iou_threshold cranked to 0.9 — kills the two sub-")
    print("   threshold sports-ball TPs that give us our only AP signal)")
    print("═" * _W)

    with tempfile.TemporaryDirectory(prefix="pg_degraded_") as tmp:
        degraded = Path(tmp)
        # copy all CSVs, then patch the two that carry the AP signal
        for f in args.baseline.glob("metrics_*.csv"):
            shutil.copy2(f, degraded / f.name)
        _patch_class_csv(
            current_good / "metrics_class.csv",
            degraded / "metrics_class.csv",
        )
        _patch_overall_csv(
            current_good / "metrics_overall.csv",
            degraded / "metrics_overall.csv",
        )

        failed = not run_gate(args.baseline, degraded, thresholds)
        if not failed:
            print("\n  WARNING: gate did not catch the planted regression.")
            print("  Check that map_slack in configs/gate_thresholds.yml is tight enough.")
        else:
            print("\n  Gate correctly identified the regression. ✓")

    # ── STEP 3: Restore — gate should PASS again ──────────────────────────────
    print("\n")
    print("═" * _W)
    print("  DEMO STEP 3 — Restore: gate returns GREEN")
    print("═" * _W)
    ok2 = run_gate(args.baseline, current_good, thresholds)
    assert ok2

    print("\n  Demo complete.")
    print(f"  Result: {'PASS → FAIL → PASS  ✓  (gate behaves correctly)' if failed else 'gate did not fire on degraded metrics — tighten map_slack'}")


if __name__ == "__main__":
    main()
