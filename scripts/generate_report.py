#!/usr/bin/env python3
"""
Generate a per-run HTML + Markdown eval report.

Minimal (eval + report only):
  python scripts/generate_report.py --dataset artifacts/dataset

Full pipeline (with baseline gate + triage + W&B):
  python scripts/generate_report.py \
    --dataset   artifacts/dataset \
    --eval      artifacts/eval \
    --baseline  artifacts/baseline \
    --triage    artifacts/triage \
    --out       artifacts/report \
    --tracker   wandb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from metrics.engine import overall_metrics, sliced_tables
from reports.renderer import render
from reports.tracker import log_run


def main() -> None:
    ap = argparse.ArgumentParser(description="PerceptorGuard report generator")
    ap.add_argument("--dataset",  type=Path, default=Path("artifacts/dataset"),
                    help="Dataset dir (for annotated failure images)")
    ap.add_argument("--eval",     type=Path, default=Path("artifacts/eval"),
                    help="Eval output dir (default: artifacts/eval)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Baseline dir for gate comparison (optional)")
    ap.add_argument("--triage",   type=Path, default=None,
                    help="Triage output dir containing failures_classified.csv")
    ap.add_argument("--out",      type=Path, default=Path("artifacts/report"),
                    help="Output dir for HTML/MD report (default: artifacts/report)")
    ap.add_argument("--model",    default="yolov8n.pt",
                    help="Model name for report header")
    ap.add_argument("--iou",      type=float, default=0.5)
    ap.add_argument("--formats",  default="html,md",
                    help="Comma-separated: html,md (default: both)")
    ap.add_argument("--tracker",  default="none",
                    choices=["none", "wandb", "mlflow"],
                    help="Experiment tracker (default: none)")
    ap.add_argument("--no-images", action="store_true",
                    help="Skip annotated failure image generation")
    args = ap.parse_args()

    matches_path = args.eval / "matches.csv"
    if not matches_path.exists():
        sys.exit(f"ERROR: {matches_path} not found — run scripts/run_eval.py first")

    print(f"  Loading matches from {matches_path}…")
    df = pd.read_csv(matches_path)
    overall = overall_metrics(df)
    tables = sliced_tables(df)

    # ── Gate results ──────────────────────────────────────────────────────────
    gate_results = None
    gate_passed = None
    if args.baseline and args.baseline.exists():
        from gates.thresholds import GateThresholds
        from gates.comparator import compare_metrics
        thresholds = GateThresholds.from_yaml()
        gate_results = compare_metrics(args.baseline, args.eval, thresholds)
        gate_passed = all(r.passed for r in gate_results)
        status = "PASSED" if gate_passed else "FAILED"
        print(f"  Gate: {status} ({sum(not r.passed for r in gate_results)} regression(s))")

    # ── Failure triage ────────────────────────────────────────────────────────
    failure_df = None
    triage_path = None
    if args.triage:
        triage_path = args.triage / "failures_classified.csv"
    else:
        # auto-detect in eval dir
        auto = args.eval / "../triage/failures_classified.csv"
        if auto.resolve().exists():
            triage_path = auto.resolve()
    if triage_path and triage_path.exists():
        failure_df = pd.read_csv(triage_path)
        print(f"  Triage: loaded {len(failure_df)} classified rows")

    # ── Render report ─────────────────────────────────────────────────────────
    dataset_dir = args.dataset if (not args.no_images and args.dataset.exists()) else None
    fmt_list = tuple(f.strip() for f in args.formats.split(","))

    outputs = render(
        df=df,
        tables=tables,
        overall=overall,
        model_name=args.model,
        dataset_dir=dataset_dir,
        gate_results=gate_results,
        failure_df=failure_df,
        out_dir=args.out,
        formats=fmt_list,
        iou_threshold=args.iou,
    )

    for fmt, path in outputs.items():
        size_kb = path.stat().st_size // 1024
        print(f"  Report ({fmt}): {path}  ({size_kb} KB)")

    # ── Optional experiment tracking ──────────────────────────────────────────
    if args.tracker != "none":
        log_run(
            overall=overall,
            tables=tables,
            model_name=args.model,
            iou_threshold=args.iou,
            gate_passed=gate_passed,
            tracker=args.tracker,
        )


if __name__ == "__main__":
    main()
