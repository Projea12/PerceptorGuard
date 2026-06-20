#!/usr/bin/env python3
"""
Run the full eval loop and produce sliced metrics.

  python scripts/run_eval.py --dataset artifacts/dataset --out artifacts/eval
  python scripts/run_eval.py --dataset artifacts/dataset --model yolov8s.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from runner.eval_runner import EvalRunner
from metrics.engine import overall_metrics, sliced_tables
from metrics.reporter import print_report, save_tables


def main() -> None:
    ap = argparse.ArgumentParser(description="PerceptorGuard eval runner")
    ap.add_argument("--dataset", type=Path, default=Path("artifacts/dataset"),
                    help="Directory containing manifest.csv and scenes/")
    ap.add_argument("--out",     type=Path, default=Path("artifacts/eval"),
                    help="Output directory for CSV tables")
    ap.add_argument("--model",   default="yolov8n.pt",
                    help="Ultralytics model name or path (default: yolov8n.pt)")
    ap.add_argument("--iou",     type=float, default=0.5,
                    help="IoU match threshold (default: 0.5)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Inference image size in pixels (default: 640; use 160 for fast CI)")
    ap.add_argument("--no-save", action="store_true",
                    help="Print report only, do not write CSVs")
    args = ap.parse_args()

    runner = EvalRunner(model_name=args.model, iou_threshold=args.iou, imgsz=args.imgsz)
    df = runner.run_dataset(args.dataset)

    overall = overall_metrics(df)
    tables  = sliced_tables(df)

    print_report(df, tables, overall,
                 model_name=args.model, iou_threshold=args.iou)

    if not args.no_save:
        save_tables(df, tables, overall, args.out)


if __name__ == "__main__":
    main()
