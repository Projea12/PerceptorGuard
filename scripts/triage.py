#!/usr/bin/env python3
"""
Failure triage: classify and cluster failures from an eval run.

  python scripts/triage.py
  python scripts/triage.py --matches artifacts/eval/matches.csv
  python scripts/triage.py --no-cluster --no-save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from metrics.failure_classifier import classify_failures
from metrics.cluster_analyzer import cluster_failures, cluster_summary
from metrics.triage_reporter import print_triage, save_triage


def main() -> None:
    ap = argparse.ArgumentParser(description="PerceptorGuard failure triage")
    ap.add_argument("--matches", type=Path, default=Path("artifacts/eval/matches.csv"),
                    help="matches.csv from run_eval.py (default: artifacts/eval/matches.csv)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/triage"),
                    help="Output directory (default: artifacts/triage)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold used in eval (default: 0.5)")
    ap.add_argument("--clusters", type=int, default=5,
                    help="Number of KMeans clusters (default: 5)")
    ap.add_argument("--no-cluster", action="store_true",
                    help="Skip clustering step")
    ap.add_argument("--no-save", action="store_true",
                    help="Print report only, do not write CSVs")
    ap.add_argument("--model", default="",
                    help="Model name for report header")
    args = ap.parse_args()

    if not args.matches.exists():
        sys.exit(f"ERROR: {args.matches} not found — run scripts/run_eval.py first")

    df = pd.read_csv(args.matches)
    df = classify_failures(df, iou_threshold=args.iou)

    csummary = pd.DataFrame()
    if not args.no_cluster:
        clustered = cluster_failures(df, n_clusters=args.clusters)
        csummary = cluster_summary(clustered)
        # merge cluster column back onto df where applicable
        if "cluster" in clustered.columns:
            df = df.merge(
                clustered[["cluster"]],
                left_index=True, right_index=True, how="left"
            )

    print_triage(df, csummary, model_name=args.model)

    if not args.no_save:
        save_triage(df, csummary, args.out)


if __name__ == "__main__":
    main()
