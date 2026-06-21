#!/usr/bin/env python3
"""End-to-end COCO eval: GT labels + model predictions → sliced metrics + report.

Usage:
    python scripts/run_coco_eval.py \\
        --gt       path/to/annotations.json \\
        --preds    path/to/results.json \\
        --out      artifacts/eval

    # With optional enrichment:
    python scripts/run_coco_eval.py \\
        --gt          path/to/annotations.json \\
        --preds       path/to/results.json \\
        --images      path/to/images/ \\
        --metadata    path/to/metadata.csv \\
        --pred-classes path/to/model_categories.json \\
        --out         artifacts/eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.coco_gt import load_coco_gt
from ingestion.coco_predictions import load_coco_predictions
from ingestion.class_map import load_or_create
from ingestion.metadata_csv import (
    load_metadata_csv,
    enrich_matches_with_metadata,
    slice_columns_from_metadata,
)
from ingestion.slice_inferrer import infer_slices, enrich_matches
from metrics.engine import overall_metrics, sliced_tables
from metrics.reporter import print_report, save_tables
from runner.matcher import match_scene


def _load_pred_categories(path: Path) -> dict[int, str]:
    """Load a model categories JSON: [{id: int, name: str}, ...]."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"--pred-classes file must be a JSON array of {{id, name}} objects: {path}"
        )
    return {int(item["id"]): item["name"] for item in raw}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PerceptorGuard COCO eval — sliced metrics from your labels and predictions"
    )
    ap.add_argument("--gt",     type=Path, required=True,
                    help="COCO annotations JSON (ground truth labels)")
    ap.add_argument("--preds",  type=Path, required=True,
                    help="COCO results JSON (model predictions array)")
    ap.add_argument("--out",    type=Path, default=Path("artifacts/eval"),
                    help="Output directory for CSV tables (default: artifacts/eval)")
    ap.add_argument("--images", type=Path, default=None,
                    help="Directory of image files — enables lighting slice")
    ap.add_argument("--metadata", type=Path, default=None,
                    help="Optional CSV with per-image metadata (weather, sensor, etc.)")
    ap.add_argument("--pred-classes", type=Path, default=None, dest="pred_classes",
                    help="Optional JSON with model category vocabulary "
                         "[{\"id\": 1, \"name\": \"car\"}, ...]. "
                         "Required when model uses different class IDs than your GT.")
    ap.add_argument("--class-map", type=Path,
                    default=Path("configs/class_map.yml"), dest="class_map",
                    help="Path to class mapping YAML (created on first run)")
    ap.add_argument("--iou",    type=float, default=0.5,
                    help="IoU match threshold (default: 0.5)")
    ap.add_argument("--no-save", action="store_true",
                    help="Print report only, do not write CSVs")
    args, _ = ap.parse_known_args()  # ignore synthetic-mode flags forwarded by CLI

    # ── 1. Load ground truth ──────────────────────────────────────────────────
    print(f"Loading GT: {args.gt}")
    gt_ds = load_coco_gt(args.gt)
    print(f"  {len(gt_ds.filename_by_id)} images  |  "
          f"{len(gt_ds.categories)} classes  |  "
          f"{sum(len(v) for v in gt_ds.gts_by_filename.values())} annotations")

    # ── 2. Load predictions ───────────────────────────────────────────────────
    pred_categories: dict[int, str] | None = None
    if args.pred_classes:
        print(f"Loading model categories: {args.pred_classes}")
        pred_categories = _load_pred_categories(args.pred_classes)
        print(f"  {len(pred_categories)} model classes")

    print(f"Loading predictions: {args.preds}")
    preds_by_filename = load_coco_predictions(
        args.preds,
        gt_ds.categories,
        gt_ds.filename_by_id,
        pred_categories=pred_categories,
    )
    total_preds = sum(len(v) for v in preds_by_filename.values())
    print(f"  {total_preds} predictions across {len(preds_by_filename)} images")

    # ── 3. Class mapping ──────────────────────────────────────────────────────
    user_classes = set(gt_ds.categories.values())
    model_classes = (
        set(pred_categories.values()) if pred_categories else user_classes
    )
    print(f"Resolving class map: {args.class_map}")
    class_map = load_or_create(user_classes, model_classes, args.class_map)
    gts_by_filename = class_map.apply_to_gts(gt_ds.gts_by_filename)

    # ── 4. Match scene by scene ───────────────────────────────────────────────
    print(f"Matching (IoU≥{args.iou}) ...")
    all_records: list[dict] = []
    all_filenames = sorted(set(gts_by_filename) | set(preds_by_filename))

    for fname in all_filenames:
        gts  = gts_by_filename.get(fname, [])
        preds = preds_by_filename.get(fname, [])
        records = match_scene(gts, preds, args.iou)
        for r in records:
            r["scene_id"] = fname
        all_records.extend(records)

    if not all_records:
        print("WARNING: no match records produced — check that GT and predictions "
              "reference the same image IDs.")
        sys.exit(1)

    df = pd.DataFrame(all_records)

    # ── 5. Auto-infer slices ──────────────────────────────────────────────────
    image_slices = infer_slices(gt_ds.gts_by_filename, args.images)
    df = enrich_matches(df, image_slices, gt_ds.image_sizes)

    # ── 6. Optional metadata slices ───────────────────────────────────────────
    extra_dims: list[str] = []
    if args.metadata:
        print(f"Loading metadata: {args.metadata}")
        metadata = load_metadata_csv(args.metadata)
        extra_dims = slice_columns_from_metadata(metadata)
        df = enrich_matches_with_metadata(df, metadata)
        print(f"  Added {len(extra_dims)} metadata slice(s): {', '.join(extra_dims)}")

    # ── 7. Metrics ────────────────────────────────────────────────────────────
    overall = overall_metrics(df)
    tables  = sliced_tables(df, extra_dims=extra_dims)

    # ── 8. Report ─────────────────────────────────────────────────────────────
    print_report(df, tables, overall, iou_threshold=args.iou)

    if not args.no_save:
        save_tables(df, tables, overall, args.out)


if __name__ == "__main__":
    main()
