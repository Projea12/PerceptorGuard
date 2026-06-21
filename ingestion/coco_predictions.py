"""Parse COCO-format model predictions into PerceptorGuard Detection schemas.

Expects the standard COCO results JSON structure (must be an array):
    [
        {
            "image_id":   int,
            "category_id": int,
            "bbox":       [x, y, w, h],
            "score":      float          # also accepted: "confidence"
        },
        ...
    ]
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from scenarios.schemas import BoundingBox, Detection


def load_coco_predictions(
    path: Path,
    categories: dict[int, str],
    filename_by_id: dict[int, str],
    pred_categories: dict[int, str] | None = None,
) -> dict[str, list[Detection]]:
    """Load a COCO results JSON file and return detections indexed by filename.

    Args:
        path: Path to the COCO results JSON file (array of prediction dicts).
        categories: Mapping of category_id -> class_name (from CocoGTDataset).
            Used when pred_categories is None (standard same-vocabulary eval).
        filename_by_id: Mapping of image_id -> file_name (from CocoGTDataset).
        pred_categories: Optional separate category mapping for the model's output.
            Pass this when the model uses different category IDs than the GT
            (e.g. evaluating a COCO-pretrained model on custom-labeled data).

    Returns:
        Dict mapping file_name -> list[Detection], sorted by confidence descending.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the JSON is not a list or predictions are malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"COCO predictions file not found: {path}")

    raw = json.loads(path.read_text())

    if not isinstance(raw, list):
        raise ValueError(
            f"COCO predictions must be a JSON array, got {type(raw).__name__}: {path}. "
            "Did you pass an annotations file instead of a results file?"
        )

    cat_lookup = pred_categories if pred_categories is not None else categories

    preds_by_filename: dict[str, list[Detection]] = {}
    skipped_unknown_image = 0
    skipped_unknown_category = 0
    skipped_degenerate = 0
    skipped_invalid_score = 0

    for pred in raw:
        iid = int(pred["image_id"])
        if iid not in filename_by_id:
            skipped_unknown_image += 1
            continue

        cid = int(pred["category_id"])
        if cid not in cat_lookup:
            skipped_unknown_category += 1
            continue

        score = pred.get("score") if pred.get("score") is not None else pred.get("confidence")
        if score is None:
            skipped_invalid_score += 1
            continue
        score = float(score)
        if not (0.0 <= score <= 1.0):
            skipped_invalid_score += 1
            continue

        x, y, w, h = (float(v) for v in pred["bbox"])
        if w <= 0 or h <= 0:
            skipped_degenerate += 1
            continue

        fname = filename_by_id[iid]
        det = Detection(
            box=BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h),
            class_id=cid,
            class_name=cat_lookup[cid],
            confidence=score,
            frame_id=fname,
        )
        preds_by_filename.setdefault(fname, []).append(det)

    if skipped_unknown_image:
        warnings.warn(
            f"Skipped {skipped_unknown_image} predictions with unknown image_id",
            stacklevel=2,
        )
    if skipped_unknown_category:
        warnings.warn(
            f"Skipped {skipped_unknown_category} predictions with unknown category_id",
            stacklevel=2,
        )
    if skipped_degenerate:
        warnings.warn(
            f"Skipped {skipped_degenerate} degenerate prediction boxes (w<=0 or h<=0)",
            stacklevel=2,
        )
    if skipped_invalid_score:
        warnings.warn(
            f"Skipped {skipped_invalid_score} predictions with missing or out-of-range score",
            stacklevel=2,
        )

    for fname in preds_by_filename:
        preds_by_filename[fname].sort(key=lambda d: d.confidence, reverse=True)

    return preds_by_filename
