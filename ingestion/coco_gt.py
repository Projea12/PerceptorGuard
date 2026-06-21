"""Parse COCO-format ground truth annotations into PerceptorGuard schemas.

Expects the standard COCO annotations JSON structure:
    {
        "images":      [{"id": int, "file_name": str, "width": int, "height": int}, ...],
        "categories":  [{"id": int, "name": str}, ...],
        "annotations": [{"id": int, "image_id": int, "category_id": int,
                          "bbox": [x, y, w, h], "iscrowd": 0|1}, ...]
    }
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from scenarios.schemas import BoundingBox, GroundTruth


@dataclass
class CocoGTDataset:
    gts_by_filename: dict[str, list[GroundTruth]]
    categories: dict[int, str]          # category_id -> class_name
    filename_by_id: dict[int, str]      # image_id -> filename
    image_sizes: dict[str, tuple[int, int]]  # filename -> (width, height)


def load_coco_gt(path: Path, tier: str = "user") -> CocoGTDataset:
    """Load a COCO annotations JSON file and return a CocoGTDataset.

    Args:
        path: Path to the COCO annotations JSON file.
        tier: Tier label to assign every GroundTruth ("user" by default;
              pass "easy"/"hard" if you want gate-level split).

    Returns:
        CocoGTDataset with gts indexed by file_name.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the JSON is not a COCO annotations object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"COCO GT file not found: {path}")

    raw = json.loads(path.read_text())

    if not isinstance(raw, dict):
        raise ValueError(f"COCO GT must be a JSON object, got {type(raw).__name__}: {path}")

    for key in ("images", "categories", "annotations"):
        if key not in raw:
            raise ValueError(f"COCO GT missing required key '{key}': {path}")

    categories: dict[int, str] = {
        int(cat["id"]): cat["name"]
        for cat in raw["categories"]
    }

    filename_by_id: dict[int, str] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for img in raw["images"]:
        iid = int(img["id"])
        fname = img["file_name"]
        filename_by_id[iid] = fname
        if "width" in img and "height" in img:
            image_sizes[fname] = (int(img["width"]), int(img["height"]))

    gts_by_filename: dict[str, list[GroundTruth]] = {}
    skipped_crowd = 0
    skipped_degenerate = 0
    skipped_unknown_image = 0
    skipped_unknown_category = 0

    for ann in raw["annotations"]:
        if ann.get("iscrowd", 0):
            skipped_crowd += 1
            continue

        iid = int(ann["image_id"])
        if iid not in filename_by_id:
            skipped_unknown_image += 1
            continue

        cid = int(ann["category_id"])
        if cid not in categories:
            skipped_unknown_category += 1
            continue

        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            skipped_degenerate += 1
            continue

        box = BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
        fname = filename_by_id[iid]
        gt = GroundTruth(
            box=box,
            class_id=cid,
            class_name=categories[cid],
            tier=tier,
            scene_id=fname,
            object_id=str(ann.get("id", "")),
        )
        gts_by_filename.setdefault(fname, []).append(gt)

    if skipped_crowd:
        warnings.warn(f"Skipped {skipped_crowd} crowd annotations (iscrowd=1)", stacklevel=2)
    if skipped_degenerate:
        warnings.warn(f"Skipped {skipped_degenerate} degenerate boxes (w<=0 or h<=0)", stacklevel=2)
    if skipped_unknown_image:
        warnings.warn(f"Skipped {skipped_unknown_image} annotations with unknown image_id", stacklevel=2)
    if skipped_unknown_category:
        warnings.warn(f"Skipped {skipped_unknown_category} annotations with unknown category_id", stacklevel=2)

    return CocoGTDataset(
        gts_by_filename=gts_by_filename,
        categories=categories,
        filename_by_id=filename_by_id,
        image_sizes=image_sizes,
    )
