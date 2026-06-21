"""Auto-infer slice labels from user data — no metadata required.

Three slices are derived from the data itself:

  object_size  — bbox area relative to image area (small / medium / large)
  clutter      — GT count per image (sparse / moderate / crowded)
  lighting     — mean pixel brightness of the image (dark / normal / bright)

Thresholds follow COCO conventions for size; clutter and lighting thresholds
are tuned to give roughly even splits across typical detection datasets.

Usage:
    slices = infer_slices(gts_by_filename, image_sizes, image_dir)
    matches_df = enrich_matches(matches_df, slices)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scenarios.schemas import BoundingBox, GroundTruth

# ── size bins (fraction of image area) — COCO small/medium/large ─────────────
_SIZE_SMALL_MAX  = 0.02   # < 2 % of image area
_SIZE_LARGE_MIN  = 0.10   # ≥ 10 % of image area

# ── clutter bins (GT count per image) ────────────────────────────────────────
_CLUTTER_SPARSE_MAX   = 2   # 1–2 objects
_CLUTTER_CROWDED_MIN  = 6   # 6+ objects

# ── lighting bins (mean pixel value, 0–255) ───────────────────────────────────
_LIGHT_DARK_MAX    = 85
_LIGHT_BRIGHT_MIN  = 170


@dataclass
class ImageSlices:
    """Per-image slice labels derived without metadata."""
    scene_id: str
    clutter_bin: str            # "sparse" | "moderate" | "crowded"
    lighting_bin: str           # "dark" | "normal" | "bright"
    object_count: int


def size_bin(box: BoundingBox, image_width: int, image_height: int) -> str:
    """Return "small" / "medium" / "large" for a single bounding box."""
    image_area = image_width * image_height
    if image_area <= 0:
        return "unknown"
    frac = box.area / image_area
    if frac < _SIZE_SMALL_MAX:
        return "small"
    if frac >= _SIZE_LARGE_MIN:
        return "large"
    return "medium"


def _clutter_bin(count: int) -> str:
    if count <= _CLUTTER_SPARSE_MAX:
        return "sparse"
    if count >= _CLUTTER_CROWDED_MIN:
        return "crowded"
    return "moderate"


def _lighting_bin(mean_brightness: float) -> str:
    if mean_brightness < _LIGHT_DARK_MAX:
        return "dark"
    if mean_brightness >= _LIGHT_BRIGHT_MIN:
        return "bright"
    return "normal"


def _mean_brightness(image_path: Path) -> float:
    """Return mean pixel brightness (0–255) across all channels."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("L")  # greyscale
    return float(np.asarray(img, dtype=np.float32).mean())


def infer_slices(
    gts_by_filename: dict[str, list[GroundTruth]],
    image_dir: Optional[Path] = None,
) -> dict[str, ImageSlices]:
    """Compute per-image slice labels from GTs and (optionally) images.

    Args:
        gts_by_filename: {filename -> [GroundTruth, ...]} from the COCO GT loader.
        image_dir: Directory containing the image files. When provided, lighting
                   is inferred from pixel brightness. When None, lighting_bin is
                   set to "unknown" for every image.

    Returns:
        {scene_id -> ImageSlices}
    """
    result: dict[str, ImageSlices] = {}
    missing_images: list[str] = []

    for fname, gts in gts_by_filename.items():
        count = len(gts)

        # ── lighting ──────────────────────────────────────────────────────────
        if image_dir is not None:
            img_path = Path(image_dir) / fname
            if img_path.exists():
                brightness = _mean_brightness(img_path)
                light = _lighting_bin(brightness)
            else:
                missing_images.append(fname)
                light = "unknown"
        else:
            light = "unknown"

        result[fname] = ImageSlices(
            scene_id=fname,
            clutter_bin=_clutter_bin(count),
            lighting_bin=light,
            object_count=count,
        )

    if missing_images:
        warnings.warn(
            f"Could not find {len(missing_images)} image(s) in {image_dir} "
            f"for lighting inference — lighting_bin set to 'unknown'. "
            f"First missing: {missing_images[0]}",
            stacklevel=2,
        )

    return result


def enrich_matches(
    matches: pd.DataFrame,
    image_slices: dict[str, ImageSlices],
    image_sizes: Optional[dict[str, tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Add slice columns to a matches DataFrame.

    Adds three new columns:
      - size_bin    ("small" / "medium" / "large" / "unknown") — per GT row
      - clutter_bin ("sparse" / "moderate" / "crowded")        — per image
      - lighting_bin("dark" / "normal" / "bright" / "unknown") — per image

    Args:
        matches: DataFrame with at least columns [scene_id, gt_box_area].
        image_slices: Output of infer_slices().
        image_sizes: {filename -> (width, height)} from CocoGTDataset.image_sizes.
                     Required for size_bin. When None, size_bin is "unknown".

    Returns:
        New DataFrame with three additional columns.
    """
    df = matches.copy()

    # ── per-image slices: clutter + lighting ──────────────────────────────────
    clutter_map = {s.scene_id: s.clutter_bin for s in image_slices.values()}
    lighting_map = {s.scene_id: s.lighting_bin for s in image_slices.values()}

    df["clutter_bin"]  = df["scene_id"].map(clutter_map).fillna("unknown")
    df["lighting_bin"] = df["scene_id"].map(lighting_map).fillna("unknown")

    # ── per-row size bin from gt_box_area ─────────────────────────────────────
    if image_sizes is not None and "gt_box_area" in df.columns:
        def _row_size_bin(row) -> str:
            sizes = image_sizes.get(row["scene_id"])
            if sizes is None or row["gt_box_area"] is None:
                return "unknown"
            w, h = sizes
            image_area = w * h
            if image_area <= 0:
                return "unknown"
            frac = row["gt_box_area"] / image_area
            if frac < _SIZE_SMALL_MAX:
                return "small"
            if frac >= _SIZE_LARGE_MIN:
                return "large"
            return "medium"

        df["size_bin"] = df.apply(_row_size_bin, axis=1)
    else:
        df["size_bin"] = "unknown"

    return df
