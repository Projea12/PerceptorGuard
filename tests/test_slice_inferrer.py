"""Tests for auto-inferred slice labels."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ingestion.slice_inferrer import (
    ImageSlices,
    enrich_matches,
    infer_slices,
    size_bin,
)
from scenarios.schemas import BoundingBox, GroundTruth


# ── helpers ───────────────────────────────────────────────────────────────────

def _box(x, y, w, h) -> BoundingBox:
    return BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)


def _gt(class_name="car", scene_id="img.jpg", box=None) -> GroundTruth:
    return GroundTruth(
        box=box or _box(0, 0, 10, 10),
        class_id=1, class_name=class_name,
        tier="user", scene_id=scene_id,
    )


def _write_grey_image(tmp_path: Path, name: str, brightness: int) -> Path:
    p = tmp_path / name
    arr = np.full((100, 100), brightness, dtype=np.uint8)
    Image.fromarray(arr, mode="L").convert("RGB").save(p)
    return p


def _matches_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── size_bin ──────────────────────────────────────────────────────────────────

class TestSizeBin:
    def test_small_box(self):
        # 10×10 = 100 area; image 1000×1000 = 1_000_000 → frac=0.0001 → small
        box = _box(0, 0, 10, 10)
        assert size_bin(box, 1000, 1000) == "small"

    def test_large_box(self):
        # 400×400 = 160_000 area; image 640×480 = 307_200 → frac≈0.52 → large
        box = _box(0, 0, 400, 400)
        assert size_bin(box, 640, 480) == "large"

    def test_medium_box(self):
        # 64×48 = 3_072 area; image 640×480 = 307_200 → frac=0.01 → small
        # use 100×100 in 640×480: frac≈0.033 → medium
        box = _box(0, 0, 100, 100)
        assert size_bin(box, 640, 480) == "medium"

    def test_exactly_at_small_boundary(self):
        # frac exactly 0.02 → medium (< threshold means small, = medium)
        # 640×480 * 0.02 = 6_144 → side ≈ 78.4; use 78×78 = 6084 < 6144 → small
        box = _box(0, 0, 78, 78)
        assert size_bin(box, 640, 480) == "small"

    def test_exactly_at_large_boundary(self):
        # 640×480 * 0.10 = 30_720 → 175×175 = 30_625 < 30_720 → medium
        box = _box(0, 0, 175, 175)
        assert size_bin(box, 640, 480) == "medium"

    def test_zero_image_area_returns_unknown(self):
        box = _box(0, 0, 10, 10)
        assert size_bin(box, 0, 0) == "unknown"


# ── infer_slices: clutter ─────────────────────────────────────────────────────

class TestInferSlicesClutter:
    def test_sparse_one_object(self):
        gts = {"a.jpg": [_gt()]}
        slices = infer_slices(gts)
        assert slices["a.jpg"].clutter_bin == "sparse"

    def test_sparse_two_objects(self):
        gts = {"a.jpg": [_gt(), _gt()]}
        slices = infer_slices(gts)
        assert slices["a.jpg"].clutter_bin == "sparse"

    def test_moderate_three_objects(self):
        gts = {"a.jpg": [_gt()] * 3}
        slices = infer_slices(gts)
        assert slices["a.jpg"].clutter_bin == "moderate"

    def test_moderate_five_objects(self):
        gts = {"a.jpg": [_gt()] * 5}
        slices = infer_slices(gts)
        assert slices["a.jpg"].clutter_bin == "moderate"

    def test_crowded_six_objects(self):
        gts = {"a.jpg": [_gt()] * 6}
        slices = infer_slices(gts)
        assert slices["a.jpg"].clutter_bin == "crowded"

    def test_crowded_many_objects(self):
        gts = {"a.jpg": [_gt()] * 20}
        slices = infer_slices(gts)
        assert slices["a.jpg"].clutter_bin == "crowded"

    def test_object_count_recorded(self):
        gts = {"a.jpg": [_gt()] * 4}
        slices = infer_slices(gts)
        assert slices["a.jpg"].object_count == 4

    def test_multiple_images_independent(self):
        gts = {"a.jpg": [_gt()], "b.jpg": [_gt()] * 10}
        slices = infer_slices(gts)
        assert slices["a.jpg"].clutter_bin == "sparse"
        assert slices["b.jpg"].clutter_bin == "crowded"


# ── infer_slices: lighting ────────────────────────────────────────────────────

class TestInferSlicesLighting:
    def test_no_image_dir_gives_unknown(self):
        gts = {"a.jpg": [_gt()]}
        slices = infer_slices(gts, image_dir=None)
        assert slices["a.jpg"].lighting_bin == "unknown"

    def test_dark_image(self, tmp_path):
        _write_grey_image(tmp_path, "dark.jpg", brightness=40)
        gts = {"dark.jpg": [_gt(scene_id="dark.jpg")]}
        slices = infer_slices(gts, image_dir=tmp_path)
        assert slices["dark.jpg"].lighting_bin == "dark"

    def test_normal_image(self, tmp_path):
        _write_grey_image(tmp_path, "normal.jpg", brightness=128)
        gts = {"normal.jpg": [_gt(scene_id="normal.jpg")]}
        slices = infer_slices(gts, image_dir=tmp_path)
        assert slices["normal.jpg"].lighting_bin == "normal"

    def test_bright_image(self, tmp_path):
        _write_grey_image(tmp_path, "bright.jpg", brightness=220)
        gts = {"bright.jpg": [_gt(scene_id="bright.jpg")]}
        slices = infer_slices(gts, image_dir=tmp_path)
        assert slices["bright.jpg"].lighting_bin == "bright"

    def test_missing_image_gives_unknown_with_warning(self, tmp_path):
        gts = {"nonexistent.jpg": [_gt(scene_id="nonexistent.jpg")]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            slices = infer_slices(gts, image_dir=tmp_path)
        assert slices["nonexistent.jpg"].lighting_bin == "unknown"
        assert any("nonexistent.jpg" in str(x.message) for x in w)

    def test_scene_id_set(self, tmp_path):
        _write_grey_image(tmp_path, "x.jpg", brightness=128)
        gts = {"x.jpg": [_gt(scene_id="x.jpg")]}
        slices = infer_slices(gts, image_dir=tmp_path)
        assert slices["x.jpg"].scene_id == "x.jpg"


# ── enrich_matches ────────────────────────────────────────────────────────────

class TestEnrichMatches:
    def _slices(self) -> dict[str, ImageSlices]:
        return {
            "a.jpg": ImageSlices("a.jpg", clutter_bin="sparse",   lighting_bin="dark",   object_count=1),
            "b.jpg": ImageSlices("b.jpg", clutter_bin="crowded",  lighting_bin="bright", object_count=8),
        }

    def test_adds_clutter_bin_column(self):
        df = _matches_df([
            {"scene_id": "a.jpg", "gt_box_area": 100.0},
            {"scene_id": "b.jpg", "gt_box_area": 100.0},
        ])
        result = enrich_matches(df, self._slices())
        assert "clutter_bin" in result.columns
        assert result.loc[result["scene_id"] == "a.jpg", "clutter_bin"].iloc[0] == "sparse"
        assert result.loc[result["scene_id"] == "b.jpg", "clutter_bin"].iloc[0] == "crowded"

    def test_adds_lighting_bin_column(self):
        df = _matches_df([
            {"scene_id": "a.jpg", "gt_box_area": 100.0},
        ])
        result = enrich_matches(df, self._slices())
        assert "lighting_bin" in result.columns
        assert result.loc[result["scene_id"] == "a.jpg", "lighting_bin"].iloc[0] == "dark"

    def test_adds_size_bin_column_with_sizes(self):
        # gt_box_area = 100; image 640×480 = 307200; frac≈0.000325 → small
        df = _matches_df([{"scene_id": "a.jpg", "gt_box_area": 100.0}])
        sizes = {"a.jpg": (640, 480)}
        result = enrich_matches(df, self._slices(), image_sizes=sizes)
        assert result["size_bin"].iloc[0] == "small"

    def test_size_bin_large(self):
        # gt_box_area = 200000; image 640×480 = 307200; frac≈0.65 → large
        df = _matches_df([{"scene_id": "a.jpg", "gt_box_area": 200_000.0}])
        sizes = {"a.jpg": (640, 480)}
        result = enrich_matches(df, self._slices(), image_sizes=sizes)
        assert result["size_bin"].iloc[0] == "large"

    def test_size_bin_unknown_without_sizes(self):
        df = _matches_df([{"scene_id": "a.jpg", "gt_box_area": 100.0}])
        result = enrich_matches(df, self._slices(), image_sizes=None)
        assert result["size_bin"].iloc[0] == "unknown"

    def test_unknown_scene_id_fills_unknown(self):
        df = _matches_df([{"scene_id": "missing.jpg", "gt_box_area": 100.0}])
        result = enrich_matches(df, self._slices())
        assert result["clutter_bin"].iloc[0] == "unknown"
        assert result["lighting_bin"].iloc[0] == "unknown"

    def test_does_not_mutate_input_df(self):
        df = _matches_df([{"scene_id": "a.jpg", "gt_box_area": 100.0}])
        original_cols = list(df.columns)
        enrich_matches(df, self._slices())
        assert list(df.columns) == original_cols

    def test_fp_rows_with_none_gt_box_area(self):
        """FP rows have gt_box_area=None; size_bin should be unknown, not crash."""
        df = _matches_df([{"scene_id": "a.jpg", "gt_box_area": None}])
        sizes = {"a.jpg": (640, 480)}
        result = enrich_matches(df, self._slices(), image_sizes=sizes)
        assert result["size_bin"].iloc[0] == "unknown"

    def test_multiple_rows_same_image(self):
        df = _matches_df([
            {"scene_id": "a.jpg", "gt_box_area": 50.0},
            {"scene_id": "a.jpg", "gt_box_area": 50.0},
        ])
        result = enrich_matches(df, self._slices())
        assert (result["clutter_bin"] == "sparse").all()
