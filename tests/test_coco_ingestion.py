"""Tests for COCO GT and predictions parsers."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from ingestion.coco_gt import load_coco_gt, CocoGTDataset
from ingestion.coco_predictions import load_coco_predictions
from scenarios.schemas import GroundTruth, Detection


# ── helpers ───────────────────────────────────────────────────────────────────

def _minimal_gt_json(
    images=None, categories=None, annotations=None
) -> dict:
    return {
        "images": images or [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "b.jpg", "width": 1920, "height": 1080},
        ],
        "categories": categories or [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "person"},
        ],
        "annotations": annotations or [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 30], "iscrowd": 0},
            {"id": 11, "image_id": 2, "category_id": 2, "bbox": [5, 5, 100, 200], "iscrowd": 0},
        ],
    }


def _write_json(tmp_path: Path, name: str, data) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ── load_coco_gt: happy path ──────────────────────────────────────────────────

class TestLoadCocoGTHappyPath:
    def test_returns_dataset(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        assert isinstance(ds, CocoGTDataset)

    def test_categories_parsed(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        assert ds.categories == {1: "car", 2: "person"}

    def test_filename_by_id(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        assert ds.filename_by_id == {1: "a.jpg", 2: "b.jpg"}

    def test_image_sizes(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        assert ds.image_sizes == {"a.jpg": (640, 480), "b.jpg": (1920, 1080)}

    def test_gts_indexed_by_filename(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        assert set(ds.gts_by_filename.keys()) == {"a.jpg", "b.jpg"}
        assert len(ds.gts_by_filename["a.jpg"]) == 1
        assert len(ds.gts_by_filename["b.jpg"]) == 1

    def test_bbox_conversion(self, tmp_path):
        """COCO [x,y,w,h] → BoundingBox(x_min,y_min,x_max,y_max)."""
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        gt = ds.gts_by_filename["a.jpg"][0]
        assert gt.box.x_min == 10
        assert gt.box.y_min == 20
        assert gt.box.x_max == 60   # 10 + 50
        assert gt.box.y_max == 50   # 20 + 30

    def test_gt_fields(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        gt = ds.gts_by_filename["a.jpg"][0]
        assert isinstance(gt, GroundTruth)
        assert gt.class_id == 1
        assert gt.class_name == "car"
        assert gt.scene_id == "a.jpg"
        assert gt.object_id == "10"

    def test_default_tier_is_user(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p)
        gt = ds.gts_by_filename["a.jpg"][0]
        assert gt.tier == "user"

    def test_custom_tier(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        ds = load_coco_gt(p, tier="hard")
        gt = ds.gts_by_filename["a.jpg"][0]
        assert gt.tier == "hard"

    def test_multiple_gts_same_image(self, tmp_path):
        data = _minimal_gt_json(annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 20, 5, 5], "iscrowd": 0},
        ])
        p = _write_json(tmp_path, "gt.json", data)
        ds = load_coco_gt(p)
        assert len(ds.gts_by_filename["a.jpg"]) == 2


# ── load_coco_gt: filtering ───────────────────────────────────────────────────

class TestLoadCocoGTFiltering:
    def test_crowd_annotations_skipped(self, tmp_path):
        data = _minimal_gt_json(annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 1},
        ])
        p = _write_json(tmp_path, "gt.json", data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = load_coco_gt(p)
        assert "a.jpg" not in ds.gts_by_filename
        assert any("crowd" in str(x.message).lower() for x in w)

    def test_zero_width_box_skipped(self, tmp_path):
        data = _minimal_gt_json(annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 0, 20], "iscrowd": 0},
        ])
        p = _write_json(tmp_path, "gt.json", data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = load_coco_gt(p)
        assert "a.jpg" not in ds.gts_by_filename
        assert any("degenerate" in str(x.message).lower() for x in w)

    def test_zero_height_box_skipped(self, tmp_path):
        data = _minimal_gt_json(annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 0], "iscrowd": 0},
        ])
        p = _write_json(tmp_path, "gt.json", data)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ds = load_coco_gt(p)
        assert "a.jpg" not in ds.gts_by_filename

    def test_unknown_image_id_skipped(self, tmp_path):
        data = _minimal_gt_json(annotations=[
            {"id": 1, "image_id": 999, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        ])
        p = _write_json(tmp_path, "gt.json", data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = load_coco_gt(p)
        assert len(ds.gts_by_filename) == 0
        assert any("unknown image_id" in str(x.message).lower() for x in w)

    def test_unknown_category_id_skipped(self, tmp_path):
        data = _minimal_gt_json(annotations=[
            {"id": 1, "image_id": 1, "category_id": 999, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        ])
        p = _write_json(tmp_path, "gt.json", data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = load_coco_gt(p)
        assert "a.jpg" not in ds.gts_by_filename
        assert any("unknown category_id" in str(x.message).lower() for x in w)

    def test_iscrowd_missing_treated_as_zero(self, tmp_path):
        data = _minimal_gt_json(annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ])
        p = _write_json(tmp_path, "gt.json", data)
        ds = load_coco_gt(p)
        assert len(ds.gts_by_filename["a.jpg"]) == 1


# ── load_coco_gt: error handling ──────────────────────────────────────────────

class TestLoadCocoGTErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_coco_gt(tmp_path / "nonexistent.json")

    def test_not_a_dict(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", [1, 2, 3])
        with pytest.raises(ValueError, match="JSON object"):
            load_coco_gt(p)

    def test_missing_images_key(self, tmp_path):
        data = {"categories": [], "annotations": []}
        p = _write_json(tmp_path, "gt.json", data)
        with pytest.raises(ValueError, match="images"):
            load_coco_gt(p)

    def test_missing_categories_key(self, tmp_path):
        data = {"images": [], "annotations": []}
        p = _write_json(tmp_path, "gt.json", data)
        with pytest.raises(ValueError, match="categories"):
            load_coco_gt(p)

    def test_missing_annotations_key(self, tmp_path):
        data = {"images": [], "categories": []}
        p = _write_json(tmp_path, "gt.json", data)
        with pytest.raises(ValueError, match="annotations"):
            load_coco_gt(p)

    def test_empty_dataset_ok(self, tmp_path):
        data = {"images": [], "categories": [], "annotations": []}
        p = _write_json(tmp_path, "gt.json", data)
        ds = load_coco_gt(p)
        assert ds.gts_by_filename == {}
        assert ds.categories == {}


# ── load_coco_predictions: happy path ────────────────────────────────────────

def _make_preds(*entries):
    return list(entries)


class TestLoadCocoPredictionsHappyPath:
    def _ds(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        return load_coco_gt(p)

    def test_returns_dict_of_detections(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.9}]
        p = _write_json(tmp_path, "preds.json", preds)
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert isinstance(result, dict)
        assert "a.jpg" in result
        assert len(result["a.jpg"]) == 1
        assert isinstance(result["a.jpg"][0], Detection)

    def test_bbox_conversion(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.8}]
        p = _write_json(tmp_path, "preds.json", preds)
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        det = result["a.jpg"][0]
        assert det.box.x_min == 10
        assert det.box.y_min == 20
        assert det.box.x_max == 40  # 10 + 30
        assert det.box.y_max == 60  # 20 + 40

    def test_score_field(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.75}]
        p = _write_json(tmp_path, "preds.json", preds)
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result["a.jpg"][0].confidence == pytest.approx(0.75)

    def test_confidence_field_accepted(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.6}]
        p = _write_json(tmp_path, "preds.json", preds)
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result["a.jpg"][0].confidence == pytest.approx(0.6)

    def test_sorted_by_confidence_descending(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.3},
            {"image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10], "score": 0.6},
        ]
        p = _write_json(tmp_path, "preds.json", preds)
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        confs = [d.confidence for d in result["a.jpg"]]
        assert confs == sorted(confs, reverse=True)

    def test_frame_id_set_to_filename(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}]
        p = _write_json(tmp_path, "preds.json", preds)
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result["a.jpg"][0].frame_id == "a.jpg"

    def test_empty_predictions_list(self, tmp_path):
        ds = self._ds(tmp_path)
        p = _write_json(tmp_path, "preds.json", [])
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result == {}

    def test_multiple_images(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 2, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.7},
        ]
        p = _write_json(tmp_path, "preds.json", preds)
        result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert set(result.keys()) == {"a.jpg", "b.jpg"}


# ── load_coco_predictions: filtering ─────────────────────────────────────────

class TestLoadCocoPredictionsFiltering:
    def _ds(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        return load_coco_gt(p)

    def test_unknown_image_id_skipped(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 999, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}]
        p = _write_json(tmp_path, "preds.json", preds)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result == {}
        assert any("unknown image_id" in str(x.message).lower() for x in w)

    def test_unknown_category_id_skipped(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 999, "bbox": [0, 0, 10, 10], "score": 0.5}]
        p = _write_json(tmp_path, "preds.json", preds)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result == {}
        assert any("unknown category_id" in str(x.message).lower() for x in w)

    def test_degenerate_box_skipped(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 10], "score": 0.5}]
        p = _write_json(tmp_path, "preds.json", preds)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result == {}
        assert any("degenerate" in str(x.message).lower() for x in w)

    def test_missing_score_skipped(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}]
        p = _write_json(tmp_path, "preds.json", preds)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result == {}
        assert any("score" in str(x.message).lower() for x in w)

    def test_score_above_one_skipped(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 1.5}]
        p = _write_json(tmp_path, "preds.json", preds)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result == {}

    def test_score_below_zero_skipped(self, tmp_path):
        ds = self._ds(tmp_path)
        preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": -0.1}]
        p = _write_json(tmp_path, "preds.json", preds)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_coco_predictions(p, ds.categories, ds.filename_by_id)
        assert result == {}


# ── load_coco_predictions: error handling ─────────────────────────────────────

class TestLoadCocoPredictionsErrors:
    def _ds(self, tmp_path):
        p = _write_json(tmp_path, "gt.json", _minimal_gt_json())
        return load_coco_gt(p)

    def test_file_not_found(self, tmp_path):
        ds = self._ds(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_coco_predictions(tmp_path / "missing.json", ds.categories, ds.filename_by_id)

    def test_json_object_raises(self, tmp_path):
        ds = self._ds(tmp_path)
        p = _write_json(tmp_path, "preds.json", {"not": "an array"})
        with pytest.raises(ValueError, match="array"):
            load_coco_predictions(p, ds.categories, ds.filename_by_id)
