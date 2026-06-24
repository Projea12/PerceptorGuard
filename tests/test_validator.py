"""Tests for the input validation gate — all 10 rules."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from ingestion.coco_gt import load_coco_gt, CocoGTDataset
from ingestion.coco_predictions import load_coco_predictions
from ingestion.exceptions import (
    DuplicateIDError,
    EvalInputError,
    FileEmptyError,
    FileMismatchError,
    MissingImageError,
)
from ingestion.validator import validate_eval_inputs, ValidationSummary


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_json(tmp_path: Path, name: str, data) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _gt_json(image_ids: list[int], with_boxes=True) -> dict:
    images = [{"id": iid, "file_name": f"{iid}.jpg", "width": 640, "height": 480}
              for iid in image_ids]
    categories = [{"id": 1, "name": "cup"}]
    annotations = []
    if with_boxes:
        for iid in image_ids:
            annotations.append({
                "id": iid * 10,
                "image_id": iid,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "iscrowd": 0,
            })
    return {"images": images, "categories": categories, "annotations": annotations}


def _preds_json(image_ids: list[int]) -> list:
    return [
        {"image_id": iid, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.9}
        for iid in image_ids
    ]


def _load_gt(tmp_path: Path, image_ids: list[int], with_boxes=True) -> CocoGTDataset:
    p = _write_json(tmp_path, "gt.json", _gt_json(image_ids, with_boxes=with_boxes))
    return load_coco_gt(p)


def _load_preds(tmp_path: Path, gt_ds: CocoGTDataset, pred_image_ids: list[int]) -> dict:
    p = _write_json(tmp_path, "preds.json", _preds_json(pred_image_ids))
    return load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)


# ── TEST 1: Perfect match — validation passes ─────────────────────────────────

class TestPerfectMatch:
    def test_validation_passes(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        summary = validate_eval_inputs(gt_ds, preds)
        assert isinstance(summary, ValidationSummary)

    def test_summary_counts_are_correct(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        summary = validate_eval_inputs(gt_ds, preds)
        assert summary.gt_image_count == 3
        assert summary.pred_image_count == 3
        assert summary.matched_image_count == 3
        assert summary.gt_box_count == 3
        assert summary.duplicate_ids_found == 0

    def test_no_exception_raised(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        try:
            validate_eval_inputs(gt_ds, preds)
        except EvalInputError:
            pytest.fail("EvalInputError raised on valid matched inputs")


# ── TEST 2: The stranger_test bug — GT covers more images than predictions ────

class TestGTExceedsPredictions:
    def test_raises_file_mismatch_error(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3, 4, 5])
        preds = _load_preds(tmp_path, gt_ds, [1, 2])
        with pytest.raises(FileMismatchError):
            validate_eval_inputs(gt_ds, preds)

    def test_error_is_eval_input_error_subclass(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3, 4, 5])
        preds = _load_preds(tmp_path, gt_ds, [1, 2])
        with pytest.raises(EvalInputError):
            validate_eval_inputs(gt_ds, preds)

    def test_error_message_contains_image_counts(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3, 4, 5])
        preds = _load_preds(tmp_path, gt_ds, [1, 2])
        with pytest.raises(FileMismatchError) as exc_info:
            validate_eval_inputs(gt_ds, preds)
        msg = str(exc_info.value)
        assert "5" in msg
        assert "2" in msg

    def test_no_metrics_produced(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3, 4, 5])
        preds = _load_preds(tmp_path, gt_ds, [1, 2])
        raised = False
        try:
            validate_eval_inputs(gt_ds, preds)
        except FileMismatchError:
            raised = True
        assert raised, "Validator must stop before metrics are produced"


# ── TEST 3: Predictions have images that exceed GT ────────────────────────────
# Note: load_coco_predictions filters unknown image IDs so pred_filenames
# will always be a subset of gt_filenames. This test confirms that behaviour
# and that the validator handles it gracefully.

class TestPredsFilteredToGT:
    def test_unknown_pred_ids_filtered_by_loader(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        # Predictions include image IDs 4, 5 which are not in GT
        preds_json = _preds_json([1, 2, 4, 5])
        p = _write_json(tmp_path, "preds.json", preds_json)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        # IDs 4 and 5 are filtered — only 1 and 2 survive
        assert set(preds.keys()) == {"1.jpg", "2.jpg"}

    def test_partial_preds_triggers_mismatch(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds_json = _preds_json([1, 2, 4, 5])
        p = _write_json(tmp_path, "preds.json", preds_json)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        # preds covers [1,2] but GT covers [1,2,3] — mismatch on image 3
        with pytest.raises(FileMismatchError):
            validate_eval_inputs(gt_ds, preds)


# ── TEST 4: Complete wrong files — zero overlap ────────────────────────────────

class TestCompleteWrongFiles:
    def test_raises_file_mismatch_error(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        # Predictions only reference IDs 4, 5, 6 — none in GT
        preds_json = _preds_json([4, 5, 6])
        p = _write_json(tmp_path, "preds.json", preds_json)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        # All pred image IDs filtered → empty preds → FileEmptyError
        with pytest.raises(EvalInputError):
            validate_eval_inputs(gt_ds, preds)

    def test_error_message_mentions_stopping(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds_json = _preds_json([4, 5, 6])
        p = _write_json(tmp_path, "preds.json", preds_json)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        with pytest.raises(EvalInputError) as exc_info:
            validate_eval_inputs(gt_ds, preds)
        assert "Stopping" in str(exc_info.value)


# ── RULE 4: Exact duplicate predictions — hard stop ──────────────────────────

class TestExactDuplicatePredictions:
    def test_raises_duplicate_id_error(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        # Same prediction entry twice — same image, class, box, score
        preds_json = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},  # exact copy
            {"image_id": 2, "category_id": 1, "bbox": [20, 20, 30, 30], "score": 0.80},
        ]
        p = _write_json(tmp_path, "preds.json", preds_json)
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        with pytest.raises(DuplicateIDError):
            validate_eval_inputs(gt_ds, preds)

    def test_error_is_eval_input_error_subclass(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds_json = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},
            {"image_id": 2, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.70},
        ]
        p = _write_json(tmp_path, "preds.json", preds_json)
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        with pytest.raises(EvalInputError):
            validate_eval_inputs(gt_ds, preds)

    def test_different_boxes_same_image_allowed(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        # Two detections on image 1 — different boxes, completely valid
        preds_json = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},
            {"image_id": 1, "category_id": 1, "bbox": [60, 60, 30, 30], "score": 0.75},
            {"image_id": 2, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.80},
            {"image_id": 3, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.80},
        ]
        p = _write_json(tmp_path, "preds.json", preds_json)
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        try:
            validate_eval_inputs(gt_ds, preds)
        except DuplicateIDError:
            pytest.fail("DuplicateIDError raised for different boxes on same image — should be allowed")

    def test_error_message_mentions_inference_script(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds_json = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},
            {"image_id": 2, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.70},
        ]
        p = _write_json(tmp_path, "preds.json", preds_json)
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        with pytest.raises(DuplicateIDError) as exc_info:
            validate_eval_inputs(gt_ds, preds)
        assert "inference script" in str(exc_info.value).lower()

    def test_same_box_different_confidence_allowed(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        # Same box, different confidence — two separate detections, allowed
        preds_json = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.91},
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.45},
            {"image_id": 2, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.80},
            {"image_id": 3, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.80},
        ]
        p = _write_json(tmp_path, "preds.json", preds_json)
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        try:
            validate_eval_inputs(gt_ds, preds)
        except DuplicateIDError:
            pytest.fail("DuplicateIDError raised for same box with different confidence — should be allowed")


# ── TEST 5: Duplicate image IDs in GT ─────────────────────────────────────────

class TestDuplicateImageIDsInGT:
    def test_raises_duplicate_id_error(self, tmp_path):
        gt_data = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
                {"id": 1, "file_name": "b.jpg", "width": 640, "height": 480},  # duplicate
                {"id": 2, "file_name": "c.jpg", "width": 640, "height": 480},
            ],
            "categories": [{"id": 1, "name": "cup"}],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
                {"id": 20, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
            ],
        }
        p = _write_json(tmp_path, "gt.json", gt_data)
        gt_ds = load_coco_gt(p)
        preds = _load_preds(tmp_path, gt_ds, list(gt_ds.filename_by_id.keys()))
        with pytest.raises(DuplicateIDError):
            validate_eval_inputs(gt_ds, preds)

    def test_error_is_eval_input_error_subclass(self, tmp_path):
        gt_data = {
            "images": [
                {"id": 42, "file_name": "a.jpg", "width": 640, "height": 480},
                {"id": 42, "file_name": "b.jpg", "width": 640, "height": 480},
            ],
            "categories": [{"id": 1, "name": "cup"}],
            "annotations": [],
        }
        p = _write_json(tmp_path, "gt.json", gt_data)
        gt_ds = load_coco_gt(p)
        preds = _load_preds(tmp_path, gt_ds, list(gt_ds.filename_by_id.keys()))
        with pytest.raises(EvalInputError):
            validate_eval_inputs(gt_ds, preds)

    def test_duplicate_id_recorded_in_dataset(self, tmp_path):
        gt_data = {
            "images": [
                {"id": 42, "file_name": "a.jpg", "width": 640, "height": 480},
                {"id": 42, "file_name": "b.jpg", "width": 640, "height": 480},
            ],
            "categories": [{"id": 1, "name": "cup"}],
            "annotations": [],
        }
        p = _write_json(tmp_path, "gt.json", gt_data)
        gt_ds = load_coco_gt(p)
        assert 42 in gt_ds.duplicate_image_ids


# ── TEST 6: Empty predictions file ────────────────────────────────────────────

class TestEmptyPredictions:
    def test_raises_file_empty_error(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        p = _write_json(tmp_path, "preds.json", [])
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        with pytest.raises(FileEmptyError):
            validate_eval_inputs(gt_ds, preds)

    def test_error_is_eval_input_error_subclass(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        p = _write_json(tmp_path, "preds.json", [])
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        with pytest.raises(EvalInputError):
            validate_eval_inputs(gt_ds, preds)

    def test_error_message_mentions_zero_predictions(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        p = _write_json(tmp_path, "preds.json", [])
        preds = load_coco_predictions(p, gt_ds.categories, gt_ds.filename_by_id)
        with pytest.raises(FileEmptyError) as exc_info:
            validate_eval_inputs(gt_ds, preds)
        assert "0" in str(exc_info.value)


# ── TEST 7: Images missing from disk ──────────────────────────────────────────

class TestImagesMissingFromDisk:
    def test_raises_missing_image_error(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        # Create only image 1 and 2 on disk — image 3 is missing
        (images_dir / "1.jpg").write_bytes(b"fake")
        (images_dir / "2.jpg").write_bytes(b"fake")
        with pytest.raises(MissingImageError):
            validate_eval_inputs(gt_ds, preds, images_dir=images_dir)

    def test_no_error_when_all_images_present(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for iid in [1, 2, 3]:
            (images_dir / f"{iid}.jpg").write_bytes(b"fake")
        summary = validate_eval_inputs(gt_ds, preds, images_dir=images_dir)
        assert summary.disk_verified is True

    def test_empty_images_directory_raises(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        with pytest.raises(MissingImageError):
            validate_eval_inputs(gt_ds, preds, images_dir=images_dir)

    def test_disk_check_skipped_without_images_dir(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3])
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        summary = validate_eval_inputs(gt_ds, preds, images_dir=None)
        assert summary.disk_verified is False


# ── Rule 7: Images with no GT boxes — warning not hard stop ──────────────────

class TestImagesWithNoBoxes:
    def test_warning_issued_not_exception(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3], with_boxes=False)
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = validate_eval_inputs(gt_ds, preds)
        assert any("no annotated boxes" in str(w.message).lower() for w in caught)
        assert summary.images_with_no_boxes == 3

    def test_evaluation_continues_not_stopped(self, tmp_path):
        gt_ds = _load_gt(tmp_path, [1, 2, 3], with_boxes=False)
        preds = _load_preds(tmp_path, gt_ds, [1, 2, 3])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                validate_eval_inputs(gt_ds, preds)
            except EvalInputError:
                pytest.fail("EvalInputError raised for images with no boxes — should be warning only")
