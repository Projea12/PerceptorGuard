"""Tests for the class mapping step."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from ingestion.class_map import ClassMap, load_or_create
from scenarios.schemas import BoundingBox, GroundTruth


# ── helpers ───────────────────────────────────────────────────────────────────

_COCO_CLASSES = {
    "car", "person", "motorcycle", "bicycle", "truck",
    "bus", "traffic light", "sports ball", "cup", "bottle",
}

_BOX = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)


def _gt(class_name: str, scene_id: str = "img.jpg") -> GroundTruth:
    return GroundTruth(
        box=_BOX, class_id=1, class_name=class_name,
        tier="user", scene_id=scene_id,
    )


def _write_map(tmp_path: Path, mappings: dict) -> Path:
    p = tmp_path / "class_map.yml"
    p.write_text(yaml.dump({"mappings": mappings}))
    return p


# ── ClassMap.resolve ──────────────────────────────────────────────────────────

class TestClassMapResolve:
    def test_known_class(self):
        cm = ClassMap(mapping={"vehicle": "car"})
        assert cm.resolve("vehicle") == "car"

    def test_unmapped_class_returns_none(self):
        cm = ClassMap(mapping={"vehicle": "car"})
        assert cm.resolve("forklift") is None

    def test_explicit_null_mapping(self):
        cm = ClassMap(mapping={"forklift": None})
        assert cm.resolve("forklift") is None


# ── ClassMap.apply_to_gts ─────────────────────────────────────────────────────

class TestApplyToGts:
    def test_remaps_class_name(self):
        cm = ClassMap(mapping={"vehicle": "car"})
        gts = {"img.jpg": [_gt("vehicle")]}
        result = cm.apply_to_gts(gts)
        assert result["img.jpg"][0].class_name == "car"

    def test_exact_match_passthrough(self):
        cm = ClassMap(mapping={"car": "car"})
        gts = {"img.jpg": [_gt("car")]}
        result = cm.apply_to_gts(gts)
        assert result["img.jpg"][0].class_name == "car"

    def test_unmapped_gt_dropped(self):
        cm = ClassMap(mapping={"forklift": None})
        gts = {"img.jpg": [_gt("forklift")]}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = cm.apply_to_gts(gts)
        assert result == {}

    def test_dropped_class_emits_warning(self):
        cm = ClassMap(mapping={"forklift": None})
        gts = {"img.jpg": [_gt("forklift")]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cm.apply_to_gts(gts)
        assert any("forklift" in str(x.message) for x in w)

    def test_mixed_mapped_and_unmapped(self):
        cm = ClassMap(mapping={"car": "car", "forklift": None})
        gts = {"img.jpg": [_gt("car"), _gt("forklift")]}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = cm.apply_to_gts(gts)
        assert len(result["img.jpg"]) == 1
        assert result["img.jpg"][0].class_name == "car"

    def test_many_to_one_mapping(self):
        """sedan, SUV, truck all collapse to car."""
        cm = ClassMap(mapping={"sedan": "car", "SUV": "car", "truck": "car"})
        gts = {"img.jpg": [_gt("sedan"), _gt("SUV"), _gt("truck")]}
        result = cm.apply_to_gts(gts)
        assert all(g.class_name == "car" for g in result["img.jpg"])

    def test_multiple_images_all_remapped(self):
        cm = ClassMap(mapping={"person": "person", "vehicle": "car"})
        gts = {
            "a.jpg": [_gt("person", "a.jpg")],
            "b.jpg": [_gt("vehicle", "b.jpg")],
        }
        result = cm.apply_to_gts(gts)
        assert result["a.jpg"][0].class_name == "person"
        assert result["b.jpg"][0].class_name == "car"

    def test_other_gt_fields_preserved(self):
        """Remapping class_name must not change box, tier, scene_id, etc."""
        cm = ClassMap(mapping={"vehicle": "car"})
        original = _gt("vehicle", scene_id="scene42.jpg")
        result = cm.apply_to_gts({"scene42.jpg": [original]})
        remapped = result["scene42.jpg"][0]
        assert remapped.box == _BOX
        assert remapped.tier == "user"
        assert remapped.scene_id == "scene42.jpg"
        assert remapped.class_id == 1

    def test_image_with_all_dropped_excluded_from_result(self):
        cm = ClassMap(mapping={"forklift": None})
        gts = {
            "a.jpg": [_gt("forklift")],
            "b.jpg": [_gt("forklift")],
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = cm.apply_to_gts(gts)
        assert result == {}

    def test_warning_lists_all_dropped_classes(self):
        cm = ClassMap(mapping={"forklift": None, "crane": None})
        gts = {"img.jpg": [_gt("forklift"), _gt("crane")]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cm.apply_to_gts(gts)
        warning_text = " ".join(str(x.message) for x in w)
        assert "forklift" in warning_text
        assert "crane" in warning_text


# ── load_or_create: auto-suggest (no existing file) ──────────────────────────

class TestAutoSuggest:
    def test_exact_case_insensitive_match(self, tmp_path):
        user = {"Car", "PERSON"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = load_or_create(user, _COCO_CLASSES, tmp_path / "map.yml")
        assert cm.resolve("Car") == "car"
        assert cm.resolve("PERSON") == "person"

    def test_fuzzy_synonym_match(self, tmp_path):
        user = {"vehicle"}  # close enough to "car"? depends on threshold
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = load_or_create(user, {"car", "truck"}, tmp_path / "map.yml")
        # vehicle doesn't fuzzy-match car well — should be None
        assert cm.resolve("vehicle") is None

    def test_fuzzy_close_match(self, tmp_path):
        """'motorcicle' is a typo for 'motorcycle' — should match."""
        user = {"motorcicle"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = load_or_create(user, _COCO_CLASSES, tmp_path / "map.yml")
        assert cm.resolve("motorcicle") == "motorcycle"

    def test_unresolvable_class_maps_to_none(self, tmp_path):
        user = {"forklift"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = load_or_create(user, _COCO_CLASSES, tmp_path / "map.yml")
        assert cm.resolve("forklift") is None

    def test_config_file_written(self, tmp_path):
        user = {"car"}
        p = tmp_path / "map.yml"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_or_create(user, _COCO_CLASSES, p)
        assert p.exists()

    def test_written_file_is_valid_yaml(self, tmp_path):
        user = {"car", "person"}
        p = tmp_path / "map.yml"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_or_create(user, _COCO_CLASSES, p)
        parsed = yaml.safe_load(p.read_text())
        assert "mappings" in parsed

    def test_written_file_contains_all_user_classes(self, tmp_path):
        user = {"car", "forklift"}
        p = tmp_path / "map.yml"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_or_create(user, _COCO_CLASSES, p)
        parsed = yaml.safe_load(p.read_text())
        assert "car" in parsed["mappings"]
        assert "forklift" in parsed["mappings"]

    def test_unresolved_class_emits_warning(self, tmp_path):
        user = {"forklift"}
        p = tmp_path / "map.yml"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_or_create(user, _COCO_CLASSES, p)
        assert any("forklift" in str(x.message) for x in w)

    def test_suggested_mapping_emits_warning(self, tmp_path):
        user = {"car"}
        p = tmp_path / "map.yml"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_or_create(user, _COCO_CLASSES, p)
        assert any("auto-suggested" in str(x.message).lower() or "→" in str(x.message) for x in w)

    def test_parent_dirs_created(self, tmp_path):
        user = {"car"}
        p = tmp_path / "deep" / "nested" / "map.yml"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            load_or_create(user, _COCO_CLASSES, p)
        assert p.exists()


# ── load_or_create: loading existing file ────────────────────────────────────

class TestLoadExisting:
    def test_loads_from_file(self, tmp_path):
        p = _write_map(tmp_path, {"vehicle": "car", "person": "person"})
        cm = load_or_create({"vehicle", "person"}, _COCO_CLASSES, p)
        assert cm.resolve("vehicle") == "car"
        assert cm.resolve("person") == "person"

    def test_null_preserved_from_file(self, tmp_path):
        p = _write_map(tmp_path, {"forklift": None})
        cm = load_or_create({"forklift"}, _COCO_CLASSES, p)
        assert cm.resolve("forklift") is None

    def test_new_class_not_in_file_warns(self, tmp_path):
        p = _write_map(tmp_path, {"car": "car"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_or_create({"car", "forklift"}, _COCO_CLASSES, p)
        assert any("forklift" in str(x.message) for x in w)

    def test_new_class_not_in_file_maps_to_none(self, tmp_path):
        p = _write_map(tmp_path, {"car": "car"})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = load_or_create({"car", "forklift"}, _COCO_CLASSES, p)
        assert cm.resolve("forklift") is None

    def test_does_not_overwrite_existing_file(self, tmp_path):
        p = _write_map(tmp_path, {"vehicle": "car"})
        mtime_before = p.stat().st_mtime
        load_or_create({"vehicle"}, _COCO_CLASSES, p)
        assert p.stat().st_mtime == mtime_before

    def test_file_takes_precedence_over_auto_suggest(self, tmp_path):
        """If user manually corrected vehicle→truck, that should be respected."""
        p = _write_map(tmp_path, {"vehicle": "truck"})
        cm = load_or_create({"vehicle"}, _COCO_CLASSES, p)
        assert cm.resolve("vehicle") == "truck"

    def test_empty_mappings_section_ok(self, tmp_path):
        p = tmp_path / "map.yml"
        p.write_text(yaml.dump({"mappings": {}}))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = load_or_create({"car"}, _COCO_CLASSES, p)
        assert cm.resolve("car") is None

    def test_missing_mappings_key_ok(self, tmp_path):
        p = tmp_path / "map.yml"
        p.write_text(yaml.dump({}))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = load_or_create({"car"}, _COCO_CLASSES, p)
        assert cm.resolve("car") is None
