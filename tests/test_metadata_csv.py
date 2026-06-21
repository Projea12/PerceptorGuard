"""Tests for optional metadata CSV ingestion."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from ingestion.metadata_csv import (
    enrich_matches_with_metadata,
    load_metadata_csv,
    slice_columns_from_metadata,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _matches(scene_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"scene_id": scene_ids, "match_type": ["tp"] * len(scene_ids)})


# ── load_metadata_csv: happy path ─────────────────────────────────────────────

class TestLoadMetadataCsvHappyPath:
    def test_returns_dict(self, tmp_path):
        p = _write_csv(tmp_path, "meta.csv", "filename,weather\nimg.jpg,rain\n")
        result = load_metadata_csv(p)
        assert isinstance(result, dict)

    def test_keyed_by_filename(self, tmp_path):
        p = _write_csv(tmp_path, "meta.csv", "filename,weather\nimg.jpg,rain\n")
        result = load_metadata_csv(p)
        assert "img.jpg" in result

    def test_slice_values_parsed(self, tmp_path):
        p = _write_csv(tmp_path, "meta.csv", "filename,weather\nimg.jpg,rain\n")
        result = load_metadata_csv(p)
        assert result["img.jpg"]["weather"] == "rain"

    def test_multiple_slice_columns(self, tmp_path):
        content = "filename,weather,sensor,distance\nimg.jpg,rain,camera,near\n"
        p = _write_csv(tmp_path, "meta.csv", content)
        result = load_metadata_csv(p)
        assert result["img.jpg"] == {"weather": "rain", "sensor": "camera", "distance": "near"}

    def test_multiple_images(self, tmp_path):
        content = "filename,weather\na.jpg,rain\nb.jpg,sunny\n"
        p = _write_csv(tmp_path, "meta.csv", content)
        result = load_metadata_csv(p)
        assert result["a.jpg"]["weather"] == "rain"
        assert result["b.jpg"]["weather"] == "sunny"

    def test_custom_id_column(self, tmp_path):
        content = "scene_id,weather\nimg.jpg,rain\n"
        p = _write_csv(tmp_path, "meta.csv", content)
        result = load_metadata_csv(p, id_column="scene_id")
        assert "img.jpg" in result

    def test_whitespace_stripped_from_values(self, tmp_path):
        content = "filename,weather\n  img.jpg , rain  \n"
        p = _write_csv(tmp_path, "meta.csv", content)
        result = load_metadata_csv(p)
        assert "img.jpg" in result
        assert result["img.jpg"]["weather"] == "rain"

    def test_whitespace_stripped_from_headers(self, tmp_path):
        content = "filename , weather , sensor\nimg.jpg,rain,camera\n"
        p = _write_csv(tmp_path, "meta.csv", content)
        result = load_metadata_csv(p)
        assert "weather" in result["img.jpg"]
        assert "sensor" in result["img.jpg"]

    def test_nan_value_becomes_empty_string(self, tmp_path):
        content = "filename,weather,sensor\nimg.jpg,rain,\n"
        p = _write_csv(tmp_path, "meta.csv", content)
        result = load_metadata_csv(p)
        assert result["img.jpg"]["sensor"] == ""


# ── load_metadata_csv: warnings and edge cases ────────────────────────────────

class TestLoadMetadataCsvEdgeCases:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metadata_csv(tmp_path / "missing.csv")

    def test_missing_id_column_raises(self, tmp_path):
        p = _write_csv(tmp_path, "meta.csv", "name,weather\nimg.jpg,rain\n")
        with pytest.raises(ValueError, match="filename"):
            load_metadata_csv(p)

    def test_error_message_lists_available_columns(self, tmp_path):
        p = _write_csv(tmp_path, "meta.csv", "name,weather\nimg.jpg,rain\n")
        with pytest.raises(ValueError, match="name"):
            load_metadata_csv(p)

    def test_no_slice_columns_warns(self, tmp_path):
        p = _write_csv(tmp_path, "meta.csv", "filename\nimg.jpg\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_metadata_csv(p)
        assert result == {"img.jpg": {}}
        assert any("no columns" in str(x.message).lower() for x in w)

    def test_duplicate_filename_warns(self, tmp_path):
        content = "filename,weather\nimg.jpg,rain\nimg.jpg,sunny\n"
        p = _write_csv(tmp_path, "meta.csv", content)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_metadata_csv(p)
        assert any("duplicate" in str(x.message).lower() for x in w)

    def test_duplicate_keeps_first(self, tmp_path):
        content = "filename,weather\nimg.jpg,rain\nimg.jpg,sunny\n"
        p = _write_csv(tmp_path, "meta.csv", content)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_metadata_csv(p)
        assert result["img.jpg"]["weather"] == "rain"


# ── enrich_matches_with_metadata ──────────────────────────────────────────────

class TestEnrichMatchesWithMetadata:
    def _meta(self) -> dict:
        return {
            "a.jpg": {"weather": "rain", "sensor": "camera"},
            "b.jpg": {"weather": "sunny", "sensor": "lidar"},
        }

    def test_adds_metadata_columns(self):
        df = _matches(["a.jpg", "b.jpg"])
        result = enrich_matches_with_metadata(df, self._meta())
        assert "weather" in result.columns
        assert "sensor" in result.columns

    def test_correct_values_mapped(self):
        df = _matches(["a.jpg", "b.jpg"])
        result = enrich_matches_with_metadata(df, self._meta())
        assert result.loc[result["scene_id"] == "a.jpg", "weather"].iloc[0] == "rain"
        assert result.loc[result["scene_id"] == "b.jpg", "sensor"].iloc[0] == "lidar"

    def test_missing_scene_id_gets_fill_value(self):
        df = _matches(["c.jpg"])
        result = enrich_matches_with_metadata(df, self._meta())
        assert result.loc[result["scene_id"] == "c.jpg", "weather"].iloc[0] == "unknown"

    def test_custom_fill_value(self):
        df = _matches(["c.jpg"])
        result = enrich_matches_with_metadata(df, self._meta(), fill_value="n/a")
        assert result.loc[result["scene_id"] == "c.jpg", "weather"].iloc[0] == "n/a"

    def test_empty_metadata_returns_copy(self):
        df = _matches(["a.jpg"])
        result = enrich_matches_with_metadata(df, {})
        assert list(result.columns) == list(df.columns)

    def test_does_not_mutate_input_df(self):
        df = _matches(["a.jpg"])
        original_cols = list(df.columns)
        enrich_matches_with_metadata(df, self._meta())
        assert list(df.columns) == original_cols

    def test_high_missing_rate_warns(self):
        # Only 1 of 10 rows has metadata → 90% missing → warn
        df = _matches(["a.jpg"] + ["x.jpg"] * 9)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enrich_matches_with_metadata(df, self._meta())
        assert any("no metadata" in str(x.message).lower() or "have no metadata" in str(x.message).lower() for x in w)

    def test_low_missing_rate_no_warning(self):
        # All rows have metadata → no warning
        df = _matches(["a.jpg", "b.jpg"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enrich_matches_with_metadata(df, self._meta())
        coverage_warnings = [x for x in w if "no metadata" in str(x.message).lower()
                             or "have no metadata" in str(x.message).lower()]
        assert len(coverage_warnings) == 0

    def test_multiple_rows_same_image(self):
        df = _matches(["a.jpg", "a.jpg", "a.jpg"])
        result = enrich_matches_with_metadata(df, self._meta())
        assert (result["weather"] == "rain").all()


# ── slice_columns_from_metadata ───────────────────────────────────────────────

class TestSliceColumnsFromMetadata:
    def test_returns_column_names(self):
        meta = {"img.jpg": {"weather": "rain", "sensor": "camera"}}
        assert set(slice_columns_from_metadata(meta)) == {"weather", "sensor"}

    def test_empty_metadata_returns_empty(self):
        assert slice_columns_from_metadata({}) == []

    def test_order_stable(self):
        meta = {"img.jpg": {"a": "1", "b": "2", "c": "3"}}
        cols = slice_columns_from_metadata(meta)
        assert cols == ["a", "b", "c"]
