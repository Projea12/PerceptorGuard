"""Load an optional user-provided metadata CSV and merge it into matches.

The CSV must have one column that identifies each image (default: "filename").
Every other column becomes a slice dimension automatically.

Example CSV:
    filename,   weather, sensor,  distance, time_of_day
    img001.jpg, rain,    camera,  near,     night
    img002.jpg, sunny,   lidar,   far,      day

Usage:
    meta = load_metadata_csv("metadata.csv")
    matches_df = enrich_matches_with_metadata(matches_df, meta)
    # → adds columns: weather, sensor, distance, time_of_day
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd


def load_metadata_csv(
    path: Path,
    id_column: str = "filename",
) -> dict[str, dict[str, str]]:
    """Parse a metadata CSV and return a lookup keyed by image filename.

    Args:
        path: Path to the CSV file.
        id_column: Column name that identifies each image. The value must match
                   the scene_id / filename used in the GT and matches DataFrame.

    Returns:
        {filename -> {column_name -> value}} for every non-id column.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if id_column is not present in the CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if id_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"id_column '{id_column}' not found in {path}. "
            f"Available columns: {available}"
        )

    slice_cols = [c for c in df.columns if c != id_column]
    if not slice_cols:
        warnings.warn(
            f"Metadata CSV {path} has no columns besides '{id_column}'. "
            "No slice dimensions will be added.",
            stacklevel=2,
        )

    duplicates = df[id_column][df[id_column].duplicated()].tolist()
    if duplicates:
        warnings.warn(
            f"Metadata CSV has {len(duplicates)} duplicate {id_column} value(s): "
            f"{duplicates[:5]}{'...' if len(duplicates) > 5 else ''}. "
            "Only the first occurrence is kept.",
            stacklevel=2,
        )
        df = df.drop_duplicates(subset=[id_column], keep="first")

    result: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = str(row[id_column]).strip()
        result[key] = {
            col: str(row[col]).strip() if pd.notna(row[col]) else ""
            for col in slice_cols
        }

    return result


def enrich_matches_with_metadata(
    matches: pd.DataFrame,
    metadata: dict[str, dict[str, str]],
    fill_value: str = "unknown",
) -> pd.DataFrame:
    """Add metadata columns to a matches DataFrame keyed on scene_id.

    Rows with no metadata entry receive fill_value for every metadata column.
    A warning is emitted if more than 20 % of rows have no coverage — usually
    means the CSV uses a different filename format than the matches.

    Args:
        matches: DataFrame with a 'scene_id' column.
        metadata: Output of load_metadata_csv().
        fill_value: Value to use for images not covered by the metadata CSV.

    Returns:
        New DataFrame with one additional column per metadata field.
    """
    if not metadata:
        return matches.copy()

    slice_cols = list(next(iter(metadata.values())).keys())
    if not slice_cols:
        return matches.copy()

    df = matches.copy()

    for col in slice_cols:
        df[col] = df["scene_id"].map(
            lambda sid, c=col: metadata.get(sid, {}).get(c, fill_value)
        )

    missing_mask = ~df["scene_id"].isin(metadata.keys())
    missing_frac = missing_mask.mean()
    if missing_frac > 0.20:
        n_missing = int(missing_mask.sum())
        sample = df.loc[missing_mask, "scene_id"].unique()[:3].tolist()
        warnings.warn(
            f"{n_missing} rows ({missing_frac:.0%}) have no metadata entry. "
            f"Sample scene_ids with no match: {sample}. "
            "Check that the metadata CSV id_column values match your scene_ids.",
            stacklevel=2,
        )

    return df


def slice_columns_from_metadata(metadata: dict[str, dict[str, str]]) -> list[str]:
    """Return the list of slice column names this metadata provides."""
    if not metadata:
        return []
    return list(next(iter(metadata.values())).keys())
