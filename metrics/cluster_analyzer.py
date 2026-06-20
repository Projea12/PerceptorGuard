"""
Cluster failures with KMeans to surface natural groupings in scenario space.

Feature vector per failure row:
  camera_distance, ambient_light, num_objects, gt_box_area (log),
  failure_mode (encoded), gt_tier (encoded), profile (encoded)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


_FAILURE_MODES = ["missed_detection", "localization_error", "wrong_class", "false_positive"]
_N_CLUSTERS = 5


def cluster_failures(
    df: pd.DataFrame,
    n_clusters: int = _N_CLUSTERS,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add a ``cluster`` column to failure rows.  Returns a copy of *df*
    with only failure rows (true_positive rows are dropped).
    """
    failures = df[df["failure_mode"] != "true_positive"].copy()
    if len(failures) == 0:
        return failures

    le_mode    = LabelEncoder().fit(_FAILURE_MODES)
    le_tier    = LabelEncoder().fit(["easy", "hard", "clutter"])
    le_profile = LabelEncoder().fit(
        ["baseline", "crowded", "dark", "far", "occluded", "steep"]
    )

    def _encode_col(series: pd.Series, le: LabelEncoder) -> np.ndarray:
        safe = series.fillna("missed_detection")  # fallback for unknowns
        try:
            return le.transform(safe).astype(float)
        except ValueError:
            # unseen labels → map to 0
            return np.array([
                le.transform([v])[0] if v in le.classes_ else 0
                for v in safe
            ], dtype=float)

    box_area = np.log1p(failures["gt_box_area"].fillna(0).clip(lower=0))

    X = np.column_stack([
        failures["camera_distance"].fillna(failures["camera_distance"].median()),
        failures["ambient_light"].fillna(0.5),
        failures["num_objects"].fillna(1),
        box_area,
        _encode_col(failures["failure_mode"], le_mode),
        _encode_col(failures.get("gt_tier", pd.Series(["easy"] * len(failures))), le_tier),
        _encode_col(failures["profile"], le_profile),
    ])

    X_scaled = StandardScaler().fit_transform(X)
    k = min(n_clusters, len(failures))
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X_scaled)
    failures["cluster"] = labels
    return failures


def cluster_summary(clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per cluster with dominant failure mode, mean scenario
    params, and count.
    """
    if clustered.empty:
        return pd.DataFrame()

    rows = []
    for cid, grp in clustered.groupby("cluster"):
        dominant_mode = grp["failure_mode"].value_counts().idxmax()
        rows.append({
            "cluster": cid,
            "count": len(grp),
            "dominant_failure_mode": dominant_mode,
            "mode_pct": grp["failure_mode"].value_counts(normalize=True).iloc[0],
            "mean_camera_distance": grp["camera_distance"].mean(),
            "mean_ambient_light": grp["ambient_light"].mean(),
            "mean_num_objects": grp["num_objects"].mean(),
            "mean_gt_box_area": grp["gt_box_area"].mean(),
            "top_profile": grp["profile"].value_counts().idxmax()
            if "profile" in grp.columns else "?",
            "top_tier": grp["gt_tier"].value_counts().dropna().idxmax()
            if "gt_tier" in grp.columns and grp["gt_tier"].notna().any() else "?",
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
