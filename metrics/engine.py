"""
Slice-based metrics engine.

Input:  matches DataFrame produced by EvalRunner (one row per TP/FP/FN).
Output: dict of per-slice DataFrames + overall metrics dict.

AP is computed via the 11-point PASCAL VOC interpolation across the full
confidence distribution (rows are not filtered by operating-point conf here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_OP_CONF = 0.25   # operating-point threshold for P/R/F1 (not AP)


# ── low-level ──────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall    = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def _slice_prf(df: pd.DataFrame, op_conf: float = _OP_CONF) -> dict:
    """TP/FP/FN at the operating-point confidence threshold."""
    op = df[
        (df["match_type"] == "fn") |
        (df["match_type"].isin(["tp", "fp"]) & (df["pred_confidence"] >= op_conf))
    ]
    tp = int((op["match_type"] == "tp").sum())
    fp = int((op["match_type"] == "fp").sum())
    fn = int((op["match_type"] == "fn").sum())
    return {**_prf(tp, fp, fn), "n_scenes": int(df["scene_id"].nunique())}


def _class_ap(df: pd.DataFrame, cls: str) -> float:
    """
    AP@0.5 for one class, using all predictions (not filtered by op_conf).
    total_gt = number of GT instances for this class (TP + FN rows).
    """
    total_gt = int(((df["match_type"].isin(["tp", "fn"])) & (df["gt_class"] == cls)).sum())
    if total_gt == 0:
        return 0.0

    # All detections of this class, sorted by confidence
    det = df[(df["pred_class"] == cls) & df["match_type"].isin(["tp", "fp"])].copy()
    if det.empty:
        return 0.0

    det = det.sort_values("pred_confidence", ascending=False)
    is_tp = (det["match_type"] == "tp").astype(float).values
    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(1 - is_tp)

    prec = cum_tp / (cum_tp + cum_fp)
    rec  = cum_tp / total_gt

    # Monotone envelope (standard VOC post-processing)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # 11-point interpolation
    ap = sum(prec[rec >= t].max() if (rec >= t).any() else 0.0
             for t in np.linspace(0, 1, 11)) / 11.0
    return float(ap)


# ── public API ─────────────────────────────────────────────────────────────────

def overall_metrics(df: pd.DataFrame) -> dict:
    m = _slice_prf(df)
    all_classes = df["gt_class"].dropna().unique()
    aps = {cls: _class_ap(df, cls) for cls in all_classes}
    m["map"] = float(np.mean(list(aps.values()))) if aps else float("nan")
    m["class_ap"] = aps
    return m


def sliced_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    # Dimension slices
    for dim in ("profile", "distance_bin", "lighting_bin", "clutter_bin"):
        if dim not in df.columns:
            continue
        rows = []
        for val, grp in df.groupby(dim, observed=True):
            m = _slice_prf(grp)
            m[dim] = val
            rows.append(m)
        if rows:
            tables[dim] = pd.DataFrame(rows).set_index(dim)

    # Tier slice — only meaningful for GT rows
    tier_rows = []
    for tier in ("easy", "hard"):
        grp = df[(df["gt_tier"] == tier) | (df["match_type"] == "fp")]
        # For tier slicing: FPs are class-agnostic; attribute them to both tiers
        # by splitting: FP rows contribute fully to the tier we're inspecting.
        gt_grp = df[df["gt_tier"] == tier]
        fp_grp = df[df["match_type"] == "fp"]
        combined = pd.concat([gt_grp, fp_grp])
        m = _slice_prf(combined)
        m["gt_tier"] = tier
        tier_rows.append(m)
    tables["gt_tier"] = pd.DataFrame(tier_rows).set_index("gt_tier")

    # Per-class breakdown
    class_rows = []
    for cls in sorted(df["gt_class"].dropna().unique()):
        cls_df = df[(df["gt_class"] == cls) | (df["pred_class"] == cls)]
        m = _slice_prf(cls_df)
        m["class"] = cls
        m["ap"] = _class_ap(df, cls)
        tiers = df[df["gt_class"] == cls]["gt_tier"].dropna()
        m["tier"] = tiers.iloc[0] if len(tiers) > 0 else "?"
        class_rows.append(m)
    if class_rows:
        tables["class"] = pd.DataFrame(class_rows).set_index("class")

    return tables
