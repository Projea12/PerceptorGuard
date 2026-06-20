"""
Classify each row in a matches DataFrame into a named failure mode.

failure_mode values
-------------------
true_positive       — correctly detected (not a failure)
missed_detection    — FN: model produced nothing near this GT
localization_error  — FN: right class nearby but IoU below threshold
wrong_class         — FN: something overlaps the GT but wrong class label
false_positive      — FP: hallucinated detection, no GT matches
"""
from __future__ import annotations

import pandas as pd

_IOU_OVERLAP_FLOOR = 0.1   # min IoU to call a box "nearby"


def classify_failures(df: pd.DataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
    """
    Add a ``failure_mode`` column to *df* (a copy).  Input is the matches
    DataFrame produced by EvalRunner / match_scene.
    """
    df = df.copy()
    modes = []

    for _, row in df.iterrows():
        mt = row["match_type"]
        if mt == "tp":
            modes.append("true_positive")
            continue
        if mt == "fp":
            modes.append("false_positive")
            continue

        # FN — use the sub-threshold overlap fields added by the updated matcher
        best_iou = row.get("best_iou_any_class", None)
        best_cls = row.get("best_pred_class_at_overlap", None)
        gt_cls   = row.get("gt_class", None)

        if best_iou is None or (best_iou != best_iou):  # None or NaN
            modes.append("missed_detection")
            continue

        if best_iou < _IOU_OVERLAP_FLOOR:
            modes.append("missed_detection")
        elif best_cls == gt_cls:
            # same-class prediction overlaps but IoU below threshold
            modes.append("localization_error")
        else:
            # something overlaps (possibly at high IoU) but wrong class
            modes.append("wrong_class")

    df["failure_mode"] = modes
    return df
