"""Unit tests for failure classification and cluster analysis."""
import pandas as pd
import pytest

from metrics.failure_classifier import classify_failures
from metrics.cluster_analyzer import cluster_failures, cluster_summary


# ── helpers ───────────────────────────────────────────────────────────────────

def _base():
    return {
        "scene_id": "s0", "profile": "baseline",
        "camera_distance": 2.5, "ambient_light": 0.8, "num_objects": 2,
        "distance_bin": "near (≤3m)", "lighting_bin": "bright (>0.7)",
        "clutter_bin": "low (1-3 obj)",
    }

def _tp():
    return {**_base(), "match_type": "tp", "iou": 0.8,
            "gt_class": "cup", "gt_tier": "easy", "gt_box_area": 400.0,
            "pred_class": "cup", "pred_confidence": 0.9, "pred_box_area": 400.0,
            "best_iou_any_class": None, "best_pred_class_at_overlap": None}

def _fn_missed():
    return {**_base(), "match_type": "fn", "iou": 0.0,
            "gt_class": "cube", "gt_tier": "hard", "gt_box_area": 100.0,
            "pred_class": None, "pred_confidence": None, "pred_box_area": None,
            "best_iou_any_class": 0.0, "best_pred_class_at_overlap": None}

def _fn_localisation():
    return {**_base(), "match_type": "fn", "iou": 0.0,
            "gt_class": "cup", "gt_tier": "easy", "gt_box_area": 400.0,
            "pred_class": None, "pred_confidence": None, "pred_box_area": None,
            "best_iou_any_class": 0.35, "best_pred_class_at_overlap": "cup"}

def _fn_wrong_class():
    return {**_base(), "match_type": "fn", "iou": 0.0,
            "gt_class": "bottle", "gt_tier": "easy", "gt_box_area": 200.0,
            "pred_class": None, "pred_confidence": None, "pred_box_area": None,
            "best_iou_any_class": 0.6, "best_pred_class_at_overlap": "cup"}

def _fp():
    return {**_base(), "match_type": "fp", "iou": 0.0,
            "gt_class": None, "gt_tier": None, "gt_box_area": None,
            "pred_class": "person", "pred_confidence": 0.4, "pred_box_area": 300.0,
            "best_iou_any_class": None, "best_pred_class_at_overlap": None}


# ── classify_failures ─────────────────────────────────────────────────────────

def test_tp_becomes_true_positive():
    df = pd.DataFrame([_tp()])
    out = classify_failures(df)
    assert out["failure_mode"].iloc[0] == "true_positive"

def test_fp_becomes_false_positive():
    df = pd.DataFrame([_fp()])
    out = classify_failures(df)
    assert out["failure_mode"].iloc[0] == "false_positive"

def test_fn_no_overlap_is_missed():
    df = pd.DataFrame([_fn_missed()])
    out = classify_failures(df)
    assert out["failure_mode"].iloc[0] == "missed_detection"

def test_fn_same_class_low_iou_is_localisation():
    df = pd.DataFrame([_fn_localisation()])
    out = classify_failures(df)
    assert out["failure_mode"].iloc[0] == "localization_error"

def test_fn_wrong_class_overlap_is_wrong_class():
    df = pd.DataFrame([_fn_wrong_class()])
    out = classify_failures(df)
    assert out["failure_mode"].iloc[0] == "wrong_class"

def test_mixed_df_all_modes_present():
    rows = [_tp(), _fp(), _fn_missed(), _fn_localisation(), _fn_wrong_class()]
    df = pd.DataFrame(rows)
    out = classify_failures(df)
    modes = set(out["failure_mode"])
    assert modes == {"true_positive", "false_positive", "missed_detection",
                     "localization_error", "wrong_class"}

def test_original_df_not_mutated():
    df = pd.DataFrame([_tp(), _fn_missed()])
    _ = classify_failures(df)
    assert "failure_mode" not in df.columns


# ── cluster_failures / cluster_summary ───────────────────────────────────────

def _make_classified_df(n=20):
    rows = []
    for i in range(n):
        r = _fn_missed() if i % 2 == 0 else _fp()
        rows.append(r)
    df = pd.DataFrame(rows)
    return classify_failures(df)

def test_cluster_adds_cluster_column():
    df = _make_classified_df(20)
    clustered = cluster_failures(df, n_clusters=3)
    assert "cluster" in clustered.columns

def test_cluster_drops_true_positives():
    rows = [_tp(), _fn_missed(), _fp()]
    df = classify_failures(pd.DataFrame(rows))
    clustered = cluster_failures(df, n_clusters=2)
    assert "true_positive" not in clustered["failure_mode"].values

def test_cluster_summary_has_expected_columns():
    df = _make_classified_df(30)
    clustered = cluster_failures(df, n_clusters=3)
    summary = cluster_summary(clustered)
    for col in ["cluster", "count", "dominant_failure_mode", "mode_pct"]:
        assert col in summary.columns

def test_cluster_summary_sorted_by_count():
    df = _make_classified_df(30)
    clustered = cluster_failures(df, n_clusters=3)
    summary = cluster_summary(clustered)
    counts = list(summary["count"])
    assert counts == sorted(counts, reverse=True)

def test_empty_df_no_crash():
    df = classify_failures(pd.DataFrame([_tp()]))
    clustered = cluster_failures(df, n_clusters=2)
    assert clustered.empty
    summary = cluster_summary(clustered)
    assert summary.empty
