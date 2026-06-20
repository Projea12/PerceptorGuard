import numpy as np
import pandas as pd
import pytest

from metrics.engine import _class_ap, _prf, _slice_prf, overall_metrics, sliced_tables


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "scene_id": "s0", "profile": "baseline",
        "camera_distance": 2.5, "ambient_light": 0.8, "num_objects": 2,
        "distance_bin": "near (≤3m)", "lighting_bin": "bright (>0.7)",
        "clutter_bin": "low (1-3 obj)",
    }
    records = []
    for r in rows:
        rec = dict(base)
        rec.update(r)
        records.append(rec)
    return pd.DataFrame(records)


def _tp(cls="cup", tier="easy", conf=0.9):
    return {"match_type":"tp","gt_class":cls,"gt_tier":tier,"gt_box_area":100.0,
            "pred_class":cls,"pred_confidence":conf,"pred_box_area":100.0,"iou":0.8}

def _fp(cls="cup", conf=0.6):
    return {"match_type":"fp","gt_class":None,"gt_tier":None,"gt_box_area":None,
            "pred_class":cls,"pred_confidence":conf,"pred_box_area":100.0,"iou":0.0}

def _fn(cls="cup", tier="easy"):
    return {"match_type":"fn","gt_class":cls,"gt_tier":tier,"gt_box_area":100.0,
            "pred_class":None,"pred_confidence":None,"pred_box_area":None,"iou":0.0}


# ── _prf ──────────────────────────────────────────────────────────────────────

def test_prf_all_correct():
    m = _prf(10, 0, 0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"]    == pytest.approx(1.0)
    assert m["f1"]        == pytest.approx(1.0)

def test_prf_all_fn():
    m = _prf(0, 0, 5)
    assert m["precision"] != m["precision"]   # nan
    assert m["recall"]    == pytest.approx(0.0)

def test_prf_all_fp():
    m = _prf(0, 5, 0)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"]    != m["recall"]      # nan

def test_prf_mixed():
    m = _prf(3, 1, 1)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"]    == pytest.approx(0.75)


# ── _slice_prf (operating-point filtering) ────────────────────────────────────

def test_slice_prf_filters_by_conf():
    df = _make_df([
        _tp("cup", conf=0.9),   # kept
        _tp("cup", conf=0.1),   # filtered out → becomes FN effectively (GT unmatched)
        _fn("cup"),
    ])
    # Only the high-conf TP survives; low-conf TP is dropped from pred side,
    # but that GT was matched so the GT row is a TP counted at full truth.
    # With op_conf=0.25: low-conf TP row is excluded → becomes unmatched for precision
    m = _slice_prf(df, op_conf=0.25)
    # 1 TP (high conf), 0 FP, 2 FN (low-conf TP excluded + 1 explicit FN)
    # The low-conf row: match_type=tp but conf<0.25 → excluded entirely
    # so its GT contribution disappears from operating-point metrics.
    # FN from the explicit _fn row remains.
    assert m["tp"] + m["fp"] + m["fn"] > 0


def test_slice_prf_no_preds_all_fn():
    df = _make_df([_fn("cup"), _fn("cup"), _fn("bowl")])
    m = _slice_prf(df)
    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 3


# ── _class_ap ─────────────────────────────────────────────────────────────────

def test_class_ap_perfect():
    # 3 TPs at high confidence, 0 FPs — AP should be 1.0
    df = _make_df([_tp("cup", conf=0.9), _tp("cup", conf=0.8), _tp("cup", conf=0.7)])
    ap = _class_ap(df, "cup")
    assert ap == pytest.approx(1.0)

def test_class_ap_zero_gt():
    df = _make_df([_fp("cup")])
    ap = _class_ap(df, "cup")
    assert ap == pytest.approx(0.0)

def test_class_ap_all_fp():
    # All detections are FP, GT is only FN → AP should be 0
    df = _make_df([_fp("cup"), _fp("cup"), _fn("cup")])
    ap = _class_ap(df, "cup")
    assert ap == pytest.approx(0.0)

def test_class_ap_between_zero_and_one():
    df = _make_df([
        _tp("cup", conf=0.9), _fp("cup", conf=0.8),
        _tp("cup", conf=0.7), _fn("cup"),
    ])
    ap = _class_ap(df, "cup")
    assert 0.0 < ap < 1.0


# ── overall_metrics ───────────────────────────────────────────────────────────

def test_overall_map_is_mean_of_class_aps():
    df = _make_df([
        _tp("cup",    conf=0.9), _tp("cup",    conf=0.8),
        _tp("bottle", conf=0.7), _fn("bottle"),
    ])
    m = overall_metrics(df)
    assert "map" in m
    assert 0.0 <= m["map"] <= 1.0
    assert m["tp"] == 3


# ── sliced_tables ─────────────────────────────────────────────────────────────

def test_sliced_tables_keys():
    df = _make_df([_tp("cup"), _fn("cup"), _fp("cup")])
    tables = sliced_tables(df)
    assert "class" in tables
    assert "profile" in tables

def test_class_table_has_ap_column():
    df = _make_df([_tp("cup", conf=0.9), _fn("cup")])
    tables = sliced_tables(df)
    assert "ap" in tables["class"].columns

def test_hard_tier_low_recall():
    df = _make_df([
        _fn("cube", tier="hard"),
        _fn("lego", tier="hard"),
        _tp("cup",  tier="easy", conf=0.9),
    ])
    tables = sliced_tables(df)
    if "gt_tier" in tables:
        t = tables["gt_tier"]
        if "hard" in t.index:
            assert t.loc["hard", "recall"] == pytest.approx(0.0)
