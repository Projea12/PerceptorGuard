import pytest
from scenarios.schemas import BoundingBox, Detection, GroundTruth
from runner.matcher import box_iou, match_scene


def _bb(x1, y1, x2, y2):
    return BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

def _gt(cls, x1, y1, x2, y2, tier="easy"):
    return GroundTruth(box=_bb(x1,y1,x2,y2), class_id=0,
                       class_name=cls, tier=tier, scene_id="s0")

def _det(cls, x1, y1, x2, y2, conf=0.9):
    return Detection(box=_bb(x1,y1,x2,y2), class_id=0,
                     class_name=cls, confidence=conf)


# ── box_iou ───────────────────────────────────────────────────────────────────

def test_iou_perfect():
    b = _bb(0, 0, 10, 10)
    assert box_iou(b, b) == pytest.approx(1.0)

def test_iou_no_overlap():
    assert box_iou(_bb(0,0,5,5), _bb(6,6,10,10)) == pytest.approx(0.0)

def test_iou_half_overlap():
    # Two 10×10 boxes overlapping by 5×10 = 50; union = 150
    v = box_iou(_bb(0,0,10,10), _bb(5,0,15,10))
    assert v == pytest.approx(50/150)

def test_iou_symmetric():
    a, b = _bb(0,0,8,8), _bb(4,4,12,12)
    assert box_iou(a, b) == pytest.approx(box_iou(b, a))


# ── match_scene ───────────────────────────────────────────────────────────────

def test_perfect_match_is_tp():
    gts  = [_gt("cup", 0,0,100,100)]
    preds = [_det("cup", 0,0,100,100)]
    recs = match_scene(gts, preds)
    assert len(recs) == 1
    assert recs[0]["match_type"] == "tp"
    assert recs[0]["iou"] == pytest.approx(1.0)

def test_class_mismatch_produces_fp_and_fn():
    gts  = [_gt("cup",  0,0,100,100)]
    preds = [_det("bowl", 0,0,100,100)]   # wrong class
    recs = match_scene(gts, preds)
    types = {r["match_type"] for r in recs}
    assert "fp" in types and "fn" in types
    assert "tp" not in types

def test_low_iou_produces_fp_and_fn():
    gts  = [_gt("cup", 0,0,10,10)]
    preds = [_det("cup", 50,50,100,100)]   # no overlap
    recs = match_scene(gts, preds)
    types = {r["match_type"] for r in recs}
    assert "fp" in types and "fn" in types

def test_unmatched_prediction_is_fp():
    gts  = []
    preds = [_det("cup", 0,0,50,50)]
    recs = match_scene(gts, preds)
    assert len(recs) == 1 and recs[0]["match_type"] == "fp"

def test_unmatched_gt_is_fn():
    gts  = [_gt("cup", 0,0,100,100)]
    preds = []
    recs = match_scene(gts, preds)
    assert len(recs) == 1 and recs[0]["match_type"] == "fn"

def test_high_conf_pred_wins_gt_over_low_conf():
    # Two predictions for the same GT box; high-confidence one should match
    gt = _gt("cup", 0,0,100,100)
    p_high = _det("cup", 0,0,100,100, conf=0.9)
    p_low  = _det("cup", 0,0,100,100, conf=0.3)
    recs = match_scene([gt], [p_high, p_low])
    tps = [r for r in recs if r["match_type"] == "tp"]
    assert len(tps) == 1
    assert tps[0]["pred_confidence"] == pytest.approx(0.9)

def test_multiple_objects_correct_counts():
    gts  = [_gt("cup", 0,0,50,50), _gt("bottle", 60,60,100,100)]
    preds = [_det("cup", 0,0,50,50), _det("bottle", 60,60,100,100)]
    recs = match_scene(gts, preds)
    types = [r["match_type"] for r in recs]
    assert types.count("tp") == 2
    assert types.count("fp") == 0
    assert types.count("fn") == 0

def test_iou_threshold_respected():
    gts  = [_gt("cup", 0,0,100,100)]
    # (0,0,100,40) ∩ (0,0,100,100) = 4000; union = 10000; IoU = 0.40 < 0.5
    preds = [_det("cup", 0,0,100,40)]
    recs = match_scene(gts, preds, iou_threshold=0.5)
    types = {r["match_type"] for r in recs}
    assert "tp" not in types
    # same box passes at a lower threshold (0.3)
    recs2 = match_scene(gts, preds, iou_threshold=0.3)
    assert any(r["match_type"] == "tp" for r in recs2)
