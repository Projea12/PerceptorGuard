"""
Greedy bipartite matching between GT boxes and predictions for one scene.

Matching is class-aware (pred.class_name must equal gt.class_name) and
uses descending-confidence ordering so high-confidence predictions have
first pick of GT boxes — consistent with VOC/COCO eval protocol.
"""
from __future__ import annotations

from scenarios.schemas import BoundingBox, Detection, GroundTruth


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x_min, b.x_min)
    iy1 = max(a.y_min, b.y_min)
    ix2 = min(a.x_max, b.x_max)
    iy2 = min(a.y_max, b.y_max)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


def match_scene(
    gts: list[GroundTruth],
    preds: list[Detection],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """
    Return one record per GT (TP or FN) and one per unmatched prediction (FP).
    Each record carries all GT and prediction fields so rows can be concatenated
    directly into a tidy DataFrame without joins.

    FN rows also include best_iou_any_class and best_pred_class_at_overlap so
    downstream failure classifiers can distinguish missed vs localisation error
    vs wrong-class failures.
    """
    ranked = sorted(range(len(preds)), key=lambda i: preds[i].confidence, reverse=True)
    used_gt: set[int] = set()
    pred_to_gt: dict[int, int] = {}

    for pi in ranked:
        pred = preds[pi]
        best_iou, best_gi = -1.0, -1
        for gi, gt in enumerate(gts):
            if gi in used_gt:
                continue
            if gt.class_name != pred.class_name:
                continue
            v = box_iou(gt.box, pred.box)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_threshold:
            used_gt.add(best_gi)
            pred_to_gt[pi] = best_gi

    records: list[dict] = []

    # TPs
    for pi, gi in pred_to_gt.items():
        gt, pred = gts[gi], preds[pi]
        records.append({
            "match_type": "tp",
            "iou": box_iou(gt.box, pred.box),
            "gt_class": gt.class_name, "gt_tier": gt.tier, "gt_box_area": gt.box.area,
            "pred_class": pred.class_name, "pred_confidence": pred.confidence,
            "pred_box_area": pred.box.area,
            "best_iou_any_class": None, "best_pred_class_at_overlap": None,
        })

    # FNs — scan all preds (including matched) to characterise why this GT was missed
    for gi, gt in enumerate(gts):
        if gi in used_gt:
            continue
        best_any, best_cls = 0.0, None
        best_same = 0.0
        for pred in preds:
            v = box_iou(gt.box, pred.box)
            if v > best_any:
                best_any = v
                best_cls = pred.class_name
            if pred.class_name == gt.class_name and v > best_same:
                best_same = v
        records.append({
            "match_type": "fn",
            "iou": 0.0,
            "gt_class": gt.class_name, "gt_tier": gt.tier, "gt_box_area": gt.box.area,
            "pred_class": None, "pred_confidence": None, "pred_box_area": None,
            "best_iou_any_class": best_any, "best_pred_class_at_overlap": best_cls,
        })

    # FPs — unmatched predictions
    matched_preds = set(pred_to_gt.keys())
    for pi, pred in enumerate(preds):
        if pi not in matched_preds:
            records.append({
                "match_type": "fp",
                "iou": 0.0,
                "gt_class": None, "gt_tier": None, "gt_box_area": None,
                "pred_class": pred.class_name, "pred_confidence": pred.confidence,
                "pred_box_area": pred.box.area,
                "best_iou_any_class": None, "best_pred_class_at_overlap": None,
            })

    return records
