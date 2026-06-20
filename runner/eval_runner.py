"""
Eval runner: iterate a generated dataset, run YOLO, build the matches DataFrame.

Inference is run at conf=0.01 so the full confidence distribution is captured
for AP computation. The caller can filter to a higher operating-point threshold
for P/R/F1 reporting.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from runner.matcher import match_scene
from scenarios.schemas import BoundingBox, Detection, GroundTruth

_INFER_CONF = 0.01   # capture full distribution; operating-point filtering is downstream


def _load_gt(path: Path) -> list[GroundTruth]:
    return [GroundTruth(**r) for r in json.loads(path.read_text())]


def _predict(model: YOLO, rgb: np.ndarray, imgsz: int = 640) -> list[Detection]:
    results = model(rgb, verbose=False, conf=_INFER_CONF, imgsz=imgsz)[0]
    dets: list[Detection] = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        dets.append(Detection(
            box=BoundingBox(
                x_min=float(xyxy[0]), y_min=float(xyxy[1]),
                x_max=float(xyxy[2]), y_max=float(xyxy[3]),
            ),
            class_id=int(box.cls),
            class_name=model.names[int(box.cls)],
            confidence=float(box.conf),
        ))
    return dets


def _add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df["distance_bin"] = pd.cut(
        df["camera_distance"],
        bins=[0, 3.0, 5.0, float("inf")],
        labels=["near (≤3m)", "mid (3-5m)", "far (>5m)"],
    )
    df["lighting_bin"] = pd.cut(
        df["ambient_light"],
        bins=[0.0, 0.25, 0.70, float("inf")],
        labels=["dark (≤0.25)", "normal (0.25-0.7)", "bright (>0.7)"],
    )
    df["clutter_bin"] = pd.cut(
        df["num_objects"],
        bins=[0, 3, 6, float("inf")],
        labels=["low (1-3 obj)", "medium (4-6 obj)", "high (7+ obj)"],
    )
    return df


class EvalRunner:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        iou_threshold: float = 0.5,
        imgsz: int = 640,
    ) -> None:
        print(f"Loading model {model_name}…", flush=True)
        self._model = YOLO(model_name)
        self._iou_threshold = iou_threshold
        self._imgsz = imgsz

    def run_dataset(self, dataset_dir: Path, verbose: bool = True) -> pd.DataFrame:
        manifest = pd.read_csv(dataset_dir / "manifest.csv")
        rows: list[dict] = []
        t0 = time.perf_counter()

        for i, (_, mrow) in enumerate(manifest.iterrows()):
            rgb = np.array(Image.open(dataset_dir / mrow["frame_path"]).convert("RGB"))
            gts = _load_gt(dataset_dir / mrow["gt_path"])
            preds = _predict(self._model, rgb, imgsz=self._imgsz)
            matches = match_scene(gts, preds, self._iou_threshold)

            scene_meta = {
                "scene_id":        mrow["scene_id"],
                "profile":         mrow["profile"],
                "camera_distance": mrow["camera_distance"],
                "camera_pitch":    mrow["camera_pitch"],
                "ambient_light":   mrow["ambient_light"],
                "num_objects":     mrow["num_objects"],
                "num_visible_gt":  mrow["num_visible"],
            }
            for m in matches:
                rows.append({**scene_meta, **m})

            if verbose and ((i + 1) % 10 == 0 or i == 0):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(manifest) - i - 1) / rate
                tp = sum(1 for r in rows if r["match_type"] == "tp")
                fn = sum(1 for r in rows if r["match_type"] == "fn")
                print(
                    f"  [{i+1:>3}/{len(manifest)}] {mrow['scene_id']}"
                    f"  preds={len(preds)}  "
                    f"cumulative TP={tp} FN={fn}  ETA {eta:.0f}s"
                )

        df = pd.DataFrame(rows)
        df = _add_bins(df)
        return df
