"""
Draw GT and prediction boxes on raw scene frames to visualise failures.

Strategy (no re-inference required):
  - GT boxes come from the ground_truth.json fixtures (exact coordinates).
  - Matched TPs are outlined green, missed FNs are outlined red with "MISSED".
  - FP predictions can't be drawn without re-inference; their count is labelled
    in the image caption only.
"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


class AnnotatedScene(NamedTuple):
    scene_id: str
    fn_count: int
    fp_count: int
    image_b64: str   # base64-encoded PNG for direct HTML embedding


def _load_gt(gt_path: Path) -> list[dict]:
    return json.loads(gt_path.read_text())


def _to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _font(size: int = 11):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _draw_box(
    draw: ImageDraw.ImageDraw,
    x1: float, y1: float, x2: float, y2: float,
    label: str,
    color: str,
    width: int = 2,
) -> None:
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    font = _font(11)
    tw = len(label) * 6 + 4
    th = 14
    draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
    draw.text((x1 + 2, y1 - th + 1), label, fill="white", font=font)


def annotate_scene(
    frame_path: Path,
    gt_path: Path,
    fn_classes: list[str],    # class names that were missed (in order)
    fp_count: int = 0,
    max_size: int = 480,
) -> Image.Image:
    """
    Return an annotated PIL image.
    Green boxes = TPs (GT objects that were detected).
    Red boxes = FNs (GT objects that were missed).
    """
    img = Image.open(frame_path).convert("RGB")

    # Resize for report embedding while keeping aspect ratio
    w, h = img.size
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        scale_x, scale_y = int(w * scale) / w, int(h * scale) / h
    else:
        scale_x = scale_y = 1.0

    draw = ImageDraw.Draw(img)

    gts = _load_gt(gt_path)
    fn_remaining = list(fn_classes)

    for gt in gts:
        b = gt["box"]
        x1 = b["x_min"] * scale_x
        y1 = b["y_min"] * scale_y
        x2 = b["x_max"] * scale_x
        y2 = b["y_max"] * scale_y
        cls = gt["class_name"]
        tier = gt.get("tier", "?")

        if cls in fn_remaining:
            fn_remaining.remove(cls)
            color = "#EF4444"   # red — missed
            label = f"MISS:{cls}({tier[0]})"
        else:
            color = "#22C55E"   # green — detected
            label = f"TP:{cls}({tier[0]})"

        _draw_box(draw, x1, y1, x2, y2, label, color)

    # FP note in corner
    if fp_count:
        draw.text((4, 4), f"{fp_count} FP(s)", fill="#F97316", font=_font(11))

    return img


def select_failure_scenes(
    df: pd.DataFrame,
    matches_df: pd.DataFrame,
    dataset_dir: Path,
    top_n: int = 4,
) -> list[AnnotatedScene]:
    """
    Pick the scenes with the most FN objects, annotate, return AnnotatedScene list.
    *df* is the matches DataFrame; *matches_df* is the same (alias for clarity).
    """
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        return []

    manifest = pd.read_csv(manifest_path).set_index("scene_id")

    fn_per_scene = (
        df[df["match_type"] == "fn"]
        .groupby("scene_id")["gt_class"]
        .apply(list)
        .rename("fn_classes")
    )
    fp_per_scene = (
        df[df["match_type"] == "fp"]
        .groupby("scene_id")
        .size()
        .rename("fp_count")
    )
    scene_stats = (
        fn_per_scene.to_frame()
        .join(fp_per_scene, how="left")
        .fillna(0)
    )
    scene_stats["fn_count"] = scene_stats["fn_classes"].apply(len)
    worst = scene_stats.sort_values("fn_count", ascending=False).head(top_n)

    results: list[AnnotatedScene] = []
    for scene_id, row in worst.iterrows():
        if scene_id not in manifest.index:
            continue
        mrow = manifest.loc[scene_id]
        frame_path = dataset_dir / mrow["frame_path"]
        gt_path = dataset_dir / mrow["gt_path"]
        if not frame_path.exists() or not gt_path.exists():
            continue

        img = annotate_scene(
            frame_path, gt_path,
            fn_classes=list(row["fn_classes"]),
            fp_count=int(row["fp_count"]),
        )
        results.append(AnnotatedScene(
            scene_id=str(scene_id),
            fn_count=int(row["fn_count"]),
            fp_count=int(row["fp_count"]),
            image_b64=_to_b64(img),
        ))

    return results
