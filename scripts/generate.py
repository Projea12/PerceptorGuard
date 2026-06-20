#!/usr/bin/env python3
"""Generate N parameterized scenarios with ground-truth labels.

Usage:
    python scripts/generate.py --count 100 --out artifacts/dataset
    python scripts/generate.py --count 500 --seed 7 --out artifacts/dataset_500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Make repo root importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.generator import GeneratorConfig, ScenarioGenerator
from runner.scene_runner import SceneRunner

# Distinct colors per class (index = class_id % len)
_PALETTE = [
    "#FF3333", "#33FF55", "#3399FF", "#FFD700",
    "#FF33FF", "#00FFFF", "#FF8800", "#AA00FF", "#888888",
]


def draw_overlay(rgb: np.ndarray, gts) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for gt in gts:
        color = _PALETTE[gt.class_id % len(_PALETTE)]
        b = gt.box
        draw.rectangle([b.x_min, b.y_min, b.x_max, b.y_max], outline=color, width=2)
        label = f"{gt.class_name}"
        draw.rectangle(
            [b.x_min, b.y_min - 12, b.x_min + len(label) * 6 + 4, b.y_min],
            fill=color,
        )
        draw.text((b.x_min + 2, b.y_min - 12), label, fill="#000000")
    return np.array(img)


def main() -> None:
    ap = argparse.ArgumentParser(description="PerceptorGuard scenario generator")
    ap.add_argument("--count",  type=int,  default=100,                     help="Number of scenarios")
    ap.add_argument("--seed",   type=int,  default=42,                      help="RNG seed")
    ap.add_argument("--out",    type=Path, default=Path("artifacts/dataset"),help="Output directory")
    ap.add_argument("--width",  type=int,  default=640,                     help="Frame width")
    ap.add_argument("--height", type=int,  default=480,                     help="Frame height")
    args = ap.parse_args()

    out_dir: Path = args.out
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} scenarios → {out_dir}")
    cfg = GeneratorConfig(
        n=args.count, seed=args.seed,
        image_width=args.width, image_height=args.height,
    )
    scenarios = ScenarioGenerator(cfg).generate()
    runner = SceneRunner()

    manifest_rows: list[dict] = []
    t0 = time.perf_counter()

    for i, scenario in enumerate(scenarios):
        scene_dir = scenes_dir / scenario.scene_id
        scene_dir.mkdir(exist_ok=True)

        rgb, gts = runner.run(scenario)

        frame_path   = scene_dir / "frame.png"
        gt_path      = scene_dir / "ground_truth.json"
        overlay_path = scene_dir / "overlay.png"

        Image.fromarray(rgb).save(frame_path)
        gt_path.write_text(json.dumps([gt.model_dump() for gt in gts], indent=2))
        Image.fromarray(draw_overlay(rgb, gts)).save(overlay_path)

        manifest_rows.append({
            "scene_id":        scenario.scene_id,
            "profile":         scenario.description,
            "frame_path":      str(frame_path.relative_to(out_dir)),
            "gt_path":         str(gt_path.relative_to(out_dir)),
            "overlay_path":    str(overlay_path.relative_to(out_dir)),
            "num_objects":     len(scenario.object_specs),
            "num_visible":     len(gts),
            "camera_distance": round(scenario.camera_distance, 3),
            "camera_pitch":    round(scenario.camera_pitch, 2),
            "camera_yaw":      round(scenario.camera_yaw, 2),
            "ambient_light":   round(scenario.ambient_light, 3),
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(scenarios) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:>{len(str(args.count))}}/{args.count}] "
                f"{scenario.scene_id}  "
                f"visible={len(gts)}/{len(scenario.object_specs)}  "
                f"profile={scenario.description:<10}  "
                f"ETA {eta:.0f}s"
            )

    df = pd.DataFrame(manifest_rows)
    df.to_csv(out_dir / "manifest.csv", index=False)

    elapsed = time.perf_counter() - t0
    print(f"\n{'─'*60}")
    print(f"Done in {elapsed:.1f}s  ({elapsed/len(scenarios)*1000:.0f} ms/scene)")
    print(f"Scenes:   {len(scenarios)}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    print(f"\nProfile breakdown:")
    for profile, grp in df.groupby("profile"):
        avg_vis = grp["num_visible"].mean()
        print(f"  {profile:<12} {len(grp):>4} scenes  avg_visible={avg_vis:.1f}")


if __name__ == "__main__":
    main()
