#!/usr/bin/env python3
"""
Chunk-2 verification — pipeline correctness, no detection model loaded.

  pytest tests/verify_chunk2.py -v          # structured test run
  python  tests/verify_chunk2.py            # standalone pass/fail report
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from runner.scene_runner import SceneRunner
from scenarios.schemas import GroundTruth, ObjectSpec, Scenario

# ── constants ─────────────────────────────────────────────────────────────────

SAMPLES_DIR = Path(__file__).parent.parent / "samples"

KNOWN_CLASSES: set[str] = {
    # easy tier — COCO vocab
    "cup", "bottle", "bowl", "teddy bear", "sports ball",
    # hard tier — off-vocabulary
    "cube", "duck", "lego", "domino",
}

_ASSETS = Path(__file__).parent.parent / "assets"

# box color per tier for overlays
_TIER_COLOR = {"easy": "#22DD44", "hard": "#FF4422"}

# ── scenario definitions ──────────────────────────────────────────────────────

def _spec(urdf: str, cid: int, name: str, tier: str,
          x: float = 0.0, y: float = 0.0, z: float = 0.5,
          yaw_deg: float = 0.0) -> ObjectSpec:
    return ObjectSpec(
        urdf=urdf, class_id=cid, class_name=name, tier=tier,
        position=(x, y, z),
        orientation_euler=(0.0, 0.0, float(np.radians(yaw_deg))),
    )

def _easy(urdf, cid, name, **kw) -> ObjectSpec:
    return _spec(urdf, cid, name, "easy", **kw)

def _hard(urdf, cid, name, **kw) -> ObjectSpec:
    return _spec(urdf, cid, name, "hard", **kw)

def _clutter(x, y) -> ObjectSpec:
    return ObjectSpec(
        urdf="cube_small.urdf", class_id=9, class_name="clutter", tier="clutter",
        position=(x, y, 0.15),
    )


VERIFICATION_SCENARIOS: list[Scenario] = [

    # ── Group A: easy — single object, centered, well-lit ────────────────────
    Scenario(
        scene_id="easy_0", description="easy: single teddy bear, centered",
        camera_distance=2.2, camera_pitch=-30.0, camera_yaw=50.0, ambient_light=0.85,
        object_specs=[_easy("teddy_vhacd.urdf", 3, "teddy bear")],
    ),
    Scenario(
        scene_id="easy_1", description="easy: single sports ball, centered",
        camera_distance=2.2, camera_pitch=-30.0, camera_yaw=50.0, ambient_light=0.85,
        object_specs=[_easy("soccerball.urdf", 4, "sports ball")],
    ),
    Scenario(
        scene_id="easy_2", description="easy: single cup, centered",
        camera_distance=2.2, camera_pitch=-30.0, camera_yaw=50.0, ambient_light=0.85,
        object_specs=[_easy("objects/mug.urdf", 0, "cup")],
    ),

    # ── Group B: occlusion ────────────────────────────────────────────────────
    # Camera at yaw=0 looks roughly along +Y; y-axis is depth from camera.
    # Objects with smaller y are closer to camera and occlude those with larger y.
    Scenario(
        scene_id="occ_0", description="occ: cube partially occluding teddy bear",
        camera_distance=2.8, camera_pitch=-25.0, camera_yaw=0.0, ambient_light=0.8,
        object_specs=[
            _hard("cube.urdf",         5, "cube",       x=0.0,  y=-0.2),   # closer
            _easy("teddy_vhacd.urdf",  3, "teddy bear", x=0.15, y=0.6),    # further+offset
        ],
    ),
    Scenario(
        scene_id="occ_1", description="occ: lego partially occluding sports ball",
        camera_distance=2.8, camera_pitch=-25.0, camera_yaw=0.0, ambient_light=0.8,
        object_specs=[
            _hard("lego/lego.urdf",   7, "lego",        x=0.0,  y=-0.1),
            _easy("soccerball.urdf",  4, "sports ball", x=0.1,  y=0.6),
        ],
    ),
    # Edge case (b): cup placed directly behind cube on same camera ray.
    # GT extractor uses 3D AABB projection (no visibility culling), so the
    # cup will still produce a box. Test verifies that box is valid, not corrupt.
    Scenario(
        scene_id="occ_2",
        description="edge-full-occ: cup directly behind cube (same screen column)",
        camera_distance=3.0, camera_pitch=-20.0, camera_yaw=0.0, ambient_light=0.8,
        object_specs=[
            _hard("cube.urdf",        5, "cube", x=0.0, y=-0.1),   # occluder
            _easy("objects/mug.urdf", 0, "cup",  x=0.0, y=1.3),    # hidden behind cube
        ],
    ),

    # ── Group C: distance variation ───────────────────────────────────────────
    Scenario(
        scene_id="dist_0", description="dist: teddy bear at distance=4.5",
        camera_distance=4.5, camera_pitch=-30.0, camera_yaw=45.0, ambient_light=0.8,
        object_specs=[_easy("teddy_vhacd.urdf", 3, "teddy bear")],
    ),
    Scenario(
        scene_id="dist_1", description="dist: sports ball at distance=6.0",
        camera_distance=6.0, camera_pitch=-30.0, camera_yaw=45.0, ambient_light=0.8,
        object_specs=[_easy("soccerball.urdf", 4, "sports ball")],
    ),
    # Edge case (c): extreme distance. gt_extractor drops sub-pixel boxes (<1px),
    # so at distance=8.5 the cube may produce no GT box. Test asserts the result
    # is either absent OR a small box — never an oversized phantom.
    Scenario(
        scene_id="dist_2",
        description="edge-far: cube at distance=8.5, expect tiny or no GT box",
        camera_distance=8.5, camera_pitch=-30.0, camera_yaw=45.0, ambient_light=0.9,
        object_specs=[_hard("cube.urdf", 5, "cube")],
    ),

    # ── Group D: high clutter ─────────────────────────────────────────────────
    Scenario(
        scene_id="clutter_0", description="clutter: 5 primary + 5 clutter objects",
        camera_distance=3.5, camera_pitch=-35.0, camera_yaw=60.0, ambient_light=0.75,
        object_specs=[
            _easy("teddy_vhacd.urdf",   3, "teddy bear",   x=-0.8, y=-0.8),
            _easy("soccerball.urdf",    4, "sports ball",  x= 0.8, y=-0.8),
            _hard("cube.urdf",          5, "cube",         x=-0.8, y= 0.8),
            _hard("duck_vhacd.urdf",    6, "duck",         x= 0.8, y= 0.8),
            _easy("objects/mug.urdf",   0, "cup",          x= 0.0, y= 0.0),
            _clutter(-0.3, -0.3), _clutter(0.3, -0.3),
            _clutter(-0.3,  0.3), _clutter(0.3,  0.3), _clutter(0.0, -0.5),
        ],
    ),
    Scenario(
        scene_id="clutter_1", description="clutter: 7 primary + 4 clutter objects",
        camera_distance=4.0, camera_pitch=-40.0, camera_yaw=30.0, ambient_light=0.7,
        object_specs=[
            _easy("teddy_vhacd.urdf",    3, "teddy bear",   x=-1.2, y=-1.2),
            _easy("soccerball.urdf",     4, "sports ball",  x= 1.2, y=-1.2),
            _hard("cube.urdf",           5, "cube",         x=-1.2, y= 1.2),
            _hard("duck_vhacd.urdf",     6, "duck",         x= 1.2, y= 1.2),
            _easy("objects/mug.urdf",    0, "cup",          x= 0.0, y=-0.6),
            _hard("lego/lego.urdf",      7, "lego",         x=-0.6, y= 0.0),
            _hard("domino/domino.urdf",  8, "domino",       x= 0.6, y= 0.0),
            _clutter(-0.3, -0.3), _clutter(0.3, -0.3),
            _clutter(0.0,  0.5),  _clutter(0.0, -0.5),
        ],
    ),
    # Edge case (a): camera_target offset so objects land near the frame border.
    # Verifies that project_aabb_to_screen clamps to [0, W] x [0, H], never
    # produces negative coords or coords > image dimensions.
    Scenario(
        scene_id="clutter_2",
        description="edge-frame: sports ball near image border + 2 clutter",
        camera_distance=2.5, camera_pitch=-30.0, camera_yaw=0.0, ambient_light=0.8,
        # Target is offset in X so that objects at x≈0 appear near the right edge
        camera_target=(1.2, 0.0, 0.0),
        object_specs=[
            _easy("soccerball.urdf",  4, "sports ball", x=0.0,  y=0.0),   # near edge
            _easy("teddy_vhacd.urdf", 3, "teddy bear",  x=-0.8, y=0.0),   # more interior
            _clutter(-0.4, 0.3), _clutter(-0.4, -0.3),
        ],
    ),
]

# ── shared state (populated once by setup_module) ─────────────────────────────

_RESULTS: dict[str, tuple[np.ndarray, list[GroundTruth]]] = {}


def setup_module(module: object) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    runner = SceneRunner()
    for scenario in VERIFICATION_SCENARIOS:
        rgb, gts = runner.run(scenario)
        _RESULTS[scenario.scene_id] = (rgb, gts)
        _save_overlay(scenario, rgb, gts)


# ── overlay helper ────────────────────────────────────────────────────────────

def _save_overlay(scenario: Scenario, rgb: np.ndarray, gts: list[GroundTruth]) -> None:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for gt in gts:
        color = _TIER_COLOR.get(gt.tier, "#AAAAAA")
        b = gt.box
        w_px = int(b.x_max - b.x_min)
        h_px = int(b.y_max - b.y_min)
        draw.rectangle([b.x_min, b.y_min, b.x_max, b.y_max], outline=color, width=2)
        label = f"{gt.class_name} ({gt.tier[0]}) {w_px}×{h_px}"
        lw = len(label) * 6 + 4
        draw.rectangle([b.x_min, b.y_min - 14, b.x_min + lw, b.y_min], fill=color)
        draw.text((b.x_min + 2, b.y_min - 13), label, fill="#000000")
    out = SAMPLES_DIR / f"overlay_{scenario.scene_id}.png"
    img.save(out)


def _scenario(scene_id: str) -> Scenario:
    return next(s for s in VERIFICATION_SCENARIOS if s.scene_id == scene_id)


def _non_clutter_count(scenario: Scenario) -> int:
    return sum(1 for s in scenario.object_specs if s.tier != "clutter")


# ── invariant 1: boxes within image bounds ────────────────────────────────────

@pytest.mark.parametrize("scene_id", [s.scene_id for s in VERIFICATION_SCENARIOS])
def test_boxes_within_bounds(scene_id: str) -> None:
    sc = _scenario(scene_id)
    _, gts = _RESULTS[scene_id]
    for gt in gts:
        b = gt.box
        assert b.x_min >= 0,              f"{scene_id}/{gt.class_name}: x_min={b.x_min:.1f} < 0"
        assert b.y_min >= 0,              f"{scene_id}/{gt.class_name}: y_min={b.y_min:.1f} < 0"
        assert b.x_max <= sc.image_width, f"{scene_id}/{gt.class_name}: x_max={b.x_max:.1f} > {sc.image_width}"
        assert b.y_max <= sc.image_height,f"{scene_id}/{gt.class_name}: y_max={b.y_max:.1f} > {sc.image_height}"


# ── invariant 2: every box has positive area ──────────────────────────────────

@pytest.mark.parametrize("scene_id", [s.scene_id for s in VERIFICATION_SCENARIOS])
def test_boxes_positive_area(scene_id: str) -> None:
    _, gts = _RESULTS[scene_id]
    for gt in gts:
        b = gt.box
        assert b.x_max > b.x_min, f"{scene_id}/{gt.class_name}: zero-width box"
        assert b.y_max > b.y_min, f"{scene_id}/{gt.class_name}: zero-height box"
        assert b.area > 0,        f"{scene_id}/{gt.class_name}: area=0"


# ── invariant 3: class labels belong to known set ────────────────────────────

@pytest.mark.parametrize("scene_id", [s.scene_id for s in VERIFICATION_SCENARIOS])
def test_class_labels_known(scene_id: str) -> None:
    _, gts = _RESULTS[scene_id]
    for gt in gts:
        assert gt.class_name in KNOWN_CLASSES, \
            f"{scene_id}: unknown class '{gt.class_name}'"


# ── invariant 4: GT count never exceeds non-clutter spec count ───────────────

@pytest.mark.parametrize("scene_id", [s.scene_id for s in VERIFICATION_SCENARIOS])
def test_gt_count_leq_requested(scene_id: str) -> None:
    sc = _scenario(scene_id)
    _, gts = _RESULTS[scene_id]
    cap = _non_clutter_count(sc)
    assert len(gts) <= cap, \
        f"{scene_id}: GT count {len(gts)} > non-clutter spec count {cap}"


# ── invariant 5: easy single-object scenes produce exactly 1 GT box ──────────

@pytest.mark.parametrize("scene_id", ["easy_0", "easy_1", "easy_2"])
def test_easy_strict_gt_count(scene_id: str) -> None:
    _, gts = _RESULTS[scene_id]
    assert len(gts) == 1, \
        f"{scene_id}: expected 1 GT box (single centered object), got {len(gts)}"


# ── invariant 6: determinism — same scenario twice yields identical GT ────────

def test_determinism() -> None:
    scenario = _scenario("easy_0")
    runner = SceneRunner()
    _, gts_a = runner.run(scenario)
    _, gts_b = runner.run(scenario)
    assert len(gts_a) == len(gts_b), "Differing GT count across runs"
    for a, b in zip(gts_a, gts_b):
        assert a.box.model_dump() == b.box.model_dump(), \
            f"Box mismatch: {a.box} vs {b.box}"


# ── edge case (a): frame-edge object must be clamped, not negative/wrapped ───

def test_edge_frame_box_clamped() -> None:
    sc = _scenario("clutter_2")
    _, gts = _RESULTS["clutter_2"]
    assert gts, "clutter_2: expected at least one visible GT box near frame edge"
    for gt in gts:
        b = gt.box
        assert b.x_min >= 0,              f"x_min={b.x_min:.1f} is negative (wrap-around?)"
        assert b.y_min >= 0,              f"y_min={b.y_min:.1f} is negative"
        assert b.x_max <= sc.image_width, f"x_max={b.x_max:.1f} exceeds image width"
        assert b.y_max <= sc.image_height,f"y_max={b.y_max:.1f} exceeds image height"
        assert b.x_max > b.x_min,        "box has non-positive width after clamp"
        assert b.y_max > b.y_min,        "box has non-positive height after clamp"
    # Confirm that at least one box touches or is near the frame boundary
    any_near_edge = any(
        gt.box.x_min < 20
        or gt.box.x_max > sc.image_width - 20
        or gt.box.y_min < 20
        or gt.box.y_max > sc.image_height - 20
        for gt in gts
    )
    assert any_near_edge, (
        "clutter_2: no GT box is near the frame edge — "
        "check camera_target offset or object placement"
    )


# ── edge case (b): fully occluded object — if GT exists it must be valid ─────

def test_edge_full_occlusion_boxes_valid() -> None:
    """
    3D AABB projection does not cull by visibility, so the cup behind the cube
    will still appear in GT. This is documented behaviour. The test asserts that
    the resulting box — occluded or not — is geometrically sound.
    """
    sc = _scenario("occ_2")
    _, gts = _RESULTS["occ_2"]
    cup_gts = [gt for gt in gts if gt.class_name == "cup"]
    if not cup_gts:
        # Fully occluded to sub-pixel — acceptable result
        return
    for gt in cup_gts:
        b = gt.box
        assert b.x_min >= 0 and b.x_max <= sc.image_width,  "cup box x out of bounds"
        assert b.y_min >= 0 and b.y_max <= sc.image_height, "cup box y out of bounds"
        assert b.area > 0, "cup box has zero area"


# ── edge case (c): far object — box is small (<10 % of frame) or absent ──────

def test_edge_far_box_small_or_absent() -> None:
    sc = _scenario("dist_2")
    _, gts = _RESULTS["dist_2"]
    if not gts:
        return  # sub-pixel at distance=8.5 — acceptable, gt_extractor filtered it
    total_pixels = sc.image_width * sc.image_height
    for gt in gts:
        fraction = gt.box.area / total_pixels
        assert fraction < 0.10, (
            f"dist_2/{gt.class_name}: box area is {fraction:.1%} of frame — "
            f"too large for distance=8.5 (expected <10%)"
        )


# ── standalone pass/fail report ───────────────────────────────────────────────

def _run_check(label: str, fn, *args) -> tuple[str, str, str]:
    try:
        fn(*args)
        return label, "PASS", ""
    except AssertionError as e:
        return label, "FAIL", str(e)
    except Exception as e:
        return label, "ERROR", repr(e)


def _standalone_report() -> None:
    print("Running scenarios…", flush=True)
    setup_module(None)

    # Print overlay locations
    print(f"\nOverlays written to  {SAMPLES_DIR}/")
    for sc in VERIFICATION_SCENARIOS:
        print(f"  overlay_{sc.scene_id}.png  — {sc.description}")

    print(f"\n{'─'*72}")
    print(f"{'TEST':<52}  {'RESULT':<6}  DETAIL")
    print(f"{'─'*72}")

    checks: list[tuple[str, str, str]] = []

    # Parametrized invariants
    for sc in VERIFICATION_SCENARIOS:
        sid = sc.scene_id
        checks.append(_run_check(f"bounds       {sid}", test_boxes_within_bounds, sid))
        checks.append(_run_check(f"pos_area     {sid}", test_boxes_positive_area, sid))
        checks.append(_run_check(f"known_class  {sid}", test_class_labels_known, sid))
        checks.append(_run_check(f"count_leq    {sid}", test_gt_count_leq_requested, sid))

    for sid in ("easy_0", "easy_1", "easy_2"):
        checks.append(_run_check(f"strict_count {sid}", test_easy_strict_gt_count, sid))

    checks.append(_run_check("determinism          easy_0", test_determinism))
    checks.append(_run_check("edge_frame           clutter_2", test_edge_frame_box_clamped))
    checks.append(_run_check("edge_full_occ        occ_2", test_edge_full_occlusion_boxes_valid))
    checks.append(_run_check("edge_far             dist_2", test_edge_far_box_small_or_absent))

    passed = failed = errors = 0
    for label, result, detail in checks:
        flag = "✓" if result == "PASS" else ("✗" if result == "FAIL" else "!")
        print(f"  {flag}  {label:<48}  {result}  {detail[:60]}")
        if result == "PASS":
            passed += 1
        elif result == "FAIL":
            failed += 1
        else:
            errors += 1

    print(f"{'─'*72}")
    print(f"  {passed} passed  |  {failed} failed  |  {errors} errors  "
          f"(total {passed+failed+errors})")

    if failed or errors:
        sys.exit(1)


if __name__ == "__main__":
    _standalone_report()
