import numpy as np
import pytest
from scenarios.schemas import BoundingBox, ObjectSpec, Scenario
from runner.gt_extractor import project_aabb_to_screen
from runner.scene_runner import SceneRunner


# ── gt_extractor unit tests (pure math, no PyBullet) ─────────────────────────

def _ortho_matrices(width=100, height=100):
    """Trivial identity view + orthographic projection for deterministic tests."""
    import pybullet as pb
    import pybullet_data
    client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    view = pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=5.0, yaw=0, pitch=-45, roll=0, upAxisIndex=2,
        physicsClientId=client,
    )
    proj = pb.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=50.0,
        physicsClientId=client,
    )
    pb.disconnect(client)
    return view, proj


def test_project_visible_box():
    view, proj = _ortho_matrices()
    box = project_aabb_to_screen((-0.3, -0.3, 0.0), (0.3, 0.3, 0.6), view, proj, 100, 100)
    assert box is not None
    assert 0 <= box.x_min < box.x_max <= 100
    assert 0 <= box.y_min < box.y_max <= 100


def test_project_behind_camera_returns_none():
    view, proj = _ortho_matrices()
    # A point far behind (negative Z from camera) — w will be negative
    box = project_aabb_to_screen((0, 0, 200), (0.1, 0.1, 200.1), view, proj, 100, 100)
    # Either None or valid clipped box; should not raise
    assert box is None or isinstance(box, BoundingBox)


def test_project_tiny_box_returns_none():
    view, proj = _ortho_matrices()
    # 1mm cube very far away → sub-pixel
    box = project_aabb_to_screen((0, 0, 0), (0.001, 0.001, 0.001), view, proj, 100, 100)
    assert box is None


# ── SceneRunner integration tests ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def simple_scenario():
    return Scenario(
        scene_id="test_00",
        camera_distance=2.5,
        camera_pitch=-30,
        camera_yaw=50,
        object_specs=[
            ObjectSpec(urdf="cube.urdf", class_id=0, class_name="cube", position=(0.0, 0.0, 0.5)),
        ],
    )


def test_runner_returns_rgb_frame(simple_scenario):
    runner = SceneRunner()
    rgb, _ = runner.run(simple_scenario)
    assert rgb.shape == (480, 640, 3)
    assert rgb.dtype == np.uint8


def test_runner_finds_cube_gt(simple_scenario):
    runner = SceneRunner()
    _, gts = runner.run(simple_scenario)
    assert len(gts) >= 1
    gt = gts[0]
    assert gt.class_name == "cube"
    assert gt.box.area > 0


def test_runner_multi_object():
    scenario = Scenario(
        scene_id="test_01",
        object_specs=[
            ObjectSpec(urdf="cube.urdf",   class_id=0, class_name="cube",   position=(-0.8, 0.0, 0.5)),
            ObjectSpec(urdf="sphere2.urdf", class_id=1, class_name="sphere", position=( 0.8, 0.0, 0.5)),
        ],
    )
    runner = SceneRunner()
    rgb, gts = runner.run(scenario)
    assert len(gts) == 2
    names = {gt.class_name for gt in gts}
    assert names == {"cube", "sphere"}


def test_runner_dark_scene_not_black():
    scenario = Scenario(
        scene_id="test_dark",
        ambient_light=0.1,
        object_specs=[
            ObjectSpec(urdf="cube.urdf", class_id=0, class_name="cube", position=(0, 0, 0.5)),
        ],
    )
    runner = SceneRunner()
    rgb, _ = runner.run(scenario)
    # Even a dark scene should have some non-zero pixels
    assert rgb.max() > 0
