import pytest
from scenarios.schemas import BoundingBox, Detection, GroundTruth, Scenario


def test_scenario_defaults():
    s = Scenario(scene_id="s0")
    assert s.image_width == 640
    assert s.image_height == 480
    assert s.fov == 60.0


def test_bounding_box_area():
    box = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
    assert box.area == 5000.0


def test_detection_confidence_bounds():
    box = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50)
    d = Detection(box=box, class_id=0, class_name="cube", confidence=0.95)
    assert d.confidence == 0.95
    with pytest.raises(Exception):
        Detection(box=box, class_id=0, class_name="cube", confidence=1.5)


def test_ground_truth_round_trip():
    box = BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1)
    gt = GroundTruth(box=box, class_id=1, class_name="sphere", scene_id="s0")
    assert gt.model_dump()["scene_id"] == "s0"
