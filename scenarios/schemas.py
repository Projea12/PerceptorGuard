import math
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)


class ObjectSpec(BaseModel):
    urdf: str
    class_id: int
    class_name: str
    tier: str = "easy"   # "easy" | "hard" | "clutter"
    position: tuple[float, float, float]
    orientation_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)


class Scenario(BaseModel):
    scene_id: str
    description: str = ""
    image_width: int = Field(default=640, gt=0)
    image_height: int = Field(default=480, gt=0)
    fov: float = Field(default=60.0, gt=0.0, lt=180.0)
    camera_distance: float = Field(default=2.5, gt=0.0)
    camera_pitch: float = Field(default=-30.0)
    camera_yaw: float = Field(default=50.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ambient_light: float = Field(default=0.8, ge=0.0, le=1.0)
    shadow: bool = True
    object_specs: list[ObjectSpec] = Field(default_factory=list)


class Detection(BaseModel):
    box: BoundingBox
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    frame_id: str = ""


class GroundTruth(BaseModel):
    box: BoundingBox
    class_id: int
    class_name: str
    tier: str = "easy"   # "easy" | "hard" | "clutter"
    scene_id: str
    object_id: str = ""
