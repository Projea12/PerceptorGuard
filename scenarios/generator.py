from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from scenarios.schemas import ObjectSpec, Scenario

# Primary objects — fixed class IDs so metrics are consistent across runs
OBJECT_CATALOG: list[dict] = [
    {"urdf": "cube.urdf",          "class_id": 0, "class_name": "cube"},
    {"urdf": "sphere2.urdf",       "class_id": 1, "class_name": "sphere"},
    {"urdf": "duck_vhacd.urdf",    "class_id": 2, "class_name": "duck"},
    {"urdf": "teddy_vhacd.urdf",   "class_id": 3, "class_name": "teddy"},
    {"urdf": "soccerball.urdf",    "class_id": 4, "class_name": "ball"},
    {"urdf": "lego/lego.urdf",     "class_id": 5, "class_name": "lego"},
    {"urdf": "domino/domino.urdf", "class_id": 6, "class_name": "domino"},
    {"urdf": "objects/mug.urdf",   "class_id": 7, "class_name": "mug"},
]

# Clutter — small objects that crowd the scene without being detection targets
CLUTTER_CATALOG: list[dict] = [
    {"urdf": "cube_small.urdf",   "class_id": 8, "class_name": "clutter"},
    {"urdf": "sphere_small.urdf", "class_id": 8, "class_name": "clutter"},
]

# Profile weights must sum to 1.0
_PROFILES: dict[str, float] = {
    "baseline":   0.30,
    "crowded":    0.15,
    "dark":       0.15,
    "far":        0.15,
    "occluded":   0.15,
    "steep":      0.10,
}


@dataclass
class GeneratorConfig:
    n: int = 100
    seed: int = 42
    image_width: int = 640
    image_height: int = 480


class ScenarioGenerator:
    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self._cfg = config or GeneratorConfig()

    def generate(self) -> list[Scenario]:
        rng = random.Random(self._cfg.seed)
        counts = _allocate(self._cfg.n, _PROFILES)

        scenarios: list[Scenario] = []
        idx = 0
        for profile, count in counts.items():
            for _ in range(count):
                s = self._make(f"scene_{idx:04d}", profile, rng)
                scenarios.append(s)
                idx += 1

        rng.shuffle(scenarios)
        # Re-index after shuffle so scene IDs are sequential
        for i, s in enumerate(scenarios):
            scenarios[i] = s.model_copy(update={"scene_id": f"scene_{i:04d}"})
        return scenarios

    # ── per-profile factories ─────────────────────────────────────────────────

    def _make(self, scene_id: str, profile: str, rng: random.Random) -> Scenario:
        cam_dist  = rng.uniform(2.0, 3.0)
        cam_pitch = rng.uniform(-45.0, -20.0)
        cam_yaw   = rng.uniform(0.0, 360.0)
        ambient   = rng.uniform(0.5, 0.9)

        if profile == "baseline":
            specs = self._random_objects(rng, rng.randint(1, 3), spread=1.5)

        elif profile == "crowded":
            cam_dist = rng.uniform(2.5, 4.0)
            specs = (
                self._random_objects(rng, rng.randint(5, 8), spread=1.5)
                + self._clutter(rng, rng.randint(3, 6), spread=2.0)
            )

        elif profile == "dark":
            ambient = rng.uniform(0.05, 0.25)
            specs = self._random_objects(rng, rng.randint(1, 4), spread=1.5)

        elif profile == "far":
            cam_dist = rng.uniform(4.0, 6.5)
            specs = self._random_objects(rng, rng.randint(2, 5), spread=1.0)

        elif profile == "occluded":
            cam_dist = rng.uniform(2.0, 3.5)
            specs = self._occluded_placement(rng, cam_dist, cam_pitch, cam_yaw)

        elif profile == "steep":
            cam_pitch = rng.uniform(-80.0, -55.0)
            cam_dist  = rng.uniform(2.0, 3.5)
            specs = self._random_objects(rng, rng.randint(2, 5), spread=1.5)

        else:
            specs = self._random_objects(rng, 2, spread=1.5)

        return Scenario(
            scene_id=scene_id,
            description=profile,
            image_width=self._cfg.image_width,
            image_height=self._cfg.image_height,
            camera_distance=cam_dist,
            camera_pitch=cam_pitch,
            camera_yaw=cam_yaw,
            ambient_light=ambient,
            object_specs=specs,
        )

    # ── placement helpers ─────────────────────────────────────────────────────

    def _random_objects(
        self, rng: random.Random, n: int, spread: float
    ) -> list[ObjectSpec]:
        specs = []
        for _ in range(n):
            obj = rng.choice(OBJECT_CATALOG)
            specs.append(ObjectSpec(
                urdf=obj["urdf"],
                class_id=obj["class_id"],
                class_name=obj["class_name"],
                position=(rng.uniform(-spread, spread), rng.uniform(-spread, spread), 0.5),
                orientation_euler=(0.0, 0.0, math.radians(rng.uniform(0, 360))),
            ))
        return specs

    def _clutter(self, rng: random.Random, n: int, spread: float) -> list[ObjectSpec]:
        specs = []
        for _ in range(n):
            obj = rng.choice(CLUTTER_CATALOG)
            specs.append(ObjectSpec(
                urdf=obj["urdf"],
                class_id=obj["class_id"],
                class_name=obj["class_name"],
                position=(rng.uniform(-spread, spread), rng.uniform(-spread, spread), 0.15),
            ))
        return specs

    def _occluded_placement(
        self,
        rng: random.Random,
        distance: float,
        pitch: float,
        yaw: float,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> list[ObjectSpec]:
        """Place primary objects then insert occluders between them and the camera."""
        cam = _camera_world_pos(distance, pitch, yaw, target)
        cam_xy = (cam[0], cam[1])

        n_primaries = rng.randint(2, 4)
        primaries = self._random_objects(rng, n_primaries, spread=1.0)
        occluders: list[ObjectSpec] = []

        for spec in primaries[: rng.randint(1, n_primaries)]:
            px, py = spec.position[0], spec.position[1]
            dx, dy = cam_xy[0] - px, cam_xy[1] - py
            dist = math.hypot(dx, dy)
            if dist < 0.3:
                continue
            # point along the line from primary toward camera
            t = rng.uniform(0.2, 0.5)
            # small lateral nudge so occlusion is partial, not total
            perp_x, perp_y = -dy / dist, dx / dist
            lateral = rng.uniform(-0.3, 0.3)
            ox = px + t * dx + lateral * perp_x
            oy = py + t * dy + lateral * perp_y
            obj = rng.choice(OBJECT_CATALOG)
            occluders.append(ObjectSpec(
                urdf=obj["urdf"],
                class_id=obj["class_id"],
                class_name=obj["class_name"],
                position=(ox, oy, 0.5),
            ))

        return primaries + occluders


# ── module-level utilities ────────────────────────────────────────────────────

def _camera_world_pos(
    distance: float,
    pitch: float,
    yaw: float,
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Compute camera world position from PyBullet YPR parameters (Z-up)."""
    pr = math.radians(pitch)
    yr = math.radians(yaw)
    dx =  distance * math.cos(pr) * math.sin(yr)
    dy = -distance * math.cos(pr) * math.cos(yr)
    dz = -distance * math.sin(pr)
    return (target[0] + dx, target[1] + dy, target[2] + dz)


def _allocate(n: int, weights: dict[str, float]) -> dict[str, int]:
    total = sum(weights.values())
    counts = {k: int(n * v / total) for k, v in weights.items()}
    remainder = n - sum(counts.values())
    for k in list(weights)[:remainder]:
        counts[k] += 1
    return counts
