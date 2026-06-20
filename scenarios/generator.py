from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from scenarios.schemas import ObjectSpec, Scenario

_ASSETS = Path(__file__).parent.parent / "assets"

# ── object catalogs ───────────────────────────────────────────────────────────
# class_names match COCO vocabulary exactly where applicable so that
# GT labels align with YOLOv8 output strings without a mapping layer.

EASY_TIER: list[dict] = [
    # COCO-recognizable — expect measurable detection signal
    {"urdf": "objects/mug.urdf",          "class_id": 0, "class_name": "cup"},
    {"urdf": str(_ASSETS / "bottle.urdf"),"class_id": 1, "class_name": "bottle"},
    {"urdf": str(_ASSETS / "bowl.urdf"),  "class_id": 2, "class_name": "bowl"},
    {"urdf": "teddy_vhacd.urdf",          "class_id": 3, "class_name": "teddy bear"},
    {"urdf": "soccerball.urdf",           "class_id": 4, "class_name": "sports ball"},
]

HARD_TIER: list[dict] = [
    # Off-vocabulary — YOLO will likely fail; failure is the documented result
    {"urdf": "cube.urdf",          "class_id": 5, "class_name": "cube"},
    {"urdf": "duck_vhacd.urdf",    "class_id": 6, "class_name": "duck"},
    {"urdf": "lego/lego.urdf",     "class_id": 7, "class_name": "lego"},
    {"urdf": "domino/domino.urdf", "class_id": 8, "class_name": "domino"},
]

CLUTTER_CATALOG: list[dict] = [
    {"urdf": "cube_small.urdf",   "class_id": 9, "class_name": "clutter"},
    {"urdf": "sphere_small.urdf", "class_id": 9, "class_name": "clutter"},
]

# Profile mix — proportions must sum to 1.0
_PROFILES: dict[str, float] = {
    "baseline": 0.30,
    "crowded":  0.15,
    "dark":     0.15,
    "far":      0.15,
    "occluded": 0.15,
    "steep":    0.10,
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
            specs = self._mixed_objects(rng, rng.randint(1, 3), spread=1.5)

        elif profile == "crowded":
            cam_dist = rng.uniform(2.5, 4.0)
            specs = (
                self._mixed_objects(rng, rng.randint(5, 8), spread=1.5)
                + self._clutter(rng, rng.randint(3, 6), spread=2.0)
            )

        elif profile == "dark":
            ambient = rng.uniform(0.05, 0.25)
            specs = self._mixed_objects(rng, rng.randint(1, 4), spread=1.5)

        elif profile == "far":
            cam_dist = rng.uniform(4.0, 6.5)
            specs = self._mixed_objects(rng, rng.randint(2, 5), spread=1.0)

        elif profile == "occluded":
            cam_dist = rng.uniform(2.0, 3.5)
            specs = self._occluded_placement(rng, cam_dist, cam_pitch, cam_yaw)

        elif profile == "steep":
            cam_pitch = rng.uniform(-80.0, -55.0)
            cam_dist  = rng.uniform(2.0, 3.5)
            specs = self._mixed_objects(rng, rng.randint(2, 5), spread=1.5)

        else:
            specs = self._mixed_objects(rng, 2, spread=1.5)

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

    def _mixed_objects(
        self, rng: random.Random, n: int, spread: float
    ) -> list[ObjectSpec]:
        """Sample from both tiers; guarantee at least one of each when n >= 2."""
        specs: list[ObjectSpec] = []
        if n >= 2:
            specs.append(self._pick_spec(rng, EASY_TIER, spread))
            specs.append(self._pick_spec(rng, HARD_TIER, spread))
            for _ in range(n - 2):
                pool = EASY_TIER if rng.random() < 0.5 else HARD_TIER
                specs.append(self._pick_spec(rng, pool, spread))
        else:
            pool = EASY_TIER if rng.random() < 0.5 else HARD_TIER
            specs.append(self._pick_spec(rng, pool, spread))
        return specs

    def _pick_spec(
        self, rng: random.Random, catalog: list[dict], spread: float
    ) -> ObjectSpec:
        obj = rng.choice(catalog)
        tier = "hard" if catalog is HARD_TIER else "easy"
        return ObjectSpec(
            urdf=obj["urdf"],
            class_id=obj["class_id"],
            class_name=obj["class_name"],
            tier=tier,
            position=(rng.uniform(-spread, spread), rng.uniform(-spread, spread), 0.5),
            orientation_euler=(0.0, 0.0, math.radians(rng.uniform(0, 360))),
        )

    def _clutter(self, rng: random.Random, n: int, spread: float) -> list[ObjectSpec]:
        specs = []
        for _ in range(n):
            obj = rng.choice(CLUTTER_CATALOG)
            specs.append(ObjectSpec(
                urdf=obj["urdf"],
                class_id=obj["class_id"],
                class_name=obj["class_name"],
                tier="clutter",
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
        primaries = self._mixed_objects(rng, n_primaries, spread=1.0)
        occluders: list[ObjectSpec] = []

        for spec in primaries[: rng.randint(1, n_primaries)]:
            px, py = spec.position[0], spec.position[1]
            dx, dy = cam_xy[0] - px, cam_xy[1] - py
            dist = math.hypot(dx, dy)
            if dist < 0.3:
                continue
            t = rng.uniform(0.2, 0.5)
            perp_x, perp_y = -dy / dist, dx / dist
            lateral = rng.uniform(-0.3, 0.3)
            ox = px + t * dx + lateral * perp_x
            oy = py + t * dy + lateral * perp_y
            # Occluder is sampled from hard tier — makes occlusion findings more interesting
            occluders.append(self._pick_spec(rng, HARD_TIER, 0.0))
            occluders[-1] = occluders[-1].model_copy(update={"position": (ox, oy, 0.5)})

        return primaries + occluders


# ── module utilities ──────────────────────────────────────────────────────────

def _camera_world_pos(
    distance: float,
    pitch: float,
    yaw: float,
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Camera world position from PyBullet YPR convention (Z-up)."""
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
