from __future__ import annotations

import numpy as np
import pybullet as pb
import pybullet_data

from runner.gt_extractor import project_aabb_to_screen
from scenarios.schemas import GroundTruth, Scenario

_SETTLE_STEPS = 60


class SceneRunner:
    """Renders one Scenario and returns (RGB frame, ground-truth boxes)."""

    def run(self, scenario: Scenario) -> tuple[np.ndarray, list[GroundTruth]]:
        client = pb.connect(pb.DIRECT)
        try:
            return self._run(client, scenario)
        finally:
            pb.disconnect(client)

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self, client: int, scenario: Scenario) -> tuple[np.ndarray, list[GroundTruth]]:
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        pb.setGravity(0, 0, -9.81, physicsClientId=client)
        pb.loadURDF("plane.urdf", physicsClientId=client)

        loaded: list[tuple] = []
        for spec in scenario.object_specs:
            try:
                bid = pb.loadURDF(
                    spec.urdf,
                    basePosition=list(spec.position),
                    baseOrientation=pb.getQuaternionFromEuler(list(spec.orientation_euler)),
                    physicsClientId=client,
                )
                loaded.append((spec, bid))
            except Exception:
                pass

        for _ in range(_SETTLE_STEPS):
            pb.stepSimulation(physicsClientId=client)

        view_matrix = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(scenario.camera_target),
            distance=scenario.camera_distance,
            yaw=scenario.camera_yaw,
            pitch=scenario.camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=client,
        )
        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=scenario.fov,
            aspect=scenario.image_width / scenario.image_height,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=client,
        )

        # lightDiffuseCoeff complementary to ambient so total stays ≤ 1
        _, _, rgba, _, _ = pb.getCameraImage(
            width=scenario.image_width,
            height=scenario.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pb.ER_TINY_RENDERER,
            lightAmbientCoeff=scenario.ambient_light,
            lightDiffuseCoeff=max(0.0, 1.0 - scenario.ambient_light * 0.8),
            lightSpecularCoeff=0.3,
            physicsClientId=client,
        )

        ground_truths: list[GroundTruth] = []
        for spec, bid in loaded:
            try:
                aabb_min, aabb_max = pb.getAABB(bid, physicsClientId=client)
            except Exception:
                continue
            box = project_aabb_to_screen(
                aabb_min, aabb_max, view_matrix, proj_matrix,
                scenario.image_width, scenario.image_height,
            )
            if box is not None:
                ground_truths.append(GroundTruth(
                    box=box,
                    class_id=spec.class_id,
                    class_name=spec.class_name,
                    scene_id=scenario.scene_id,
                    object_id=str(bid),
                ))

        rgb = np.array(rgba, dtype=np.uint8).reshape(
            scenario.image_height, scenario.image_width, 4
        )[:, :, :3]
        return rgb, ground_truths
