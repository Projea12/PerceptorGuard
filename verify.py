"""
Chunk 0 verification: import everything, spawn PyBullet, render one RGB frame, save to disk.
"""
import sys
from pathlib import Path

# ── 1. Verify all package imports ────────────────────────────────────────────
import pybullet as pb
import pybullet_data
import torch
import ultralytics
import pandas as pd
import pydantic
import numpy as np
from PIL import Image

print(f"pybullet     {pb.__version__ if hasattr(pb, '__version__') else 'ok'}")
print(f"torch        {torch.__version__}  (MPS={torch.backends.mps.is_available()})")
print(f"ultralytics  {ultralytics.__version__}")
print(f"pandas       {pd.__version__}")
print(f"pydantic     {pydantic.__version__}")

# ── 2. Verify local schemas ───────────────────────────────────────────────────
from scenarios.schemas import BoundingBox, Detection, GroundTruth, Scenario

scene = Scenario(
    scene_id="verify_0",
    description="Chunk-0 smoke test",
    objects=["cube"],
)
print(f"Scenario     {scene.scene_id!r}  {scene.image_width}x{scene.image_height}")

# ── 3. Spawn PyBullet (offscreen, no GUI) ────────────────────────────────────
client = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
pb.setGravity(0, 0, -9.81, physicsClientId=client)

plane_id = pb.loadURDF("plane.urdf", physicsClientId=client)
cube_id = pb.loadURDF(
    "cube.urdf",
    basePosition=[0, 0, 0.5],
    physicsClientId=client,
)
pb.stepSimulation(physicsClientId=client)
print(f"PyBullet     plane={plane_id}  cube={cube_id}")

# ── 4. Render one RGB frame ───────────────────────────────────────────────────
W, H = scene.image_width, scene.image_height

view_matrix = pb.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0, 0, 0.5],
    distance=scene.camera_distance,
    yaw=scene.camera_yaw,
    pitch=scene.camera_pitch,
    roll=0,
    upAxisIndex=2,
    physicsClientId=client,
)
proj_matrix = pb.computeProjectionMatrixFOV(
    fov=scene.fov,
    aspect=W / H,
    nearVal=0.1,
    farVal=100.0,
    physicsClientId=client,
)
_, _, rgba, _, _ = pb.getCameraImage(
    width=W,
    height=H,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    physicsClientId=client,
)

pb.disconnect(physicsClientId=client)

# ── 5. Save frame to disk ────────────────────────────────────────────────────
rgb = np.array(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
out_dir = Path("artifacts")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "verify_frame_0.png"
Image.fromarray(rgb).save(out_path)
print(f"Frame saved  {out_path}  ({W}x{H})")

print("\nChunk 0 verification PASSED.")
