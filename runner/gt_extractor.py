import itertools
from typing import Optional

import numpy as np

from scenarios.schemas import BoundingBox


def project_aabb_to_screen(
    aabb_min: tuple[float, float, float],
    aabb_max: tuple[float, float, float],
    view_matrix: tuple,
    proj_matrix: tuple,
    width: int,
    height: int,
) -> Optional[BoundingBox]:
    """Project a 3-D axis-aligned bounding box onto the image plane.

    PyBullet matrices are column-major (OpenGL convention); we reshape then
    transpose to get row-major before doing MVP multiplication.
    """
    V = np.array(view_matrix, dtype=np.float64).reshape(4, 4).T
    P = np.array(proj_matrix, dtype=np.float64).reshape(4, 4).T
    MVP = P @ V

    xs: list[float] = []
    ys: list[float] = []
    for corner in itertools.product(*zip(aabb_min, aabb_max)):
        clip = MVP @ np.array([*corner, 1.0])
        w = clip[3]
        if w <= 1e-7:
            continue
        ndc_x = clip[0] / w
        ndc_y = clip[1] / w
        xs.append((ndc_x + 1.0) / 2.0 * width)
        ys.append((1.0 - ndc_y) / 2.0 * height)

    if not xs:
        return None

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(width), max(xs))
    y_max = min(float(height), max(ys))

    if x_max - x_min < 1.0 or y_max - y_min < 1.0:
        return None

    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
