"""
Export helpers for downstream reconstruction tooling.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .vggt_slam import VGGT_SLAM


def export_colmap_model(slam: "VGGT_SLAM", output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    trajectory = slam.get_trajectory()
    frames = slam._frames  # intentionally internal utility use

    cameras_path = os.path.join(output_dir, "cameras.txt")
    images_path = os.path.join(output_dir, "images.txt")
    points_path = os.path.join(output_dir, "points3D.txt")

    if frames:
        K = frames[0]["intrinsics"]
        height, width = frames[0]["depth"].shape
    else:
        K = np.eye(3, dtype=np.float32)
        width = height = 0

    with open(cameras_path, "w", encoding="utf-8") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write(f"1 PINHOLE {width} {height} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")

    with open(images_path, "w", encoding="utf-8") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        for frame, pose in zip(frames, trajectory):
            R = pose[:3, :3]
            t = pose[:3, 3]
            qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
            fh.write(
                f"{frame['frame_id'] + 1} {qw} {qx} {qy} {qz} "
                f"{t[0]} {t[1]} {t[2]} 1 frame_{frame['frame_id']:06d}.png\n\n"
            )

    points = slam.get_point_cloud()
    with open(points_path, "w", encoding="utf-8") as fh:
        fh.write("# 3D point list:\n")
        for idx, point in enumerate(points):
            fh.write(f"{idx + 1} {point[0]} {point[1]} {point[2]} 255 255 255 1.0 1 1\n")


def _rotation_matrix_to_quaternion(R: np.ndarray):
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        diag = np.argmax(np.diag(R))
        if diag == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif diag == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)
