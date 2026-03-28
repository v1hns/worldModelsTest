"""
Small geometry helpers shared across the SLAM pipeline.
"""

from __future__ import annotations

import numpy as np


def ensure_homogeneous(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = pose
        return out
    raise ValueError(f"Expected 3x4 or 4x4 pose, got {pose.shape}")


def relative_pose(anchor_pose: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(anchor_pose) @ pose).astype(np.float32)


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b).astype(np.float32)


def translation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
