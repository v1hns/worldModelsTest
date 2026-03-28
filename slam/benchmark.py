"""
Evaluation helpers for TUM RGB-D style trajectory benchmarking.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def compute_ate(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    if len(predicted) == 0 or len(ground_truth) == 0:
        return float("nan")
    n = min(len(predicted), len(ground_truth))
    pred = predicted[:n, :3, 3]
    gt = ground_truth[:n, :3, 3]
    offset = gt[0] - pred[0]
    aligned = pred + offset
    return float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))


def run_tum_evaluation(predicted: np.ndarray, ground_truth: np.ndarray, output_path: str):
    ate = compute_ate(predicted, ground_truth)
    report = {
        "frames_evaluated": int(min(len(predicted), len(ground_truth))),
        "ate_rmse": ate,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_tum_ground_truth(path: str) -> np.ndarray:
    poses: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(token) for token in line.split()]
            if len(parts) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = parts
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = quaternion_to_matrix(qw, qx, qy, qz)
            pose[:3, 3] = [tx, ty, tz]
            poses.append(pose)
    return np.stack(poses, axis=0) if poses else np.empty((0, 4, 4), dtype=np.float32)


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
