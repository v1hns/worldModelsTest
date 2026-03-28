"""
Open3D-based visualization helpers.
"""

from __future__ import annotations

from typing import List

import numpy as np


class Open3DViewer:
    def __init__(self):
        try:
            import open3d as o3d  # type: ignore

            self._o3d = o3d
        except ImportError:
            self._o3d = None
        self.last_geometry = {}

    @property
    def available(self) -> bool:
        return self._o3d is not None

    def update(self, trajectory: np.ndarray, point_cloud: np.ndarray, labeled_objects: List[dict]):
        self.last_geometry = {
            "trajectory": trajectory,
            "point_cloud": point_cloud,
            "labels": labeled_objects,
        }
        if self._o3d is None:
            return

        pcd = self._o3d.geometry.PointCloud()
        if point_cloud.size:
            pcd.points = self._o3d.utility.Vector3dVector(point_cloud)

        line_set = self._o3d.geometry.LineSet()
        if len(trajectory) >= 2:
            positions = trajectory[:, :3, 3]
            line_set.points = self._o3d.utility.Vector3dVector(positions)
            lines = [[idx, idx + 1] for idx in range(len(positions) - 1)]
            line_set.lines = self._o3d.utility.Vector2iVector(lines)

        self._o3d.visualization.draw_geometries([pcd, line_set])
