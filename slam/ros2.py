"""
ROS2 wrapper for streaming RGB frames through the SLAM pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ROS2SLAMNode:
    def __init__(self, slam):
        self.slam = slam
        self._rclpy = None
        self.node = None
        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore

            self._rclpy = rclpy
            self.node = Node("vggt_slam")
        except ImportError:
            self._rclpy = None

    @property
    def available(self) -> bool:
        return self._rclpy is not None and self.node is not None

    def process_rgb_frame(self, frame: np.ndarray):
        return self.slam.process_frame(frame)
