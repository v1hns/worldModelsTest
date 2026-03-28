"""
test_slam.py — unit-level tests for VGGT-SLAM components.

Run with:  python test_slam.py
"""

import sys
import numpy as np
import torch
import unittest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_image(H=240, W=320):
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAttentionRetrieval(unittest.TestCase):

    def setUp(self):
        from slam.retrieval import AttentionRetrieval
        self.ret = AttentionRetrieval(device="cpu")

    def test_empty_query(self):
        results = self.ret.query(0, k=5)
        self.assertEqual(results, [])

    def test_add_and_query(self):
        for i in range(30):
            feat = torch.rand(256)
            self.ret.add(i, feat)

        results = self.ret.query(29, k=3, min_distance=10)
        self.assertLessEqual(len(results), 3)
        for fid, score in results:
            self.assertLess(fid, 20)         # must be ≥10 frames back
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_reset(self):
        self.ret.add(0, torch.rand(64))
        self.ret.reset()
        self.assertEqual(self.ret.size, 0)


class TestFactorGraph(unittest.TestCase):

    def _make_submap(self, pose, submap_id=0):
        from slam.submap import Submap
        frames = [{"image": random_image(), "tensor": torch.rand(3, 240, 320)}]
        poses = [pose]
        sm = Submap(submap_id=submap_id, frames=frames, poses=poses)
        return sm

    def setUp(self):
        from slam.factor_graph import FactorGraph
        self.fg = FactorGraph(device="cpu")

    def test_add_submaps(self):
        for i in range(4):
            pose = np.eye(4)
            pose[0, 3] = float(i)
            self.fg.add_submap(self._make_submap(pose, submap_id=i))
        self.assertEqual(self.fg.num_submaps, 4)
        self.assertEqual(len(self.fg.odometry_factors), 3)

    def test_loop_closure(self):
        for i in range(6):
            pose = np.eye(4)
            pose[0, 3] = float(i)
            self.fg.add_submap(self._make_submap(pose, submap_id=i))
        self.fg.add_loop_closure(0, 5, score=0.9)
        self.assertEqual(self.fg.num_loop_closures, 1)

    def test_optimize_returns_poses(self):
        for i in range(4):
            pose = np.eye(4)
            pose[0, 3] = float(i) * 0.5
            self.fg.add_submap(self._make_submap(pose, submap_id=i))
        self.fg.add_loop_closure(0, 3, score=0.8)
        optimised = self.fg.optimize(iterations=5)
        self.assertEqual(len(optimised), 4)
        for p in optimised:
            self.assertEqual(p.shape, (4, 4))

    def test_reset(self):
        pose = np.eye(4)
        self.fg.add_submap(self._make_submap(pose))
        self.fg.reset()
        self.assertEqual(self.fg.num_submaps, 0)


class TestSubmap(unittest.TestCase):

    def test_build_point_cloud_with_depth(self):
        from slam.submap import Submap

        H, W = 60, 80
        depth = np.ones((H, W), dtype=np.float32) * 2.0
        frame = {
            "image": random_image(H, W),
            "tensor": torch.rand(3, H, W),
            "depth": depth,
        }
        pose = np.eye(4, dtype=np.float32)
        sm = Submap(submap_id=0, frames=[frame], poses=[pose])
        sm.build_point_cloud()

        self.assertIsNotNone(sm.point_cloud)
        self.assertEqual(sm.point_cloud.shape[1], 3)
        self.assertGreater(len(sm.point_cloud), 0)

    def test_build_point_cloud_no_depth(self):
        from slam.submap import Submap

        frame = {"image": random_image(), "tensor": torch.rand(3, 240, 320)}
        sm = Submap(submap_id=0, frames=[frame], poses=[np.eye(4)])
        sm.build_point_cloud()
        self.assertEqual(sm.point_cloud.shape, (0, 3))


class TestVGGTSLAM(unittest.TestCase):
    """Integration-level test (dummy mode, no VGGT weights needed)."""

    def _make_slam(self):
        from slam.vggt_slam import VGGT_SLAM, SLAMConfig
        cfg = SLAMConfig(submap_size=5, verbose=False, device="cpu")
        return VGGT_SLAM(config=cfg)

    def test_single_frame(self):
        slam = self._make_slam()
        result = slam.process_frame(random_image())
        self.assertEqual(result.frame_id, 0)
        self.assertEqual(result.pose.shape, (4, 4))

    def test_multiple_frames(self):
        slam = self._make_slam()
        N = 12
        for _ in range(N):
            slam.process_frame(random_image())
        traj = slam.get_trajectory()
        self.assertEqual(traj.shape[0], N)
        self.assertEqual(traj.shape[1:], (4, 4))

    def test_submap_creation(self):
        slam = self._make_slam()
        for _ in range(10):
            slam.process_frame(random_image())
        # With submap_size=5, we expect at least one submap after 10 frames
        self.assertGreaterEqual(len(slam.submaps), 1)

    def test_reset(self):
        slam = self._make_slam()
        for _ in range(6):
            slam.process_frame(random_image())
        slam.reset()
        self.assertEqual(slam._frame_count, 0)
        self.assertEqual(len(slam.submaps), 0)

    def test_point_cloud(self):
        slam = self._make_slam()
        for _ in range(10):
            slam.process_frame(random_image())
        pc = slam.get_point_cloud()
        # Point cloud is empty in dummy mode (no depth attached to frames)
        self.assertEqual(pc.ndim, 2)
        self.assertEqual(pc.shape[1], 3)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("VGGT-SLAM 2.0 — Unit Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
