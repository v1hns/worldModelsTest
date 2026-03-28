"""
Unit and integration tests for the VGGT-SLAM pipeline.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

from slam import (
    Open3DViewer,
    ROS2SLAMNode,
    SLAMConfig,
    VGGT_SLAM,
    compute_ate,
    export_colmap_model,
    load_tum_ground_truth,
    run_tum_evaluation,
)
from slam.factor_graph import FactorGraph
from slam.retrieval import AttentionRetrieval
from slam.submap import Submap


def random_image(H=96, W=128):
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)


class TestAttentionRetrieval(unittest.TestCase):
    def test_add_and_query(self):
        retrieval = AttentionRetrieval(device="cpu", backend="bruteforce")
        for idx in range(20):
            vec = torch.zeros(8)
            vec[idx % 8] = 1.0
            retrieval.add(idx, vec)

        results = retrieval.query(19, k=3, min_distance=4)
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all(fid <= 15 for fid, _ in results))

    def test_reset(self):
        retrieval = AttentionRetrieval(device="cpu", backend="bruteforce")
        retrieval.add(0, torch.rand(8))
        retrieval.reset()
        self.assertEqual(retrieval.size, 0)


class TestFactorGraph(unittest.TestCase):
    def _submap(self, submap_id: int, tx: float):
        frames = [
            {
                "frame_id": submap_id,
                "image": random_image(),
                "depth": np.ones((16, 16), dtype=np.float32),
                "intrinsics": np.eye(3, dtype=np.float32),
                "detections": [],
            }
        ]
        local_poses = [np.eye(4, dtype=np.float32)]
        anchor = np.eye(4, dtype=np.float32)
        anchor[0, 3] = tx
        return Submap(
            submap_id=submap_id,
            frame_ids=[submap_id],
            frames=frames,
            local_poses=local_poses,
            anchor_pose=anchor,
        )

    def test_add_submaps_and_loop(self):
        fg = FactorGraph(device="cpu", backend="linear")
        for idx in range(3):
            fg.add_submap(self._submap(idx, float(idx)))
        rel = np.eye(4, dtype=np.float32)
        rel[0, 3] = 0.2
        fg.add_loop_closure(0, 2, rel, score=0.9)
        self.assertEqual(fg.num_submaps, 3)
        self.assertEqual(fg.num_loop_closures, 1)

    def test_optimize_returns_poses(self):
        fg = FactorGraph(device="cpu", backend="linear")
        for idx in range(4):
            fg.add_submap(self._submap(idx, float(idx)))
        rel = np.eye(4, dtype=np.float32)
        rel[0, 3] = 0.1
        fg.add_loop_closure(0, 3, rel, score=0.8)
        poses = fg.optimize(iterations=4)
        self.assertEqual(len(poses), 4)
        self.assertEqual(poses[0].shape, (4, 4))


class TestSubmap(unittest.TestCase):
    def test_build_point_cloud_with_depth(self):
        depth = np.ones((20, 24), dtype=np.float32) * 2.0
        frame = {
            "frame_id": 0,
            "image": random_image(20, 24),
            "depth": depth,
            "intrinsics": np.array([[12.0, 0.0, 12.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            "detections": [],
        }
        submap = Submap(
            submap_id=0,
            frame_ids=[0],
            frames=[frame],
            local_poses=[np.eye(4, dtype=np.float32)],
            anchor_pose=np.eye(4, dtype=np.float32),
        )
        submap.build_point_cloud(sample_stride=2)
        self.assertGreater(len(submap.point_cloud_local), 0)
        self.assertEqual(submap.point_cloud.shape[1], 3)


class TestBenchmarkAndExport(unittest.TestCase):
    def test_compute_ate(self):
        pred = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0)
        gt = pred.copy()
        gt[1, 0, 3] = 1.0
        gt[2, 0, 3] = 2.0
        pred[1, 0, 3] = 1.1
        pred[2, 0, 3] = 2.1
        ate = compute_ate(pred, gt)
        self.assertGreaterEqual(ate, 0.0)

    def test_tum_ground_truth_loading_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = os.path.join(tmpdir, "groundtruth.txt")
            with open(gt_path, "w", encoding="utf-8") as fh:
                fh.write("0.0 0 0 0 0 0 0 1\n")
                fh.write("0.1 1 0 0 0 0 0 1\n")
            gt = load_tum_ground_truth(gt_path)
            self.assertEqual(gt.shape, (2, 4, 4))

            out_path = os.path.join(tmpdir, "report.json")
            report = run_tum_evaluation(gt, gt, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("ate_rmse", report)


class TestVGGTSLAM(unittest.TestCase):
    def _make_slam(self, **kwargs):
        cfg = SLAMConfig(
            submap_size=3,
            device="cpu",
            verbose=False,
            optimizer_backend="linear",
            retrieval_backend="bruteforce",
            async_inference=False,
            **kwargs,
        )
        return VGGT_SLAM(config=cfg)

    def test_public_exports(self):
        from slam import FrameResult, SLAMConfig as ExportedConfig

        self.assertIs(ExportedConfig, SLAMConfig)
        self.assertEqual(FrameResult.__name__, "FrameResult")

    def test_single_frame(self):
        p = self._make_slam()
        r = p.process_frame(random_image())
        self.assertEqual(r.frame_id, 0)
        self.assertEqual(r.pose.shape, (4, 4))

    def test_submap_creation_and_point_cloud(self):
        slam = self._make_slam()
        for _ in range(6):
            slam.process_frame(random_image())
        self.assertGreaterEqual(len(slam.submaps), 2)
        point_cloud = slam.get_point_cloud()
        self.assertEqual(point_cloud.shape[1], 3)
        self.assertGreater(len(point_cloud), 0)

    def test_optimized_anchors_propagate_to_trajectory(self):
        slam = self._make_slam()
        for _ in range(6):
            slam.process_frame(random_image())
        baseline = slam.get_trajectory().copy()

        shifted = [submap.optimized_anchor_pose.copy() for submap in slam.submaps]
        shifted[-1][:3, 3] += np.array([2.0, 0.0, 0.0], dtype=np.float32)
        slam.factor_graph.optimize = lambda iterations=10: shifted  # type: ignore[assignment]
        slam._optimize_submaps()

        updated = slam.get_trajectory()
        self.assertFalse(np.allclose(baseline[-1], updated[-1]))

    def test_loop_closure_uses_submap_ids(self):
        slam = self._make_slam()
        for _ in range(9):
            slam.process_frame(random_image())

        current_submap = slam.submaps[-1]
        slam.retrieval.query = lambda query_idx, k=5, min_distance=6: [(0, 0.95)]  # type: ignore[assignment]
        recorded = {}

        def capture(submap_i, submap_j, relative_pose, score, source_frame_i=None, source_frame_j=None):
            recorded["submap_i"] = submap_i
            recorded["submap_j"] = submap_j
            recorded["source_frame_i"] = source_frame_i
            recorded["source_frame_j"] = source_frame_j

        slam.factor_graph.add_loop_closure = capture  # type: ignore[assignment]
        slam._try_loop_closure(current_submap)
        self.assertEqual(recorded["submap_i"], 0)
        self.assertEqual(recorded["submap_j"], current_submap.submap_id)
        self.assertEqual(recorded["source_frame_i"], 0)

    def test_colmap_export(self):
        slam = self._make_slam()
        for _ in range(3):
            slam.process_frame(random_image())

        with tempfile.TemporaryDirectory() as tmpdir:
            export_colmap_model(slam, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "cameras.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "images.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "points3D.txt")))

    def test_viewer_and_ros_wrapper(self):
        slam = self._make_slam()
        for _ in range(3):
            slam.process_frame(random_image())

        viewer = Open3DViewer()
        viewer.update(slam.get_trajectory(), slam.get_point_cloud(), slam.get_labeled_objects())
        self.assertIn("trajectory", viewer.last_geometry)

        ros_node = ROS2SLAMNode(slam)
        result = ros_node.process_rgb_frame(random_image())
        self.assertEqual(result.pose.shape, (4, 4))


# ============================================================
# Model-agnostic contract tests
# ============================================================

class _ModelContractMixin:
    """
    Inherit from this + unittest.TestCase and set self.model in setUp()
    to run the full interface contract against any BaseReconstructionModel.
    """

    model = None

    def test_name_is_nonempty_string(self):
        self.assertIsInstance(self.model.name, str)
        self.assertGreater(len(self.model.name), 0)

    def test_load_does_not_raise(self):
        self.model.load("cpu")

    def test_preprocess_shape_and_dtype(self):
        t = self.model.preprocess(random_image(60, 80))
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, (3, 60, 80))

    def test_preprocess_value_range(self):
        img = (np.ones((60, 80, 3)) * 200).astype(np.uint8)
        t = self.model.preprocess(img)
        self.assertLessEqual(float(t.max()), 1.0 + 1e-5)
        self.assertGreaterEqual(float(t.min()), 0.0)

    def test_predict_single_frame_shapes(self):
        t = self.model.preprocess(random_image(60, 80))
        pred = self.model.predict([t])
        self.assertEqual(pred.extrinsics.shape, (1, 3, 4))
        self.assertEqual(pred.intrinsics.shape, (1, 3, 3))
        self.assertEqual(pred.depth.shape[0], 1)
        self.assertEqual(len(pred.retrieval_features), 1)

    def test_predict_multi_frame_shapes(self):
        tensors = [self.model.preprocess(random_image(60, 80)) for _ in range(4)]
        pred = self.model.predict(tensors)
        self.assertEqual(pred.extrinsics.shape, (4, 3, 4))
        self.assertEqual(pred.depth.shape[0], 4)
        self.assertEqual(len(pred.retrieval_features), 4)

    def test_retrieval_features_are_1d_tensors(self):
        t = self.model.preprocess(random_image(60, 80))
        pred = self.model.predict([t])
        for feat in pred.retrieval_features:
            self.assertIsInstance(feat, torch.Tensor)
            self.assertEqual(feat.ndim, 1)


class TestDummyModelContract(_ModelContractMixin, unittest.TestCase):
    def setUp(self):
        from slam.models import DummyModel
        self.model = DummyModel(feat_dim=64, seed=0)
        self.model.load("cpu")

    def test_determinism(self):
        from slam.models import DummyModel
        m1 = DummyModel(seed=7)
        m2 = DummyModel(seed=7)
        m1.load("cpu")
        m2.load("cpu")
        t = m1.preprocess(random_image(30, 40))
        self.assertTrue(torch.allclose(m1.predict([t]).extrinsics, m2.predict([t]).extrinsics))

    def test_different_seeds_differ(self):
        from slam.models import DummyModel
        m1 = DummyModel(seed=1)
        m2 = DummyModel(seed=2)
        m1.load("cpu")
        m2.load("cpu")
        t = m1.preprocess(random_image(30, 40))
        self.assertFalse(torch.allclose(m1.predict([t]).extrinsics, m2.predict([t]).extrinsics))


class TestVGGTModelContract(_ModelContractMixin, unittest.TestCase):
    """Contract tests against VGGTModel in dummy-fallback mode (no weights needed)."""

    def setUp(self):
        from slam.models import VGGTModel
        self.model = VGGTModel()
        self.model.load("cpu")


class TestSLAMPipeline(unittest.TestCase):
    """Integration tests for the model-agnostic SLAMPipeline."""

    def _make_pipeline(self, **kwargs):
        from slam import SLAMPipeline, SLAMConfig
        from slam.models import DummyModel
        cfg = SLAMConfig(
            submap_size=5, verbose=False, device="cpu",
            optimizer_backend="linear", retrieval_backend="bruteforce",
            async_inference=False, **kwargs,
        )
        return SLAMPipeline(model=DummyModel(), config=cfg)

    def test_single_frame(self):
        p = self._make_pipeline()
        r = p.process_frame(random_image())
        self.assertEqual(r.frame_id, 0)
        self.assertEqual(r.pose.shape, (4, 4))

    def test_trajectory_length(self):
        p = self._make_pipeline()
        N = 12
        for _ in range(N):
            p.process_frame(random_image())
        self.assertEqual(p.get_trajectory().shape, (N, 4, 4))

    def test_submap_creation(self):
        p = self._make_pipeline()
        for _ in range(10):
            p.process_frame(random_image())
        self.assertGreaterEqual(len(p.submaps), 1)

    def test_point_cloud_shape(self):
        p = self._make_pipeline()
        for _ in range(10):
            p.process_frame(random_image())
        pc = p.get_point_cloud()
        self.assertEqual(pc.ndim, 2)
        self.assertEqual(pc.shape[1], 3)

    def test_reset(self):
        p = self._make_pipeline()
        for _ in range(6):
            p.process_frame(random_image())
        p.reset()
        self.assertEqual(p._frame_count, 0)
        self.assertEqual(len(p.submaps), 0)

    def test_swap_model_gives_different_trajectories(self):
        from slam import SLAMPipeline, SLAMConfig
        from slam.models import DummyModel
        cfg = SLAMConfig(
            submap_size=5, verbose=False, device="cpu",
            optimizer_backend="linear", retrieval_backend="bruteforce",
            async_inference=False,
        )
        p1 = SLAMPipeline(model=DummyModel(seed=1), config=cfg)
        p2 = SLAMPipeline(model=DummyModel(seed=99), config=cfg)
        for _ in range(3):
            p1.process_frame(random_image())
            p2.process_frame(random_image())
        self.assertEqual(p1.get_trajectory().shape, p2.get_trajectory().shape)
        self.assertFalse(np.allclose(p1.get_trajectory(), p2.get_trajectory()))


if __name__ == "__main__":
    print("=" * 60)
    print("VGGT-SLAM 2.0 — Tests")
    print("=" * 60)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
