"""
test_slam.py — tests for the SLAM pipeline and model interface.

Structure
---------
TestModelContract       — every model must pass these (parametrized over all models)
TestDummyModel          — DummyModel-specific behaviour
TestVGGTModel           — VGGTModel-specific behaviour (dummy-fallback, no weights needed)
TestAttentionRetrieval  — retrieval index unit tests
TestFactorGraph         — factor graph unit tests
TestSubmap              — submap / point-cloud unit tests
TestSLAMPipeline        — integration tests (runs with DummyModel, no weights needed)

Run with:  python test_slam.py
"""

import sys
import numpy as np
import torch
import unittest


def random_image(H=240, W=320):
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)


# ============================================================
# Contract tests — any model must satisfy these
# ============================================================

class TestModelContract:
    """
    Mixin: subclass + set self.model in setUp() to run the full contract.
    Each concrete model test class inherits this and overrides setUp().
    """

    model = None   # set in subclass

    def test_name_is_string(self):
        self.assertIsInstance(self.model.name, str)
        self.assertGreater(len(self.model.name), 0)

    def test_load_does_not_raise(self):
        self.model.load("cpu")   # must not raise

    def test_preprocess_output_shape(self):
        img = random_image(60, 80)
        t = self.model.preprocess(img)
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, (3, 60, 80))

    def test_preprocess_value_range(self):
        img = (np.ones((60, 80, 3)) * 128).astype(np.uint8)
        t = self.model.preprocess(img)
        self.assertGreaterEqual(float(t.min()), 0.0)
        self.assertLessEqual(float(t.max()), 1.0 + 1e-5)

    def test_predict_single_frame(self):
        img = random_image(60, 80)
        t = self.model.preprocess(img)
        pred = self.model.predict([t])
        self.assertEqual(pred.extrinsics.shape, (1, 3, 4))
        self.assertEqual(pred.intrinsics.shape, (1, 3, 3))
        self.assertEqual(pred.depth.shape[0], 1)
        self.assertEqual(len(pred.retrieval_features), 1)

    def test_predict_multi_frame(self):
        tensors = [self.model.preprocess(random_image(60, 80)) for _ in range(4)]
        pred = self.model.predict(tensors)
        self.assertEqual(pred.extrinsics.shape, (4, 3, 4))
        self.assertEqual(pred.depth.shape[0], 4)
        self.assertEqual(len(pred.retrieval_features), 4)

    def test_retrieval_features_are_tensors(self):
        t = self.model.preprocess(random_image(60, 80))
        pred = self.model.predict([t])
        for feat in pred.retrieval_features:
            self.assertIsInstance(feat, torch.Tensor)
            self.assertEqual(feat.ndim, 1)


# ============================================================
# Concrete model test classes
# ============================================================

class TestDummyModel(TestModelContract, unittest.TestCase):

    def setUp(self):
        from slam.models import DummyModel
        self.model = DummyModel(feat_dim=64, seed=0)
        self.model.load("cpu")

    def test_determinism(self):
        """Same seed → same predictions."""
        from slam.models import DummyModel
        m1 = DummyModel(seed=7)
        m2 = DummyModel(seed=7)
        m1.load("cpu")
        m2.load("cpu")
        t = m1.preprocess(random_image(30, 40))
        p1 = m1.predict([t])
        p2 = m2.predict([t])
        self.assertTrue(torch.allclose(p1.extrinsics, p2.extrinsics))

    def test_different_seeds_differ(self):
        from slam.models import DummyModel
        m1 = DummyModel(seed=1)
        m2 = DummyModel(seed=2)
        m1.load("cpu")
        m2.load("cpu")
        t = m1.preprocess(random_image(30, 40))
        p1 = m1.predict([t])
        p2 = m2.predict([t])
        self.assertFalse(torch.allclose(p1.extrinsics, p2.extrinsics))


class TestVGGTModel(TestModelContract, unittest.TestCase):
    """Runs contract tests against VGGTModel in dummy-fallback mode (no weights)."""

    def setUp(self):
        from slam.models import VGGTModel
        self.model = VGGTModel()
        self.model.load("cpu")   # will warn + fall back to dummy outputs


# ============================================================
# Component unit tests
# ============================================================

class TestAttentionRetrieval(unittest.TestCase):

    def setUp(self):
        from slam.retrieval import AttentionRetrieval
        self.ret = AttentionRetrieval(device="cpu")

    def test_empty_query(self):
        self.assertEqual(self.ret.query(0, k=5), [])

    def test_add_and_query(self):
        for i in range(30):
            self.ret.add(i, torch.rand(256))
        results = self.ret.query(29, k=3, min_distance=10)
        self.assertLessEqual(len(results), 3)
        for fid, score in results:
            self.assertLess(fid, 20)
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
        return Submap(submap_id=submap_id, frames=frames, poses=[pose])

    def setUp(self):
        from slam.factor_graph import FactorGraph
        self.fg = FactorGraph(device="cpu")

    def test_add_submaps(self):
        for i in range(4):
            pose = np.eye(4); pose[0, 3] = float(i)
            self.fg.add_submap(self._make_submap(pose, submap_id=i))
        self.assertEqual(self.fg.num_submaps, 4)
        self.assertEqual(len(self.fg.odometry_factors), 3)

    def test_loop_closure(self):
        for i in range(6):
            pose = np.eye(4); pose[0, 3] = float(i)
            self.fg.add_submap(self._make_submap(pose, submap_id=i))
        self.fg.add_loop_closure(0, 5, score=0.9)
        self.assertEqual(self.fg.num_loop_closures, 1)

    def test_optimize_returns_poses(self):
        for i in range(4):
            pose = np.eye(4); pose[0, 3] = float(i) * 0.5
            self.fg.add_submap(self._make_submap(pose, submap_id=i))
        self.fg.add_loop_closure(0, 3, score=0.8)
        optimised = self.fg.optimize(iterations=5)
        self.assertEqual(len(optimised), 4)
        for p in optimised:
            self.assertEqual(p.shape, (4, 4))

    def test_reset(self):
        self.fg.add_submap(self._make_submap(np.eye(4)))
        self.fg.reset()
        self.assertEqual(self.fg.num_submaps, 0)


class TestSubmap(unittest.TestCase):

    def test_build_point_cloud_with_depth(self):
        from slam.submap import Submap
        H, W = 60, 80
        depth = np.ones((H, W), dtype=np.float32) * 2.0
        frame = {"image": random_image(H, W), "tensor": torch.rand(3, H, W), "depth": depth}
        sm = Submap(submap_id=0, frames=[frame], poses=[np.eye(4, dtype=np.float32)])
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


# ============================================================
# Integration tests — pipeline with DummyModel
# ============================================================

class TestSLAMPipeline(unittest.TestCase):

    def _make_pipeline(self, **kwargs):
        from slam import SLAMPipeline, SLAMConfig
        from slam.models import DummyModel
        cfg = SLAMConfig(submap_size=5, verbose=False, device="cpu", **kwargs)
        return SLAMPipeline(model=DummyModel(), config=cfg)

    def test_single_frame(self):
        p = self._make_pipeline()
        r = p.process_frame(random_image())
        self.assertEqual(r.frame_id, 0)
        self.assertEqual(r.pose.shape, (4, 4))

    def test_multiple_frames_trajectory_length(self):
        p = self._make_pipeline()
        N = 12
        for _ in range(N):
            p.process_frame(random_image())
        traj = p.get_trajectory()
        self.assertEqual(traj.shape, (N, 4, 4))

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

    def test_swap_model_at_construction(self):
        """Swapping the model changes predictions but the pipeline API is identical."""
        from slam import SLAMPipeline, SLAMConfig
        from slam.models import DummyModel
        cfg = SLAMConfig(submap_size=5, verbose=False, device="cpu")
        p1 = SLAMPipeline(model=DummyModel(seed=1), config=cfg)
        p2 = SLAMPipeline(model=DummyModel(seed=99), config=cfg)
        # Process 2 frames so the second frame uses real model predictions
        for _ in range(2):
            r1 = p1.process_frame(random_image())
            r2 = p2.process_frame(random_image())
        # API shape is the same regardless of model
        self.assertEqual(r1.pose.shape, r2.pose.shape)
        # Trajectories differ because models differ
        t1 = p1.get_trajectory()
        t2 = p2.get_trajectory()
        self.assertFalse(np.allclose(t1, t2))


# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SLAM Pipeline — Model-Agnostic Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
