# Product Requirements Document
## VGGT-SLAM 2.0 — Real-time Dense Feed-forward Scene Reconstruction

**Version**: 0.1
**Paper**: arXiv:[2601.19887](https://arxiv.org/abs/2601.19887) — Maggio & Carlone
**Repo**: https://github.com/v1hns/worldModelsTest
**Status**: Active Development

---

## 1. Overview

VGGT-SLAM 2.0 is a real-time RGB SLAM (Simultaneous Localization and Mapping) system. It wraps the VGGT (Visual Geometry Grounded Transformer) feed-forward reconstruction model with a drift-correcting factor graph and an attention-based loop-closure system — enabling accurate 3D scene reconstruction from a monocular RGB camera with no depth sensor.

The core value proposition: **one model, one forward pass per frame, real-time pose + dense depth + point cloud.**

---

## 2. Problem Statement

Classical SLAM systems (ORB-SLAM, LIO-SAM, etc.) rely on hand-crafted feature extractors and explicit geometric solvers that break down in texture-less, dynamic, or cluttered environments. Learned monocular reconstruction models like VGGT solve this but suffer from:

- **15-DOF drift** — scale, rotation, and translation ambiguity accumulates over time in feed-forward predictions
- **No loop closure** — no mechanism to detect revisited places and correct accumulated error
- **Batch-only** — existing feed-forward models require all frames upfront, not suitable for online use

VGGT-SLAM 2.0 addresses all three.

---

## 3. Goals

| # | Goal | Priority |
|---|------|----------|
| G1 | Process RGB frames online (no offline batch required) | P0 |
| G2 | Remove 15-DOF drift via factor graph optimisation | P0 |
| G3 | Detect loop closures using VGGT attention features (no extra training) | P0 |
| G4 | Output globally-consistent camera trajectory + dense point cloud | P0 |
| G5 | Run on consumer GPU / edge device (Jetson-class) in real time | P1 |
| G6 | Provide a clean Python API for downstream use | P1 |
| G7 | Support open-set object detection on the reconstructed scene | P2 |

---

## 4. Non-Goals

- Training new models from scratch (we use VGGT pretrained weights)
- Supporting non-RGB sensors (LiDAR, stereo, depth cameras) in v1
- Web or mobile deployment
- Full production-grade GTSAM integration (v1 uses a lightweight optimizer)

---

## 5. System Architecture

```
RGB Frame Stream
      │
      ▼
┌─────────────────────┐
│   VGGT Backbone     │  ← facebook/VGGT-1B (HuggingFace)
│  (feed-forward)     │
└──────┬──────────────┘
       │  per-frame: extrinsics, intrinsics, depth, attention features
       ▼
┌─────────────────────┐      ┌──────────────────────┐
│  Submap Builder     │ ───► │  Attention Retrieval  │
│  (groups N frames)  │      │  (cosine-sim index)   │
└──────┬──────────────┘      └──────────┬───────────┘
       │                                │ loop-closure candidates
       ▼                                ▼
┌─────────────────────────────────────────────────┐
│              Factor Graph Optimizer              │
│  - odometry factors (consecutive submaps)        │
│  - loop-closure factors (retrieval matches)      │
│  - gauge fixing (removes 15-DOF ambiguity)       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
         Global Trajectory (N×4×4)
         Dense Point Cloud (M×3)
```

---

## 6. Core Components

### 6.1 VGGT Backbone (`slam/vggt_slam.py`)
- Loads `facebook/VGGT-1B` from HuggingFace
- Runs inference on a sliding window of frames
- Extracts: extrinsic poses (3×4), intrinsics (3×3), depth maps (H×W), attention feature vectors
- Falls back to **dummy mode** (random outputs) if VGGT package is not installed — all downstream components still function

### 6.2 Submap (`slam/submap.py`)
- Groups `N` frames (configurable, default 10) into a locally-consistent submap
- Back-projects depth maps using pinhole model → fused point cloud
- Stores anchor pose (first frame's pose) used by the factor graph

### 6.3 Factor Graph (`slam/factor_graph.py`)
- Collects odometry factors between consecutive submaps
- Collects loop-closure factors from retrieval
- Runs a Gauss-Newton pose-graph optimisation to minimise drift
- Returns corrected submap anchor poses

### 6.4 Attention Retrieval (`slam/retrieval.py`)
- Indexes L2-normalised VGGT attention features per frame
- Top-k cosine-similarity query for loop-closure candidate selection
- Configurable similarity threshold (default 0.75) to gate false positives

---

## 7. API

```python
from slam import VGGT_SLAM, SLAMConfig

slam = VGGT_SLAM(config=SLAMConfig(
    submap_size=10,
    loop_closure_k=5,
    loop_closure_thresh=0.75,
    device="cuda",
))

for rgb_frame in source:            # numpy H×W×3 uint8
    result = slam.process_frame(rgb_frame)
    result.pose        # 4×4 world-camera transform
    result.depth       # H×W depth map
    result.intrinsics  # 3×3

trajectory  = slam.get_trajectory()   # (N, 4, 4)
point_cloud = slam.get_point_cloud()  # (M, 3)
```

---

## 8. What's Been Done

### Completed

| Component | Status | Notes |
|-----------|--------|-------|
| `slam/vggt_slam.py` — main SLAM class | Done | Sliding-window VGGT inference, dummy fallback, frame buffer, submap trigger |
| `slam/submap.py` — submap + point cloud fusion | Done | Pinhole back-projection, anchor pose |
| `slam/factor_graph.py` — pose graph | Done | Odometry + loop-closure factors, Gauss-Newton optimizer |
| `slam/retrieval.py` — attention retrieval index | Done | Cosine-similarity, top-k query, min-distance gating |
| `slam/__init__.py` | Done | Clean public exports |
| `demo.py` | Done | Supports video file, image folder, synthetic frames; saves .npz + trajectory plot |
| `test_slam.py` | Done | 14 unit + integration tests, all passing |
| `requirements.txt` + `setup.py` | Done | Pip-installable package |
| `README.md` | Done | Setup, usage, API reference |
| GitHub push | Done | https://github.com/v1hns/worldModelsTest |

### Test Coverage

```
TestAttentionRetrieval   — empty query, add+query, reset
TestFactorGraph          — add submaps, loop closure, optimize, reset
TestSubmap               — point cloud with depth, point cloud without depth
TestVGGTSLAM             — single frame, multiple frames, submap creation, reset, point cloud

Ran 14 tests in 0.204s — OK
```

---

## 9. What's Left / Roadmap

### v0.2 — Accuracy
- [ ] Replace lightweight optimizer with **GTSAM** LevenbergMarquardt for full nonlinear pose-graph optimization
- [ ] ICP-refined relative pose for loop-closure factors (instead of assuming identity residual)
- [ ] Proper gauge-fixing: fix first pose + scale to remove all 15 DOF simultaneously
- [ ] Evaluate on **TUM RGB-D** benchmark and compare ATE against numbers in paper (23% improvement claim)

### v0.3 — Performance
- [ ] FAISS approximate nearest-neighbour index in `retrieval.py` for sub-millisecond lookup
- [ ] Batched VGGT inference with async frame queue
- [ ] Profile + optimize memory layout for Jetson deployment
- [ ] FP16 inference mode

### v0.4 — Features
- [ ] Real-time 3D visualisation (Viser / Open3D viewer)
- [ ] COLMAP-format pose export for Gaussian Splatting pipelines
- [ ] ROS2 node wrapper for robotics integration
- [ ] Open-set object detection overlay (paper Section V)

### v1.0 — Production
- [ ] Merge upstream VGGT-SLAM 2.0 official code when released
- [ ] Jetson Thor end-to-end benchmark
- [ ] Docker image with CUDA 12 + all dependencies

---

## 10. Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `submap_size` | 10 | Frames per submap |
| `loop_closure_k` | 5 | Top-k retrieval candidates per query |
| `loop_closure_thresh` | 0.75 | Cosine-similarity gate for accepting a loop |
| `device` | auto (cuda/cpu) | Inference device |
| `verbose` | True | Print submap + loop-closure events |

---

## 11. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0 | VGGT inference + tensor ops |
| torchvision | ≥0.15 | Image transforms |
| numpy | ≥1.24 | Array ops |
| Pillow | ≥10.0 | Image I/O |
| huggingface_hub | ≥0.20 | VGGT-1B weight download |
| opencv-python | ≥4.8 | Video + image loading in demo |
| matplotlib | ≥3.7 | Trajectory visualisation |
| scipy | ≥1.10 | Rotation utilities (future) |
| tqdm | ≥4.65 | Progress bars |
| vggt (from source) | latest | VGGT backbone |
