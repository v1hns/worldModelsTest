# VGGT-SLAM 2.0

Implementation based on **"VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction"**
Maggio & Carlone — arXiv:[2601.19887](https://arxiv.org/abs/2601.19887)

---

## What is this?

VGGT-SLAM 2.0 is a real-time RGB SLAM system that wraps the [VGGT](https://github.com/facebookresearch/vggt) feed-forward reconstruction model with:

| Component | What it does |
|-----------|--------------|
| **VGGT backbone** | Predicts per-frame camera poses, depth maps, and point maps in a single forward pass |
| **Factor graph** | Removes 15-DOF drift (scale/rotation/translation ambiguity) from monocular feed-forward predictions |
| **Attention retrieval** | Uses VGGT attention features for loop-closure detection without extra training |
| **Submaps** | Groups frames into locally-consistent submaps; aligns them via loop-closure constraints |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install VGGT (optional, needed for real reconstruction)

```bash
git clone https://github.com/facebookresearch/vggt.git
pip install -e ./vggt
```

> Without VGGT installed the system runs in **dummy mode** — all pipeline
> components work, but pose/depth values are random placeholders.

### 3. Run tests

```bash
python test_slam.py
```

### 4. Run the demo

```bash
# Synthetic frames (no data needed)
python demo.py --synthetic --num_frames 50 --visualize

# From a video file
python demo.py --source path/to/video.mp4 --visualize --output results.npz

# From a folder of images
python demo.py --source path/to/frames/ --submap_size 8 --output results.npz
```

---

## Code structure

```
worldModelsTest/
├── slam/
│   ├── __init__.py
│   ├── vggt_slam.py      # Main SLAM class — entry point for processing frames
│   ├── submap.py         # Submap: groups frames + fuses point cloud
│   ├── factor_graph.py   # Pose-graph optimisation (drift removal + loop closures)
│   └── retrieval.py      # Cosine-similarity attention feature index
├── demo.py               # CLI demo: video / image-folder / synthetic
├── test_slam.py          # Unit + integration tests
├── requirements.txt
└── setup.py
```

---

## Using the API

```python
from slam import VGGT_SLAM, SLAMConfig
import numpy as np

cfg = SLAMConfig(
    submap_size=10,           # frames per submap
    loop_closure_k=5,         # top-k retrieval candidates
    loop_closure_thresh=0.75, # cosine-similarity gate
    device="cuda",            # or "cpu"
    verbose=True,
)

slam = VGGT_SLAM(config=cfg)

for rgb_frame in your_video_source:          # numpy HxWx3 uint8
    result = slam.process_frame(rgb_frame)
    print(result.pose)                       # 4x4 world-camera transform
    print(result.depth.shape)                # HxW depth map

trajectory  = slam.get_trajectory()          # (N, 4, 4)
point_cloud = slam.get_point_cloud()         # (M, 3)
```

---

## Notes

- The official VGGT-SLAM 2.0 code has **not yet been released** (paper says "code will be released upon publication"). This repo implements the core architecture from the paper description.
- The factor-graph optimiser here is a lightweight linear relaxation. For production accuracy, swap it with [GTSAM](https://gtsam.org/) or [g2o](https://github.com/RainerKuemmerle/g2o).
- Tested on Python 3.10, PyTorch 2.3, CUDA 12.1.
