"""
demo.py — Run VGGT-SLAM on a video file or a folder of images.

Usage
-----
# From a video file:
python demo.py --source path/to/video.mp4

# From a folder of images (jpg/png, sorted by name):
python demo.py --source path/to/frames/

# Synthetic test (no real data needed):
python demo.py --synthetic --num_frames 50

Options
-------
--source        Path to a video file or image directory.
--synthetic     Generate random frames instead of reading a file.
--num_frames    Number of synthetic frames to generate (default: 30).
--submap_size   Frames per submap (default: 10).
--output        Save trajectory + point-cloud to this .npz file.
--visualize     Show a live matplotlib trajectory plot.
--export_colmap Export poses/intrinsics in COLMAP text format.
--benchmark_gt  Path to a TUM-style ground-truth file for ATE evaluation.
"""

import argparse
import sys
import os
import numpy as np


def load_frames_from_video(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_frames_from_dir(path: str):
    import cv2
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(
        f for f in os.listdir(path) if f.lower().endswith(exts)
    )
    frames = []
    for fn in files:
        img = cv2.imread(os.path.join(path, fn))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def synthetic_frames(n: int, H: int = 252, W: int = 336):
    """Generate n random RGB frames using VGGT-compatible dimensions."""
    frames = []
    for i in range(n):
        # Slowly shifting colour gradient to simulate motion
        base = np.zeros((H, W, 3), dtype=np.uint8)
        offset = int((i / n) * 200)
        base[:, :, 0] = np.clip(offset + np.random.randint(0, 30, (H, W)), 0, 255)
        base[:, :, 1] = np.clip(100 + np.random.randint(0, 30, (H, W)), 0, 255)
        base[:, :, 2] = np.clip(200 - offset + np.random.randint(0, 30, (H, W)), 0, 255)
        frames.append(base)
    return frames


def visualize_trajectory(trajectory: np.ndarray):
    """Simple 3-D scatter of camera positions."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for CI
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        positions = trajectory[:, :3, 3]   # (N, 3) translation vectors

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                "b-o", markersize=3, linewidth=1, label="Camera trajectory")
        ax.scatter(*positions[0], c="g", s=60, zorder=5, label="Start")
        ax.scatter(*positions[-1], c="r", s=60, zorder=5, label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("VGGT-SLAM 2.0 — Estimated Trajectory")
        ax.legend()
        plt.tight_layout()

        out_path = "trajectory.png"
        plt.savefig(out_path, dpi=120)
        print(f"[demo] Trajectory plot saved to {out_path}")
        plt.close()
    except Exception as e:
        print(f"[demo] Visualisation skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="VGGT-SLAM 2.0 demo")
    parser.add_argument("--source", default=None,
                        help="Path to video file or image directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic random frames")
    parser.add_argument("--num_frames", type=int, default=30,
                        help="Number of synthetic frames")
    parser.add_argument("--submap_size", type=int, default=10)
    parser.add_argument("--output", default=None,
                        help="Save results to .npz file")
    parser.add_argument("--visualize", action="store_true",
                        help="Save trajectory plot as trajectory.png")
    parser.add_argument("--export_colmap", default=None,
                        help="Export COLMAP text files into this directory")
    parser.add_argument("--benchmark_gt", default=None,
                        help="Path to TUM-format ground-truth trajectory")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable fp16 inference when running on CUDA")
    parser.add_argument("--async_inference", action="store_true",
                        help="Run inference through the async worker")
    parser.add_argument("--enable_detection", action="store_true",
                        help="Enable Grounding DINO object detection if installed")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load frames
    # ------------------------------------------------------------------
    if args.synthetic:
        print(f"[demo] Generating {args.num_frames} synthetic frames …")
        frames = synthetic_frames(args.num_frames)
    elif args.source is not None:
        if os.path.isdir(args.source):
            print(f"[demo] Loading frames from directory: {args.source}")
            frames = load_frames_from_dir(args.source)
        else:
            print(f"[demo] Loading frames from video: {args.source}")
            frames = load_frames_from_video(args.source)
    else:
        print("[demo] No source specified. Running with --synthetic (30 frames).")
        frames = synthetic_frames(30)

    print(f"[demo] {len(frames)} frames ready.")

    # ------------------------------------------------------------------
    # Run SLAM
    # ------------------------------------------------------------------
    from slam.vggt_slam import VGGT_SLAM, SLAMConfig
    from slam.benchmark import load_tum_ground_truth, run_tum_evaluation

    cfg = SLAMConfig(
        submap_size=args.submap_size,
        verbose=True,
        fp16=args.fp16,
        async_inference=args.async_inference,
        enable_object_detection=args.enable_detection,
    )
    slam = VGGT_SLAM(config=cfg)

    from tqdm import tqdm
    for frame in tqdm(frames, desc="Processing frames"):
        slam.process_frame(frame)

    trajectory = slam.get_trajectory()
    point_cloud = slam.get_point_cloud()

    print(f"\n[demo] Done.")
    print(f"       Frames processed : {slam._frame_count}")
    print(f"       Submaps created  : {len(slam.submaps)}")
    print(f"       Loop closures    : {slam.factor_graph.num_loop_closures}")
    print(f"       Trajectory shape : {trajectory.shape}")
    print(f"       Point cloud pts  : {len(point_cloud)}")

    # ------------------------------------------------------------------
    # Save / visualise
    # ------------------------------------------------------------------
    if args.output:
        np.savez_compressed(
            args.output,
            trajectory=trajectory,
            point_cloud=point_cloud,
        )
        print(f"[demo] Results saved to {args.output}")

    if args.export_colmap:
        slam.export_colmap(args.export_colmap)
        print(f"[demo] COLMAP export written to {args.export_colmap}")

    if args.visualize:
        visualize_trajectory(trajectory)

    if args.benchmark_gt:
        gt = load_tum_ground_truth(args.benchmark_gt)
        report = run_tum_evaluation(trajectory, gt, "tum_eval_report.json")
        print(f"[demo] TUM evaluation report saved to tum_eval_report.json: {report}")


if __name__ == "__main__":
    main()
