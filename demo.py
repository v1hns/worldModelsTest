"""
demo.py — Run the SLAM pipeline on any supported model.

Usage
-----
# Synthetic frames, default model (vggt):
python demo.py --synthetic --num_frames 50

# Choose a model explicitly:
python demo.py --model vggt   --source path/to/video.mp4
python demo.py --model dummy  --synthetic --num_frames 30

# Save results:
python demo.py --synthetic --output results.npz --visualize

Options
-------
--model         Model to use: vggt | dummy  (default: vggt)
--source        Path to a video file or image directory.
--synthetic     Generate random frames instead of reading a file.
--num_frames    Number of synthetic frames (default: 30).
--submap_size   Frames per submap (default: 10).
--output        Save trajectory + point-cloud to this .npz file.
--visualize     Save trajectory plot as trajectory.png.
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


def synthetic_frames(n: int, H: int = 240, W: int = 320):
    """Generate n random RGB frames (simulates a camera moving through noise)."""
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
        ax.set_title("SLAM Pipeline — Estimated Trajectory")
        ax.legend()
        plt.tight_layout()

        out_path = "trajectory.png"
        plt.savefig(out_path, dpi=120)
        print(f"[demo] Trajectory plot saved to {out_path}")
        plt.close()
    except Exception as e:
        print(f"[demo] Visualisation skipped: {e}")


MODELS = {
    "vggt":  lambda: __import__("slam.models", fromlist=["VGGTModel"]).VGGTModel(),
    "dummy": lambda: __import__("slam.models", fromlist=["DummyModel"]).DummyModel(),
}


def main():
    parser = argparse.ArgumentParser(description="SLAM pipeline demo")
    parser.add_argument("--model", default="vggt", choices=list(MODELS),
                        help="Model backend to use (default: vggt)")
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
    from slam import SLAMPipeline, SLAMConfig

    model = MODELS[args.model]()
    print(f"[demo] Using model: {model.name}")
    cfg = SLAMConfig(submap_size=args.submap_size, verbose=True)
    slam = SLAMPipeline(model=model, config=cfg)

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

    if args.visualize:
        visualize_trajectory(trajectory)


if __name__ == "__main__":
    main()
