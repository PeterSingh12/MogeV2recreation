from pathlib import Path
import glob
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import json
import shutil

DATA_ROOT = Path("/Users/petersingh/Desktop/Datasetsmr/tartanair_tools")

ENVIRONMENTS = [
    "westerndesert",
]

samples = []

for env in ENVIRONMENTS:
    env_dir = DATA_ROOT / env / "Easy"

    if not env_dir.exists():
        print(f"Missing {env}")
        continue

    trajectories = sorted(
        d for d in env_dir.iterdir()
        if d.is_dir() and d.name.startswith("P")
    )

    print(f"{env}: {len(trajectories)} trajectories")

    for traj in trajectories:

        image_dir = traj / "image_left"
        depth_dir = traj / "depth_left"
        pose_file = traj / "pose_left.txt"

        if not (image_dir.exists() and depth_dir.exists() and pose_file.exists()):
            print(f"Skipping {traj}")
            continue

        images = sorted(image_dir.glob("*_left.png"))

        for img in images:

            stem = img.stem.replace("_left", "")

            depth = depth_dir / f"{stem}_left_depth.npy"

            if not depth.exists():
                continue

            samples.append(
                {
                    "env": env,
                    "traj": traj.name,
                    "image": img,
                    "depth": depth,
                    "pose": pose_file,
                }
            )


def load_poses(pose_file):
    poses = []

    with open(pose_file, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))

            if len(vals) != 7:
                raise ValueError(f"Expected 7 values, got {len(vals)}")

            poses.append(vals)

    return poses

def build_intrinsics():
    fx = 320.0
    fy = 320.0
    cx = 320.0
    cy = 240.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return K

def pose_to_matrix(pose_vec):
    """
    TartanAir pose format:
    [x, y, z, qx, qy, qz, qw]
    """

    x, y, z, qx, qy, qz, qw = pose_vec

    T = np.eye(4, dtype=np.float32)

    T[:3, :3] = Rotation.from_quat(
        [qx, qy, qz, qw]
    ).as_matrix()

    T[:3, 3] = [x, y, z]

    return T

from pathlib import Path

def write_one_sample(sample, sample_id=0):
    out_dir = Path("/Users/petersingh/Desktop/Datasetsmr/moge_dataset_tartanair") / f"sample{sample_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Image ----------
    img = cv2.imread(str(sample["image"]))
    height, width = img.shape[:2]

    cv2.imwrite(
        str(out_dir / "image.jpg"),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # ---------- Depth ----------
    depth = np.load(sample["depth"]).astype(np.float32)

    depth16 = (depth * 1000.0).astype(np.uint16)

    cv2.imwrite(
        str(out_dir / "depth.png"),
        depth16,
    )

    # ---------- Intrinsics ----------
    K = build_intrinsics()

    # ---------- Camera pose ----------
    frame_idx = int(sample["image"].stem.split("_")[0])

    poses = load_poses(sample["pose"])

    pose = pose_to_matrix(poses[frame_idx])

    meta = {
        "intrinsics": K.tolist(),
        "camera_pose": pose.tolist(),
        "image_size": [width, height],
        "depth_scale": 1.0,
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Finished", out_dir)

print()
print(f"Total frames: {len(samples)}")
print()

for s in samples[:5]:
    print(s)


print("\n----- Writing Sample -----")

OUTPUT_DIR = Path("/Users/petersingh/Desktop/Datasetsmr/moge_dataset_tartanair")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / ".index.txt", "w") as index:

    for i, sample in enumerate(samples):
        write_one_sample(sample, i)
        index.write(f"sample{i:06d}\n")

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(samples)}")