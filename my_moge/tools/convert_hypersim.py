#!/usr/bin/env python3
"""
convert_hypersim.py

Convert Apple Hypersim scenes into the dataset format expected by
Microsoft my_moge.

Output format:

output/
│
├── sample000000/
│   ├── image.jpg
│   ├── depth.png
│   └── meta.json
│
├── sample000001/
│   ├── image.jpg
│   ├── depth.png
│   └── meta.json
│
└── .index.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

# Microsoft's IO utilities
from utils.io import (
    write_depth,
    write_image,
    write_json,
)

###############################################################################
# HDF5
###############################################################################

def load_hdf5(path: Path):
    """
    Hypersim stores all arrays under the key 'dataset'.
    """

    with h5py.File(path, "r") as f:
        return f["dataset"][:]


###############################################################################
# Camera
###############################################################################

def load_camera(scene_root: Path):
    """
    Load camera poses for one camera trajectory.

    Returns
    -------
    positions : (N,3)
    rotations : (N,3,3)
    """

    cam = scene_root / "_detail" / "cam_00"

    positions = load_hdf5(
        cam / "camera_keyframe_positions.hdf5"
    )

    rotations = load_hdf5(
        cam / "camera_keyframe_orientations.hdf5"
    )

    return positions, rotations


###############################################################################
# Scene scale
###############################################################################

def load_scene_scale(scene_root: Path):
    """
    Read meters_per_asset_unit from metadata_scene.csv.
    """

    csv_file = scene_root / "_detail" / "metadata_scene.csv"

    meters = 1.0

    with open(csv_file) as f:

        next(f)

        for line in f:

            name, value = line.strip().split(",")

            if name == "meters_per_asset_unit":
                meters = float(value)

    return meters


###############################################################################
# Intrinsics
###############################################################################

def build_intrinsics(width, height, fov_deg=60.0):
    """
    Temporary intrinsic matrix.

    NOTE:
    Hypersim derives camera intrinsics from its rendering
    camera. We'll replace this with the exact computation
    after verifying the camera specification.
    """

    fov = np.deg2rad(fov_deg)

    fx = width / (2.0 * np.tan(fov / 2.0))
    fy = fx

    cx = width / 2.0
    cy = height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


###############################################################################
# Output folders
###############################################################################

def make_sample_folder(out_root: Path, index: int):

    folder = out_root / f"sample{index:06d}"

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return folder


###############################################################################
# Frame discovery
###############################################################################

def discover_frames(scene_root: Path):
    """
    Find matching RGB + depth files.

    Returns
    -------
    List[(rgb_path, depth_path)]
    """

    rgb_root = (
    scene_root
    / "images"
    / "scene_cam_00_geometry_preview"
    )

    depth_root = (
    scene_root
    / "images"
    / "scene_cam_00_geometry_hdf5"
    )

    rgb_files = sorted(
        rgb_root.glob("frame.*.color.jpg")
    )

    frames = []

    for rgb in rgb_files:

        frame_id = rgb.stem.split(".")[1]

        depth = (
            depth_root
            / f"frame.{frame_id}.depth_meters.hdf5"
        )

        if depth.exists():

            frames.append(
                (
                    rgb,
                    depth,
                )
            )

    return frames


###############################################################################
# Argument parser
###############################################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Hypersim root directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="my_moge dataset output",
    )

    return parser.parse_args()

###############################################################################
# Sample conversion
###############################################################################

def convert_frame(
    rgb_path: Path,
    depth_path: Path,
    out_dir: Path,
    position: np.ndarray,
    rotation: np.ndarray,
    meters_per_asset_unit: float,
):
    """
    Convert one Hypersim frame into one my_moge sample.
    """

    # ------------------------------------------------------------------
    # RGB
    # ------------------------------------------------------------------

    image = imageio.imread(rgb_path)

    height, width = image.shape[:2]

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------

    depth = load_hdf5(depth_path).astype(np.float32)

    # Hypersim stores asset units.
    # Convert to meters.

    depth *= meters_per_asset_unit

    # ------------------------------------------------------------------
    # Intrinsics
    # ------------------------------------------------------------------

    K = build_intrinsics(
        width,
        height,
    )

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    pose = np.eye(4, dtype=np.float32)

    pose[:3, :3] = rotation.astype(np.float32)

    pose[:3, 3] = position.astype(np.float32)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------

    write_image(
        out_dir / "image.jpg",
        image,
    )

    write_depth(
        out_dir / "depth.png",
        depth,
    )

    meta = {

        "intrinsics": K.tolist(),

        "camera_pose": pose.tolist(),

        "image_size": [

            int(width),

            int(height),

        ],

        "depth_scale": 1.0,

    }

    write_json(
        out_dir / "meta.json",
        meta,
    )

###############################################################################
# Scene conversion
###############################################################################

def convert_scene(
    scene_root: Path,
    out_root: Path,
    start_index: int,
):
    """
    Convert one Hypersim scene.
    """

    print(f"\nConverting {scene_root.name}")

    positions, rotations = load_camera(scene_root)

    scale = load_scene_scale(scene_root)

    frames = discover_frames(scene_root)

    dataset_index = []

    sample_index = start_index

    for i, (rgb, depth) in enumerate(
        tqdm(frames)
    ):

        if i >= len(positions):

            break

        sample_dir = make_sample_folder(

            out_root,

            sample_index,

        )

        convert_frame(

            rgb,

            depth,

            sample_dir,

            positions[i],

            rotations[i],

            scale,

        )

        dataset_index.append(

            sample_dir.name

        )

        sample_index += 1

    return dataset_index, sample_index

###############################################################################
# Dataset conversion
###############################################################################

def convert_dataset(
    input_root: Path,
    output_root: Path,
):
    """
    Convert every Hypersim scene under the input directory.
    """

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_dirs = sorted(

        [

            d

            for d in input_root.iterdir()

            if d.is_dir()

            and d.name.startswith("ai_")

        ]

    )

    print(f"Found {len(scene_dirs)} scenes.")

    global_index = []

    sample_index = 0

    for scene in scene_dirs:

        scene_index, sample_index = convert_scene(

            scene,

            output_root,

            sample_index,

        )

        global_index.extend(scene_index)

    with open(

        output_root / ".index.txt",

        "w",

    ) as f:

        for item in global_index:

            f.write(item + "\n")

    print()

    print("=" * 60)

    print(f"Finished converting {len(global_index)} samples.")

    print("=" * 60)

###############################################################################
# Main
###############################################################################

def main():

    args = parse_args()

    convert_dataset(

        args.input,

        args.output,

    )


if __name__ == "__main__":

    main()