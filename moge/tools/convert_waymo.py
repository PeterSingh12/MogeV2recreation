#!/usr/bin/env python3

from pathlib import Path
import argparse
import io
import cv2

import tensorflow as tf
import pyarrow.parquet as pq
import numpy as np
from PIL import Image

from moge.utils.io import (
    write_image,
    write_json,
)
###############################################################################
# Arguments
###############################################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()

###############################################################################
# Read parquet
###############################################################################

def load_table(path):

    return pq.read_table(path).to_pandas()

###############################################################################
# LiDAR
###############################################################################

def load_lidar(root):

    lidar = load_table(
        next((root / "lidar").glob("*.parquet"))
    )

    calib = load_table(
        next((root / "lidar_calibration").glob("*.parquet"))
    )

    pose = load_table(
        next((root / "lidar_pose").glob("*.parquet"))
    )

    projection = load_table(
        next((root / "lidar_camera_projection").glob("*.parquet"))
    )

    return lidar, calib, pose, projection

###############################################################################
# Range image
###############################################################################

def range_image(row):

    values = np.asarray(

        row["[LiDARComponent].range_image_return1.values"],

        dtype=np.float32,

    )

    shape = row["[LiDARComponent].range_image_return1.shape"]

    return values.reshape(shape)

###############################################################################
# Camera projection
###############################################################################

def camera_projection(row):

    values = np.asarray(

        row["[LiDARCameraProjectionComponent].range_image_return1.values"],

        dtype=np.float32,

    )

    shape = row["[LiDARCameraProjectionComponent].range_image_return1.shape"]

    return values.reshape(shape)

###############################################################################
# LiDAR pose
###############################################################################

def lidar_pose(row):

    values = np.asarray(

        row["[LiDARPoseComponent].range_image_return1.values"],

        dtype=np.float32,

    )

    shape = row["[LiDARPoseComponent].range_image_return1.shape"]

    return values.reshape(shape)

###############################################################################
# Decode jpeg
###############################################################################

def decode_image(image_bytes):

    return np.asarray(
        Image.open(io.BytesIO(image_bytes))
    )

###############################################################################
# Intrinsics
###############################################################################

def build_intrinsics(row):

    return np.array(

        [

            [
                row["[CameraCalibrationComponent].intrinsic.f_u"],
                0,
                row["[CameraCalibrationComponent].intrinsic.c_u"],
            ],

            [
                0,
                row["[CameraCalibrationComponent].intrinsic.f_v"],
                row["[CameraCalibrationComponent].intrinsic.c_v"],
            ],

            [
                0,
                0,
                1,
            ],

        ],

        dtype=np.float32,

    )

###############################################################################
# Camera pose
###############################################################################

def build_pose(row):

    return np.array(

        row["[CameraImageComponent].pose.transform"],

        dtype=np.float32,

    ).reshape(4,4)

###############################################################################
# Beam inclinations
###############################################################################

def beam_inclinations(row, H):

    values = row["[LiDARCalibrationComponent].beam_inclination.values"]

    if values is not None and len(values) > 0:
        return np.asarray(values, dtype=np.float32)

    mn = row["[LiDARCalibrationComponent].beam_inclination.min"]
    mx = row["[LiDARCalibrationComponent].beam_inclination.max"]

    return np.linspace(mx, mn, H, dtype=np.float32)

###############################################################################
# Extrinsic
###############################################################################

def lidar_extrinsic(row):

    return np.asarray(

        row["[LiDARCalibrationComponent].extrinsic.transform"],

        dtype=np.float32,

    ).reshape(4,4)

def pixel_pose_tensor(lp):

    pose = lp.astype(np.float32)

    rot = tf.convert_to_tensor(pose[..., :3])

    trans = tf.convert_to_tensor(pose[..., 3:])

    R = get_rotation_matrix(
        rot[...,0],
        rot[...,1],
        rot[...,2],
    )

    return get_transform(R, trans)

def frame_pose_tensor(camera_pose):

    return tf.convert_to_tensor(
        camera_pose.astype(np.float32)
    )

def get_yaw_rotation(yaw, name=None):
  """Gets a rotation matrix given yaw only.

  Args:
    yaw: x-rotation in radians. This tensor can be any shape except an empty
      one.
    name: the op name.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape, 3 ,3].
  """
  with tf.compat.v1.name_scope(name, 'GetYawRotation', [yaw]):
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    return tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ],
                    axis=-2)


def get_yaw_rotation_2d(yaw):
  """Gets a rotation matrix given yaw only for 2d.

  Args:
    yaw: x-rotation in radians. This tensor can be any shape except an empty
      one.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape, 2, 2].
  """
  with tf.name_scope('GetYawRotation2D'):
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)

    return tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw], axis=-1),
        tf.stack([sin_yaw, cos_yaw], axis=-1),
    ],
                    axis=-2)


def get_rotation_matrix(roll, pitch, yaw, name=None):
  """Gets a rotation matrix given roll, pitch, yaw.

  roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
  x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

  https://en.wikipedia.org/wiki/Euler_angles
  http://planning.cs.uiuc.edu/node102.html

  Args:
    roll : x-rotation in radians.
    pitch: y-rotation in radians. The shape must be the same as roll.
    yaw: z-rotation in radians. The shape must be the same as roll.
    name: the op name.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape_of_yaw, 3 ,3].
  """
  with tf.compat.v1.name_scope(name, 'GetRotationMatrix', [yaw, pitch, roll]):
    cos_roll = tf.cos(roll)
    sin_roll = tf.sin(roll)
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    cos_pitch = tf.cos(pitch)
    sin_pitch = tf.sin(pitch)

    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    r_roll = tf.stack([
        tf.stack([ones, zeros, zeros], axis=-1),
        tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
        tf.stack([zeros, sin_roll, cos_roll], axis=-1),
    ],
                      axis=-2)
    r_pitch = tf.stack([
        tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        tf.stack([zeros, ones, zeros], axis=-1),
        tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
    ],
                       axis=-2)
    r_yaw = tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ],
                     axis=-2)

    return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
  """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

  Args:
    rotation: [..., N, N] rotation tensor.
    translation: [..., N] translation tensor. This must have the same type as
      rotation.

  Returns:
    transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
      rotation.
  """
  with tf.name_scope('GetTransform'):
    # [..., N, 1]
    translation_n_1 = translation[..., tf.newaxis]
    # [..., N, N+1]
    transform = tf.concat([rotation, translation_n_1], axis=-1)
    # [..., N]
    last_row = tf.zeros_like(translation)
    # [..., N+1]
    last_row = tf.concat([last_row, tf.ones_like(last_row[..., 0:1])], axis=-1)
    # [..., N+1, N+1]
    transform = tf.concat([transform, last_row[..., tf.newaxis, :]], axis=-2)
    return transform
  
def compute_inclination(inclination_range, height, scope=None):
  """Computes uniform inclination range based the given range and height.

  Args:
    inclination_range: [..., 2] tensor. Inner dims are [min inclination, max
      inclination].
    height: an integer indicates height of the range image.
    scope: the name scope.

  Returns:
    inclination: [..., height] tensor. Inclinations computed.
  """
  with tf.compat.v1.name_scope(scope, 'ComputeInclination',
                               [inclination_range]):
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (
        (.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) /
        tf.cast(height, dtype=inclination_range.dtype) *
        tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1])
    return inclination
  
def extract_point_cloud_from_range_image(range_image,
                                         extrinsic,
                                         inclination,
                                         pixel_pose=None,
                                         frame_pose=None,
                                         dtype=tf.float32,
                                         scope=None):
  """Extracts point cloud from range image.

  Args:
    range_image: [B, H, W] tensor. Lidar range images.
    extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
    inclination: [B, H] tensor. Inclination for each row of the range image.
      0-th entry corresponds to the 0-th row of the range image.
    pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
      image pixel.
    frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
      decides the vehicle frame at which the cartesian points are computed.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in vehicle
    frame.
  """
  with tf.compat.v1.name_scope(
      scope, 'ExtractPointCloudFromRangeImage',
      [range_image, extrinsic, inclination, pixel_pose, frame_pose]):
    range_image_polar = compute_range_image_polar(
        range_image, extrinsic, inclination, dtype=dtype)
    range_image_cartesian = compute_range_image_cartesian(
        range_image_polar,
        extrinsic,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        dtype=dtype)
    return range_image_cartesian
  
def compute_range_image_cartesian(range_image_polar,
                                  extrinsic,
                                  pixel_pose=None,
                                  frame_pose=None,
                                  dtype=tf.float32,
                                  scope=None):
  """Computes range image cartesian coordinates from polar ones.

  Args:
    range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
      coordinate in sensor frame.
    extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
    pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
      range image pixel.
    frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
      It decides the vehicle frame at which the cartesian points are computed.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_cartesian: [B, H, W, 3] cartesian coordinates.
  """
  range_image_polar_dtype = range_image_polar.dtype
  range_image_polar = tf.cast(range_image_polar, dtype=dtype)
  extrinsic = tf.cast(extrinsic, dtype=dtype)
  if pixel_pose is not None:
    pixel_pose = tf.cast(pixel_pose, dtype=dtype)
  if frame_pose is not None:
    frame_pose = tf.cast(frame_pose, dtype=dtype)

  with tf.compat.v1.name_scope(
      scope, 'ComputeRangeImageCartesian',
      [range_image_polar, extrinsic, pixel_pose, frame_pose]):
    azimuth, inclination, range_image_range = tf.unstack(
        range_image_polar, axis=-1)

    cos_azimuth = tf.cos(azimuth)
    sin_azimuth = tf.sin(azimuth)
    cos_incl = tf.cos(inclination)
    sin_incl = tf.sin(inclination)

    # [B, H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [B, H, W, 3]
    range_image_points = tf.stack([x, y, z], -1)
    # [B, 3, 3]
    rotation = extrinsic[..., 0:3, 0:3]
    # translation [B, 1, 3]
    translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

    # To vehicle frame.
    # [B, H, W, 3]
    range_image_points = tf.einsum('bkr,bijr->bijk', rotation,
                                   range_image_points) + translation
    if pixel_pose is not None:
      # To global frame.
      # [B, H, W, 3, 3]
      pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
      # [B, H, W, 3]
      pixel_pose_translation = pixel_pose[..., 0:3, 3]
      # [B, H, W, 3]
      range_image_points = tf.einsum(
          'bhwij,bhwj->bhwi', pixel_pose_rotation,
          range_image_points) + pixel_pose_translation
      if frame_pose is None:
        raise ValueError('frame_pose must be set when pixel_pose is set.')
      # To vehicle frame corresponding to the given frame_pose
      # [B, 4, 4]
      world_to_vehicle = tf.linalg.inv(frame_pose)
      world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
      world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
      # [B, H, W, 3]
      range_image_points = tf.einsum(
          'bij,bhwj->bhwi', world_to_vehicle_rotation,
          range_image_points) + world_to_vehicle_translation[:, tf.newaxis,
                                                             tf.newaxis, :]

    range_image_points = tf.cast(
        range_image_points, dtype=range_image_polar_dtype)
    return range_image_points


def compute_range_image_polar(range_image,
                              extrinsic,
                              inclination,
                              dtype=tf.float32,
                              scope=None):
  """Computes range image polar coordinates.

  Args:
    range_image: [B, H, W] tensor. Lidar range images.
    extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
    inclination: [B, H] tensor. Inclination for each row of the range image.
      0-th entry corresponds to the 0-th row of the range image.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_polar: [B, H, W, 3] polar coordinates.
  """
  # pylint: disable=unbalanced-tuple-unpacking
  _, height, width = _combined_static_and_dynamic_shape(range_image)
  range_image_dtype = range_image.dtype
  range_image = tf.cast(range_image, dtype=dtype)
  extrinsic = tf.cast(extrinsic, dtype=dtype)
  inclination = tf.cast(inclination, dtype=dtype)

  with tf.compat.v1.name_scope(scope, 'ComputeRangeImagePolar',
                               [range_image, extrinsic, inclination]):
    with tf.compat.v1.name_scope('Azimuth'):
      # [B].
      az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
      # [W].
      ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - .5) / tf.cast(
          width, dtype=dtype)
      # [B, W].
      azimuth = (ratios * 2. - 1.) * np.pi - tf.expand_dims(az_correction, -1)

    # [B, H, W]
    azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
    # [B, H, W]
    inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
    range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image],
                                 axis=-1)
    return tf.cast(range_image_polar, dtype=range_image_dtype)

def _combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(input=tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

###############################################################################
# Build depth map
###############################################################################

def build_depth(points, projection, camera_name, H, W):

    depth = np.full((H, W), np.inf, dtype=np.float32)

    xyz = points.reshape(-1,3)
    cp = projection.reshape(-1, 6)

    for p, c in zip(xyz, cp):

        if c[0] == camera_name:

            x = int(c[1])
            y = int(c[2])

        elif c[3] == camera_name:

            x = int(c[4])
            y = int(c[5])

        else:
            continue

        if x < 0 or x >= W:
            continue

        if y < 0 or y >= H:
            continue

        d = np.linalg.norm(p)

        if d <= 0:
            continue

        if d < depth[y, x]:
            depth[y, x] = d

    depth[np.isinf(depth)] = 0

    return depth

###############################################################################
# Main
###############################################################################

def main():

    args = parse_args()

    image_file = next(
        (args.input / "camera_image").glob("*.parquet")
    )

    calib_file = next(
        (args.input / "camera_calibration").glob("*.parquet")
    )

    print("Reading camera image...")

    image_df = load_table(image_file)

    print("Reading calibration...")

    calib_df = load_table(calib_file)

    print(image_df.shape)

    print(calib_df.shape)

    print("Reading LiDAR...")

    lidar_df, lidar_calib_df, lidar_pose_df, projection_df = load_lidar(args.input)

    print(lidar_df.shape)
    print(lidar_calib_df.shape)
    print(lidar_pose_df.shape)
    print(projection_df.shape)

    for i in range(len(image_df)):

        image_row = image_df.iloc[i]
        segment = image_row["key.segment_context_name"]
        timestamp = image_row["key.frame_timestamp_micros"]

        lidar_row = lidar_df[
            (lidar_df["key.segment_context_name"] == segment) &
            (lidar_df["key.frame_timestamp_micros"] == timestamp)
        ].iloc[0]

        pose_row = lidar_pose_df[
            (lidar_pose_df["key.segment_context_name"] == segment) &
            (lidar_pose_df["key.frame_timestamp_micros"] == timestamp)
        ].iloc[0]

        proj_row = projection_df[
            (projection_df["key.segment_context_name"] == segment) &
            (projection_df["key.frame_timestamp_micros"] == timestamp)
        ].iloc[0]

        calib_row = calib_df[
            calib_df["key.camera_name"] == image_row["key.camera_name"]
        ].iloc[0]

        image = decode_image(
            image_row["[CameraImageComponent].image"]
        )

        K = build_intrinsics(calib_row)

        pose = build_pose(image_row)

        ri = range_image(lidar_row)
        lp = lidar_pose(pose_row)
        cp = camera_projection(proj_row)

        print()

        print("Range image:", ri.shape)
        print("Pose image:", lp.shape)
        print("Projection image:", cp.shape)
        
        print()

        print("Range min/max:", ri[...,0].min(), ri[...,0].max())

        calib_row = lidar_calib_df.iloc[0]

        extrinsic = lidar_extrinsic(calib_row)

        inclinations = beam_inclinations(
            calib_row,
            ri.shape[0],
        )

        range_tf = tf.convert_to_tensor(
        ri[...,0][None,...],
        dtype=tf.float32,
        )

        extrinsic_tf = tf.convert_to_tensor(
            extrinsic[None,...],
            dtype=tf.float32,
        )

        inclination_tf = tf.convert_to_tensor(
            inclinations[None,...],
            dtype=tf.float32,
        )

        pixel_pose_tf = pixel_pose_tensor(lp)[None,...]

        frame_pose_tf = frame_pose_tensor(pose)[None,...]

        points = extract_point_cloud_from_range_image(
            range_tf,
            extrinsic_tf,
            inclination_tf,
            pixel_pose=pixel_pose_tf,
            frame_pose=frame_pose_tf,
        )

        points = points.numpy()[0]

        camera_name = image_row["key.camera_name"]

        depth = build_depth(
            points,
            cp,
            camera_name,
            image.shape[0],
            image.shape[1],
        )

        print("Camera Z:", depth[depth > 0].min(), depth.max())

        print(depth.shape)

        print(depth.min(), depth.max())

        valid = depth > 0

        print("Valid pixels:", valid.sum())
        print("Coverage:", valid.mean())

        vis = depth.copy()

        vis[valid] = np.log1p(vis[valid])

        vis = vis / vis.max()

        cv2.imwrite(
            "debug_depth_vis.png",
            (vis * 255).astype(np.uint8),
        )

        print(points.shape)

        print()

        print(extrinsic)

        print()

        print(inclinations.shape)

        sample_dir = args.output / f"sample{i:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(sample_dir / "image.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

        depth16 = (depth * 1000.0).astype(np.uint16)

        cv2.imwrite(
            str(sample_dir / "depth.png"),
            depth16,
        )

        meta = {
            "intrinsics": K.tolist(),
            "pose": pose.tolist(),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "source": "waymo",
        }

        write_json(
            sample_dir / "meta.json",
            meta,
        )

        with open(args.output / ".index.txt", "a") as f:
            f.write("sample000000\n")

        print("Done:", sample_dir)


if __name__ == "__main__":
    main()