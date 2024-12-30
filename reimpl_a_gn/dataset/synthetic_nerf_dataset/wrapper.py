"""This module wraps the original dataset loader to provide JAX arrays in a format that's easy for us to use."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy

from reimpl_a_gn.threed.camera import CameraParams

from ._original_code import load_llff_data


@dataclass
class SyntheticNeRFData:
    images: jax.Array
    """All images. Shape: (image_count, height, width, RGB); normalized to [0, 1]."""
    poses: jax.Array
    """Camera poses for all images. Shape: (image_count, 3, 5).
    3x4 camera-to-world matrix, then (height, width, focal_length) in the fifth column.
    """
    bds: jax.Array
    """Near and far bounds for each image. Shape: (image_count, 2). Last axis: near depth, then far depth."""
    render_poses: jax.Array
    """Camera poses for novel views. Shape: (novel_view_count, 3, 5).
    3x4 camera-to-world matrix, then (height, width, focal_length) in the fifth column.
    """
    i_test: int
    """Index of the test image. Not supposed to be used in training."""
    cameras: list[CameraParams]
    """Camera parameters from `poses`."""


def _get_camera_parameters(poses, imgs):
    """Extract the extrinsic and intrinsic parameters of all cameras from the Synthetic NeRF dataset.

    Ignore camera distortion. Assue a pinhole model.

    Thanks Claude Sonnet 3.5.
    """
    # Extrinsic matrix (camera-to-world transform)
    c2w = poses[:, :3, :4]  # Shape: (N, 3, 4) where N is the number of images
    new_homogeneous_rows = jnp.repeat(
        jnp.array([[[0, 0, 0, 1]]]), repeats=c2w.shape[0], axis=0
    )
    c2w = jnp.concatenate(
        [c2w, new_homogeneous_rows],
        axis=1,
    )

    # Intrinsic matrix
    H, W = imgs[0].shape[:2]  # Image height and width
    focal = poses[
        0, 2, 4
    ]  # Focal length is stored in poses[:, 2, 4], we assume that it's the same for every camera

    intrinsic_matrix = jnp.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])

    return jnp.array(c2w), intrinsic_matrix


def load_synthetic_nerf_dataset(
    basedir: Path | str,
    factor=8,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    path_zflat=False,
) -> SyntheticNeRFData:
    """Load data from the Synthetic NeRF Dataset as JAX arrays.

    Format is the same as the original loader, besides using JAX arrays instead of NumPy arrays.

    Original loader: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L243
    """
    images, poses, bds, render_poses, i_test = load_llff_data(
        str(basedir),
        factor=factor,
        recenter=recenter,
        bd_factor=bd_factor,
        spherify=spherify,
        path_zflat=path_zflat,
    )
    camera_to_world_matrices, intrinsic_matrix = _get_camera_parameters(poses, images)
    camera_params: list[CameraParams] = []
    for camera_to_world in camera_to_world_matrices:
        extrinsic_matrix = jnp.array(numpy.linalg.inv(camera_to_world[:, :4]))
        camera_params.append(CameraParams(extrinsic_matrix, intrinsic_matrix))
    return SyntheticNeRFData(
        jnp.array(images),
        jnp.array(poses),
        jnp.array(bds),
        jnp.array(render_poses),
        int(i_test),
        camera_params,
    )
