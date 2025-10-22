"""This module implements classes that load training batches from a Synthetic NeRF Dataset instance."""

import jax.numpy as jnp
from jax import Array
from jax.random import randint, split, uniform

from ..common import NeRFTrainingSamples, RayAndColorDataset
from .wrapper import SyntheticNeRFData


class SyntheticNeRFDatasetForTraining(RayAndColorDataset):
    def __init__(self, all_data: SyntheticNeRFData):
        self.all_data = all_data

    def _get_batch_of_rays(self, rng_key: Array) -> NeRFTrainingSamples:
        coordinate_choice_key, image_choice_key = split(rng_key)

        image_count, image_height, image_width, _ = self.all_data.images.shape

        # choose images (with replacement)
        chosen_image_indices = randint(
            image_choice_key, shape=(self.batch_size,), minval=0, maxval=image_count
        )
        del image_choice_key

        # choose pixel coordinates for every image (same sizes)
        chosen_pixel_xy = uniform(
            coordinate_choice_key,
            shape=(self.batch_size, 2),
            minval=jnp.zeros((self.batch_size, 2), dtype=float),
            maxval=jnp.repeat(
                jnp.array([[image_width, image_height]], dtype=float),
                axis=0,
                repeats=self.batch_size,
            ),
        )
        del coordinate_choice_key

        # compute ray direction vectors in camera frame
        pixel_to_camera_transform = jnp.linalg.inv(self.all_data.intrinsic_matrix)
        assert isinstance(
            pixel_to_camera_transform, jnp.ndarray
        ) and pixel_to_camera_transform.shape == (3, 3)
        chosen_pixel_xy_hom = jnp.concat(
            [
                chosen_pixel_xy,
                jnp.ones((chosen_pixel_xy.shape[0], 1), dtype=chosen_pixel_xy.dtype),
            ],
            axis=1,
        )
        ray_directions_camera_frame = chosen_pixel_xy_hom @ pixel_to_camera_transform.T
        # homogeneous weight should be zero for vectors
        assert jnp.all(ray_directions_camera_frame[..., 3] == 0.0)

        # to world frame
        chosen_extrinsic_matrices = jnp.take_along_axis(
            self.all_data.extrinsic_matrices, chosen_image_indices, axis=0
        )
        camera_to_world_transforms = jnp.linalg.inv(chosen_extrinsic_matrices)
        ray_directions_world_frame = (
            ray_directions_camera_frame @ camera_to_world_transforms.T
        )
