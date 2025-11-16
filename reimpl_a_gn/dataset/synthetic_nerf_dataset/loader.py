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
        # add homogeneous weight with value 1 (at this point, during projection, Z has been normalized away)
        chosen_pixel_xyw = jnp.concat(
            [
                chosen_pixel_xy,
                jnp.ones((chosen_pixel_xy.shape[0], 1), dtype=chosen_pixel_xy.dtype),
            ],
            axis=1,
        )
        ray_directions_camera_frame_xyw = chosen_pixel_xyw @ pixel_to_camera_transform.T
        # add Z = 1
        ray_directions_camera_frame_xyzw = jnp.concat(
            [
                ray_directions_camera_frame_xyw,
                jnp.ones_like(
                    ray_directions_camera_frame_xyw,
                    shape=(*ray_directions_camera_frame_xyw.shape[:-1], 1),
                ),
            ],
            axis=-1,
        )

        # to world frame
        chosen_extrinsic_matrices = jnp.take(
            self.all_data.extrinsic_matrices, chosen_image_indices, axis=0
        )
        camera_to_world_transforms: Array = jnp.linalg.inv(chosen_extrinsic_matrices)
        assert isinstance(camera_to_world_transforms, Array)
        ray_directions_world_frame = jnp.einsum(
            "ij,ikj->ik", ray_directions_camera_frame_xyzw, camera_to_world_transforms
        )
        ray_origins_camera_frame = jnp.zeros((self.batch_size, 4), dtype=float)
        ray_origins_world_frame = jnp.einsum(
            "ij,ikj->ik", ray_origins_camera_frame, camera_to_world_transforms
        )
        # should be vectors
        assert jnp.all(ray_origins_world_frame[..., 3] == 0.0)
        rays_world_frame = jnp.concat(
            [
                ray_origins_world_frame[..., :3] / ray_origins_world_frame[..., 3:],
                ray_directions_world_frame[..., :3],  # should be vectors
            ],
            axis=-1,
        )

        # read colors from images
        integer_pixel_coordinates = chosen_pixel_xy.astype(int)
        chosen_images = jnp.take(self.all_data.images, chosen_image_indices, axis=0)
        pixel_color_values = chosen_images[
            chosen_image_indices,
            integer_pixel_coordinates[:, 0],
            integer_pixel_coordinates[:, 1],
        ]

        return NeRFTrainingSamples(
            rays=rays_world_frame,
            colors=pixel_color_values,
            extrinsic_matrices=chosen_extrinsic_matrices,
            dataset_info=[
                {"image_index": index} for index in chosen_image_indices.tolist()
            ],
        )
