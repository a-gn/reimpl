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

        # choose pixel coordinates
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

        chosen_image_indices = randint(
            image_choice_key, shape=(self.batch_size,), minval=0, maxval=image_count
        )
        del image_choice_key
