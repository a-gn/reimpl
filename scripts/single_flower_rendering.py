"""Loads one image and pose from the flower datasets, renders one color per pixel.

Will only output noise as of now since I didn't train anything.
"""

from pathlib import Path

import jax.numpy as jnp
import kagglehub
from flax.nnx import Rngs
from jax.random import key, split

from reimpl_a_gn.dataset.synthetic_nerf_dataset import load_synthetic_nerf_dataset
from reimpl_a_gn.random import piecewise_uniform
from reimpl_a_gn.threed.nerf import (
    CoarseMLP,
    FineMLP,
    compute_fine_sampling_distribution,
    compute_nerf_positional_encoding,
    sample_coarse_mlp_inputs,
)
from reimpl_a_gn.threed.rendering import sample_rays_towards_pixels
from reimpl_a_gn.threed.visualization.export_array import array_to_csv

rng_key = key(seed=7)


def get_flower_dataset():
    """Download the dataset if needed, and return the path to the flower subfolder."""
    dataset_path = (
        Path(kagglehub.dataset_download("arenagrenade/llff-dataset-full"))
        / "nerf_llff_data"
        / "flower"
    )
    return load_synthetic_nerf_dataset(dataset_path)


flower_dataset = get_flower_dataset()

first_image = flower_dataset.images[0]
assert first_image.ndim == 3  # height, width, RGB
first_camera = flower_dataset.cameras[0]

near_distance = 0.01
far_distance = 5


def get_rays():
    image_height, image_width, _ = first_image.shape
    pixel_xs, pixel_ys = jnp.meshgrid(
        jnp.arange(0, image_height), jnp.arange(0, image_width)
    )
    pixel_coords = jnp.stack([pixel_xs.flatten(), pixel_ys.flatten()], axis=1)
    assert pixel_coords.ndim == 2 and pixel_coords.shape[1] == 2

    pixel_rays = sample_rays_towards_pixels(first_camera, pixel_coords)
    return pixel_rays


rays = get_rays()

# coarse MLP

rng_key, rng_subkey = split(rng_key)
coarse_positions = sample_coarse_mlp_inputs(
    rays,
    near_distance=near_distance,
    far_distance=far_distance,
    bins_per_ray=5,
    prng_key=rng_subkey,
)
del rng_subkey

encoded_coarse_positions = compute_nerf_positional_encoding(coarse_positions, 2)

rng_key, rng_subkey = split(rng_key)
coarse_network = CoarseMLP(
    input_features=6 * 2 * 2,
    mid_features=(64, 64),
    out_features=4,
    rngs=Rngs(rng_subkey),
)
del rng_subkey

coarse_logits = coarse_network(encoded_coarse_positions)

# fine MLP

coarse_densities = coarse_logits[..., 3]
coarse_positions_on_rays = jnp.linalg.norm(coarse_positions[..., :3], axis=-1)
fine_position_distribution = compute_fine_sampling_distribution(
    densities=coarse_densities, sampling_positions=coarse_positions_on_rays
)

rng_key, rng_subkey = split(rng_key)
fine_positions_on_rays = piecewise_uniform(
    key=rng_subkey,
    intervals=coarse_positions_on_rays,
    pdf_values=fine_position_distribution,
    sample_count_per_distribution=5,
)
del rng_subkey

ray_unit_direction_vectors = rays[..., 3:6] / jnp.linalg.norm(
    rays[..., 3:6], axis=-1, keepdims=True
)
fine_positions = jnp.expand_dims(rays[..., :3], -2) + jnp.expand_dims(
    ray_unit_direction_vectors, -2
) * jnp.expand_dims(fine_positions_on_rays, -1)
# add ray directions
fine_positions = jnp.concat(
    [
        fine_positions,
        jnp.repeat(jnp.expand_dims(ray_unit_direction_vectors, -2), axis=-2, repeats=5),
    ],
    axis=-1,
)

encoded_fine_positions = compute_nerf_positional_encoding(fine_positions, 2)

rng_key, rng_subkey = split(rng_key)
fine_network = FineMLP(6 * 2 * 2, (64, 64), 4, rngs=Rngs(rng_subkey))
del rng_subkey

fine_predictions = fine_network(encoded_fine_positions)
