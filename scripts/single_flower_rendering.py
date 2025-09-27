"""Loads one image and pose from the flower datasets, renders one color per pixel.

Will only output noise as of now since I don't train anything.
"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import kagglehub
import matplotlib.pyplot as plt
from flax.nnx import Rngs
from jax.random import key, split

from reimpl_a_gn.dataset.synthetic_nerf_dataset import load_synthetic_nerf_dataset
from reimpl_a_gn.random import piecewise_uniform
from reimpl_a_gn.threed.nerf import (
    CoarseMLP,
    FineMLP,
    blend_ray_features_with_nerf_paper_method,
    compute_fine_sampling_distribution,
    compute_nerf_positional_encoding,
    sample_coarse_mlp_inputs,
)
from reimpl_a_gn.threed.rendering import CameraParams, sample_rays_towards_pixels


def get_flower_dataset():
    """Download the dataset if needed, and return the path to the flower subfolder."""
    dataset_path = (
        Path(kagglehub.dataset_download("arenagrenade/llff-dataset-full"))
        / "nerf_llff_data"
        / "flower"
    )
    return load_synthetic_nerf_dataset(dataset_path)


def get_rays(image_height: int, image_width: int, camera: CameraParams):
    pixel_xs, pixel_ys = jnp.meshgrid(
        jnp.arange(0, image_height), jnp.arange(0, image_width)
    )
    pixel_coords = jnp.stack([pixel_xs.flatten(), pixel_ys.flatten()], axis=1)
    assert pixel_coords.ndim == 2 and pixel_coords.shape[1] == 2

    pixel_rays = sample_rays_towards_pixels(camera, pixel_coords)
    return pixel_coords, pixel_rays


@partial(jax.jit, static_argnames=("camera", "coarse_network", "fine_network"))
def render_single_image(
    image: jnp.ndarray,
    camera: CameraParams,
    rng_key: jnp.ndarray,
    coarse_network: CoarseMLP,
    fine_network: FineMLP,
    near_distance: float = 0.01,
    far_distance: float = 5,
):
    image_height, image_width, _ = image.shape

    _, rays = get_rays(image_height, image_width, camera)

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
        interval_probabilities=fine_position_distribution,
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
            jnp.repeat(
                jnp.expand_dims(ray_unit_direction_vectors, -2), axis=-2, repeats=5
            ),
        ],
        axis=-1,
    )

    encoded_fine_positions = compute_nerf_positional_encoding(fine_positions, 2)

    fine_predictions = fine_network(encoded_fine_positions)

    blending_inputs = jnp.concat([fine_positions, fine_predictions], axis=-1)
    blended_colors_per_ray = blend_ray_features_with_nerf_paper_method(
        ray_features=blending_inputs
    )
    blended_colors_per_ray = blended_colors_per_ray.reshape(
        (image_height, image_width, 3)
    )
    return blended_colors_per_ray


rng_key = key(seed=7)

flower_dataset = get_flower_dataset()

first_image = flower_dataset.images[0]
assert first_image.ndim == 3  # height, width, RGB
first_camera = flower_dataset.cameras[0]

rng_key, rng_subkey = split(rng_key)
coarse_network = CoarseMLP(
    input_features=6 * 2 * 2,
    mid_features=(64, 64),
    out_features=4,
    rngs=Rngs(rng_subkey),
)
del rng_subkey

rng_key, rng_subkey = split(rng_key)
fine_network = FineMLP(6 * 2 * 2, (64, 64), 4, rngs=Rngs(rng_subkey))
del rng_subkey

colors = render_single_image(
    image=first_image,
    camera=first_camera,
    rng_key=rng_key,
    coarse_network=coarse_network,
    fine_network=fine_network,
)
plt.imshow(colors)
plt.show()
