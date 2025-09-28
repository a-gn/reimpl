"""Loads one image and pose from the flower datasets, renders one color per pixel.

Will only output noise as of now since I don't train anything.
"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.typing as jt
import kagglehub
import matplotlib.pyplot as plt
from flax.nnx import Rngs
from jax.lax import scan
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


@partial(
    jax.jit,
    static_argnames=["camera", "coarse_network", "fine_network", "ray_batch_size"],
)
def render_image(
    image: jt.ArrayLike,
    camera: CameraParams,
    rng_key: jax.Array,
    coarse_network: CoarseMLP,
    fine_network: FineMLP,
    ray_batch_size: int,
):
    """Predict and render colors for all pixels in an image, batch by batch.

    @param image RGB image with shape (height, width, 3).
    @param ray_batch_size Divisor of (image.shape[0] * image.shape[1]). We will process batches of this number of rays.
    """
    image = jnp.array(image)
    image_height, image_width, _ = image.shape

    _, rays = get_rays(image_height, image_width, camera)
    ray_count = rays.shape[0]

    # batch all rays except a possible incomplete last batch
    if ray_batch_size > rays.shape[0]:
        print(
            f"batch size {ray_batch_size} is larger than total ray count {rays.shape[0]}, we won't batch or pad"
        )
        batches = jnp.expand_dims(rays, 0)
        remaining_rays = 0
    elif ray_count % ray_batch_size == 0:
        # clean split
        batches = jnp.reshape(rays, (-1, ray_batch_size, 6))
        remaining_rays = 0
    else:
        remaining_rays = ray_count % ray_batch_size
        # create the first, complete batches
        batches = jnp.reshape(rays[:-remaining_rays], (-1, ray_batch_size, 6))
        # pad the last batch and append it
        last_batch = jnp.concat(
            [
                rays[-remaining_rays:],
                jnp.zeros((ray_batch_size - remaining_rays, 6)),
            ],
            axis=0,
        )
        last_batch = jnp.expand_dims(last_batch, 0)  # batch axis
        batches = jnp.concat([batches, last_batch], axis=0)

    split_rng_keys = jax.random.split(rng_key, batches.shape[0])

    def render_single_batch(batch_index: int, ray_batch: jnp.ndarray):
        return batch_index + 1, render_rays(
            ray_batch,
            rng_key=split_rng_keys[batch_index],
            coarse_network=coarse_network,
            fine_network=fine_network,
        )

    print(f"scanning through {batches.shape[0]} batches...")
    _, ray_batch_renders = scan(render_single_batch, 0, batches)
    # collapse batch axis
    ray_batch_renders = ray_batch_renders.reshape(-1, 3)
    # remove padding from last batch
    if remaining_rays != 0:
        ray_batch_renders = ray_batch_renders[: -ray_batch_size + remaining_rays]
    # back to image shape
    ray_batch_renders = ray_batch_renders.reshape(image.shape)
    return ray_batch_renders


def render_rays(
    rays: jt.ArrayLike,
    rng_key: jnp.ndarray,
    coarse_network: CoarseMLP,
    fine_network: FineMLP,
    near_distance: float = 0.01,
    far_distance: float = 5,
):
    rays = jnp.array(rays)

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

colors = render_image(
    image=first_image,
    camera=first_camera,
    rng_key=rng_key,
    coarse_network=coarse_network,
    fine_network=fine_network,
    ray_batch_size=2**16,
)
plt.imshow(colors)
plt.show()
