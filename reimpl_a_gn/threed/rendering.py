from functools import partial

import jax
import jax.numpy as jnp
import jax.typing as jt
from jax.lax import scan
from jax.random import split

from reimpl_a_gn.random import piecewise_uniform
from reimpl_a_gn.threed.coord_utils import (
    blend_ray_features_with_nerf_paper_method,
    compute_fine_sampling_distribution,
    compute_nerf_positional_encoding,
    get_rays,
    sample_coarse_mlp_inputs,
)
from reimpl_a_gn.threed.nerf.nerf import CoarseMLP, FineMLP


@partial(
    jax.jit,
    static_argnames=["coarse_network", "fine_network", "ray_batch_size"],
)
def render_image(
    image: jt.ArrayLike,
    intrinsic_matrix: jt.ArrayLike,
    rng_key: jax.Array,
    coarse_network: CoarseMLP,
    fine_network: FineMLP,
    ray_batch_size: int,
):
    """Predict and render colors for all pixels in an image, batch by batch.

    @param image RGB image with shape (height, width, 3).
    @param intrinsic_matrix Intrinsic matrix from camera frame to image coordinates. Shape: (3, 3).
    @param rng_key JAX random number generator key for stochastic sampling.
    @param coarse_network Coarse MLP network for initial density prediction.
    @param fine_network Fine MLP network for refined rendering.
    @param ray_batch_size Divisor of (image.shape[0] * image.shape[1]). We will process batches of this number of rays.
    @return Rendered image with same shape as input image.
    """
    image = jnp.array(image)
    image_height, image_width, _ = image.shape

    _, rays = get_rays(image_height, image_width, intrinsic_matrix)
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
