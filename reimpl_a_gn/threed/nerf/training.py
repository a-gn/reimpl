from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.random import split
from optax import adam
from tqdm import tqdm

from reimpl_a_gn.dataset.common import RayAndColorDataset
from reimpl_a_gn.threed.coord_utils import (
    compute_fine_sampling_distribution,
    sample_coarse_mlp_positions,
    sample_from_fine_sampling_distribution,
)
from reimpl_a_gn.threed.rendering import blend_ray_features_with_nerf_paper_method

from .nerf import CoarseMLP, FineMLP


def mse_loss(logits: Array, target: Array):
    return jnp.sum((logits - target) ** 2)


def train_nerf(
    coarse_mlp: CoarseMLP,
    fine_mlp: FineMLP,
    dataset: RayAndColorDataset,
    epoch_count: int,
    batch_size: int,
    batches_per_epoch: int,
    position_encoder: Callable[[Array], Array],
    direction_encoder: Callable[[Array], Array],
    rng_key: Array,
):
    # we need two RNG keys per batch: coarse and fine position sampling
    epoch_keys = split(rng_key, (epoch_count, 2))

    dataset_iterator = iter(dataset)

    for epoch_id in tqdm(range(epoch_count), desc="epochs"):
        for _ in tqdm(range(batches_per_epoch), desc="batches"):
            coarse_input_sampling_key, fine_input_sampling_key = epoch_keys[epoch_id]
            batch_items = [next(dataset_iterator) for _ in range(batch_size)]
            joined_rays = jnp.concatenate([item.rays for item in batch_items], axis=0)

            coarse_positions = sample_coarse_mlp_positions(
                joined_rays,
                near_distance=0.1,
                far_distance=5.0,
                bins_per_ray=5,
                prng_key=coarse_input_sampling_key,
            )
            # Encode coarse positions for MLP
            coarse_inputs = jnp.concat(
                [
                    position_encoder(coarse_positions[..., :3]),
                    direction_encoder(coarse_positions[..., 3:]),
                ],
                axis=-1,
            )
            coarse_predictions = coarse_mlp(coarse_inputs)

            fine_input_distribution = compute_fine_sampling_distribution(
                densities=coarse_predictions[..., 3],
                sampling_positions=coarse_positions[..., :3],
            )
            fine_positions = sample_from_fine_sampling_distribution(
                pdf=fine_input_distribution,
                rays=joined_rays,
                positions=coarse_positions,
                sample_count_per_distribution=3,
                rng_key=fine_input_sampling_key,
            )
            fine_inputs = jnp.concat(
                [
                    position_encoder(fine_positions[..., :3]),
                    direction_encoder(fine_positions[..., 3:]),
                ],
                axis=-1,
            )
            fine_predictions = fine_mlp(fine_inputs)

            blended_colors = blend_ray_features_with_nerf_paper_method()
