from flax.nnx import Rngs
from jax.random import key, split

from reimpl_a_gn.dataset.synthetic_nerf_dataset import (
    get_flower_dataset,
    load_synthetic_nerf_dataset,
)
from reimpl_a_gn.threed.nerf import CoarseMLP, FineMLP
from reimpl_a_gn.threed.nerf.training import train_nerf
from reimpl_a_gn.threed.rendering import compute_nerf_positional_encoding


def main():
    coarse_init_key, fine_init_key, train_key = split(key(7), 3)
    coarse_net = CoarseMLP(3, (32, 32), 32, Rngs(coarse_init_key))
    fine_net = FineMLP(3, (32, 32), 32, Rngs(fine_init_key))
    dataset = get_flower_dataset()
    train_nerf(
        coarse_mlp=coarse_net,
        fine_mlp=fine_net,
        dataset=dataset,
        epoch_count=3,
        batch_size=4,
        batches_per_epoch=1000,
        position_encoder=compute_nerf_positional_encoding,
        direction_encoder=compute_nerf_positional_encoding,
        rng_key=train_key,
    )


if __name__ == "__main__":
    main()
