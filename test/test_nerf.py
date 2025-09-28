import flax.nnx as nnx
import jax
import pytest

import reimpl_a_gn.threed.nerf as nerf


@pytest.fixture()
def rngs_7():
    return nnx.Rngs(7)


def test_pass_data_through_coarse_mlp(rngs_7: nnx.Rngs):
    mlp = nerf.CoarseMLP(6, (64, 64, 64), 9, rngs=rngs_7)
    prng_key = jax.random.key(7)
    data = jax.random.uniform(
        prng_key, (4, 6), float, -10000, 10000
    )  # 6D instead of 32D
    result = mlp(data)
    assert result.shape == (4, 9)


def test_pass_data_through_fine_mlp(rngs_7: nnx.Rngs):
    mlp = nerf.FineMLP(6, (64, 64, 64), 9, rngs=rngs_7)
    prng_key = jax.random.key(7)
    data = jax.random.uniform(
        prng_key, (4, 6), float, -10000, 10000
    )  # 6D instead of 32D
    result = mlp(data)
    assert result.shape == (4, 9)
