import reimpl_a_gn.threed.nerf as nerf

import jax
import jax.numpy as jnp


def test_pass_data_through_coarse_mlp():
    mlp = nerf.CoarseMLP((64, 64, 64), 6)
    prng_key = jax.random.key(7)
    data = jax.random.uniform(prng_key, (4, 32), float, -10000, 10000)
    params = mlp.init(prng_key, data)
    mlp.apply(params, data)


def test_pass_data_through_fine_mlp():
    mlp = nerf.FineMLP((64, 64, 64), 6)
    prng_key = jax.random.key(7)
    data = jax.random.uniform(prng_key, (4, 32), float, -10000, 10000)
    params = mlp.init(prng_key, data)
    mlp.apply(params, data)


def test_compute_fine_sampling_distribution():
    densities = jnp.array([0.5, 0.7, 4.5, 0.0, 3.0])
    sampling_positions = jnp.array([0.0, 0.3, 1.5, 100.3, 102.0])
    expected_distribution = jnp.zeros(len(densities) - 1, dtype=float)  # TODO

    computed_distribution = nerf.compute_fine_sampling_distribution(
        densities, sampling_positions
    )
    assert jnp.allclose(computed_distribution, expected_distribution)
