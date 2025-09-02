import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import reimpl_a_gn.threed.nerf as nerf


@pytest.fixture()
def rngs_7():
    return nnx.Rngs(7)


def test_pass_data_through_coarse_mlp(rngs_7: nnx.Rngs):
    mlp = nerf.CoarseMLP(6, (64, 64, 64), 9, rngs=rngs_7)
    prng_key = jax.random.key(7)
    data = jax.random.uniform(prng_key, (4, 6), float, -10000, 10000)  # 6D instead of 32D
    result = mlp(data)
    assert result.shape == (4, 9)


def test_pass_data_through_fine_mlp(rngs_7: nnx.Rngs):
    mlp = nerf.FineMLP(6, (64, 64, 64), 9, rngs=rngs_7)
    prng_key = jax.random.key(7)
    data = jax.random.uniform(prng_key, (4, 6), float, -10000, 10000)  # 6D instead of 32D
    result = mlp(data)
    assert result.shape == (4, 9)


def test_compute_fine_sampling_distribution():
    """Make sure that the fine sampling distribution doesn't regress to incorrectness."""
    densities = jnp.array([[0.5, 0.7, 4.5, 0.0, 162.8]])
    sampling_positions = jnp.array([[0.0, 0.3, 1.5, 100.3, 102.0]])
    expected_distribution = jnp.zeros(
        (sampling_positions.shape[0], sampling_positions.shape[1] - 1), dtype=float
    )
    cumulative_transmittance = jnp.cumulative_sum(
        -densities[:, :-1] * (sampling_positions[:, 1:] - sampling_positions[:, :-1]),
        axis=1,
        include_initial=True,
    )
    for interval_index in range(0, sampling_positions.shape[1] - 1):
        expected_distribution = expected_distribution.at[:, interval_index].set(
            jnp.exp(cumulative_transmittance[:, interval_index])
            * (
                1
                - jnp.exp(
                    -densities[:, interval_index]
                    * (
                        sampling_positions[:, interval_index + 1]
                        - sampling_positions[:, interval_index]
                    )
                )
            )
        )

    computed_distribution = nerf.compute_fine_sampling_distribution(
        densities, sampling_positions
    )
    print(f"computed:\n{computed_distribution}\nexpected:\n{expected_distribution}")
    assert jnp.allclose(computed_distribution, expected_distribution)


@pytest.mark.parametrize("ray_features,expected_result,description,atol", [
    # Test case 1: Simple case with 2 points to make calculation clearer
    (
        jnp.array([
            [
                [
                    # Point 1: position (0,0,0), color (1,0,0), density 1.0
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    # Point 2: position (1,0,0), color (0,1,0), density 2.0
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0],
                ]
            ]
        ]),
        lambda: jnp.array([[[
            1.0 * (1 - jnp.exp(-1.0)) * 1.0,  # Red: only first point contributes
            1.0 * (1 - jnp.exp(-1.0)) * 0.0,  # Green: only first point contributes  
            1.0 * (1 - jnp.exp(-1.0)) * 0.0,  # Blue: only first point contributes
        ]]]),
        "two_points_with_density",
        1e-5,
    ),
    # Test case 2: Edge case - zero density should pass through completely
    (
        jnp.array([
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Red, no density
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Green, no density
                ]
            ]
        ]),
        lambda: jnp.zeros((1, 1, 3)),
        "zero_density_passthrough",
        1e-6,
    ),
    # Test case 3: Single point with density (boundary case)
    (
        jnp.array([
            [
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.3,
                        0.8,
                        2.0,
                    ],  # Purple point with density 2.0
                ]
            ]
        ]),
        None,  # No specific expected result, just check shape
        "single_point_boundary_case",
        1e-6,
    ),
])
def test_blend_ray_features_with_nerf_paper_method(ray_features, expected_result, description, atol):
    """Test the NeRF paper's volume rendering implementation."""
    result = nerf.blend_ray_features_with_nerf_paper_method(ray_features)
    
    # Always check shape
    assert result.shape == (1, 1, 3), f"Expected shape (1, 1, 3), got {result.shape}"
    
    # Check expected result if provided
    if expected_result is not None:
        expected = expected_result()
        assert jnp.allclose(result, expected, atol=atol), (
            f"Test case '{description}': Expected {expected}, got {result}"
        )
