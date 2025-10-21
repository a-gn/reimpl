import jax
import jax.numpy as jnp
import pytest

import reimpl_a_gn.threed.coord_utils as coord_utils


@pytest.fixture
def first_flower_camera_matrices() -> tuple[jnp.ndarray, jnp.ndarray]:
    """First camera from the flowers dataset - returns (intrinsic, extrinsic) matrices."""
    extrinsic_matrix = jnp.array(
        [
            [0.99982196, 0.01527655, -0.01106967, 0.33459657],
            [-0.01512637, 0.9997941, 0.01352618, -0.14106995],
            [0.01127402, -0.01335633, 0.9998473, 0.08382977],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    intrinsic_matrix = jnp.array(
        [[446.88232, 0.0, 252.0], [0.0, 446.88232, 189.0], [0.0, 0.0, 1.0]], dtype=float
    )
    return intrinsic_matrix, extrinsic_matrix


@pytest.fixture
def flower_rays(
    first_flower_camera_matrices: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    intrinsic_matrix, extrinsic_matrix = first_flower_camera_matrices
    return coord_utils.compute_rays_in_world_frame(
        intrinsic_matrix, extrinsic_matrix, (-3, 4), (-2, 2)
    )


def test_extrinsic_camera_from_valid_camera_params():
    forward_direction = jnp.array([-1.0, -2.0, 1.0, 0.0])
    up_direction = jnp.array([2.0, 1.0, 4.0, 0.0])
    camera_origin = jnp.array([4.0, 5.5, -12.2, 1.0])

    extrinsic_matrix = coord_utils.extrinsic_matrix_from_pose(
        camera_origin, forward_direction, up_direction
    )

    assert jnp.linalg.det(extrinsic_matrix) != 0


def test_extrinsic_camera_with_non_orthogonal_directions():
    forward_direction = jnp.array([-1.0, -2.0, 1.0, 0.0])
    up_direction = jnp.array([-2.0, 1.0, 4.0, 0.0])
    camera_origin = jnp.array([4.0, 5.5, -12.2, 1.0])

    with pytest.raises(ValueError):
        coord_utils.extrinsic_matrix_from_pose(
            camera_origin, forward_direction, up_direction
        )


def test_sample_coarse_positions(flower_rays: jnp.ndarray):
    prng_key = jax.random.key(7)
    actual_result = coord_utils.sample_coarse_mlp_inputs(
        rays=flower_rays,
        near_distance=0.5,
        far_distance=5.0,
        bins_per_ray=3,
        prng_key=prng_key,
    )
    # Verify the output shape is correct (should be 6D now instead of 8D)
    assert actual_result.shape[-1] == 6


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

    computed_distribution = coord_utils.compute_fine_sampling_distribution(
        densities, sampling_positions
    )
    print(f"computed:\n{computed_distribution}\nexpected:\n{expected_distribution}")
    assert jnp.allclose(computed_distribution, expected_distribution)


@pytest.mark.parametrize(
    "ray_features,expected_result,description,atol",
    [
        # Test case 1: Simple case with 2 points to make calculation clearer
        (
            jnp.array(
                [
                    [
                        [
                            # Point 1: position (0,0,0), color (1,0,0), density 1.0
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                            # Point 2: position (1,0,0), color (0,1,0), density 2.0
                            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0],
                        ]
                    ]
                ]
            ),
            lambda: jnp.array(
                [
                    [
                        [
                            1.0
                            * (1 - jnp.exp(-1.0))
                            * 1.0,  # Red: only first point contributes
                            1.0
                            * (1 - jnp.exp(-1.0))
                            * 0.0,  # Green: only first point contributes
                            1.0
                            * (1 - jnp.exp(-1.0))
                            * 0.0,  # Blue: only first point contributes
                        ]
                    ]
                ]
            ),
            "two_points_with_density",
            1e-5,
        ),
        # Test case 2: Edge case - zero density should pass through completely
        (
            jnp.array(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Red, no density
                            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Green, no density
                        ]
                    ]
                ]
            ),
            lambda: jnp.zeros((1, 1, 3)),
            "zero_density_passthrough",
            1e-6,
        ),
        # Test case 3: Single point with density (boundary case)
        (
            jnp.array(
                [
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
                ]
            ),
            None,  # No specific expected result, just check shape
            "single_point_boundary_case",
            1e-6,
        ),
    ],
)
def test_blend_ray_features_with_nerf_paper_method(
    ray_features, expected_result, description, atol
):
    """Test the NeRF paper's volume rendering implementation."""
    result = coord_utils.blend_ray_features_with_nerf_paper_method(ray_features)

    # Always check shape
    assert result.shape == (1, 1, 3), f"Expected shape (1, 1, 3), got {result.shape}"

    # Check expected result if provided
    if expected_result is not None:
        expected = expected_result()
        assert jnp.allclose(result, expected, atol=atol), (
            f"Test case '{description}': Expected {expected}, got {result}"
        )
