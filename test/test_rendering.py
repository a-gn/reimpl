import jax
import jax.numpy as jnp
import pytest

import reimpl_a_gn.threed.rendering as rendering
from reimpl_a_gn.threed.nerf import (
    compute_rays_in_world_frame,
    sample_coarse_mlp_inputs,
)


@pytest.fixture
def first_flower_camera() -> rendering.CameraParams:
    """First camera from the flowers dataset."""
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
    return rendering.CameraParams(
        extrinsic_matrix=extrinsic_matrix, intrinsic_matrix=intrinsic_matrix
    )


@pytest.fixture
def flower_rays(first_flower_camera: rendering.CameraParams) -> jnp.ndarray:
    return compute_rays_in_world_frame(first_flower_camera, (-3, 4), (-2, 2))


def test_extrinsic_camera_from_valid_camera_params():
    forward_direction = jnp.array([-1.0, -2.0, 1.0, 0.0])
    up_direction = jnp.array([2.0, 1.0, 4.0, 0.0])
    camera_origin = jnp.array([4.0, 5.5, -12.2, 1.0])

    extrinsic_matrix = rendering.extrinsic_matrix_from_pose(
        camera_origin, forward_direction, up_direction
    )

    assert jnp.linalg.det(extrinsic_matrix) != 0


def test_extrinsic_camera_with_non_orthogonal_directions():
    forward_direction = jnp.array([-1.0, -2.0, 1.0, 0.0])
    up_direction = jnp.array([-2.0, 1.0, 4.0, 0.0])
    camera_origin = jnp.array([4.0, 5.5, -12.2, 1.0])

    with pytest.raises(ValueError):
        rendering.extrinsic_matrix_from_pose(
            camera_origin, forward_direction, up_direction
        )


def test_sample_coarse_positions(flower_rays: jnp.ndarray):
    prng_key = jax.random.key(7)
    actual_result = sample_coarse_mlp_inputs(
        rays=flower_rays,
        near_distance=0.5,
        far_distance=5.0,
        bins_per_ray=3,
        prng_key=prng_key,
    )
