import jax
import jax.numpy as jnp
import jax.typing as jt
import pytest

import reimpl_a_gn.threed.rendering as rendering


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
