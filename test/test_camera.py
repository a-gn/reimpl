"""Test 3D camera functions."""

import jax as jax
import jax.numpy as jnp
import jax.typing as jt
import numpy

from reimpl_a_gn.threed import camera as camera


def test_pose_to_extrinsic():
    position = jnp.array([-2.34, 5.5, -7], dtype=float)
    direction = jnp.array([3.5, 2.4, 23], dtype=float)
    extrinsic = camera.extrinsic_matrix_from_pose(position, direction)
    assert extrinsic.shape == (4, 4)
    assert numpy.linalg.det(extrinsic) != 0  # invertible
