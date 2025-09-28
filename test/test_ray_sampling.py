import jax.numpy as jnp

from reimpl_a_gn.threed.rendering import sample_regular_positions_along_rays


class TestRaySampling:
    def test_sample_regular_positions_basic_shape(self):
        # Create simple rays
        rays = jnp.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # ray along x-axis
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # ray along y-axis
            ]
        )
        ray_count = 2
        near_distance = 1.0
        far_distance = 5.0
        pos_per_ray = 4

        positions = sample_regular_positions_along_rays(
            rays, ray_count, near_distance, far_distance, pos_per_ray
        )

        # Check output shape: (ray_count, pos_per_ray, 3)
        assert positions.shape == (ray_count, pos_per_ray, 3)

    def test_sample_regular_positions_along_x_axis(self):
        # Ray along x-axis from origin
        rays = jnp.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        ray_count = 1
        near_distance = 2.0
        far_distance = 6.0
        pos_per_ray = 4

        positions = sample_regular_positions_along_rays(
            rays, ray_count, near_distance, far_distance, pos_per_ray
        )

        # Check that positions are along x-axis
        assert jnp.allclose(positions[0, :, 1], 0.0)  # y = 0
        assert jnp.allclose(positions[0, :, 2], 0.0)  # z = 0

        # Check that x coordinates are regularly spaced between near and far
        x_coords = positions[0, :, 0]
        expected_interval = (far_distance - near_distance) / pos_per_ray
        expected_x = jnp.array(
            [
                near_distance + expected_interval,  # 3.0
                near_distance + 2 * expected_interval,  # 4.0
                near_distance + 3 * expected_interval,  # 5.0
                near_distance + 4 * expected_interval,  # 6.0
            ]
        )
        assert jnp.allclose(x_coords, expected_x)

    def test_sample_regular_positions_different_directions(self):
        # Two rays in different directions
        rays = jnp.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # x direction
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # z direction
            ]
        )
        ray_count = 2
        near_distance = 1.0
        far_distance = 3.0
        pos_per_ray = 2

        positions = sample_regular_positions_along_rays(
            rays, ray_count, near_distance, far_distance, pos_per_ray
        )

        # First ray should sample along x-axis
        assert jnp.allclose(positions[0, :, 1], 0.0)  # y = 0
        assert jnp.allclose(positions[0, :, 2], 0.0)  # z = 0
        assert positions[0, 0, 0] > 0  # positive x

        # Second ray should sample along z-axis
        assert jnp.allclose(positions[1, :, 0], 0.0)  # x = 0
        assert jnp.allclose(positions[1, :, 1], 0.0)  # y = 0
        assert positions[1, 0, 2] > 0  # positive z

    def test_sample_regular_positions_non_unit_direction(self):
        # Ray with non-unit direction vector
        rays = jnp.array([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]])  # 2x unit vector in x
        ray_count = 1
        near_distance = 1.0
        far_distance = 3.0
        pos_per_ray = 2

        positions = sample_regular_positions_along_rays(
            rays, ray_count, near_distance, far_distance, pos_per_ray
        )

        # Should still produce normalized results
        x_coords = positions[0, :, 0]
        expected_x = jnp.array([2.0, 3.0])  # at distances 1.0 + 1.0 and 1.0 + 2.0
        assert jnp.allclose(x_coords, expected_x)

    def test_sample_regular_positions_from_offset_origin(self):
        # Ray starting from non-zero origin
        rays = jnp.array(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0]]
        )  # origin at (1,2,3), direction +x
        ray_count = 1
        near_distance = 1.0
        far_distance = 3.0
        pos_per_ray = 2

        positions = sample_regular_positions_along_rays(
            rays, ray_count, near_distance, far_distance, pos_per_ray
        )

        # Positions should be offset by the origin
        expected_positions = jnp.array(
            [
                [1.0 + 2.0, 2.0, 3.0],  # origin + 2 * direction
                [1.0 + 3.0, 2.0, 3.0],  # origin + 3 * direction
            ]
        )
        assert jnp.allclose(positions[0], expected_positions)
