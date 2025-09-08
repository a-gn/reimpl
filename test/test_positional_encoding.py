import jax.numpy as jnp
import pytest

from reimpl_a_gn.threed.nerf import compute_nerf_positional_encoding


class TestPositionalEncoding:
    def test_basic_shape_single_point(self):
        # Single point with direction (needs to be 2D for the function)
        point_and_dir = jnp.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        components = 4

        result = compute_nerf_positional_encoding(point_and_dir, components)

        # Should have shape (1, 6, 2 * components)
        expected_shape = (1, 6, 2 * components)
        assert result.shape == expected_shape

    def test_basic_shape_batch(self):
        # Batch of points with directions
        points_and_dirs = jnp.array(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]]
        )
        components = 3

        result = compute_nerf_positional_encoding(points_and_dirs, components)

        # Should have shape (batch, 6, 2 * components)
        expected_shape = (2, 6, 2 * components)
        assert result.shape == expected_shape

    def test_encoding_values_properties(self):
        # Test that encoding alternates between sin and cos
        point_and_dir = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        components = 2

        result = compute_nerf_positional_encoding(point_and_dir, components)

        # For the first coordinate (x=1.0), check sin/cos pattern
        x_encoding = result[0, 0, :]  # encoding for x coordinate of first point

        # Should have sin(2^0 * pi * 1), cos(2^0 * pi * 1), sin(2^1 * pi * 1), cos(2^1 * pi * 1)
        expected_x = jnp.array(
            [
                jnp.sin(jnp.pi),  # sin(2^0 * pi * 1)
                jnp.cos(jnp.pi),  # cos(2^0 * pi * 1)
                jnp.sin(2 * jnp.pi),  # sin(2^1 * pi * 1)
                jnp.cos(2 * jnp.pi),  # cos(2^1 * pi * 1)
            ]
        )

        assert jnp.allclose(x_encoding, expected_x, atol=1e-6)

    def test_zero_input(self):
        # Zero input should give predictable pattern
        zero_point = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        components = 2

        result = compute_nerf_positional_encoding(zero_point, components)

        # sin(0) = 0, cos(0) = 1 for all components
        # Each coordinate should have pattern [0, 1, 0, 1, ...]
        for coord_idx in range(6):
            coord_encoding = result[0, coord_idx, :]
            for comp_idx in range(components):
                assert jnp.isclose(coord_encoding[comp_idx * 2], 0.0)  # sin
                assert jnp.isclose(coord_encoding[comp_idx * 2 + 1], 1.0)  # cos

    def test_different_components_count(self):
        point_and_dir = jnp.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])

        for components in [1, 3, 5]:
            result = compute_nerf_positional_encoding(point_and_dir, components)
            assert result.shape == (1, 6, 2 * components)

    def test_invalid_input_shape(self):
        # Wrong number of coordinates
        with pytest.raises(ValueError):
            bad_input = jnp.array([1.0, 2.0, 3.0])  # Only 3 coordinates, need 6
            compute_nerf_positional_encoding(bad_input, 2)

        with pytest.raises(ValueError):
            bad_input = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # 7 coordinates
            compute_nerf_positional_encoding(bad_input, 2)

    def test_power_of_two_progression(self):
        # Verify that frequencies are powers of 2
        point_and_dir = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        components = 3

        result = compute_nerf_positional_encoding(point_and_dir, components)

        x_encoding = result[0, 0, :]  # encoding for x=1.0

        # Check that we get the expected frequencies
        for power in range(components):
            frequency = 2**power
            expected_sin = jnp.sin(frequency * jnp.pi)
            expected_cos = jnp.cos(frequency * jnp.pi)

            assert jnp.isclose(x_encoding[power * 2], expected_sin, atol=1e-6)
            assert jnp.isclose(x_encoding[power * 2 + 1], expected_cos, atol=1e-6)

    def test_batch_consistency(self):
        # Two identical points in batch should have same encoding
        batch_point = jnp.array(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]
        )
        components = 2

        result = compute_nerf_positional_encoding(batch_point, components)

        assert jnp.allclose(result[0], result[1])
