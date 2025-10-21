import jax.numpy as jnp
import pytest

from reimpl_a_gn.threed.rendering import compute_nerf_positional_encoding


class TestPositionalEncoding:
    def test_basic_shape_single_point(self):
        # Single point with direction (needs to be 2D for the function)
        point_and_dir = jnp.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        components = 4

        result = compute_nerf_positional_encoding(point_and_dir, components)

        # Should have shape (1, 6 * 2 * components)
        expected_shape = (1, 6 * 2 * components)
        assert result.shape == expected_shape

    def test_basic_shape_batch(self):
        # Batch of points with directions
        points_and_dirs = jnp.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        components = 3

        result = compute_nerf_positional_encoding(points_and_dirs, components)

        # Should have shape (batch, 6 * 2 * components)
        expected_shape = (1, 6 * 2 * components)
        assert result.shape == expected_shape

    def test_encoding_values(self):
        point_and_dir = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        components = 2

        result = compute_nerf_positional_encoding(point_and_dir, components)

        x_encoding = result[:, :4]  # encoding for x coordinate of first point

        expected_x = jnp.array(
            [
                jnp.sin(jnp.pi),
                jnp.sin(2 * jnp.pi),
                jnp.cos(jnp.pi),
                jnp.cos(2 * jnp.pi),
            ]
        )

        assert jnp.allclose(x_encoding, expected_x, atol=1e-6)

    def test_zero_input(self):
        # Zero input should give predictable pattern
        zero_point = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        components = 2

        result = compute_nerf_positional_encoding(zero_point, components)

        # sines, then cosines, for every input number
        expected_encoding = jnp.repeat(
            jnp.array([[[0.0, 0.0, 1.0, 1.0]]]), repeats=6, axis=-2
        ).reshape(6 * 2 * 2)

        assert jnp.allclose(result, expected_encoding, atol=1e-6)

    def test_different_components_count(self):
        point_and_dir = jnp.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])

        for components in [1, 2]:
            result = compute_nerf_positional_encoding(point_and_dir, components)
            assert result.shape == (1, 6 * 2 * components)

    def test_invalid_input_shape(self):
        # Wrong number of coordinates
        with pytest.raises(ValueError):
            bad_input = jnp.array([1.0, 2.0, 3.0])  # Only 3 coordinates, need 6
            compute_nerf_positional_encoding(bad_input, 2)

        with pytest.raises(ValueError):
            bad_input = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # 7 coordinates
            compute_nerf_positional_encoding(bad_input, 2)

    def test_batch_consistency(self):
        # Identical point should produce deterministic encoding
        point = jnp.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        components = 2

        result1 = compute_nerf_positional_encoding(point, components)
        result2 = compute_nerf_positional_encoding(point, components)

        assert jnp.allclose(result1, result2)
