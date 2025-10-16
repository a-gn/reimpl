import jax.numpy as jnp

from reimpl_a_gn.threed.rendering import (
    from_homogeneous,
    to_homogeneous_points,
    to_homogeneous_vectors,
)


class TestHomogeneousCoordinates:
    def test_to_homogeneous_points_single(self):
        point = jnp.array([1.0, 2.0, 3.0])
        result = to_homogeneous_points(point)
        expected = jnp.array([1.0, 2.0, 3.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_to_homogeneous_points_batch(self):
        points = jnp.array([[1.0, 2.0, 3.0]])
        result = to_homogeneous_points(points)
        expected = jnp.array([[1.0, 2.0, 3.0, 1.0]])
        assert jnp.allclose(result, expected)

    def test_to_homogeneous_vectors_single(self):
        vector = jnp.array([1.0, 2.0, 3.0])
        result = to_homogeneous_vectors(vector)
        expected = jnp.array([1.0, 2.0, 3.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_to_homogeneous_vectors_batch(self):
        vectors = jnp.array([[1.0, 2.0, 3.0]])
        result = to_homogeneous_vectors(vectors)
        expected = jnp.array([[1.0, 2.0, 3.0, 0.0]])
        assert jnp.allclose(result, expected)

    def test_from_homogeneous_points(self):
        # Points have w != 0
        homo_points = jnp.array([[2.0, 4.0, 6.0, 2.0]])
        result = from_homogeneous(homo_points)
        expected = jnp.array([[1.0, 2.0, 3.0]])
        assert jnp.allclose(result, expected)

    def test_from_homogeneous_vectors(self):
        # Vectors have w = 0
        homo_vectors = jnp.array([[1.0, 2.0, 3.0, 0.0]])
        result = from_homogeneous(homo_vectors)
        expected = jnp.array([[1.0, 2.0, 3.0]])
        assert jnp.allclose(result, expected)

    def test_from_homogeneous_mixed(self):
        # Mix of points and vectors
        homo_coords = jnp.array([[2.0, 4.0, 6.0, 2.0], [1.0, 2.0, 3.0, 0.0]])
        result = from_homogeneous(homo_coords)
        expected = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        assert jnp.allclose(result, expected)

    def test_roundtrip_points(self):
        original = jnp.array([[1.0, 2.0, 3.0]])
        homo = to_homogeneous_points(original)
        recovered = from_homogeneous(homo)
        assert jnp.allclose(original, recovered)

    def test_roundtrip_vectors(self):
        original = jnp.array([[1.0, 2.0, 3.0]])
        homo = to_homogeneous_vectors(original)
        recovered = from_homogeneous(homo)
        assert jnp.allclose(original, recovered)
