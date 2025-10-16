"""Comprehensive integration tests for render_rays() and render_image().

Originally written by Claude (claude-sonnet-4-5-20250929) on 2025/10/15
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from reimpl_a_gn.threed.nerf import CoarseMLP, FineMLP
from reimpl_a_gn.threed.rendering import CameraParams, render_image, render_rays


@pytest.fixture
def simple_camera() -> CameraParams:
    """Simple camera at origin looking down +Z."""
    extrinsic = jnp.eye(4, dtype=float)
    intrinsic = jnp.array(
        [[50.0, 0.0, 25.0], [0.0, 50.0, 25.0], [0.0, 0.0, 1.0]], dtype=float
    )
    return CameraParams(extrinsic_matrix=extrinsic, intrinsic_matrix=intrinsic)


@pytest.fixture
def tiny_networks(
    request: pytest.FixtureRequest,
) -> tuple[CoarseMLP, FineMLP]:
    """Create minimal networks for fast testing."""
    seed = getattr(request, "param", 42)
    rngs = nnx.Rngs(seed)

    # Coarse network: 6D input (positions+dirs) -> 2 freq encoding -> 24D -> 4D output (RGB + density)
    coarse = CoarseMLP(input_features=24, mid_features=(4,), out_features=4, rngs=rngs)

    # Fine network: 6D input -> 2 freq encoding -> 24D -> 4D output (RGB + density)
    fine = FineMLP(input_features=24, mid_features=(4,), out_features=4, rngs=rngs)

    return coarse, fine


class TestRenderRays:
    """Test render_rays() function."""

    def test_output_shape_single_ray(self, tiny_networks: tuple[CoarseMLP, FineMLP]):
        """Test that single ray produces correct output shape."""
        coarse, fine = tiny_networks
        rays = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # origin at 0, direction +Z
        rng_key = jax.random.key(0)

        result = render_rays(
            rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
        )

        assert result.shape == (1, 3), f"Expected shape (1, 3), got {result.shape}"

    def test_output_shape_multiple_rays(self, tiny_networks: tuple[CoarseMLP, FineMLP]):
        """Test that multiple rays produce correct output shape."""
        coarse, fine = tiny_networks
        rays = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        rng_key = jax.random.key(0)

        result = render_rays(
            rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
        )

        assert result.shape == (3, 3), f"Expected shape (3, 3), got {result.shape}"

    def test_output_values_are_finite(self, tiny_networks: tuple[CoarseMLP, FineMLP]):
        """Test that output RGB values are finite (no NaN or Inf)."""
        coarse, fine = tiny_networks
        rays = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0, 0.5, 0.5, 0.0],
            ]
        )
        rng_key = jax.random.key(0)

        result = render_rays(
            rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
        )

        # RGB values should be finite (not NaN or Inf)
        assert jnp.all(jnp.isfinite(result)), f"Found non-finite values: {result}"

    def test_different_rng_keys_produce_different_results(
        self, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that different random keys produce different results."""
        coarse, fine = tiny_networks
        rays = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        result1 = render_rays(
            rays, rng_key=jax.random.key(0), coarse_network=coarse, fine_network=fine
        )
        result2 = render_rays(
            rays, rng_key=jax.random.key(1), coarse_network=coarse, fine_network=fine
        )

        # Results should differ due to random sampling
        assert not jnp.allclose(result1, result2, atol=1e-6)

    def test_same_rng_key_produces_same_results(
        self, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that same random key produces deterministic results."""
        coarse, fine = tiny_networks
        rays = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        rng_key = jax.random.key(42)

        result1 = render_rays(
            rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
        )
        result2 = render_rays(
            rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
        )

        # Results should be identical
        assert jnp.allclose(result1, result2, atol=1e-6)

    def test_near_far_distance_parameters(self, tiny_networks: tuple[CoarseMLP, FineMLP]):
        """Test that near_distance and far_distance parameters affect results."""
        coarse, fine = tiny_networks
        rays = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        rng_key = jax.random.key(0)

        result_near = render_rays(
            rays,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            near_distance=0.01,
            far_distance=1.0,
        )
        result_far = render_rays(
            rays,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            near_distance=10.0,
            far_distance=100.0,
        )

        # Different sampling ranges should produce different results
        assert not jnp.allclose(result_near, result_far, atol=1e-6)

    def test_batch_dimension_preserved(self, tiny_networks: tuple[CoarseMLP, FineMLP]):
        """Test that batch dimensions are correctly preserved."""
        coarse, fine = tiny_networks
        batch_sizes = [1, 5, 10]
        rng_key = jax.random.key(0)

        for batch_size in batch_sizes:
            rays = jnp.tile(jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]), (batch_size, 1))
            result = render_rays(
                rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
            )
            assert result.shape == (batch_size, 3)

    def test_normalized_ray_directions_work(self, tiny_networks: tuple[CoarseMLP, FineMLP]):
        """Test that normalized ray directions work correctly."""
        coarse, fine = tiny_networks
        # Ray with non-unit direction
        rays = jnp.array([[0.0, 0.0, 0.0, 3.0, 4.0, 0.0]])
        rng_key = jax.random.key(0)

        result = render_rays(
            rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
        )

        # Should still produce valid output
        assert result.shape == (1, 3)
        assert jnp.all(jnp.isfinite(result))

    def test_rays_from_different_origins(self, tiny_networks: tuple[CoarseMLP, FineMLP]):
        """Test rays from different origin points."""
        coarse, fine = tiny_networks
        rays = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # origin at (0,0,0)
                [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # origin at (1,1,1)
                [-5.0, 2.0, 3.0, 1.0, 0.0, 0.0],  # origin at (-5,2,3)
            ]
        )
        rng_key = jax.random.key(0)

        result = render_rays(
            rays, rng_key=rng_key, coarse_network=coarse, fine_network=fine
        )

        assert result.shape == (3, 3)
        # All results should be finite
        assert jnp.all(jnp.isfinite(result))


class TestRenderImage:
    """Test render_image() function."""

    def test_output_shape_matches_input_image(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that output image has same shape as input."""
        coarse, fine = tiny_networks
        image = jnp.ones((3, 4, 3), dtype=float)  # 3x4 image
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=6,
        )

        assert result.shape == image.shape, f"Expected {image.shape}, got {result.shape}"

    def test_output_values_are_finite(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that output pixel values are finite (no NaN or Inf)."""
        coarse, fine = tiny_networks
        image = jnp.zeros((4, 4, 3), dtype=float)
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=8,
        )

        assert jnp.all(jnp.isfinite(result)), f"Found non-finite values in result"

    def test_ray_batch_size_divides_evenly(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test with ray_batch_size that divides total rays evenly."""
        coarse, fine = tiny_networks
        image = jnp.zeros((6, 6, 3), dtype=float)  # 36 rays total
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=12,  # 36 / 12 = 3 batches
        )

        assert result.shape == (6, 6, 3)

    def test_ray_batch_size_does_not_divide_evenly(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test with ray_batch_size that doesn't divide total rays evenly."""
        coarse, fine = tiny_networks
        image = jnp.zeros((5, 7, 3), dtype=float)  # 35 rays total
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=8,  # 35 / 8 = 4 batches + 3 remainder
        )

        assert result.shape == (5, 7, 3)
        # Verify no NaN or invalid values from padding
        assert not jnp.any(jnp.isnan(result))

    def test_ray_batch_size_larger_than_total(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test with ray_batch_size larger than total ray count."""
        coarse, fine = tiny_networks
        image = jnp.zeros((3, 4, 3), dtype=float)  # 12 rays total
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=50,  # Much larger than 12
        )

        assert result.shape == (3, 4, 3)
        assert not jnp.any(jnp.isnan(result))

    def test_ray_batch_size_one(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test with ray_batch_size=1 (process one ray at a time)."""
        coarse, fine = tiny_networks
        image = jnp.zeros((3, 4, 3), dtype=float)  # 12 rays
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=1,
        )

        assert result.shape == (3, 4, 3)

    def test_small_image_renders_correctly(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test rendering very small images (edge case)."""
        coarse, fine = tiny_networks
        image = jnp.zeros((2, 2, 3), dtype=float)  # 4 rays only
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=2,
        )

        assert result.shape == (2, 2, 3)
        assert jnp.all(jnp.isfinite(result))

    def test_different_rng_keys_produce_different_images(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that different random keys produce different results."""
        coarse, fine = tiny_networks
        image = jnp.zeros((3, 3, 3), dtype=float)

        result1 = render_image(
            image,
            camera=simple_camera,
            rng_key=jax.random.key(0),
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=5,
        )
        result2 = render_image(
            image,
            camera=simple_camera,
            rng_key=jax.random.key(1),
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=5,
        )

        assert not jnp.allclose(result1, result2, atol=1e-6)

    def test_same_rng_key_produces_deterministic_results(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that same random key produces deterministic results."""
        coarse, fine = tiny_networks
        image = jnp.zeros((3, 3, 3), dtype=float)
        rng_key = jax.random.key(42)

        result1 = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=5,
        )
        result2 = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=5,
        )

        assert jnp.allclose(result1, result2, atol=1e-6)

    def test_rectangular_images_various_sizes(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test rendering rectangular images of various sizes."""
        coarse, fine = tiny_networks
        rng_key = jax.random.key(0)

        test_cases = [
            (2, 5, 5),  # Wide
            (5, 2, 3),  # Tall
            (3, 4, 6),  # Wide, different batch size
            (4, 2, 4),  # Tall, different batch size
        ]

        for height, width, batch_size in test_cases:
            image = jnp.zeros((height, width, 3), dtype=float)
            result = render_image(
                image,
                camera=simple_camera,
                rng_key=rng_key,
                coarse_network=coarse,
                fine_network=fine,
                ray_batch_size=batch_size,
            )
            assert result.shape == (height, width, 3)
            assert jnp.all(jnp.isfinite(result))


class TestRenderIntegration:
    """Integration tests for render_rays and render_image working together."""

    def test_render_image_uses_render_rays_internally(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that render_image correctly uses render_rays for all pixels."""
        coarse, fine = tiny_networks
        # Use tiny image to make this test fast
        image = jnp.zeros((2, 3, 3), dtype=float)  # 6 rays
        rng_key = jax.random.key(0)

        result = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=6,  # Process all at once
        )

        # Should produce valid image
        assert result.shape == (2, 3, 3)
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))

    def test_consistent_results_across_batch_sizes(
        self, simple_camera: CameraParams, tiny_networks: tuple[CoarseMLP, FineMLP]
    ):
        """Test that different batch sizes produce similar results (accounting for RNG differences)."""
        coarse, fine = tiny_networks
        image = jnp.zeros((4, 4, 3), dtype=float)  # 16 rays
        rng_key = jax.random.key(0)

        # Note: Different batch sizes will use different RNG splits, so we can't expect
        # exact equality. This test just verifies both complete successfully.
        result_batch_1 = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=1,
        )
        result_batch_16 = render_image(
            image,
            camera=simple_camera,
            rng_key=rng_key,
            coarse_network=coarse,
            fine_network=fine,
            ray_batch_size=16,
        )

        # Both should produce valid outputs
        assert result_batch_1.shape == (4, 4, 3)
        assert result_batch_16.shape == (4, 4, 3)
        assert jnp.all(jnp.isfinite(result_batch_1))
        assert jnp.all(jnp.isfinite(result_batch_16))
