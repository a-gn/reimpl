"""Unit tests for SyntheticNeRFDatasetForTraining and RayAndColorDataset classes.

Originally written by Claude Sonnet 4 on 2025/10/04
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from reimpl_a_gn.dataset.common import NeRFTrainingSamples, RayAndColorDataset
from reimpl_a_gn.dataset.synthetic_nerf_dataset.loader import (
    SyntheticNeRFDatasetForTraining,
)
from reimpl_a_gn.dataset.synthetic_nerf_dataset.wrapper import SyntheticNeRFData


class MockRayAndColorDataset(RayAndColorDataset):
    """Mock implementation of RayAndColorDataset for testing base class behavior."""

    def __init__(self, rng_key: Array, batch_size: int):
        super().__init__(rng_key, batch_size)
        self.call_count = 0
        self.received_keys = []

    def _get_batch_of_rays(self, rng_key: Array) -> NeRFTrainingSamples:
        """Mock implementation that tracks calls and keys."""
        self.call_count += 1
        self.received_keys.append(rng_key.copy())

        # Return a simple mock batch
        batch_size = self.batch_size
        rays = jnp.ones((batch_size, 6))
        colors = jnp.ones((batch_size, 3))
        extrinsic_matrices = jnp.stack([jnp.eye(4) for _ in range(batch_size)])
        dataset_info = [{"mock": f"sample_{i}"} for i in range(batch_size)]

        return NeRFTrainingSamples(
            rays=rays,
            colors=colors,
            extrinsic_matrices=extrinsic_matrices,
            dataset_info=dataset_info,
        )


@pytest.fixture
def test_synthetic_nerf_data():
    """Create a SyntheticNeRFData instance for testing."""
    # Create test data with known dimensions
    image_count, height, width, channels = 2, 10, 10, 3

    # Create test images with distinct values for each image
    images = jnp.stack(
        [jnp.full((height, width, channels), i * 0.1) for i in range(image_count)]
    )

    # Create test poses - simple identity transformations with different focal lengths
    poses = jnp.stack(
        [
            jnp.array(
                [
                    [1.0, 0.0, 0.0, float(i)],  # x translation varies by image
                    [0.0, 1.0, 0.0, 0.0],  # y translation constant
                    [0.0, 0.0, 1.0, 0.0],  # z translation constant
                    [
                        float(height),
                        float(width),
                        100.0 + i * 10,
                        0.0,
                    ],  # last column: h, w, focal, dummy
                ]
            ).T
            for i in range(image_count)
        ]
    )

    # Create test bounds
    bds = jnp.array([[1.0, 10.0] for _ in range(image_count)])

    # Create render poses (same as regular poses for simplicity)
    render_poses = poses.copy()

    # Test image index (must be less than image_count)
    i_test = 0

    # Create intrinsic matrix (shared for all cameras)
    focal_length = 100.0
    intrinsic_matrix = jnp.array(
        [
            [focal_length, 0.0, width / 2],
            [0.0, focal_length, height / 2],
            [0.0, 0.0, 1.0],
        ]
    )

    # Create extrinsic matrices (world-to-camera)
    extrinsic_matrices = jnp.stack(
        [
            jnp.eye(4).at[0, 3].set(-i)  # move camera in x
            for i in range(image_count)
        ]
    )

    return SyntheticNeRFData(
        images=images,
        poses=poses,
        bds=bds,
        render_poses=render_poses,
        i_test=i_test,
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrices=extrinsic_matrices,
    )


@pytest.fixture
def test_dataset(test_synthetic_nerf_data):
    """Create a SyntheticNeRFDatasetForTraining instance for testing."""
    rng_key = jax.random.PRNGKey(7)
    batch_size = 2
    dataset = SyntheticNeRFDatasetForTraining(test_synthetic_nerf_data)
    dataset.rng_key = rng_key
    dataset.batch_size = batch_size
    return dataset


class TestRayAndColorDataset:
    """Test the base RayAndColorDataset class behavior."""

    def test_initialization(self):
        """Test that initialization sets the correct attributes."""
        rng_key = jax.random.PRNGKey(123)
        batch_size = 2

        dataset = MockRayAndColorDataset(rng_key, batch_size)

        assert jnp.array_equal(dataset.rng_key, rng_key)
        assert dataset.batch_size == batch_size

    def test_iteration_uses_different_rng_keys(self):
        """Test that each iteration uses a different RNG key."""
        rng_key = jax.random.PRNGKey(456)
        batch_size = 4

        dataset = MockRayAndColorDataset(rng_key, batch_size)
        iterator = iter(dataset)

        # Get several batches and verify they're valid
        num_batches = 3
        for _ in range(num_batches):
            batch = next(iterator)
            assert isinstance(batch, NeRFTrainingSamples)

        # Should have called _get_batch_of_rays the expected number of times
        assert dataset.call_count == num_batches

        # Each call should have received a different key
        keys = dataset.received_keys
        assert len(keys) == 3

        # Keys should be different from each other
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert not jnp.array_equal(keys[i], keys[j])

        # Keys should be different from the original key
        for key in keys:
            assert not jnp.array_equal(key, rng_key)

    def test_iteration_updates_internal_rng_key(self):
        """Test that the internal RNG key is updated after each iteration."""
        initial_key = jax.random.PRNGKey(789)
        batch_size = 2

        dataset = MockRayAndColorDataset(initial_key, batch_size)
        original_key = dataset.rng_key.copy()

        iterator = iter(dataset)
        next(iterator)

        # The internal key should have changed
        assert not jnp.array_equal(dataset.rng_key, original_key)

        # Get another batch
        key_after_first = dataset.rng_key.copy()
        next(iterator)

        # The key should have changed again
        assert not jnp.array_equal(dataset.rng_key, key_after_first)

    def test_iteration_returns_correct_batch_structure(self):
        """Test that iteration returns properly structured NeRFTrainingSamples."""
        rng_key = jax.random.PRNGKey(101)
        batch_size = 2

        dataset = MockRayAndColorDataset(rng_key, batch_size)
        iterator = iter(dataset)

        batch = next(iterator)

        # Check that we get a NeRFTrainingSamples object
        assert isinstance(batch, NeRFTrainingSamples)

        # Check dimensions
        assert batch.rays.shape == (batch_size, 6)
        assert batch.colors.shape == (batch_size, 3)
        assert batch.extrinsic_matrices.shape == (batch_size, 4, 4)
        assert len(batch.dataset_info) == batch_size

    def test_multiple_iterations(self):
        """Test that the iterator can generate multiple batches without stopping."""
        rng_key = jax.random.PRNGKey(202)
        batch_size = 3

        dataset = MockRayAndColorDataset(rng_key, batch_size)
        iterator = iter(dataset)

        # Should be able to get many batches
        num_batches = 3
        for _ in range(num_batches):
            batch = next(iterator)
            assert isinstance(batch, NeRFTrainingSamples)

        assert dataset.call_count == num_batches


class TestSyntheticNeRFDatasetForTraining:
    """Test the SyntheticNeRFDatasetForTraining class behavior."""

    def test_initialization(self, test_synthetic_nerf_data):
        """Test that initialization properly stores the data."""
        dataset = SyntheticNeRFDatasetForTraining(test_synthetic_nerf_data)
        assert dataset.all_data is test_synthetic_nerf_data

    def test_get_batch_of_rays_shape_consistency(self, test_dataset):
        """Test that _get_batch_of_rays returns consistently shaped data."""
        rng_key = jax.random.PRNGKey(303)

        batch = test_dataset._get_batch_of_rays(rng_key)

        expected_batch_size = test_dataset.batch_size

        # Check basic structure
        assert isinstance(batch, NeRFTrainingSamples)

        # Check that all arrays have the correct batch dimension
        assert batch.rays.shape == (expected_batch_size, 6)
        assert batch.colors.shape == (expected_batch_size, 3)
        assert batch.extrinsic_matrices.shape == (expected_batch_size, 4, 4)
        assert len(batch.dataset_info) == expected_batch_size

        # Check data types
        assert batch.rays.dtype == jnp.float32
        assert batch.colors.dtype == jnp.float32

        # Check that colors are in valid range [0, 1]
        assert jnp.all(batch.colors >= 0.0)
        assert jnp.all(batch.colors <= 1.0)

    def test_rays_have_valid_structure(self, test_dataset):
        """Test that generated rays have valid structure and properties."""
        rng_key = jax.random.PRNGKey(404)

        batch = test_dataset._get_batch_of_rays(rng_key)

        # Check ray structure: (batch_size, 6) where last axis is (x, y, z, dx, dy, dz)
        assert batch.rays.shape == (test_dataset.batch_size, 6)

        # Check that ray directions (last 3 components) are unit vectors
        ray_directions = batch.rays[:, 3:6]
        direction_norms = jnp.linalg.norm(ray_directions, axis=1)
        assert jnp.allclose(direction_norms, 1.0, rtol=1e-5), (
            "Ray directions should be unit vectors"
        )

    def test_dataset_info_contains_valid_image_indices(self, test_dataset):
        """Test that dataset info contains valid image indices."""
        rng_key = jax.random.PRNGKey(505)

        batch = test_dataset._get_batch_of_rays(rng_key)

        # Check that all image indices are within valid range
        max_image_index = test_dataset.all_data.images.shape[0] - 1
        for info in batch.dataset_info:
            assert "image_index" in info
            image_idx = info["image_index"]
            assert isinstance(image_idx, int)
            assert 0 <= image_idx <= max_image_index

    def test_different_rng_keys_produce_different_results(self, test_dataset):
        """Test that different RNG keys produce different sampling results."""
        rng_key1 = jax.random.PRNGKey(606)
        rng_key2 = jax.random.PRNGKey(607)

        batch1 = test_dataset._get_batch_of_rays(rng_key1)
        batch2 = test_dataset._get_batch_of_rays(rng_key2)

        # With high probability, different keys should produce different results
        # Check that at least some rays are different
        rays_different = not jnp.allclose(batch1.rays, batch2.rays)
        colors_different = not jnp.allclose(batch1.colors, batch2.colors)

        # At least one should be different (with very high probability)
        assert rays_different or colors_different, (
            "Different RNG keys should produce different samples"
        )

    def test_extrinsic_matrices_count_matches_batch_size(self, test_dataset):
        """Test that the number of returned extrinsic matrices matches batch size."""
        rng_key = jax.random.PRNGKey(707)

        batch = test_dataset._get_batch_of_rays(rng_key)

        # Check that we get the correct number of extrinsic matrices
        assert batch.extrinsic_matrices.shape == (test_dataset.batch_size, 4, 4)

    def test_deterministic_behavior_with_same_key(self, test_dataset):
        """Test that the same RNG key produces the same results."""
        rng_key = jax.random.PRNGKey(808)

        batch1 = test_dataset._get_batch_of_rays(rng_key)
        batch2 = test_dataset._get_batch_of_rays(rng_key)

        # Same RNG key should produce identical results
        assert jnp.allclose(batch1.rays, batch2.rays), (
            "Same RNG key should produce identical rays"
        )
        assert jnp.allclose(batch1.colors, batch2.colors), (
            "Same RNG key should produce identical colors"
        )

        # Dataset info should also be identical
        assert len(batch1.dataset_info) == len(batch2.dataset_info)
        for info1, info2 in zip(batch1.dataset_info, batch2.dataset_info):
            assert info1 == info2

    def test_color_sampling_from_correct_images(self, test_dataset):
        """Test that colors correspond to the correct images based on distinct image values."""
        # Generate multiple batches to test color correspondence
        num_test_batches = 3
        for i in range(num_test_batches):
            test_key = jax.random.PRNGKey(909 + i)
            batch = test_dataset._get_batch_of_rays(test_key)

            # For each sample in the batch, verify color consistency
            for j in range(test_dataset.batch_size):
                image_index = batch.dataset_info[j]["image_index"]
                sampled_color = batch.colors[j]

                # Our test images have distinct constant values: image i has value i * 0.1
                expected_color_value = image_index * 0.1
                expected_color = jnp.array([expected_color_value] * 3)  # RGB

                # The sampled color should match the expected value for that image
                assert jnp.allclose(sampled_color, expected_color, rtol=1e-6), (
                    f"Color mismatch: got {sampled_color}, expected {expected_color} for image {image_index}"
                )

    def test_ray_origins_in_camera_coordinate_system(self, test_dataset):
        """Test that ray origins are correctly positioned relative to camera."""
        rng_key = jax.random.PRNGKey(1010)

        batch = test_dataset._get_batch_of_rays(rng_key)

        # Check that ray origins are at the camera center in world coordinates
        # The rays are in world coordinates, so origins should be the camera positions
        for i in range(test_dataset.batch_size):
            ray_origin = batch.rays[i, :3]  # First 3 components are origin
            extrinsic_matrix = batch.extrinsic_matrices[i]
            camera_to_world = jnp.linalg.inv(extrinsic_matrix)

            # Camera position in world coordinates is the translation part of camera_to_world
            expected_origin = camera_to_world[:3, 3]

            assert jnp.allclose(ray_origin, expected_origin, rtol=1e-5), (
                f"Ray origin {ray_origin} doesn't match camera position {expected_origin}"
            )

    def test_extrinsic_matrices_are_from_dataset(self, test_dataset):
        """Test that returned extrinsic matrices are from the original dataset."""
        rng_key = jax.random.PRNGKey(1111)

        batch = test_dataset._get_batch_of_rays(rng_key)

        # Each extrinsic matrix in the batch should be one of the matrices from the dataset
        dataset_extrinsics = test_dataset.all_data.extrinsic_matrices
        for i in range(test_dataset.batch_size):
            image_index = batch.dataset_info[i]["image_index"]
            batch_extrinsic = batch.extrinsic_matrices[i]
            expected_extrinsic = dataset_extrinsics[image_index]

            # The matrices should be identical
            assert jnp.allclose(batch_extrinsic, expected_extrinsic)
