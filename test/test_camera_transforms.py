"""Comprehensive unit tests for CameraParams coordinate transformations.

Originally written by Claude (claude-sonnet-4-5-20250929) on 2025/10/15
"""
import jax.numpy as jnp
import pytest

from reimpl_a_gn.threed.rendering import CameraParams


@pytest.fixture
def identity_camera() -> CameraParams:
    """Camera at origin looking down +Z with standard intrinsics."""
    extrinsic = jnp.eye(4, dtype=float)
    intrinsic = jnp.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=float
    )
    return CameraParams(extrinsic_matrix=extrinsic, intrinsic_matrix=intrinsic)


@pytest.fixture
def translated_camera() -> CameraParams:
    """Camera translated to (1, 2, 3) looking down +Z."""
    extrinsic = jnp.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    intrinsic = jnp.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=float
    )
    return CameraParams(extrinsic_matrix=extrinsic, intrinsic_matrix=intrinsic)


class TestCameraParamsInit:
    """Test CameraParams initialization and validation."""

    def test_valid_initialization(self):
        """Test that valid matrices initialize correctly."""
        extrinsic = jnp.eye(4, dtype=float)
        intrinsic = jnp.eye(3, dtype=float)
        camera = CameraParams(extrinsic_matrix=extrinsic, intrinsic_matrix=intrinsic)

        assert camera.world_to_camera.shape == (4, 4)
        assert camera.camera_to_world.shape == (4, 4)
        assert camera.camera_to_image.shape == (3, 3)

    def test_invalid_extrinsic_shape(self):
        """Test that invalid extrinsic matrix shape raises ValueError."""
        extrinsic = jnp.eye(3, dtype=float)
        intrinsic = jnp.eye(3, dtype=float)

        with pytest.raises(ValueError, match="Expected camera matrix to have shape.*4, 4"):
            CameraParams(extrinsic_matrix=extrinsic, intrinsic_matrix=intrinsic)

    def test_invalid_intrinsic_shape(self):
        """Test that invalid intrinsic matrix shape raises ValueError."""
        extrinsic = jnp.eye(4, dtype=float)
        intrinsic = jnp.eye(4, dtype=float)

        with pytest.raises(ValueError, match="Expected camera matrix to have shape.*3, 3"):
            CameraParams(extrinsic_matrix=extrinsic, intrinsic_matrix=intrinsic)

    def test_camera_to_world_is_inverse(self, identity_camera: CameraParams):
        """Test that camera_to_world is the inverse of world_to_camera."""
        product = identity_camera.world_to_camera @ identity_camera.camera_to_world
        assert jnp.allclose(product, jnp.eye(4))


class TestImageToCameraTransform:
    """Test image_to_camera transformation."""

    def test_center_pixel_maps_to_forward_direction(self, identity_camera: CameraParams):
        """Test that the center pixel (principal point) maps to camera +Z direction."""
        # Principal point is at (50, 50) from intrinsic matrix
        center_point = jnp.array([[50.0, 50.0]])
        camera_direction = identity_camera.image_to_camera(center_point)

        # Should be unit vector along +Z in camera frame (homogeneous: [0, 0, 1, 0])
        expected = jnp.array([[0.0, 0.0, 1.0, 0.0]])
        assert jnp.allclose(camera_direction, expected, atol=1e-6)

    def test_output_is_unit_vector(self, identity_camera: CameraParams):
        """Test that output direction vectors are normalized."""
        points = jnp.array([[0.0, 0.0], [100.0, 100.0], [25.0, 75.0]])
        directions = identity_camera.image_to_camera(points)

        # Check each direction has unit norm (ignoring homogeneous coordinate)
        norms = jnp.linalg.norm(directions[:, :3], axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_homogeneous_weight_is_zero(self, identity_camera: CameraParams):
        """Test that output homogeneous weight is zero (direction, not point)."""
        points = jnp.array([[0.0, 0.0], [100.0, 100.0]])
        directions = identity_camera.image_to_camera(points)

        assert jnp.all(directions[:, 3] == 0.0)

    def test_left_of_center_has_negative_x(self, identity_camera: CameraParams):
        """Test that pixels left of center have negative X in camera frame."""
        # Pixel at (0, 50) is left of center (50, 50)
        left_point = jnp.array([[0.0, 50.0]])
        direction = identity_camera.image_to_camera(left_point)

        # X component should be negative
        assert direction[0, 0] < 0

    def test_right_of_center_has_positive_x(self, identity_camera: CameraParams):
        """Test that pixels right of center have positive X in camera frame."""
        # Pixel at (100, 50) is right of center (50, 50)
        right_point = jnp.array([[100.0, 50.0]])
        direction = identity_camera.image_to_camera(right_point)

        # X component should be positive
        assert direction[0, 0] > 0

    def test_above_center_has_negative_y(self, identity_camera: CameraParams):
        """Test that pixels above center have negative Y in camera frame."""
        # In image coordinates, Y increases downward, but camera Y is upward
        above_point = jnp.array([[50.0, 0.0]])
        direction = identity_camera.image_to_camera(above_point)

        # Y component should be negative (image Y=0 is top)
        assert direction[0, 1] < 0

    def test_below_center_has_positive_y(self, identity_camera: CameraParams):
        """Test that pixels below center have positive Y in camera frame."""
        below_point = jnp.array([[50.0, 100.0]])
        direction = identity_camera.image_to_camera(below_point)

        # Y component should be positive
        assert direction[0, 1] > 0

    def test_batch_processing(self, identity_camera: CameraParams):
        """Test that multiple points are processed correctly."""
        points = jnp.array([[50.0, 50.0], [0.0, 0.0], [100.0, 100.0]])
        directions = identity_camera.image_to_camera(points)

        assert directions.shape == (3, 4)
        # All should be unit vectors
        norms = jnp.linalg.norm(directions[:, :3], axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)


class TestImageToWorldTransform:
    """Test image_to_world transformation."""

    def test_identity_camera_matches_image_to_camera(self, identity_camera: CameraParams):
        """Test that for identity camera, world and camera frames are the same."""
        points = jnp.array([[50.0, 50.0], [0.0, 0.0]])
        camera_directions = identity_camera.image_to_camera(points)
        world_directions = identity_camera.image_to_world(points)

        # For identity extrinsic, camera and world should match
        assert jnp.allclose(camera_directions, world_directions, atol=1e-6)

    def test_output_is_unit_vector(self, translated_camera: CameraParams):
        """Test that output direction vectors are normalized."""
        points = jnp.array([[0.0, 0.0], [100.0, 100.0], [50.0, 50.0]])
        directions = translated_camera.image_to_world(points)

        # Check each direction has unit norm (ignoring homogeneous coordinate)
        norms = jnp.linalg.norm(directions[:, :3], axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_homogeneous_weight_is_zero(self, translated_camera: CameraParams):
        """Test that output homogeneous weight is zero (direction, not point)."""
        points = jnp.array([[0.0, 0.0], [100.0, 100.0]])
        directions = translated_camera.image_to_world(points)

        assert jnp.all(directions[:, 3] == 0.0)

    def test_translation_does_not_affect_directions(self):
        """Test that camera translation doesn't change ray directions (only origins change)."""
        # Create two cameras with same orientation but different positions
        extrinsic1 = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        extrinsic2 = jnp.array(
            [
                [1.0, 0.0, 0.0, -5.0],
                [0.0, 1.0, 0.0, -10.0],
                [0.0, 0.0, 1.0, -15.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        intrinsic = jnp.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=float
        )

        camera1 = CameraParams(extrinsic_matrix=extrinsic1, intrinsic_matrix=intrinsic)
        camera2 = CameraParams(extrinsic_matrix=extrinsic2, intrinsic_matrix=intrinsic)

        points = jnp.array([[50.0, 50.0], [0.0, 0.0], [100.0, 100.0]])
        directions1 = camera1.image_to_world(points)
        directions2 = camera2.image_to_world(points)

        # Ray directions should be identical despite different camera positions
        assert jnp.allclose(directions1, directions2, atol=1e-6)

    def test_rotated_camera_changes_directions(self):
        """Test that camera rotation changes ray directions in world frame."""
        # Camera rotated 90 degrees around Y axis
        extrinsic_rotated = jnp.array(
            [
                [0.0, 0.0, -1.0, 0.0],  # X' = -Z
                [0.0, 1.0, 0.0, 0.0],  # Y' = Y
                [1.0, 0.0, 0.0, 0.0],  # Z' = X
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        intrinsic = jnp.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=float
        )

        identity_cam = CameraParams(extrinsic_matrix=jnp.eye(4), intrinsic_matrix=intrinsic)
        rotated_cam = CameraParams(
            extrinsic_matrix=extrinsic_rotated, intrinsic_matrix=intrinsic
        )

        center_point = jnp.array([[50.0, 50.0]])
        dir_identity = identity_cam.image_to_world(center_point)
        dir_rotated = rotated_cam.image_to_world(center_point)

        # Directions should be different
        assert not jnp.allclose(dir_identity, dir_rotated, atol=1e-6)

        # Identity camera looks along +Z: [0, 0, 1, 0]
        assert jnp.allclose(dir_identity, jnp.array([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)

        # Rotated camera looks along +X: [1, 0, 0, 0]
        assert jnp.allclose(dir_rotated, jnp.array([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6)

    def test_batch_processing(self, translated_camera: CameraParams):
        """Test that multiple points are processed correctly."""
        points = jnp.array([[50.0, 50.0], [0.0, 0.0], [100.0, 100.0]])
        directions = translated_camera.image_to_world(points)

        assert directions.shape == (3, 4)
        # All should be unit vectors
        norms = jnp.linalg.norm(directions[:, :3], axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)


class TestFocalLengthProperties:
    """Test fx and fy property accessors."""

    def test_fx_returns_correct_value(self):
        """Test that fx returns the [0,0] element of intrinsic matrix."""
        intrinsic = jnp.array(
            [[123.45, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=float
        )
        camera = CameraParams(extrinsic_matrix=jnp.eye(4), intrinsic_matrix=intrinsic)

        assert camera.fx == 123.45

    def test_fy_returns_correct_value(self):
        """Test that fy returns the [1,1] element of intrinsic matrix."""
        intrinsic = jnp.array(
            [[100.0, 0.0, 50.0], [0.0, 234.56, 50.0], [0.0, 0.0, 1.0]], dtype=float
        )
        camera = CameraParams(extrinsic_matrix=jnp.eye(4), intrinsic_matrix=intrinsic)

        assert camera.fy == 234.56

    def test_different_focal_lengths(self):
        """Test camera with different fx and fy (non-square pixels)."""
        intrinsic = jnp.array(
            [[200.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=float
        )
        camera = CameraParams(extrinsic_matrix=jnp.eye(4), intrinsic_matrix=intrinsic)

        assert camera.fx == 200.0
        assert camera.fy == 100.0
        assert camera.fx != camera.fy
