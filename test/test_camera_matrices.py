import jax.numpy as jnp

from reimpl_a_gn.threed.rendering import (
    CameraParams,
    intrinsic_matrix_from_params,
    sample_rays_towards_pixels,
)


class TestCameraMatrices:
    def test_intrinsic_matrix_basic_properties(self):
        focal_length = (500.0, 500.0)
        image_height = 480
        image_width = 640

        intrinsic = intrinsic_matrix_from_params(
            focal_length, image_height, image_width
        )

        # Check shape
        assert intrinsic.shape == (3, 3)

        # Check focal lengths are set correctly
        assert intrinsic[0, 0] == focal_length[0]
        assert intrinsic[1, 1] == focal_length[1]

        # Check principal point is at image center
        assert intrinsic[0, 2] == image_width / 2
        assert intrinsic[1, 2] == image_height / 2

        # Check bottom row is standard
        assert jnp.allclose(intrinsic[2, :], jnp.array([0, 0, 1]))

    def test_intrinsic_matrix_with_different_focal_lengths(self):
        focal_length = (400.0, 300.0)
        image_height = 240
        image_width = 320

        intrinsic = intrinsic_matrix_from_params(
            focal_length, image_height, image_width
        )

        assert intrinsic[0, 0] == 400.0
        assert intrinsic[1, 1] == 300.0
        assert intrinsic[0, 2] == 160.0  # width / 2
        assert intrinsic[1, 2] == 120.0  # height / 2

    def test_intrinsic_matrix_with_skew(self):
        focal_length = (500.0, 500.0)
        image_height = 480
        image_width = 640
        skew = 10.0

        intrinsic = intrinsic_matrix_from_params(
            focal_length, image_height, image_width, skew
        )

        assert intrinsic[0, 1] == skew

    def test_sample_rays_towards_pixels_basic(self):
        # Create a simple camera setup
        extrinsic = jnp.eye(4)
        intrinsic = intrinsic_matrix_from_params((500.0, 500.0), 480, 640)
        camera = CameraParams(extrinsic, intrinsic)

        # Sample rays towards a few pixels
        pixels = jnp.array([[320.0, 240.0], [100.0, 100.0]])  # center and corner
        rays = sample_rays_towards_pixels(camera, pixels)

        # Check output shape
        assert rays.shape == (2, 6)

        # Ray origins should be at (0, 0, 0) in camera coordinates
        assert jnp.allclose(rays[:, :3], 0.0)

        # Check that directions are unit vectors
        directions = rays[:, 3:6]
        norms = jnp.linalg.norm(directions, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_sample_rays_towards_center_pixel(self):
        # Camera looking down the z-axis
        extrinsic = jnp.eye(4)
        intrinsic = intrinsic_matrix_from_params((500.0, 500.0), 480, 640)
        camera = CameraParams(extrinsic, intrinsic)

        # Ray towards center pixel should point down z-axis
        center_pixel = jnp.array([[320.0, 240.0]])
        rays = sample_rays_towards_pixels(camera, center_pixel)

        direction = rays[0, 3:6]
        # Should be approximately (0, 0, 1) - pointing down z-axis
        assert abs(direction[0]) < 0.1  # small x component
        assert abs(direction[1]) < 0.1  # small y component
        assert direction[2] > 0.9  # large positive z component

    def test_sample_rays_different_pixels_different_directions(self):
        extrinsic = jnp.eye(4)
        intrinsic = intrinsic_matrix_from_params((500.0, 500.0), 480, 640)
        camera = CameraParams(extrinsic, intrinsic)

        # Sample rays towards different pixels
        pixels = jnp.array([[0.0, 0.0], [640.0, 480.0]])  # corners
        rays = sample_rays_towards_pixels(camera, pixels)

        # Directions should be different
        dir1, dir2 = rays[0, 3:6], rays[1, 3:6]
        assert not jnp.allclose(dir1, dir2)
