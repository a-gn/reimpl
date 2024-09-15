import jax
import jax.numpy as jnp
import jax.typing as jt
import numpy


class CameraParams:
    """Parameters of a pinhole camera."""

    def __init__(
        self,
        extrinsic_matrix: jt.ArrayLike,
        intrinsic_matrix: jt.ArrayLike,
        focal_length: float,
    ):
        """Initialize camera parameters.

        @param extrinsic_matrix Extrinsic parameters, from world frame to camera frame. Shape: (4, 4).
        @param intrinsic_matrix Intrinsic parameters, from camera frame to image coordinates. Shape: (3, 3).
        @param focal_length Focal length of the camera.
        """
        self.world_to_camera = jnp.array(extrinsic_matrix)
        if self.world_to_camera.shape != (4, 4):
            raise ValueError(
                f"Expected camera matrix to have shape (3, 4), got {self.world_to_camera.shape}"
            )
        self.camera_to_world = jnp.array(numpy.linalg.inv(self.world_to_camera))

        self.camera_to_image = jnp.array(intrinsic_matrix)
        if self.camera_to_image.shape != (3, 3):
            raise ValueError(
                f"Expected camera matrix to have shape (3, 3), got {self.camera_to_image.shape}"
            )
        self._image_to_camera = jnp.array(numpy.linalg.inv(self.camera_to_image))

        self.focal_length = focal_length
        self.pixel_size_x = (1 / self.camera_to_image[0, 0]) * focal_length
        self.pixel_size_y = (1 / self.camera_to_image[1, 1]) * focal_length

    def image_points_to_camera(self, image_points: jt.ArrayLike) -> jax.Array:
        """Compute the direction of a ray from the camera origin to a point in the image.

        This is a function because we go through homogeneous coordinates, but the input coordinates are in pixels.

        @param image_points Image points, pixel coordinates. Shape: (point_count, 2). Last axis: x, y.
        @return Point coordinates in the camera coordinate system. Shape: (point_count, 3). Last axis: x, y, z.
        """
        image_points = jnp.array(image_points)
        image_points_homogeneous = jnp.concatenate(
            [image_points, jnp.ones((image_points.shape[0], 1))], axis=1
        )
        inverse_camera_matrix = self._image_to_camera
        return (
            inverse_camera_matrix @ image_points_homogeneous.transpose()
        ).transpose()
