import jax
import jax.numpy as jnp
import jax.typing as jt


class CameraParams:
    """Parameters of a pinhole camera."""

    def __init__(
        self,
        extrinsic_matrix: jt.ArrayLike,
        intrinsic_matrix: jt.ArrayLike,
    ):
        """Initialize camera parameters.

        @param extrinsic_matrix Extrinsic parameters, from world frame to camera frame. Shape: (4, 4).
        @param intrinsic_matrix Intrinsic parameters, from camera frame to image coordinates. Shape: (3, 3).
        """
        self.world_to_camera = jnp.array(extrinsic_matrix)
        if self.world_to_camera.shape != (4, 4):
            raise ValueError(
                f"Expected camera matrix to have shape (3, 4), got {self.world_to_camera.shape}"
            )
        self.camera_to_world = jnp.linalg.inv(self.world_to_camera)

        self.camera_to_image = jnp.array(intrinsic_matrix)
        if self.camera_to_image.shape != (3, 3):
            raise ValueError(
                f"Expected camera matrix to have shape (3, 3), got {self.camera_to_image.shape}"
            )
        self._image_to_camera = jnp.linalg.inv(self.camera_to_image)

    def image_to_camera(self, image_points: jt.ArrayLike) -> jax.Array:
        """Compute the direction of a ray from the camera origin to a point in the image, in the camera frame.

        This is a function because we go through homogeneous coordinates, but the input coordinates are in pixels.

        @param image_points Image points, pixel coordinates. Shape: (point_count, 2). Last axis: x, y.
        @return Unit direction vector in the camera coordinate system. Shape: (point_count, 4). Last axis: x, y, z, 0.
        """
        image_points = jnp.array(image_points)
        image_points_homogeneous = jnp.concatenate(
            [image_points, jnp.ones((image_points.shape[0], 1))], axis=1
        )
        camera_points_inhomogeneous = image_points_homogeneous @ self._image_to_camera.T
        # normalize to unit vectors
        camera_points_inhomogeneous = camera_points_inhomogeneous / jnp.linalg.norm(
            camera_points_inhomogeneous, axis=1, keepdims=True
        )
        # add homogeneous weight of zero (direction vectors, not points)
        camera_points_homogeneous = jnp.concat(
            [
                camera_points_inhomogeneous,
                jnp.zeros([camera_points_inhomogeneous.shape[0], 1]),
            ],
            axis=-1,
        )
        assert len(camera_points_homogeneous.shape) == 2
        assert camera_points_homogeneous.shape[1] == 4
        return camera_points_homogeneous

    def image_to_world(self, image_points: jt.ArrayLike) -> jax.Array:
        """Compute the direction of a ray from the camera origin to a point in the image, in the world frame.

        This is a function because we go through homogeneous coordinates, but the input coordinates are in pixels.

        @param image_points Image points, pixel coordinates. Shape: (point_count, 2). Last axis: x, y.
        @return Unit direction vector in the world coordinate system. Shape: (point_count, 4). Last axis: x, y, z, 0.
        """
        camera_directions = self.image_to_camera(image_points)
        world_directions = camera_directions @ self.camera_to_world.T
        # normalize to unit vectors (homogeneous weight is zero, ignore it)
        world_directions = world_directions.at[:, :3].set(
            world_directions[:, :3]
            / jnp.linalg.norm(world_directions[:, :3], axis=1, keepdims=True)
        )
        return world_directions

    @property
    def fx(self):
        """Focal length divided by the pixel size in x. Point [0, 0] in the intrinsic matrix."""

        return self.camera_to_image[0, 0]

    @property
    def fy(self):
        """Focal length divided by the pixel size in y. Point [1, 1] in the intrinsic matrix."""

        return self.camera_to_image[1, 1]


def extrinsic_matrix_from_pose(position: jt.ArrayLike, direction: jt.ArrayLike):
    """Create a pinhole camera's extrinsic matrix from its position and direction.

    @param position Origin of the camera in world coordinates. Shape: (3,). Order: x, y, z.
    @param direction Viewing direction of the camera in world coordinates. Shape: (3,). Order: dx, dy, dz.
    @return Extrinsic matrix, transforms from world coordinates to camera coordinates. Shape: (4, 4).
    """
    direction = jnp.array(direction)
    position = jnp.array(position)
    position = position / jnp.sqrt(jnp.sum(position**2))
    # compute the inverse: from camera coordinates to world coordinates
    inverse_extrinsic = jnp.array(
        [
            [direction[0], 0, 0, position[0]],
            [0, direction[1], 0, position[1]],
            [0, 0, direction[2], position[2]],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    extrinsic = jnp.linalg.inv(inverse_extrinsic)
    assert extrinsic.shape == (4, 4)
    return extrinsic


def intrinsic_matrix_from_params(
    focal_length: tuple[float, float],
    image_height: int,
    image_width: int,
    skew: float = 0,
):
    """Create the intrinsic matrix of a pinhole camera.

    @param focal_length Focal lengths for x and y axes, in meters.
    @param principal_point x and y pixel coordinates of the intersection between the image plane and the camera's
    z-axis.
    @param pixel_size Size of a pixel on the x, then y axes.
    @param image_height Number of rows of pixels in the image.
    @param image_width Number of columns of pixels in the image.
    @param skew Skew parameter.
    @return Intrinsic matrix from camera frame to image frame. Size: (3, 3).
    """

    principal_point_x = image_width / 2
    principal_point_y = image_height / 2
    return jnp.array(
        [
            [
                focal_length[0],
                skew,
                principal_point_x,
            ],
            [
                0,
                focal_length[1],
                principal_point_y,
            ],
            [0, 0, 1],
        ],
        dtype=float,
    )
