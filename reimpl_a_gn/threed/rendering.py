import jax
import jax.numpy as jnp
import jax.typing as jt


def norm_eucl_3d(
    points: jt.ArrayLike,
    homogeneous: bool = True,
    keepdims: bool = False,
    add_batch_dimension: bool = False,
) -> jax.Array:
    """Compute the Euclidean distance between the origin and 3D points.

    We always compute the norm over the second dimension (the coordinates).
    In the homogeneous case, we handle both points (non-zero homogeneous weight) and direction vectors (zero weight).

    @param Single or multiple 3D points in an array.
    Shape: `(point_count, 4)`, `(point_count, 3)`, `(4,)`, or `(3,)`, depending on the other arguments.
    When the homogeneous weight is zero, we assume a direction vector and ignore the weight.

    @param homogeneous Whether the points are in homogeneous coordinates or not.
    If True, we expect 4 coordinates. Otherwise, we expect 3.

    @param keepdims If True, the output will have the same number of dimensions as the input and the dimension we
    compute the norm over will have size 1. Otherwise, the dimension we compute the norm over will be squished.

    @param add_batch_dimension If True, we expect a single point in a (3,)- or (4,)-sized array. Otherwise, expect
    a `(point_count, 4)`- or `(point_count, 3)`-sized array of points.

    @return The norm for every point or vector in the input. Shape: either (point_count,) or (point_count, 1), depending
    on the `keepdim` parameter.
    """
    points = jnp.array(points, float)
    if add_batch_dimension and points.ndim != 1:
        raise ValueError(
            f"expected a single point or vector without a batch axis, got shape {points.shape}"
        )
    if not add_batch_dimension and points.ndim != 2:
        raise ValueError(
            f"expected points array to have two dimensions, got shape {points.shape}"
        )

    if add_batch_dimension:
        points = jnp.expand_dims(points, 0)
        assert points.ndim == 2

    if homogeneous:
        if points.shape[1] != 4:
            raise ValueError(
                f"expected points array to have shape (N, 4) because homogeneous=True, got shape {points.shape}"
            )
        points_non_homogeneous = jnp.where(
            points[:, 3:4] == 0,
            points[:, :3],  # vectors have homogeneous weight zero
            points[:, :3] / points[:, 3:4],  # divide point coords by weight
        )
    else:
        if points.shape[1] != 3:
            raise ValueError(
                f"expected points array to have shape (N, 3) because homogeneous=False, got shape {points.shape}"
            )
        points_non_homogeneous = points
    # this is the step that squashes the norm dimension, or not
    norms = jnp.linalg.norm(points_non_homogeneous, ord=2, axis=1, keepdims=keepdims)
    return norms


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
        self.world_to_camera: jax.Array = jnp.array(extrinsic_matrix)
        if self.world_to_camera.shape != (4, 4):
            raise ValueError(
                f"Expected camera matrix to have shape (4, 4), got {self.world_to_camera.shape}"
            )
        self.camera_to_world: jax.Array = jnp.linalg.inv(self.world_to_camera)

        self.camera_to_image: jax.Array = jnp.array(intrinsic_matrix)
        if self.camera_to_image.shape != (3, 3):
            raise ValueError(
                f"Expected camera matrix to have shape (3, 3), got {self.camera_to_image.shape}"
            )
        self._image_to_camera: jax.Array = jnp.linalg.inv(self.camera_to_image)

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
        camera_points_inhomogeneous = camera_points_inhomogeneous / norm_eucl_3d(
            camera_points_inhomogeneous, homogeneous=False, keepdims=True
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
        # These are directions, they have homogeneous weight zero.
        camera_directions = self.image_to_camera(image_points)
        world_directions = camera_directions @ self.camera_to_world.T
        # Normalize to unit vectors (homogeneous weight is zero, ignore it).
        assert jnp.all(world_directions[:, 3] == 0)
        world_directions = world_directions.at[:, :3].set(
            world_directions[:, :3]
            / norm_eucl_3d(world_directions, homogeneous=True, keepdims=True)
        )
        return world_directions

    @property
    def fx(self):
        """Focal length divided by the pixel size in x. Element [0, 0] in the intrinsic matrix."""

        return self.camera_to_image[0, 0]

    @property
    def fy(self):
        """Focal length divided by the pixel size in y. Element [1, 1] in the intrinsic matrix."""

        return self.camera_to_image[1, 1]


def extrinsic_matrix_from_pose(
    camera_origin_world: jt.ArrayLike,
    viewing_direction_world: jt.ArrayLike,
    up_direction_world: jt.ArrayLike,
):
    """Create a pinhole camera's extrinsic matrix from its position and direction.

    @param position Origin of the camera in world coordinates. Shape: (4,). Order: x, y, z, w.
    @param viewing_direction_world Viewing direction of the camera in world coordinates.
        Shape: (4,). Order: dx, dy, dz, 0.
    @param up_direction_world y axis's direction in world coordinates. Must be perpendicular to the viewing direction.
        Shape: (4,). Order: dx, dy, dz, 0.
    @return Extrinsic matrix, transforms from world coordinates to camera coordinates. Shape: (4, 4).

    """
    viewing_direction_world = jnp.array(viewing_direction_world)
    up_direction_world = jnp.array(up_direction_world)
    camera_origin_world = jnp.array(camera_origin_world)

    if viewing_direction_world.shape != (4,):
        raise ValueError(
            f"expected a 3D vector as the viewing direction, got shape {viewing_direction_world.shape}"
        )
    if viewing_direction_world[3] != 0:
        raise ValueError(
            f"expected a 3D vector with homogeneous weight zero as the viewing direction"
            f", got weight {viewing_direction_world[3]}"
        )
    if up_direction_world.shape != (4,):
        raise ValueError(
            f"expected a 3D vector as the up direction, got shape {up_direction_world.shape}"
        )
    if up_direction_world[3] != 0:
        raise ValueError(
            f"expected a 3D vector with homogeneous weight zero as the up direction"
            f", got weight {up_direction_world[3]}"
        )
    if camera_origin_world.shape != (4,):
        raise ValueError(
            f"expected a 3D point as the camera origin, got shape {camera_origin_world.shape}"
        )
    if camera_origin_world[3] == 0:
        raise ValueError(
            f"expected a 3D point with non-zero homogeneous weight as the camera origin"
            f", got weight {camera_origin_world[3]}"
        )

    # compute inhomogeneous, unit vectors for all axes
    viewing_direction_world = (
        viewing_direction_world
        / norm_eucl_3d(viewing_direction_world, add_batch_dimension=True)
    )[:3]
    up_direction_world = (
        up_direction_world / norm_eucl_3d(up_direction_world, add_batch_dimension=True)
    )[:3]
    if 1e-3 < abs((viewing_direction_world @ up_direction_world).item()):
        raise ValueError(
            f"viewing direction {viewing_direction_world.tolist()} and up direction {up_direction_world.tolist()} "
            "do not seem orthogonal (vectors shown here have been normalized to unit vectors)"
        )

    sideways_direction = jnp.cross(viewing_direction_world, up_direction_world)
    assert jnp.allclose(
        norm_eucl_3d(sideways_direction, homogeneous=False, add_batch_dimension=True),
        1.0,
    ), (
        f"sideways direction {sideways_direction.tolist()} does not have Euclidean norm 1.0"
    )

    # compute the inverse: from camera coordinates to world coordinates
    rotation_block = jnp.stack(
        [
            sideways_direction[:3],
            up_direction_world[:3],
            viewing_direction_world[:3],
            jnp.zeros_like(sideways_direction[:3]),
        ],
        axis=0,
    )
    assert rotation_block.shape == (4, 3)
    translation_block = jnp.concat(
        [
            camera_origin_world[:3] / camera_origin_world[3],
            jnp.ones_like(camera_origin_world, shape=(1,)),
        ],
        axis=0,
    )
    translation_block = jnp.expand_dims(translation_block, axis=1)
    assert translation_block.shape == (4, 1)
    inverse_extrinsic = jnp.concat([rotation_block, translation_block], axis=1)
    assert inverse_extrinsic.shape == (4, 4)

    # extrinsic matrix is the inverse
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


def sample_rays_towards_pixels(
    camera_params: CameraParams,
    points: jt.ArrayLike,
) -> jax.Array:
    """Sample parameters of rays towards pixels in a pinhole camera.

    @param camera_params Pinhole camera parameters.
    @param points Pixel coordinates in the camera's image. Coordinates start in the upper-left corner.
    Shape: (point_count, 2) where the second axis is x, y.
    @return Ray parameters, homogeneous. Shape: (point_count, 8). Third axis: x, y, z, w1, dx, dy, dz, zeroes.
    The origin of rays is always at the camera center. Since we're in camera frame, the origin is always zero.
    Dimension 7 is all-zeroes because those are homogeneous direction vectors.
    """
    points = jnp.array(points)
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    ray_coords = jnp.zeros((points.shape[0], 8), dtype=float)
    # all origins are at zero, set their homogeneous weight to 1
    ray_coords = ray_coords.at[:, 3].set(1)
    # compute directions of rays
    ray_directions = camera_params.image_to_camera(points)
    ray_coords = ray_coords.at[:, 4:8].set(ray_directions)
    return ray_coords

