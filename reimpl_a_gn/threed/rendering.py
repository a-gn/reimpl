from functools import partial

import jax
import jax.numpy as jnp
import jax.typing as jt


# @partial(jax.jit, static_argnames=["homogeneous", "keepdims"])
def norm_eucl_3d(
    points: jt.ArrayLike, homogeneous: bool = True, keepdims: bool = False
):
    """Compute the Euclidean distance between the origin and 3D points.

    We always compute the norm over the second dimension (the coordinates).
    In the homogeneous case, we handle both points (non-zero homogeneous weight) and direction vectors (zero weight).

    @param points 2D array of 3D points or vectors in homogeneous coordinates.
    Shape: (point_count, 4) or (point_count, 3), depending on the `homogeneous` argument.
    When the homogeneous weight is zero, we assume a direction vector and ignore the weight.
    @param homogeneous Whether the points are in homogeneous coordinates or not.
    If True, we expect 4 coordinates. Otherwise, we expect 3.
    @param keepdims If True, the output will have the same number of dimensions as the input and the dimension we
    compute the norm over will have size 1. Otherwise, the dimension we compute the norm over will be squished.
    @return The norm for every point or vector in the input. Shape: either (point_count,) or (point_count, 1), depending
    on the `keepdim` parameter.
    """
    points = jnp.array(points, float)
    assert len(points.shape) == 2
    if homogeneous:
        assert points.shape[1] == 4
        points_non_homogeneous = jnp.where(
            points[:, 3:4] == 0,
            points[:, :3],  # vectors have homogeneous weight zero
            points[:, :3] / points[:, 3:4],  # divide point coords by weight
        )
    else:
        assert points.shape[1] == 3
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


# @partial(jax.jit, static_argnames=["camera_params"])
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


# @partial(jax.jit, static_argnames=["ray_count", "pos_per_ray"])
def sample_regular_positions_along_rays(
    rays: jt.ArrayLike,
    ray_count: int,
    near_distance: float,
    far_distance: float,
    pos_per_ray: int,
) -> jax.Array:
    """Compute regular positions along a set of rays.

    @param rays Ray origins and directions. Shape: (ray_count, 8). Last axis: x, y, z, w, dx, dy, dz, 0.
    @param ray_count Number of rays in the input. Must be equal to the size of rays' first dimension.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param pos_per_ray Number of positions to sample along each ray.
    @param out_shape Shape of the output array. Last two axes must have size pos_per_ray and 4, respectively.
    @return Positions along rays. Shape: (ray_count, pos_per_ray, 4). Last axis: x, y, z, w.
    """
    rays = jnp.array(rays)
    result = jnp.zeros([ray_count, pos_per_ray, 4], dtype=float)
    # make coordinates non-homogeneous
    ray_origins = rays[:, :3] / rays[..., 3:4]
    ray_directions = rays[:, 4:7]
    norm_ray_directions = ray_directions / norm_eucl_3d(
        ray_directions, homogeneous=False, keepdims=True
    )
    assert ray_origins.shape[-1] == 3
    assert norm_ray_directions.shape[-1] == 3
    distance_interval = (far_distance - near_distance) / pos_per_ray
    for position_index in range(pos_per_ray):
        sampled_positions = ray_origins + norm_ray_directions * (
            near_distance + (position_index + 1) * distance_interval
        )
        assert sampled_positions.shape == (ray_count, 3)
        # make samples homogeneous again
        sampled_positions = jnp.concatenate(
            [sampled_positions, jnp.ones(tuple(sampled_positions.shape[:-1]) + (1,))],
            axis=-1,
        )
        assert sampled_positions.shape[-1] == 4
        result = result.at[:, position_index, :].set(sampled_positions)
    return result


# @partial(
#     jax.jit,
#     static_argnames=["ray_count", "near_distance", "far_distance", "bins_per_ray"],
# )
def sample_nerf_rendering_positions_along_rays(
    rays: jt.ArrayLike,
    ray_count: int,
    near_distance: float,
    far_distance: float,
    bins_per_ray: int,
    prng_key: jax.Array,
):
    """Split (near, far) into regularly-sized bins, then randomly sample one position per bin uniformly.

    @param rays Ray origins and directions. Shape: (..., 8). Last axis: x, y, z, w, dx, dy, dz, 0.
    @param ray_count Number of rays in the input. Must be equal to the size of rays' first dimension.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param bins_per_ray Number of bins to split (near_distance, far_distance) into.
    @return Points sampled uniformly for each bin, for each ray. Shape: (..., bins_per_ray, 4). Last axis: x, y, z, w.
    """
    rays = jnp.array(rays)
    result = jnp.zeros([ray_count, bins_per_ray, 4], dtype=float)
    # make coordinates non-homogeneous
    ray_origins = rays[..., :3] / rays[..., 3:4]
    ray_directions = rays[..., 4:7]
    norm_ray_directions = ray_directions / norm_eucl_3d(
        ray_directions, homogeneous=False, keepdims=True
    )
    bin_width = (far_distance - near_distance) / bins_per_ray
    positions_shape_per_bin = list(rays.shape[:-1]) + [1]
    for bin_i in range(1, bins_per_ray + 1):
        bin_start = near_distance + bin_i * bin_width
        bin_end = near_distance + (bin_i + 1) * bin_width
        position_on_ray = jax.random.uniform(
            prng_key,
            # keep first axes of rays, just remove the last one and replace with 3
            positions_shape_per_bin,
            dtype=float,
            minval=bin_start,
            maxval=bin_end,
        )
        sampled_positions = ray_origins + norm_ray_directions * position_on_ray
        assert sampled_positions.shape[-1] == 3
        # make samples homogeneous again
        sampled_positions = jnp.concatenate(
            [sampled_positions, jnp.ones((*sampled_positions.shape[:-1], 1))], axis=-1
        )
        assert sampled_positions.shape[-1] == 4
        result = result.at[:, bin_i - 1, :].set(sampled_positions)
    return result


# @jax.jit
def blend_ray_features_with_nerf_paper_method(
    ray_features: jax.Array, bins_per_ray: int
) -> jax.Array:
    """Compute one color for each ray, by using the NeRF paper's rendering method.

    We split the (near_point, far_point) interval into N regularly-sized bins, then sample one point, $t_i$, uniformly
    inside each bin. We then use alpha-rendering with this formula:

    $C(r) = \\sum_{i=1}^{N}{c(t_i)T(t_i)(1-\\exp(-\\sigma(t_i)\\delta(t_i)))}$

    where $T(t_i) = \\exp{-\\sum_{j=1}^{i-1}{\\sigma(t_j)}}$ is the weight of each color (goes down exponentially with
    the sum of densities of the previous intervals) and $\\delta(T-i)$ is the distance between the previous point
    $t_{i-1}$ and $t_i$.

    @param ray_features Coordinates, color, and transparency sampled along rays. Shape: (..., pos_per_ray, 7).
    Second axis: x, y, z, R, G, B, sigma.
    @return One color per ray. Shape: (num_rays, ..., 3). Last axis: R, G, B.
    """
    ray_features = jnp.array(ray_features)
    # compute length and center of intervals between samples
    interval_lengths = ray_features[..., 1::, :3] - ray_features[..., ::-1, :3]
    interval_centers = ray_features[..., ::-1, :3] + (
        (ray_features[..., 1::, :3] - ray_features[..., ::-1, :3]) / 2
    )
    # compute distances between origin and interval centers and samples
    origin_center_distances = norm_eucl_3d(interval_centers)
    origin_sample_distances = norm_eucl_3d(ray_features[..., :, :3])
    # interpolate values at midpoints between samples
    center_values = jnp.interp(
        origin_center_distances, origin_sample_distances, ray_features[..., :, 3:]
    )
    # weight interpolated colors at midpoints with interval lengths and interpolated sigma values
    blended_values = jnp.sum(
        center_values[..., :3] * center_values[..., 3] * interval_lengths, axis=-2
    )
    return blended_values


# @partial(jax.jit, static_argnames="components")
def compute_nerf_positional_encoding(
    points_and_directions: jt.ArrayLike, components: int
):
    """Compute the NeRF paper's positional encoding of a set of points and associated directions.

    @param points_and_directions Data to encode. Shape: (..., 6). Last axis: x, y, z, dx, dy, dz.
    @return Positional encoding of the points. Shape: (..., 2 * components).
    """

    points_and_directions = jnp.array(points_and_directions)
    result = jnp.zeros(
        list(points_and_directions.shape[:-1]) + [6, 2 * components], dtype=float
    )
    for power_of_two in range(components):
        result = result.at[..., power_of_two * 2].set(
            jnp.sin(jnp.pow(2, power_of_two) * jnp.pi * points_and_directions)
        )
        result = result.at[..., power_of_two * 2 + 1].set(
            jnp.cos(jnp.pow(2, power_of_two) * jnp.pi * points_and_directions)
        )
    return result
