import jax
import jax.numpy as jnp
import jax.typing as jt
import jax.lax as lax
import numpy  # some operations aren't supported on jax-metal


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
        self.image_to_camera = jnp.array(numpy.linalg.inv(self.camera_to_image))

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
        image_points_homogeneous = jnp.concatenate([image_points, jnp.ones((image_points.shape[0], 1))], axis=1)
        inverse_camera_matrix = self.image_to_camera
        return (inverse_camera_matrix @ image_points_homogeneous.transpose()).transpose()


def get_ray_to_image_coordinates(camera_params: CameraParams, pixel_row: float, pixel_col: float) -> jax.Array:
    """Compute the direction of a ray from the camera origin to a point in the image.

    Assumes that the camera is at the origin and looks along the z-axis.

    @param camera_params Camera parameters.
    @param pixel_row Row index of the pixel. Can be a float to represent subpixel positions.
    @param pixel_col Column index of the pixel. Can be a float to represent subpixel positions.
    @return Unit vector in the direction of the ray from camera origin to a point in the image plane.
    Shape: (image_height, image_width, 3). Last axis: x, y, z.
    """
    principal_point_world = jnp.array([0, 0, camera_params.focal_length])
    pixel_center = jnp.array([
        principal_point_world[0] + camera_params.pixel_size_x * pixel_col,
        principal_point_world[1] + camera_params.pixel_size_y * pixel_row,
        principal_point_world[2],
    ])
    origin_to_pixel = pixel_center
    assert jnp.linalg.norm(origin_to_pixel, ord=2) > 0.0
    origin_to_pixel = origin_to_pixel / jnp.linalg.norm(origin_to_pixel, ord=2)
    return origin_to_pixel


def sample_rays_towards_all_pixels(
    camera_params: CameraParams,
    image_height: int,
    image_width: int,
) -> jax.Array:
    """Sample parameters of rays through a pinhole camera, which can be used to render an image.

    @param camera_params Camera matrix. Shape: (3, 4).
    @param image_height Pixel row count of the image we want to render.
    @param image_width Pixel column count of the image we want to render.
    @return Ray parameters. Shape: (image_height, image_width, 6). Second axis: x, y, z, dx, dy, dz.
    """
    ray_coords = jnp.zeros((image_height, image_width, 6), dtype=jnp.float32)
    for row_i in range(image_height):
        for col_i in range(image_width):
            ray_direction = get_ray_to_image_coordinates(camera_params, row_i, col_i)
            ray_coords = ray_coords.at[row_i, col_i, 3:6].set(ray_direction)
    return ray_coords


def sample_regular_positions_along_rays(
    rays: jax.Array, near_distance: float, far_distance: float, pos_per_ray: int
) -> jax.Array:
    """Compute regular positions along a set of rays.

    @param rays Ray parameters. Shape: (..., 6). Last axis: x, y, z, dx, dy, dz.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param pos_per_ray Number of positions to sample along each ray.
    @return Positions along rays. Shape: (..., pos_per_ray, 3). Last axis: x, y, z.
    """
    rays = jnp.array(rays)
    result = jnp.zeros(list(rays.shape[:-1]) + [pos_per_ray, 3], dtype=jnp.float32)
    ray_origins = rays[..., :3]
    ray_directions = rays[..., 3:]
    norm_ray_directions = ray_directions / jnp.expand_dims(
        (jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2), -1
    )
    distance_interval = (far_distance - near_distance) / pos_per_ray
    for pos_i in range(1, pos_per_ray + 1):
        result = result.at[..., pos_i - 1, :].set(
            ray_origins
            + norm_ray_directions * (near_distance + pos_i * distance_interval)
        )
    return result


def sample_nerf_rendering_positions_along_rays(
    rays: jax.Array,
    near_distance: float,
    far_distance: float,
    bins_per_ray: int,
    prng_key: jax.Array,
):
    """Split (near, far) into regularly-sized bins, then randomly sample one position per bin uniformly.

    @param rays Ray parameters. Shape: (..., 6). Last axis: x, y, z, dx, dy, dz.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param bins_per_ray Number of bins to split (near_distance, far_distance) into.
    @return Points sampled uniformly for each bin, for each ray. Shape: (..., bins_per_ray, 3). Last axis: x, y, z.
    """
    rays = jnp.array(rays)
    result = jnp.zeros(list(rays.shape[:-1]) + [bins_per_ray, 3], dtype=jnp.float32)
    ray_origins = rays[..., :3]
    ray_directions = rays[..., 3:]
    norm_ray_directions = ray_directions / jnp.expand_dims(
        (jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2), -1
    )
    bin_width = (far_distance - near_distance) / bins_per_ray
    positions_shape_per_bin = list(rays.shape[:-1])
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
        sampled_points = ray_origins + norm_ray_directions * jnp.expand_dims(
            position_on_ray, -1
        )
        result = result.at[..., bin_i - 1, :].set(sampled_points)
    return result


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
    origin_center_distances = jnp.linalg.norm(interval_centers, axis=-1, ord=2) ** 1 / 2
    origin_sample_distances = (
        jnp.linalg.norm(ray_features[..., :, :3], axis=-1, ord=2) ** 1 / 2
    )
    # interpolate values at midpoints between samples
    center_values = jnp.interp(
        origin_center_distances, origin_sample_distances, ray_features[..., :, 3:]
    )
    # weight interpolated colors at midpoints with interval lengths and interpolated sigma values
    blended_values = jnp.sum(
        center_values[..., :3] * center_values[..., 3] * interval_lengths, axis=-2
    )
    return blended_values


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
