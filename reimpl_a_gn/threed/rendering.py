from functools import partial

import jax
import jax.numpy as jnp
import jax.typing as jt

from reimpl_a_gn.threed.camera import CameraParams


@partial(jax.jit, static_argnames=["camera_params"])
def get_ray_to_image_coordinates(
    camera_params: CameraParams, pixel_row: float, pixel_col: float
) -> jax.Array:
    """Compute the direction of a ray from the camera origin to a point in the image.

    Assumes that the camera is at the origin and looks along the z-axis.

    @param camera_params Camera parameters.
    @param pixel_row Row index of the pixel. Can be a float to represent subpixel positions.
    @param pixel_col Column index of the pixel. Can be a float to represent subpixel positions.
    @return Unit vector in the direction of the ray from camera origin to a point in the image plane.
    Shape: (image_height, image_width, 4). Last axis: x, y, z, w.
    """
    principal_point_world = jnp.array([0, 0, camera_params.focal_length])
    pixel_center = jnp.array(
        [
            principal_point_world[0] + camera_params.pixel_size_x * pixel_col,
            principal_point_world[1] + camera_params.pixel_size_y * pixel_row,
            principal_point_world[2],
            1,
        ]
    )
    origin_to_pixel = pixel_center
    origin_to_pixel = origin_to_pixel / jnp.linalg.norm(origin_to_pixel, ord=2)
    return origin_to_pixel


# @partial(jax.jit, static_argnames=["camera_params"])
def sample_rays_towards_pixels(
    camera_params: CameraParams,
    points: jt.ArrayLike,
) -> jax.Array:
    """Sample parameters of rays towards pixels in a pinhole camera.

    @param camera_params Pinhole camera parameters.
    @param points Pixel coordinates in the camera's image. Coordinates start in the upper-left corner.
    Shape: (point_count, 2) where the second axis is x, y.
    @return Ray parameters, homogeneous. Shape: (point_count, 6). Third axis: x, y, z, w1, dx, dy, dz, w2.
    The origin of rays is always at the camera center. Since we're in camera frame, the origin is always zero.
    """
    points = jnp.array(points)
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    ray_coords = jnp.zeros((len(points), 8), dtype=jnp.float32)
    # all origins are at zero, set their homogeneous weight to 1
    ray_coords = ray_coords.at[:, 3].set(1)
    for point_index, (x, y) in enumerate(points):
        ray_direction = get_ray_to_image_coordinates(camera_params, x, y)
        ray_coords = ray_coords.at[point_index, 4:8].set(ray_direction)
    return ray_coords


@partial(jax.jit, static_argnames=["camera_params"])
def sample_random_rays_toward_image(
    camera_params: CameraParams,
    image_height: int,
    image_width: int,
    sample_count: int,
    prng_key: jt.ArrayLike,
) -> jax.Array:
    """Sample parameters of rays through a pinhole camera, with random directions.

    Choose random points continuously, uniformly inside the image, and return their direction from the camera.

    @param camera_params Pinhole camera parameters.
    @param image_height Pixel row count of the image.
    @param image_width Pixel column count of the image.
    @param sample_count Number of points to sample.
    @param prng_key Key for jax.random.
    @return Ray parameters in camera coordinates. Shape: (sample_count, 8). Second axis: x, y, z, w1, dx, dy, dz, w2.
    """
    prng_key = jnp.array(prng_key)
    positions_in_image = jax.random.uniform(
        prng_key,
        (sample_count, 2),
        jnp.array((0, 0)),
        jnp.array((image_height, image_width)),
    )
    rays = jnp.zeros((sample_count, 8), dtype=float)
    for sample_id in range(sample_count):
        rays = rays.at[sample_id].set(
            get_ray_to_image_coordinates(
                camera_params,
                positions_in_image[sample_id, 0].item(),
                positions_in_image[sample_id, 1].item(),
            )
        )
    return rays


# @partial(jax.jit, static_argnames=["pos_per_ray"])
def sample_regular_positions_along_rays(
    rays: jt.ArrayLike, near_distance: float, far_distance: float, pos_per_ray: int
) -> jax.Array:
    """Compute regular positions along a set of rays.

    @param rays Ray origins and directions. Shape: (..., 8). Last axis: x, y, z, w, dx, dy, dz, 0.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param pos_per_ray Number of positions to sample along each ray.
    @return Positions along rays. Shape: (..., pos_per_ray, 4). Last axis: x, y, z, w.
    """
    rays = jnp.array(rays)
    assert rays.shape[-1] == 8
    assert jnp.all(rays[:, 7] == 0)
    result_shape = jnp.array([*rays.shape[:-1], pos_per_ray, 4], dtype=int)
    result = jnp.zeros(result_shape, dtype=jnp.float32)
    # make coordinates non-homogeneous
    ray_origins = rays[..., :3] / rays[..., 3:4]
    ray_directions = rays[..., 4:7]
    norm_ray_directions = ray_directions / jnp.expand_dims(
        (jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2), -1
    )
    assert ray_origins.shape[-1] == 3
    assert norm_ray_directions.shape[-1] == 3
    distance_interval = (far_distance - near_distance) / pos_per_ray
    for pos_i in range(1, pos_per_ray + 1):
        sampled_positions = ray_origins + norm_ray_directions * (
            near_distance + pos_i * distance_interval
        )
        assert sampled_positions.shape[-1] == 3
        # make samples homogeneous again
        sampled_positions = jnp.concatenate(
            [sampled_positions, jnp.ones(tuple(sampled_positions.shape[:-1]) + (1,))],
            axis=-1,
        )
        assert sampled_positions.shape[-1] == 4
        result = result.at[..., pos_i - 1, :].set(sampled_positions)
    return result


@partial(jax.jit, static_argnames=["near_distance", "far_distance", "bins_per_ray"])
def sample_nerf_rendering_positions_along_rays(
    rays: jt.ArrayLike,
    near_distance: float,
    far_distance: float,
    bins_per_ray: int,
    prng_key: jax.Array,
):
    """Split (near, far) into regularly-sized bins, then randomly sample one position per bin uniformly.

    @param rays Ray origins and directions. Shape: (..., 8). Last axis: x, y, z, w, dx, dy, dz, 0.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param bins_per_ray Number of bins to split (near_distance, far_distance) into.
    @return Points sampled uniformly for each bin, for each ray. Shape: (..., bins_per_ray, 4). Last axis: x, y, z, w.
    """
    rays = jnp.array(rays)
    assert rays.shape[-1] == 8
    assert jnp.all(rays[:, 7] == 0)
    result = jnp.zeros(list(rays.shape[:-1]) + [bins_per_ray, 4], dtype=jnp.float32)
    # make coordinates non-homogeneous
    ray_origins = rays[..., :3] / rays[..., 3:4]
    ray_directions = rays[..., 4:7]
    norm_ray_directions = ray_directions / jnp.expand_dims(
        (jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2), -1
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
        result = result.at[..., bin_i - 1, :].set(sampled_positions)
    return result


@jax.jit
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


@partial(jax.jit, static_argnames="components")
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
