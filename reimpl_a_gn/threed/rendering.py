from functools import partial

import jax
import jax.experimental.checkify as checkify
import jax.numpy as jnp
import jax.typing as jt

from reimpl_a_gn.threed.camera import CameraParams


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


def _validate_rays(rays: jt.ArrayLike, ray_count: int) -> jax.Array:
    rays = jnp.array(rays)
    checkify.check(
        len(rays.shape) == 2, "a ray array must have two dimensions (rays then coordinates)"
    )
    checkify.check(
        jnp.all(rays.shape[0] == ray_count),
        "ray_count is not consistent with the array's first dimension's size",
    )
    checkify.check(
        jnp.all(rays.shape[1] == 8),
        "dimension 1 must have size 8: positions then direction vectors in homogeneous coordinates",
    )
    checkify.check(
        jnp.all(rays[:, 3] != 0),
        "ray origins must have non-zero homogeneous weights, those are 3D points",
    )
    checkify.check(
        jnp.all(rays[:, 7] == 0),
        "dimension 7 must be all-zeroes, those are direction vectors",
    )
    return rays


@partial(jax.jit, static_argnames=["ray_count", "pos_per_ray"])
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
    rays = _validate_rays(rays, ray_count)
    result = jnp.zeros([ray_count, pos_per_ray, 4], dtype=float)
    # make coordinates non-homogeneous
    ray_origins = rays[:, :3] / rays[..., 3:4]
    ray_directions = rays[:, 4:7]
    norm_ray_directions = ray_directions / jnp.expand_dims(
        (jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2), -1
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


@partial(
    jax.jit,
    static_argnames=["ray_count", "near_distance", "far_distance", "bins_per_ray"],
)
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
    rays = _validate_rays(rays, ray_count)
    result = jnp.zeros([ray_count, bins_per_ray, 4], dtype=float)
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
        result = result.at[:, bin_i - 1, :].set(sampled_positions)
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
