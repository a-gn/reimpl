"""Coordinate transformation and ray manipulation utilities for 3D graphics.

Originally written by Claude Sonnet 4.5 on 2025/10/21
"""

import jax
import jax.numpy as jnp
import jax.typing as jt

from reimpl_a_gn.random import piecewise_uniform


def to_homogeneous_points(points: jt.ArrayLike) -> jax.Array:
    """Convert 3D points to homogeneous coordinates.

    @param points Shape: (..., 3). Last axis: x, y, z.
    @return Homogeneous points. Shape: (..., 4). Last axis: x, y, z, w=1.
    """
    points = jnp.array(points)
    ones = jnp.ones(points.shape[:-1] + (1,))
    return jnp.concatenate([points, ones], axis=-1)


def to_homogeneous_vectors(vectors: jt.ArrayLike) -> jax.Array:
    """Convert 3D vectors to homogeneous coordinates.

    @param vectors Shape: (..., 3). Last axis: dx, dy, dz.
    @return Homogeneous vectors. Shape: (..., 4). Last axis: dx, dy, dz, w=0.
    """
    vectors = jnp.array(vectors)
    zeros = jnp.zeros(vectors.shape[:-1] + (1,))
    return jnp.concatenate([vectors, zeros], axis=-1)


def from_homogeneous(coords: jt.ArrayLike) -> jax.Array:
    """Convert homogeneous coordinates back to 3D.

    @param coords Shape: (..., 4). Last axis: x, y, z, w.
    @return 3D coordinates. Shape: (..., 3). For points, divides by w. For vectors (w=0), keeps xyz.
    """
    coords = jnp.array(coords)
    is_vector = coords[..., 3] == 0
    # For vectors (w=0), just take xyz. For points, divide by w.
    result = jnp.where(
        jnp.expand_dims(is_vector, -1),
        coords[..., :3],  # vectors: keep xyz as-is
        coords[..., :3] / jnp.expand_dims(coords[..., 3], -1),  # points: divide by w
    )
    return result


def make_non_homogeneous(point: jt.ArrayLike):
    """Convert homogeneous point to non-homogeneous coordinates.

    @param point Homogeneous point. Shape: (4,). Last axis: x, y, z, w.
    @return Non-homogeneous point. Shape: (3,). Order: x/w, y/w, z/w.
    """
    point = jnp.array(point)
    return point[:3] / point[3]


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
    viewing_direction_world = jnp.array(
        viewing_direction_world / jnp.linalg.norm(viewing_direction_world[:3])
    )[:3]
    up_direction_world = jnp.array(
        up_direction_world / jnp.linalg.norm(up_direction_world[:3])
    )[:3]
    if 1e-3 < abs((viewing_direction_world @ up_direction_world).item()):
        raise ValueError(
            f"viewing direction {viewing_direction_world.tolist()} and up direction {up_direction_world.tolist()} "
            "do not seem orthogonal (vectors shown here have been normalized to unit vectors)"
        )

    sideways_direction = jnp.cross(viewing_direction_world, up_direction_world)
    assert jnp.allclose(
        jnp.linalg.norm(sideways_direction),
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


def image_to_camera(
    image_points: jt.ArrayLike, intrinsic_matrix: jt.ArrayLike
) -> jax.Array:
    """Compute direction of rays from camera origin to image points, in camera frame.

    @param image_points Image points, pixel coordinates. Shape: (point_count, 2). Last axis: x, y.
    @param intrinsic_matrix Intrinsic matrix from camera frame to image coordinates. Shape: (3, 3).
    @return Unit direction vectors in camera coordinate system. Shape: (point_count, 4). Last axis: x, y, z, 0.
    """
    image_points = jnp.array(image_points)
    intrinsic_matrix = jnp.array(intrinsic_matrix)

    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            f"Expected intrinsic matrix to have shape (3, 3), got {intrinsic_matrix.shape}"
        )

    image_to_camera_matrix = jnp.linalg.inv(intrinsic_matrix)

    # Convert to homogeneous coordinates
    image_points_homogeneous = jnp.concatenate(
        [image_points, jnp.ones((image_points.shape[0], 1))], axis=1
    )

    # Transform to camera coordinates
    camera_points_inhomogeneous = image_points_homogeneous @ image_to_camera_matrix.T

    # Normalize to unit vectors
    camera_points_inhomogeneous = camera_points_inhomogeneous / jnp.linalg.norm(
        camera_points_inhomogeneous, axis=1, keepdims=True
    )

    # Add homogeneous weight of zero (direction vectors, not points)
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


def image_to_world(
    image_points: jt.ArrayLike,
    intrinsic_matrix: jt.ArrayLike,
    extrinsic_matrix: jt.ArrayLike,
) -> jax.Array:
    """Compute direction of rays from camera origin to image points, in world frame.

    @param image_points Image points, pixel coordinates. Shape: (point_count, 2). Last axis: x, y.
    @param intrinsic_matrix Intrinsic matrix from camera frame to image coordinates. Shape: (3, 3).
    @param extrinsic_matrix Extrinsic matrix from world frame to camera frame. Shape: (4, 4).
    @return Unit direction vectors in world coordinate system. Shape: (point_count, 4). Last axis: x, y, z, 0.
    """
    extrinsic_matrix = jnp.array(extrinsic_matrix)

    if extrinsic_matrix.shape != (4, 4):
        raise ValueError(
            f"Expected extrinsic matrix to have shape (4, 4), got {extrinsic_matrix.shape}"
        )

    camera_to_world = jnp.linalg.inv(extrinsic_matrix)

    # Get directions in camera frame
    camera_directions = image_to_camera(image_points, intrinsic_matrix)

    # Transform to world frame
    world_directions = camera_directions @ camera_to_world.T

    # Normalize to unit vectors (homogeneous weight is zero, ignore it)
    assert jnp.all(world_directions[:, 3] == 0)
    world_directions = world_directions.at[:, :3].set(
        world_directions[:, :3]
        / jnp.linalg.norm(world_directions[:, :3], axis=1, keepdims=True)
    )

    return world_directions


def compute_rays_in_world_frame(
    intrinsic_matrix: jt.ArrayLike,
    extrinsic_matrix: jt.ArrayLike,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
):
    """Compute the origin and direction of rays from the camera origin to pixels in the image.

    @param intrinsic_matrix Intrinsic matrix from camera frame to image coordinates. Shape: (3, 3).
    @param extrinsic_matrix Extrinsic matrix from world frame to camera frame. Shape: (4, 4).
    @param x_range Range of x pixel coordinates to generate rays for.
    @param y_range Range of y pixel coordinates to generate rays for.
    @return ray directions and origins. Shape: (ray_count, 6). Second axis: (x, y, z, dx, dy, dz).
    """
    extrinsic_matrix = jnp.array(extrinsic_matrix)
    camera_to_world = jnp.linalg.inv(extrinsic_matrix)

    # compute rays in world frame
    ray_targets_x_image, ray_targets_y_image = jnp.meshgrid(
        jnp.arange(*x_range), jnp.arange(*y_range)
    )
    ray_targets_image = jnp.stack(
        [ray_targets_x_image, ray_targets_y_image], axis=-1
    ).reshape(-1, 2)
    ray_directions_world_homo = image_to_world(
        ray_targets_image, intrinsic_matrix, extrinsic_matrix
    )
    # origin of rays is origin of camera
    ray_origins_world_homo = jnp.array([[0.0, 0.0, 0.0, 1.0]]) @ camera_to_world.T
    # same origin for all rays, concatenate needs axis 0 to have the same size as directions
    ray_origins_world_homo = jnp.repeat(
        ray_origins_world_homo, axis=0, repeats=ray_directions_world_homo.shape[0]
    )

    # Convert to inhomogeneous coordinates
    ray_origins_world = from_homogeneous(ray_origins_world_homo)
    ray_directions_world = from_homogeneous(ray_directions_world_homo)

    ray_directions_and_origins_world = jnp.concatenate(
        [ray_origins_world, ray_directions_world], axis=1
    )
    return ray_directions_and_origins_world


def sample_rays_towards_pixels(
    intrinsic_matrix: jt.ArrayLike,
    points: jt.ArrayLike,
) -> jax.Array:
    """Sample parameters of rays towards pixels in a pinhole camera.

    @param intrinsic_matrix Intrinsic matrix from camera frame to image coordinates. Shape: (3, 3).
    @param points Pixel coordinates in the camera's image. Coordinates start in the upper-left corner.
    Shape: (point_count, 2) where the second axis is x, y.
    @return Ray parameters, inhomogeneous. Shape: (point_count, 6). Last axis: x, y, z, dx, dy, dz.
    The origin of rays is always at the camera center (0, 0, 0) in camera coordinates.
    """
    points = jnp.array(points)
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    ray_coords = jnp.zeros((points.shape[0], 6), dtype=float)
    # origins are at zero (already set)
    # compute directions of rays (convert from homogeneous to inhomogeneous)
    ray_directions_homo = image_to_camera(points, intrinsic_matrix)
    ray_directions = from_homogeneous(ray_directions_homo)
    ray_coords = ray_coords.at[:, 3:6].set(ray_directions)
    return ray_coords


def get_rays(image_height: int, image_width: int, intrinsic_matrix: jt.ArrayLike):
    """Generate ray parameters for all pixels in an image.

    @param image_height Number of rows of pixels in the image.
    @param image_width Number of columns of pixels in the image.
    @param intrinsic_matrix Intrinsic matrix from camera frame to image coordinates. Shape: (3, 3).
    @return Tuple of (pixel_coords, pixel_rays) where pixel_coords has shape (height*width, 2)
        and pixel_rays has shape (height*width, 6).
    """
    pixel_xs, pixel_ys = jnp.meshgrid(
        jnp.arange(0, image_height), jnp.arange(0, image_width)
    )
    pixel_coords = jnp.stack([pixel_xs.flatten(), pixel_ys.flatten()], axis=1)
    assert pixel_coords.ndim == 2 and pixel_coords.shape[1] == 2

    pixel_rays = sample_rays_towards_pixels(intrinsic_matrix, pixel_coords)
    return pixel_coords, pixel_rays


def sample_regular_positions_along_rays(
    rays: jt.ArrayLike,
    ray_count: int,
    near_distance: float,
    far_distance: float,
    pos_per_ray: int,
) -> jax.Array:
    """Compute regular positions along a set of rays.

    @param rays Ray origins and directions. Shape: (ray_count, 6). Last axis: x, y, z, dx, dy, dz.
    @param ray_count Number of rays in the input. Must be equal to the size of rays' first dimension.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param pos_per_ray Number of positions to sample along each ray.
    @return Positions along rays. Shape: (ray_count, pos_per_ray, 3). Last axis: x, y, z.
    """
    rays = jnp.array(rays)
    result = jnp.zeros([ray_count, pos_per_ray, 3], dtype=float)
    ray_origins = rays[:, :3]
    ray_directions = rays[:, 3:6]
    norm_ray_directions = ray_directions / jnp.linalg.norm(
        ray_directions, axis=1, keepdims=True
    )
    assert ray_origins.shape[-1] == 3
    assert norm_ray_directions.shape[-1] == 3
    distance_interval = (far_distance - near_distance) / pos_per_ray
    for position_index in range(pos_per_ray):
        sampled_positions = ray_origins + norm_ray_directions * (
            near_distance + (position_index + 1) * distance_interval
        )
        assert sampled_positions.shape == (ray_count, 3)
        result = result.at[:, position_index, :].set(sampled_positions)
    return result


def sample_coarse_mlp_positions(
    rays: jt.ArrayLike,
    near_distance: float,
    far_distance: float,
    bins_per_ray: int,
    prng_key: jax.Array,
):
    """Split (near, far) into regularly-sized bins, then randomly sample one position per bin uniformly.

    @param rays Ray origins and directions. Shape: (ray_count, 6). Last axis: x, y, z, dx, dy, dz.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param bins_per_ray Number of bins to split (near_distance, far_distance) into.
    @return Points and direction vectors for each bin, for each ray. Shape: (ray_count, bins_per_ray, 6).
        Last axis: x, y, z, dx, dy, dz.
    """
    rays = jnp.array(rays)
    ray_origins = rays[:, :3]
    ray_directions = rays[:, 3:6]
    norm_ray_directions = ray_directions / jnp.linalg.norm(
        ray_directions, axis=1, keepdims=True
    )

    # sample points on rays
    bin_boundaries = jnp.reshape(
        # N bins means N + 1 bounds
        jnp.linspace(near_distance, far_distance, bins_per_ray + 1),
        (1, bins_per_ray + 1, 1),
    )  # shape: (1, bins_per_ray + 1, 1)
    sampled_distances_along_rays = jax.random.uniform(
        prng_key,
        (rays.shape[0], bins_per_ray, 1),  # with coordinate axis
        minval=bin_boundaries[:, :-1],
        maxval=bin_boundaries[:, 1:],
    )
    sampled_positions = (
        jnp.expand_dims(ray_origins, 1)  # shape: (ray_count, 1, 3)
        + jnp.expand_dims(norm_ray_directions, 1)  # shape: (ray_count, 1, 3)
        * sampled_distances_along_rays
    )

    # add direction vectors to the result
    sampled_points_and_directions = jnp.concatenate(
        [
            sampled_positions,
            jnp.repeat(
                ray_directions.reshape(rays.shape[0], 1, 3), bins_per_ray, axis=1
            ),
        ],
        axis=-1,
    )

    return sampled_points_and_directions


def compute_fine_sampling_distribution(
    densities: jt.ArrayLike, sampling_positions: jt.ArrayLike
):
    """Compute the distributions from which to sample points to pass through the fine MLP, for a single ray.

    We compute weights for each sampling position along the ray, adjusting the probability down according to densities.

    This is meant to sample from the more computationally expensive MLP using the results of the coarse MLP.
    See the NeRF paper for details.

    @param densities Density values predicted by the coarse MLP. Shape: (..., num_samples,).
    @param sampling_positions Positions along the ray at which `densities` were predicted. Same shape as `densities`.
    Must be strictly increasing along the last axis.
    @return A piecewise-uniform probability distribution represented as a `(..., num_samples - 1,)`-shaped array of
    probability values. Item with index `n` is the distribution's value in the `n`th interval.
    """
    densities = jnp.array(densities)
    sampling_positions = jnp.array(sampling_positions)

    interval_lengths = sampling_positions[..., 1:] - sampling_positions[..., :-1]

    # T(r(s))
    accumulated_transmittance = jnp.exp(
        jnp.cumulative_sum(
            -densities[..., :-2] * interval_lengths[..., :-1],
            axis=-1,
            include_initial=True,
        )
    )
    unnormalized_pdf = accumulated_transmittance * (
        1 - jnp.exp(-densities[..., :-1] * interval_lengths)
    )
    pdf = unnormalized_pdf / jnp.sum(unnormalized_pdf, axis=-1, keepdims=True)
    return pdf


def sample_from_fine_sampling_distribution(
    pdf: jax.Array,
    rays: jax.Array,
    positions: jax.Array,
    sample_count_per_distribution: int,
    rng_key: jax.Array,
):
    """Sample positions for the fine MLP of a NeRF from a PDF over ray positions and convert them to 3D positions.

    @param pdf Piecewise-uniform probability densities for each interval. Shape: (ray_count, interval_count)
    @param rays 3D origin and unit direction vector of each ray. Shape: (ray_count, 6). Last axis: x, y, z, dx, dy, dz.
    @param positions 3D coordinates corresponding to the PDF intervals. Shape: (ray_count, interval_count + 1)
    @param rng_key Key to use to generate pseudorandom numbers.
    """

    assert rays.ndim == 2 and rays.shape[1] == 6
    ray_origins = rays[..., 0:3]
    ray_directions = rays[..., 3:6]

    # Broadcast origins, compute distances from origins to positions
    positions_on_rays = jnp.linalg.norm(positions - jnp.expand_dims(ray_origins, -1))
    interval_lengths = positions_on_rays[..., 1:] - positions_on_rays[..., :-1]
    interval_probabilities = interval_lengths * pdf
    sampled_positions_on_rays = piecewise_uniform(
        key=rng_key,
        intervals=positions_on_rays,
        interval_probabilities=interval_probabilities,
        sample_count_per_distribution=sample_count_per_distribution,
    )
    sampled_positions = (
        jnp.expand_dims(ray_origins, -1)
        + jnp.expand_dims(ray_directions, -1) * sampled_positions_on_rays
    )
    return sampled_positions


def compute_nerf_positional_encoding(
    points_and_directions: jt.ArrayLike, components: int
):
    """Compute the NeRF paper's positional encoding of a set of points and associated directions.

    @param points_and_directions Rays to encode. Shape: (..., 6). Last axis: x, y, z, dx, dy, dz.
    @return Positional encoding of the points. Shape: (..., 6 * 2 * components). The embeddings of all coordinates are
    concatenated on the last dimension (the output has the same number of dimension as the input).
    """

    points_and_directions = jnp.array(points_and_directions)
    if points_and_directions.ndim < 2 or points_and_directions.shape[-1] != 6:
        raise ValueError(
            f"expected input shape (..., 6), got shape {points_and_directions.shape}"
        )

    exponents = jnp.arange(0, components).reshape(
        # 1s over all input axes
        *([1] * (len(points_and_directions.shape))),
        # all components on a new axis so that product broadcasts each of the 6 input coordinates over all exponents
        components,
    )
    points_and_directions_with_broadcast_axis = points_and_directions.reshape(
        *points_and_directions.shape, 1
    )
    arguments = (
        jnp.pow(2, exponents) * jnp.pi * points_and_directions_with_broadcast_axis
    )
    sine_results = jnp.sin(arguments)
    cosine_results = jnp.cos(arguments)
    full_results = jnp.concatenate([sine_results, cosine_results], axis=-1)
    full_results = full_results.reshape(
        *(points_and_directions.shape[:-1]), 6 * 2 * components
    )

    return full_results


def blend_ray_features_with_nerf_paper_method(ray_features: jt.ArrayLike) -> jax.Array:
    """Compute one color for each ray, by using the NeRF paper's rendering method.

    We split the (near_point, far_point) interval into N regularly-sized bins, then sample one point, $t_i$, uniformly
    inside each bin. We then use alpha-rendering with this formula:

    $C(r) = \\sum_{i=1}^{N}{c(t_i)T(t_i)(1-\\exp(-\\sigma(t_i)\\delta(t_i)))}$

    where $T(t_i) = \\exp{-\\sum_{j=1}^{i-1}{\\sigma(t_j)}}$ is the weight of each color (goes down exponentially with
    the sum of densities of the previous intervals) and $\\delta(T-i)$ is the distance between the previous point
    $t_{i-1}$ and $t_i$.

    @param ray_features Coordinates, color, and transparency sampled along rays. Shape: (..., pos_per_ray, 7).
    Last axis: x, y, z, R, G, B, sigma.
    @return One color per ray. Shape: (..., 3). Last axis: R, G, B.
    """
    ray_features = jnp.array(ray_features)

    densities = ray_features[..., :-1, 6:7]
    colors = ray_features[..., :-1, 3:6]
    positions = ray_features[..., :, 0:3]
    interval_lengths = jnp.linalg.norm(
        positions[..., 1:, :] - positions[..., :-1, :], axis=-1, keepdims=True
    )

    density_distance_products = densities * interval_lengths
    accumulated_densities = jnp.cumulative_sum(
        density_distance_products, axis=-2, include_initial=True
    )[..., :-1, :]
    remaining_transmittance = jnp.exp(-accumulated_densities)

    alpha_values = 1 - jnp.exp(-densities * interval_lengths)
    individual_rendering_components = remaining_transmittance * alpha_values * colors

    final_rendered_colors = jnp.sum(
        individual_rendering_components, axis=-2, keepdims=False
    )
    return final_rendered_colors
