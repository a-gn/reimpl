import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt

from .rendering import CameraParams, from_homogeneous, norm_eucl_3d


class CoarseMLP(nn.Module):
    """The coarse network that lets us choose interesting sampling positions in a NeRF."""

    mid_features: tuple[int, ...]
    out_features: int

    @nn.compact
    def __call__(
        self,
        rays: jt.ArrayLike,
    ):
        """Predict features for the given rays.

        @param rays Origins and direction unit vectors for all rays.
            Shape: `(number_of_rays, 6)`. Last axis: x, y, z, dx, dy, dz.
        @return Predicted features. Shapes: `(number_of_rays, self.out_features)`.

        """
        rays = jnp.array(rays)
        for out_feat_count in self.mid_features:
            rays = nn.Dense(out_feat_count)(rays)
            rays = nn.relu(rays)
        return nn.Dense(self.out_features)(rays)


class FineMLP(nn.Module):
    """The fine network that gives us more precise details in a NeRF."""

    mid_features: tuple[int, ...]
    out_features: int

    @nn.compact
    def __call__(
        self,
        rays: jt.ArrayLike,
    ):
        """Predict features for the given rays.

        @param rays Origins and direction unit vectors for all rays.
            Shape: `(number_of_rays, 6)`. Last axis: x, y, z, dx, dy, dz.
        @return Predicted features. Shapes: `(number_of_rays, self.out_features)`.

        """
        rays = jnp.array(rays)
        for out_feat_count in self.mid_features:
            rays = nn.Dense(out_feat_count)(rays)
            rays = nn.relu(rays)
        return nn.Dense(self.out_features)(rays)


class FullNeRF(nn.Module):
    """A full NeRF network as described in the paper."""

    coarse_mid_features: tuple[int, ...]
    fine_mid_features: int
    prng_key: jt.ArrayLike

    @nn.compact
    def __call__(
        self,
        points: jt.ArrayLike,
    ):
        points = jnp.array(points)
        coarse_features = CoarseMLP(self.coarse_mid_features, self.fine_mid_features)(
            points
        )
        fine_sampling_distributions = self.compute_fine_sampling_distribution(
            coarse_features
        )
        fine_sampling_points = jax.random.uniform(
            self.prng_key,
        )


def compute_rays_in_world_frame(
    camera: CameraParams, x_range: tuple[int, int], y_range: tuple[int, int]
):
    """Compute the origin and direction of rays from the camera origin to pixels in the image.

    @return ray directions and origins. Shape: (ray_count, 6). Second axis: (x, y, z, dx, dy, dz).

    """

    # compute rays in world frame
    ray_targets_x_image, ray_targets_y_image = jnp.meshgrid(
        jnp.arange(*x_range), jnp.arange(*y_range)
    )
    ray_targets_image = jnp.stack(
        [ray_targets_x_image, ray_targets_y_image], axis=-1
    ).reshape(-1, 2)
    ray_directions_world_homo = camera.image_to_world(ray_targets_image)
    # origin of rays is origin of camera
    ray_origins_world_homo = (
        jnp.array([[0.0, 0.0, 0.0, 1.0]]) @ camera.camera_to_world.T
    )
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


@jax.jit
def compute_fine_sampling_distribution(
    densities: jt.ArrayLike,
    sampling_positions: jt.ArrayLike,
):
    """Compute the distributions from which to sample points to pass through the fine MLP, for a single ray.

    We compute weights for each sampling position along the ray, adjusting the probability down according to densities.

    This is meant to sample from the more computationally expensive MLP using the results of the coarse MLP.
    See the NeRF paper for details.

    @param densities Density values predicted by the coarse MLP. Shape: (num_rays, num_samples,).
    @param sampling_positions Positions along the ray at which `densities` were predicted. Same shape as `densities`.
    Must be strictly increasing along the last axis.
    @return A piecewise-uniform probability distribution represented as a `(num_rays, num_samples + 1,)`-shaped array of
    probability values. Item with index `n` is the distribution's value in the `n`th interval.
    """
    densities = jnp.array(densities)
    sampling_positions = jnp.array(sampling_positions)

    result = jnp.zeros((densities.shape[0], densities.shape[1] - 1), dtype=float)

    cumulative_transmittance = jnp.cumulative_sum(
        -densities[:, :-1] * (sampling_positions[:, 1:] - sampling_positions[:, :-1]),
        axis=1,
        include_initial=True,
    )
    for interval_index in range(0, sampling_positions.shape[1] - 1):
        result = result.at[:, interval_index].set(
            jnp.exp(cumulative_transmittance[:, interval_index])
            * (
                1
                - jnp.exp(
                    -densities[:, interval_index]
                    * (
                        sampling_positions[:, interval_index + 1]
                        - sampling_positions[:, interval_index]
                    )
                )
            )
        )
    return result


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
        result = result.at[:, position_index, :].set(sampled_positions)
    return result


def sample_coarse_mlp_inputs(
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
    norm_ray_directions = ray_directions / norm_eucl_3d(
        ray_directions, homogeneous=False, keepdims=True
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


def blend_ray_features_with_nerf_paper_method(ray_features: jax.Array) -> jax.Array:
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
    origin_center_distances = norm_eucl_3d(interval_centers, homogeneous=False)
    origin_sample_distances = norm_eucl_3d(ray_features[..., :, :3], homogeneous=False)
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

    @param points_and_directions Rays to encode. Shape: (..., 6). Last axis: x, y, z, dx, dy, dz.
    @return Positional encoding of the points. Shape: (..., 6, 2 * components).
    """

    points_and_directions = jnp.array(points_and_directions)
    if points_and_directions.ndim < 2 or points_and_directions.shape[-1] != 6:
        raise ValueError(
            f"expected input shape (..., 6), got shape {points_and_directions.shape}"
        )

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
