import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
from .rendering import CameraParams


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
            Shape: `(number_of_rays, 7)`. Last axis: x, y, z, w, dx, dy, dz.
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
            Shape: `(number_of_rays, 7)`. Last axis: x, y, z, w, dx, dy, dz.
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

    @return ray directions and origins. Shape: (ray_count, 8). Second axis: (x, y, z, w, dx, dy, dz, 0).

    """

    # compute rays in world frame
    ray_targets_x_image, ray_targets_y_image = jnp.meshgrid(
        jnp.arange(*x_range), jnp.arange(*y_range)
    )
    ray_targets_image = jnp.stack(
        [ray_targets_x_image, ray_targets_y_image], axis=-1
    ).reshape(-1, 2)
    ray_directions_world = camera.image_to_world(ray_targets_image)
    # origin of rays is origin of camera
    ray_origins_world = jnp.array([[0.0, 0.0, 0.0, 1.0]]) @ camera.camera_to_world.T
    # same origin for all rays, concatenate needs axis 0 to have the same size as directions
    ray_origins_world = jnp.repeat(
        ray_origins_world, axis=0, repeats=ray_directions_world.shape[0]
    )
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
