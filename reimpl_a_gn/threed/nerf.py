import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt


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
    if sampling_positions.shape != densities.shape or sampling_positions.ndim != 2:
        raise ValueError(
            "densities and sampling positions must have the same shape and exactly two axes each"
            f", but we got {densities.shape} and {sampling_positions.shape}"
        )
    if jnp.any(sampling_positions[:, 1:] < sampling_positions[:, :-1]).item():
        raise ValueError(
            "sampling positions must be strictly increasing along the last axis, but they aren't"
        )
    if jnp.any(densities < 0).item():
        raise ValueError(
            f"densities must be positive or zero, but their minimum is {densities.min().item()}"
        )

    num_rays, num_samples = densities.shape
    num_intervals = num_samples - 1
    unnormalized_pdf_values = jnp.zeros(
        [num_rays, num_intervals], dtype=float, device=sampling_positions.device
    )
    previous_accumulated_weighted_densities = jnp.zeros(num_rays, dtype=float)
    # compute all probability density values after the (near_distance, first sampling position) interval
    for interval_index in range(num_intervals):
        distance_to_next_sample = (
            sampling_positions[:, interval_index + 1]
            - sampling_positions[:, interval_index]
        )
        current_density = densities[:, interval_index]
        unnormalized_pdf_values = unnormalized_pdf_values.at[:, interval_index].set(
            # term that removes importance from our current position according to previous densities
            jnp.exp(-previous_accumulated_weighted_densities)
            # term that adds influence to our current position according to its own density
            * (1 - jnp.exp(-current_density * distance_to_next_sample))
        )
        previous_accumulated_weighted_densities += (
            current_density * distance_to_next_sample
        )
    return unnormalized_pdf_values / unnormalized_pdf_values.sum(axis=1)
