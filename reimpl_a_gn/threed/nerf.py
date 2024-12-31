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
        points: jt.ArrayLike,
    ):
        points = jnp.array(points)
        for out_feat_count in self.mid_features:
            points = nn.Dense(out_feat_count)(points)
            points = nn.relu(points)
        return nn.Dense(self.out_features)(points)


class FineMLP(nn.Module):
    """The fine network that gives us more precise details in a NeRF."""

    mid_features: tuple[int, ...]
    out_features: int

    @nn.compact
    def __call__(
        self,
        points: jt.ArrayLike,
    ):
        points = jnp.array(points)
        for out_feat_count in self.mid_features:
            points = nn.Dense(out_feat_count)(points)
            points = nn.relu(points)
        return nn.Dense(self.out_features)(points)


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

    @staticmethod
    def compute_fine_sampling_distribution(coarse_features: jt.ArrayLike):
        """Compute the distributions from which to sample points to pass through the fine MLP.

        This is meant to sample from the more computationally expensive MLP using the results of the coarse MLP.
        See the NeRF paper for details.

        @param coarse_features Features predicted by the coarse MLP. Shape: (..., 4). Last axis: red, green, blue,
        density (sigma).
        """
        coarse_features = jnp.array(coarse_features)
        assert len(coarse_features) >= 2
        assert coarse_features.shape[-1] == 4
        flat_coarse_features = coarse_features.reshape(-1, 4)
        # compute rendering weights (seeing color computation as alpha-rendering, those are the weights)
        un_normalized_weights = jnp.zeros(())
