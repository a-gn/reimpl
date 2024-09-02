import flax
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
