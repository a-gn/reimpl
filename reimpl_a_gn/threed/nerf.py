import flax
import flax.nnx
import jax
import jax.numpy as jnp
import jax.typing as jt


class CoarseMLP(flax.nnx.Module):
    """The coarse network that lets us choose interesting sampling positions in a NeRF."""

    def __init__(self):
        self.mlp_layers = flax.nnx.List

    def __call__(self, points: jt.ArrayLike):
        pass


class FineMLP(flax.nnx.Module):
    """The fine network that gives us more precise details in a NeRF."""

    def __init__(self):
        pass

    def __call__(self, points: jt.ArrayLike):
        pass
