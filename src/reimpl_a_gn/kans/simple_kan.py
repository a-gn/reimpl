"""https://arxiv.org/abs/2404.19756"""

import jax
import jax.nn as nn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import jax.typing as jt

from ..common.modules import TrainableModule


def b_spline(d: int, k: int, x: float):
    """Compute the `k`-th B-spline basis function of degree `d` at `x`."""
    raise NotImplementedError()


class KANLayer(TrainableModule):
    """One layer of a dense Kolmogorov-Arnold network."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # there is one B-spline with a coefficient for every (input, output) pair
        self.spline_coefficients = jnp.zeros(
            (self.input_dim, self.output_dim), dtype=jnp.float32
        )

    def __call__(self, x: jt.ArrayLike):
        """Compute the layer's output.

        @param x Vector of length `self.input_dim`.
        @return Vector of length `self.output_dim`.
        """

        outputs = jnp.zeros(self.output_dim, dtype=jnp.float32)
        for in_spline_idx in range(self.input_dim):
            for out_spline_idx in range(self.output_dim):
                outputs[out_spline_idx] += b_spline()
