"""Re-implementation of Behind the Scenes (CVPR 23)."""

from typing import Callable

import jax.numpy as jnp
import jax.random as random
from jax import lax


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(x, 0)


class UNet2DConvBlock:
    """One convolution + activation + normalization block in a U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        activation: Callable[[jnp.ndarray], jnp.ndarray] = relu,
        init_key: random.KeyArray = random.key(0),
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = random.normal(
            init_key,
            (out_channels, in_channels, kernel_size[0], kernel_size[1]),
        )
        self.activation = activation

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        x = lax.conv(image, self.kernel, window_strides=(1, 1), padding="SAME")
        x = self.activation(x)
        return x
