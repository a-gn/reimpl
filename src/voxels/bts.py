"""Re-implementation of Behind the Scenes (CVPR 23)."""

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.numpy import ndarray


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(x, 0)


def maxpool2d(
    x: jnp.ndarray,
    window_size: tuple[int, int] = (2, 2),
    strides: tuple[int, int] = (2, 2),
) -> jnp.ndarray:
    """Max pooling with a 2x2 window."""
    return lax.reduce_window(
        x,
        init_value=jnp.zeros_like(x),
        computation=jnp.max,
        window_dimensions=window_size,
        window_strides=strides,
        padding="SAME",
    )


class TrainableModule(ABC):
    """Module with parameters that can be trained."""

    @abstractmethod
    def get_trainable_parameters(self) -> list[jnp.ndarray]:
        """Get all of this module's trainable parameters."""
        pass


class UNet2DConvBlock(TrainableModule):
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

    def get_trainable_parameters(self) -> list[ndarray]:
        return [self.kernel]


class UNet2DEncoderLevel(TrainableModule):
    """One level of the encoder in a U-Net.

    Goes through multiple convolution-activation blocks.
    Output always has the same spatial size as the input.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_count: int = 3,
        kernel_size: tuple[int, int] = (3, 3),
        activation: Callable[[jnp.ndarray], jnp.ndarray] = relu,
        init_key: random.KeyArray = random.key(0),
    ):
        self.levels = [
            UNet2DConvBlock(in_chan, out_channels, kernel_size, activation, init_key)
            for in_chan in [in_channels] + [out_channels] * (block_count - 1)
        ]

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        """Compute the output of this level, with and without max-pooling."""
        x = image
        for level in self.levels:
            x = level(x)
        return x

    def get_trainable_parameters(self) -> list[ndarray]:
        return [p for level in self.levels for p in level.get_trainable_parameters()]
