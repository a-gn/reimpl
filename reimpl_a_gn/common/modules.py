from typing import Protocol

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random


def relu(x: jax.Array) -> jax.Array:
    return jnp.maximum(x, 0)


def maxpool2d(
    x: jax.Array,
    window_size: tuple[int, int] = (2, 2),
    strides: tuple[int, int] = (2, 2),
) -> jax.Array:
    """Max pooling."""
    return lax.reduce_window(
        x,
        init_value=0.0,
        computation=lax.max,
        window_dimensions=(1, 1, window_size[0], window_size[1]),
        window_strides=(1, 1, strides[0], strides[1]),
        padding="SAME",
    )


class TrainableModule(Protocol):
    """Module with parameters that can be trained."""

    def get_trainable_parameters(self) -> list[jax.Array]:
        """Get all of this module's trainable parameters."""
        ...


class UpConv2D(TrainableModule):
    """Upsample a 2D image with a transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int] = (2, 2),
        padding: tuple[tuple[int, int], tuple[int, int]] = (
            (1, 1),
            (1, 1),
        ),
        strides: tuple[int, int] = (1, 1),
        init_key: jax.Array = random.key(0),
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.padding = padding
        self.strides = strides

        self.kernel = random.normal(
            init_key,
            (out_channels, in_channels, kernel_size[0], kernel_size[1]),
        )

    def __call__(self, image: jax.Array) -> jax.Array:
        x = lax.conv_general_dilated(
            jnp.array(image),
            self.kernel,
            window_strides=self.strides,
            lhs_dilation=self.dilation,
            padding=self.padding,
        )
        return x

    def get_trainable_parameters(self) -> list[jax.Array]:
        return [self.kernel]


class Dense(TrainableModule):
    """A dense linear layer."""

    def __init__(self, input_dim: int, output_dim: int, with_bias: bool):
        self.main_params = jnp.zeros((output_dim, input_dim), dtype=jnp.float32)
        self.bias_params: jax.Array | None = (
            jnp.zeros((output_dim,), dtype=jnp.float32) if with_bias else None
        )

    def __call__(self, input: jax.Array):
        result = self.main_params @ input
        if self.bias_params is not None:
            result = result + self.bias_params
        return result

    def get_trainable_parameters(self) -> list[jax.Array]:
        result = [self.main_params]
        if self.bias_params is not None:
            result.append(self.bias_params)
        return result
