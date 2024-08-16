"""Re-implementation of Behind the Scenes (CVPR 23)."""

from typing import Callable

import jax.numpy as jnp
import jax.random as random
from jax import lax

from ..common.modules import TrainableModule, UpConv2D, maxpool2d, relu


class UNet2DConvBlock(TrainableModule):
    """One convolution + activation + normalization block in a U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        strides: tuple[int, int] = (1, 1),
        activation: Callable[[jnp.ndarray], jnp.ndarray] = relu,
        init_key: jnp.ndarray = random.key(0),
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = random.normal(
            init_key,
            (out_channels, in_channels, kernel_size[0], kernel_size[1]),
        )
        self.strides = strides
        self.activation = activation

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        x = lax.conv(image, self.kernel, window_strides=self.strides, padding="SAME")
        x = self.activation(x)
        return x

    def get_trainable_parameters(self) -> list[jnp.ndarray]:
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
        init_key: jnp.ndarray = random.key(0),
    ):
        self.levels = [
            UNet2DConvBlock(
                in_chan,
                out_channels,
                kernel_size,
                activation=activation,
                init_key=init_key,
            )
            for in_chan in [in_channels] + [out_channels] * (block_count - 1)
        ]

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        """Compute the output of this level, with and without max-pooling."""
        x = image
        for level in self.levels:
            x = level(x)
        return x

    def get_trainable_parameters(self) -> list[jnp.ndarray]:
        return [p for level in self.levels for p in level.get_trainable_parameters()]


class UNet2DDecoderLevel(TrainableModule):
    """One level of the decoder in a U-Net.

    Takes a skip-input and an up-input that gets upsampled to the skip-input's spatial size. Concatenate the result and
    runs it through multiple convolution-activation blocks.

    """

    def __init__(
        self,
        skip_in_channels: int,
        up_in_channels: int,
        out_channels: int,
        conv_count_after_merge: int = 2,
    ):
        self.skip_in_channels = skip_in_channels
        self.up_in_channels = up_in_channels
        self.out_channels = out_channels

        self.up_conv = UpConv2D(
            up_in_channels, up_in_channels, kernel_size=(2, 2), dilation=(2, 2)
        )
        self.merge_conv = UNet2DConvBlock(
            skip_in_channels + up_in_channels, out_channels, kernel_size=(1, 1)
        )
        self.later_convs = [
            UNet2DConvBlock(out_channels, out_channels, kernel_size=(3, 3))
            for _ in range(conv_count_after_merge)
        ]

    def __call__(self, skip_input: jnp.ndarray, up_input: jnp.ndarray) -> jnp.ndarray:
        upsampled = self.up_conv(up_input)
        merged = jnp.concatenate([skip_input, upsampled], axis=1)
        x = self.merge_conv(merged)
        for conv in self.later_convs:
            x = conv(x)
        return x

    def get_trainable_parameters(self) -> list[jnp.ndarray]:
        return (
            [p for conv in self.later_convs for p in conv.get_trainable_parameters()]
            + self.up_conv.get_trainable_parameters()
            + self.merge_conv.get_trainable_parameters()
        )


class UNet2D(TrainableModule):
    """2D U-Net image segmentation network."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.in_conv = UNet2DConvBlock(in_channels, hidden_channels)

        self.encoder_levels = [
            UNet2DEncoderLevel(hidden_channels, hidden_channels * 2),
            UNet2DEncoderLevel(hidden_channels * 2, hidden_channels * 4),
            UNet2DEncoderLevel(hidden_channels * 4, hidden_channels * 8),
            UNet2DEncoderLevel(hidden_channels * 8, hidden_channels * 16),
        ]

        self.bridge = [
            UNet2DConvBlock(
                hidden_channels * 16, hidden_channels * 16, kernel_size=(3, 3)
            ),
            UNet2DConvBlock(
                hidden_channels * 16, hidden_channels * 16, kernel_size=(3, 3)
            ),
        ]

        self.decoder_levels = [
            UNet2DDecoderLevel(
                hidden_channels * 16, hidden_channels * 16, hidden_channels * 8
            ),
            UNet2DDecoderLevel(
                hidden_channels * 8, hidden_channels * 8, hidden_channels * 4
            ),
            UNet2DDecoderLevel(
                hidden_channels * 4, hidden_channels * 4, hidden_channels * 2
            ),
            UNet2DDecoderLevel(
                hidden_channels * 2, hidden_channels * 2, hidden_channels
            ),
        ]

        self.out_conv = UNet2DConvBlock(
            hidden_channels, out_channels, kernel_size=(1, 1)
        )

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        """Compute the output of the U-Net."""

        x = self.in_conv(image)
        encoder_outputs: list[jnp.ndarray] = []
        for level in self.encoder_levels:
            x = level(x)
            encoder_outputs.append(x)
            x = maxpool2d(x)
        for bridge_part in self.bridge:
            x = bridge_part(x)
        for level, encoder_output in zip(
            self.decoder_levels, reversed(encoder_outputs)
        ):
            x = level(encoder_output, x)
        x = self.out_conv(x)

        return x

    def get_trainable_parameters(self) -> list[jnp.ndarray]:
        return (
            self.in_conv.get_trainable_parameters()
            + [
                p
                for level in self.encoder_levels
                for p in level.get_trainable_parameters()
            ]
            + [
                p
                for level in self.decoder_levels
                for p in level.get_trainable_parameters()
            ]
            + [p for level in self.bridge for p in level.get_trainable_parameters()]
            + self.out_conv.get_trainable_parameters()
        )
