from typing import Any

import jax.numpy as jnp
import jax.random as random
import pytest

from voxels.bts import UNet2DConvBlock, UNet2DEncoderLevel

KEY = random.key(0)


@pytest.mark.parametrize(
    ["in_channels", "out_channels", "kernel_size", "input_image"],
    [
        [
            2,
            1,
            (3, 3),
            jnp.ones((2, 2, 8, 8)),
        ]
    ],
)
def test_pass_data_through_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    input_image: jnp.ndarray,
):
    block = UNet2DConvBlock(in_channels, out_channels, kernel_size, init_key=KEY)
    output = block(input_image)
    assert output.shape == (
        input_image.shape[0],
        out_channels,
        input_image.shape[2],
        input_image.shape[3],
    )


@pytest.mark.parametrize(
    ["in_channels", "out_channels", "other_encoder_block_args", "input_image"],
    [
        [
            2,
            1,
            {},
            jnp.ones((2, 2, 8, 8)),
        ],
    ],
)
def test_pass_data_through_encoder_level(
    in_channels: int,
    out_channels: int,
    other_encoder_block_args: dict[str, Any],
    input_image: jnp.ndarray,
):
    level = UNet2DEncoderLevel(
        in_channels, out_channels, init_key=KEY, **other_encoder_block_args
    )
    output = level(input_image)
    assert output.shape == (
        input_image.shape[0],
        out_channels,
        input_image.shape[2],
        input_image.shape[3],
    )
