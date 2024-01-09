from typing import Any

import jax.numpy as jnp
import jax.random as random
import pytest

from voxels.unet import UNet2D, UNet2DConvBlock, UNet2DDecoderLevel, UNet2DEncoderLevel

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


@pytest.mark.parametrize(
    [
        "skip_channels",
        "up_channels",
        "out_channels",
        "other_decoder_block_args",
        "skip_input_image",
        "up_input_image",
    ],
    [
        [
            2,
            3,
            6,
            {},
            jnp.ones((2, 2, 8, 8)),
            jnp.ones((2, 3, 4, 4)),
        ],
    ],
)
def test_pass_data_through_decoder_level(
    skip_channels: int,
    up_channels: int,
    out_channels: int,
    other_decoder_block_args: dict[str, Any],
    skip_input_image: jnp.ndarray,
    up_input_image: jnp.ndarray,
):
    level = UNet2DDecoderLevel(
        skip_channels, up_channels, out_channels, **other_decoder_block_args
    )
    output = level(skip_input_image, up_input_image)
    assert output.shape == (
        skip_input_image.shape[0],
        out_channels,
        skip_input_image.shape[2],
        skip_input_image.shape[3],
    )


@pytest.mark.parametrize(
    [
        "in_channels",
        "hidden_channels",
        "out_channels",
        "other_unet_args",
        "input_image",
    ],
    [
        [
            2,
            4,
            2,
            {},
            jnp.ones((2, 2, 64, 64)),
        ],
    ],
)
def test_pass_data_through_full_unet(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    other_unet_args: dict[str, Any],
    input_image: jnp.ndarray,
):
    net = UNet2D(in_channels, hidden_channels, out_channels, **other_unet_args)
    output = net(input_image)
    assert output.shape == (
        input_image.shape[0],
        out_channels,
        input_image.shape[2],
        input_image.shape[3],
    )
