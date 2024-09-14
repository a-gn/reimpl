"""This module wraps the original dataset loader to provide JAX arrays in a format that's easy for us to use."""

from ._original_code import load_llff_data

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jax


@dataclass
class SyntheticNeRFData:
    images: jax.Array
    poses: jax.Array
    bds: jax.Array
    render_poses: jax.Array
    i_test: jax.Array


def load_synthetic_nerf_dataset(
    basedir: Path | str,
    factor=8,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    path_zflat=False,
):
    """Load data from the Synthetic NeRF Dataset as JAX arrays.

    Format is the same as the original loader, besides using JAX arrays instead of NumPy arrays.

    Original loader: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L243
    """
    images, poses, bds, render_poses, i_test = load_llff_data(
        str(basedir),
        factor=factor,
        recenter=recenter,
        bd_factor=bd_factor,
        spherify=spherify,
        path_zflat=path_zflat,
    )
    return SyntheticNeRFData(
        *(jnp.array(x) for x in (images, poses, bds, render_poses, i_test))
    )
