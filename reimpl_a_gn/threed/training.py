import jax
import jax.numpy as jnp
import jax.typing as jt

import reimpl_a_gn.threed.rendering as render

from .camera import CameraParams
from .nerf import FullNeRF


class NeRFTrainer:
    """Stores a NeRF network and trains it."""

    def __init__(self, nerf_module: FullNeRF):
        self.nerf = nerf_module
