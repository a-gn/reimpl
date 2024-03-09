"""NeRF rendered for BTS."""

from logging import getLogger
from typing import Callable

import jax
import jax.numpy as jnp

from .modules import TrainableModule
from .unet import UNet2D

_log = getLogger(__name__)


class NeRFRenderer:
    """Composes images from ray features."""

    pass


class BTSPrediction:
    """Wraps an image and its associated feature map to render new views of a scene."""

    def __init__(self, image: jnp.ndarray, feature_grid: jnp.ndarray):
        assert image.shape[0] == feature_grid.shape[0]
        assert image.shape[2] == feature_grid.shape[2]
        assert image.shape[3] == feature_grid.shape[3]

        self.image = image
        self.feature_grid = feature_grid

    def sample_features(self, rays: jnp.ndarray):
        """Sample the feature grid along multiple rays.

        @param rays Ray coordinates: point and direction.
        Shape: [num_rays, 6] with axis 2 containing: x, y, z, dx, dy, dz.

        """
        raise NotImplementedError()

    def compute_new_views(self, camera_params: jnp.ndarray):
        """Compute images of the scene from given points of view.

        @param camera_params Parameters of the new views.
        @return Image array with shape [view_count, 3, height, width]

        """
        raise NotImplementedError()


class BTSModel(TrainableModule):
    """Model that predicts a feature grid from an image."""

    def __init__(self, backbone: Callable[[jnp.ndarray], jnp.ndarray]):
        """
        @param backbone Module that predicts a feature map from an image.
        """
        self.backbone = backbone

    def __call__(self, image: jnp.ndarray):
        backbone_out = self.backbone(image)
