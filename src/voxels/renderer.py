"""NeRF rendered for BTS."""

from abc import abstractmethod
from logging import getLogger
from typing import Protocol

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


class BTSBackbone(TrainableModule, Protocol):
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        ...


class BTSModel(TrainableModule):
    """Model that predicts a feature grid from an image."""

    def __init__(self, backbone: TrainableModule, feature_depth: int):
        """
        @param backbone Module that predicts a feature map from an image.
        """
        self.backbone = backbone
        self.feature_depth = feature_depth

    def __call__(self, image: jnp.ndarray) -> BTSPrediction:
        backbone_out = self.backbone(image)
        expected_shape = (
            image.shape[0],
            self.feature_depth,
            image.shape[2],
            image.shape[3],
        )
        if backbone_out.shape != expected_shape:
            raise ValueError(
                f"backbone output shape {backbone_out.shape} does not match expected shape {expected_shape}"
            )
        return BTSPrediction(image, backbone_out)
