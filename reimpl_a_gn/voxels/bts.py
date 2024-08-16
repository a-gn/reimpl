"""NeRF rendered for BTS."""

from logging import getLogger
from typing import Protocol

import jax
import jax.numpy as jnp
import jax.typing as jt

from ..common.modules import TrainableModule

_log = getLogger(__name__)


class BTSPrediction:
    """Wraps an image and its associated feature map to render new views of a scene."""

    def __init__(self, image: jax.Array, feature_grid: jax.Array):
        image = jnp.array(image)
        feature_grid = jnp.array(feature_grid)
        assert image.shape[0] == feature_grid.shape[0]
        assert image.shape[2] == feature_grid.shape[2]
        assert image.shape[3] == feature_grid.shape[3]

        self.image = image
        self.feature_grid = feature_grid

    def sample_features(self, rays: jax.Array):
        """Sample the feature grid along multiple rays.

        @param rays Ray coordinates: point and direction.
        Shape: [num_rays, 6] with axis 2 containing: x, y, z, dx, dy, dz.

        """
        raise NotImplementedError()

    def compute_new_views(self, camera_params: jax.Array):
        """Compute images of the scene from given points of view.

        @param camera_params Parameters of the new views.
        @return Image array with shape [view_count, 3, height, width]

        """
        raise NotImplementedError()


class BTSBackbone(TrainableModule, Protocol):

    def __call__(self, image: jax.Array) -> jax.Array:
        ...


class BTSModel(TrainableModule):
    """Model that predicts a feature grid from an image and decodes densities along rays."""

    def __init__(self, backbone: BTSBackbone, feature_depth: int):
        """
        @param backbone Module that predicts a feature map from an image.
        """
        self.backbone = backbone
        self.feature_depth = feature_depth

    def predict_feature_grid(self, image: jax.Array) -> jax.Array:
        """Predict the pixel-aligned feature map described in BTS.

        @return An array of predicted features. Shape: (image_height, image_width, feature_depth).
        """
        image = jnp.array(image)
        backbone_out = self.backbone(image)
        expected_shape = (
            image.shape[0],  # batch
            self.feature_depth,
            image.shape[2],  # height
            image.shape[3],  # width
        )
        if backbone_out.shape != expected_shape:
            raise ValueError(
                f"backbone output shape {backbone_out.shape} does not match expected shape {expected_shape}"
            )
        backbone_out = backbone_out.reshape(
            (image.shape[0], image.shape[2], image.shape[3], self.feature_depth)
        )
        return backbone_out


class BTSDensityDecoder(TrainableModule):
    """MLP that predicts density from features predicted by a BTS model.

    Store a feature grid, receive coordinates, interpolate features at them, pass them through a dense linear layer.
    """

    def __init__(self, feature_grid: jax.Array, world_to_camera: jax.Array):
        """
        @param feature_grid Predicted features, each one aligned with input image pixels.
        Shape: (height, width, feature_depth).
        @param world_to_camera Matrix that transforms world 3D coordinates into image/grid 2D coordinates.
        Shape: (3, 4).
        """
        assert len(feature_grid.shape) == 3
        self.feature_grid = feature_grid
        self.feature_depth = self.feature_grid.shape[-1]
        assert world_to_camera.shape == (3, 4)
        self.world_to_camera = world_to_camera

    def __call__(self, coordinates: jax.Array):
        """Compute density at given points from the stored feature grid.

        @param coordinates 3D world points at which to predict density; shape: (num_points, 3).
        """
        assert coordinates.shape[-1] == 3
        flat_coords_world = jnp.reshape(coordinates, (-1, 3))
        # turn into homogeneous coordinates
        flat_coords_world_hom = jnp.stack([flat_coords_world, jnp.ones((1,))], axis=-1)
        flat_coords_image = self.world_to_camera @ flat_coords_world_hom
        # convert back
        flat_coords_image = flat_coords_image[:, :2] / flat_coords_image[:, 2]
        # interpolate features at the exact positions given
        flat_interpolated_features = jnp.zeros(
            (flat_coords_world.size // 3, self.feature_depth), dtype=jnp.float32
        )
        for ix, iy in flat_coords_image:
            x_min = jnp.floor(ix)
            y_min = jnp.floor(iy)
            features_around_us = (
                self.feature_grid[y_min, x_min],
                self.feature_grid[y_min, x_min + 1],
                self.feature_grid[y_min + 1, x_min],
                self.feature_grid[y_min + 1, x_min + 1],
            )
            flat_interpolated_features = features_around_us[]
