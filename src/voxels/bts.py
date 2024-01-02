"""Re-implementation of Behind the Scenes (CVPR 23)."""

import jax.numpy as jnp
import jax.random as random


class FeatureGrid:
    """3D grid of features, positioned in space, that can be sampled from.

    There are two coordinate systems: grid indices and spatial coordinates.
    This object can e.g. sample at coordinates (x, y, z) by converting those to grid coordinates.

    """

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        corner_min: tuple[float, float, float],
        corner_max: tuple[float, float, float],
        random_init_seed: int = 0,
    ):
        """Initialize the feature grid.

        :param width: Number of columns in the grid. Corresponds to X spatial dimension.
        :param height: Number of rows in the grid. Corresponds to Y spatial dimension.
        :param depth: Number of channels in the grid. Corresponds to Z spatial dimension.
        :param corner_min: x, y, z coordinates of the corner of the grid corresponding to minimum indices.
        :param corner_max: x, y, z coordinates of the corner of the grid corresponding to maximum indices.

        """
        self.feature_grid = random.normal(
            random.key(random_init_seed), (height, width, depth), dtype=jnp.float32
        )
        self.spatial_to_grid_projection = jnp.array(
            [
                [width / (corner_max[0] - corner_min[0]), 0, 0],
                [0, height / (corner_max[1] - corner_min[1]), 0],
                [0, 0, depth / (corner_max[2] - corner_min[2])],
            ]
        )

    def sample(self, points: jnp.ndarray) -> jnp.ndarray:
        """Sample features from the grid.

        :param points: Points to sample features at. Shape: (N, 3).
        :return: Features at the given points. Shape: (N, D).

        """
