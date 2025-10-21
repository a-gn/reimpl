from abc import ABC, abstractmethod
from dataclasses import dataclass

from jax import Array
from jax.random import split


@dataclass
class NeRFTrainingSamples:
    """Ray coordinates, ground truth colors, and other information for a NeRF training batch."""

    rays: Array
    """Ray coordinates in world frame. Dimensions: (batch, 6). Last axis: (x, y, z, dx, dy, dz). DType: float."""
    colors: Array
    """Ground truth colors for all rays. Shape: (batch, 3). Last axis: RGB. DType: float. Range: [0, 1]."""
    extrinsic_matrices: Array
    """Extrinsic matrices (world to camera) for cameras corresponding to all rays. Shape: (batch, 4, 4)."""
    dataset_info: list[dict[str, str]] | None
    """Source-specific information that can be added to e.g. find the data the samples come from."""


class RayAndColorDataset(ABC):
    """Object that generates ray coordinates and pixel colors from a training dataset."""

    def __init__(self, rng_key: Array, batch_size: int):
        """
        @param rng_key JAX random number generation key to use. We will create splits of it internally, repeatedly, to
        generate successive samples.
        @param batch_size Number of samples to choose for each yielded batch.
        """
        self.rng_key = rng_key
        self.batch_size = batch_size

    @abstractmethod
    def _get_batch_of_rays(self, rng_key: Array) -> NeRFTrainingSamples:
        """Choose a batch of rays and load them with associated data.

        @param rng_key JAX random number generation key to use. Will not be split or modified. Make sure to pass
        different keys for every call if you want different training samples.
        @return A batch of `batch_size` training samples.
        """
        ...

    def __iter__(self):
        """Generate new training samples indefinitely."""

        while True:
            subkey, self.rng_key = split(self.rng_key, 2)
            yield self._get_batch_of_rays(subkey)
            del subkey
