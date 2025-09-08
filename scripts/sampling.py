from pathlib import Path

import jax.numpy as jnp
import jax.random as rand
import kagglehub
from flax import nnx

from reimpl_a_gn.dataset.synthetic_nerf_dataset import load_synthetic_nerf_dataset
from reimpl_a_gn.threed.nerf import (
    CoarseMLP,
    compute_nerf_positional_encoding,
    sample_coarse_mlp_inputs,
)
from reimpl_a_gn.threed.rendering import sample_rays_towards_pixels

POSITIONAL_ENCODING_COMPONENTS_COARSE = 5
POSITIONAL_ENCODING_COMPONENTS_FINE = 6

RNGS = nnx.Rngs(7)


def get_flower_dataset():
    """Download the dataset if needed, and return the path to the flower subfolder."""
    dataset_path = (
        Path(kagglehub.dataset_download("arenagrenade/llff-dataset-full"))
        / "nerf_llff_data"
        / "flower"
    )
    return load_synthetic_nerf_dataset(dataset_path)


def compute_pixel_coords() -> jnp.ndarray:
    """Compute pixel coordinates for a grid of pixels.

    @return: Pixel (x, y) coordinate array, shape (N, 2).
    """
    pixels_x = jnp.arange(-10, 10)
    pixels_y = jnp.arange(-15, 15)
    pixels_x, pixels_y = jnp.meshgrid(pixels_x, pixels_y)
    pixel_coords = jnp.stack([pixels_x, pixels_y], axis=1).reshape(-1, 2)
    return pixel_coords


camera_0 = get_flower_dataset().cameras[0]
rays = sample_rays_towards_pixels(camera_0, compute_pixel_coords())
prng_key = rand.key(7)

prng_key, coarse_sampling_key = rand.split(prng_key)
coarse_points = sample_coarse_mlp_inputs(rays, 0.01, 0.1, 5, coarse_sampling_key)
# print(coarse_points)
# ax = plt.figure().add_subplot(projection="3d")
# ax.scatter(
#     coarse_points.reshape(-1, 4)[:, 0] / coarse_points.reshape(-1, 4)[:, 3:4],
#     coarse_points.reshape(-1, 4)[:, 1] / coarse_points.reshape(-1, 4)[:, 3:4],
#     coarse_points.reshape(-1, 4)[:, 2] / coarse_points.reshape(-1, 4)[:, 3:4],
# )
# plt.show()
# del coarse_sampling_key


encoded_coarse_points = compute_nerf_positional_encoding(
    coarse_points, POSITIONAL_ENCODING_COMPONENTS_COARSE
)
coarse_mlp = CoarseMLP(
    POSITIONAL_ENCODING_COMPONENTS_COARSE * 2, (32, 32), 4, rngs=RNGS
)
coarse_predictions = coarse_mlp(encoded_coarse_points)
print(coarse_predictions)
