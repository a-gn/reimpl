"""Loads one image and pose from the flower datasets, renders one color per pixel.

Will only output noise as of now since I don't train anything.
"""

import matplotlib.pyplot as plt
from flax.nnx import Rngs
from jax.random import key, split

from reimpl_a_gn.dataset.synthetic_nerf_dataset import get_flower_dataset
from reimpl_a_gn.threed.nerf import CoarseMLP, FineMLP
from reimpl_a_gn.threed.rendering import render_image

rng_key = key(seed=7)

flower_dataset = get_flower_dataset()

first_image = flower_dataset.images[0]
assert first_image.ndim == 3  # height, width, RGB
first_camera = flower_dataset.cameras[0]

rng_key, rng_subkey = split(rng_key)
coarse_network = CoarseMLP(
    input_features=6 * 2 * 2,
    mid_features=(64, 64),
    out_features=4,
    rngs=Rngs(rng_subkey),
)
del rng_subkey

rng_key, rng_subkey = split(rng_key)
fine_network = FineMLP(6 * 2 * 2, (64, 64), 4, rngs=Rngs(rng_subkey))
del rng_subkey

colors = render_image(
    image=first_image,
    camera=first_camera,
    rng_key=rng_key,
    coarse_network=coarse_network,
    fine_network=fine_network,
    ray_batch_size=2**16,
)
plt.imshow(colors)
plt.show()
