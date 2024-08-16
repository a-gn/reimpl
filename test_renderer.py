import jax.numpy as jnp

from reimpl_a_gn.voxels.render import (
    sample_rays_for_image_render,
    sample_positions_along_rays,
    blend_ray_features,
)


camera_origin = jnp.array([0.0, 0.0, 0.0, 1.0])
camera_params = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
print(sample_rays_for_image_render(camera_origin, camera_params, 10, 10))
