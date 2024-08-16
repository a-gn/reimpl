import jax.numpy as jnp

from reimpl_a_gn.voxels.render import (
    sample_rays_for_image_render,
    sample_positions_along_rays,
    blend_ray_features,
)


camera_origin = jnp.array([0.0, 0.0, 0.0, 1.0])
camera_params = jnp.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
)
rays = sample_rays_for_image_render(camera_origin, camera_params, 10, 10)
positions_along_rays = sample_positions_along_rays(rays, 0.0, 1.0, 10)
print(positions_along_rays)
