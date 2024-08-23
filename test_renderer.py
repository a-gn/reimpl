import matplotlib.pyplot as plt
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
rays = sample_rays_for_image_render(camera_origin, camera_params, 1, 1)
positions_along_rays = sample_positions_along_rays(rays, 0.0, 1.0, 5)
print(positions_along_rays)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# flat_positions = positions_along_rays.reshape(-1, 3)
# ax.scatter(
#     flat_positions[:, 0],
#     flat_positions[:, 1],
#     flat_positions[:, 2],
# )
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
flat_positions = rays.reshape(-1, 6)[:, :3]
ax.scatter(
    flat_positions[:, 0],
    flat_positions[:, 1],
    flat_positions[:, 2],
)
plt.show()
