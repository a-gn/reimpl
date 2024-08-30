import matplotlib.pyplot as plt
import jax.numpy as jnp

from reimpl_a_gn.voxels.render import (
    sample_rays_towards_all_pixels,
    sample_positions_along_rays,
    blend_ray_features,
    CameraParams,
)


# camera_params = CameraParams(
#     jnp.array([
#         [2.0, 0.0, 0.0, -2.0],
#         [0.0, 1.0, 0.0, 1.0],
#         [0.0, 0.0, 1.0, 3.0],
#     ]),
#     4.0,
#     5.0,
#     5.0,
# )
camera_params = CameraParams(
    jnp.array([
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]),
    2.0,
)

print(camera_params.image_points_to_world(jnp.array([[0.0, 0.0]])))

all_x, all_y = pixel_grid = jnp.meshgrid(jnp.arange(-2, 3, 0.7), jnp.arange(-5, 5, 0.5))
all_grid_points = jnp.stack([all_x, all_y], axis=-1).reshape(-1, 2)

all_rays_towards_grid_points = camera_params.image_points_to_world(all_grid_points)
all_rays_towards_grid_points = jnp.concatenate(
    [
        jnp.zeros([all_rays_towards_grid_points.shape[0], 3]),
        all_rays_towards_grid_points[:, :3],
    ],
    1,
)
positions_along_rays = sample_positions_along_rays(
    all_rays_towards_grid_points, 0.1, 1.1, 4
)
print(positions_along_rays)

fig_rays = plt.figure()
ax_rays = fig_rays.add_subplot(projection="3d")
ax_rays.text(1, 0, 0, "x-axis", "x")
ax_rays.text(0, 1, 0, "y-axis", "y")
ax_rays.text(0, 0, 1, "z-axis", "z")
flat_positions = positions_along_rays.reshape(-1, 3)
ax_rays.scatter(
    flat_positions[:, 0],
    flat_positions[:, 1],
    flat_positions[:, 2],
)

grid_points_world = camera_params.image_points_to_world(all_grid_points)
ax_rays.text(1, 0, 0, "x-axis", "x")
ax_rays.text(0, 1, 0, "y-axis", "y")
ax_rays.text(0, 0, 1, "z-axis", "z")
flat_positions = positions_along_rays.reshape(-1, 3)
ax_rays.scatter(
    grid_points_world[:, 0], grid_points_world[:, 1], grid_points_world[:, 2], "red"
)
plt.show()

# rays = sample_rays_towards_all_pixels(camera_params, 10, 10)
# positions_along_rays = sample_positions_along_rays(rays, 0.0, 1.0, 5)
# print(positions_along_rays)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# flat_positions = positions_along_rays.reshape(-1, 3)
# ax.scatter(
#     flat_positions[:, 0],
#     flat_positions[:, 1],
#     flat_positions[:, 2],
# )
# # fig = plt.figure()
# # ax = fig.add_subplot(projection="3d")
# # flat_positions = rays.reshape(-1, 6)[:, :3]
# # ax.scatter(
# #     flat_positions[:, 0],
# #     flat_positions[:, 1],
# #     flat_positions[:, 2],
# # )
# plt.show()
