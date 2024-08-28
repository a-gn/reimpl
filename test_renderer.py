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

all_x, all_y = pixel_grid = jnp.meshgrid(jnp.arange(1, 5, 0.5), jnp.arange(1, 5, 0.5))
all_grid_points = jnp.stack([all_x, all_y], axis=-1).reshape(-1, 2)
fig_2d = plt.figure()
plt.scatter(all_grid_points[:, 0], all_grid_points[:, 1])

all_grid_points_world = camera_params.image_points_to_world(all_grid_points)
all_grid_points_world_non_hom = jnp.stack(
    [
        all_grid_points_world[:, 0] / all_grid_points_world[:, 3],
        all_grid_points_world[:, 1] / all_grid_points_world[:, 3],
        all_grid_points_world[:, 2] / all_grid_points_world[:, 3],
    ],
    axis=-1,
)
fig_3d = plt.figure()
ax = fig_3d.add_subplot(projection="3d")
ax.scatter(
    all_grid_points_world_non_hom[:, 0],
    all_grid_points_world_non_hom[:, 1],
    all_grid_points_world_non_hom[:, 2],
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
