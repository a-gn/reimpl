import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from reimpl_a_gn.threed.nerf import (
    sample_rays_towards_all_pixels,
    sample_regular_positions_along_rays,
    sample_nerf_rendering_positions_along_rays,
    CameraParams,
    compute_nerf_positional_encoding,
)

print(compute_nerf_positional_encoding(jnp.array([[2, 3, 4.5, 0.5, 1.0, 0.3]]), 4))


# # camera_params = CameraParams(
# #     jnp.array([
# #         [2.0, 0.0, 0.0, -2.0],
# #         [0.0, 1.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0, 3.0],
# #     ]),
# #     4.0,
# #     5.0,
# #     5.0,
# # )
# camera_params = CameraParams(
#     jnp.array([
#         [4.0, 0.0, 0.0, 0.0],
#         [0.0, 4.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#     ]),
#     2.0,
# )

# print(camera_params.image_points_to_world(jnp.array([[0.0, 0.0]])))

# all_x, all_y = jnp.meshgrid(jnp.arange(-2, 3, 0.7), jnp.arange(-5, 5, 0.5))
# all_grid_points = jnp.stack([all_x, all_y], axis=-1).reshape(-1, 2)

# all_grid_points_camera_frame = camera_params.image_points_to_world(all_grid_points)
# all_rays_towards_grid_points = jnp.concatenate(
#     [
#         jnp.zeros([all_grid_points_camera_frame.shape[0], 3]),
#         all_grid_points_camera_frame[:, :3],
#     ],
#     1,
# )


# def plot_pixels(fig: plt.Figure, ax: plt.Axes, points: jnp.ndarray):
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2])


# regularly_sampled_rays = sample_regular_positions_along_rays(
#     all_rays_towards_grid_points, 0.5, 3.0, 3
# )
# print(regularly_sampled_rays)
# fig_rays, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
# ax_rays_regular, ax_rays_nerf = axes

# ax_rays_regular.set_title("rays towards pixels, regular sampling")
# ax_rays_regular.text(1, 0, 0, "x-axis", "x")  # type: ignore
# ax_rays_regular.text(0, 1, 0, "y-axis", "y")  # type: ignore
# ax_rays_regular.text(0, 0, 1, "z-axis", "z")  # type: ignore
# flat_positions = regularly_sampled_rays.reshape(-1, 3)
# ax_rays_regular.scatter(
#     flat_positions[:, 0],
#     flat_positions[:, 1],
#     flat_positions[:, 2],
# )
# plot_pixels(fig_rays, ax_rays_regular, all_grid_points_camera_frame)

# nerf_sampled_rays = sample_nerf_rendering_positions_along_rays(
#     all_rays_towards_grid_points, 0.5, 3.0, 3, jax.random.PRNGKey(0)
# )
# ax_rays_nerf.set_title("rays towards pixels, NeRF (bins) sampling")
# ax_rays_nerf.text(1, 0, 0, "x-axis", "x")  # type: ignore
# ax_rays_nerf.text(0, 1, 0, "y-axis", "y")  # type: ignore
# ax_rays_nerf.text(0, 0, 1, "z-axis", "z")  # type: ignore
# flat_positions = nerf_sampled_rays.reshape(-1, 3)
# ax_rays_nerf.scatter(
#     flat_positions[:, 0],
#     flat_positions[:, 1],
#     flat_positions[:, 2],
# )
# plot_pixels(fig_rays, ax_rays_nerf, all_grid_points_camera_frame)

# plt.show()
