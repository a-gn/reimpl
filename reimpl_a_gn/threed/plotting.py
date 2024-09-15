"""Utilities that plot various 3D data."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .camera import CameraParams


def make_non_homogeneous(point):
    return point[:3] / point[3]


def plot_cameras(cameras: list[CameraParams], marked_camera: int | None = None):
    """Plot the position, direction, and image pixels for multiple cameras. Run `plt.show()`, blocking.

    @param marked_camera Plot this camera with a different style. Useful to e.g. show the holdout view in a NeRF.
    """

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for camera_index, camera_params in enumerate(cameras):
        color = "red" if camera_index == marked_camera else "blue"
        # plot the axes
        origin_world = camera_params.camera_to_world @ (jnp.array([0, 0, 0, 1]))
        origin_world_non_homogeneous = make_non_homogeneous(origin_world)
        ax.scatter(
            [origin_world_non_homogeneous[0]],
            [origin_world_non_homogeneous[1]],
            [origin_world_non_homogeneous[2]],
            color=color,
        )
        x_point = make_non_homogeneous(
            camera_params.camera_to_world @ jnp.array([1, 0, 0, 1])
        )
        y_point = make_non_homogeneous(
            camera_params.camera_to_world @ jnp.array([0, 1, 0, 1])
        )
        z_point = make_non_homogeneous(
            camera_params.camera_to_world @ jnp.array([0, 0, 1, 1])
        )
        x_direction = jnp.array([x_point - origin_world_non_homogeneous]).squeeze(0)
        x_direction /= jnp.linalg.norm(x_direction) * 15
        y_direction = jnp.array([y_point - origin_world_non_homogeneous]).squeeze(0)
        y_direction /= jnp.linalg.norm(y_direction) * 15
        z_direction = jnp.array([z_point - origin_world_non_homogeneous]).squeeze(0)
        z_direction /= jnp.linalg.norm(z_direction) * 15
        ax.quiver(*origin_world_non_homogeneous, *x_direction, color=color)
        ax.quiver(*origin_world_non_homogeneous, *y_direction, color=color)
        ax.quiver(*origin_world_non_homogeneous, *z_direction, color=color)

    plt.show()
