"""Utilities that plot various 3D data."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .camera import CameraParams


def make_non_homogeneous(point):
    return point[:3] / point[3]


def plot_cameras(cameras: list[CameraParams]):
    """Plot the position, direction, and image pixels for multiple cameras. Run `plt.show()`, blocking."""

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for camera_params in cameras:
        # plot the axes
        origin_world = camera_params.camera_to_world @ (jnp.array([0, 0, 0, 1]))
        origin_world_non_homogeneous = make_non_homogeneous(origin_world)
        ax.scatter(
            [origin_world_non_homogeneous[0]],
            [origin_world_non_homogeneous[1]],
            [origin_world_non_homogeneous[2]],
            color="red",
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
        ax.quiver(*origin_world_non_homogeneous, *x_direction)
        ax.quiver(*origin_world_non_homogeneous, *y_direction)
        ax.quiver(*origin_world_non_homogeneous, *z_direction)

    plt.show()
