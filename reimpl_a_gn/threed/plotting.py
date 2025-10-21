"""Utilities that plot various 3D data."""

import jax.numpy as jnp
import jax.typing as jt
import matplotlib.pyplot as plt


def make_non_homogeneous(point: jt.ArrayLike):
    """Convert homogeneous point to non-homogeneous coordinates.

    @param point Homogeneous point. Shape: (4,). Last axis: x, y, z, w.
    @return Non-homogeneous point. Shape: (3,). Order: x/w, y/w, z/w.
    """
    point = jnp.array(point)
    return point[:3] / point[3]


def plot_cameras(
    camera_to_world_matrices: list[jt.ArrayLike], marked_camera: int | None = None
):
    """Plot the position and direction for multiple cameras. Run `plt.show()`. Blocking if the pyplot backend blocks.

    @param camera_to_world_matrices List of camera-to-world transformation matrices. Each has shape (4, 4).
    @param marked_camera Plot this camera with a different style. Useful to e.g. show the holdout view in a NeRF.
    """

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for camera_index, camera_to_world in enumerate(camera_to_world_matrices):
        camera_to_world = jnp.array(camera_to_world)
        color = (
            "red"
            if marked_camera is not None and camera_index == marked_camera
            else "blue"
        )
        # plot the axes
        origin_world = camera_to_world @ (jnp.array([0, 0, 0, 1]))
        origin_world_non_homogeneous = make_non_homogeneous(origin_world)
        ax.scatter(
            [origin_world_non_homogeneous[0]],
            [origin_world_non_homogeneous[1]],
            [origin_world_non_homogeneous[2]],
            color=color,
        )
        x_point = make_non_homogeneous(camera_to_world @ jnp.array([1, 0, 0, 1]))
        y_point = make_non_homogeneous(camera_to_world @ jnp.array([0, 1, 0, 1]))
        z_point = make_non_homogeneous(camera_to_world @ jnp.array([0, 0, 1, 1]))
        x_direction = jnp.array([x_point - origin_world_non_homogeneous])
        x_direction /= jnp.linalg.norm(x_direction, axis=1, keepdims=True) * 15
        y_direction = jnp.array([y_point - origin_world_non_homogeneous])
        y_direction /= jnp.linalg.norm(y_direction, axis=1, keepdims=True) * 15
        z_direction = jnp.array([z_point - origin_world_non_homogeneous])
        z_direction /= jnp.linalg.norm(z_direction, axis=1, keepdims=True) * 15
        ax.quiver(*origin_world_non_homogeneous, *(x_direction.squeeze(0)), color=color)
        ax.quiver(*origin_world_non_homogeneous, *(y_direction.squeeze(0)), color=color)
        ax.quiver(*origin_world_non_homogeneous, *(z_direction.squeeze(0)), color=color)

    plt.show()
