"""Utilities that plot various 3D data."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .camera import CameraParams


def plot_cameras(cameras: list[CameraParams]):
    """Plot the position, direction, and image pixels for multiple cameras. Run `plt.show()`, blocking."""

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for camera_params in cameras:
        # plot the axes
        origin_world = camera_params.camera_to_world @ (jnp.array([0, 0, 0, 1]))
        ax.scatter(
            [origin_world[0] / origin_world[3]],
            [origin_world[1] / origin_world[3]],
            [origin_world[2] / origin_world[3]],
            color="red",
        )

    plt.show()
