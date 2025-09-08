"""Camera and coordinate system visualization for NeRF implementation.

Originally written by Claude Sonnet 4 on 2025/09/07
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.typing as jt
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

from ..rendering import CameraParams, from_homogeneous
from .visualization_utils import (
    PlotManager,
    compute_axis_vectors,
    compute_plot_bounds,
    plot_coordinate_axes,
    plot_world_origin,
    set_axis_bounds,
    setup_3d_plot,
)


def visualize_cameras(
    cameras: Sequence[CameraParams],
    marked_camera: int | None = None,
    show_world_axes: bool = True,
    camera_scale: float = 1.0,
    world_scale: float = 2.0,
    title: str = "Camera Positions and Orientations",
    figsize: tuple[int, int] = (12, 10),
    show_plot: bool = True,
) -> tuple[Figure, Axes3D]:
    """Visualize multiple cameras with their positions and orientations.

    @param cameras: List of camera parameters to visualize.
    @param marked_camera: Index of camera to highlight (useful for holdout views).
    @param show_world_axes: Whether to show world coordinate axes.
    @param camera_scale: Scale factor for camera coordinate axes.
    @param world_scale: Scale factor for world coordinate axes.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    # Collect all camera origins for computing bounds
    all_origins = []

    for camera_index, camera_params in enumerate(cameras):
        # Determine colors for this camera
        is_marked = marked_camera is not None and camera_index == marked_camera
        base_color = "red" if is_marked else "blue"
        alpha = 0.9 if is_marked else 0.7

        # Get camera origin and axes
        origin, x_axis, y_axis, z_axis = compute_axis_vectors(
            camera_params, camera_scale
        )
        all_origins.append(origin)

        # Plot camera origin as a larger marker
        marker_size = 120 if is_marked else 80
        ax.scatter(
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
            color=base_color,
            s=marker_size,
            alpha=alpha,
            marker="o",
            edgecolors="black",
            linewidth=1 if is_marked else 0.5,
        )

        # Plot camera coordinate axes
        camera_colors = (
            ("darkred", "darkgreen", "darkblue")
            if is_marked
            else ("lightcoral", "lightgreen", "lightblue")
        )
        plot_coordinate_axes(
            ax,
            origin,
            x_axis,
            y_axis,
            z_axis,
            colors=camera_colors,
            alpha=alpha,
            linewidth=2.5 if is_marked else 1.5,
        )

        # Add camera label
        ax.text(
            float(origin[0]),
            float(origin[1]),
            float(origin[2] + camera_scale * 0.3),
            f"Cam {camera_index}" + (" (marked)" if is_marked else ""),
            fontsize=10 if is_marked else 8,
            fontweight="bold" if is_marked else "normal",
            ha="center",
        )

    # Show world coordinate axes if requested
    if show_world_axes:
        plot_world_origin(ax, world_scale)
        ax.text(
            0.0,
            0.0,
            float(world_scale * 1.2),
            "World Origin",
            fontsize=12,
            fontweight="bold",
            ha="center",
            color="black",
        )

    # Compute and set appropriate bounds
    if all_origins:
        # Include world origin in bounds calculation
        points_for_bounds = jnp.array(all_origins)
        if show_world_axes:
            world_points = jnp.array([[0, 0, 0]])
            points_for_bounds = jnp.vstack([points_for_bounds, world_points])

        bounds = compute_plot_bounds(points_for_bounds, padding=0.2)
        set_axis_bounds(ax, bounds)

    # Add legend
    legend_elements = []
    if marked_camera is not None:
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    label="Marked Camera",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="blue",
                    markersize=8,
                    label="Other Cameras",
                ),
            ]
        )
    else:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Cameras",
            )
        )

    if show_world_axes:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=8,
                label="World Origin",
            )
        )

    ax.legend(handles=legend_elements, loc="upper left")

    if show_plot:
        plt.show()

    return fig, ax


def visualize_single_camera_detailed(
    camera: CameraParams,
    pixel_coordinates: jt.ArrayLike | None = None,
    camera_scale: float = 1.0,
    world_scale: float = 2.0,
    show_frustum: bool = True,
    frustum_depth: float = 3.0,
    title: str = "Detailed Camera View",
    figsize: tuple[int, int] = (12, 10),
    show_plot: bool = True,
) -> tuple[Figure, Axes3D]:
    """Visualize a single camera with detailed information.

    @param camera: Camera parameters to visualize.
    @param pixel_coordinates: Optional pixel coordinates to show rays for.
    @param camera_scale: Scale factor for camera coordinate axes.
    @param world_scale: Scale factor for world coordinate axes.
    @param show_frustum: Whether to show camera frustum outline.
    @param frustum_depth: Depth of frustum visualization.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    # Get camera origin and axes
    origin, x_axis, y_axis, z_axis = compute_axis_vectors(camera, camera_scale)

    # Plot camera origin
    ax.scatter(
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        color="red",
        s=150,
        alpha=0.9,
        marker="o",
        edgecolors="black",
        linewidth=2,
    )

    # Plot camera coordinate axes
    plot_coordinate_axes(
        ax,
        origin,
        x_axis,
        y_axis,
        z_axis,
        colors=("red", "green", "blue"),
        alpha=0.9,
        linewidth=3.0,
    )

    # Add axis labels
    x_tip = origin + x_axis
    y_tip = origin + y_axis
    z_tip = origin + z_axis

    ax.text(
        float(x_tip[0]),
        float(x_tip[1]),
        float(x_tip[2]),
        "X",
        fontsize=12,
        fontweight="bold",
        color="red",
    )
    ax.text(
        float(y_tip[0]),
        float(y_tip[1]),
        float(y_tip[2]),
        "Y",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    ax.text(
        float(z_tip[0]),
        float(z_tip[1]),
        float(z_tip[2]),
        "Z",
        fontsize=12,
        fontweight="bold",
        color="blue",
    )

    # Show camera frustum if requested
    if show_frustum:
        _draw_camera_frustum(ax, camera, frustum_depth)

    # Show world coordinate axes
    plot_world_origin(ax, world_scale)
    ax.text(
        0.0,
        0.0,
        float(world_scale * 1.2),
        "World Origin",
        fontsize=12,
        fontweight="bold",
        ha="center",
        color="black",
    )

    # Draw rays to specific pixels if provided
    if pixel_coordinates is not None:
        _draw_pixel_rays(ax, camera, pixel_coordinates, frustum_depth)

    # Add camera information as text
    info_text = (
        f"Camera Info:\\n"
        f"fx: {camera.fx:.2f}\\n"
        f"fy: {camera.fy:.2f}\\n"
        f"Position: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})"
    )
    ax.text2D(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Compute bounds including frustum
    points_for_bounds = jnp.array([origin, [0, 0, 0]])  # Camera origin and world origin
    if show_frustum:
        # Add frustum corners to bounds calculation
        frustum_corners = _compute_frustum_corners(camera, frustum_depth)
        points_for_bounds = jnp.vstack([points_for_bounds, frustum_corners])

    bounds = compute_plot_bounds(points_for_bounds, padding=0.2)
    set_axis_bounds(ax, bounds)

    if show_plot:
        plt.show()

    return fig, ax


def _draw_camera_frustum(
    ax: Axes3D,
    camera: CameraParams,
    depth: float,
    alpha: float = 0.3,
    color: str = "cyan",
) -> None:
    """Draw camera frustum wireframe.

    @param ax: Matplotlib 3D axes object.
    @param camera: Camera parameters.
    @param depth: Depth of frustum to draw.
    @param alpha: Transparency of frustum lines.
    @param color: Color of frustum lines.
    """
    # Get camera origin
    origin, _, _, _ = compute_axis_vectors(camera, 1.0)

    # Compute frustum corners at given depth
    frustum_corners = _compute_frustum_corners(camera, depth)

    # Draw lines from camera center to each corner
    for corner in frustum_corners:
        ax.plot(
            [origin[0], corner[0]],
            [origin[1], corner[1]],
            [origin[2], corner[2]],
            color=color,
            alpha=alpha,
            linewidth=1.5,
            linestyle="--",
        )

    # Draw frustum rectangle at depth
    corners = frustum_corners.reshape(2, 2, 3)  # [height, width, xyz]

    # Draw horizontal lines
    for i in range(2):
        ax.plot(
            [corners[i, 0, 0], corners[i, 1, 0]],
            [corners[i, 0, 1], corners[i, 1, 1]],
            [corners[i, 0, 2], corners[i, 1, 2]],
            color=color,
            alpha=alpha * 1.5,
            linewidth=2,
        )

    # Draw vertical lines
    for j in range(2):
        ax.plot(
            [corners[0, j, 0], corners[1, j, 0]],
            [corners[0, j, 1], corners[1, j, 1]],
            [corners[0, j, 2], corners[1, j, 2]],
            color=color,
            alpha=alpha * 1.5,
            linewidth=2,
        )


def _compute_frustum_corners(camera: CameraParams, depth: float) -> jnp.ndarray:
    """Compute frustum corners at given depth.

    @param camera: Camera parameters.
    @param depth: Depth at which to compute corners.
    @return: Frustum corner positions in world coordinates. Shape: (4, 3).
    """
    # Use a simple image size for frustum visualization
    image_width, image_height = 100, 100

    # Corner pixels (top-left, top-right, bottom-left, bottom-right)
    corner_pixels = jnp.array(
        [[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]],
        dtype=float,
    )

    # Get ray directions for corner pixels
    ray_directions_world = camera.image_to_world(corner_pixels)
    ray_directions = jnp.array(
        [from_homogeneous(ray_directions_world[i : i + 1])[0] for i in range(4)]
    )

    # Camera origin in world coordinates
    origin, _, _, _ = compute_axis_vectors(camera, 1.0)

    # Compute corner positions at given depth
    corners = origin + ray_directions * depth
    return corners


def _draw_pixel_rays(
    ax: Axes3D,
    camera: CameraParams,
    pixel_coordinates: jt.ArrayLike,
    ray_length: float,
    color: str = "orange",
    alpha: float = 0.7,
) -> None:
    """Draw rays from camera to specific pixel coordinates.

    @param ax: Matplotlib 3D axes object.
    @param camera: Camera parameters.
    @param pixel_coordinates: Pixel coordinates. Shape: (N, 2).
    @param ray_length: Length of rays to draw.
    @param color: Color of rays.
    @param alpha: Transparency of rays.
    """
    pixel_coordinates = jnp.array(pixel_coordinates)
    if pixel_coordinates.ndim == 1:
        pixel_coordinates = pixel_coordinates.reshape(1, -1)

    # Get camera origin
    origin, _, _, _ = compute_axis_vectors(camera, 1.0)

    # Get ray directions
    ray_directions_world = camera.image_to_world(pixel_coordinates)
    ray_directions = jnp.array(
        [
            from_homogeneous(ray_directions_world[i : i + 1])[0]
            for i in range(len(pixel_coordinates))
        ]
    )

    # Draw rays
    for i, direction in enumerate(ray_directions):
        end_point = origin + direction * ray_length
        ax.plot(
            [origin[0], end_point[0]],
            [origin[1], end_point[1]],
            [origin[2], end_point[2]],
            color=color,
            alpha=alpha,
            linewidth=2,
        )

        # Add pixel label
        label_point = origin + direction * (ray_length * 0.8)
        ax.text(
            float(label_point[0]),
            float(label_point[1]),
            float(label_point[2]),
            f"px({pixel_coordinates[i, 0]:.0f},{pixel_coordinates[i, 1]:.0f})",
            fontsize=8,
            color=color,
        )


def create_camera_comparison_plot(
    cameras: Sequence[CameraParams],
    camera_names: Sequence[str] | None = None,
    figsize: tuple[int, int] = (16, 12),
    show_plot: bool = True,
) -> PlotManager:
    """Create a plot manager with individual camera visualizations for comparison.

    @param cameras: List of camera parameters to visualize.
    @param camera_names: Optional names for each camera.
    @param figsize: Figure size for each subplot.
    @param show_plot: Whether to display plots immediately.
    @return: PlotManager instance with individual camera plots.
    """
    if camera_names is None:
        camera_names = [f"Camera_{i}" for i in range(len(cameras))]

    if len(camera_names) != len(cameras):
        raise ValueError("Number of camera names must match number of cameras")

    plot_manager = PlotManager(figsize)

    for i, (camera, name) in enumerate(zip(cameras, camera_names)):
        fig, ax = plot_manager.create_plot(f"camera_{i}", f"Detailed View - {name}")

        # Use the detailed single camera visualization
        visualize_single_camera_detailed(
            camera,
            camera_scale=1.0,
            world_scale=2.0,
            show_frustum=True,
            title=f"Detailed View - {name}",
            figsize=figsize,
            show_plot=False,  # Don't show individual plots yet
        )

    if show_plot:
        plot_manager.show_all(block=False)

    return plot_manager
