"""Ray visualization functions for NeRF implementation.

Originally written by Claude Sonnet 4 on 2025/09/07
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.typing as jt
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

from ..rendering import CameraParams, from_homogeneous
from .visualization_utils import (
    compute_axis_vectors,
    compute_plot_bounds,
    plot_world_origin,
    set_axis_bounds,
    setup_3d_plot,
)


def visualize_ray_sampling(
    camera: CameraParams,
    pixel_coordinates: jt.ArrayLike,
    rays: jt.ArrayLike,
    sampling_points: jt.ArrayLike | None = None,
    ray_colors: jt.ArrayLike | None = None,
    show_camera: bool = True,
    show_world_origin: bool = True,
    max_rays_to_show: int = 10,
    ray_length: float = 5.0,
    title: str = "Ray Sampling Visualization",
    figsize: tuple[int, int] = (14, 12),
    show_plot: bool = True,
) -> tuple[Figure, Axes3D]:  # type: ignore
    """Visualize ray sampling from camera through pixels.

    @param camera: Camera parameters.
    @param pixel_coordinates: Pixel coordinates. Shape: (N, 2).
    @param rays: Ray parameters (origin + direction). Shape: (N, 6).
    @param sampling_points: Optional 3D sampling points along rays. Shape: (N, K, 3).
    @param ray_colors: Optional colors for each ray. Shape: (N, 3).
    @param show_camera: Whether to show camera axes and origin.
    @param show_world_origin: Whether to show world coordinate system.
    @param max_rays_to_show: Maximum number of rays to visualize (for performance).
    @param ray_length: Length of ray arrows to draw.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    # Convert inputs to arrays
    pixel_coordinates = jnp.array(pixel_coordinates)
    rays = jnp.array(rays)

    # Limit number of rays for visualization
    num_rays = min(len(rays), max_rays_to_show)
    if num_rays < len(rays):
        # Sample evenly spaced rays
        indices = jnp.linspace(0, len(rays) - 1, num_rays, dtype=int)
        pixel_coordinates = pixel_coordinates[indices]
        rays = rays[indices]
        if sampling_points is not None:
            sampling_points = jnp.array(sampling_points)[indices]
        if ray_colors is not None:
            ray_colors = jnp.array(ray_colors)[indices]

    # Extract ray origins and directions
    ray_origins = rays[:, :3]
    ray_directions = rays[:, 3:6]

    # Normalize ray directions
    ray_dir_norms = jnp.linalg.norm(ray_directions, axis=1, keepdims=True)
    ray_directions_normalized = ray_directions / ray_dir_norms

    # Plot camera if requested
    if show_camera:
        origin, x_axis, y_axis, z_axis = compute_axis_vectors(camera, 1.0)
        ax.scatter(
            [origin[0]],
            [origin[1]],
            [origin[2]],
            color="red",
            s=120,
            alpha=0.9,
            marker="o",
            edgecolors="black",
            linewidth=1,
            label="Camera",
        )

        # Plot camera axes (smaller)
        from .visualization_utils import plot_coordinate_axes

        plot_coordinate_axes(
            ax,
            origin,
            x_axis * 0.5,
            y_axis * 0.5,
            z_axis * 0.5,
            colors=("red", "green", "blue"),
            alpha=0.7,
            linewidth=1.5,
        )

    # Plot world origin if requested
    if show_world_origin:
        plot_world_origin(ax, scale=1.5, alpha=0.5)

    # Default ray colors if not provided
    if ray_colors is None:
        # Use a colormap for different rays
        cmap = plt.cm.viridis
        ray_colors = np.array(
            [cmap(i / max(1, num_rays - 1))[:3] for i in range(num_rays)]
        )
    else:
        ray_colors = jnp.array(ray_colors)

    # Plot rays as arrows
    for i in range(num_rays):
        color = ray_colors[i] if len(ray_colors.shape) > 1 else ray_colors

        # Draw ray as arrow
        ax.quiver(
            ray_origins[i, 0],
            ray_origins[i, 1],
            ray_origins[i, 2],
            ray_directions_normalized[i, 0] * ray_length,
            ray_directions_normalized[i, 1] * ray_length,
            ray_directions_normalized[i, 2] * ray_length,
            color=color,
            alpha=0.7,
            arrow_length_ratio=0.1,
            linewidth=2,
        )

        # Add pixel label
        pixel_x, pixel_y = pixel_coordinates[i]
        label_point = ray_origins[i] + ray_directions_normalized[i] * (ray_length * 0.9)
        ax.text(
            label_point[0],
            label_point[1],
            label_point[2],
            f"px({pixel_x:.0f},{pixel_y:.0f})",
            fontsize=8,
            color=color,
            alpha=0.9,
        )

    # Plot sampling points along rays if provided
    if sampling_points is not None:
        _plot_sampling_points(ax, sampling_points, ray_colors)

    # Collect all points for bounds calculation
    all_points = [ray_origins]
    if show_camera:
        all_points.append(jnp.array([origin]))
    if show_world_origin:
        all_points.append(jnp.array([[0, 0, 0]]))
    if sampling_points is not None:
        all_points.append(sampling_points.reshape(-1, 3))
    else:
        # Include ray endpoints
        ray_endpoints = ray_origins + ray_directions_normalized * ray_length
        all_points.append(ray_endpoints)

    # Set bounds
    bounds = compute_plot_bounds(jnp.vstack(all_points), padding=0.1)
    set_axis_bounds(ax, bounds)

    # Add legend
    legend_elements = []
    if show_camera:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Camera",
            )
        )
    if show_world_origin:
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
    legend_elements.extend(
        [
            plt.Line2D([0], [0], color="blue", linewidth=2, label="Rays"),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="purple",
                markersize=6,
                label="Sampling Points",
            ),
        ]
    )
    ax.legend(handles=legend_elements, loc="upper right")

    if show_plot:
        plt.show()

    return fig, ax


def visualize_pixel_grid(
    camera: CameraParams,
    pixel_grid: jt.ArrayLike,
    highlight_pixels: jt.ArrayLike | None = None,
    show_rays: bool = True,
    ray_length: float = 3.0,
    title: str = "Pixel Grid Visualization",
    figsize: tuple[int, int] = (12, 10),
    show_plot: bool = True,
) -> tuple[Figure, Axes3D]:  # type: ignore
    """Visualize a grid of pixels and their corresponding rays.

    @param camera: Camera parameters.
    @param pixel_grid: Grid of pixel coordinates. Shape: (H, W, 2).
    @param highlight_pixels: Optional specific pixels to highlight. Shape: (N, 2).
    @param show_rays: Whether to show rays from pixels.
    @param ray_length: Length of ray visualization.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    pixel_grid = jnp.array(pixel_grid)

    # Flatten pixel grid for processing
    if pixel_grid.ndim == 3:
        h, w = pixel_grid.shape[:2]
        pixel_coords_flat = pixel_grid.reshape(-1, 2)
    else:
        pixel_coords_flat = pixel_grid
        h, w = (
            int(jnp.sqrt(len(pixel_coords_flat))),
            int(jnp.sqrt(len(pixel_coords_flat))),
        )

    # Get camera origin
    origin, _, _, _ = compute_axis_vectors(camera, 1.0)

    # Plot camera
    ax.scatter(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        color="red",
        s=150,
        alpha=0.9,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="Camera",
    )

    if show_rays:
        # Get ray directions for all pixels
        ray_directions_world = camera.image_to_world(pixel_coords_flat)
        ray_directions = jnp.array(
            [
                from_homogeneous(ray_directions_world[i : i + 1])[0]
                for i in range(len(pixel_coords_flat))
            ]
        )

        # Sample a subset of pixels for visualization (grid pattern)
        step = max(1, len(pixel_coords_flat) // 50)  # Show ~50 rays max
        indices = jnp.arange(0, len(pixel_coords_flat), step)

        # Plot rays with slight transparency
        for i in indices:
            end_point = origin + ray_directions[i] * ray_length
            ax.plot(
                [origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]],
                color="lightblue",
                alpha=0.3,
                linewidth=0.8,
            )

        # Plot highlighted pixels if provided
        if highlight_pixels is not None:
            highlight_pixels = jnp.array(highlight_pixels)
            highlight_rays = camera.image_to_world(highlight_pixels)
            highlight_directions = jnp.array(
                [
                    from_homogeneous(highlight_rays[i : i + 1])[0]
                    for i in range(len(highlight_pixels))
                ]
            )

            for i, direction in enumerate(highlight_directions):
                end_point = origin + direction * ray_length
                ax.plot(
                    [origin[0], end_point[0]],
                    [origin[1], end_point[1]],
                    [origin[2], end_point[2]],
                    color="orange",
                    alpha=0.8,
                    linewidth=3,
                    label="Highlighted Rays" if i == 0 else "",
                )

                # Add pixel coordinate label
                label_point = origin + direction * (ray_length * 0.9)
                ax.text(
                    label_point[0],
                    label_point[1],
                    label_point[2],
                    f"({highlight_pixels[i, 0]:.0f},{highlight_pixels[i, 1]:.0f})",
                    fontsize=10,
                    color="orange",
                    fontweight="bold",
                )

    # Show world origin
    plot_world_origin(ax, scale=1.0, alpha=0.6)

    # Set bounds
    if show_rays:
        ray_endpoints = origin + ray_directions[::step] * ray_length
        points_for_bounds = jnp.vstack(
            [
                jnp.array([origin]),
                jnp.array([[0, 0, 0]]),  # World origin
                ray_endpoints,
            ]
        )
    else:
        points_for_bounds = jnp.array([origin, [0, 0, 0]])

    bounds = compute_plot_bounds(points_for_bounds, padding=0.15)
    set_axis_bounds(ax, bounds)

    # Add info text
    info_text = f"Pixel Grid: {h}×{w}\\nTotal pixels: {len(pixel_coords_flat)}"
    if show_rays:
        info_text += f"\\nShowing every {step} rays"

    ax.text2D(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.legend(loc="upper left")

    if show_plot:
        plt.show()

    return fig, ax


def _plot_sampling_points(
    ax: Axes3D,  # type: ignore
    sampling_points: jt.ArrayLike,
    ray_colors: jt.ArrayLike,
    point_size: float = 20,
    alpha: float = 0.6,
) -> None:
    """Plot sampling points along rays.

    @param ax: Matplotlib 3D axes object.
    @param sampling_points: 3D sampling points. Shape: (N_rays, N_samples, 3).
    @param ray_colors: Colors for each ray. Shape: (N_rays, 3).
    @param point_size: Size of sampling point markers.
    @param alpha: Transparency of points.
    """
    sampling_points = jnp.array(sampling_points)
    ray_colors = jnp.array(ray_colors)

    num_rays, num_samples = sampling_points.shape[:2]

    for ray_idx in range(num_rays):
        color = ray_colors[ray_idx] if len(ray_colors.shape) > 1 else ray_colors
        points = sampling_points[ray_idx]  # Shape: (num_samples, 3)

        # Plot all sampling points for this ray
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            s=point_size,
            alpha=alpha,
            marker="o",
        )

        # Connect sampling points with a line to show ray path
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            alpha=alpha * 0.5,
            linewidth=1,
            linestyle=":",
        )


def visualize_ray_comparison(
    camera: CameraParams,
    pixel_coordinates: jt.ArrayLike,
    coarse_rays: jt.ArrayLike,
    fine_rays: jt.ArrayLike | None = None,
    coarse_points: jt.ArrayLike | None = None,
    fine_points: jt.ArrayLike | None = None,
    max_rays: int = 5,
    ray_length: float = 4.0,
    title: str = "Coarse vs Fine Ray Sampling",
    figsize: tuple[int, int] = (16, 12),
    show_plot: bool = True,
) -> tuple[Figure, Axes3D]:  # type: ignore
    """Compare coarse and fine ray sampling side by side.

    @param camera: Camera parameters.
    @param pixel_coordinates: Pixel coordinates. Shape: (N, 2).
    @param coarse_rays: Coarse ray parameters. Shape: (N, 6).
    @param fine_rays: Optional fine ray parameters. Shape: (N, 6).
    @param coarse_points: Optional coarse sampling points. Shape: (N, K1, 3).
    @param fine_points: Optional fine sampling points. Shape: (N, K2, 3).
    @param max_rays: Maximum number of rays to visualize.
    @param ray_length: Length of ray visualization.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    # Convert inputs and limit rays
    pixel_coordinates = jnp.array(pixel_coordinates)[:max_rays]
    coarse_rays = jnp.array(coarse_rays)[:max_rays]
    if fine_rays is not None:
        fine_rays = jnp.array(fine_rays)[:max_rays]
    if coarse_points is not None:
        coarse_points = jnp.array(coarse_points)[:max_rays]
    if fine_points is not None:
        fine_points = jnp.array(fine_points)[:max_rays]

    num_rays = len(coarse_rays)

    # Get camera origin and plot it
    origin, _, _, _ = compute_axis_vectors(camera, 1.0)
    ax.scatter(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        color="red",
        s=150,
        alpha=0.9,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="Camera",
    )

    # Plot world origin
    plot_world_origin(ax, scale=1.5, alpha=0.5)

    # Colors for coarse and fine
    coarse_color = "blue"
    fine_color = "green"

    # Plot coarse rays and sampling points
    coarse_origins = coarse_rays[:, :3]
    coarse_directions = coarse_rays[:, 3:6]
    coarse_directions_norm = coarse_directions / jnp.linalg.norm(
        coarse_directions, axis=1, keepdims=True
    )

    for i in range(num_rays):
        # Coarse ray
        end_point = coarse_origins[i] + coarse_directions_norm[i] * ray_length
        ax.plot(
            [coarse_origins[i, 0], end_point[0]],
            [coarse_origins[i, 1], end_point[1]],
            [coarse_origins[i, 2], end_point[2]],
            color=coarse_color,
            alpha=0.7,
            linewidth=2,
            label="Coarse Rays" if i == 0 else "",
        )

        # Pixel label
        pixel_x, pixel_y = pixel_coordinates[i]
        label_point = coarse_origins[i] + coarse_directions_norm[i] * (
            ray_length * 0.95
        )
        ax.text(
            label_point[0],
            label_point[1],
            label_point[2],
            f"px({pixel_x:.0f},{pixel_y:.0f})",
            fontsize=8,
            color=coarse_color,
        )

    # Plot coarse sampling points
    if coarse_points is not None:
        for i in range(num_rays):
            points = coarse_points[i]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=coarse_color,
                s=30,
                alpha=0.8,
                marker="o",
                label="Coarse Points" if i == 0 else "",
            )

    # Plot fine rays and sampling points if provided
    if fine_rays is not None:
        fine_origins = fine_rays[:, :3]
        fine_directions = fine_rays[:, 3:6]
        fine_directions_norm = fine_directions / jnp.linalg.norm(
            fine_directions, axis=1, keepdims=True
        )

        for i in range(num_rays):
            # Fine ray (slightly offset for visibility)
            offset = jnp.array([0.05, 0.05, 0.05]) * i  # Small offset per ray
            end_point = fine_origins[i] + fine_directions_norm[i] * ray_length + offset
            ax.plot(
                [fine_origins[i, 0] + offset[0], end_point[0]],
                [fine_origins[i, 1] + offset[1], end_point[1]],
                [fine_origins[i, 2] + offset[2], end_point[2]],
                color=fine_color,
                alpha=0.7,
                linewidth=2,
                linestyle="--",
                label="Fine Rays" if i == 0 else "",
            )

    # Plot fine sampling points
    if fine_points is not None:
        for i in range(num_rays):
            points = fine_points[i]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=fine_color,
                s=20,
                alpha=0.8,
                marker="^",
                label="Fine Points" if i == 0 else "",
            )

    # Set bounds
    all_points = [coarse_origins, jnp.array([origin]), jnp.array([[0, 0, 0]])]
    if coarse_points is not None:
        all_points.append(coarse_points.reshape(-1, 3))
    if fine_points is not None:
        all_points.append(fine_points.reshape(-1, 3))

    bounds = compute_plot_bounds(jnp.vstack(all_points), padding=0.15)
    set_axis_bounds(ax, bounds)

    ax.legend(loc="upper right")

    if show_plot:
        plt.show()

    return fig, ax
