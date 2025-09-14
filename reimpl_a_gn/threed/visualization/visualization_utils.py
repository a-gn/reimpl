"""Visualization utilities for NeRF implementation.

Originally written by Claude Sonnet 4 on 2025/09/07
"""

import jax.numpy as jnp
import jax.typing as jt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ..rendering import CameraParams


def make_non_homogeneous(point: jt.ArrayLike) -> jnp.ndarray:
    """Convert homogeneous coordinates to non-homogeneous.

    @param point: Homogeneous coordinates. Shape: (..., 4).
    @return: Non-homogeneous coordinates. Shape: (..., 3).
    """
    point = jnp.array(point)
    return point[..., :3] / point[..., 3:4]


def normalize_colors(colors: jt.ArrayLike) -> jnp.ndarray:
    """Normalize color values to [0, 1] range for plotting.

    @param colors: Color values. Shape: (..., 3).
    @return: Normalized colors. Shape: (..., 3).
    """
    colors = jnp.array(colors)
    colors = jnp.clip(colors, 0.0, None)  # Remove negative values
    colors_max = jnp.max(colors)
    if colors_max > 1e-8:
        colors = colors / colors_max
    return jnp.clip(colors, 0.0, 1.0)


def create_custom_colormap(
    name: str = "density_alpha", colors: tuple[str, ...] = ("white", "red", "darkred")
) -> LinearSegmentedColormap:
    """Create a custom colormap for density visualization.

    @param name: Name of the colormap.
    @param colors: Sequence of color names for the colormap.
    @return: Custom matplotlib colormap.
    """
    return LinearSegmentedColormap.from_list(name, colors)


def setup_3d_plot(
    figsize: tuple[int, int] = (12, 10), title: str = "NeRF Visualization"
) -> tuple[Figure, Axes3D]:
    """Set up a 3D matplotlib plot with standard configuration.

    @param figsize: Figure size in inches.
    @param title: Plot title.
    @return: Tuple of (figure, axes) objects.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)

    # Set equal aspect ratio for better visualization
    ax.set_box_aspect((1, 1, 1))
    return fig, ax


def compute_axis_vectors(
    camera: CameraParams, scale: float = 1.0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute camera origin and axis vectors in world coordinates.

    @param camera: Camera parameters.
    @param scale: Scale factor for axis vectors.
    @return: Tuple of (origin, x_axis, y_axis, z_axis) in world coordinates.
    """
    # Camera origin in world coordinates
    origin_world = camera.camera_to_world @ jnp.array([0, 0, 0, 1])
    origin_world = make_non_homogeneous(origin_world)

    # Axis endpoints in camera coordinates, then transform to world
    x_point = make_non_homogeneous(camera.camera_to_world @ jnp.array([scale, 0, 0, 1]))
    y_point = make_non_homogeneous(camera.camera_to_world @ jnp.array([0, scale, 0, 1]))
    z_point = make_non_homogeneous(camera.camera_to_world @ jnp.array([0, 0, scale, 1]))

    # Convert to direction vectors
    x_axis = x_point - origin_world
    y_axis = y_point - origin_world
    z_axis = z_point - origin_world

    return origin_world, x_axis, y_axis, z_axis


def plot_coordinate_axes(
    ax: Axes3D,
    origin: jt.ArrayLike,
    x_axis: jt.ArrayLike,
    y_axis: jt.ArrayLike,
    z_axis: jt.ArrayLike,
    colors: tuple[str, str, str] = ("red", "green", "blue"),
    alpha: float = 0.8,
    arrow_length_ratio: float = 0.1,
    linewidth: float = 2.0,
) -> None:
    """Plot coordinate axes as arrows.

    @param ax: Matplotlib 3D axes object.
    @param origin: Origin point. Shape: (3,).
    @param x_axis: X-axis direction vector. Shape: (3,).
    @param y_axis: Y-axis direction vector. Shape: (3,).
    @param z_axis: Z-axis direction vector. Shape: (3,).
    @param colors: Colors for (x, y, z) axes.
    @param alpha: Transparency of arrows.
    @param arrow_length_ratio: Arrow head length as fraction of total length.
    @param linewidth: Line width for arrows.
    """
    origin = jnp.array(origin)
    axes = [jnp.array(x_axis), jnp.array(y_axis), jnp.array(z_axis)]

    for axis_vector, color in zip(axes, colors):
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_vector[0],
            axis_vector[1],
            axis_vector[2],
            color=color,
            alpha=alpha,
            arrow_length_ratio=arrow_length_ratio,
            linewidth=linewidth,
        )


def plot_world_origin(
    ax: Axes3D,
    scale: float = 1.0,
    colors: tuple[str, str, str] = ("darkred", "darkgreen", "darkblue"),
    alpha: float = 0.6,
) -> None:
    """Plot world coordinate system origin and axes.

    @param ax: Matplotlib 3D axes object.
    @param scale: Scale factor for axis vectors.
    @param colors: Colors for (x, y, z) axes.
    @param alpha: Transparency of arrows.
    """
    origin = jnp.array([0.0, 0.0, 0.0])
    x_axis = jnp.array([scale, 0.0, 0.0])
    y_axis = jnp.array([0.0, scale, 0.0])
    z_axis = jnp.array([0.0, 0.0, scale])

    plot_coordinate_axes(ax, origin, x_axis, y_axis, z_axis, colors, alpha)

    # Add origin marker
    ax.scatter([0], [0], [0], color="black", s=100, alpha=0.8, marker="o")


def compute_plot_bounds(
    points: jt.ArrayLike, padding: float = 0.1
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute appropriate plot bounds for 3D data.

    @param points: 3D points. Shape: (..., 3).
    @param padding: Fraction of range to add as padding.
    @return: Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    """
    points = jnp.array(points)
    if points.size == 0:
        return ((-1, 1), (-1, 1), (-1, 1))

    # Flatten to 2D array of points
    points_flat = points.reshape(-1, 3)

    mins = jnp.min(points_flat, axis=0)
    maxs = jnp.max(points_flat, axis=0)
    ranges = maxs - mins

    # Add padding
    padding_amounts = ranges * padding

    x_bounds = (
        float(mins[0] - padding_amounts[0]),
        float(maxs[0] + padding_amounts[0]),
    )
    y_bounds = (
        float(mins[1] - padding_amounts[1]),
        float(maxs[1] + padding_amounts[1]),
    )
    z_bounds = (
        float(mins[2] - padding_amounts[2]),
        float(maxs[2] + padding_amounts[2]),
    )

    return x_bounds, y_bounds, z_bounds


def set_axis_bounds(
    ax: Axes3D,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    """Set 3D axis bounds.

    @param ax: Matplotlib 3D axes object.
    @param bounds: Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)


def convert_density_to_alpha(
    densities: jt.ArrayLike,
    distances: jt.ArrayLike | None = None,
    default_distance: float = 1.0,
) -> jnp.ndarray:
    """Convert densities to alpha values using the NeRF formula.

    @param densities: Density values. Shape: (...,).
    @param distances: Distance intervals for each density. Shape same as densities.
    @param default_distance: Default distance if distances not provided.
    @return: Alpha values in [0, 1]. Shape same as densities.
    """
    densities = jnp.array(densities)
    if distances is None:
        distances = jnp.full_like(densities, default_distance)
    else:
        distances = jnp.array(distances)

    alpha_values = 1.0 - jnp.exp(-densities * distances)
    return jnp.clip(alpha_values, 0.0, 1.0)


def create_color_alpha_array(colors: jt.ArrayLike, alphas: jt.ArrayLike) -> np.ndarray:
    """Create RGBA array from colors and alpha values.

    @param colors: RGB colors. Shape: (..., 3).
    @param alphas: Alpha values. Shape: (...,) or (..., 1).
    @return: RGBA array. Shape: (..., 4).
    """
    colors = jnp.array(colors)
    alphas = jnp.array(alphas)

    # Ensure alphas has correct shape
    if alphas.ndim == colors.ndim - 1:
        alphas = jnp.expand_dims(alphas, -1)

    rgba = jnp.concatenate([colors, alphas], axis=-1)
    return np.array(rgba)


class PlotManager:
    """Manager class for handling multiple visualization plots.

    Originally written by Claude Sonnet 4 on 2025/09/07
    """

    def __init__(self, figsize: tuple[int, int] = (15, 12)):
        """Initialize plot manager.

        @param figsize: Default figure size for plots.
        """
        self.figsize = figsize
        self.plots: dict[str, tuple[plt.Figure, plt.Axes]] = {}
        self.current_plot: str | None = None

    def create_plot(
        self,
        name: str,
        title: str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[Figure, Axes3D]:
        """Create a new 3D plot.

        @param name: Unique name for the plot.
        @param title: Plot title (defaults to name if not provided).
        @param figsize: Figure size (uses default if not provided).
        @return: Tuple of (figure, axes) objects.
        """
        if title is None:
            title = name.replace("_", " ").title()

        if figsize is None:
            figsize = self.figsize

        fig, ax = setup_3d_plot(figsize, title)
        self.plots[name] = (fig, ax)
        self.current_plot = name
        return fig, ax

    def get_plot(self, name: str) -> tuple[Figure, Axes3D]:
        """Get an existing plot.

        @param name: Name of the plot.
        @return: Tuple of (figure, axes) objects.
        @raises KeyError: If plot with given name doesn't exist.
        """
        if name not in self.plots:
            raise KeyError(
                f"Plot '{name}' not found. Available plots: {list(self.plots.keys())}"
            )

        self.current_plot = name
        return self.plots[name]

    def show_plot(self, name: str | None = None, block: bool = True) -> None:
        """Show a specific plot or the current plot.

        @param name: Name of plot to show (uses current if None).
        @param block: Whether to block execution until plot is closed.
        """
        if name is None:
            name = self.current_plot

        if name is None or name not in self.plots:
            raise ValueError("No plot specified and no current plot available")

        fig, _ = self.plots[name]
        plt.figure(fig.number)
        plt.show(block=block)

    def show_all(self, block: bool = True) -> None:
        """Show all plots.

        @param block: Whether to block execution until all plots are closed.
        """
        for name in self.plots:
            self.show_plot(name, block=False)

        if block:
            plt.show(block=True)

    def clear_plot(self, name: str) -> None:
        """Clear a specific plot.

        @param name: Name of plot to clear.
        """
        if name in self.plots:
            fig, ax = self.plots[name]
            ax.clear()

    def close_plot(self, name: str) -> None:
        """Close and remove a specific plot.

        @param name: Name of plot to close.
        """
        if name in self.plots:
            fig, _ = self.plots[name]
            plt.close(fig)
            del self.plots[name]
            if self.current_plot == name:
                self.current_plot = None

    def close_all(self) -> None:
        """Close all plots."""
        for name in list(self.plots.keys()):
            self.close_plot(name)
