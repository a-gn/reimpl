"""NeRF prediction and volume rendering visualization.

Originally written by Claude Sonnet 4 on 2025/09/07
"""

import jax.numpy as jnp
import jax.typing as jt
import matplotlib.pyplot as plt

from .visualization_utils import (
    compute_plot_bounds,
    convert_density_to_alpha,
    normalize_colors,
    plot_world_origin,
    set_axis_bounds,
    setup_3d_plot,
)


def visualize_nerf_predictions(
    sampling_points: jt.ArrayLike,
    predicted_colors: jt.ArrayLike,
    predicted_densities: jt.ArrayLike,
    ray_origins: jt.ArrayLike | None = None,
    max_rays: int = 8,
    density_threshold: float = 0.1,
    alpha_scale: float = 1.0,
    title: str = "NeRF Network Predictions",
    figsize: tuple[int, int] = (16, 12),
    show_plot: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize NeRF network predictions (colors and densities) along rays.

    @param sampling_points: 3D sampling points. Shape: (N_rays, N_samples, 3).
    @param predicted_colors: Predicted RGB colors. Shape: (N_rays, N_samples, 3).
    @param predicted_densities: Predicted densities. Shape: (N_rays, N_samples).
    @param ray_origins: Optional ray origins for context. Shape: (N_rays, 3).
    @param max_rays: Maximum number of rays to visualize.
    @param density_threshold: Minimum density to display.
    @param alpha_scale: Scale factor for alpha values.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    # Convert inputs to arrays and limit rays
    sampling_points = jnp.array(sampling_points)[:max_rays]
    predicted_colors = jnp.array(predicted_colors)[:max_rays]
    predicted_densities = jnp.array(predicted_densities)[:max_rays]

    if ray_origins is not None:
        ray_origins = jnp.array(ray_origins)[:max_rays]

    num_rays, num_samples = sampling_points.shape[:2]

    # Plot world origin for reference
    plot_world_origin(ax, scale=1.0, alpha=0.5)

    # Plot ray origins if provided
    if ray_origins is not None:
        ax.scatter(
            ray_origins[:, 0],
            ray_origins[:, 1],
            ray_origins[:, 2],
            color="red",
            s=80,
            alpha=0.8,
            marker="o",
            edgecolors="black",
            linewidth=1,
            label="Ray Origins",
        )

        # Connect origins to first sampling points
        for ray_idx in range(num_rays):
            first_point = sampling_points[ray_idx, 0]
            ax.plot(
                [ray_origins[ray_idx, 0], first_point[0]],
                [ray_origins[ray_idx, 1], first_point[1]],
                [ray_origins[ray_idx, 2], first_point[2]],
                color="gray",
                alpha=0.3,
                linewidth=1,
                linestyle="--",
            )

    # Normalize colors for display
    colors_normalized = normalize_colors(predicted_colors)

    # Convert densities to alpha values
    alpha_values = convert_density_to_alpha(predicted_densities) * alpha_scale

    # Plot sampling points with colors and densities
    for ray_idx in range(num_rays):
        points = sampling_points[ray_idx]  # Shape: (num_samples, 3)
        colors = colors_normalized[ray_idx]  # Shape: (num_samples, 3)
        densities = predicted_densities[ray_idx]  # Shape: (num_samples,)
        alphas = alpha_values[ray_idx]  # Shape: (num_samples,)

        # Filter points by density threshold
        valid_mask = densities >= density_threshold
        if not jnp.any(valid_mask):
            continue

        points_valid = points[valid_mask]
        colors_valid = colors[valid_mask]
        alphas_valid = alphas[valid_mask]
        densities_valid = densities[valid_mask]

        # Plot points with colors and alpha based on density
        for i in range(len(points_valid)):
            point = points_valid[i]
            color = colors_valid[i]
            alpha = float(alphas_valid[i])
            density = float(densities_valid[i])

            # Point size based on density
            point_size = 20 + density * 50

            ax.scatter(
                [point[0]],
                [point[1]],
                [point[2]],
                color=color,
                s=point_size,
                alpha=alpha,
                marker="o",
                edgecolors="black",
                linewidth=0.5,
            )

        # Connect points along the ray
        if len(points_valid) > 1:
            ax.plot(
                points_valid[:, 0],
                points_valid[:, 1],
                points_valid[:, 2],
                color="gray",
                alpha=0.4,
                linewidth=1,
                linestyle=":",
            )

        # Add ray label
        if len(points_valid) > 0:
            label_point = points_valid[-1]
            ax.text(
                label_point[0],
                label_point[1],
                label_point[2],
                f"Ray {ray_idx}",
                fontsize=8,
                alpha=0.8,
            )

    # Create custom legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Ray Origins",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=6,
            label="Low Density",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="purple",
            markersize=12,
            label="High Density",
        ),
        plt.Line2D([0], [0], color="gray", linestyle=":", label="Ray Path"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Add density threshold info
    info_text = f"Density Threshold: {density_threshold:.3f}\\nShowing {num_rays} rays"
    ax.text2D(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Set bounds
    all_points = [sampling_points.reshape(-1, 3)]
    if ray_origins is not None:
        all_points.append(ray_origins)
    all_points.append(jnp.array([[0, 0, 0]]))  # World origin

    bounds = compute_plot_bounds(jnp.vstack(all_points), padding=0.15)
    set_axis_bounds(ax, bounds)

    if show_plot:
        plt.show()

    return fig, ax


def visualize_volume_rendering(
    sampling_points: jt.ArrayLike,
    predicted_colors: jt.ArrayLike,
    predicted_densities: jt.ArrayLike,
    rendered_colors: jt.ArrayLike | None = None,
    ray_origins: jt.ArrayLike | None = None,
    segment_length: float = 0.1,
    max_rays: int = 6,
    title: str = "Volume Rendering Visualization",
    figsize: tuple[int, int] = (16, 12),
    show_plot: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize volume rendering as alpha-blended segments along rays.

    @param sampling_points: 3D sampling points. Shape: (N_rays, N_samples, 3).
    @param predicted_colors: Predicted RGB colors. Shape: (N_rays, N_samples, 3).
    @param predicted_densities: Predicted densities. Shape: (N_rays, N_samples).
    @param rendered_colors: Final rendered colors per ray. Shape: (N_rays, 3).
    @param ray_origins: Optional ray origins. Shape: (N_rays, 3).
    @param segment_length: Length of each volume segment.
    @param max_rays: Maximum number of rays to visualize.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    # Convert inputs to arrays and limit rays
    sampling_points = jnp.array(sampling_points)[:max_rays]
    predicted_colors = jnp.array(predicted_colors)[:max_rays]
    predicted_densities = jnp.array(predicted_densities)[:max_rays]

    if rendered_colors is not None:
        rendered_colors = jnp.array(rendered_colors)[:max_rays]
    if ray_origins is not None:
        ray_origins = jnp.array(ray_origins)[:max_rays]

    num_rays, num_samples = sampling_points.shape[:2]

    # Plot world origin
    plot_world_origin(ax, scale=1.0, alpha=0.5)

    # Plot ray origins if provided
    if ray_origins is not None:
        ax.scatter(
            ray_origins[:, 0],
            ray_origins[:, 1],
            ray_origins[:, 2],
            color="red",
            s=100,
            alpha=0.8,
            marker="o",
            edgecolors="black",
            linewidth=1,
            label="Camera/Ray Origins",
        )

    # Normalize colors
    colors_normalized = normalize_colors(predicted_colors)

    # For each ray, draw volume segments
    for ray_idx in range(num_rays):
        points = sampling_points[ray_idx]  # Shape: (num_samples, 3)
        colors = colors_normalized[ray_idx]  # Shape: (num_samples, 3)
        densities = predicted_densities[ray_idx]  # Shape: (num_samples,)

        # Skip rays with no significant density
        if jnp.max(densities) < 0.01:
            continue

        # Calculate segment intervals
        if len(points) > 1:
            # Use actual distances between points
            distances = jnp.linalg.norm(jnp.diff(points, axis=0), axis=1)
            avg_distance = jnp.mean(distances)
        else:
            avg_distance = segment_length

        # Convert densities to alpha values
        alphas = convert_density_to_alpha(densities, default_distance=avg_distance)

        # Draw volume segments
        for i in range(len(points) - 1):
            if alphas[i] < 0.01:  # Skip transparent segments
                continue

            start_point = points[i]
            end_point = points[i + 1]
            color = colors[i]
            alpha = float(alphas[i])

            # Draw segment as a thick line with alpha
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color,
                alpha=alpha,
                linewidth=8,  # Thick line to represent volume
                solid_capstyle="round",
            )

            # Add density markers at points
            density_val = float(densities[i])
            marker_size = 10 + density_val * 40
            ax.scatter(
                [start_point[0]],
                [start_point[1]],
                [start_point[2]],
                color=color,
                s=marker_size,
                alpha=alpha * 0.7,
                marker="o",
                edgecolors="black",
                linewidth=0.3,
            )

        # Connect to ray origin if provided
        if ray_origins is not None:
            ax.plot(
                [ray_origins[ray_idx, 0], points[0, 0]],
                [ray_origins[ray_idx, 1], points[0, 1]],
                [ray_origins[ray_idx, 2], points[0, 2]],
                color="gray",
                alpha=0.3,
                linewidth=1,
                linestyle="--",
            )

        # Show final rendered color if provided
        if rendered_colors is not None:
            final_color = rendered_colors[ray_idx]
            # Place rendered color indicator at the end of the ray
            end_indicator = points[-1] + jnp.array([0.2, 0.2, 0.2])
            ax.scatter(
                [end_indicator[0]],
                [end_indicator[1]],
                [end_indicator[2]],
                color=final_color,
                s=100,
                alpha=0.9,
                marker="s",
                edgecolors="black",
                linewidth=2,
                label="Final Rendered Color" if ray_idx == 0 else "",
            )

            # Connect to last point
            ax.plot(
                [points[-1, 0], end_indicator[0]],
                [points[-1, 1], end_indicator[1]],
                [points[-1, 2], end_indicator[2]],
                color=final_color,
                alpha=0.5,
                linewidth=3,
            )

        # Add ray label
        mid_point = points[len(points) // 2]
        ax.text(
            mid_point[0],
            mid_point[1],
            mid_point[2] + 0.1,
            f"Ray {ray_idx}",
            fontsize=9,
            alpha=0.8,
            ha="center",
        )

    # Set bounds
    all_points = [sampling_points.reshape(-1, 3)]
    if ray_origins is not None:
        all_points.append(ray_origins)
    all_points.append(jnp.array([[0, 0, 0]]))  # World origin

    bounds = compute_plot_bounds(jnp.vstack(all_points), padding=0.2)
    set_axis_bounds(ax, bounds)

    # Create legend
    legend_elements = []
    if ray_origins is not None:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Ray Origins",
            )
        )
    legend_elements.extend(
        [
            plt.Line2D(
                [0], [0], color="blue", linewidth=8, alpha=0.7, label="Volume Segments"
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="purple",
                markersize=6,
                label="Density Points",
            ),
        ]
    )
    if rendered_colors is not None:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="green",
                markersize=10,
                label="Final Rendered Color",
            )
        )

    ax.legend(handles=legend_elements, loc="upper right")

    # Add info
    info_text = f"Volume Rendering\\n{num_rays} rays visualized\\nSegment length: {segment_length:.3f}"
    ax.text2D(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    if show_plot:
        plt.show()

    return fig, ax


def visualize_coarse_vs_fine_predictions(
    coarse_points: jt.ArrayLike,
    coarse_colors: jt.ArrayLike,
    coarse_densities: jt.ArrayLike,
    fine_points: jt.ArrayLike,
    fine_colors: jt.ArrayLike,
    fine_densities: jt.ArrayLike,
    ray_origins: jt.ArrayLike | None = None,
    max_rays: int = 4,
    title: str = "Coarse vs Fine Network Predictions",
    figsize: tuple[int, int] = (18, 12),
    show_plot: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare coarse and fine network predictions side by side.

    @param coarse_points: Coarse sampling points. Shape: (N_rays, N_coarse, 3).
    @param coarse_colors: Coarse predicted colors. Shape: (N_rays, N_coarse, 3).
    @param coarse_densities: Coarse predicted densities. Shape: (N_rays, N_coarse).
    @param fine_points: Fine sampling points. Shape: (N_rays, N_fine, 3).
    @param fine_colors: Fine predicted colors. Shape: (N_rays, N_fine, 3).
    @param fine_densities: Fine predicted densities. Shape: (N_rays, N_fine).
    @param ray_origins: Optional ray origins. Shape: (N_rays, 3).
    @param max_rays: Maximum number of rays to visualize.
    @param title: Plot title.
    @param figsize: Figure size in inches.
    @param show_plot: Whether to display the plot immediately.
    @return: Tuple of (figure, axes) objects.
    """
    fig, ax = setup_3d_plot(figsize, title)

    # Convert inputs and limit rays
    coarse_points = jnp.array(coarse_points)[:max_rays]
    coarse_colors = jnp.array(coarse_colors)[:max_rays]
    coarse_densities = jnp.array(coarse_densities)[:max_rays]
    fine_points = jnp.array(fine_points)[:max_rays]
    fine_colors = jnp.array(fine_colors)[:max_rays]
    fine_densities = jnp.array(fine_densities)[:max_rays]

    if ray_origins is not None:
        ray_origins = jnp.array(ray_origins)[:max_rays]

    num_rays = len(coarse_points)

    # Plot world origin
    plot_world_origin(ax, scale=1.0, alpha=0.4)

    # Plot ray origins if provided
    if ray_origins is not None:
        ax.scatter(
            ray_origins[:, 0],
            ray_origins[:, 1],
            ray_origins[:, 2],
            color="red",
            s=100,
            alpha=0.8,
            marker="o",
            edgecolors="black",
            linewidth=1,
            label="Ray Origins",
        )

    # Normalize colors
    coarse_colors_norm = normalize_colors(coarse_colors)
    fine_colors_norm = normalize_colors(fine_colors)

    # Convert densities to alpha
    coarse_alphas = convert_density_to_alpha(coarse_densities)
    fine_alphas = convert_density_to_alpha(fine_densities)

    # Plot coarse and fine predictions
    for ray_idx in range(num_rays):
        # Coarse predictions
        c_points = coarse_points[ray_idx]
        c_colors = coarse_colors_norm[ray_idx]
        c_densities = coarse_densities[ray_idx]
        c_alphas = coarse_alphas[ray_idx]

        # Fine predictions
        f_points = fine_points[ray_idx]
        f_colors = fine_colors_norm[ray_idx]
        f_densities = fine_densities[ray_idx]
        f_alphas = fine_alphas[ray_idx]

        # Plot coarse points
        coarse_mask = c_densities >= 0.01
        if jnp.any(coarse_mask):
            valid_c_points = c_points[coarse_mask]
            valid_c_colors = c_colors[coarse_mask]
            valid_c_alphas = c_alphas[coarse_mask]
            valid_c_densities = c_densities[coarse_mask]

            for i in range(len(valid_c_points)):
                point_size = 30 + float(valid_c_densities[i]) * 50
                ax.scatter(
                    [valid_c_points[i, 0]],
                    [valid_c_points[i, 1]],
                    [valid_c_points[i, 2]],
                    color=valid_c_colors[i],
                    s=point_size,
                    alpha=float(valid_c_alphas[i]),
                    marker="o",
                    edgecolors="blue",
                    linewidth=1.5,
                    label="Coarse Predictions" if ray_idx == 0 and i == 0 else "",
                )

            # Connect coarse points
            if len(valid_c_points) > 1:
                ax.plot(
                    valid_c_points[:, 0],
                    valid_c_points[:, 1],
                    valid_c_points[:, 2],
                    color="blue",
                    alpha=0.4,
                    linewidth=2,
                    linestyle="-",
                )

        # Plot fine points
        fine_mask = f_densities >= 0.01
        if jnp.any(fine_mask):
            valid_f_points = f_points[fine_mask]
            valid_f_colors = f_colors[fine_mask]
            valid_f_alphas = f_alphas[fine_mask]
            valid_f_densities = f_densities[fine_mask]

            for i in range(len(valid_f_points)):
                point_size = 20 + float(valid_f_densities[i]) * 40
                ax.scatter(
                    [valid_f_points[i, 0]],
                    [valid_f_points[i, 1]],
                    [valid_f_points[i, 2]],
                    color=valid_f_colors[i],
                    s=point_size,
                    alpha=float(valid_f_alphas[i]),
                    marker="^",
                    edgecolors="green",
                    linewidth=1,
                    label="Fine Predictions" if ray_idx == 0 and i == 0 else "",
                )

            # Connect fine points
            if len(valid_f_points) > 1:
                ax.plot(
                    valid_f_points[:, 0],
                    valid_f_points[:, 1],
                    valid_f_points[:, 2],
                    color="green",
                    alpha=0.4,
                    linewidth=1.5,
                    linestyle="--",
                )

        # Connect to ray origin if provided
        if ray_origins is not None:
            if len(c_points) > 0:
                ax.plot(
                    [ray_origins[ray_idx, 0], c_points[0, 0]],
                    [ray_origins[ray_idx, 1], c_points[0, 1]],
                    [ray_origins[ray_idx, 2], c_points[0, 2]],
                    color="gray",
                    alpha=0.3,
                    linewidth=1,
                    linestyle=":",
                )

        # Ray label
        if len(c_points) > 0:
            mid_point = c_points[len(c_points) // 2]
            ax.text(
                mid_point[0],
                mid_point[1],
                mid_point[2] + 0.15,
                f"Ray {ray_idx}",
                fontsize=9,
                ha="center",
                alpha=0.8,
            )

    # Set bounds
    all_points = [
        coarse_points.reshape(-1, 3),
        fine_points.reshape(-1, 3),
        jnp.array([[0, 0, 0]]),  # World origin
    ]
    if ray_origins is not None:
        all_points.append(ray_origins)

    bounds = compute_plot_bounds(jnp.vstack(all_points), padding=0.15)
    set_axis_bounds(ax, bounds)

    # Legend
    ax.legend(loc="upper right")

    # Info text
    info_text = (
        f"Comparison View\\n"
        f"{num_rays} rays shown\\n"
        f"Coarse: {coarse_points.shape[1]} samples\\n"
        f"Fine: {fine_points.shape[1]} samples"
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

    if show_plot:
        plt.show()

    return fig, ax
