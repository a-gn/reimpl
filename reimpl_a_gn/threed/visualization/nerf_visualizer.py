"""High-level interactive NeRF visualization interface.

Originally written by Claude Sonnet 4 on 2025/09/07
"""

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import jax.typing as jt
import matplotlib.pyplot as plt

from ..rendering import CameraParams
from .camera_visualization import (
    create_camera_comparison_plot,
    visualize_cameras,
)
from .nerf_visualization import (
    visualize_coarse_vs_fine_predictions,
    visualize_nerf_predictions,
    visualize_volume_rendering,
)
from .ray_visualization import (
    visualize_ray_sampling,
)
from .visualization_utils import PlotManager


@dataclass
class NeRFVisualizationData:
    """Container for NeRF visualization data.

    Originally written by Claude Sonnet 4 on 2025/09/07
    """

    # Camera and scene data
    cameras: Sequence[CameraParams] | None = None
    marked_camera: int | None = None

    # Pixel and ray data
    pixel_coordinates: jt.ArrayLike | None = None
    rays: jt.ArrayLike | None = None

    # Sampling data
    coarse_sampling_points: jt.ArrayLike | None = None
    fine_sampling_points: jt.ArrayLike | None = None

    # Network predictions
    coarse_colors: jt.ArrayLike | None = None
    coarse_densities: jt.ArrayLike | None = None
    fine_colors: jt.ArrayLike | None = None
    fine_densities: jt.ArrayLike | None = None

    # Final rendering results
    rendered_colors: jt.ArrayLike | None = None

    def validate(self) -> list[str]:
        """Validate the data and return list of issues.

        @return: List of validation error messages.
        """
        issues = []

        if self.pixel_coordinates is not None and self.rays is not None:
            pixel_coords = jnp.array(self.pixel_coordinates)
            rays = jnp.array(self.rays)
            if pixel_coords.shape[0] != rays.shape[0]:
                issues.append(
                    "Pixel coordinates and rays must have same number of entries"
                )

        if self.coarse_sampling_points is not None and self.coarse_colors is not None:
            points = jnp.array(self.coarse_sampling_points)
            colors = jnp.array(self.coarse_colors)
            if points.shape[:2] != colors.shape[:2]:
                issues.append("Coarse sampling points and colors must have same shape")

        if (
            self.coarse_sampling_points is not None
            and self.coarse_densities is not None
        ):
            points = jnp.array(self.coarse_sampling_points)
            densities = jnp.array(self.coarse_densities)
            if points.shape[:2] != densities.shape:
                issues.append(
                    "Coarse sampling points and densities must have compatible shapes"
                )

        return issues


class NeRFVisualizer:
    """High-level interface for NeRF visualization.

    Originally written by Claude Sonnet 4 on 2025/09/07
    """

    def __init__(self, figsize: tuple[int, int] = (15, 12)):
        """Initialize the NeRF visualizer.

        @param figsize: Default figure size for plots.
        """
        self.plot_manager = PlotManager(figsize)
        self.data: NeRFVisualizationData | None = None

    def load_data(self, data: NeRFVisualizationData) -> None:
        """Load visualization data.

        @param data: NeRF visualization data container.
        @raises ValueError: If data validation fails.
        """
        issues = data.validate()
        if issues:
            raise ValueError(f"Data validation failed: {', '.join(issues)}")

        self.data = data

    def show_cameras(
        self,
        show_world_axes: bool = True,
        camera_scale: float = 1.0,
        world_scale: float = 2.0,
        create_individual_views: bool = False,
    ) -> tuple[plt.Figure, plt.Axes] | PlotManager:
        """Visualize camera positions and orientations.

        @param show_world_axes: Whether to show world coordinate system.
        @param camera_scale: Scale factor for camera axes.
        @param world_scale: Scale factor for world axes.
        @param create_individual_views: Whether to create individual camera views.
        @return: Figure and axes, or PlotManager for individual views.
        @raises ValueError: If no camera data is loaded.
        """
        if self.data is None or self.data.cameras is None:
            raise ValueError("No camera data loaded")

        if create_individual_views:
            return create_camera_comparison_plot(
                self.data.cameras, figsize=self.plot_manager.figsize, show_plot=False
            )
        else:
            return visualize_cameras(
                self.data.cameras,
                marked_camera=self.data.marked_camera,
                show_world_axes=show_world_axes,
                camera_scale=camera_scale,
                world_scale=world_scale,
                show_plot=False,
            )

    def show_ray_sampling(
        self,
        max_rays: int = 10,
        ray_length: float = 5.0,
        show_sampling_points: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize ray sampling from camera through pixels.

        @param max_rays: Maximum number of rays to show.
        @param ray_length: Length of ray arrows.
        @param show_sampling_points: Whether to show sampling points along rays.
        @return: Figure and axes objects.
        @raises ValueError: If required data is not loaded.
        """
        if (
            self.data is None
            or self.data.pixel_coordinates is None
            or self.data.rays is None
        ):
            raise ValueError("Pixel coordinates and rays data required")

        # Use first camera as reference if available
        camera = None
        if self.data.cameras is not None:
            camera = self.data.cameras[0]

        sampling_points = None
        if show_sampling_points and self.data.coarse_sampling_points is not None:
            sampling_points = self.data.coarse_sampling_points

        return visualize_ray_sampling(
            camera,
            self.data.pixel_coordinates,
            self.data.rays,
            sampling_points=sampling_points,
            max_rays_to_show=max_rays,
            ray_length=ray_length,
            show_plot=False,
        )

    def show_coarse_predictions(
        self,
        max_rays: int = 8,
        density_threshold: float = 0.1,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize coarse network predictions.

        @param max_rays: Maximum number of rays to visualize.
        @param density_threshold: Minimum density to display.
        @return: Figure and axes objects.
        @raises ValueError: If required data is not loaded.
        """
        if (
            self.data is None
            or self.data.coarse_sampling_points is None
            or self.data.coarse_colors is None
            or self.data.coarse_densities is None
        ):
            raise ValueError("Coarse network prediction data required")

        # Get ray origins if available
        ray_origins = None
        if self.data.rays is not None:
            rays = jnp.array(self.data.rays)
            ray_origins = rays[:, :3]  # First 3 components are origins

        return visualize_nerf_predictions(
            self.data.coarse_sampling_points,
            self.data.coarse_colors,
            self.data.coarse_densities,
            ray_origins=ray_origins,
            max_rays=max_rays,
            density_threshold=density_threshold,
            title="Coarse Network Predictions",
            show_plot=False,
        )

    def show_fine_predictions(
        self,
        max_rays: int = 8,
        density_threshold: float = 0.1,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize fine network predictions.

        @param max_rays: Maximum number of rays to visualize.
        @param density_threshold: Minimum density to display.
        @return: Figure and axes objects.
        @raises ValueError: If required data is not loaded.
        """
        if (
            self.data is None
            or self.data.fine_sampling_points is None
            or self.data.fine_colors is None
            or self.data.fine_densities is None
        ):
            raise ValueError("Fine network prediction data required")

        # Get ray origins if available
        ray_origins = None
        if self.data.rays is not None:
            rays = jnp.array(self.data.rays)
            ray_origins = rays[:, :3]

        return visualize_nerf_predictions(
            self.data.fine_sampling_points,
            self.data.fine_colors,
            self.data.fine_densities,
            ray_origins=ray_origins,
            max_rays=max_rays,
            density_threshold=density_threshold,
            title="Fine Network Predictions",
            show_plot=False,
        )

    def show_coarse_vs_fine(self, max_rays: int = 4) -> tuple[plt.Figure, plt.Axes]:
        """Compare coarse and fine network predictions.

        @param max_rays: Maximum number of rays to visualize.
        @return: Figure and axes objects.
        @raises ValueError: If required data is not loaded.
        """
        if self.data is None or any(
            getattr(self.data, attr) is None
            for attr in [
                "coarse_sampling_points",
                "coarse_colors",
                "coarse_densities",
                "fine_sampling_points",
                "fine_colors",
                "fine_densities",
            ]
        ):
            raise ValueError("Both coarse and fine network prediction data required")

        # Get ray origins if available
        ray_origins = None
        if self.data.rays is not None:
            rays = jnp.array(self.data.rays)
            ray_origins = rays[:, :3]

        return visualize_coarse_vs_fine_predictions(
            self.data.coarse_sampling_points,
            self.data.coarse_colors,
            self.data.coarse_densities,
            self.data.fine_sampling_points,
            self.data.fine_colors,
            self.data.fine_densities,
            ray_origins=ray_origins,
            max_rays=max_rays,
            show_plot=False,
        )

    def show_volume_rendering(
        self,
        max_rays: int = 6,
        segment_length: float = 0.1,
        use_fine_network: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize volume rendering process.

        @param max_rays: Maximum number of rays to visualize.
        @param segment_length: Length of volume segments.
        @param use_fine_network: Whether to use fine network data (if available).
        @return: Figure and axes objects.
        @raises ValueError: If required data is not loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Choose which network data to use
        if (
            use_fine_network
            and self.data.fine_sampling_points is not None
            and self.data.fine_colors is not None
            and self.data.fine_densities is not None
        ):
            points = self.data.fine_sampling_points
            colors = self.data.fine_colors
            densities = self.data.fine_densities
            title = "Volume Rendering (Fine Network)"
        elif (
            self.data.coarse_sampling_points is not None
            and self.data.coarse_colors is not None
            and self.data.coarse_densities is not None
        ):
            points = self.data.coarse_sampling_points
            colors = self.data.coarse_colors
            densities = self.data.coarse_densities
            title = "Volume Rendering (Coarse Network)"
        else:
            raise ValueError("Network prediction data required for volume rendering")

        # Get ray origins and rendered colors
        ray_origins = None
        if self.data.rays is not None:
            rays = jnp.array(self.data.rays)
            ray_origins = rays[:, :3]

        return visualize_volume_rendering(
            points,
            colors,
            densities,
            rendered_colors=self.data.rendered_colors,
            ray_origins=ray_origins,
            segment_length=segment_length,
            max_rays=max_rays,
            title=title,
            show_plot=False,
        )

    def show_pipeline_overview(
        self, max_rays: int = 3, figsize_per_plot: tuple[int, int] = (12, 8)
    ) -> PlotManager:
        """Show complete NeRF pipeline overview with multiple plots.

        @param max_rays: Maximum rays per visualization.
        @param figsize_per_plot: Figure size for each subplot.
        @return: PlotManager with all pipeline visualizations.
        """
        overview_manager = PlotManager(figsize_per_plot)

        # 1. Camera setup
        if self.data is not None and self.data.cameras is not None:
            fig, ax = overview_manager.create_plot("cameras", "Camera Setup")
            visualize_cameras(
                self.data.cameras,
                marked_camera=self.data.marked_camera,
                show_plot=False,
            )

        # 2. Ray sampling
        if (
            self.data is not None
            and self.data.pixel_coordinates is not None
            and self.data.rays is not None
        ):
            fig, ax = overview_manager.create_plot("rays", "Ray Sampling")
            camera = self.data.cameras[0] if self.data.cameras else None
            visualize_ray_sampling(
                camera,
                self.data.pixel_coordinates,
                self.data.rays,
                sampling_points=self.data.coarse_sampling_points,
                max_rays_to_show=max_rays,
                show_plot=False,
            )

        # 3. Coarse predictions
        if self.data is not None and all(
            getattr(self.data, attr) is not None
            for attr in ["coarse_sampling_points", "coarse_colors", "coarse_densities"]
        ):
            fig, ax = overview_manager.create_plot("coarse", "Coarse Network")
            self.show_coarse_predictions(max_rays=max_rays)

        # 4. Fine predictions (if available)
        if self.data is not None and all(
            getattr(self.data, attr) is not None
            for attr in ["fine_sampling_points", "fine_colors", "fine_densities"]
        ):
            fig, ax = overview_manager.create_plot("fine", "Fine Network")
            self.show_fine_predictions(max_rays=max_rays)

        # 5. Volume rendering
        try:
            fig, ax = overview_manager.create_plot("volume", "Volume Rendering")
            self.show_volume_rendering(max_rays=max_rays)
        except ValueError:
            pass  # Skip if data not available

        return overview_manager

    def interactive_exploration(self) -> None:
        """Start an interactive exploration session.

        Provides a simple command-line interface for exploring the data.
        """
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return

        print("NeRF Interactive Exploration")
        print("=" * 40)
        print("Available commands:")
        print("  cameras    - Show camera positions")
        print("  rays       - Show ray sampling")
        print("  coarse     - Show coarse predictions")
        print("  fine       - Show fine predictions (if available)")
        print("  compare    - Compare coarse vs fine")
        print("  volume     - Show volume rendering")
        print("  overview   - Show complete pipeline")
        print("  quit       - Exit")
        print()

        while True:
            try:
                command = input("Enter command: ").strip().lower()

                if command in ["q", "quit", "exit"]:
                    break
                elif command == "cameras":
                    self.show_cameras()
                    plt.show()
                elif command == "rays":
                    self.show_ray_sampling()
                    plt.show()
                elif command == "coarse":
                    try:
                        self.show_coarse_predictions()
                        plt.show()
                    except ValueError as e:
                        print(f"Error: {e}")
                elif command == "fine":
                    try:
                        self.show_fine_predictions()
                        plt.show()
                    except ValueError as e:
                        print(f"Error: {e}")
                elif command == "compare":
                    try:
                        self.show_coarse_vs_fine()
                        plt.show()
                    except ValueError as e:
                        print(f"Error: {e}")
                elif command == "volume":
                    try:
                        self.show_volume_rendering()
                        plt.show()
                    except ValueError as e:
                        print(f"Error: {e}")
                elif command == "overview":
                    manager = self.show_pipeline_overview()
                    manager.show_all()
                else:
                    print(f"Unknown command: {command}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def save_visualizations(
        self,
        output_dir: str = "nerf_visualizations",
        formats: Sequence[str] = ("png",),
        dpi: int = 300,
    ) -> None:
        """Save all available visualizations to files.

        @param output_dir: Directory to save visualizations.
        @param formats: File formats to save (e.g., 'png', 'pdf', 'svg').
        @param dpi: Resolution for raster formats.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        visualizations_to_save = []

        # Camera visualization
        if self.data is not None and self.data.cameras is not None:
            try:
                fig, _ = self.show_cameras()
                visualizations_to_save.append((fig, "cameras"))
            except Exception as e:
                print(f"Skipping cameras visualization: {e}")

        # Ray sampling
        if (
            self.data is not None
            and self.data.pixel_coordinates is not None
            and self.data.rays is not None
        ):
            try:
                fig, _ = self.show_ray_sampling()
                visualizations_to_save.append((fig, "ray_sampling"))
            except Exception as e:
                print(f"Skipping ray sampling visualization: {e}")

        # Network predictions
        for network_type in ["coarse", "fine"]:
            try:
                if network_type == "coarse":
                    fig, _ = self.show_coarse_predictions()
                else:
                    fig, _ = self.show_fine_predictions()
                visualizations_to_save.append((fig, f"{network_type}_predictions"))
            except Exception as e:
                print(f"Skipping {network_type} predictions visualization: {e}")

        # Volume rendering
        try:
            fig, _ = self.show_volume_rendering()
            visualizations_to_save.append((fig, "volume_rendering"))
        except Exception as e:
            print(f"Skipping volume rendering visualization: {e}")

        # Save all figures
        for fig, name in visualizations_to_save:
            for fmt in formats:
                filepath = os.path.join(output_dir, f"{name}.{fmt}")
                fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
                print(f"Saved {filepath}")

        print(f"Saved {len(visualizations_to_save)} visualizations to {output_dir}")


def create_quick_visualizer(
    cameras: Sequence[CameraParams] | None = None,
    pixel_coordinates: jt.ArrayLike | None = None,
    rays: jt.ArrayLike | None = None,
    coarse_points: jt.ArrayLike | None = None,
    coarse_colors: jt.ArrayLike | None = None,
    coarse_densities: jt.ArrayLike | None = None,
    fine_points: jt.ArrayLike | None = None,
    fine_colors: jt.ArrayLike | None = None,
    fine_densities: jt.ArrayLike | None = None,
    rendered_colors: jt.ArrayLike | None = None,
    marked_camera: int | None = None,
) -> NeRFVisualizer:
    """Create a NeRF visualizer with data in one step.

    @param cameras: Camera parameters.
    @param pixel_coordinates: Pixel coordinates.
    @param rays: Ray parameters.
    @param coarse_points: Coarse sampling points.
    @param coarse_colors: Coarse predicted colors.
    @param coarse_densities: Coarse predicted densities.
    @param fine_points: Fine sampling points.
    @param fine_colors: Fine predicted colors.
    @param fine_densities: Fine predicted densities.
    @param rendered_colors: Final rendered colors.
    @param marked_camera: Index of camera to highlight.
    @return: Configured NeRF visualizer.
    """
    data = NeRFVisualizationData(
        cameras=cameras,
        marked_camera=marked_camera,
        pixel_coordinates=pixel_coordinates,
        rays=rays,
        coarse_sampling_points=coarse_points,
        coarse_colors=coarse_colors,
        coarse_densities=coarse_densities,
        fine_sampling_points=fine_points,
        fine_colors=fine_colors,
        fine_densities=fine_densities,
        rendered_colors=rendered_colors,
    )

    visualizer = NeRFVisualizer()
    visualizer.load_data(data)
    return visualizer
