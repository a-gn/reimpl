"""3D graphics and NeRF implementation modules.

Originally written by Claude Sonnet 4 on 2025/09/07
"""

from .nerf import (
    CoarseMLP,
    FineMLP,
    blend_ray_features_with_nerf_paper_method,
    compute_fine_sampling_distribution,
    compute_nerf_positional_encoding,
    compute_rays_in_world_frame,
    sample_coarse_mlp_inputs,
    sample_regular_positions_along_rays,
)
from .plotting import make_non_homogeneous, plot_cameras
from .rendering import (
    CameraParams,
    extrinsic_matrix_from_pose,
    from_homogeneous,
    intrinsic_matrix_from_params,
    norm_eucl_3d,
    sample_rays_towards_pixels,
    to_homogeneous_points,
    to_homogeneous_vectors,
)
from .visualization import *

__all__ = [
    # Core NeRF
    "CoarseMLP",
    "FineMLP",
    "blend_ray_features_with_nerf_paper_method",
    "compute_fine_sampling_distribution",
    "compute_nerf_positional_encoding",
    "compute_rays_in_world_frame",
    "sample_coarse_mlp_inputs",
    "sample_regular_positions_along_rays",
    # Rendering
    "CameraParams",
    "extrinsic_matrix_from_pose",
    "from_homogeneous",
    "intrinsic_matrix_from_params",
    "norm_eucl_3d",
    "sample_rays_towards_pixels",
    "to_homogeneous_points",
    "to_homogeneous_vectors",
    # Legacy plotting
    "make_non_homogeneous",
    "plot_cameras",
    # Camera visualization
    "create_camera_comparison_plot",
    "visualize_cameras",
    "visualize_single_camera_detailed",
    # Ray visualization
    "visualize_pixel_grid",
    "visualize_ray_comparison",
    "visualize_ray_sampling",
    # NeRF prediction visualization
    "visualize_coarse_vs_fine_predictions",
    "visualize_nerf_predictions",
    "visualize_volume_rendering",
    # High-level interface
    "NeRFVisualizationData",
    "NeRFVisualizer",
    "create_quick_visualizer",
    # Utilities
    "PlotManager",
    "compute_plot_bounds",
    "convert_density_to_alpha",
    "create_custom_colormap",
    "normalize_colors",
    "setup_3d_plot",
]
