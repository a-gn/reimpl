#!/usr/bin/env python3
"""Example script demonstrating NeRF visualization tools.

Originally written by Claude Sonnet 4 on 2025/09/07

This script shows how to use the comprehensive NeRF visualization system
with your existing data structures and network outputs.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as rand
import kagglehub
import matplotlib.pyplot as plt

from reimpl_a_gn.dataset.synthetic_nerf_dataset import load_synthetic_nerf_dataset
from reimpl_a_gn.threed import (
    NeRFVisualizationData,
    NeRFVisualizer,
    PlotManager,
    create_quick_visualizer,
    sample_coarse_mlp_inputs,
    sample_rays_towards_pixels,
    visualize_cameras,
    visualize_nerf_predictions,
    visualize_ray_sampling,
    visualize_volume_rendering,
)


def get_flower_dataset():
    """Download the dataset if needed, and return the path to the flower subfolder."""
    dataset_path = (
        Path(kagglehub.dataset_download("arenagrenade/llff-dataset-full"))
        / "nerf_llff_data"
        / "flower"
    )
    return load_synthetic_nerf_dataset(dataset_path)


def compute_pixel_coords(width: int = 20, height: int = 30) -> jnp.ndarray:
    """Compute pixel coordinates for a grid of pixels.

    @param width: Number of pixels in x direction.
    @param height: Number of pixels in y direction.
    @return: Pixel (x, y) coordinate array, shape (N, 2).
    """
    pixels_x = jnp.arange(-width // 2, width // 2)
    pixels_y = jnp.arange(-height // 2, height // 2)
    pixels_x, pixels_y = jnp.meshgrid(pixels_x, pixels_y)
    pixel_coords = jnp.stack([pixels_x, pixels_y], axis=1).reshape(-1, 2)
    return pixel_coords


def generate_mock_network_outputs(
    coarse_points: jnp.ndarray, fine_points: jnp.ndarray, prng_key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate mock network outputs for demonstration.

    @param coarse_points: Coarse sampling points. Shape: (N_rays, N_coarse, 3).
    @param fine_points: Fine sampling points. Shape: (N_rays, N_fine, 3).
    @param prng_key: Random key for generation.
    @return: Tuple of (coarse_colors, coarse_densities, fine_colors, fine_densities).
    """
    key1, key2, key3, key4 = rand.split(prng_key, 4)

    # Generate colors based on position (add some structure)
    coarse_colors = jnp.abs(coarse_points) * 0.5 + rand.uniform(
        key1, coarse_points.shape, minval=0.0, maxval=0.5
    )
    coarse_colors = jnp.clip(coarse_colors, 0.0, 1.0)

    fine_colors = jnp.abs(fine_points) * 0.6 + rand.uniform(
        key2, fine_points.shape, minval=0.0, maxval=0.4
    )
    fine_colors = jnp.clip(fine_colors, 0.0, 1.0)

    # Generate densities with some structure (higher near center)
    coarse_center_distances = jnp.linalg.norm(coarse_points, axis=2)
    coarse_densities = (
        jnp.exp(-coarse_center_distances * 0.5)
        + rand.uniform(key3, coarse_center_distances.shape) * 0.3
    )

    fine_center_distances = jnp.linalg.norm(fine_points, axis=2)
    fine_densities = (
        jnp.exp(-fine_center_distances * 0.4)
        + rand.uniform(key4, fine_center_distances.shape) * 0.2
    )

    return coarse_colors, coarse_densities, fine_colors, fine_densities


def demo_basic_visualizations():
    """Demonstrate basic visualization functions."""
    print("🎥 Demo: Basic Visualizations")
    print("=" * 50)

    # Load dataset
    print("Loading dataset...")
    dataset = get_flower_dataset()
    cameras = dataset.cameras[:3]  # Use first 3 cameras

    # 1. Camera visualization
    print("1. Visualizing cameras...")
    fig, ax = visualize_cameras(
        cameras,
        marked_camera=0,  # Highlight first camera
        title="Camera Positions (Basic Demo)",
        show_plot=False,
    )
    plt.show()

    # 2. Ray sampling visualization
    print("2. Visualizing ray sampling...")
    pixel_coords = compute_pixel_coords(10, 8)  # Small grid for demo
    rays = sample_rays_towards_pixels(cameras[0], pixel_coords)

    fig, ax = visualize_ray_sampling(
        cameras[0],
        pixel_coords[:5],  # Show first 5 rays only
        rays[:5],
        max_rays_to_show=5,
        title="Ray Sampling (Basic Demo)",
        show_plot=False,
    )
    plt.show()

    print("Basic visualizations demo complete!\n")


def demo_nerf_pipeline():
    """Demonstrate full NeRF pipeline visualization."""
    print("🧠 Demo: NeRF Pipeline Visualization")
    print("=" * 50)

    # Setup
    dataset = get_flower_dataset()
    camera = dataset.cameras[0]
    pixel_coords = compute_pixel_coords(8, 6)  # Small for demo
    rays = sample_rays_towards_pixels(camera, pixel_coords)

    # Generate sampling points
    prng_key = rand.key(42)
    prng_key, coarse_key, fine_key = rand.split(prng_key, 3)

    # Coarse sampling
    coarse_points = sample_coarse_mlp_inputs(
        rays[:10],
        near_distance=0.5,
        far_distance=3.0,
        bins_per_ray=8,
        prng_key=coarse_key,
    )

    # Fine sampling (mock - in practice this would use coarse network output)
    fine_points = sample_coarse_mlp_inputs(
        rays[:10],
        near_distance=0.8,
        far_distance=2.5,
        bins_per_ray=12,
        prng_key=fine_key,
    )

    # Generate mock network outputs
    prng_key, mock_key = rand.split(prng_key)
    coarse_colors, coarse_densities, fine_colors, fine_densities = (
        generate_mock_network_outputs(coarse_points, fine_points, mock_key)
    )

    # Mock final rendered colors (would come from volume rendering)
    rendered_colors = jnp.mean(fine_colors, axis=1)  # Simple average

    print("1. Coarse network predictions...")
    visualize_nerf_predictions(
        coarse_points,
        coarse_colors,
        coarse_densities,
        ray_origins=rays[:10, :3],
        max_rays=6,
        title="Coarse Network Predictions (Pipeline Demo)",
        show_plot=True,
    )

    print("2. Fine network predictions...")
    visualize_nerf_predictions(
        fine_points,
        fine_colors,
        fine_densities,
        ray_origins=rays[:10, :3],
        max_rays=6,
        title="Fine Network Predictions (Pipeline Demo)",
        show_plot=True,
    )

    print("3. Volume rendering...")
    visualize_volume_rendering(
        fine_points,
        fine_colors,
        fine_densities,
        rendered_colors=rendered_colors,
        ray_origins=rays[:10, :3],
        max_rays=4,
        title="Volume Rendering (Pipeline Demo)",
        show_plot=True,
    )

    print("NeRF pipeline demo complete!\n")


def demo_high_level_interface():
    """Demonstrate the high-level NeRF visualizer interface."""
    print("🚀 Demo: High-Level Interface")
    print("=" * 50)

    # Setup data
    dataset = get_flower_dataset()
    cameras = dataset.cameras[:2]
    camera = cameras[0]
    pixel_coords = compute_pixel_coords(6, 6)
    rays = sample_rays_towards_pixels(camera, pixel_coords[:15])

    # Generate mock data
    prng_key = rand.key(123)
    keys = rand.split(prng_key, 4)

    coarse_points = sample_coarse_mlp_inputs(
        rays, near_distance=0.4, far_distance=2.8, bins_per_ray=6, prng_key=keys[0]
    )
    fine_points = sample_coarse_mlp_inputs(
        rays, near_distance=0.6, far_distance=2.4, bins_per_ray=10, prng_key=keys[1]
    )

    coarse_colors, coarse_densities, fine_colors, fine_densities = (
        generate_mock_network_outputs(coarse_points, fine_points, keys[2])
    )
    rendered_colors = jnp.mean(fine_colors, axis=1)

    # Method 1: Using the data container and visualizer
    print("Method 1: Using NeRFVisualizationData + NeRFVisualizer")

    data = NeRFVisualizationData(
        cameras=cameras,
        marked_camera=0,
        pixel_coordinates=pixel_coords[:15],
        rays=rays,
        coarse_sampling_points=coarse_points,
        coarse_colors=coarse_colors,
        coarse_densities=coarse_densities,
        fine_sampling_points=fine_points,
        fine_colors=fine_colors,
        fine_densities=fine_densities,
        rendered_colors=rendered_colors,
    )

    visualizer = NeRFVisualizer(figsize=(14, 10))
    visualizer.load_data(data)

    print("Showing camera setup...")
    visualizer.show_cameras()
    plt.show()

    print("Showing coarse vs fine comparison...")
    visualizer.show_coarse_vs_fine(max_rays=3)
    plt.show()

    # Method 2: Quick visualizer
    print("\nMethod 2: Using create_quick_visualizer")

    quick_viz = create_quick_visualizer(
        cameras=cameras,
        pixel_coordinates=pixel_coords[:15],
        rays=rays,
        coarse_points=coarse_points,
        coarse_colors=coarse_colors,
        coarse_densities=coarse_densities,
        fine_points=fine_points,
        fine_colors=fine_colors,
        fine_densities=fine_densities,
        rendered_colors=rendered_colors,
        marked_camera=0,
    )

    print("Showing complete pipeline overview...")
    overview = quick_viz.show_pipeline_overview(max_rays=2)
    overview.show_all()

    print("High-level interface demo complete!\n")


def demo_interactive_exploration():
    """Demonstrate interactive exploration mode."""
    print("🔍 Demo: Interactive Exploration")
    print("=" * 50)

    # Create mock data
    dataset = get_flower_dataset()
    cameras = dataset.cameras[:2]
    pixel_coords = compute_pixel_coords(4, 4)
    rays = sample_rays_towards_pixels(cameras[0], pixel_coords[:8])

    prng_key = rand.key(456)
    keys = rand.split(prng_key, 3)

    coarse_points = sample_coarse_mlp_inputs(
        rays, near_distance=0.5, far_distance=2.5, bins_per_ray=5, prng_key=keys[0]
    )
    coarse_colors, coarse_densities, _, _ = generate_mock_network_outputs(
        coarse_points,
        coarse_points,
        keys[1],  # Use same points for simplicity
    )

    # Create visualizer
    visualizer = create_quick_visualizer(
        cameras=cameras,
        pixel_coordinates=pixel_coords[:8],
        rays=rays,
        coarse_points=coarse_points,
        coarse_colors=coarse_colors,
        coarse_densities=coarse_densities,
    )

    print("Starting interactive exploration...")
    print("Try commands like: cameras, rays, coarse, overview, quit")

    # Uncomment the line below to start interactive mode
    # visualizer.interactive_exploration()
    print("(Interactive mode disabled in demo - uncomment to enable)")

    print("Interactive exploration demo complete!\n")


def demo_plot_manager():
    """Demonstrate the PlotManager for handling multiple plots."""
    print("📊 Demo: Plot Manager")
    print("=" * 50)

    # Create plot manager
    manager = PlotManager(figsize=(10, 8))

    # Create multiple plots
    dataset = get_flower_dataset()
    cameras = dataset.cameras[:3]

    # Plot 1: All cameras
    fig, ax = manager.create_plot("all_cameras", "All Camera Positions")
    visualize_cameras(cameras, show_plot=False)

    # Plot 2: Marked camera
    fig, ax = manager.create_plot("marked_camera", "Camera 0 Highlighted")
    visualize_cameras(cameras, marked_camera=0, show_plot=False)

    # Plot 3: Different view
    fig, ax = manager.create_plot("camera_details", "Detailed Camera View")
    from reimpl_a_gn.threed import visualize_single_camera_detailed

    visualize_single_camera_detailed(cameras[0], show_plot=False)

    print("Created 3 plots. Showing all...")
    manager.show_all()

    print("Plot manager demo complete!\n")


def main():
    """Run all demonstration examples."""
    print("🎨 NeRF Visualization Tools Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive NeRF visualization system.")
    print("Each section demonstrates different aspects of the tools.\n")

    try:
        # Run all demos
        demo_basic_visualizations()
        demo_nerf_pipeline()
        demo_high_level_interface()
        demo_plot_manager()
        demo_interactive_exploration()

        print("🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("- Adapt the examples to your specific data")
        print("- Explore the interactive mode")
        print("- Try saving visualizations with save_visualizations()")
        print("- Customize colors, scales, and other parameters")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Make sure you have the required dataset and dependencies.")
        raise


if __name__ == "__main__":
    main()
