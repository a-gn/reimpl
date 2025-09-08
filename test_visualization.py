#!/usr/bin/env python3
"""Simple test script for NeRF visualization tools.

Originally written by Claude Sonnet 4 on 2025/09/07
"""

import jax.numpy as jnp
import jax.random as rand
import matplotlib.pyplot as plt

# Test imports
try:
    from reimpl_a_gn.threed import (
        CameraParams,
        NeRFVisualizationData,
        NeRFVisualizer,
        PlotManager,
        create_quick_visualizer,
        extrinsic_matrix_from_pose,
        intrinsic_matrix_from_params,
        sample_rays_towards_pixels,
        visualize_cameras,
        visualize_ray_sampling,
    )

    print("✅ All visualization imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def create_test_camera():
    """Create a simple test camera."""
    # Camera looking down negative z axis from (0, 0, 2)
    extrinsic = extrinsic_matrix_from_pose(
        camera_origin_world=jnp.array([0.0, 0.0, 2.0, 1.0]),
        viewing_direction_world=jnp.array([0.0, 0.0, -1.0, 0.0]),
        up_direction_world=jnp.array([0.0, 1.0, 0.0, 0.0]),
    )

    intrinsic = intrinsic_matrix_from_params(
        focal_length=(100.0, 100.0), image_height=400, image_width=400
    )

    return CameraParams(extrinsic, intrinsic)


def test_basic_functionality():
    """Test basic visualization functionality."""
    print("\n🧪 Testing Basic Functionality")
    print("-" * 40)

    # Create test cameras
    cameras = []
    positions = [(0, 0, 2), (1, 1, 2), (-1, 0, 2)]

    for i, (x, y, z) in enumerate(positions):
        extrinsic = extrinsic_matrix_from_pose(
            camera_origin_world=jnp.array([float(x), float(y), float(z), 1.0]),
            viewing_direction_world=jnp.array([0.0, 0.0, -1.0, 0.0]),
            up_direction_world=jnp.array([0.0, 1.0, 0.0, 0.0]),
        )

        intrinsic = intrinsic_matrix_from_params(
            focal_length=(50.0, 50.0), image_height=200, image_width=200
        )

        cameras.append(CameraParams(extrinsic, intrinsic))

    print(f"Created {len(cameras)} test cameras")

    # Test camera visualization
    try:
        fig, ax = visualize_cameras(cameras, marked_camera=0, show_plot=False)
        print("✅ Camera visualization works")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Camera visualization failed: {e}")

    # Test ray sampling
    try:
        pixel_coords = jnp.array([[50, 50], [100, 100], [150, 150]])
        rays = sample_rays_towards_pixels(cameras[0], pixel_coords)

        fig, ax = visualize_ray_sampling(
            cameras[0], pixel_coords, rays, max_rays_to_show=3, show_plot=False
        )
        print("✅ Ray sampling visualization works")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Ray sampling visualization failed: {e}")

    # Test plot manager
    try:
        manager = PlotManager()
        fig, ax = manager.create_plot("test", "Test Plot")
        ax.scatter([0, 1, 2], [0, 1, 4], color="red")
        print("✅ PlotManager works")
        plt.close(fig)
    except Exception as e:
        print(f"❌ PlotManager failed: {e}")


def test_nerf_visualization():
    """Test NeRF-specific visualizations."""
    print("\n🎯 Testing NeRF Visualizations")
    print("-" * 40)

    # Create mock data
    num_rays = 3
    num_samples = 5

    # Mock sampling points
    sampling_points = rand.uniform(
        rand.key(1), (num_rays, num_samples, 3), minval=-1.0, maxval=1.0
    )

    # Mock colors and densities
    colors = rand.uniform(
        rand.key(2), (num_rays, num_samples, 3), minval=0.0, maxval=1.0
    )
    densities = rand.uniform(
        rand.key(3), (num_rays, num_samples), minval=0.0, maxval=2.0
    )

    # Test NeRF prediction visualization
    try:
        from reimpl_a_gn.threed import visualize_nerf_predictions

        fig, ax = visualize_nerf_predictions(
            sampling_points, colors, densities, max_rays=num_rays, show_plot=False
        )
        print("✅ NeRF predictions visualization works")
        plt.close(fig)
    except Exception as e:
        print(f"❌ NeRF predictions visualization failed: {e}")

    # Test volume rendering visualization
    try:
        from reimpl_a_gn.threed import visualize_volume_rendering

        fig, ax = visualize_volume_rendering(
            sampling_points, colors, densities, max_rays=num_rays, show_plot=False
        )
        print("✅ Volume rendering visualization works")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Volume rendering visualization failed: {e}")


def test_high_level_interface():
    """Test the high-level visualization interface."""
    print("\n🚀 Testing High-Level Interface")
    print("-" * 40)

    # Create test data
    camera = create_test_camera()
    pixel_coords = jnp.array([[50, 50], [100, 100]])
    rays = sample_rays_towards_pixels(camera, pixel_coords)

    # Mock sampling points and predictions
    coarse_points = rand.uniform(rand.key(10), (2, 4, 3), minval=-0.5, maxval=0.5)
    coarse_colors = rand.uniform(rand.key(11), (2, 4, 3), minval=0.0, maxval=1.0)
    coarse_densities = rand.uniform(rand.key(12), (2, 4), minval=0.0, maxval=1.0)

    # Test NeRFVisualizationData
    try:
        data = NeRFVisualizationData(
            cameras=[camera],
            pixel_coordinates=pixel_coords,
            rays=rays,
            coarse_sampling_points=coarse_points,
            coarse_colors=coarse_colors,
            coarse_densities=coarse_densities,
        )

        # Validate data
        issues = data.validate()
        if issues:
            print(f"⚠️ Data validation issues: {issues}")
        else:
            print("✅ NeRFVisualizationData validation works")
    except Exception as e:
        print(f"❌ NeRFVisualizationData failed: {e}")
        return

    # Test NeRFVisualizer
    try:
        visualizer = NeRFVisualizer()
        visualizer.load_data(data)

        # Test camera visualization
        fig, ax = visualizer.show_cameras()
        plt.close(fig)

        print("✅ NeRFVisualizer works")
    except Exception as e:
        print(f"❌ NeRFVisualizer failed: {e}")

    # Test quick visualizer
    try:
        quick_viz = create_quick_visualizer(
            cameras=[camera],
            pixel_coordinates=pixel_coords,
            rays=rays,
            coarse_points=coarse_points,
            coarse_colors=coarse_colors,
            coarse_densities=coarse_densities,
        )
        print("✅ create_quick_visualizer works")
    except Exception as e:
        print(f"❌ create_quick_visualizer failed: {e}")


def main():
    """Run all tests."""
    print("🧪 NeRF Visualization Tools - Quick Test")
    print("=" * 50)

    test_basic_functionality()
    test_nerf_visualization()
    test_high_level_interface()

    print("\n✨ Test completed!")
    print("\nTo see the full demo, run:")
    print("  python examples/visualization_demo.py")


if __name__ == "__main__":
    main()
