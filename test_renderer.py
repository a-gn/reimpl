import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy
import jax
import jax.typing as jt

from reimpl_a_gn.threed.rendering import (
    sample_rays_towards_all_pixels,
    sample_regular_positions_along_rays,
    sample_nerf_rendering_positions_along_rays,
    CameraParams,
    compute_nerf_positional_encoding,
)


def plot_pos_encoding():
    # print(compute_nerf_positional_encoding(jnp.array([[2, 3, 4.5, 0.5, 1.0, 0.3]]), 4))

    plot_min_val, plot_max_val = -1.3, 1.3
    plot_sample_count = 1000
    component_count = 5
    sample_plot_points = jnp.array([-1.0, -0.7, -0.3, -0.05, 0.2, 0.5, 0.9])

    fig, axes = plt.subplots(5, 2)
    fig.set_size_inches(12, 12)
    fig.suptitle(
        f"NeRF positional encoding with {component_count} components\n"
        f"(encoding values: {', '.join([str(x) for x in sample_plot_points])})",
        fontsize=16,
    )

    base_plot_points = jnp.linspace(plot_min_val, plot_max_val, plot_sample_count)
    for power_of_two in range(component_count):

        def f1(z):
            return jnp.sin(jnp.pow(2, power_of_two) * jnp.pi * z)

        def f2(z):
            return jnp.cos(jnp.pow(2, power_of_two) * jnp.pi * z)

        # plot trig functions
        f1_plot_vals = f1(base_plot_points)
        f2_plot_vals = f2(base_plot_points)
        axes[power_of_two, 0].plot(base_plot_points, f1_plot_vals)
        axes[power_of_two, 1].plot(base_plot_points, f2_plot_vals)

        axes[power_of_two, 0].set_xticks(sample_plot_points)
        axes[power_of_two, 1].set_xticks(sample_plot_points)
        axes[power_of_two, 0].vlines(sample_plot_points, -1.5, 1.5, color="orange")
        axes[power_of_two, 1].vlines(sample_plot_points, -1.5, 1.5, color="orange")

        sample_f1_values = f1(sample_plot_points)
        sample_f2_values = f2(sample_plot_points)
        axes[power_of_two, 0].scatter(sample_plot_points, sample_f1_values, color="red")
        axes[power_of_two, 1].scatter(sample_plot_points, sample_f2_values, color="red")

        axes[power_of_two, 0].set_title(f"sin(2^{power_of_two} * pi * x)", loc="left")
        axes[power_of_two, 1].set_title(f"cos(2^{power_of_two} * pi * x)", loc="right")

    fig.tight_layout(pad=0.1)
    plt.show()

    y = jnp.array([[2, 3, 4.5, 0.5, 1.0, 0.3]])
    components = 3
    result = jnp.zeros(list(y.shape[:-1]) + [6, 2 * components], dtype=float)
    for power_of_two in range(components):
        result = result.at[..., power_of_two * 2].set(
            jnp.sin(jnp.pow(2, power_of_two) * jnp.pi * y)
        )
        result = result.at[..., power_of_two * 2 + 1].set(
            jnp.cos(jnp.pow(2, power_of_two) * jnp.pi * y)
        )

    y = jnp.array([[0.4, 0.6, 0.1, 0.5, 0.3, 0.8]], dtype=float)
    pe = compute_nerf_positional_encoding(y, 12)
    print(pe)


# plot_pos_encoding()


CAMERA_FRAME_ORIGIN = jnp.array([3.0, 2.5, 4.0, 1.0])
CAMERA_FRAME_DIRECTIONS = jnp.array(
    [[-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.0, 0.5]]
)
CAMERA_TO_WORLD_MATRIX = jnp.zeros((4, 4))
CAMERA_TO_WORLD_MATRIX = CAMERA_TO_WORLD_MATRIX.at[:3, :3].set(
    CAMERA_FRAME_DIRECTIONS.transpose()
)
CAMERA_TO_WORLD_MATRIX = CAMERA_TO_WORLD_MATRIX.at[:, 3].set(CAMERA_FRAME_ORIGIN)
WORLD_TO_CAMERA_MATRIX = jnp.array(numpy.linalg.inv(CAMERA_TO_WORLD_MATRIX))
CAMERA_INTRINSICS = jnp.array([[2.0, 0, 6], [0, 2.0, 6], [0, 0, 1]])


def plot_coordinate_systems():
    def plot_basis(origin: jt.ArrayLike, dxdydz: jt.ArrayLike, plt_axis):
        """Plot quivers representing an orthonormal basis in 3D.

        @param origin Origin of the coordinate system in world coordinates.
        @param dxdydz Mutually orthogonal vectors representing the axes. Shape: (3, 3).
        Dimensions: vector, then vector coordinates.
        """

        origin = jnp.array(origin)
        assert origin.shape == (3,)
        dxdydz = jnp.array(dxdydz)
        assert dxdydz.shape == (3, 3)
        for direction in range(3):
            plt_axis.quiver(
                origin[0],
                origin[1],
                origin[2],
                dxdydz[direction, 0],
                dxdydz[direction, 1],
                dxdydz[direction, 2],
                normalize=True,
            )

    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plot_basis(CAMERA_FRAME_ORIGIN, CAMERA_FRAME_DIRECTIONS, ax)
    plt.show()


# plot_coordinate_systems()


def plot_sampled_rays():
    camera_params = CameraParams(
        WORLD_TO_CAMERA_MATRIX,
        CAMERA_INTRINSICS,
        4.0,
    )

    print(camera_params.image_points_to_camera(jnp.array([[0.0, 0.0]])))

    all_x, all_y = jnp.meshgrid(jnp.arange(-2, 3, 0.7), jnp.arange(-5, 5, 0.5))
    all_grid_points = jnp.stack([all_x, all_y], axis=-1).reshape(-1, 2)

    all_grid_points_camera_frame = camera_params.image_points_to_camera(all_grid_points)
    all_rays_towards_grid_points = jnp.concatenate(
        [
            jnp.zeros([all_grid_points_camera_frame.shape[0], 3]),
            all_grid_points_camera_frame[:, :3],
        ],
        1,
    )

    def plot_pixels(fig: plt.Figure, ax: plt.Axes, points: jnp.ndarray):  # type: ignore
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    regularly_sampled_rays = sample_regular_positions_along_rays(
        all_rays_towards_grid_points, 0.5, 3.0, 3
    )
    print(regularly_sampled_rays)
    fig_rays, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    ax_rays_regular, ax_rays_nerf = axes

    ax_rays_regular.set_title("rays towards pixels, regular sampling")
    ax_rays_regular.text(1, 0, 0, "x-axis", "x")  # type: ignore
    ax_rays_regular.text(0, 1, 0, "y-axis", "y")  # type: ignore
    ax_rays_regular.text(0, 0, 1, "z-axis", "z")  # type: ignore
    flat_positions = regularly_sampled_rays.reshape(-1, 3)
    ax_rays_regular.scatter(
        flat_positions[:, 0],
        flat_positions[:, 1],
        flat_positions[:, 2],
    )
    plot_pixels(fig_rays, ax_rays_regular, all_grid_points_camera_frame)

    nerf_sampled_rays = sample_nerf_rendering_positions_along_rays(
        all_rays_towards_grid_points, 0.5, 3.0, 3, jax.random.PRNGKey(0)
    )
    ax_rays_nerf.set_title("rays towards pixels, NeRF (bins) sampling")
    ax_rays_nerf.text(1, 0, 0, "x-axis", "x")  # type: ignore
    ax_rays_nerf.text(0, 1, 0, "y-axis", "y")  # type: ignore
    ax_rays_nerf.text(0, 0, 1, "z-axis", "z")  # type: ignore
    flat_positions = nerf_sampled_rays.reshape(-1, 3)
    ax_rays_nerf.scatter(
        flat_positions[:, 0],
        flat_positions[:, 1],
        flat_positions[:, 2],
    )
    plot_pixels(fig_rays, ax_rays_nerf, all_grid_points_camera_frame)

    plt.show()


plot_sampled_rays()