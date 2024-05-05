import jax.numpy as jnp


def sample_rays_for_image_render(
    camera_params: jnp.ndarray, image_height: int, image_width: int
) -> jnp.ndarray:
    """Sample parameters of rays through a pinhole camera, which can be used to render an image.

    @param camera_params Camera matrix.
    @param image_height Pixel row count of the image we want to render.
    @param image_width Pixel column count of the image we want to render.
    @return Ray parameters. Shape: (num_rays, 6). Second axis: x, y, z, dx, dy, dz.
    """
    raise NotImplementedError()


def sample_positions_along_rays(
    rays: jnp.ndarray, near_distance: float, far_distance: float, pos_per_ray: int
) -> jnp.ndarray:
    """Compute regular positions along a set of rays.

    @param rays Ray parameters. Shape: (num_rays, 6). Second axis: x, y, z, dx, dy, dz.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param pos_per_ray Number of positions to sample along each ray.
    @return Positions along rays. Shape: (num_rays, pos_per_ray, 3). Last axis: x, y, z.
    """
    raise NotImplementedError()


def render_image_from_rays(
    ray_features: jnp.ndarray, image_height: int, image_width: int
) -> jnp.ndarray:
    """Render an image from (R, G, B, sigma) predicted along a set of rays.

    We interpolate the input ray features to get values for each pixel's center.

    @param ray_features Array of colors and transparency predicted along rays. Shape: (num_rays, pos_per_ray, 7).
    Last axis: x, y, z, R, G, B, sigma.
    @return The rendered RGB image. Shape: (3, image_height, image_width).
    """
    raise NotImplementedError()
