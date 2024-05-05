import jax
import jax.numpy as jnp
import jax.typing as jt
import jax.lax as lax


def sample_rays_for_image_render(
    camera_origin: jt.ArrayLike,
    camera_params: jt.ArrayLike,
    image_height: int,
    image_width: int,
) -> jax.Array:
    """Sample parameters of rays through a pinhole camera, which can be used to render an image.

    @param camera_origin Position of the camera. Shape: (4,).
    @param camera_params Camera matrix. Shape: (3, 4).
    @param image_height Pixel row count of the image we want to render.
    @param image_width Pixel column count of the image we want to render.
    @return Ray parameters. Shape: (image_height, image_width, 6). Second axis: x, y, z, dx, dy, dz.
    """
    image_to_world = jnp.linalg.inv(camera_params)
    ray_coords = jnp.zeros((image_height, image_width))
    for row_i in range(image_height):
        for col_i in range(image_width):
            image_point = jnp.array([row_i + 0.5, col_i + 0.5, 1], dtype=jnp.float32)
            world_point = image_to_world @ image_point
            ray_coords.at[row_i, col_i, 0:3].set(world_point)
            camera_to_wp = world_point - camera_origin
            ray_direction = camera_to_wp / (
                jnp.linalg.norm(camera_to_wp[:2] / camera_to_wp[2], axis=0) ** 1 / 2
            )
            ray_coords.at[row_i, col_i, 3:6].set(ray_direction)
    return ray_coords


def sample_positions_along_rays(
    rays: jt.ArrayLike, near_distance: float, far_distance: float, pos_per_ray: int
) -> jax.Array:
    """Compute regular positions along a set of rays.

    @param rays Ray parameters. Shape: (..., 6). Last axis: x, y, z, dx, dy, dz.
    @param near_distance Smallest distance from origin to sample at.
    @param far_distance Largest distance from origin to sample at.
    @param pos_per_ray Number of positions to sample along each ray.
    @return Positions along rays. Shape: (..., pos_per_ray, 3). Last axis: x, y, z.
    """
    rays = jnp.array(rays)
    result = jnp.zeros(list(rays.shape[:-1]) + [pos_per_ray, 3], dtype=jnp.float32)
    ray_origins = rays[..., :3]
    ray_directions = rays[..., 3:]
    norm_ray_directions = ray_directions / (
        jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2
    )
    distance_interval = (far_distance - near_distance) / pos_per_ray
    for pos_i in range(pos_per_ray + 1):
        result[..., pos_i, :] = ray_origins + norm_ray_directions * (
            near_distance + pos_i * distance_interval
        )
    return result


def blend_ray_features(ray_features: jt.ArrayLike) -> jax.Array:
    """Compute one color for each ray.

    @param ray_features Coordinates, color, and transparency sampled along rays. Shape: (num_rays, pos_per_ray, 7).
    Second axis: x, y, z, R, G, B, sigma.
    @return One color per ray. Shape: (num_rays, ..., 3). Last axis: R, G, B.
    """
    raise NotImplementedError()


@jax.jit
def render_image_from_rays(
    ray_features: jt.ArrayLike, image_height: int, image_width: int
) -> jax.Array:
    """Render an image from (R, G, B, sigma) predicted along a set of rays.

    We interpolate the input ray features to get values for each pixel's center.

    @param ray_features Array of colors and transparency predicted along rays. Shape: (num_rays, pos_per_ray, 7).
    Last axis: x, y, z, R, G, B, sigma.
    @return The rendered RGB image. Shape: (3, image_height, image_width).
    """
    color_per_ray = blend_ray_features(ray_features)
    final_image = jnp.zeros((3, image_height, image_width), dtype=jnp.float32)
