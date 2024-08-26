import jax
import jax.numpy as jnp
import jax.typing as jt
import jax.lax as lax


def get_ray_to_image_coordinates(
    camera_params: jax.Array, pixel_row: float, pixel_col: float, pixel_size_x: float, pixel_size_y: float
) -> jax.Array:
    """Compute the direction of a ray from the camera origin to a point in the image.

    Assumes that the camera is at the origin and looks along the z-axis.

    @param camera_params Camera matrix. Shape: (3, 4).
    @param pixel_row Row index of the pixel. Can be a float to represent subpixel positions.
    @param pixel_col Column index of the pixel. Can be a float to represent subpixel positions.
    @param pixel_size_x Width of a pixel in the image plane.
    @param pixel_size_y Height of a pixel in the image plane.
    @return Unit vector in the direction of the ray from camera origin to a point in the image plane.
    Shape: (image_height, image_width, 3). Last axis: x, y, z.
    """
    camera_params = jnp.array(camera_params)
    focal_length = camera_params[0, 0] * pixel_size_x
    principal_point = jnp.array([0, 0, focal_length])
    pixel_center = jnp.array([
        principal_point[0] + pixel_size_x * pixel_col,
        principal_point[1] - pixel_size_y * pixel_row,
        principal_point[2],
    ])
    origin_to_pixel = pixel_center
    origin_to_pixel = origin_to_pixel / jnp.linalg.norm(origin_to_pixel, ord=2)
    return origin_to_pixel


def sample_rays_for_image_render(
    camera_origin: jax.Array,
    camera_params: jax.Array,
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
    camera_origin = camera_origin[:3] / camera_origin[3]
    ray_coords = jnp.zeros((image_height, image_width, 6), dtype=jnp.float32)
    for row_i in range(image_height):
        for col_i in range(image_width):
            ray_direction = get_ray_to_image_coordinates(camera_params, row_i, col_i, 1.0, 1.0)
            ray_coords = ray_coords.at[row_i, col_i, 3:6].set(ray_direction)
    return ray_coords


def sample_positions_along_rays(
    rays: jax.Array, near_distance: float, far_distance: float, pos_per_ray: int
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
    norm_ray_directions = ray_directions / jnp.expand_dims(
        (jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2), 2
    )
    distance_interval = (far_distance - near_distance) / pos_per_ray
    for pos_i in range(pos_per_ray + 1):
        result = result.at[..., pos_i, :].set(
            ray_origins
            + norm_ray_directions * (near_distance + pos_i * distance_interval)
        )
    return result


def blend_ray_features(ray_features: jax.Array) -> jax.Array:
    """Compute one color for each ray.

    @param ray_features Coordinates, color, and transparency sampled along rays. Shape: (..., pos_per_ray, 7).
    Second axis: x, y, z, R, G, B, sigma.
    @return One color per ray. Shape: (num_rays, ..., 3). Last axis: R, G, B.
    """
    ray_features = jnp.array(ray_features)
    # compute length and center of intervals between samples
    interval_lengths = ray_features[..., 1::, :3] - ray_features[..., ::-1, :3]
    interval_centers = ray_features[..., ::-1, :3] + (
        (ray_features[..., 1::, :3] - ray_features[..., ::-1, :3]) / 2
    )
    # compute distances between origin and interval centers and samples
    origin_center_distances = jnp.linalg.norm(interval_centers, axis=-1, ord=2) ** 1 / 2
    origin_sample_distances = (
        jnp.linalg.norm(ray_features[..., :, :3], axis=-1, ord=2) ** 1 / 2
    )
    # interpolate values at midpoints between samples
    center_values = jnp.interp(
        origin_center_distances, origin_sample_distances, ray_features[..., :, 3:]
    )
    # weight interpolated colors at midpoints with interval lengths and interpolated sigma values
    blended_values = jnp.sum(
        center_values[..., :3] * center_values[..., 3] * interval_lengths, axis=-2
    )
    return blended_values
