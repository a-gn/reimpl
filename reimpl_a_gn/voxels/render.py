import jax
import jax.numpy as jnp
import jax.typing as jt
import jax.lax as lax


class CameraParams:
    """Parameters of a pinhole camera."""

    def __init__(self, camera_matrix: jt.ArrayLike, focal_length: float):
        """Initialize camera parameters.

        @param camera_matrix Camera matrix. Shape: (3, 4).
        @param focal_length Focal length of the camera.
        """
        self.camera_matrix = jnp.array(camera_matrix)
        if self.camera_matrix.shape != (3, 4):
            raise ValueError(f"Expected camera matrix to have shape (3, 4), got {self.camera_matrix.shape}")
        self.focal_length = focal_length
        self.pixel_size_x = (1 / self.camera_matrix[0, 0]) * focal_length
        self.pixel_size_y = (1 / self.camera_matrix[1, 1]) * focal_length
        self._inverse_camera_matrix = None

    def world_points_to_image(self, world_points: jt.ArrayLike) -> jax.Array:
        return self.camera_matrix @ world_points

    def image_points_to_world(self, image_points: jt.ArrayLike) -> jax.Array:
        """Compute the direction of a ray from the camera origin to a point in the image.

        @param image_points Image points, pixel coordinates. Shape: (point_count, 2). Last axis: x, y.
        @return Point coordinates in the world coordinate system. Shape: (point_count, 3). Last axis: x, y, z.
        """
        image_points = jnp.array(image_points)
        image_points_homogeneous = jnp.concatenate([image_points, jnp.ones((image_points.shape[0], 1))], axis=1)
        inverse_camera_matrix = self.compute_inverse_camera_matrix()
        return (inverse_camera_matrix @ image_points_homogeneous.transpose()).transpose()

    def compute_inverse_camera_matrix(self):
        if self._inverse_camera_matrix is None:
            self._inverse_camera_matrix = jnp.linalg.pinv(self.camera_matrix)
        return self._inverse_camera_matrix


def get_ray_to_image_coordinates(camera_params: CameraParams, pixel_row: float, pixel_col: float) -> jax.Array:
    """Compute the direction of a ray from the camera origin to a point in the image.

    Assumes that the camera is at the origin and looks along the z-axis.

    @param camera_params Camera parameters.
    @param pixel_row Row index of the pixel. Can be a float to represent subpixel positions.
    @param pixel_col Column index of the pixel. Can be a float to represent subpixel positions.
    @return Unit vector in the direction of the ray from camera origin to a point in the image plane.
    Shape: (image_height, image_width, 3). Last axis: x, y, z.
    """
    principal_point_pixels = jnp.array([])
    principal_point_world = jnp.array([0, 0, camera_params.focal_length])
    pixel_center = jnp.array([
        principal_point_world[0] + camera_params.pixel_size_x * pixel_col,
        principal_point_world[1] + camera_params.pixel_size_y * pixel_row,
        principal_point_world[2],
    ])
    origin_to_pixel = pixel_center
    assert jnp.linalg.norm(origin_to_pixel, ord=2) > 0.0
    origin_to_pixel = origin_to_pixel / jnp.linalg.norm(origin_to_pixel, ord=2)
    return origin_to_pixel


def sample_rays_towards_all_pixels(
    camera_params: CameraParams,
    image_height: int,
    image_width: int,
) -> jax.Array:
    """Sample parameters of rays through a pinhole camera, which can be used to render an image.

    @param camera_params Camera matrix. Shape: (3, 4).
    @param image_height Pixel row count of the image we want to render.
    @param image_width Pixel column count of the image we want to render.
    @return Ray parameters. Shape: (image_height, image_width, 6). Second axis: x, y, z, dx, dy, dz.
    """
    ray_coords = jnp.zeros((image_height, image_width, 6), dtype=jnp.float32)
    for row_i in range(image_height):
        for col_i in range(image_width):
            ray_direction = get_ray_to_image_coordinates(camera_params, row_i, col_i)
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
        (jnp.linalg.norm(ray_directions, axis=-1, ord=2) ** 1 / 2), -1
    )
    distance_interval = (far_distance - near_distance) / pos_per_ray
    for pos_i in range(1, pos_per_ray + 1):
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
