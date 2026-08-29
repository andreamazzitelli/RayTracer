from __future__ import annotations

import math

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import translation


class Camera:
    """A camera defined by image dimensions, field of view, and a
    transform placing/orienting it in world space."""

    __slots__ = (
        "hsize", "vsize", "field_of_view", "transform",
        "half_width", "half_height", "pixel_size",
    )

    def __init__(
        self,
        hsize: int,
        vsize: int,
        field_of_view: float,
        transform: Matrix | None = None,
    ) -> None:
        self.hsize = hsize
        self.vsize = vsize
        self.field_of_view = field_of_view
        self.transform = transform if transform is not None else Matrix.identity(4)

        half_view = math.tan(field_of_view / 2)
        aspect = hsize / vsize

        if aspect >= 1:
            self.half_width = half_view
            self.half_height = half_view / aspect
        else:
            self.half_width = half_view * aspect
            self.half_height = half_view

        self.pixel_size = (self.half_width * 2) / hsize

    def ray_for_pixel(self, px: int, py: int) -> Ray:
        # Offset from canvas edge to pixel center
        x_offset = (px + 0.5) * self.pixel_size
        y_offset = (py + 0.5) * self.pixel_size

        # Camera looks down -z; world x is to the left of camera x,
        # so subtract for x, and image y grows down while camera y grows up.
        world_x = self.half_width - x_offset
        world_y = self.half_height - y_offset

        inverse_transform = self.transform.inverse()

        # Canvas sits at z = -1 in camera space
        pixel = inverse_transform.apply_to_point(Point(world_x, world_y, -1))
        origin = inverse_transform.apply_to_point(Point(0, 0, 0))
        direction = (pixel - origin).normalize()

        return Ray(origin, direction)