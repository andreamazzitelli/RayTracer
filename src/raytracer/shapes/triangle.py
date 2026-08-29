from __future__ import annotations
import math

from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.shapes.shape import Shape
from raytracer.geometry.bounding_box import BoundingBox

TRIANGLE_EPSILON = 1e-9


class Triangle(Shape):
    __slots__ = ("p1", "p2", "p3", "e1", "e2", "_normal")

    def __init__(self, p1: Point, p2: Point, p3: Point, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.e1 = p2 - p1
        self.e2 = p3 - p1
        self._normal = self.e2.cross(self.e1).normalize()
        self.bounding_box = BoundingBox.from_points([p1, p2, p3])

    def intersect(self, ray: Ray) -> list[Intersection]:
        local_ray = self._to_local_ray(ray)

        direction_cross_e2 = local_ray.direction.cross(self.e2)
        det = self.e1.dot(direction_cross_e2)

        if math.isclose(det, 0, abs_tol=TRIANGLE_EPSILON):
            return []

        f = 1.0 / det
        p1_to_origin = local_ray.origin - self.p1
        u = f * p1_to_origin.dot(direction_cross_e2)

        if u < 0 or u > 1:
            return []

        origin_cross_e1 = p1_to_origin.cross(self.e1)
        v = f * local_ray.direction.dot(origin_cross_e1)

        if v < 0 or (u + v) > 1:
            return []

        t = f * self.e2.dot(origin_cross_e1)
        return [Intersection(t, self)]

    def normal_at(self, point: Point) -> Vector:
        world_normal = self._inverse_transpose_transform.apply_to_vector(self._normal)
        return world_normal.normalize()

    def _to_local_ray(self, ray: Ray) -> Ray:
        inverse = self._inverse_transform
        return Ray(inverse.apply_to_point(ray.origin), inverse.apply_to_vector(ray.direction))