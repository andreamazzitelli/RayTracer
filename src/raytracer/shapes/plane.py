from __future__ import annotations
import math

from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.shapes.shape import Shape

PLANE_EPSILON = 1e-5


class Plane(Shape):
    def intersect(self, ray: Ray) -> list[Intersection]:
        local_ray = self._to_local_ray(ray)

        if math.isclose(local_ray.direction.y, 0, abs_tol=PLANE_EPSILON):
            return []

        t = -local_ray.origin.y / local_ray.direction.y
        return [Intersection(t, self)]

    def normal_at(self, point: Point) -> Vector:
        world_normal = self._inverse_transpose_transform.apply_to_vector(Vector(0, 1, 0))
        return world_normal.normalize()

    def _to_local_ray(self, ray: Ray) -> Ray:
        inverse = self._inverse_transform
        return Ray(inverse.apply_to_point(ray.origin), inverse.apply_to_vector(ray.direction))