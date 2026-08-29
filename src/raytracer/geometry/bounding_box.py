from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray


class BoundingBox:
    __slots__ = ("min_point", "max_point")

    def __init__(self, min_point: Point, max_point: Point) -> None:
        self.min_point = min_point
        self.max_point = max_point

    @staticmethod
    def empty() -> "BoundingBox":
        inf = math.inf
        return BoundingBox(Point(inf, inf, inf), Point(-inf, -inf, -inf))

    def merge(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            Point(
                min(self.min_point.x, other.min_point.x),
                min(self.min_point.y, other.min_point.y),
                min(self.min_point.z, other.min_point.z),
            ),
            Point(
                max(self.max_point.x, other.max_point.x),
                max(self.max_point.y, other.max_point.y),
                max(self.max_point.z, other.max_point.z),
            ),
        )

    def contains_point(self, point: Point) -> bool:
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def intersects(self, ray: Ray) -> bool:
        t_min = -math.inf
        t_max = math.inf

        axes = (
            (self.min_point.x, self.max_point.x, ray.origin.x, ray.direction.x),
            (self.min_point.y, self.max_point.y, ray.origin.y, ray.direction.y),
            (self.min_point.z, self.max_point.z, ray.origin.z, ray.direction.z),
        )

        for axis_min, axis_max, origin, direction in axes:
            if direction == 0:
                if origin < axis_min or origin > axis_max:
                    return False
                continue

            t1 = (axis_min - origin) / direction
            t2 = (axis_max - origin) / direction
            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return False

        return True

    @staticmethod
    def from_points(points) -> "BoundingBox":
        box = BoundingBox.empty()
        for p in points:
            box = box.merge(BoundingBox(p, p))
        return box