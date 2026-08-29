from __future__ import annotations

from raytracer.geometry.bounding_box import BoundingBox
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector


def _unit_box() -> BoundingBox:
    return BoundingBox(Point(-1, -1, -1), Point(1, 1, 1))


def test_ray_hits_box_from_six_directions():
    box = _unit_box()
    cases = [
        (Point(5, 0.5, 0), Vector(-1, 0, 0)),
        (Point(-5, 0.5, 0), Vector(1, 0, 0)),
        (Point(0.5, 5, 0), Vector(0, -1, 0)),
        (Point(0.5, -5, 0), Vector(0, 1, 0)),
        (Point(0.5, 0, 5), Vector(0, 0, -1)),
        (Point(0, 0.5, 0), Vector(0, 0, 1)),
    ]
    for origin, direction in cases:
        assert box.intersects(Ray(origin, direction.normalize()))


def test_ray_misses_box():
    box = _unit_box()
    cases = [
        (Point(-2, 0, 0), Vector(0.2673, 0.5345, 0.8018)),
        (Point(0, -2, 0), Vector(0.8018, 0.2673, 0.5345)),
        (Point(0, 0, -2), Vector(0.5345, 0.8018, 0.2673)),
        (Point(2, 0, 2), Vector(0, 0, -1)),
        (Point(0, 2, 2), Vector(0, -1, 0)),
        (Point(2, 2, 0), Vector(-1, 0, 0)),
    ]
    for origin, direction in cases:
        assert not box.intersects(Ray(origin, direction.normalize()))


def test_ray_parallel_to_axis_outside_slab_misses():
    box = _unit_box()
    r = Ray(Point(5, 5, 0), Vector(0, 0, 1))
    assert not box.intersects(r)


def test_ray_parallel_to_axis_inside_slab_hits():
    box = _unit_box()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    assert box.intersects(r)


def test_merge_produces_smallest_containing_box():
    a = BoundingBox(Point(-1, -1, -1), Point(1, 1, 1))
    b = BoundingBox(Point(0, 0, 0), Point(3, 3, 3))
    merged = a.merge(b)
    assert merged.min_point == Point(-1, -1, -1)
    assert merged.max_point == Point(3, 3, 3)


def test_empty_box_is_merge_identity():
    a = BoundingBox(Point(-1, -1, -1), Point(1, 1, 1))
    merged = a.merge(BoundingBox.empty())
    assert merged.min_point == a.min_point
    assert merged.max_point == a.max_point


def test_from_points_matches_triangle_vertices():
    p1, p2, p3 = Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0)
    box = BoundingBox.from_points([p1, p2, p3])
    assert box.min_point == Point(-1, 0, 0)
    assert box.max_point == Point(1, 1, 0)


def test_contains_point():
    box = _unit_box()
    assert box.contains_point(Point(0, 0, 0))
    assert not box.contains_point(Point(2, 0, 0))