from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.shapes.plane import Plane


def test_intersect_parallel_ray():
    p = Plane()
    r = Ray(Point(0, 10, 0), Vector(0, 0, 1))
    assert p.intersect(r) == []


def test_intersect_coplanar_ray():
    p = Plane()
    r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
    assert p.intersect(r) == []


def test_intersect_from_above():
    p = Plane()
    r = Ray(Point(0, 1, 0), Vector(0, -1, 0))
    xs = p.intersect(r)
    assert len(xs) == 1
    assert math.isclose(xs[0].t, 1)


def test_intersect_from_below():
    p = Plane()
    r = Ray(Point(0, -1, 0), Vector(0, 1, 0))
    xs = p.intersect(r)
    assert len(xs) == 1
    assert math.isclose(xs[0].t, 1)


def test_normal_constant_everywhere():
    p = Plane()
    n1 = p.normal_at(Point(0, 0, 0))
    n2 = p.normal_at(Point(10, 0, -10))
    n3 = p.normal_at(Point(-5, 0, 150))
    assert n1 == n2 == n3 == Vector(0, 1, 0)
