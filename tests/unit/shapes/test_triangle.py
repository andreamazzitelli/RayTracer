from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.shapes.triangle import Triangle


def test_intersect_misses_p1_p3_edge():
    t = Triangle(Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0))
    r = Ray(Point(1, 1, -2), Vector(0, 0, 1))
    assert t.intersect(r) == []


def test_intersect_misses_p1_p2_edge():
    t = Triangle(Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0))
    r = Ray(Point(-1, 1, -2), Vector(0, 0, 1))
    assert t.intersect(r) == []


def test_intersect_misses_p2_p3_edge():
    t = Triangle(Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0))
    r = Ray(Point(0, -1, -2), Vector(0, 0, 1))
    assert t.intersect(r) == []


def test_intersect_hits_interior():
    t = Triangle(Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0))
    r = Ray(Point(0, 0.5, -2), Vector(0, 0, 1))
    xs = t.intersect(r)
    assert len(xs) == 1
    assert math.isclose(xs[0].t, 2)


def test_normal_constant_regardless_of_point():
    t = Triangle(Point(0, 1, 0), Point(-1, 0, 0), Point(1, 0, 0))
    n1 = t.normal_at(Point(0, 0.5, 0))
    n2 = t.normal_at(Point(-0.5, 0.75, 0))
    n3 = t.normal_at(Point(0.5, 0.25, 0))
    assert n1 == n2 == n3
