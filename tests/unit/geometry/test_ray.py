from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector


def test_position_at_zero():
    r = Ray(Point(2, 3, 4), Vector(1, 0, 0))
    assert r.position_at(0) == Point(2, 3, 4)


def test_position_at_positive_t():
    r = Ray(Point(2, 3, 4), Vector(1, 0, 0))
    assert r.position_at(1) == Point(3, 3, 4)


def test_position_at_fractional_t():
    r = Ray(Point(2, 3, 4), Vector(1, 0, 0))
    assert r.position_at(2.5) == Point(4.5, 3, 4)


def test_position_at_negative_t():
    r = Ray(Point(2, 3, 4), Vector(1, 0, 0))
    assert r.position_at(-1) == Point(1, 3, 4)


def test_position_at_matches_formula_property():
    r = Ray(Point(1, -2, 3), Vector(0.5, 1, -1))
    for t in (0, 0.3, 1, 5, -2.5):
        expected = r.origin + r.direction * t
        assert r.position_at(t) == expected


def test_equality_between_separately_constructed_rays():
    r1 = Ray(Point(1, 2, 3), Vector(4, 5, 6))
    r2 = Ray(Point(1, 2, 3), Vector(4, 5, 6))
    assert r1 == r2


def test_equality_against_non_ray_does_not_crash():
    r = Ray(Point(0, 0, 0), Vector(1, 0, 0))
    assert (r == None) is False
