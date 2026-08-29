from __future__ import annotations

import math

from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.shapes.sphere import Sphere


def test_construction():
    s = Sphere(Point(0, 0, 0), 1)
    i = Intersection(3.5, s)
    assert math.isclose(i.t, 3.5)
    assert i.object is s


def test_equality():
    s = Sphere(Point(0, 0, 0), 1)
    i1 = Intersection(3.5, s)
    i2 = Intersection(3.5, s)
    assert i1 == i2
