from __future__ import annotations

import math

from raytracer.geometry.computations import prepare_computations
from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.shapes.sphere import Sphere


def test_computations_outside_hit():
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    shape = Sphere(Point(0, 0, 0), 1)
    i = Intersection(4, shape)

    comps = prepare_computations(i, r)

    assert comps.inside is False
    assert comps.point == Point(0, 0, -1)
    assert comps.eye_vector == Vector(0, 0, -1)
    assert comps.normal_vector == Vector(0, 0, -1)


def test_computations_inside_hit_flips_normal():
    r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
    shape = Sphere(Point(0, 0, 0), 1)
    i = Intersection(1, shape)

    comps = prepare_computations(i, r)

    assert comps.point == Point(0, 0, 1)
    assert comps.eye_vector == Vector(0, 0, -1)
    assert comps.inside is True
    # normal would be (0,0,1) unflipped; should be inverted
    assert comps.normal_vector == Vector(0, 0, -1)


def test_over_point_is_offset_from_surface():
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    shape = Sphere(Point(0, 0, 0), 1)
    i = Intersection(4, shape)

    comps = prepare_computations(i, r)

    assert comps.over_point.z < -1
    assert comps.point.z > comps.over_point.z
