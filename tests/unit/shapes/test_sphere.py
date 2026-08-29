from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import scaling, translation
from raytracer.geometry.vector import Vector
from raytracer.rendering.material import Material
from raytracer.image.color import Color
from raytracer.shapes.sphere import Sphere


def test_ray_misses_sphere():
    s = Sphere(Point(0, 0, 0), 1)
    r = Ray(Point(0, 2, -5), Vector(0, 0, 1))
    assert s.intersect(r) == []


def test_ray_hits_sphere_at_two_points():
    s = Sphere(Point(0, 0, 0), 1)
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    xs = s.intersect(r)
    assert len(xs) == 2
    assert math.isclose(xs[0].t, 4.0)
    assert math.isclose(xs[1].t, 6.0)


def test_ray_tangent_to_sphere():
    s = Sphere(Point(0, 0, 0), 1)
    r = Ray(Point(0, 1, -5), Vector(0, 0, 1))
    xs = s.intersect(r)
    assert len(xs) == 2
    assert math.isclose(xs[0].t, xs[1].t)
    assert math.isclose(xs[0].t, 5.0)


def test_ray_originates_inside_sphere():
    s = Sphere(Point(0, 0, 0), 1)
    r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
    xs = s.intersect(r)
    assert len(xs) == 2
    assert math.isclose(xs[0].t, -1.0)
    assert math.isclose(xs[1].t, 1.0)


def test_sphere_behind_ray():
    s = Sphere(Point(0, 0, 0), 1)
    r = Ray(Point(0, 0, 5), Vector(0, 0, 1))
    xs = s.intersect(r)
    assert len(xs) == 2
    assert math.isclose(xs[0].t, -6.0)
    assert math.isclose(xs[1].t, -4.0)


def test_intersections_sorted_ascending():
    s = Sphere(Point(0, 0, 0), 1)
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    xs = s.intersect(r)
    assert xs[0].t < xs[1].t


def test_normal_at_axis_aligned_points():
    s = Sphere(Point(0, 0, 0), 1)
    assert s.normal_at(Point(1, 0, 0)) == Vector(1, 0, 0)
    assert s.normal_at(Point(0, 1, 0)) == Vector(0, 1, 0)
    assert s.normal_at(Point(0, 0, 1)) == Vector(0, 0, 1)


def test_normal_is_unit_length_property():
    s = Sphere(Point(0, 0, 0), 1)
    n = s.normal_at(Point(
        math.sqrt(3) / 3, math.sqrt(3) / 3, math.sqrt(3) / 3
    ))
    assert math.isclose(n.magnitude(), 1, abs_tol=1e-9)


def test_default_material():
    s = Sphere(Point(0, 0, 0), 1)
    assert s.material == Material(Color(1, 1, 1))


def test_custom_material_stored():
    m = Material(Color(1, 0, 0), ambient=0.5)
    s = Sphere(Point(0, 0, 0), 1, material=m)
    assert s.material == m


def test_intersect_translated_sphere():
    s = Sphere(Point(0, 0, 0), 1, transform=translation(5, 0, 0))
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    assert s.intersect(r) == []


def test_intersect_scaled_sphere():
    s = Sphere(Point(0, 0, 0), 1, transform=scaling(2, 2, 2))
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    xs = s.intersect(r)
    assert len(xs) == 2
    assert math.isclose(xs[0].t, 3.0)
    assert math.isclose(xs[1].t, 7.0)


def test_normal_on_translated_sphere():
    s = Sphere(Point(0, 0, 0), 1, transform=translation(0, 1, 0))
    n = s.normal_at(Point(0, 1.70711, -0.70711))
    assert n.x == 0
    assert math.isclose(n.y, 0.70711, abs_tol=1e-4)
    assert math.isclose(n.z, -0.70711, abs_tol=1e-4)


def test_normal_on_nonuniformly_scaled_sphere_is_unit_length():
    s = Sphere(Point(0, 0, 0), 1, transform=scaling(1, 0.5, 1))
    n = s.normal_at(Point(0, math.sqrt(2) / 2, -math.sqrt(2) / 2))
    assert math.isclose(n.magnitude(), 1, abs_tol=1e-9)
