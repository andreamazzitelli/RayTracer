from __future__ import annotations

import math

from raytracer.geometry.computations import prepare_computations, UNDER_EPSILON
from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import scaling, translation
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.material import Material
from raytracer.shapes.sphere import Sphere


def glass_sphere(**kwargs) -> Sphere:
    material = kwargs.pop("material", None) or Material(
        Color(1, 1, 1), transparency=1.0, refractive_index=1.5
    )
    return Sphere(Point(0, 0, 0), 1.0, material=material, **kwargs)


def test_n1_n2_at_various_intersections_of_three_glass_spheres():
    a = glass_sphere(transform=scaling(2, 2, 2))
    a.material.refractive_index = 1.5

    b = glass_sphere(transform=translation(0, 0, -0.25))
    b.material.refractive_index = 2.0

    c = glass_sphere(transform=translation(0, 0, 0.25))
    c.material.refractive_index = 2.5

    r = Ray(Point(0, 0, -4), Vector(0, 0, 1))
    xs = [
        Intersection(2, a),
        Intersection(2.75, b),
        Intersection(3.25, c),
        Intersection(4.75, b),
        Intersection(5.25, c),
        Intersection(6, a),
    ]

    expected = [
        (1.0, 1.5),
        (1.5, 2.0),
        (2.0, 2.5),
        (2.5, 2.5),
        (2.5, 1.5),
        (1.5, 1.0),
    ]

    for index, (n1, n2) in enumerate(expected):
        comps = prepare_computations(xs[index], r, xs)
        assert math.isclose(comps.n1, n1), f"index {index}: n1"
        assert math.isclose(comps.n2, n2), f"index {index}: n2"


def test_under_point_is_offset_below_the_surface():
    shape = glass_sphere(transform=translation(0, 0, 1))
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    i = Intersection(5, shape)

    comps = prepare_computations(i, r, [i])

    assert comps.under_point.z > UNDER_EPSILON / 2
    assert comps.point.z < comps.under_point.z


def test_prepare_computations_defaults_all_intersections_to_single_hit():
    # When all_intersections is omitted, n1/n2 should still be computable
    # (defaulting to a single-element list), even if not accounting for
    # containment by other transparent objects.
    shape = glass_sphere()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    i = Intersection(5, shape)

    comps = prepare_computations(i, r)

    assert math.isclose(comps.n1, 1.0)
    assert math.isclose(comps.n2, 1.5)