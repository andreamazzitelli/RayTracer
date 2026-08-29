from __future__ import annotations

import math

from raytracer.geometry.computations import prepare_computations
from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.material import Material
from raytracer.shapes.sphere import Sphere
from raytracer.scene.world import World


def glass_sphere(**kwargs) -> Sphere:
    material = kwargs.pop("material", None) or Material(
        Color(1, 1, 1), transparency=1.0, refractive_index=1.5
    )
    return Sphere(Point(0, 0, 0), 1.0, material=material, **kwargs)


def test_refracted_color_for_opaque_material_is_black():
    w = World.default_world()
    shape = w.objects[0]
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    xs = [Intersection(4, shape), Intersection(6, shape)]

    comps = prepare_computations(xs[0], r, xs)

    assert w.refracted_color(comps, remaining=5) == Color(0, 0, 0)


def test_refracted_color_at_max_recursion_depth_is_black():
    w = World.default_world()
    shape = w.objects[0]
    shape.material.transparency = 1.0
    shape.material.refractive_index = 1.5

    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    xs = [Intersection(4, shape), Intersection(6, shape)]

    comps = prepare_computations(xs[0], r, xs)

    assert w.refracted_color(comps, remaining=0) == Color(0, 0, 0)


def test_refracted_color_under_total_internal_reflection_is_black():
    shape = glass_sphere()
    r = Ray(Point(0, 0, math.sqrt(2) / 2), Vector(0, 1, 0))
    xs = [
        Intersection(-math.sqrt(2) / 2, shape),
        Intersection(math.sqrt(2) / 2, shape),
    ]

    # the ray originates inside the sphere, so we shade the second
    # intersection — the one where light exits
    comps = prepare_computations(xs[1], r, xs)
    w = World(objects=[shape], lights=[])

    assert w.refracted_color(comps, remaining=5) == Color(0, 0, 0)


def test_refracted_color_produces_nonblack_result_for_refracting_ray():
    w = World.default_world()
    a = w.objects[0]
    a.material.ambient = 1.0

    b = w.objects[1]
    b.material.transparency = 1.0
    b.material.refractive_index = 1.5

    r = Ray(Point(0, 0, 0.1), Vector(0, 1, 0))
    xs = [
        Intersection(-0.9899, a),
        Intersection(-0.4899, b),
        Intersection(0.4899, b),
        Intersection(0.9899, a),
    ]

    comps = prepare_computations(xs[2], r, xs)
    color = w.refracted_color(comps, remaining=5)

    assert color != Color(0, 0, 0)


def test_shade_hit_with_transparent_material_matches_expected_color():
    from raytracer.geometry.transform import translation
    from raytracer.shapes.plane import Plane

    w = World.default_world()
    floor = Plane(
        transform=translation(0, -1, 0),
        material=Material(Color(1, 1, 1), transparency=0.5, refractive_index=1.5),
    )
    w.objects.append(floor)

    ball = Sphere(
        Point(0, 0, 0),
        1.0,
        material=Material(Color(1, 0, 0), ambient=0.5),
        transform=translation(0, -3.5, -0.5),
    )
    w.objects.append(ball)

    r = Ray(Point(0, 0, -3), Vector(0, -math.sqrt(2) / 2, math.sqrt(2) / 2))
    i = Intersection(math.sqrt(2), floor)
    comps = prepare_computations(i, r, [i])

    color = w.shade_hit(comps, remaining=5)

    assert color == Color(0.9364250822069577, 0.6864250822069577, 0.6864250822069577)