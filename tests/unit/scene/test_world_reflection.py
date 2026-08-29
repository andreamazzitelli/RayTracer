from __future__ import annotations

import math

from raytracer.geometry.computations import prepare_computations
from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import translation
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material
from raytracer.shapes.plane import Plane
from raytracer.scene.world import World


def test_reflected_color_for_nonreflective_material():
    w = World.default_world()
    r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
    shape = w.objects[1]
    shape.material.ambient = 1.0

    i = Intersection(1, shape)
    comps = prepare_computations(i, r, [i])

    assert w.reflected_color(comps) == Color(0, 0, 0)


def test_reflected_color_for_reflective_material():
    w = World.default_world()
    plane = Plane(
        material=Material(Color(1, 1, 1), reflective=0.5),
        transform=translation(0, -1, 0),
    )
    w.objects.append(plane)

    r = Ray(Point(0, 0, -3), Vector(0, -math.sqrt(2) / 2, math.sqrt(2) / 2))
    i = Intersection(math.sqrt(2), plane)
    comps = prepare_computations(i, r, [i])

    color = w.reflected_color(comps)
    assert color.r > 0
    assert color.g > 0
    assert color.b > 0


def test_shade_hit_with_reflective_material_is_brighter_than_surface_alone():
    w = World.default_world()
    plane = Plane(
        material=Material(Color(1, 1, 1), reflective=0.5),
        transform=translation(0, -1, 0),
    )
    w.objects.append(plane)

    r = Ray(Point(0, 0, -3), Vector(0, -math.sqrt(2) / 2, math.sqrt(2) / 2))
    i = Intersection(math.sqrt(2), plane)
    comps = prepare_computations(i, r, [i])

    with_reflection = w.shade_hit(comps)
    without_reflection = w.reflected_color(comps, remaining=0)

    # shade_hit's total must be at least as bright as the surface color
    # alone would be, since reflection only adds light, never removes it
    surface_only = w.shade_hit(comps, remaining=0)
    assert with_reflection.r >= surface_only.r
    assert with_reflection.g >= surface_only.g
    assert with_reflection.b >= surface_only.b


def test_reflected_color_at_max_recursion_depth_returns_black():
    w = World.default_world()
    plane = Plane(
        material=Material(Color(1, 1, 1), reflective=0.5),
        transform=translation(0, -1, 0),
    )
    w.objects.append(plane)

    r = Ray(Point(0, 0, -3), Vector(0, -math.sqrt(2) / 2, math.sqrt(2) / 2))
    i = Intersection(math.sqrt(2), plane)
    comps = prepare_computations(i, r, [i])

    assert w.reflected_color(comps, remaining=0) == Color(0, 0, 0)


def test_color_at_terminates_with_mutually_reflective_surfaces():
    # Two infinite mirrors facing each other — without a depth limit this
    # recurses forever between them. This test's only real assertion is
    # that it returns at all, rather than raising RecursionError.
    light = PointLight(Point(0, 0, 0), Color(1, 1, 1))
    lower = Plane(
        material=Material(Color(1, 1, 1), reflective=1.0),
        transform=translation(0, -1, 0),
    )
    upper = Plane(
        material=Material(Color(1, 1, 1), reflective=1.0),
        transform=translation(0, 1, 0),
    )
    w = World(objects=[lower, upper], lights=[light])

    r = Ray(Point(0, 0, 0), Vector(0, 1, 0))
    color = w.color_at(r)

    assert color is not None