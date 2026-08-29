from __future__ import annotations

import math

from raytracer.geometry.computations import prepare_computations
from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.light import PointLight
from raytracer.scene.world import World, hit


def test_intersect_returns_combined_sorted_list():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    xs = w.intersect(r)
    assert len(xs) == 4
    assert [math.isclose(x.t, e) for x, e in zip(xs, [4.0, 4.5, 5.5, 6.0])] == [True] * 4


def test_hit_returns_none_on_empty_list():
    assert hit([]) is None


def test_hit_returns_none_when_all_negative():
    s = World.default_world().objects[0]
    xs = [Intersection(-2, s), Intersection(-1, s)]
    assert hit(xs) is None


def test_hit_returns_closest_positive_from_mixed_list():
    s = World.default_world().objects[0]
    xs = [Intersection(-1, s), Intersection(1, s), Intersection(2, s)]
    result = hit(xs)
    assert result is not None
    assert math.isclose(result.t, 1)


def test_shade_hit_outside():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    i = Intersection(4, w.objects[0])
    comps = prepare_computations(i, r)
    color = w.shade_hit(comps)
    assert color == Color(0.3806609553101071, 0.47582619413763383, 0.2854957164825803)


def test_shade_hit_inside():
    w = World.default_world()
    w.lights = [PointLight(Point(0, 0.25, 0), Color(1, 1, 1))]
    r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
    i = Intersection(0.5, w.objects[1])
    comps = prepare_computations(i, r)
    color = w.shade_hit(comps)
    assert color == Color(0.9049812520679432, 0.9049812520679432, 0.9049812520679432)


def test_shade_hit_in_shadow_uses_ambient_only():
    from raytracer.geometry.transform import translation
    from raytracer.shapes.sphere import Sphere

    s1 = Sphere(Point(0, 0, 0), 1)
    s2 = Sphere(Point(0, 0, 0), 1, transform=translation(0, 0, 10))
    light = PointLight(Point(0, 0, -10), Color(1, 1, 1))
    w = World(objects=[s1, s2], lights=[light])

    r = Ray(Point(0, 0, 5), Vector(0, 0, 1))
    i = Intersection(4, s2)
    comps = prepare_computations(i, r)

    color = w.shade_hit(comps)
    assert color == Color(0.1, 0.1, 0.1)


def test_color_at_ray_misses():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 1, 0))
    assert w.color_at(r) == Color(0, 0, 0)


def test_color_at_ray_hits():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    color = w.color_at(r)
    assert color == Color(0.3806609553101071, 0.47582619413763383, 0.2854957164825803)


def test_color_at_matches_manual_pipeline():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))

    manual_hit = hit(w.intersect(r))
    manual_comps = prepare_computations(manual_hit, r)
    manual_color = w.shade_hit(manual_comps)

    assert w.color_at(r) == manual_color


def test_is_shadowed_nothing_between_point_and_light():
    w = World.default_world()
    p = Point(0, 10, 0)
    assert w.is_shadowed(p, w.lights[0]) is False


def test_is_shadowed_object_between_point_and_light():
    w = World.default_world()
    p = Point(10, -10, 10)
    assert w.is_shadowed(p, w.lights[0]) is True


def test_is_shadowed_object_behind_light():
    w = World.default_world()
    p = Point(-20, 20, -20)
    assert w.is_shadowed(p, w.lights[0]) is False


def test_is_shadowed_point_between_light_and_object():
    w = World.default_world()
    p = Point(-2, 2, -2)
    assert w.is_shadowed(p, w.lights[0]) is False
