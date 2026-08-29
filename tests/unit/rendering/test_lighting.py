from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.light import PointLight
from raytracer.rendering.lighting import lighting
from raytracer.rendering.material import Material


def test_eye_between_light_and_surface_both_straight_on():
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, 0, -1)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 0, -10), Color(1, 1, 1))

    result = lighting(m, light, p, eye, normal)
    assert result == Color(1.9, 1.9, 1.9)


def test_eye_offset_45_degrees_light_straight_on():
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, math.sqrt(2) / 2, -math.sqrt(2) / 2)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 0, -10), Color(1, 1, 1))

    result = lighting(m, light, p, eye, normal)
    assert result == Color(1.0, 1.0, 1.0)


def test_eye_straight_on_light_offset_45_degrees():
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, 0, -1)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 10, -10), Color(1, 1, 1))

    result = lighting(m, light, p, eye, normal)
    expected = 0.1 + 0.9 * (math.sqrt(2) / 2)
    assert math.isclose(result.r, expected, abs_tol=1e-9)


def test_eye_in_path_of_reflection_vector():
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, -math.sqrt(2) / 2, -math.sqrt(2) / 2)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 10, -10), Color(1, 1, 1))

    result = lighting(m, light, p, eye, normal)
    # ambient + diffuse (from previous test) + near-maximal specular
    assert result.r > 1.6


def test_light_behind_surface():
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, 0, -1)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 0, 10), Color(1, 1, 1))

    result = lighting(m, light, p, eye, normal)
    assert result == Color(0.1, 0.1, 0.1)


def test_in_shadow_returns_ambient_only():
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, 0, -1)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 0, -10), Color(1, 1, 1))

    result = lighting(m, light, p, eye, normal, in_shadow=True)
    assert result == Color(0.1, 0.1, 0.1)


def test_in_shadow_suppresses_even_maximal_specular_configuration():
    # Same geometry as test_eye_in_path_of_reflection_vector, which would
    # otherwise produce a bright specular highlight — shadow should still
    # reduce it to ambient only.
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, -math.sqrt(2) / 2, -math.sqrt(2) / 2)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 10, -10), Color(1, 1, 1))

    result = lighting(m, light, p, eye, normal, in_shadow=True)
    assert result == Color(0.1, 0.1, 0.1)


def test_default_in_shadow_is_false_matches_unshadowed_result():
    m = Material(Color(1, 1, 1))
    p = Point(0, 0, 0)
    eye = Vector(0, 0, -1)
    normal = Vector(0, 0, -1)
    light = PointLight(Point(0, 0, -10), Color(1, 1, 1))

    with_default = lighting(m, light, p, eye, normal)
    explicit_false = lighting(m, light, p, eye, normal, in_shadow=False)
    assert with_default == explicit_false
