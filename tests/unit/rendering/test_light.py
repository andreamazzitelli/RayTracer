from __future__ import annotations

from raytracer.geometry.point import Point
from raytracer.image.color import Color
from raytracer.rendering.light import PointLight


def test_construction_stores_fields():
    position = Point(0, 0, 0)
    intensity = Color(1, 1, 1)
    light = PointLight(position, intensity)
    assert light.position == position
    assert light.intensity == intensity


def test_equality_with_tolerance():
    l1 = PointLight(Point(0, 0, 0), Color(1, 1, 1))
    l2 = PointLight(Point(1e-10, 0, 0), Color(1, 1, 1))
    assert l1 == l2
