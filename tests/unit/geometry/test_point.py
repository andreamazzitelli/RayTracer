from __future__ import annotations

import pytest

from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector


def test_point_minus_point_is_vector():
    result = Point(3, 2, 1) - Point(5, 6, 7)
    assert result == Vector(-2, -4, -6)
    assert isinstance(result, Vector)


def test_point_minus_vector_is_point():
    result = Point(3, 2, 1) - Vector(5, 6, 7)
    assert result == Point(-2, -4, -6)
    assert isinstance(result, Point)


def test_point_plus_vector():
    result = Point(3, 2, 1) + Vector(5, 6, 7)
    assert result == Point(8, 8, 8)


def test_equality_tolerance():
    assert Point(1.0000000001, 2, 3) == Point(1, 2, 3)


def test_equality_against_non_point_does_not_crash():
    assert (Point(1, 2, 3) == None) is False
    assert (Point(1, 2, 3) == 5) is False


def test_subtract_invalid_type_raises_or_not_implemented():
    with pytest.raises(TypeError):
        Point(1, 2, 3) - "not a point or vector"
