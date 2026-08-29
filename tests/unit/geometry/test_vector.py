from __future__ import annotations

import math

import pytest

from raytracer.geometry.vector import Vector


def test_add():
    assert Vector(1, 2, 3) + Vector(4, 5, 6) == Vector(5, 7, 9)


def test_subtract():
    assert Vector(4, 5, 6) - Vector(1, 2, 3) == Vector(3, 3, 3)


def test_negate():
    assert -Vector(1, -2, 3) == Vector(-1, 2, -3)


def test_scalar_multiply():
    assert Vector(1, 2, 3) * 2 == Vector(2, 4, 6)
    assert 2 * Vector(1, 2, 3) == Vector(2, 4, 6)


def test_scalar_divide():
    assert Vector(2, 4, 6) / 2 == Vector(1, 2, 3)


def test_magnitude_axis_aligned():
    assert math.isclose(Vector(1, 0, 0).magnitude(), 1)


def test_magnitude_nontrivial():
    assert math.isclose(Vector(1, 2, 2).magnitude(), 3)


def test_normalize():
    v = Vector(4, 0, 0).normalize()
    assert v == Vector(1, 0, 0)
    assert math.isclose(v.magnitude(), 1)


def test_normalize_zero_vector():
    assert Vector(0, 0, 0).normalize() == Vector(0, 0, 0)


def test_dot():
    assert math.isclose(Vector(1, 2, 3).dot(Vector(2, 3, 4)), 20)


def test_dot_commutative():
    a, b = Vector(1, 2, 3), Vector(-2, 5, 0.5)
    assert math.isclose(a.dot(b), b.dot(a))


def test_cross():
    a, b = Vector(1, 0, 0), Vector(0, 1, 0)
    assert a.cross(b) == Vector(0, 0, 1)
    assert b.cross(a) == Vector(0, 0, -1)


def test_cross_perpendicular_property():
    a, b = Vector(1, 2, 3), Vector(-2, 5, 0.5)
    cross = a.cross(b)
    assert math.isclose(cross.dot(a), 0, abs_tol=1e-9)
    assert math.isclose(cross.dot(b), 0, abs_tol=1e-9)


def test_cross_anticommutative():
    a, b = Vector(1, 2, 3), Vector(-2, 5, 0.5)
    assert a.cross(b) == -b.cross(a)


def test_equality_tolerance():
    assert Vector(1.0000000001, 2, 3) == Vector(1, 2, 3)


def test_equality_against_non_vector_does_not_crash():
    assert (Vector(1, 2, 3) == None) is False
    assert (Vector(1, 2, 3) == "vector") is False
    assert (Vector(1, 2, 3) == 5) is False

def test_reflect_axis_aligned():
    v = Vector(1, -1, 0)
    n = Vector(0, 1, 0)
    assert v.reflect(n) == Vector(1, 1, 0)


def test_reflect_at_angle():
    v = Vector(0, -1, 0)
    n = Vector(math.sqrt(2) / 2, math.sqrt(2) / 2, 0)
    result = v.reflect(n)
    assert result == Vector(1, 0, 0)
