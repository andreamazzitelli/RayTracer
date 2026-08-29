from __future__ import annotations

import math

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.geometry.transform import (
    rotation_x,
    rotation_y,
    rotation_z,
    scaling,
    shearing,
    translation,
)
from raytracer.geometry.vector import Vector


def test_translation_point():
    t = translation(5, -3, 2)
    assert t.apply_to_point(Point(-3, 4, 5)) == Point(2, 1, 7)


def test_translation_vector_unaffected():
    t = translation(5, -3, 2)
    v = Vector(-3, 4, 5)
    assert t.apply_to_vector(v) == v


def test_translation_inverse():
    t = translation(5, -3, 2)
    inv = t.inverse()
    expected = translation(-5, 3, -2)
    assert inv == expected


def test_scaling_point():
    s = scaling(2, 3, 4)
    assert s.apply_to_point(Point(-4, 6, 8)) == Point(-8, 18, 32)


def test_scaling_vector():
    s = scaling(2, 3, 4)
    assert s.apply_to_vector(Vector(-4, 6, 8)) == Vector(-8, 18, 32)


def test_scaling_inverse():
    s = scaling(2, 3, 4)
    expected = scaling(0.5, 1 / 3, 0.25)
    assert s.inverse() == expected


def test_reflection_via_negative_scaling():
    s = scaling(-1, 1, 1)
    assert s.apply_to_point(Point(2, 3, 4)) == Point(-2, 3, 4)


def test_rotation_x_quarter_turn():
    half_quarter = rotation_x(math.pi / 4)
    full_quarter = rotation_x(math.pi / 2)
    p = Point(0, 1, 0)
    assert half_quarter.apply_to_point(p) == Point(0, math.sqrt(2) / 2, math.sqrt(2) / 2)
    assert full_quarter.apply_to_point(p) == Point(0, 0, 1)


def test_rotation_y_quarter_turn():
    full_quarter = rotation_y(math.pi / 2)
    p = Point(0, 0, 1)
    assert full_quarter.apply_to_point(p) == Point(1, 0, 0)


def test_rotation_z_quarter_turn():
    full_quarter = rotation_z(math.pi / 2)
    p = Point(0, 1, 0)
    assert full_quarter.apply_to_point(p) == Point(-1, 0, 0)


def test_rotation_x_periodicity():
    r = rotation_x(math.pi / 2)
    p = Point(0, 1, 0)
    for _ in range(4):
        p = r.apply_to_point(p)
    assert p == Point(0, 1, 0)


def test_rotation_y_periodicity():
    r = rotation_y(math.pi / 2)
    p = Point(1, 0, 0)
    for _ in range(4):
        p = r.apply_to_point(p)
    assert p == Point(1, 0, 0)


def test_rotation_z_periodicity():
    r = rotation_z(math.pi / 2)
    p = Point(1, 0, 0)
    for _ in range(4):
        p = r.apply_to_point(p)
    assert p == Point(1, 0, 0)


def test_shearing_xy():
    s = shearing(1, 0, 0, 0, 0, 0)
    assert s.apply_to_point(Point(2, 3, 4)) == Point(5, 3, 4)


def test_shearing_yx():
    s = shearing(0, 0, 1, 0, 0, 0)
    assert s.apply_to_point(Point(2, 3, 4)) == Point(2, 5, 4)


def test_shearing_zy():
    s = shearing(0, 0, 0, 0, 0, 1)
    assert s.apply_to_point(Point(2, 3, 4)) == Point(2, 3, 7)


def test_composition_order_step_by_step_matches_combined():
    p = Point(1, 0, 1)
    a = rotation_x(math.pi / 2)
    b = scaling(5, 5, 5)
    c = translation(10, 5, 7)

    p2 = a.apply_to_point(p)
    p3 = b.apply_to_point(p2)
    p4 = c.apply_to_point(p3)

    combined = c @ b @ a
    assert combined.apply_to_point(p) == p4
