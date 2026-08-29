from __future__ import annotations

from raytracer.image.color import Color


def test_addition():
    assert Color(0.9, 0.6, 0.75) + Color(0.7, 0.1, 0.25) == Color(1.6, 0.7, 1.0)


def test_scalar_multiplication():
    assert Color(0.2, 0.3, 0.4) * 2 == Color(0.4, 0.6, 0.8)


def test_hadamard_product_distinct_from_scalar_multiply():
    c1 = Color(1, 0.5, 0)
    c2 = Color(0.5, 0.5, 1)
    assert c1.hadamard(c2) == Color(0.5, 0.25, 0)


def test_equality_with_tolerance():
    assert Color(0.1, 0.2, 0.3) == Color(0.1 + 1e-10, 0.2, 0.3)


def test_equality_against_non_color_does_not_crash():
    assert (Color(1, 1, 1) == None) is False  # noqa: E711
    assert (Color(1, 1, 1) == 5) is False
