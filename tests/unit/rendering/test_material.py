from __future__ import annotations

from raytracer.image.color import Color
from raytracer.rendering.material import Material


def test_default_coefficients():
    m = Material(Color(1, 1, 1))
    assert m.ambient == 0.1
    assert m.diffuse == 0.9
    assert m.specular == 0.9
    assert m.shininess == 200.0


def test_equality_with_tolerance():
    m1 = Material(Color(1, 1, 1), ambient=0.2)
    m2 = Material(Color(1, 1, 1), ambient=0.2 + 1e-10)
    assert m1 == m2


def test_inequality_on_differing_field():
    m1 = Material(Color(1, 1, 1), ambient=0.2)
    m2 = Material(Color(1, 1, 1), ambient=0.5)
    assert m1 != m2


def test_equality_against_non_material_does_not_crash():
    m = Material(Color(1, 1, 1))
    assert (m == None) is False  # noqa: E711
    assert (m == 5) is False
