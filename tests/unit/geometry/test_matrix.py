from __future__ import annotations

import math

import pytest

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector


def test_construction_and_element_access():
    m = Matrix([[1, 2], [3, 4]])
    assert m[0, 0] == 1
    assert m[0, 1] == 2
    assert m[1, 0] == 3
    assert m[1, 1] == 4


def test_construction_ragged_rows_raises():
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3]])


def test_shape_non_square():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.shape == (2, 3)


def test_equality():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[1, 2], [3, 4]])
    c = Matrix([[1, 2], [3, 5]])
    assert a == b
    assert a != c


def test_identity():
    ident = Matrix.identity(4)
    for i in range(4):
        for j in range(4):
            assert math.isclose(ident[i, j], 1.0 if i == j else 0.0)


def test_identity_multiplication_property():
    m = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    assert Matrix.identity(4) @ m == m


def test_matrix_multiplication_hand_computed():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    expected = Matrix([[19, 22], [43, 50]])
    assert (a @ b) == expected


def test_matrix_multiplication_noncommutative():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    assert (a @ b) != (b @ a)


def test_transpose():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    expected = Matrix([[1, 4], [2, 5], [3, 6]])
    assert m.transpose() == expected


def test_apply_to_point_identity():
    p = Point(1, 2, 3)
    assert Matrix.identity(4).apply_to_point(p) == p


def test_apply_to_vector_identity():
    v = Vector(1, 2, 3)
    assert Matrix.identity(4).apply_to_vector(v) == v


def test_submatrix():
    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = Matrix([[1, 3], [7, 9]])
    assert m.submatrix(1, 1) == expected


def test_minor_simple_case():
    m = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])
    assert math.isclose(m.minor(1, 0), 25)


def test_cofactor_sign_alternation():
    m = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])
    minor00 = m.minor(0, 0)
    cofactor00 = m.cofactor(0, 0)
    minor10 = m.minor(1, 0)
    cofactor10 = m.cofactor(1, 0)
    assert math.isclose(cofactor00, minor00)
    assert math.isclose(cofactor10, -minor10)


def test_determinant_2x2():
    m = Matrix([[1, 5], [-3, 2]])
    assert math.isclose(m.determinant(), 17)


def test_is_invertible_true():
    m = Matrix([[6, 4, 4, 4], [5, 5, 7, 6], [4, -9, 3, -7], [9, 1, 7, -6]])
    assert m.is_invertible()


def test_is_invertible_false_singular():
    m = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]])
    assert not m.is_invertible()


def test_inverse_raises_on_singular():
    m = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]])
    with pytest.raises(ValueError):
        m.inverse()


def test_inverse_property_translation():
    from raytracer.geometry.transform import translation

    m = translation(5, -3, 2)
    assert (m @ m.inverse()) == Matrix.identity(4)


def test_inverse_property_scaling():
    from raytracer.geometry.transform import scaling

    m = scaling(2, 3, 4)
    assert (m @ m.inverse()) == Matrix.identity(4)


def test_inverse_property_rotation():
    from raytracer.geometry.transform import rotation_x

    m = rotation_x(math.pi / 3)
    assert (m @ m.inverse()) == Matrix.identity(4)


def test_inverse_of_product_reverses_order():
    a = Matrix([[3, -9, 7, 3], [3, -8, 2, -9], [-4, 4, 4, 1], [-6, 5, -1, 1]])
    b = Matrix([[8, 2, 2, 2], [3, -1, 7, 0], [7, 0, 5, 4], [6, -2, 0, 5]])
    c = a @ b
    assert c.inverse() == (b.inverse() @ a.inverse())
