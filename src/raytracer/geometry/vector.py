from __future__ import annotations
import math

# vector.py
class Vector:
    __slots__ = ("_x", "_y", "_z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self._x + other._x, self._y + other._y, self._z + other._z)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self._x - other._x, self._y - other._y, self._z - other._z)

    def __neg__(self) -> "Vector":
        return Vector(-self._x, -self._y, -self._z)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self._x * scalar, self._y * scalar, self._z * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector":
        return Vector(self._x / scalar, self._y / scalar, self._z / scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return (
            math.isclose(self._x, other._x, abs_tol=1e-9)
            and math.isclose(self._y, other._y, abs_tol=1e-9)
            and math.isclose(self._z, other._z, abs_tol=1e-9)
        )

    def magnitude(self) -> float:
        return math.sqrt(self._x**2 + self._y**2 + self._z**2)

    def normalize(self) -> "Vector":
        m = self.magnitude()
        if math.isclose(m, 0, abs_tol=1e-9):
            return Vector(0, 0, 0)
        return self / m

    def dot(self, other: "Vector") -> float:
        return self._x * other._x + self._y * other._y + self._z * other._z

    def cross(self, other: "Vector") -> "Vector":
        return Vector(
            self._y * other._z - self._z * other._y,
            self._z * other._x - self._x * other._z,
            self._x * other._y - self._y * other._x,
        )

    def reflect(self, normal: "Vector") -> "Vector":
        return self - (normal * (2 * self.dot(normal)))