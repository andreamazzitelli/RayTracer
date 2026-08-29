from __future__ import annotations

import numpy as np
import math

from raytracer.geometry.vector import Vector

class Point:
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

    def __sub__(self, other: Point | Vector) -> Vector | Point:
        """Point - Point -> Vector; Point - Vector -> Point.

        Note: a single __sub__ signature can't cleanly express this overload
        in the type system alone. Consider @overload from typing, or splitting
        into two explicit methods (e.g. subtract_point / subtract_vector) if
        operator overloading here starts to feel like it's fighting mypy.
        """

        if isinstance(other, Point):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        if isinstance(other, Vector):
            return Point(self.x - other.x, self.y - other.y, self.z - other.z)

        return NotImplemented

    def __add__(self, other: Vector) -> Point:
        return Point( self.x + other.x, self.y + other.y, self.z + other.z)

    def __eq__(self, other: object) -> bool: 
        """Tolerance-based equality, not exact float comparison."""

        if not isinstance(other, Point):
            return NotImplemented

        return (
            math.isclose(self.x, other.x, abs_tol=1e-9) 
            and math.isclose(self.y, other.y,  abs_tol=1e-9) 
            and math.isclose(self.z, other.z, abs_tol=1e-9)
        )

    def __repr__(self) -> str:
        return f"Point({self.x},{self.y},{self.z})"