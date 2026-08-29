from __future__ import annotations

import numpy as np
import math


class Color:
    """An RGB color as relative intensities. Not clamped or scaled here —
    values may legitimately exceed [0,1] or be negative as an intermediate
    result of combining multiple light contributions."""

    __slots__ = ("_r", "_g", "_b")

    def __init__(self, r: float, g: float, b: float) -> None:
        self._r = float(r)
        self._g = float(g)
        self._b = float(b)

    @property
    def r(self) -> float:
        return self._r

    @property
    def g(self) -> float:
        return self._g

    @property
    def b(self) -> float: 
        return self._b

    def __add__(self, other: Color) -> Color:
        """Additive color combination, e.g. summing multiple light contributions."""
        return Color(self.r + other.r, self.g + other.g, self.b + other.b,)

    def __mul__(self, scalar: float) -> Color:
        """Scale intensity, e.g. attenuating by light intensity or material coefficient."""
        return Color(self.r * scalar, self.g * scalar, self.b * scalar)

    def __rmul__(self, scalar: float) -> Color:
        return self.__mul__(scalar)


    def hadamard(self, other: Color) -> Color:
        """Return the component-wise product of two colors."""
        values = np.multiply(
            np.array([self.r, self.g, self.b]),
            np.array([other.r, other.g, other.b]),
        )

        return Color.from_np_array(values)

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, Color):
            return NotImplemented

        return (
            math.isclose(self.r, other.r, abs_tol=1e-9) 
            and math.isclose(self.g, other.g,  abs_tol=1e-9) 
            and math.isclose(self.b, other.b, abs_tol=1e-9)
        )

    def __repr__(self) -> str:
        return f"Color({self.r}, {self.g}, {self.b})"

    @staticmethod
    def from_np_array(array: np.ndarray) -> Color:
        if array.shape != (3,):
            raise ValueError("Expected a NumPy array with shape (3,)")

        return Color(
            float(array[0]),
            float(array[1]),
            float(array[2])
        )