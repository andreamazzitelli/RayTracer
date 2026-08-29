from __future__ import annotations

import numpy as np
import math

from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector


_inverse_call_count = 0

class Matrix:
    """A general MxN matrix, used at 4x4 size for 3D transformations,
    but kept general enough to support smaller sizes for hand-verifying
    determinant/inverse logic in tests later."""

    __slots__ = ("_data", "_flat")

    def __init__(self, rows: list[list[float]]) -> None:
        """Construct from a list of rows, e.g. [[1,0],[0,1]] for a 2x2
        identity. Validate that all rows have equal length."""
        length = len(rows[0])
        for row in rows:
            if len(row) != length:
                raise ValueError("All rows should have the same length")
        self._data = np.array(rows, dtype=np.float64)
        self._flat = tuple(float(v) for row in rows for v in row)



    @classmethod
    def _from_ndarray(cls, array: np.ndarray) -> Matrix:
        """Construct directly from an already-validated ndarray, bypassing
        the list-of-lists validation in __init__ — used internally by methods
        that already have a correctly-shaped NumPy result."""
        instance = cls.__new__(cls)
        instance._data = array.astype(np.float64)
        instance._flat = tuple(float(v) for v in array.flatten())
        return instance

    @staticmethod
    def identity(size: int = 4) -> Matrix:
        """The size x size identity matrix."""
        return Matrix._from_ndarray(np.eye(size))

    @property
    def shape(self) -> tuple[int, int]:
        """(num_rows, num_cols)."""
        return self._data.shape

    def __getitem__(self, row_col: tuple[int, int]) -> float:
        """m[row, col] element access."""
        return float(self._data[row_col[0]][row_col[1]])

    def __eq__(self, other: object) -> bool:
        """Tolerance-based equality, elementwise."""
        if not isinstance(other, Matrix):
            return NotImplemented

        if self.shape != other.shape:
            return False

        return bool(np.allclose(self._data, other._data, atol=1e-9))

    def __repr__(self) -> str:
        return f"Matrix({self._data.tolist()})"

    def __matmul__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        return Matrix._from_ndarray(self._data @ other._data)
        
    def transpose(self) -> Matrix:
        return Matrix._from_ndarray(self._data.T)

    def apply_to_point(self, point: Point) -> Point:
        """Transform a Point: treat it as a homogeneous column [x,y,z,1],
        multiply, and convert the first three components of the result
        back into a Point."""
        m = self._flat  # plain Python tuple of 16 floats, row-major
        x, y, z = point.x, point.y, point.z
        return Point(
            m[0]*x + m[1]*y + m[2]*z + m[3],
            m[4]*x + m[5]*y + m[6]*z + m[7],
            m[8]*x + m[9]*y + m[10]*z + m[11],
        )

    def apply_to_vector(self, vector: Vector) -> Vector:
        """Transform a Vector: treat it as a homogeneous column [x,y,z,0].
        Same multiplication, different w, hence a separate method rather
        than one overloaded operation."""
        m = self._flat
        x, y, z = vector.x, vector.y, vector.z
        return Vector(
            m[0]*x + m[1]*y + m[2]*z,
            m[4]*x + m[5]*y + m[6]*z,
            m[8]*x + m[9]*y + m[10]*z,
        )

    def submatrix(self, row: int, col: int) -> Matrix:
        """Return the matrix formed by deleting the given row and column."""
        arr = np.delete(self._data, row, axis=0)
        return Matrix._from_ndarray(np.delete(arr, col, axis=1))

    def minor(self, row: int, col: int) -> float:
        """The determinant of submatrix(row, col)."""
        return self.submatrix(row, col).determinant()

    def cofactor(self, row: int, col: int) -> float:
        """minor(row, col) with the alternating (-1)^(row+col) sign applied."""
        sign = -1.0 if (row + col) % 2 == 1 else 1.0
        return sign * self.minor(row, col)
        

    def determinant(self) -> float:
        """The determinant of this (square) matrix."""
        return float(np.linalg.det(self._data))

    def is_invertible(self) -> bool:
        """True if determinant() is nonzero (within tolerance)."""
        return not math.isclose(self.determinant(), 0, abs_tol=1e-9)

    def inverse(self) -> Matrix:
        """The inverse matrix."""
        global _inverse_call_count
        _inverse_call_count += 1

        if not self.is_invertible():
            raise ValueError("Matrix is not invertible")

        return Matrix._from_ndarray(np.linalg.inv(self._data))