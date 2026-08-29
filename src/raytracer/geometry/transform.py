from __future__ import annotations

import math
import numpy as np

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector


def translation(x: float, y: float, z: float) -> Matrix:
    """A 4x4 matrix that translates points by (x, y, z).
    Leaves vectors unaffected (w=0 zeroes out the translation column
    when applied via Matrix.apply_to_vector)."""
    transform = np.identity(4, dtype=float)
    transform[0:3, 3] = [x, y, z]
    return Matrix._from_ndarray(transform)


def scaling(x: float, y: float, z: float) -> Matrix:
    """A 4x4 matrix that scales by (x, y, z) along each axis."""
    transform = np.diag([x, y, z, 1.0])
    return Matrix._from_ndarray(transform)


def rotation_x(radians: float) -> Matrix:
    """A 4x4 matrix rotating about the X axis by `radians`."""
    cosine = math.cos(radians)
    sine = math.sin(radians)
    transform = np.identity(4, dtype=float)
    transform[1:3, 1:3] = [
        [cosine, -sine],
        [sine,  cosine]
    ]
    return Matrix._from_ndarray(transform)


def rotation_y(radians: float) -> Matrix:
    """A 4x4 matrix rotating about the Y axis by `radians`."""
    cosine = math.cos(radians)
    sine = math.sin(radians)
    transform = np.identity(4, dtype=float)
    transform[0, 0] = cosine
    transform[0, 2] = sine
    transform[2, 0] = -sine
    transform[2, 2] = cosine
    return Matrix._from_ndarray(transform)


def rotation_z(radians: float) -> Matrix:
    """A 4x4 matrix rotating about the Z axis by `radians`."""
    cosine = math.cos(radians)
    sine = math.sin(radians)
    transform = np.identity(4, dtype=float)
    transform[0:2, 0:2] = [
        [cosine, -sine],
        [sine,  cosine]
    ]
    return Matrix._from_ndarray(transform)


def shearing(
    xy: float, xz: float, yx: float, yz: float, zx: float, zy: float
) -> Matrix:
    """A 4x4 shearing (skew) matrix. Each parameter controls how much
    one axis shifts in proportion to another — e.g. `xy` controls how
    much x moves in proportion to y."""
    transform = np.identity(4, dtype=float)
    transform[0, 1] = xy
    transform[0, 2] = xz
    transform[1, 0] = yx
    transform[1, 2] = yz
    transform[2, 0] = zx
    transform[2, 1] = zy
    return Matrix._from_ndarray(transform)

def view_transform(frm: Point, to: Point, up: Vector) -> Matrix:
    """Build a camera transform: an orthonormal basis (left, true_up,
    -forward) placing a camera at `frm`, looking toward `to`, oriented
    by `up`."""
    forward = (to - frm).normalize()
    upn = up.normalize()
    left = forward.cross(upn)
    true_up = left.cross(forward)

    orientation = Matrix([
        [left.x, left.y, left.z, 0],
        [true_up.x, true_up.y, true_up.z, 0],
        [-forward.x, -forward.y, -forward.z, 0],
        [0, 0, 0, 1],
    ])
    return orientation @ translation(-frm.x, -frm.y, -frm.z)