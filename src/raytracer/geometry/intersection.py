from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raytracer.shapes.shape import Shape

import math


class Intersection:
    """Records where along a ray (t) an intersection occurred, and with
    which object — so later stages know what was hit, not just how far."""

    __slots__ = ("t", "object")

    def __init__(self, t: float, obj: "Shape") -> None:
        self.t = t
        self.object = obj

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, Intersection):
            return NotImplemented
        
        return (
            math.isclose(self.t, other.t, abs_tol=1e-9)
            and self.object == other.object
        )
    

    def __repr__(self) -> str:
        return f"Intercepts Object: {self.object} at position {self.t}"