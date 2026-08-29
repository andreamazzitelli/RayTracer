from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector


class Ray:
    """A ray defined by an origin point and a direction vector: R(t) = O + tD."""

    __slots__ = ("origin", "direction")

    def __init__(self, origin: Point, direction: Vector) -> None:
        self.origin = origin
        self.direction = direction

    def position_at(self, t: float) -> Point:
        """Evaluate R(t) = O + tD."""
        return self.origin + (t * self.direction)
        

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ray):
            return NotImplemented

        return (
            self.origin == other.origin
            and self.direction == other.direction  
        )

    def __repr__(self) -> str:
        return f"Origin: {self.origin}; Direction: {self.direction}"