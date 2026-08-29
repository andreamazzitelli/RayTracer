from __future__ import annotations

from raytracer.geometry.point import Point
from raytracer.image.color import Color


class PointLight:
    """A light source with no size, emitting uniformly in all directions
    from a single position."""

    __slots__ = ("position", "intensity")

    def __init__(self, position: Point, intensity: Color) -> None:
        self.position = position
        self.intensity = intensity

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointLight):
            return NotImplemented

        return (
            self.position == other.position
            and self.intensity == other.intensity
        )

    def __repr__(self) -> str:
        return f"PointLight positioned at {self.position} with intensity {self.intensity}"