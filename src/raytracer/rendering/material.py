from __future__ import annotations

import math

from raytracer.image.color import Color


class Material:
    __slots__ = (
        "color", "ambient", "diffuse", "specular", "shininess",
        "reflective", "transparency", "refractive_index", "emissive",
    )

    def __init__(
        self,
        color: Color,
        ambient: float = 0.1,
        diffuse: float = 0.9,
        specular: float = 0.9,
        shininess: float = 200.0,
        reflective: float = 0.0,
        transparency: float = 0.0,
        refractive_index: float = 1.0,
        emissive: Color | None = None,
    ) -> None:
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflective = reflective
        self.transparency = transparency
        self.refractive_index = refractive_index
        self.emissive = emissive if emissive is not None else Color(0, 0, 0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented
        return (
            self.color == other.color
            and math.isclose(self.ambient, other.ambient, abs_tol=1e-9)
            and math.isclose(self.diffuse, other.diffuse, abs_tol=1e-9)
            and math.isclose(self.specular, other.specular, abs_tol=1e-9)
            and math.isclose(self.shininess, other.shininess, abs_tol=1e-9)
            and math.isclose(self.reflective, other.reflective, abs_tol=1e-9)
            and math.isclose(self.transparency, other.transparency, abs_tol=1e-9)
            and math.isclose(self.refractive_index, other.refractive_index, abs_tol=1e-9)
            and self.emissive == other.emissive
        )

    def __repr__(self) -> str:
        return (
            f"Color: {self.color}, Ambient: {self.ambient}, "
            f"Diffuse: {self.diffuse}, Specular: {self.specular}, "
            f"Shininess: {self.shininess}, Reflective: {self.reflective}, "
            f"Transparency: {self.transparency}, RefractiveIndex: {self.refractive_index}, "
            f"Emissive: {self.emissive}"
        )