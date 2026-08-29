from __future__ import annotations

from abc import ABC, abstractmethod

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.material import Material


class Shape(ABC):
    def __init__(self, material: Material | None = None, transform: Matrix | None = None) -> None:
        self.material = material if material is not None else Material(Color(1, 1, 1))
        self._transform = Matrix.identity(4)
        self._inverse_transform = Matrix.identity(4)
        self._inverse_transpose_transform = Matrix.identity(4)
        self.transform = transform if transform is not None else Matrix.identity(4)

    @property
    def transform(self) -> Matrix:
        return self._transform

    @transform.setter
    def transform(self, value: Matrix) -> None:
        """Setting transform recomputes and caches its inverse and
        inverse-transpose ONCE here, rather than paying for a full
        matrix inversion on every single ray this shape is tested
        against — which was the actual measured bottleneck (3.9M
        redundant inverse() calls in a single render)."""
        self._transform = value
        self._inverse_transform = value.inverse()
        self._inverse_transpose_transform = self._inverse_transform.transpose()

    @abstractmethod
    def intersect(self, ray: Ray):
        ...

    @abstractmethod
    def normal_at(self, point: Point) -> Vector:
        ...

    def _to_local_ray(self, ray: Ray) -> Ray:
        inverse = self._inverse_transform
        return Ray(inverse.apply_to_point(ray.origin), inverse.apply_to_vector(ray.direction))