import math

from raytracer.geometry.intersection import Intersection
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector
from raytracer.shapes.shape import Shape
from raytracer.rendering.material import Material
from raytracer.geometry.matrix import Matrix

class Sphere(Shape):
    """A unit-ish sphere defined by a center and radius, positioned and
    oriented in world space via its inherited `transform`."""

    __slots__ = ("center", "radius")

    def __init__(
        self,
        center: Point,
        radius: float,
        material: Material | None = None,
        transform: Matrix | None = None,
    ) -> None:
        super().__init__(material, transform)
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray) -> list[Intersection]:
        """Transform the ray into this sphere's local (untransformed)
        space via the inverse transform, then solve the quadratic exactly
        as before — the sphere's own math never needs to know it was
        scaled/rotated/moved."""
        local_ray = self._to_local_ray(ray)

        sphere_to_ray = local_ray.origin - self.center
        a = local_ray.direction.dot(local_ray.direction)
        b = 2 * local_ray.direction.dot(sphere_to_ray)
        c = sphere_to_ray.dot(sphere_to_ray) - self.radius**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return []

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        return [Intersection(t1, self), Intersection(t2, self)]

    def normal_at(self, point: Point) -> Vector:
        """`point` is given in WORLD space. Convert to local space, compute
        the normal there (simple, since it's still just (p-c)/r locally),
        then convert the normal back to world space using the inverse
        TRANSPOSE of the transform — not the transform itself.

        Why inverse-transpose: under non-uniform scaling, a normal
        transformed the same way as a point would no longer stay
        perpendicular to the (correctly transformed) surface. The
        inverse-transpose is the standard fix that preserves
        perpendicularity regardless of scale. This is a real, nontrivial
        linear-algebra fact — worth taking on faith for now and verifying
        empirically via tests (a sphere scaled non-uniformly, normal
        checked for perpendicularity to the transformed surface) rather
        than re-deriving from scratch here."""
        local_point = self._inverse_transform.apply_to_point(point)
        local_normal = (local_point - self.center).normalize()
        world_normal = self._inverse_transpose_transform.apply_to_vector(local_normal)
        return world_normal.normalize()

    def _to_local_ray(self, ray: Ray) -> Ray:
        inverse = self._inverse_transform
        return Ray(inverse.apply_to_point(ray.origin), inverse.apply_to_vector(ray.direction))