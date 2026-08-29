from __future__ import annotations

import math

from raytracer.geometry.computations import Computations, prepare_computations
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import scaling
from raytracer.image.color import Color
from raytracer.rendering.light import PointLight
from raytracer.rendering.lighting import lighting
from raytracer.rendering.material import Material
from raytracer.shapes.sphere import Sphere

MAX_REFLECTION_DEPTH = 5


class World:
    """A collection of objects and lights forming a scene, along with the
    logic for determining what color a ray sees when fired into it."""

    __slots__ = ("objects", "lights")

    def __init__(self, objects: list | None = None, lights: list | None = None) -> None:
        self.objects = objects if objects is not None else []
        self.lights = lights if lights is not None else []

    def intersect(self, ray: Ray):
        """All intersections of `ray` against every object in the world,
        sorted ascending by t."""
        intersections = []
        for obj in self.objects:
            intersections.extend(obj.intersect(ray))
        intersections.sort(key=lambda i: i.t)
        return intersections

    def is_shadowed(self, point: Point, light: PointLight) -> bool:
        """True if `point` is blocked from `light` by any object in the world."""
        light_vector = light.position - point
        distance = light_vector.magnitude()
        direction = light_vector.normalize()

        shadow_ray = Ray(point, direction)
        shadow_hit = hit(self.intersect(shadow_ray))

        return shadow_hit is not None and shadow_hit.t < distance

    @staticmethod
    def default_world() -> "World":
        """Standard test-fixture world: two concentric unit spheres at the
        origin (outer full-sized, inner scaled to half), fixed materials,
        one light. Shared across tests so expected values only need to be
        hand-computed once."""
        light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

        s1 = Sphere(Point(0, 0, 0), 1.0)
        s1.material = Material(Color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)

        s2 = Sphere(Point(0, 0, 0), 1.0)
        s2.transform = scaling(0.5, 0.5, 0.5)

        return World(objects=[s1, s2], lights=[light])

    def shade_hit(self, comps: Computations, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
        """Shade a single precomputed intersection: sum lighting()
        contributions from every light (accounting for shadows), plus
        any reflected color contribution."""
        surface_color = Color(0, 0, 0)
        for light in self.lights:
            in_shadow = self.is_shadowed(comps.over_point, light)
            surface_color = surface_color + lighting(
                comps.object.material, light, comps.over_point,
                comps.eye_vector, comps.normal_vector, in_shadow,
            )

        reflected = self.reflected_color(comps, remaining)
        refracted = self.refracted_color(comps, remaining)

        return surface_color + reflected + refracted

    def color_at(self, ray: Ray, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
        """The color seen by `ray` in this world: find the closest hit
        (if any), shade it, or return black if the ray hits nothing."""
        intersections = self.intersect(ray)
        intersection = hit(intersections)

        if intersection is None:
            return Color(0, 0, 0)

        comps = prepare_computations(intersection, ray, intersections)
        return self.shade_hit(comps, remaining)

    def reflected_color(self, comps: Computations, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
        """The color contributed by reflection at this intersection.
        Returns black if the recursion depth is exhausted or the
        material isn't reflective, to avoid unnecessary secondary rays
        and to guarantee termination between mutually reflective surfaces."""
        if remaining <= 0:
            return Color(0, 0, 0)

        if math.isclose(comps.object.material.reflective, 0, abs_tol=1e-9):
            return Color(0, 0, 0)

        reflect_ray = Ray(comps.over_point, comps.reflect_vector)
        color = self.color_at(reflect_ray, remaining - 1)

        return color * comps.object.material.reflective

    def refracted_color(self, comps: Computations, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
        if remaining <= 0:
            return Color(0, 0, 0)

        if math.isclose(comps.object.material.transparency, 0, abs_tol=1e-9):
            return Color(0, 0, 0)

        n_ratio = comps.n1 / comps.n2
        cos_i = comps.eye_vector.dot(comps.normal_vector)
        sin2_t = (n_ratio**2) * (1 - cos_i**2)

        if sin2_t > 1:
            return Color(0, 0, 0)  # total internal reflection

        cos_t = math.sqrt(1.0 - sin2_t)
        direction = (comps.normal_vector * (n_ratio * cos_i - cos_t)) - (comps.eye_vector * n_ratio)

        refract_ray = Ray(comps.under_point, direction)
        color = self.color_at(refract_ray, remaining - 1)

        return color * comps.object.material.transparency

def hit(intersections):
    """The visible intersection: closest with t > 0. None if no such hit
    exists. Kept as a free function since it operates purely on a list of
    Intersections, and relies on that list already being sorted ascending
    by t (as World.intersect guarantees)."""
    for intersection in intersections:
        if intersection.t > 0:
            return intersection
    return None