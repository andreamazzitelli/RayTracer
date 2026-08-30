from __future__ import annotations

import math
import random

from raytracer.geometry.computations import Computations, prepare_computations
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.sampling import random_cosine_weighted_hemisphere
from raytracer.geometry.transform import scaling
from raytracer.image.color import Color
from raytracer.rendering.light import PointLight
from raytracer.rendering.lighting import lighting
from raytracer.rendering.material import Material
from raytracer.shapes.sphere import Sphere

MAX_REFLECTION_DEPTH = 5
MAX_DEPTH_BEFORE_ROULETTE = 3
ROULETTE_SURVIVAL_PROBABILITY = 0.8


class World:
    __slots__ = ("objects", "lights")

    def __init__(self, objects=None, lights=None) -> None:
        self.objects = objects if objects is not None else []
        self.lights = lights if lights is not None else []

    def intersect(self, ray: Ray):
        intersections = []
        for obj in self.objects:
            intersections.extend(obj.intersect(ray))
        intersections.sort(key=lambda i: i.t)
        return intersections

    def is_shadowed(self, point: Point, light: PointLight) -> bool:
        light_vector = light.position - point
        distance = light_vector.magnitude()
        direction = light_vector.normalize()
        shadow_ray = Ray(point, direction)
        shadow_hit = hit(self.intersect(shadow_ray))
        return shadow_hit is not None and shadow_hit.t < distance

    @staticmethod
    def default_world() -> "World":
        light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))
        s1 = Sphere(Point(0, 0, 0), 1.0)
        s1.material = Material(Color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)
        s2 = Sphere(Point(0, 0, 0), 1.0)
        s2.transform = scaling(0.5, 0.5, 0.5)
        return World(objects=[s1, s2], lights=[light])

    def shade_hit(self, comps: Computations, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
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
        intersections = self.intersect(ray)
        intersection = hit(intersections)
        if intersection is None:
            return Color(0, 0, 0)
        comps = prepare_computations(intersection, ray, intersections)
        return self.shade_hit(comps, remaining)

    def reflected_color(self, comps: Computations, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
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
            return Color(0, 0, 0)

        cos_t = math.sqrt(1.0 - sin2_t)
        direction = (comps.normal_vector * (n_ratio * cos_i - cos_t)) - (comps.eye_vector * n_ratio)

        refract_ray = Ray(comps.under_point, direction)
        color = self.color_at(refract_ray, remaining - 1)
        return color * comps.object.material.transparency

    def path_trace(self, ray: Ray, depth: int, rng: random.Random, _bounce: int = 0) -> Color:
        """One Monte Carlo sample of the rendering equation along `ray`.
        A single call is a single noisy sample, not a converged pixel
        color — call many times per pixel and average (see
        path_tracer.render_path_traced).

        `depth`: hard recursion-depth safety cap, decremented each
        bounce; at depth <= 0, only direct lighting + emission is
        returned (no further indirect bounce).

        `_bounce`: internal, increments each recursive call regardless
        of `depth`'s starting value — used purely to trigger Russian
        roulette after MAX_DEPTH_BEFORE_ROULETTE bounces, independent of
        how large a `depth` budget the caller passed in. Callers should
        never pass this explicitly.
        """
        intersections = self.intersect(ray)
        intersection = hit(intersections)

        if intersection is None:
            return Color(0, 0, 0)

        comps = prepare_computations(intersection, ray, intersections)
        material = comps.object.material
        emitted = material.emissive

        # Direct lighting only (no reflection/refraction recursion) —
        # reuses shade_hit with remaining=0, which is exactly what
        # color_at(ray, remaining=0) also computes. This equivalence is
        # deliberate: at depth<=0 this method must match color_at's
        # direct-lighting-only result exactly.
        direct = self.shade_hit(comps, remaining=0)

        if depth <= 0:
            return emitted + direct

        roulette_weight = 1.0
        if _bounce >= MAX_DEPTH_BEFORE_ROULETTE:
            if rng.random() > ROULETTE_SURVIVAL_PROBABILITY:
                return emitted
            roulette_weight = 1.0 / ROULETTE_SURVIVAL_PROBABILITY

        sample_direction, _pdf = random_cosine_weighted_hemisphere(comps.normal_vector, rng)
        bounce_ray = Ray(comps.over_point, sample_direction)
        incoming = self.path_trace(bounce_ray, depth - 1, rng, _bounce=_bounce + 1)

        indirect = material.color.hadamard(incoming) * roulette_weight

        return emitted + direct + indirect


    # def path_trace(self, ray: Ray, depth: int, rng) -> Color:
    #     """Estimate radiance along `ray` via Monte Carlo path tracing:
    #     direct lighting (same as shade_hit) plus a single stochastic
    #     indirect-diffuse bounce, deterministic reflection/refraction,
    #     and Russian-roulette termination for the indirect bounce so
    #     recursion is bounded without introducing bias."""
    #     import math
    #     from raytracer.geometry.sampling import random_cosine_weighted_hemisphere

    #     intersections = self.intersect(ray)
    #     intersection = hit(intersections)
    #     if intersection is None:
    #         return Color(0, 0, 0)

    #     comps = prepare_computations(intersection, ray, intersections)
    #     material = comps.object.material

    #     direct = Color(0, 0, 0)
    #     for light in self.lights:
    #         in_shadow = self.is_shadowed(comps.over_point, light)
    #         direct = direct + lighting(
    #             material, light, comps.over_point,
    #             comps.eye_vector, comps.normal_vector, in_shadow,
    #         )

    #     if depth <= 0:
    #         return direct

    #     albedo = material.color * material.diffuse
    #     survival = min(max(albedo.r, albedo.g, albedo.b), 0.95)

    #     indirect = Color(0, 0, 0)
    #     if survival > 1e-6 and rng.random() < survival:
    #         direction, pdf = random_cosine_weighted_hemisphere(comps.normal_vector, rng)
    #         if pdf > 1e-6:
    #             bounce_ray = Ray(comps.over_point, direction)
    #             incoming = self.path_trace(bounce_ray, depth - 1, rng)
    #             # Cosine-weighted sampling cancels cos(theta)/pdf exactly
    #             # for a Lambertian BRDF -- see docs/15-path-tracing.md.
    #             # Divide by survival to keep the Russian-roulette estimator unbiased.
    #             indirect = albedo.hadamard(incoming) * (1.0 / survival)

    #     reflected = Color(0, 0, 0)
    #     if material.reflective > 1e-9:
    #         reflect_ray = Ray(comps.over_point, comps.reflect_vector)
    #         reflected = self.path_trace(reflect_ray, depth - 1, rng) * material.reflective

    #     refracted = Color(0, 0, 0)
    #     if material.transparency > 1e-9:
    #         n_ratio = comps.n1 / comps.n2
    #         cos_i = comps.eye_vector.dot(comps.normal_vector)
    #         sin2_t = (n_ratio ** 2) * (1 - cos_i ** 2)
    #         if sin2_t <= 1:
    #             cos_t = math.sqrt(1.0 - sin2_t)
    #             refract_dir = (comps.normal_vector * (n_ratio * cos_i - cos_t)) - (
    #                 comps.eye_vector * n_ratio
    #             )
    #             refract_ray = Ray(comps.under_point, refract_dir)
    #             refracted = self.path_trace(refract_ray, depth - 1, rng) * material.transparency

    #     return direct + indirect + reflected + refracted


def hit(intersections):
    for intersection in intersections:
        if intersection.t > 0:
            return intersection
    return None