from __future__ import annotations

import math
import random

from raytracer.geometry.vector import Vector


def _orthonormal_basis(normal: Vector):
    """Arbitrary orthonormal basis with `normal` as one axis, used to
    transform a locally-sampled hemisphere direction into world space."""
    if abs(normal.x) > 0.9:
        helper = Vector(0, 1, 0)
    else:
        helper = Vector(1, 0, 0)

    tangent = helper.cross(normal).normalize()
    bitangent = normal.cross(tangent)
    return tangent, bitangent, normal


def random_cosine_weighted_hemisphere(
    normal: Vector, rng: random.Random
) -> tuple[Vector, float]:
    """One cosine-weighted random direction on the hemisphere around
    `normal`, via Malley's method, plus the probability density (pdf)
    of having sampled that exact direction: pdf = cos(theta) / pi.

    For a Lambertian BRDF (f_r = albedo / pi), the Monte Carlo estimator
    f_r * L_i * cos(theta) / pdf simplifies to just `albedo * L_i` —
    this is why callers combining Lambertian materials with this sampler
    can multiply by the material's color directly without an explicit
    division by pdf. The pdf is still returned explicitly (rather than
    baking that cancellation in silently) so it's available for any
    future non-Lambertian BRDF that doesn't get to skip the division.
    """
    u1 = rng.random()
    u2 = rng.random()

    r = math.sqrt(u1)
    theta = 2 * math.pi * u2
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1 - u1))

    tangent, bitangent, n = _orthonormal_basis(normal)
    direction = (tangent * x + bitangent * y + n * z).normalize()

    cos_theta = z  # z-component in the local frame IS cos(theta) to the normal
    pdf = max(cos_theta, 1e-9) / math.pi

    return direction, pdf