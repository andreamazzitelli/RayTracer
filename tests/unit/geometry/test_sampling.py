from __future__ import annotations

import math
import random

from raytracer.geometry.sampling import random_cosine_weighted_hemisphere
from raytracer.geometry.vector import Vector


def test_sampled_directions_are_unit_length():
    rng = random.Random(1)
    normal = Vector(0, 1, 0)
    for _ in range(2000):
        direction, _ = random_cosine_weighted_hemisphere(normal, rng)
        assert math.isclose(direction.magnitude(), 1, abs_tol=1e-9)


def test_sampled_directions_stay_on_correct_hemisphere():
    rng = random.Random(2)
    normal = Vector(0, 1, 0)
    for _ in range(2000):
        direction, _ = random_cosine_weighted_hemisphere(normal, rng)
        assert direction.dot(normal) >= -1e-9


def test_pdf_is_always_positive():
    rng = random.Random(3)
    normal = Vector(0, 1, 0)
    for _ in range(2000):
        _, pdf = random_cosine_weighted_hemisphere(normal, rng)
        assert pdf > 0


def test_sampling_works_for_arbitrary_normal_orientations():
    rng = random.Random(4)
    normals = [
        Vector(1, 0, 0),
        Vector(0, 0, 1),
        Vector(1, 1, 1).normalize(),
        Vector(-1, 0.5, 0.3).normalize(),
    ]
    for normal in normals:
        for _ in range(500):
            direction, pdf = random_cosine_weighted_hemisphere(normal, rng)
            assert math.isclose(direction.magnitude(), 1, abs_tol=1e-9)
            assert direction.dot(normal) >= -1e-9
            assert pdf > 0


def test_directions_concentrate_near_normal_not_uniform():
    # Cosine-weighted sampling should produce more directions close to
    # the normal than far from it -- a rough statistical check that this
    # is NOT uniform hemisphere sampling.
    rng = random.Random(5)
    normal = Vector(0, 1, 0)
    close_to_normal = 0
    total = 5000
    for _ in range(total):
        direction, _ = random_cosine_weighted_hemisphere(normal, rng)
        if direction.dot(normal) > 0.7:  # within ~45 degrees of straight up
            close_to_normal += 1

    # Uniform hemisphere sampling would put only ~13% of samples within
    # cos(theta) > 0.7 (proportional to solid angle); cosine-weighted
    # sampling should put substantially more there.
    assert close_to_normal / total > 0.3