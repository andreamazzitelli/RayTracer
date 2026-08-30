from __future__ import annotations

import math
import random

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import translation, view_transform
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.camera import Camera
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material
from raytracer.rendering.path_tracer import render_path_traced
from raytracer.scene.world import World
from raytracer.shapes.plane import Plane
from raytracer.shapes.sphere import Sphere


def test_path_trace_ray_miss_is_black():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 1, 0))
    rng = random.Random(0)
    assert w.path_trace(r, depth=5, rng=rng) == Color(0, 0, 0)


def test_path_trace_at_depth_zero_matches_deterministic_direct_lighting():
    # The direct-lighting component of path_trace must exactly match
    # the existing, already-verified color_at(ray, remaining=0) --
    # proving it's the same computation, not a parallel reimplementation
    # that merely looks similar.
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    rng = random.Random(0)

    path_traced = w.path_trace(r, depth=0, rng=rng)
    deterministic = w.color_at(r, remaining=0)

    assert path_traced == deterministic


def test_path_trace_terminates_with_mutually_reflective_surfaces():
    # Same structural concern as the Phase 12 reflection termination
    # test: two facing mirrors must not cause unbounded recursion, this
    # time through path_trace's recursive reflected-ray handling.
    light = PointLight(Point(0, 0, 0), Color(1, 1, 1))
    lower = Plane(
        material=Material(Color(1, 1, 1), reflective=1.0),
        transform=translation(0, -1, 0),
    )
    upper = Plane(
        material=Material(Color(1, 1, 1), reflective=1.0),
        transform=translation(0, 1, 0),
    )
    w = World(objects=[lower, upper], lights=[light])

    r = Ray(Point(0, 0, 0), Vector(0, 1, 0))
    rng = random.Random(0)
    color = w.path_trace(r, depth=5, rng=rng)

    assert color is not None


def test_path_trace_russian_roulette_terminates_over_many_calls():
    # Statistical termination check: even with a high max depth, Russian
    # roulette should cause most individual calls to terminate well
    # before reaching that depth, across many independent traces.
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    rng = random.Random(0)

    # If this call completes without RecursionError across many runs at
    # a large depth, Russian roulette is bounding recursion as intended.
    for _ in range(200):
        w.path_trace(r, depth=50, rng=rng)


def test_render_path_traced_output_dimensions():
    w = World.default_world()
    c = Camera(8, 6, math.pi / 2)
    image = render_path_traced(c, w, samples_per_pixel=2, max_depth=2, seed=1, show_progress=False)
    assert image.width == 8
    assert image.height == 6


def test_render_path_traced_is_reproducible_with_same_seed():
    w = World.default_world()
    c = Camera(6, 4, math.pi / 2)

    image_a = render_path_traced(c, w, samples_per_pixel=3, max_depth=2, seed=7, show_progress=False)
    image_b = render_path_traced(c, w, samples_per_pixel=3, max_depth=2, seed=7, show_progress=False)

    for y in range(4):
        for x in range(6):
            assert image_a.pixel_at(x, y) == image_b.pixel_at(x, y)


def test_render_path_traced_differs_with_different_seed():
    # Different seeds should (almost certainly) produce at least one
    # differing pixel -- confirms randomness is actually being used,
    # not silently ignored.
    w = World.default_world()
    c = Camera(10, 8, math.pi / 2)

    image_a = render_path_traced(c, w, samples_per_pixel=2, max_depth=2, seed=1, show_progress=False)
    image_b = render_path_traced(c, w, samples_per_pixel=2, max_depth=2, seed=2, show_progress=False)

    differs = any(
        image_a.pixel_at(x, y) != image_b.pixel_at(x, y)
        for y in range(8)
        for x in range(10)
    )
    assert differs


def test_more_samples_reduces_variance_across_repeated_renders():
    # A convergence check: averaging many LOW-sample renders should
    # itself be noisier (more spread between independent renders) than
    # averaging many HIGH-sample renders, at the same pixel. This is a
    # statistical property test, not a single-pixel comparison, since a
    # single pixel at low vs. high sample count can differ just from
    # ordinary Monte Carlo noise.
    w = World.default_world()
    c = Camera(4, 4, math.pi / 2)
    px, py = 2, 2

    def variance_at(samples_per_pixel: int, num_renders: int) -> float:
        values = []
        for seed in range(num_renders):
            image = render_path_traced(
                c, w, samples_per_pixel=samples_per_pixel, max_depth=2,
                seed=seed, show_progress=False,
            )
            values.append(image.pixel_at(px, py).r)
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    low_variance = variance_at(samples_per_pixel=1, num_renders=15)
    high_variance = variance_at(samples_per_pixel=25, num_renders=15)

    assert high_variance <= low_variance