"""
examples/path_tracing_test.py

Validates the path tracer in three stages, cheapest first:
  1. A fast, deterministic check that cosine-weighted samples never fall
     below the hemisphere (catches a sign/math error before wasting time
     on slow renders).
  2. A numeric convergence check: render the same single pixel at
     increasing sample counts and print how much it changes between
     steps — should shrink toward zero as noise averages out.
  3. Full small renders at increasing sample counts, saved as separate
     PPM files so you can visually compare noise levels side by side.

Run:
    PYTHONPATH=src python examples/path_tracing_test.py
"""

from __future__ import annotations

import math
import random
import time

from raytracer.geometry.point import Point
from raytracer.geometry.transform import translation, view_transform
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.image.ppm import write_ppm
from raytracer.rendering.camera import Camera
from raytracer.rendering.material import Material
from raytracer.rendering.path_tracer import _cosine_weighted_sample, trace_path
from raytracer.rendering.renderer import render_path_traced
from raytracer.scene.world import World
from raytracer.shapes.plane import Plane
from raytracer.shapes.sphere import Sphere


def step_1_hemisphere_sanity_check(num_samples: int = 1000) -> None:
    print("=== Step 1: hemisphere sanity check ===")
    rng = random.Random(0)
    normal = Vector(0, 1, 0)

    failures = 0
    for _ in range(num_samples):
        sample = _cosine_weighted_sample(normal, rng)
        if sample.dot(normal) < 0:
            failures += 1

    if failures:
        print(f"FAILED: {failures}/{num_samples} samples fell below the hemisphere.")
        print("Stopping here — fix _orthonormal_basis / _cosine_weighted_sample "
              "before running any renders.")
        raise SystemExit(1)

    print(f"OK: all {num_samples} samples stayed within the hemisphere.\n")


def build_test_scene() -> World:
    """An emissive ceiling lighting a diffuse sphere — no PointLight at
    all, so this isolates the path tracer's own light-transport
    mechanism rather than mixing it with the existing Phong direct-light
    path."""
    ceiling = Plane(
        material=Material(Color(0, 0, 0), emissive=Color(4, 4, 4)),
        transform=translation(0, 5, 0),
    )
    floor = Plane(material=Material(Color(0.8, 0.8, 0.8)))
    sphere = Sphere(
        Point(0, 0, 0), 1,
        transform=translation(0, 1, 0),
        material=Material(Color(0.7, 0.3, 0.3)),
    )
    return World(objects=[ceiling, floor, sphere], lights=[])


def step_2_single_pixel_convergence(world: World, sample_counts: list[int]) -> None:
    print("=== Step 2: single-pixel numeric convergence ===")
    camera = Camera(
        20, 10, math.pi / 3,
        transform=view_transform(Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0)),
    )

    px, py = 10, 5  # a central pixel, likely to see both floor and sphere
    ray = camera.ray_for_pixel(px, py)

    previous_avg = None
    for samples in sample_counts:
        rng = random.Random(42)  # fixed seed: isolates sample-count effect
        accumulated = Color(0, 0, 0)
        for _ in range(samples):
            accumulated = accumulated + trace_path(world, ray, rng)
        avg = accumulated * (1.0 / samples)

        if previous_avg is not None:
            delta = math.sqrt(
                (avg.r - previous_avg.r) ** 2
                + (avg.g - previous_avg.g) ** 2
                + (avg.b - previous_avg.b) ** 2
            )
            print(f"  {samples:>4} samples -> {avg}  (change from previous: {delta:.4f})")
        else:
            print(f"  {samples:>4} samples -> {avg}")

        previous_avg = avg

    print("  Expect the 'change from previous' column to shrink as samples increase.\n")


def step_3_full_renders(world: World, sample_counts: list[int]) -> None:
    print("=== Step 3: full renders at increasing sample counts ===")
    camera = Camera(
        40, 20, math.pi / 3,
        transform=view_transform(Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0)),
    )

    for samples in sample_counts:
        start = time.perf_counter()
        canvas = render_path_traced(camera, world, samples_per_pixel=samples, seed=7)
        elapsed = time.perf_counter() - start

        path = f"renders/path_traced_{samples:04d}spp.ppm"
        write_ppm(canvas, path)
        print(f"  {samples:>4} spp -> {path} ({elapsed:.2f}s)")

    print("\n  Convert to PNG and compare visually — noise should visibly drop as spp increases:")
    print("    python -c \"from PIL import Image; "
          "[Image.open(f'renders/path_traced_{s:04d}spp.ppm')"
          ".save(f'renders/path_traced_{s:04d}spp.png') for s in "
          f"{sample_counts}]\"")


def main() -> None:
    step_1_hemisphere_sanity_check()

    world = build_test_scene()

    convergence_samples = [4, 16, 64, 256, 1024]
    step_2_single_pixel_convergence(world, convergence_samples)

    render_samples = [4, 16, 64, 256]
    step_3_full_renders(world, render_samples)


if __name__ == "__main__":
    main()