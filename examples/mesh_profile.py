# examples/mesh_profile.py
"""
Profiles a scene containing a single triangle-mesh sphere (generated
in-memory, not loaded from a file), to confirm Triangle.intersect is
actually the bottleneck before building a BVH around it.

Usage:
    PYTHONPATH=src python examples/mesh_profile.py --triangles 800
"""

from __future__ import annotations

import argparse
import cProfile
import io
import math
import pstats

from raytracer.geometry.point import Point
from raytracer.geometry.transform import view_transform
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.image.ppm import write_ppm
from raytracer.rendering.camera import Camera
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material
from raytracer.rendering.renderer import render
from raytracer.scene.mesh_generators import generate_uv_sphere
from raytracer.scene.world import World
from raytracer.shapes.plane import Plane


def build_mesh_world(rings: int, segments: int) -> World:
    material = Material(Color(0.6, 0.8, 1.0), diffuse=0.7, specular=0.3)
    triangles = generate_uv_sphere(radius=1.5, rings=rings, segments=segments, material=material)

    floor = Plane(material=Material(Color(1, 1, 1), specular=0))
    light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    return World(objects=[floor, *triangles], lights=[light])


def rings_segments_for_target(triangle_count: int) -> tuple[int, int]:
    """Roughly solve 2*rings*segments = triangle_count for rings == segments."""
    n = max(4, round(math.sqrt(triangle_count / 2)))
    return n, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triangles", type=int, default=800)
    parser.add_argument("--hsize", type=int, default=200)
    parser.add_argument("--vsize", type=int, default=100)
    args = parser.parse_args()

    rings, segments = rings_segments_for_target(args.triangles)
    world = build_mesh_world(rings, segments)
    actual_triangle_count = len(world.objects) - 1  # minus the floor

    print(f"Mesh: {rings} rings x {segments} segments = {actual_triangle_count} triangles")

    camera = Camera(
        args.hsize, args.vsize, math.pi / 3,
        transform=view_transform(Point(0, 1.5, -5), Point(0, 0.5, 0), Vector(0, 1, 0)),
    )

    profiler = cProfile.Profile()
    profiler.enable()
    canvas = render(camera, world, show_progress=False)
    profiler.disable()

    write_ppm(canvas, "renders/mesh_profile.ppm")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)
    print(stream.getvalue())


if __name__ == "__main__":
    main()