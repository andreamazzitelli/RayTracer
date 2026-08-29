"""
examples/render_scene.py

Builds a simple scene (a floor plane and three spheres, one reflective)
and renders it to a PPM file. Run from the project root:

    PYTHONPATH=src python examples/render_scene.py
"""

from __future__ import annotations

import math
import sys
import time

from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector
from raytracer.geometry.transform import translation, scaling, rotation_y, view_transform
from raytracer.image.color import Color
from raytracer.image.ppm import write_ppm
from raytracer.rendering.camera import Camera
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material
from raytracer.rendering.renderer import render
from raytracer.scene.world import World
from raytracer.shapes.plane import Plane
from raytracer.shapes.sphere import Sphere


def build_scene() -> World:
    floor = Plane(material=Material(Color(1, 0.9, 0.9), specular=0))

    middle = Sphere(
        Point(0, 0, 0), 1,
        transform=translation(-0.5, 1, 0.5),
        material=Material(Color(0.1, 1, 0.5), diffuse=0.7, specular=0.3),
    )

    right = Sphere(
        Point(0, 0, 0), 1,
        transform=translation(1.5, 0.5, -0.5) @ scaling(0.5, 0.5, 0.5),
        material=Material(Color(0.5, 1, 0.1), diffuse=0.7, specular=0.3, reflective=0.4),
    )

    left = Sphere(
        Point(0, 0, 0), 1,
        transform=translation(-1.5, 0.33, -0.75) @ scaling(0.33, 0.33, 0.33),
        material=Material(Color(1, 0.8, 0.1), diffuse=0.7, specular=0.3),
    )

    light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    return World(objects=[floor, middle, right, left], lights=[light])


def build_camera(hsize: int, vsize: int) -> Camera:
    return Camera(
        hsize, vsize, math.pi / 3,
        transform=view_transform(Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0)),
    )


def main() -> None:
    hsize = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    vsize = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out_path = sys.argv[3] if len(sys.argv) > 3 else "renders/scene.ppm"

    world = build_scene()
    camera = build_camera(hsize, vsize)

    start = time.perf_counter()
    canvas = render(camera, world)
    elapsed = time.perf_counter() - start

    write_ppm(canvas, out_path)
    print(f"Rendered {hsize}x{vsize} to {out_path} in {elapsed:.3f}s")


if __name__ == "__main__":
    main()