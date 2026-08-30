"""
examples/generate_render.py

A single, configurable entry point for generating renders — either the
existing deterministic Phong/reflection/refraction pipeline or the
Phase 15 path tracer, with scene composition, camera placement, and
output all controllable via command-line arguments.

Examples:
    # Deterministic render, default scene, default camera
    PYTHONPATH=src python examples/generate_render.py

    # Higher resolution, custom output path
    PYTHONPATH=src python examples/generate_render.py --hsize 400 --vsize 200 --output renders/big.ppm

    # Path-traced render with 128 samples per pixel, fixed seed for reproducibility
    PYTHONPATH=src python examples/generate_render.py --mode path-trace --samples 128 --seed 7

    # Custom camera position/target/fov
    PYTHONPATH=src python examples/generate_render.py \\
        --camera-from 0 2 -6 --camera-to 0 1 0 --fov 60

    # Also save a PNG alongside the PPM (requires Pillow)
    PYTHONPATH=src python examples/generate_render.py --png
"""

from __future__ import annotations

import argparse
import math
import os
import time

from raytracer.geometry.point import Point
from raytracer.geometry.transform import scaling, translation, view_transform
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.image.ppm import write_ppm
from raytracer.rendering.camera import Camera
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material
from raytracer.rendering.renderer import render
from raytracer.rendering.path_tracer import render_path_traced
from raytracer.scene.world import World
from raytracer.shapes.plane import Plane
from raytracer.shapes.sphere import Sphere


def build_default_scene(emissive_ceiling: bool = False) -> World:
    """A simple three-sphere-and-floor scene, matching the one used
    earlier in this project for both deterministic and path-traced
    testing. `emissive_ceiling` swaps the point light for an emissive
    plane, needed for a meaningful path-traced result (path tracing
    needs *something* emitting light to bounce off of)."""
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

    objects = [floor, middle, right, left]

    if emissive_ceiling:
        ceiling = Plane(
            material=Material(Color(0, 0, 0), emissive=Color(4, 4, 4)),
            transform=translation(0, 5, 0),
        )
        objects.append(ceiling)
        lights = []
    else:
        lights = [PointLight(Point(-10, 10, -10), Color(1, 1, 1))]

    return World(objects=objects, lights=lights)


def build_camera(args: argparse.Namespace) -> Camera:
    frm = Point(*args.camera_from)
    to = Point(*args.camera_to)
    up = Vector(0, 1, 0)
    fov_radians = math.radians(args.fov)

    return Camera(
        args.hsize, args.vsize, fov_radians,
        transform=view_transform(frm, to, up),
    )


def convert_to_png(ppm_path: str) -> str:
    from PIL import Image

    png_path = os.path.splitext(ppm_path)[0] + ".png"
    Image.open(ppm_path).save(png_path)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a render with configurable scene, camera, and rendering mode."
    )

    parser.add_argument("--hsize", type=int, default=200, help="Output width in pixels.")
    parser.add_argument("--vsize", type=int, default=100, help="Output height in pixels.")
    parser.add_argument(
        "--output", type=str, default="renders/render.ppm",
        help="Output PPM file path.",
    )

    parser.add_argument(
        "--mode", choices=["deterministic", "path-trace"], default="deterministic",
        help="'deterministic' uses the Phong/reflection/refraction pipeline (fast, "
             "noise-free). 'path-trace' uses Monte Carlo path tracing (slower, "
             "converges toward a physically based result as --samples increases).",
    )
    parser.add_argument(
        "--samples", type=int, default=16,
        help="Samples per pixel, only used in --mode path-trace.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=5,
        help="Max recursion depth for path tracing (hard safety cap; "
             "Russian roulette usually terminates paths earlier). "
             "Only used in --mode path-trace.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for path tracing, for reproducible noisy renders. "
             "Ignored in deterministic mode.",
    )

    parser.add_argument(
        "--fov", type=float, default=60.0,
        help="Camera field of view, in degrees.",
    )
    parser.add_argument(
        "--camera-from", type=float, nargs=3, default=[0, 1.5, -5],
        metavar=("X", "Y", "Z"), help="Camera position.",
    )
    parser.add_argument(
        "--camera-to", type=float, nargs=3, default=[0, 1, 0],
        metavar=("X", "Y", "Z"), help="Point the camera looks at.",
    )

    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar (useful when scripting many renders).",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="Also save a .png alongside the .ppm (requires Pillow: pip install pillow).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    emissive_ceiling = args.mode == "path-trace"
    world = build_default_scene(emissive_ceiling=emissive_ceiling)
    camera = build_camera(args)

    show_progress = not args.no_progress

    print(f"Mode: {args.mode}, resolution: {args.hsize}x{args.vsize}")
    start = time.perf_counter()

    if args.mode == "path-trace":
        print(f"Samples per pixel: {args.samples}, seed: {args.seed}")
        canvas = render_path_traced(
            camera, world,
            samples_per_pixel=args.samples,
            max_depth=args.max_depth,
            seed=args.seed,
            show_progress=show_progress,
        )
    else:
        canvas = render(camera, world, show_progress=show_progress)

    elapsed = time.perf_counter() - start

    write_ppm(canvas, args.output)
    print(f"Rendered in {elapsed:.2f}s -> {args.output}")

    if args.png:
        try:
            png_path = convert_to_png(args.output)
            print(f"Saved PNG -> {png_path}")
        except ImportError:
            print("Pillow not installed — skipping PNG conversion "
                  "(pip install pillow --break-system-packages)")


if __name__ == "__main__":
    main()