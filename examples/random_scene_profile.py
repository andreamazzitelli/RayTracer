"""
examples/random_scene_profiler.py  (updated: saves + displays each render)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import math
import os
import pstats
import random
import time
from dataclasses import dataclass, field

from raytracer.geometry import matrix as matrix_module
from raytracer.geometry.point import Point
from raytracer.geometry.transform import scaling, translation, view_transform
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.image.ppm import write_ppm
from raytracer.rendering.camera import Camera
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material
from raytracer.rendering.renderer import render
from raytracer.scene.world import World
from raytracer.shapes.plane import Plane
from raytracer.shapes.sphere import Sphere

AVAILABLE_SIZES = [(50, 25), (100, 50), (200, 100), (400, 200)]
OBJECT_COUNT_RANGE = (3, 12)
PLACEMENT_RANGE = (-5.0, 5.0)
SCALE_RANGE = (0.3, 1.5)
RENDER_OUTPUT_DIR = "renders/profiler_batch"


@dataclass
class RunResult:
    scene_id: int
    hsize: int
    vsize: int
    num_objects: int
    num_lights: int
    seconds: float
    image_path: str
    top_functions: list = field(default_factory=list)


def random_color(rng: random.Random) -> Color:
    return Color(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1))


def random_point(rng: random.Random, y_min: float = 0.0) -> Point:
    return Point(
        rng.uniform(*PLACEMENT_RANGE),
        rng.uniform(y_min, PLACEMENT_RANGE[1]),
        rng.uniform(*PLACEMENT_RANGE),
    )


def random_material(rng: random.Random) -> Material:
    return Material(
        random_color(rng),
        ambient=rng.uniform(0.0, 0.3),
        diffuse=rng.uniform(0.4, 0.9),
        specular=rng.uniform(0.0, 0.9),
        shininess=rng.uniform(10, 300),
        reflective=rng.choice([0.0, 0.0, 0.0, rng.uniform(0.1, 0.6)]),
        transparency=rng.choice([0.0, 0.0, 0.0, rng.uniform(0.1, 0.5)]),
        refractive_index=rng.choice([1.0, 1.5]),
    )


def random_sphere(rng: random.Random) -> Sphere:
    center_point = random_point(rng, y_min=0.3)
    scale = rng.uniform(*SCALE_RANGE)
    return Sphere(
        Point(0, 0, 0),
        1.0,
        material=random_material(rng),
        transform=translation(center_point.x, center_point.y, center_point.z)
        @ scaling(scale, scale, scale),
    )


def random_world(rng: random.Random, num_objects: int, num_lights: int) -> World:
    floor = Plane(material=Material(random_color(rng), specular=0.0))
    objects = [floor] + [random_sphere(rng) for _ in range(num_objects)]
    lights = [
        PointLight(random_point(rng, y_min=2.0), random_color(rng))
        for _ in range(num_lights)
    ]
    return World(objects=objects, lights=lights)


def random_camera(rng: random.Random, hsize: int, vsize: int) -> Camera:
    frm = Point(rng.uniform(-8, 8), rng.uniform(1, 6), rng.uniform(-10, -3))
    to = Point(rng.uniform(-1, 1), rng.uniform(0, 2), rng.uniform(-1, 1))
    up = Vector(0, 1, 0)
    fov = rng.uniform(math.pi / 4, math.pi / 2)
    return Camera(hsize, vsize, fov, transform=view_transform(frm, to, up))


def profile_single_render(
    scene_id: int, rng: random.Random, output_dir: str, top_n: int = 10
) -> RunResult:
    hsize, vsize = rng.choice(AVAILABLE_SIZES)
    num_objects = rng.randint(*OBJECT_COUNT_RANGE)
    num_lights = rng.randint(1, 3)

    world = random_world(rng, num_objects, num_lights)
    camera = random_camera(rng, hsize, vsize)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    canvas = render(camera, world, show_progress=False)
    profiler.disable()
    elapsed = time.perf_counter() - start

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"scene_{scene_id:03d}.ppm")
    write_ppm(canvas, image_path)

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")

    top_functions = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        top_functions.append({
            "function": f"{func[0].split('/')[-1]}:{func[1]}({func[2]})",
            "ncalls": nc,
            "tottime": tt,
            "cumtime": ct,
        })
    top_functions.sort(key=lambda r: r["cumtime"], reverse=True)
    top_functions = top_functions[:top_n]

    return RunResult(
        scene_id=scene_id,
        hsize=hsize,
        vsize=vsize,
        num_objects=num_objects + 1,
        num_lights=num_lights,
        seconds=elapsed,
        image_path=image_path,
        top_functions=top_functions,
    )


def run_experiment(
    num_scenes: int, seed: int | None = None, output_dir: str = RENDER_OUTPUT_DIR
) -> list[RunResult]:
    rng = random.Random(seed)
    results = []

    for i in range(num_scenes):
        print(f"--- Scene {i + 1}/{num_scenes} ---")

        matrix_module._inverse_call_count = 0  # reset before this scene
        result = profile_single_render(i, rng, output_dir)
        inverse_calls = matrix_module._inverse_call_count  # read immediately after

        print(
            f"  {result.hsize}x{result.vsize}, "
            f"{result.num_objects} objects, {result.num_lights} lights "
            f"-> {result.seconds:.3f}s -> {result.image_path} "
            f"| inverse() calls: {inverse_calls}"
        )
        results.append(result)

    return results


def _load_ppm_as_array(path: str):
    """Load a PPM file into a numpy array via Pillow, for display purposes."""
    from PIL import Image
    import numpy as np

    img = Image.open(path)
    return np.array(img)


def display_renders(results: list[RunResult], columns: int = 4) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping image display.")
        return

    columns = min(columns, len(results))
    rows = (len(results) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 2), squeeze=False)
    axes = axes.flatten()

    for ax, result in zip(axes, results):
        try:
            image_array = _load_ppm_as_array(result.image_path)
            ax.imshow(image_array)
        except ImportError:
            ax.text(0.5, 0.5, "Pillow not installed", ha="center", va="center")

        ax.set_title(
            f"#{result.scene_id} {result.hsize}x{result.vsize}\n"
            f"{result.num_objects} obj, {result.seconds:.2f}s",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def summarize(results: list[RunResult]) -> None:
    try:
        import pandas as pd

        summary_df = pd.DataFrame([
            {
                "scene_id": r.scene_id,
                "resolution": f"{r.hsize}x{r.vsize}",
                "pixels": r.hsize * r.vsize,
                "num_objects": r.num_objects,
                "num_lights": r.num_lights,
                "seconds": r.seconds,
                "image_path": r.image_path,
            }
            for r in results
        ])
        print("\n=== Summary ===")
        print(summary_df.to_string(index=False))

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].scatter(summary_df["pixels"], summary_df["seconds"])
            axes[0].set_xlabel("Pixels")
            axes[0].set_ylabel("Seconds")
            axes[0].set_title("Render time vs. resolution")

            axes[1].scatter(summary_df["num_objects"], summary_df["seconds"])
            axes[1].set_xlabel("Number of objects")
            axes[1].set_ylabel("Seconds")
            axes[1].set_title("Render time vs. object count")

            plt.tight_layout()
            plt.show()
        except ImportError:
            pass

        print("\n=== Top functions by cumulative time (slowest scene) ===")
        slowest = max(results, key=lambda r: r.seconds)
        top_df = pd.DataFrame(slowest.top_functions)
        print(top_df.to_string(index=False))

    except ImportError:
        print("\n=== Summary (plain text; install pandas for a table) ===")
        for r in results:
            print(
                f"scene {r.scene_id}: {r.hsize}x{r.vsize}, "
                f"{r.num_objects} objects, {r.num_lights} lights, "
                f"{r.seconds:.3f}s, saved to {r.image_path}"
            )

    display_renders(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile random ray tracer scenes.")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=RENDER_OUTPUT_DIR)
    args = parser.parse_args()

    results = run_experiment(args.num_scenes, seed=args.seed, output_dir=args.output_dir)
    summarize(results)


if __name__ == "__main__":
    main()