from __future__ import annotations

import random

from raytracer.image.canvas import Canvas
from raytracer.image.color import Color
from raytracer.rendering.camera import Camera
from raytracer.scene.world import World

DEFAULT_MAX_DEPTH = 5


def render_path_traced(
    camera: Camera,
    world: World,
    samples_per_pixel: int = 16,
    max_depth: int = DEFAULT_MAX_DEPTH,
    seed: int | None = None,
    show_progress: bool = True,
) -> Canvas:
    """Render `world` via Monte Carlo path tracing: averages
    `samples_per_pixel` independent calls to World.path_trace per pixel.
    Slower and noisier per-sample than the deterministic render(), but
    converges toward a physically based result as samples_per_pixel
    increases."""
    rng = random.Random(seed)
    image = Canvas(camera.hsize, camera.vsize)

    rows = range(camera.vsize)
    if show_progress:
        try:
            from tqdm import tqdm
            rows = tqdm(rows, desc="Path tracing", unit="row")
        except ImportError:
            pass

    for y in rows:
        for x in range(camera.hsize):
            accumulated = Color(0, 0, 0)
            for _ in range(samples_per_pixel):
                ray = camera.ray_for_pixel(x, y)
                accumulated = accumulated + world.path_trace(ray, depth=max_depth, rng=rng)
            averaged = accumulated * (1.0 / samples_per_pixel)
            image.write_pixel(x, y, averaged)

    return image