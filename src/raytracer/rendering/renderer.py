from __future__ import annotations

from raytracer.image.canvas import Canvas
from raytracer.rendering.camera import Camera
from raytracer.scene.world import World


def render(camera: Camera, world: World, show_progress: bool = True) -> Canvas:
    image = Canvas(camera.hsize, camera.vsize)

    rows = range(camera.vsize)
    if show_progress:
        try:
            from tqdm import tqdm
            rows = tqdm(rows, desc="Rendering", unit="row")
        except ImportError:
            pass

    for y in rows:
        for x in range(camera.hsize):
            ray = camera.ray_for_pixel(x, y)
            color = world.color_at(ray)
            image.write_pixel(x, y, color)

    return image