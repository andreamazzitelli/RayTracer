from __future__ import annotations

from tqdm import tqdm

from raytracer.geometry.ray import Ray
from raytracer.image.canvas import Canvas
from raytracer.image.color import Color
from raytracer.rendering.camera import Camera
from raytracer.rendering.lighting import lighting
from raytracer.scene.world import World
from raytracer.scene.world import hit as find_hit

SHADOW_EPSILON = 1e-5


def color_at(world: World, ray: Ray) -> Color:
    """The color seen by `ray` in `world`."""
    intersections = world.intersect(ray)
    intersection = find_hit(intersections)

    if intersection is None:
        return Color(0, 0, 0)

    obj = intersection.object
    point = ray.position_at(intersection.t)
    normal = obj.normal_at(point)
    eye = -ray.direction

    # nudge the point along the normal to avoid shadow acne / self-intersection
    over_point = point + normal * SHADOW_EPSILON

    color = Color(0, 0, 0)
    for light in world.lights:
        in_shadow = world.is_shadowed(over_point, light)
        color = color + lighting(
            obj.material, light, over_point, eye, normal, in_shadow
        )

    return color


def render(camera: Camera, world: World, show_progress: bool = True) -> Canvas:
    """Render `world` as seen by `camera`, returning a fully-populated Canvas."""
    image = Canvas(camera.hsize, camera.vsize)

    rows = range(camera.vsize)
    if show_progress:
        rows = tqdm(rows, desc="Rendering", unit="row")

    for y in rows:
        for x in range(camera.hsize):
            ray = camera.ray_for_pixel(x, y)
            color = world.color_at(ray)
            image.write_pixel(x, y, color)

    return image