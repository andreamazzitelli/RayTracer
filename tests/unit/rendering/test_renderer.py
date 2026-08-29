from __future__ import annotations

import math

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import translation
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.camera import Camera
from raytracer.rendering.renderer import render
from raytracer.scene.world import World


def _view_transform(frm: Point, to: Point, up: Vector) -> Matrix:
    """Standard look-at construction, used only to build a camera transform
    for the render() test below."""
    forward = (to - frm).normalize()
    upn = up.normalize()
    left = forward.cross(upn)
    true_up = left.cross(forward)
    orientation = Matrix([
        [left.x, left.y, left.z, 0],
        [true_up.x, true_up.y, true_up.z, 0],
        [-forward.x, -forward.y, -forward.z, 0],
        [0, 0, 0, 1],
    ])
    return orientation @ translation(-frm.x, -frm.y, -frm.z)


def test_color_at_ray_hits_matches_world_color_at():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    assert w.color_at(r) == Color(0.3806609553101071, 0.47582619413763383, 0.2854957164825803)


def test_color_at_ray_misses_is_black():
    w = World.default_world()
    r = Ray(Point(0, 0, -5), Vector(0, 1, 0))
    assert w.color_at(r) == Color(0, 0, 0)


def test_render_output_canvas_dimensions():
    w = World.default_world()
    c = Camera(11, 9, math.pi / 2)
    image = render(c, w)
    assert image.width == 11
    assert image.height == 9


def test_render_default_world_center_pixel():
    w = World.default_world()
    frm = Point(0, 0, -5)
    to = Point(0, 0, 0)
    up = Vector(0, 1, 0)
    c = Camera(11, 11, math.pi / 2, transform=_view_transform(frm, to, up))

    image = render(c, w)

    assert image.pixel_at(5, 5) == Color(
        0.3806609553101071, 0.47582619413763383, 0.2854957164825803
    )
