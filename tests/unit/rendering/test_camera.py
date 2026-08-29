from __future__ import annotations

import math

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.geometry.transform import rotation_y, translation
from raytracer.geometry.vector import Vector
from raytracer.rendering.camera import Camera


def test_pixel_size_horizontal_canvas():
    c = Camera(200, 125, math.pi / 2)
    assert math.isclose(c.pixel_size, 0.01, abs_tol=1e-9)


def test_pixel_size_vertical_canvas():
    c = Camera(125, 200, math.pi / 2)
    assert math.isclose(c.pixel_size, 0.01, abs_tol=1e-9)


def test_default_transform_is_identity():
    c = Camera(160, 120, math.pi / 2)
    assert c.transform == Matrix.identity(4)


def test_ray_through_center_of_canvas():
    c = Camera(201, 101, math.pi / 2)
    r = c.ray_for_pixel(100, 50)
    assert r.origin == Point(0, 0, 0)
    assert r.direction == Vector(0, 0, -1)


def test_ray_through_corner_of_canvas():
    c = Camera(201, 101, math.pi / 2)
    r = c.ray_for_pixel(0, 0)
    assert r.origin == Point(0, 0, 0)
    assert r.direction == Vector(0.6651864261194508, 0.3325932130597254, -0.6685123582500481)


def test_ray_with_transformed_camera():
    transform = rotation_y(math.pi / 4) @ translation(0, -2, 5)
    c = Camera(201, 101, math.pi / 2, transform=transform)
    r = c.ray_for_pixel(100, 50)
    assert r.origin == Point(0, 2, -5)
    assert r.direction == Vector(math.sqrt(2) / 2, 0, -math.sqrt(2) / 2)
