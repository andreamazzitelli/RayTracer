from __future__ import annotations

import math

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.transform import view_transform
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.camera import Camera
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material
from raytracer.rendering.renderer import render
from raytracer.scene.mesh_generators import uv_sphere_triangles
from raytracer.scene.world import World
from raytracer.shapes.mesh import Mesh, build_bvh, intersect_bvh
from raytracer.shapes.plane import Plane
from raytracer.shapes.triangle import Triangle


def _small_triangle_set() -> list[Triangle]:
    return uv_sphere_triangles(radius=1.0, latitude_segments=4, longitude_segments=4)


def test_bvh_root_bounding_box_contains_all_triangles():
    triangles = _small_triangle_set()
    root = build_bvh(triangles, leaf_size=4)

    for t in triangles:
        assert root.bounding_box.contains_point(t.p1)
        assert root.bounding_box.contains_point(t.p2)
        assert root.bounding_box.contains_point(t.p3)


def test_bvh_leaf_size_respected():
    triangles = _small_triangle_set()
    root = build_bvh(triangles, leaf_size=4)

    def check(node):
        if node.is_leaf():
            assert len(node.triangles) <= 4
        else:
            check(node.left)
            check(node.right)

    check(root)


def test_bvh_all_triangles_reachable():
    triangles = _small_triangle_set()
    root = build_bvh(triangles, leaf_size=4)

    def collect(node):
        if node.is_leaf():
            return list(node.triangles)
        return collect(node.left) + collect(node.right)

    reachable = collect(root)
    assert len(reachable) == len(triangles)
    assert set(id(t) for t in reachable) == set(id(t) for t in triangles)


def test_mesh_intersect_matches_brute_force_on_hit():
    triangles = uv_sphere_triangles(radius=1.0, latitude_segments=6, longitude_segments=6)
    mesh = Mesh(triangles)

    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))

    brute_force = []
    for t in triangles:
        brute_force.extend(t.intersect(r))
    brute_force.sort(key=lambda i: i.t)

    via_bvh = mesh.intersect(r)
    via_bvh.sort(key=lambda i: i.t)

    assert len(via_bvh) == len(brute_force)
    for a, b in zip(via_bvh, brute_force):
        assert math.isclose(a.t, b.t, abs_tol=1e-9)


def test_mesh_intersect_empty_on_miss():
    triangles = uv_sphere_triangles(radius=1.0, latitude_segments=6, longitude_segments=6)
    mesh = Mesh(triangles)

    # A ray whose entire infinite line passes nowhere near the sphere —
    # not just "behind" it (which would still yield real, if negative,
    # t values under this project's unfiltered-intersection convention).
    r = Ray(Point(100, 100, 100), Vector(0, 1, 0))
    assert mesh.intersect(r) == []


def test_full_render_matches_brute_force_pixel_for_pixel():
    # The load-bearing correctness test: a BVH-accelerated render must
    # produce IDENTICAL output to the brute-force equivalent, not just
    # "similar" or "faster" — any pixel mismatch means the BVH is
    # incorrectly pruning triangles a ray actually needed to test.
    material = Material(Color(0.6, 0.8, 1.0), diffuse=0.7, specular=0.3)
    triangles = uv_sphere_triangles(
        radius=1.5, latitude_segments=10, longitude_segments=10, material=material
    )

    floor = Plane(material=Material(Color(1, 1, 1), specular=0))
    light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    world_brute = World(objects=[floor, *triangles], lights=[light])

    mesh = Mesh(triangles)
    world_bvh = World(objects=[floor, mesh], lights=[light])

    camera = Camera(
        30, 15, math.pi / 3,
        transform=view_transform(Point(0, 1.5, -5), Point(0, 0.5, 0), Vector(0, 1, 0)),
    )

    canvas_brute = render(camera, world_brute, show_progress=False)
    canvas_bvh = render(camera, world_bvh, show_progress=False)

    for y in range(15):
        for x in range(30):
            assert canvas_brute.pixel_at(x, y) == canvas_bvh.pixel_at(x, y), (
                f"Mismatch at pixel ({x},{y})"
            )