# src/raytracer/scene/mesh_generators.py
from __future__ import annotations

import math

from raytracer.geometry.matrix import Matrix
from raytracer.geometry.point import Point
from raytracer.rendering.material import Material
from raytracer.shapes.triangle import Triangle


def generate_uv_sphere(
    radius: float = 1.0,
    rings: int = 10,
    segments: int = 10,
    material: Material | None = None,
    transform: Matrix | None = None,
) -> list[Triangle]:
    """Generate a UV-sphere approximated as a triangle mesh, entirely
    in-memory — bypasses OBJ file parsing so you can control triangle
    count directly via `rings`/`segments` for profiling purposes.

    Triangle count is exactly 2 * rings * segments (two triangles per
    quad, except the polar rings which only need one triangle each —
    handled by the shared vertex at each pole collapsing one edge).

    `rings` = latitude divisions (top pole to bottom pole)
    `segments` = longitude divisions (around the equator)
    """
    vertices: list[Point] = []

    for ring in range(rings + 1):
        theta = math.pi * ring / rings  # 0 at north pole, pi at south pole
        y = radius * math.cos(theta)
        ring_radius = radius * math.sin(theta)

        for segment in range(segments):
            phi = 2 * math.pi * segment / segments
            x = ring_radius * math.cos(phi)
            z = ring_radius * math.sin(phi)
            vertices.append(Point(x, y, z))

    def vertex_at(ring: int, segment: int) -> Point:
        return vertices[ring * segments + (segment % segments)]

    triangles: list[Triangle] = []

    for ring in range(rings):
        for segment in range(segments):
            top_left = vertex_at(ring, segment)
            top_right = vertex_at(ring, segment + 1)
            bottom_left = vertex_at(ring + 1, segment)
            bottom_right = vertex_at(ring + 1, segment + 1)

            # Skip degenerate triangles at the poles, where top_left ==
            # top_right (ring 0) or bottom_left == bottom_right (last ring)
            if ring != 0:
                triangles.append(
                    Triangle(top_left, bottom_left, bottom_right,
                             material=material, transform=transform)
                )
            if ring != rings - 1:
                triangles.append(
                    Triangle(top_left, bottom_right, top_right,
                             material=material, transform=transform)
                )

    return triangles

def uv_sphere_triangles(
    radius: float = 1.0,
    center: Point | None = None,
    latitude_segments: int = 20,
    longitude_segments: int = 20,
    material: Material | None = None,
) -> list[Triangle]:
    """Tessellate a sphere into a UV grid of quads, each split into two
    triangles. Total triangle count is
    2 * latitude_segments * longitude_segments — e.g. 20x20 produces 800
    triangles, a convenient, easily-scaled way to generate a mesh dense
    enough to be a genuine profiling target.
    """
    center = center or Point(0, 0, 0)
    kwargs = {"material": material} if material is not None else {}

    vertices: list[list[Point]] = []
    for lat_i in range(latitude_segments + 1):
        theta = math.pi * lat_i / latitude_segments  # 0 (top) to pi (bottom)
        row = []
        for lon_i in range(longitude_segments + 1):
            phi = 2 * math.pi * lon_i / longitude_segments
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.cos(theta)
            z = radius * math.sin(theta) * math.sin(phi)
            row.append(Point(center.x + x, center.y + y, center.z + z))
        vertices.append(row)

    triangles: list[Triangle] = []
    for lat_i in range(latitude_segments):
        for lon_i in range(longitude_segments):
            top_left = vertices[lat_i][lon_i]
            top_right = vertices[lat_i][lon_i + 1]
            bottom_left = vertices[lat_i + 1][lon_i]
            bottom_right = vertices[lat_i + 1][lon_i + 1]

            # Each quad becomes two triangles. Winding order matters for
            # normal direction (see Triangle's e2.cross(e1) convention) —
            # keep it consistent across the whole mesh or normals will
            # point inward on some faces and outward on others.
            triangles.append(Triangle(top_left, bottom_left, top_right, **kwargs))
            triangles.append(Triangle(top_right, bottom_left, bottom_right, **kwargs))

    return triangles


def triangle_count_for(latitude_segments: int, longitude_segments: int) -> int:
    """Convenience for predicting mesh size before generating it."""
    return 2 * latitude_segments * longitude_segments