# src/raytracer/scene/obj_parser.py
from __future__ import annotations

from dataclasses import dataclass, field

from raytracer.geometry.point import Point
from raytracer.shapes.triangle import Triangle


@dataclass
class ParseResult:
    vertices: list[Point] = field(default_factory=list)
    triangles: list[Triangle] = field(default_factory=list)
    ignored_lines: int = 0


def fan_triangulate(vertices: list[Point]) -> list[Triangle]:
    """Fan triangulation: for a face with vertices v0, v1, ..., vn,
    produce triangles (v0,v1,v2), (v0,v2,v3), ..., (v0,vn-1,vn)."""
    triangles = []
    for i in range(1, len(vertices) - 1):
        triangles.append(Triangle(vertices[0], vertices[i], vertices[i + 1]))
    return triangles


def parse_obj(source: str) -> ParseResult:
    """Parse Wavefront OBJ text into vertices and triangulated faces.
    OBJ vertex indices are 1-based; vertices[0] is a dummy so that
    1-based face indices map directly without an off-by-one subtraction
    scattered through the parsing logic."""
    result = ParseResult()
    result.vertices.append(Point(0, 0, 0))  # index 0 unused, keeps indexing 1-based

    for line in source.splitlines():
        tokens = line.split()
        if not tokens:
            result.ignored_lines += 1
            continue

        keyword, *args = tokens

        if keyword == "v":
            x, y, z = (float(v) for v in args[:3])
            result.vertices.append(Point(x, y, z))

        elif keyword == "f":
            face_indices = [int(idx) for idx in args]
            face_vertices = [result.vertices[i] for i in face_indices]
            result.triangles.extend(fan_triangulate(face_vertices))

        else:
            result.ignored_lines += 1

    return result


def parse_obj_file(path: str) -> ParseResult:
    with open(path, "r", encoding="utf-8") as f:
        return parse_obj(f.read())