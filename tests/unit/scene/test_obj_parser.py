from __future__ import annotations

from raytracer.geometry.point import Point
from raytracer.scene.obj_parser import parse_obj


def test_ignoring_unrecognized_lines():
    source = (
        "There was a young lady named Bright\n"
        "who traveled much faster than light.\n"
    )
    result = parse_obj(source)
    assert result.ignored_lines == 2


def test_parsing_vertex_records():
    source = """
v -1 1 0
v -1.0000 0.5000 0.0000
v 1 0 0
v 1 1 0
"""
    result = parse_obj(source)
    assert result.vertices[1] == Point(-1, 1, 0)
    assert result.vertices[2] == Point(-1, 0.5, 0)
    assert result.vertices[3] == Point(1, 0, 0)
    assert result.vertices[4] == Point(1, 1, 0)


def test_parsing_triangle_faces():
    source = """
v -1 1 0
v -1 0 0
v 1 0 0
v 1 1 0

f 1 2 3
f 1 3 4
"""
    result = parse_obj(source)
    assert len(result.triangles) == 2

    t1 = result.triangles[0]
    t2 = result.triangles[1]

    assert t1.p1 == result.vertices[1]
    assert t1.p2 == result.vertices[2]
    assert t1.p3 == result.vertices[3]

    assert t2.p1 == result.vertices[1]
    assert t2.p2 == result.vertices[3]
    assert t2.p3 == result.vertices[4]


def test_triangulating_polygons():
    source = """
v -1 1 0
v -1 0 0
v 1 0 0
v 1 1 0
v 0 2 0

f 1 2 3 4 5
"""
    result = parse_obj(source)
    assert len(result.triangles) == 3

    v = result.vertices

    assert result.triangles[0].p1 == v[1]
    assert result.triangles[0].p2 == v[2]
    assert result.triangles[0].p3 == v[3]

    assert result.triangles[1].p1 == v[1]
    assert result.triangles[1].p2 == v[3]
    assert result.triangles[1].p3 == v[4]

    assert result.triangles[2].p1 == v[1]
    assert result.triangles[2].p2 == v[4]
    assert result.triangles[2].p3 == v[5]


def test_dummy_zero_index_vertex_is_unused_placeholder():
    source = "v 1 2 3\n"
    result = parse_obj(source)
    # index 0 is a sentinel so 1-based OBJ indices map directly
    assert result.vertices[0] == Point(0, 0, 0)
    assert result.vertices[1] == Point(1, 2, 3)


def test_blank_lines_are_ignored_without_crashing():
    source = "\n\nv 0 0 0\n\n\nf 1 1 1\n"
    result = parse_obj(source)
    # blank lines are counted as ignored, not as errors
    assert result.ignored_lines >= 4
