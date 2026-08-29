from __future__ import annotations

import pytest

from raytracer.image.canvas import Canvas
from raytracer.image.color import Color


def test_construction_default_fill_black():
    c = Canvas(10, 20)
    for y in range(20):
        for x in range(10):
            assert c.pixel_at(x, y) == Color(0, 0, 0)


def test_construction_custom_default_fill():
    fill = Color(1, 0.5, 0.25)
    c = Canvas(3, 3, default=fill)
    assert c.pixel_at(1, 1) == fill


def test_write_and_read_pixel_round_trip():
    c = Canvas(10, 20)
    red = Color(1, 0, 0)
    c.write_pixel(2, 3, red)
    assert c.pixel_at(2, 3) == red


def test_non_square_canvas_edge_pixel():
    # Regression test for the [x][y] vs [y][x] transposition bug —
    # a square canvas cannot distinguish correct indexing from swapped.
    c = Canvas(4, 2)
    marker = Color(1, 1, 1)
    c.write_pixel(3, 1, marker)  # last column, last row
    assert c.pixel_at(3, 1) == marker
    # every other pixel should remain the default (black)
    assert c.pixel_at(0, 0) == Color(0, 0, 0)
    assert c.pixel_at(3, 0) == Color(0, 0, 0)
    assert c.pixel_at(0, 1) == Color(0, 0, 0)


def test_write_pixel_out_of_bounds_negative_x():
    c = Canvas(5, 5)
    with pytest.raises(IndexError):
        c.write_pixel(-1, 0, Color(1, 1, 1))


def test_write_pixel_out_of_bounds_negative_y():
    c = Canvas(5, 5)
    with pytest.raises(IndexError):
        c.write_pixel(0, -1, Color(1, 1, 1))


def test_write_pixel_out_of_bounds_too_large_x():
    c = Canvas(5, 5)
    with pytest.raises(IndexError):
        c.write_pixel(5, 0, Color(1, 1, 1))


def test_write_pixel_out_of_bounds_too_large_y():
    c = Canvas(5, 5)
    with pytest.raises(IndexError):
        c.write_pixel(0, 5, Color(1, 1, 1))


def test_pixel_at_out_of_bounds_negative():
    c = Canvas(5, 5)
    with pytest.raises(IndexError):
        c.pixel_at(-1, 0)


def test_pixel_at_out_of_bounds_too_large():
    c = Canvas(5, 5)
    with pytest.raises(IndexError):
        c.pixel_at(5, 5)


def test_pixels_property_reflects_writes():
    c = Canvas(2, 2)
    c.write_pixel(1, 1, Color(1, 1, 1))
    assert c.pixels[1][1].tolist() == [1.0, 1.0, 1.0]
