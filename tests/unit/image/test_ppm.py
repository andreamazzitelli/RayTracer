from __future__ import annotations

import os
import tempfile

from raytracer.image.canvas import Canvas
from raytracer.image.color import Color
from raytracer.image.ppm import canvas_to_ppm, write_ppm


def test_ppm_header():
    c = Canvas(5, 3)
    ppm = canvas_to_ppm(c)
    lines = ppm.splitlines()
    assert lines[0] == "P3"
    assert lines[1] == "5 3"
    assert lines[2] == "255"


def test_ppm_row_ordering_no_transposition():
    c = Canvas(2, 1)
    c.write_pixel(0, 0, Color(1, 0, 0))
    c.write_pixel(1, 0, Color(0, 1, 0))
    ppm = canvas_to_ppm(c)
    lines = ppm.splitlines()
    pixel_line = lines[3]
    assert pixel_line.split() == ["255", "0", "0", "0", "255", "0"]


def test_ppm_clamps_above_one():
    c = Canvas(1, 1)
    c.write_pixel(0, 0, Color(1.5, 1.5, 1.5))
    ppm = canvas_to_ppm(c)
    values = ppm.splitlines()[3].split()
    assert values == ["255", "255", "255"]


def test_ppm_clamps_below_zero():
    c = Canvas(1, 1)
    c.write_pixel(0, 0, Color(-0.5, -0.5, -0.5))
    ppm = canvas_to_ppm(c)
    values = ppm.splitlines()[3].split()
    assert values == ["0", "0", "0"]


def test_ppm_line_wrap_at_70_chars():
    c = Canvas(10, 2, default=Color(1, 0.8, 0.6))
    ppm = canvas_to_ppm(c)
    lines = ppm.splitlines()
    for line in lines:
        assert len(line) <= 70


def test_write_ppm_creates_file_matching_string():
    c = Canvas(2, 2, default=Color(0.5, 0.5, 0.5))
    expected = canvas_to_ppm(c)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.ppm")
        write_ppm(c, path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

    assert content == expected
