from __future__ import annotations

import numpy as np

from raytracer.image.color import Color


class Canvas:
    """A 2D grid of pixels, width x height, each holding a Color.
    Image coordinates: (x, y) with x growing rightward, y growing downward,
    (0,0) at top-left."""

    __slots__ = ("_width", "_height", "_pixels")

    def __init__(self, width: int, height: int, default: Color | None = None) -> None:
        """default: fill color for every pixel; consider what a sane default is
        (e.g. black) if none is given."""
        self._width = width
        self._height = height

        if default is None:
            default = Color(0, 0, 0)

        self._pixels = np.full(
            (height, width, 3),
            (default.r, default.g, default.b ),
            dtype=np.float64
        )

    @property
    def width(self) -> int:
        return int(self._width)

    @property
    def height(self) -> int:
        return int(self._height)

    @property
    def pixels(self) -> np.ndarray:
        """Read-only view of the raw (height, width, 3) pixel data,
        for callers (like ppm.py) that want to vectorize over the
        whole canvas instead of iterating pixel-by-pixel."""
        
        return self._pixels

    def write_pixel(self, x: int, y: int, color: Color) -> None:
        """Raises IndexError (or a subclass) if (x, y) is out of bounds."""
        self._check_bounds(x, y)

        self._pixels[y][x] = (color.r, color.g, color.b)


    def pixel_at(self, x: int, y: int) -> Color:
        """Raises IndexError (or a subclass) if (x, y) is out of bounds."""
        self._check_bounds(x, y)

        return Color.from_np_array(self._pixels[y][x])

    def _check_bounds(self, x: int, y: int) -> None:
        """Shared bounds check used by write_pixel/pixel_at."""
        if x >= self.width or x < 0: 
            raise IndexError("x value is out of bounds")
        if y >= self.height or y < 0:
            raise IndexError("y value is out of bounds")