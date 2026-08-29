from __future__ import annotations
from raytracer.image.canvas import Canvas

import numpy as np

def _quantize(canvas: Canvas, max_color_value: int) -> np.ndarray:
    """Clamp, scale, and round the entire canvas to integers in one pass.
    This is the only place clamping happens."""
    clamped = np.clip(canvas.pixels, 0.0, 1.0)
    scaled = clamped * max_color_value
    return np.round(scaled).astype(np.int64)

def canvas_to_ppm(canvas: Canvas, max_color_value: int = 255) -> str:
    """Serialize a Canvas to a PPM (P3, plain-text) formatted string.
    Clamps each color channel from its unclamped float range into
    [0, max_color_value] as integers. This is the ONLY place clamping happens.
    """
    width = canvas.width
    height = canvas.height

    quantized = _quantize(canvas, max_color_value)  # shape: (height, width, 3), dtype int

    lines = ["P3", f"{width} {height}", f"{max_color_value}"]

    for y in range(height):
        row_values = quantized[y].reshape(-1)  # flatten (width, 3) -> r,g,b,r,g,b,...

        current_line = ""
        for value in row_values:
            candidate = f"{current_line} {value}".strip()

            if len(candidate) > 70:
                lines.append(current_line)
                current_line = str(value)
            else:
                current_line = candidate

        lines.append(current_line)

    return "\n".join(lines) + "\n"



def write_ppm(canvas: Canvas, path: str, max_color_value: int = 255) -> None:
    """Write the PPM-formatted output of `canvas` to a file at `path`."""

    canvas_string = canvas_to_ppm(canvas, max_color_value)

    with open(path, "w", encoding="utf-8") as file:
        file.write(canvas_string)
