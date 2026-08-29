# Phase 8 — Camera

## 1. Theory / Purpose

This is where every previously built piece — rays, transforms, world
intersection, shading — connects into "generate a full image from a
described scene." The camera's job is a mapping: given a pixel coordinate
`(px, py)`, produce the world-space ray a viewer at the camera's position,
looking through that pixel, would see.

## 2. Geometric Setup

Conceptually, a virtual image plane ("canvas," distinct from the `Canvas`
pixel-grid class) sits exactly 1 unit in front of the camera, in camera
space, with the camera at the origin looking down `-z` (a standard
convention). The field of view is the full angle subtended by this virtual
canvas as seen from the camera.

### Half-view from field of view

```
half_view = tan(field_of_view / 2)
```

Straightforward right-triangle trig: half the FOV angle, opposite side at
distance 1 (adjacent side), gives half the canvas's extent via `tan`.

### Aspect ratio branch

```
aspect = hsize / vsize

if aspect >= 1:
    half_width  = half_view
    half_height = half_view / aspect
else:
    half_width  = half_view * aspect
    half_height = half_view
```

A wide (landscape) image's field of view should be constrained by width,
not height, or the result is distorted — the branch selects whichever
dimension the FOV genuinely constrains and derives the other from the
aspect ratio.

### Pixel size and per-pixel ray construction

```
pixel_size = (half_width * 2) / hsize
```

For pixel `(px, py)`:

```
x_offset = (px + 0.5) * pixel_size
y_offset = (py + 0.5) * pixel_size

world_x = half_width  - x_offset
world_y = half_height - y_offset
```

The `+0.5` offsets target the pixel's *center*, not its corner. The
subtraction (rather than addition) for `world_x`/`world_y` accounts for the
canvas edge being at `+half_width`/`+half_height` while pixel coordinates
increase rightward/downward — getting this sign backwards mirrors the
rendered image, a bug that is easy to miss until an asymmetric scene is
rendered.

The camera-space point `(world_x, world_y, -1)` (canvas sits at `z = -1`)
and the camera's own origin `(0,0,0)` are both transformed into world space
via `camera.transform.inverse()` (the same inverse-transform-to-world-space
pattern used for objects' local space, applied here in the opposite
direction: camera space to world space). The ray's direction is the
normalized vector from the transformed origin to the transformed pixel
point.

## 3. Python / NumPy Representation

```python
class Camera:
    __slots__ = (
        "hsize", "vsize", "field_of_view", "transform",
        "half_width", "half_height", "pixel_size",
    )

    def __init__(self, hsize, vsize, field_of_view, transform=None) -> None:
        # half_width, half_height, pixel_size computed ONCE here,
        # not recomputed per pixel in ray_for_pixel
        ...

    def ray_for_pixel(self, px: int, py: int) -> Ray:
        x_offset = (px + 0.5) * self.pixel_size
        y_offset = (py + 0.5) * self.pixel_size

        world_x = self.half_width - x_offset
        world_y = self.half_height - y_offset

        inverse_transform = self.transform.inverse()
        pixel = inverse_transform.apply_to_point(Point(world_x, world_y, -1))
        origin = inverse_transform.apply_to_point(Point(0, 0, 0))
        direction = (pixel - origin).normalize()

        return Ray(origin, direction)
```

## 4. API / Design Decisions

- **`half_width`/`half_height`/`pixel_size` computed once in `__init__`**,
  not recomputed per pixel — these depend only on the camera's fixed
  properties (`hsize`, `vsize`, `field_of_view`), not on which pixel is
  being queried; recomputing per pixel would redo identical trigonometry
  for every pixel in the image, a real and avoidable cost at full
  resolution.
- **`render()` iterates rows in the outer loop**, matching the row-major
  `(height, width, 3)` `Canvas` storage layout from Phase 3 — a minor
  cache-friendliness detail, secondary to correctness at this stage but
  worth doing consistently.
- **The renderer's `color_at`/`shade_hit` responsibility was later moved
  onto `World`** (see the Phase 9 doc) rather than staying inline in
  `renderer.py` — `render()` itself stays a thin loop: generate a ray per
  pixel, ask the world for its color, write it to the canvas.

## 5. Testing Strategy

- `pixel_size` for a landscape canvas — hand-computed.
- `pixel_size` for a portrait canvas — hand-computed.
- `ray_for_pixel` through the exact center of the canvas, identity
  transform — should point straight down `-z`.
- `ray_for_pixel` through a corner pixel (e.g. `(0,0)`) — hand-verified
  direction components.
- `ray_for_pixel` with a non-identity transform (camera rotated and/or
  translated) — confirms the inverse-transform wiring, not just the
  identity case.
- Full `render()` on a small, fully specified test world — spot-check a
  hand-reasoned-about pixel (e.g. dead center) against an expected color.

## 6. What This Will Be Used For Later

- `render()` is the top-level entry point for producing any actual image
  from this project going forward — every later phase (reflection,
  refraction, path tracing) changes what `world.color_at`/`shade_hit`
  compute per ray, but the camera's job (map pixel -> ray) never changes.
- The inverse-transform pattern established here for camera-space-to-
  world-space conversion is structurally identical to the object-space
  conversion already used in `Sphere`/`Plane`/`Triangle` — one mental
  model covers both.
