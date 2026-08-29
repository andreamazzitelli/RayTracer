# Phase 3 — Canvas and Image Output

## 1. Theory / Purpose

This phase has comparatively little new mathematics; its purpose is
plumbing — turning "numbers computed by the renderer" into "an image a human
can actually look at." This closes the loop on the project's own learning
cycle (theory → implement → test → **visualize** → document), which up to
this point had no visual output stage at all.

## 2. Core Concepts

- **Canvas**: a 2D grid of pixels, `width x height`, where each pixel holds
  a color.
- **Color**: represented as an RGB triple of *unclamped* floating-point
  intensities, not integers in `[0, 255]`. This is a deliberate choice, not
  an oversight — see "Why colors stay unclamped internally" below.
- **Image coordinate system**: `(x, y)` with `x` growing rightward and `y`
  growing **downward**, origin `(0,0)` at the top-left. This is the opposite
  vertical handedness from typical math-class Cartesian axes (where `y`
  grows upward) and is the standard convention for raster image formats.
  Mixing this up with world-space `y`-up conventions is a common source of
  "my render is upside-down" bugs once camera ray generation (Phase 8) maps
  world coordinates onto this pixel grid.
- **PPM (Portable Pixmap) format, plain-text "P3" variant**: the simplest
  possible image format — no compression, a three-line header, then
  whitespace-separated integer RGB triples, row by row, top to bottom.
  Chosen as the first output target specifically because it has zero
  external dependencies and is human-readable, so small test images can be
  sanity-checked by eye directly in the raw text.

### Why colors stay unclamped internally

Colors will eventually be **summed** — multiple light contributions
(ambient + diffuse + specular in Phase 5, and eventually multiple lights,
reflections, and indirect illumination) are added together before a final
image is produced. If color values were clamped to `[0,1]` immediately after
each individual computation, information would be lost every time an
intermediate sum exceeded `1.0`, producing visibly wrong results once
contributions start stacking. Clamping is therefore deferred to the single
point where floating-point color must become a displayable integer: PPM
serialization.

## 3. PPM Format Specification (P3)

```
P3
<width> <height>
<max_color_value>
<r> <g> <b> <r> <g> <b> ...   (whitespace-separated, row by row, top to bottom)
```

Additional constraint implemented in this project: no line in the pixel-data
section may exceed 70 characters — a legacy convention from the original PPM
spec for line-oriented tools. This requires accumulating values into a
line buffer and wrapping to a new line once appending the next value would
exceed the limit.

## 4. Python / NumPy Representation

### Color

```python
class Color:
    __slots__ = ("_data",)

    def __init__(self, r: float, g: float, b: float) -> None:
        self._data = np.array([r, g, b], dtype=np.float64)
```

Two distinct multiplication-like operations are exposed, and are **not**
interchangeable:

- `__mul__` / `__rmul__` — **scalar** multiplication: `Color * float ->
  Color`. Used for scaling a color's overall intensity (e.g. by a material
  coefficient).
- `hadamard(other: Color) -> Color` — **component-wise** product: `(r1*r2,
  g1*g2, b1*b2)`. This is how a surface color combines with an incoming
  light's color (a red surface under a pure blue light should render black,
  which only the Hadamard product produces correctly — scalar multiplication
  cannot express "combine two different colors").

### Canvas

Internally backed by a single `(height, width, 3)` `np.float64` ndarray
(not a 2D array of `Color` objects), specifically to support **vectorized**
whole-canvas operations later:

```python
class Canvas:
    __slots__ = ("_width", "_height", "_pixels")
```

`write_pixel`/`pixel_at` convert to/from `Color` objects at the boundary,
keeping `Color` as the public interface while `_pixels` remains a raw array
internally. Row-major indexing is used: `self._pixels[y][x]`, matching the
`(height, width, 3)` allocation shape — **not** `[x][y]`, which is a subtle
transposition bug that a square test canvas cannot catch (see Common
Mistakes).

### PPM serialization

```python
def _quantize(canvas: Canvas, max_color_value: int) -> np.ndarray:
    clamped = np.clip(canvas.pixels, 0.0, 1.0)
    scaled = clamped * max_color_value
    return np.round(scaled).astype(np.int64)
```

This single vectorized function replaces what would otherwise be a
per-channel, per-pixel Python loop calling clamp/scale/round `3 * width *
height` times. The subsequent text-formatting step (building 70-character
wrapped lines) is **not** vectorizable — it is an inherently sequential,
stateful string-building process, not a numeric operation — and remains a
Python-level loop, now operating on already-quantized integers rather than
doing arithmetic per iteration.

## 5. API / Design Decisions

- **Clamping happens in exactly one place**: the PPM serialization step
  (`_quantize`), never inside `Color` or `Canvas`. This keeps both types
  format-agnostic — if a PNG writer were added later, it would reuse
  `Canvas`/`Color` unchanged and implement its own quantization/encoding
  step.
- **Out-of-bounds pixel access raises `IndexError`**, rather than silently
  clamping or ignoring the write/read. An out-of-bounds pixel coordinate
  during rendering is always a bug (typically a camera/ray-generation error
  mapping pixel coordinates incorrectly) — silently ignoring it would let
  that bug produce a subtly wrong image instead of an immediate, traceable
  crash.
- **`np.round` (banker's rounding) vs. Python's built-in `round()`**: NumPy's
  rounding uses round-half-to-even tie-breaking, which differs from naive
  "round half away from zero" behavior at exact `.5` boundaries. This is a
  real, documented difference — any test pinning an exact boundary case
  (e.g. a channel value that scales to exactly `127.5`) must account for
  which rounding rule is in effect, or the expected value may be off by one.
- **`Canvas.pixels` exposes the raw array for callers (like the PPM writer)
  that need whole-canvas vectorized access.** Whether this returns a direct
  reference or a defensive copy is a real encapsulation-vs-performance
  tradeoff: a direct reference risks external code mutating canvas state
  through the "read-only" property; a copy costs memory/time on every
  access. For a property expected to be read once per render (by the PPM
  writer), a copy is the safer default.

## 6. Testing Strategy

- Canvas construction with a default fill color (and a sane default, e.g.
  black, when none given).
- Single-pixel write/read round-trip.
- **Non-square canvas tests** (e.g. `width=4, height=2`) writing to a pixel
  near an edge — critical, since a square canvas cannot distinguish correct
  `[y][x]` indexing from an accidentally transposed `[x][y]`.
- Out-of-bounds raises `IndexError` on **both** `write_pixel` and
  `pixel_at`, for negative indices *and* indices `>= width`/`>= height`
  independently.
- `Color` arithmetic: `+`, scalar `*`, and specifically a `hadamard` test
  using two colors with **different** values per channel (not e.g. two
  `Color(1,1,1)` instances, which cannot distinguish Hadamard product from
  scalar multiplication).
- PPM header format (`P3`, dimensions, max value).
- PPM row/column ordering, using a small canvas with visually distinct
  per-pixel colors to catch any transposition.
- PPM clamping of both above-`1.0` and below-`0.0` input values to the
  correct integer range.
- 70-character line-wrap boundary behavior.

## 7. Common Mistakes Encountered (and Fixed)

1. **Transposed pixel indexing (`[x][y]` instead of `[y][x]`).** The
   internal array is allocated as `(height, width)` (or `(height, width,
   3)`), so the first index is the row (`y`) and the second is the column
   (`x`). An initial implementation indexed `self._pixels[x][y]`. On a
   *square* canvas this does not crash and does not obviously misbehave,
   because both dimensions share the same valid index range — the bug is
   invisible until the canvas is non-square, at which point it either
   raises an `IndexError` for legitimate coordinates or silently writes to
   the wrong pixel. This is precisely why a non-square test canvas is
   listed as required above, not optional.

2. **`pixel_at` skipped its own bounds check.** `write_pixel` correctly
   called `_check_bounds`, but an initial `pixel_at` implementation did
   not. The consequence was worse than a missing crash: NumPy's own
   indexing silently supports *negative* indices as "count from the end,"
   so `pixel_at(-1, 0)` returned the last row of the canvas instead of
   raising — a genuine correctness bug hiding behind an unused bounds-check
   method that existed but wasn't being called everywhere it needed to be.

3. **Shared mutable default when filling a `dtype=object` array
   (superseded design).** An earlier `Canvas` design stored `Color` objects
   directly in a `dtype=object` NumPy array via `np.full((height, width),
   default_color, dtype=object)`. This fills every cell with the *same*
   object reference, not independent copies — safe only because `Color` has
   no in-place mutating methods. This is documented here as a "trap that
   happened to be safe" specifically so it is not repeated in a context
   where the stored object *is* mutable.

## 8. What This Will Be Used For Later

- Every rendered scene ultimately produces its output by writing to a
  `Canvas` and serializing via `write_ppm` — this is the terminal step of
  the entire rendering pipeline from Phase 9 onward.
- The `Color` type (with its `+` and `hadamard` operations) is the return
  type of the Phase 5 `lighting()` function and is combined further in
  Phases 6, 12, and 13 (shadows, reflection, refraction).
- The vectorized quantization pattern established here — do the numeric
  work in NumPy across the whole array, keep only genuinely sequential
  logic (text formatting) as a Python loop — is a pattern worth
  recognizing and reapplying, particularly once performance optimization
  becomes a concern in Phase 14.
