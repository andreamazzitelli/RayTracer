# Phase 2 — Rays

## 1. Mathematical Theory

A ray is the fundamental probe of a ray tracer: a half-line starting at an
**origin** point and extending infinitely in a **direction**. Every pixel in
the final image corresponds to (at least) one ray fired from the camera into
the scene; "ray tracing" as a technique is precisely the process of finding
what each such ray hits.

## 2. Geometric Intuition

A ray is parametrized by a single scalar `t`, which can be thought of as
"how far along the direction vector have we traveled from the origin."
`t = 0` is the origin itself. Increasing `t` moves forward along the
direction. Negative `t` would move backward, behind the origin — not
physically part of the ray as usually drawn, but a mathematically valid
evaluation of the same equation, and useful for reasoning about sphere
intersections in Phase 4 (e.g. "the sphere is entirely behind the ray").

## 3. Equations

### The parametric ray equation

```
R(t) = O + t*D
```

where `O` is the origin `Point` and `D` is the direction `Vector`.

**Important subtlety**: if `D` is not a unit vector, `t` is not literally a
distance traveled — it is scaled by `|D|`. A `t` of `1` moves exactly `|D|`
units along the ray, not `1` unit. This matters once `t` values from
different rays need to be compared or sorted (Phase 4 intersection tests,
Phase 6 shadow rays, Phase 9 closest-hit selection) — comparisons are only
meaningful when direction vectors are normalized consistently, or when the
comparison is scoped to a single ray's own `t` values (which is the common
case and generally safe).

## 4. Python / NumPy Representation

`Ray` is a thin, immutable-by-convention composite of the two types from
Phase 1 — it does not introduce any new NumPy usage itself; all numerical
work is delegated to `Point`/`Vector`.

```python
class Ray:
    __slots__ = ("origin", "direction")

    def __init__(self, origin: Point, direction: Vector) -> None:
        self.origin = origin
        self.direction = direction

    def position_at(self, t: float) -> Point:
        return self.origin + (t * self.direction)
```

## 5. API / Design Decisions

- **Public plain attributes (`origin`, `direction`) rather than
  properties.** `Ray` is a thin composite of two already-encapsulated,
  effectively-immutable types (`Point`, `Vector` have no in-place mutators),
  so the extra indirection of `@property` accessors was judged unnecessary
  here — a deliberate departure from the `Point`/`Vector` pattern, made
  consciously rather than by default.
- **`position_at` implements the general formula directly, with no special
  case for `t = 0`.** An earlier draft special-cased `t=0` to return
  `self.origin` directly; this was removed because it is both unnecessary
  (the general formula already reduces correctly to `origin` when `t=0`,
  since `0 * direction` is the zero vector) and actively harmful for
  testing — a special case for `t=0` means a `t=0` test can never actually
  exercise (and thus never actually validate) the general
  addition/multiplication path.
- **`direction` is not normalized by the constructor.** Left as a
  conscious, documented choice rather than silently normalizing — doing so
  would change the meaning of `t` (see the equations section above) in ways
  that could surprise callers who deliberately pass a scaled direction.
- **Equality (`__eq__`) delegates to `Point.__eq__` and `Vector.__eq__`**
  rather than re-implementing tolerance-based float comparison — avoids
  duplicating comparison logic across types.

## 6. Testing Strategy

- `position_at(0)` equals the origin exactly (via the general formula, not
  a special case — see above).
- Hand-computed `position_at(t)` for positive `t`, a fractional `t`, and a
  negative `t`.
- **Property-style test**: `ray.position_at(t) == ray.origin + ray.direction
  * t` computed independently, for several `t` values — tests the equation
  itself, not just fixed examples.
- Equality between two **separately constructed** `Ray` objects with equal
  (but not identical) `origin`/`direction` components — see Common Mistakes
  below for why this specific test matters.

## 7. Common Mistakes Encountered (and Fixed)

1. **`isinstance(object, Ray)` instead of `isinstance(other, Ray)`.** The
   built-in name `object` was passed to `isinstance` instead of the method's
   actual parameter, `other`. Since `object` (the built-in type) is never an
   instance of `Ray`, the check was always `True`, and `__eq__` always
   returned `NotImplemented`.

   This bug is more dangerous than a typo that simply crashes: when
   `__eq__` returns `NotImplemented`, Python retries by calling
   `other.__eq__(self)`. If `other` is also a `Ray` with the identical bug,
   *that* call also returns `NotImplemented`, and Python silently falls
   back to **identity comparison** (`is`). The net effect: two separately
   constructed but value-equal `Ray` objects compared unequal, with no
   error raised anywhere — a silent wrong answer rather than a crash.

   This is also a clear illustration of why **`assert my_ray == my_ray`**
   (comparing an object to itself) is an insufficient test: identity
   comparison happens to agree with value comparison when it's the same
   object, so that test would pass even with this bug present. The test
   that actually catches this constructs two distinct `Ray` instances with
   equal component values and asserts equality between them.

2. **Redundant `t = 0` special case in `position_at`** (see Design
   Decisions above) — not a correctness bug on its own, but a latent one:
   it would have silently masked a bug in `Vector.__mul__` or
   `Point.__add__` specifically for the `t=0` case, since that code path
   was never exercised for zero. Removed so that the general formula is
   what gets tested, including at `t=0`.

## 8. What This Will Be Used For Later

- Every intersection test (`Sphere.intersect`, and later `Plane`,
  `Triangle`) is fundamentally "solve for `t` such that `R(t)` lies on the
  shape's surface."
- Shadow rays (Phase 6) are `Ray`s constructed from a hit point toward a
  light.
- Reflected and refracted rays (Phases 12–13) are new `Ray`s built from a
  hit point and a computed direction (via `Vector.reflect`, and Snell's law
  for refraction).
- Camera ray generation (Phase 8) constructs one `Ray` per pixel, mapping
  image-plane coordinates to world-space direction vectors.
