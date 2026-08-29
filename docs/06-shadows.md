# Phase 6 — Shadows

## 1. Theory

Shadows introduce no new formulas beyond what already exists — they are a
direct application of ray-object intersection (Phase 4) and ray
construction (Phase 2) to answer a binary geometric question: *is anything
standing between this point and this light?* The real content of this phase
is a numerical-precision subtlety, not new algebra.

## 2. The Algorithm

For a shaded point `P` and a light source:

1. Compute the vector from `P` toward the light:
   `light_vector = light.position - P`, and its length:
   `distance = light_vector.magnitude()`.
2. Build a **shadow ray**: origin `P`, direction
   `light_vector.normalize()`.
3. Intersect the shadow ray against every object in the scene.
4. Find the closest intersection with `0 < t < distance` (using the same
   `hit()` logic from Phase 9's `World`). If one exists, `P` is in shadow;
   otherwise it is lit.

### Why `t < distance`, not just `t > 0`

Filtering only on `t > 0` (as `hit()` already does for ordinary visibility)
is not sufficient here: an object positioned *beyond* the light itself
would still register as a valid `t > 0` hit along the shadow ray, but it
cannot possibly be blocking that light, since it's farther away than the
light is. The upper bound `t < distance` excludes exactly this case —
only obstructions strictly *between* `P` and the light count.

## 3. Geometric Intuition

Picture a point on a sphere's surface, and a light somewhere else in the
scene. If a second object sits directly on the line segment connecting the
point to the light, that object physically occludes the light from that
point's perspective — the point should render as if only ambient light
reaches it (diffuse and specular both suppressed), exactly mirroring the
"surface faces away from the light" clamp-to-zero behavior from Phase 5,
but here the cause is occlusion rather than surface orientation.

## 4. The Precision Problem: Shadow Acne

This is the conceptual core of the phase. `P` — the point being shaded — is
itself computed as `ray.position_at(t)` for some intersection `t`. Due to
ordinary floating-point rounding in that computation, `P` may not sit
*exactly* on the surface it was computed from — it may be an infinitesimal
distance *inside* the object.

If the shadow ray's origin is a hair inside its own object, that same
object can register a spurious self-intersection at a `t` value extremely
close to zero along the shadow ray — the object incorrectly "shadows
itself." This artifact is called **acne**: speckled, incorrect
self-shadowing across a surface that should be uniformly lit or unlit.

### The fix: an epsilon offset along the normal

Before constructing the shadow ray, nudge its origin slightly outward along
the surface normal:

```
shadow_ray_origin = P + N * epsilon
```

for a small `epsilon` (commonly ~`1e-5` to `1e-4` — large enough to
reliably clear floating-point noise in `P`'s computed position, small
enough to be visually imperceptible). This pushes the ray's starting point
just outside the surface, so it can no longer spuriously re-intersect the
geometry it originated from.

### This epsilon is conceptually different from `abs_tol` comparisons

It is worth being precise about two superficially similar but distinct
uses of "a small number to handle floating point":

| | Purpose | Used where |
|---|---|---|
| `abs_tol` (e.g. `1e-9`) | Decide whether two numbers *are the same value*, for equality comparisons | `Point`/`Vector`/`Color` `__eq__` |
| Shadow-acne `epsilon` | **Deliberately displace** a point's actual position to sidestep a self-intersection bug | Shadow ray origin construction |

The former is a tolerance for comparison; the latter is a geometric offset
that changes an actual coordinate. Conflating the two is an easy mistake —
both "feel like" floating-point workarounds — but they solve different
problems and are not interchangeable.

## 5. Python / NumPy Representation

Shadow testing requires iterating *every* object in a scene, not a single
shape — this pulled forward the essential piece of the `World` abstraction
(properly introduced in Phase 9) rather than testing shadows against a
throwaway single-object setup.

```python
class World:
    __slots__ = ("objects", "lights")

    def intersect(self, ray: Ray) -> list[Intersection]:
        intersections = []
        for obj in self.objects:
            intersections.extend(obj.intersect(ray))
        intersections.sort(key=lambda i: i.t)
        return intersections

    def is_shadowed(self, point: Point, light: PointLight) -> bool:
        light_vector = light.position - point
        distance = light_vector.magnitude()
        direction = light_vector.normalize()

        shadow_ray = Ray(point, direction)
        shadow_hit = hit(self.intersect(shadow_ray))

        return shadow_hit is not None and shadow_hit.t < distance


def hit(intersections: list[Intersection]) -> Intersection | None:
    for intersection in intersections:
        if intersection.t > 0:
            return intersection
    return None
```

## 6. API / Design Decisions

- **`hit()` is a free function, not a `World` method.** It operates purely
  on a `list[Intersection]` with no dependency on `World` or `Ray` — keeping
  it standalone reflects that it belongs conceptually with the data it
  processes, not with whichever class happens to produce that data.
- **`hit()` relies on its input already being sorted ascending by `t`.**
  `World.intersect` guarantees this by sorting before returning, which lets
  `hit()` do a single linear scan for the first `t > 0` entry rather than
  re-sorting defensively. This creates a real coupling: calling `hit()`
  with a hand-built, unsorted list (e.g. directly in a unit test) will
  silently produce the wrong answer with no error. This tradeoff (trust the
  invariant vs. defensively re-sort every call) is made deliberately here,
  favoring performance, but is worth flagging any time `hit()` is called
  outside of the `World.intersect -> hit` pipeline.
- **The epsilon offset is *not* baked into `is_shadowed`.** `is_shadowed`
  is kept as a "pure" geometric query — given a point and a light, is it
  occluded — with no built-in knowledge of *why* an offset might be needed.
  Responsibility for nudging the point along its normal before calling
  `is_shadowed` is left to the caller (the renderer, in Phase 9's shading
  pipeline), which already has both the hit point and its normal available
  together. This is a deliberate separation-of-concerns choice: geometric
  occlusion testing and floating-point-precision workarounds are different
  concerns, even though they show up adjacent to each other in practice.

## 7. Testing Strategy

For `hit()`:
- Returns `None` on an empty intersection list.
- Returns `None` when every intersection has `t <= 0`.
- Returns the correct (closest, positive-`t`) intersection when the list
  contains a mix of negative and positive `t` values.

For `World.intersect`:
- Ray against a world containing two objects returns the *combined*,
  correctly *sorted* list of intersections from both.

For `is_shadowed`, four cases, corresponding directly to the geometric
reasoning above:
1. Nothing between the point and the light — not shadowed.
2. An object directly between the point and the light — shadowed.
3. An object positioned *beyond* the light (light is between the point and
   the object) — **not** shadowed (this is exactly the case the `t <
   distance` upper bound exists to handle).
4. The point itself positioned beyond the object relative to the light
   (object is on the far side of the point from the light) — not shadowed.

## 8. Common Mistakes Encountered (and Fixed)

1. **`World.__init__` left as a stub (`...`)** in an intermediate draft —
   `self.objects`/`self.lights` were never assigned, so the very first call
   to `World.intersect` raised `AttributeError`. Also relevant: the `None`
   defaults for `objects`/`lights` parameters must each be replaced with a
   **fresh** empty list inside `__init__` (`objects if objects is not None
   else []`), never a mutable default argument value (`objects: list =
   []`) — a shared mutable default would silently alias the same list
   across every `World` instance constructed without an explicit argument.

2. **`hit()`'s first draft ignored the `t > 0` filter entirely** and simply
   returned `intersections[0]` after sorting — meaning a ray that misses
   everything (or that only intersects objects behind it) would return the
   *most negative* `t` value as if it were a valid, visible hit, rather
   than the documented `None`.

3. **`hit()`'s first draft crashed on an empty list.** `intersections[0]`
   on `[]` raises `IndexError` rather than returning `None` as the
   docstring promised — this is the actual, common case of "the ray hit
   nothing at all," not just an unlikely edge case.

4. **`hit()`'s first draft mutated its input in place** via
   `intersections.sort(...)`, a side effect on a list the caller may not
   expect to be reordered. Superseded by relying on `World.intersect`'s
   pre-established sort invariant rather than re-sorting (and thus
   mutating) inside `hit()` at all.

## 9. What This Will Be Used For Later

- The `World`/`hit()` pair introduced here to support shadow testing is the
  same machinery Phase 9 formalizes as the renderer's core "find what this
  ray sees" logic — shadows did not require a separate, parallel
  implementation.
- The epsilon-offset pattern (nudge a computed point slightly along its
  normal to avoid self-intersection) recurs in Phase 12 (reflection) and
  Phase 13 (refraction), where reflected/refracted rays originate from a
  surface point and face the identical self-intersection risk.
- `is_shadowed`'s "occluded -> ambient only" behavior is wired into the
  full `lighting()` call in the renderer once Phase 9's rendering pipeline
  is built, by simply passing an `in_shadow: bool` flag through to zero out
  diffuse/specular exactly as Phase 5 already does for surfaces facing away
  from a light.
