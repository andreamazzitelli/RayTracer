# Phase 9 — World / Scene (Formalized)

## 1. Purpose

Much of `World`'s mechanics (`intersect`, `is_shadowed`, `hit()`) were built
ahead of schedule to support shadows (Phase 6) and the initial renderer
(Phase 8). Formalizing this phase properly means: (1) introducing a shared
data bundle (`Computations`) so per-intersection geometric state is computed
once and reused by shading, reflection, and refraction alike, rather than
recomputed inline in `color_at`; and (2) closing a real gap — `Sphere` (and
shapes generally) needed a `transform` attribute for Phase 7's matrices to
actually take effect during rendering, which had not yet been wired in.

## 2. Why `Computations` Exists

Reflection (Phase 12) and refraction (Phase 13) both need the *same* bundle
of per-intersection data that ordinary shading needs — hit point, normal,
eye vector, shadow-offset point — plus additional derived values later
(reflection vector, whether the ray originated inside the object). Without
a shared type, that computation would be duplicated three times, or
`color_at` would grow increasingly tangled trying to do everything at once.
`Computations` is that shared bundle, computed once per intersection via
`prepare_computations(intersection, ray)`.

## 3. The "Inside" Case and Normal Flipping

### Theory

`Shape.normal_at` always returns a normal pointing *outward* from the
surface. If a ray originates from **inside** an object (e.g. a shadow ray
or a refracted ray that has entered a transparent sphere), that outward
normal is geometrically backwards for shading purposes — `lighting()`'s
diffuse/specular math assumes the normal faces toward the incoming
light/eye side of the surface.

### Detecting "inside" via the normal-eye dot product

`eye_vector` points from the hit point back toward the ray's origin.

- If the ray approached from **outside**, the (outward) normal and the eye
  vector point in roughly the *same* general direction (both "outward"
  relative to the surface) — their dot product is **positive**.
- If the ray originated from **inside**, the eye vector points back toward
  the interior, roughly *opposite* the outward normal — their dot product
  is **negative**.

```python
if normal_vector.dot(eye_vector) < 0:
    inside = True
    normal_vector = -normal_vector
else:
    inside = False
```

This is the same sign-based reasoning already used for `light_dot_normal`
back in Phase 5 — a negative dot product signaling "wrong side" recurs as a
pattern throughout the renderer.

### `over_point` must be computed after any flip

The shadow-acne epsilon offset (Phase 6) pushes a point *outward, away from
the surface*, along the normal. If the normal was flipped because the ray
started inside the object, "outward" now means the *flipped* direction —
computing `over_point` from the original, unflipped normal would push the
point the wrong way, reintroducing the exact self-intersection bug the
offset exists to prevent.

## 4. Python / NumPy Representation

```python
class Computations:
    __slots__ = ("t", "object", "point", "eye_vector", "normal_vector",
                 "inside", "over_point")


def prepare_computations(intersection: Intersection, ray: Ray) -> Computations:
    point = ray.position_at(intersection.t)
    eye_vector = -ray.direction
    normal_vector = intersection.object.normal_at(point)

    if normal_vector.dot(eye_vector) < 0:
        inside = True
        normal_vector = -normal_vector
    else:
        inside = False

    over_point = point + normal_vector * SHADOW_EPSILON
    return Computations(intersection.t, intersection.object, point,
                         eye_vector, normal_vector, inside, over_point)
```

`World` gains `shade_hit` and `color_at`, consolidating shading
responsibility that previously lived inline in `renderer.py`:

```python
class World:
    def shade_hit(self, comps: Computations) -> Color:
        color = Color(0, 0, 0)
        for light in self.lights:
            in_shadow = self.is_shadowed(comps.over_point, light)
            color = color + lighting(
                comps.object.material, light, comps.over_point,
                comps.eye_vector, comps.normal_vector, in_shadow,
            )
        return color

    def color_at(self, ray: Ray) -> Color:
        intersection = hit(self.intersect(ray))
        if intersection is None:
            return Color(0, 0, 0)
        comps = prepare_computations(intersection, ray)
        return self.shade_hit(comps)
```

`renderer.py`'s `render()` is correspondingly simplified to a thin loop
delegating entirely to `world.color_at(ray)` per pixel — all "what color
does this ray see" logic now lives on `World`, with a single implementation
rather than a parallel one in the renderer.

### Closing the transform gap: `Shape.transform`

`Shape` gains a `transform: Matrix` attribute (defaulting to identity)
alongside `material`. `Sphere.intersect`/`normal_at` are updated to convert
incoming rays/points into local space via `transform.inverse()` before
doing their existing math, and `normal_at` converts the resulting local
normal back to world space via the **inverse transpose** (not the plain
inverse) of the transform:

```python
def normal_at(self, point: Point) -> Vector:
    local_point = self.transform.inverse().apply_to_point(point)
    local_normal = (local_point - self.center).normalize()
    world_normal = self.transform.inverse().transpose().apply_to_vector(local_normal)
    return world_normal.normalize()
```

**Why inverse-transpose for normals, specifically**: under non-uniform
scaling, a normal transformed the same way as a point would no longer stay
perpendicular to the (correctly transformed) surface. The inverse-transpose
is the standard correction that preserves perpendicularity regardless of
scale — accepted here as an established linear-algebra result rather than
re-derived from scratch, and verified empirically via a non-uniformly
scaled sphere's normal being tested for perpendicularity. The final
`.normalize()` is necessary (not optional): the inverse-transpose operation
can produce a non-unit-length result even from a unit input.

## 5. API / Design Decisions

- **`default_world()` as a static factory on `World`**: a fixed, well-known
  test fixture (two concentric unit spheres, one scaled to half via its
  `transform`, one light) shared across many tests, so expected values
  only need to be hand-computed once rather than per test.
- **Shading responsibility consolidated onto `World`, not left split
  between `World` and `renderer.py`.** Before this phase, `color_at` lived
  in `renderer.py` and computed hit-point/normal/eye-vector/shadow-offset
  inline; that logic is now entirely `World.shade_hit`/`color_at`, with
  `render()` reduced to pixel iteration only.

## 6. Testing Strategy

- `prepare_computations` for a ray hitting from outside — `inside == False`,
  normal unchanged.
- `prepare_computations` for a ray originating inside a sphere —
  `inside == True`, normal is the negation of the unflipped result.
- `over_point`'s z-component (or relevant axis) confirms an actual nudge
  occurred, not left exactly on the surface — precise regression test for
  the shadow-acne fix taking effect through this new code path.
- `shade_hit` on `default_world()`, ray hitting the outer sphere from
  outside — cross-checked against calling `lighting()` directly with
  equivalent parameters.
- `shade_hit` with the same setup but the point in shadow — dimmer
  (ambient-only) result, matching `lighting(..., in_shadow=True)`.
- `World.color_at` — miss returns black; hit matches the manual
  `intersect -> hit -> prepare_computations -> shade_hit` pipeline.
- Full render regression: confirm `render()` through the refactored
  `world.color_at` path produces identical output to the pre-refactor
  version on a small scene — proves the consolidation didn't silently
  change behavior.
- `Sphere` with a non-identity `transform`: translated sphere still
  intersects correctly at the expected world-space location; non-uniformly
  scaled sphere's normal is still unit-length and perpendicular to the
  (transformed) surface at a hand-picked point.

## 7. What This Will Be Used For Later

- `Computations` gains additional fields in Phase 12 (a `reflect_vector`)
  and Phase 13 (refraction indices, an `under_point` analogous to
  `over_point` but for refracted rays) — built by extending
  `prepare_computations`, not by introducing a parallel data structure.
- The `inside`/normal-flip logic is directly reused by refraction: a
  refracted ray traveling through a transparent object repeatedly crosses
  surfaces from inside and outside, and needs this same detection at every
  crossing.
- The `Shape.transform` + inverse-transpose-normal pattern established here
  for `Sphere` is copied directly into `Plane` and `Triangle` in Phase 10 —
  the same local-space conversion applies to every shape uniformly.
