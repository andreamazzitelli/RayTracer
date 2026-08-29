# Phase 4 — Spheres and Ray Intersection

## 1. Mathematical Theory

A sphere with center `C` and radius `r` is the set of all points `P`
satisfying:

```
|P - C|^2 = r^2
```

To find where a ray `R(t) = O + tD` intersects this sphere, substitute the
ray equation for `P`:

```
|O + tD - C|^2 = r^2
```

Let `L = O - C` (the vector from the sphere's center to the ray's origin).
Expand the squared magnitude as a dot product of the vector with itself:

```
(D*t + L) . (D*t + L) = r^2
(D.D)t^2 + 2(D.L)t + (L.L) - r^2 = 0
```

This is a standard quadratic in `t`:

```
a*t^2 + b*t + c = 0

a = D . D
b = 2 * (D . L)         where L = O - C
c = (L . L) - r^2
```

## 2. Geometric Intuition

A line (the ray, extended infinitely in both directions) can intersect a
sphere's surface in at most two points. The **discriminant** of the
quadratic, `b^2 - 4ac`, determines how many real solutions exist, which
maps directly onto the geometric picture:

| Discriminant | Geometric meaning |
|---|---|
| `< 0` | Line misses the sphere entirely — no real roots |
| `= 0` | Line is tangent to the sphere — touches at exactly one point (a repeated root) |
| `> 0` | Line passes through the sphere — two distinct intersection points |

## 3. Equations

### The quadratic formula

```
t = (-b +/- sqrt(discriminant)) / (2a)
```

producing `t1` (the `-` root, always the smaller/closer value since
`sqrt(discriminant) >= 0`) and `t2` (the `+` root).

### Surface normal

For a point `P` known to lie on the sphere's surface, the normal at `P`
points radially outward from the center:

```
N = (P - C) / r     (equivalently: (P - C).normalize())
```

This holds for **every** point on a sphere's surface, by the sphere's very
definition — unlike, say, a cube, where the normal depends on *which face*
was hit.

## 4. Python / NumPy Representation

```python
def intersect(self, ray: Ray) -> list[Intersection]:
    sphere_to_ray = ray.origin - self.center       # L, a Vector

    a = ray.direction.dot(ray.direction)
    b = 2 * ray.direction.dot(sphere_to_ray)
    c = sphere_to_ray.dot(sphere_to_ray) - self.radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return []

    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    return [Intersection(t1, self), Intersection(t2, self)]

def normal_at(self, point: Point) -> Vector:
    return (point - self.center).normalize()
```

`sphere_to_ray` (i.e. `L = O - C`) is computed once and reused for both `b`
and `c`, avoiding redundant vector subtraction.

## 5. API / Design Decisions

### Why `Intersection` exists as its own type

An earlier design considered returning raw `t` values (a `list[float]`)
from `intersect`. This was rejected: later stages (the renderer, shadow
testing) need to know not just *where* along a ray something was hit, but
*what object* was hit — a scene will have many shapes, and "the closest hit"
needs to carry a reference back to the object it belongs to. `Intersection`
bundles `(t, object)` together for exactly this reason.

```python
class Intersection:
    __slots__ = ("t", "object")
```

### `intersect` returns *all* roots unfiltered

`Sphere.intersect` deliberately returns both `t` values regardless of sign,
and even when they are numerically equal (the tangent case) — it does
**not** filter out negative `t` (behind the ray) or deduplicate the tangent
case. Filtering for "valid, visible" hits is treated as the responsibility
of a later stage (`hit()` in Phase 9's `World`), not of the shape's own
`intersect` method. This keeps `Sphere.intersect`'s contract simple and
uniform: "here is everywhere this ray touches this shape's surface,
mathematically," with visibility semantics layered on top separately.

### Ordering

Since `t1 = (-b - sqrt_discriminant) / (2a)` and `t2 = (-b +
sqrt_discriminant) / (2a)`, and `sqrt_discriminant >= 0`, the returned list
`[Intersection(t1, ...), Intersection(t2, ...)]` is always ascending by
construction (assuming `a > 0`, which holds for any nonzero ray direction).
This ordering is treated as part of the method's implicit contract and is
relied upon later by `World.intersect`'s sort and `hit()`'s "first
qualifying" scan (Phase 9).

### `normal_at` assumes its input lies on the surface

`normal_at` does not verify that `point` is actually on the sphere — doing
so would cost an extra `sqrt`/comparison on every call, and this method is
called once per intersection per shading computation, a genuine hot path
once rendering full scenes. The precondition is documented rather than
defended against at runtime.

## 6. Testing Strategy

Directly from the project's own phase plan:

- Ray misses the sphere (`discriminant < 0`, expect `[]`).
- Ray hits the sphere at two points — verify both `t` values against
  hand-computed expectations.
- Ray is tangent to the sphere (`discriminant == 0`) — both returned `t`
  values equal.
- Ray originates **inside** the sphere — expect one negative and one
  positive `t` (worked out by hand, not assumed, before writing the test).
- Sphere entirely **behind** the ray — expect both `t` values negative.
- Intersection ordering — confirm ascending by `t`.
- `normal_at` on axis-aligned points on a unit sphere at the origin (easy to
  hand-verify, e.g. normal at `(1,0,0)` should be `(1,0,0)`).
- **Property test**: `normal_at(p).magnitude() ~= 1` for any point `p` on
  the sphere's surface — should always hold given division by `radius` in
  the normal formula.

## 7. Common Mistakes Encountered (and Fixed)

No implementation bugs were found in the final `Sphere.intersect` /
`normal_at` code during review — the main work in this phase was correctly
translating each term of the algebraic derivation (`a`, `b`, `c`,
discriminant, both roots) into code using the `Vector.dot` method already
built in Phase 1, rather than reaching for `math.pow`/manual component
arithmetic that duplicates functionality already available on `Vector`.

## 8. What This Will Be Used For Later

- `Intersection` is the return type threaded through `World.intersect` and
  `hit()` (Phase 9), which generalize "closest visible intersection" from a
  single sphere to an entire scene of arbitrary shapes.
- `normal_at` (as an abstract method on the `Shape` base class) is
  implemented per-shape (`Plane`, `Triangle` in Phase 10) and consumed
  directly by the Phase 5 `lighting()` function.
- The quadratic-solving pattern here — substitute the ray equation into an
  implicit surface equation, solve for `t` — is the general technique
  revisited for any new implicit-surface shape; explicit-surface shapes
  (triangles, via barycentric coordinates) use a different intersection
  strategy introduced in Phase 10.
