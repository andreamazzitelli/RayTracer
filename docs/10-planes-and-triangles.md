# Phase 10 — Planes and Triangles

## 1. Theory

Unlike `Sphere` (an implicit surface, solved via substituting the ray
equation into `|P-C|^2 = r^2` and solving a quadratic), planes and
triangles use different intersection strategies suited to their own
geometry.

## 2. Plane

### Theory

An infinite plane, in local space, is conventionally the `xz`-plane
(`y = 0`), with a constant normal `(0, 1, 0)` everywhere — unlike a sphere,
a flat plane's normal doesn't vary by position at all.

### Derivation

A ray `R(t) = O + tD` intersects `y = 0` where:

```
O.y + t*D.y = 0
t = -O.y / D.y
```

**Special case**: if `D.y` is near zero, the ray is parallel to the plane —
either it never touches it (no intersection) or it lies exactly within the
plane (infinitely many "intersections," conventionally treated as no
intersection for rendering, since a ray lying exactly inside an infinitely
thin plane has no single meaningful hit point to render).

## 3. Triangle

### Theory

A triangle is defined by three vertices `p1, p2, p3`. Two edge vectors,
`e1 = p2 - p1` and `e2 = p3 - p1`, span the triangle's flat plane. Their
cross product gives a constant normal across the entire triangle (computed
once at construction, not recomputed per query, since a flat triangle's
normal never varies — unlike a sphere's).

### Barycentric coordinates

Any point in the triangle's plane can be written as
`P = u*p1 + v*p2 + w*p3` with `u + v + w = 1`. A point lies **inside** the
triangle exactly when `u >= 0`, `v >= 0`, and `w >= 0` (equivalently,
`u + v <= 1` given `w = 1 - u - v`).

### Möller–Trumbore derivation

Setting the ray equation equal to the barycentric form:

```
O + tD = p1 + u*e1 + v*e2
```

Let `T = O - p1`. This is a 3-equation, 3-unknown linear system in
`t, u, v`, solved via Cramer's rule using cross products:

```
P_vec = D x e2
det = e1 . P_vec

u = (T . P_vec) / det
Q_vec = T x e1
v = (D . Q_vec) / det
t = (e2 . Q_vec) / det
```

`det ≈ 0` means the ray is parallel to the triangle's plane. The three
rejection conditions map onto barycentric containment:

- `u < 0 or u > 1` — fails the `u`-boundary
- `v < 0` — fails the `v`-boundary
- `u + v > 1` — fails the combined boundary (equivalent to `w < 0`)

Each is a geometrically distinct way for the ray to miss the triangle by
passing outside one particular edge — worth testing independently (see
Testing Strategy), since a bug isolated to one condition would not be
caught by tests only exercising the other two.

## 4. Python / NumPy Representation

```python
class Plane(Shape):
    def intersect(self, ray: Ray) -> list[Intersection]:
        local_ray = self._to_local_ray(ray)
        if math.isclose(local_ray.direction.y, 0, abs_tol=PLANE_EPSILON):
            return []
        t = -local_ray.origin.y / local_ray.direction.y
        return [Intersection(t, self)]

    def normal_at(self, point: Point) -> Vector:
        world_normal = self.transform.inverse().transpose().apply_to_vector(Vector(0, 1, 0))
        return world_normal.normalize()


class Triangle(Shape):
    __slots__ = ("p1", "p2", "p3", "e1", "e2", "_normal")

    def __init__(self, p1, p2, p3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.e1 = p2 - p1
        self.e2 = p3 - p1
        self._normal = self.e2.cross(self.e1).normalize()

    def intersect(self, ray: Ray) -> list[Intersection]:
        local_ray = self._to_local_ray(ray)
        direction_cross_e2 = local_ray.direction.cross(self.e2)
        det = self.e1.dot(direction_cross_e2)

        if math.isclose(det, 0, abs_tol=TRIANGLE_EPSILON):
            return []

        f = 1.0 / det
        p1_to_origin = local_ray.origin - self.p1
        u = f * p1_to_origin.dot(direction_cross_e2)
        if u < 0 or u > 1:
            return []

        origin_cross_e1 = p1_to_origin.cross(self.e1)
        v = f * local_ray.direction.dot(origin_cross_e1)
        if v < 0 or (u + v) > 1:
            return []

        t = f * self.e2.dot(origin_cross_e1)
        return [Intersection(t, self)]

    def normal_at(self, point: Point) -> Vector:
        world_normal = self.transform.inverse().transpose().apply_to_vector(self._normal)
        return world_normal.normalize()
```

Both shapes reuse the `_to_local_ray` pattern established for `Sphere` in
Phase 9 (transform incoming ray via `self.transform.inverse()`) and the
inverse-transpose pattern for converting local normals to world space.

## 5. API / Design Decisions

- **`Triangle._normal` is precomputed once in `__init__`**, not recomputed
  per `normal_at` call — since it's genuinely constant across the whole
  triangle, unlike `Sphere`'s position-dependent normal, recomputing it
  per query would be pure waste.
- **`Triangle.normal_at`'s `point` parameter is unused.** This is a
  deliberate interface tradeoff: `Shape.normal_at` is a uniform abstract
  method across all shapes, and `Triangle` is forced to accept a parameter
  it doesn't need purely to satisfy that shared interface. Accepted as a
  common, reasonable cost of interface uniformity across genuinely
  different shapes, rather than treated as an oversight.
- **`e2.cross(e1)` (not `e1.cross(e2)`) for the normal** — the order
  determines which way the resulting normal faces, tied to the triangle's
  vertex winding order convention; verified against a hand-picked triangle
  where the expected outward direction is known.

## 6. Testing Strategy

**Plane**: ray parallel to the plane (no hit); ray exactly coplanar (no
hit, per the "infinite intersections treated as none" convention); ray from
above and from below intersecting at the expected `t`; normal identical
regardless of which point on the plane is queried.

**Triangle**: hits the interior (hand-computed `t` for a simple
axis-aligned triangle); three separate miss tests, one per edge (`u`
boundary, `v` boundary, `u+v` boundary), since Möller–Trumbore's rejection
logic has three independent failure paths; ray parallel to the triangle's
plane; normal constant regardless of which point on the triangle's surface
is queried.

## 7. What This Will Be Used For Later

- `Triangle` is the building block for mesh support (Phase 11) — an OBJ
  file's faces are parsed directly into `Triangle` instances via fan
  triangulation.
- Both shapes participate in `World.intersect`/`shade_hit` identically to
  `Sphere`, with no special-casing needed elsewhere in the renderer — this
  is the payoff of the `Shape` abstraction and the transform-to-local-space
  pattern being applied uniformly.
- Once acceleration structures (Phase 14, BVH) are introduced, triangle
  meshes (potentially thousands of `Triangle` instances per model) are the
  primary motivating case — a linear scan of `World.objects` becomes a real
  bottleneck specifically once meshes are loaded via Phase 11's OBJ parser.
