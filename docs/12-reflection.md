# Phase 12 — Reflection

## 1. Theory

A reflective surface (mirror, polished metal, water) doesn't just show
ambient/diffuse/specular highlights from direct light sources — it shows a
dimmed copy of whatever else is visible in the direction the surface
reflects. Rendering this requires casting a **new ray** from the hit point,
in the mirrored direction, and recursively asking "what color does *that*
ray see?" — the same `color_at` logic already built for primary rays,
called again on a secondary ray.

## 2. Geometric Intuition

The direction a ray bounces off a mirror-like surface is exactly the
reflection vector already derived in Phase 5: `Vector.reflect`, originally
built to reflect a *light* direction for specular highlights, is reused
here to reflect the *ray's own direction of travel* off the surface
normal. The formula and the geometry are identical — a mirror image is a
mirror image whether the thing being reflected is incoming light or an
incoming ray.

Since a reflected ray can itself strike another reflective surface, this
process is naturally **recursive**. Two mirrors facing each other would
recurse infinitely without a limit — a **depth counter** bounds how many
bounces are computed before giving up and treating any further
contribution as black.

## 3. Equations

Reflection direction, reusing the Phase 5 formula directly on the ray's
direction rather than a light direction:

```
reflect_vector = ray.direction.reflect(normal_vector)
```

No new derivation is needed — this is the same `R = d - 2(d.N)N` geometry,
applied to a different input vector.

## 4. Python / NumPy Representation

`Material` gains a `reflective` coefficient in `[0, 1]` (0 = no
reflection, 1 = perfect mirror):

```python
class Material:
    __slots__ = ("color", "ambient", "diffuse", "specular", "shininess", "reflective")
```

`Computations` gains a precomputed `reflect_vector`, derived alongside
everything else in `prepare_computations`:

```python
reflect_vector = ray.direction.reflect(normal_vector)
```

`World` gains the actual recursive machinery:

```python
MAX_REFLECTION_DEPTH = 5

class World:
    def shade_hit(self, comps: Computations, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
        surface_color = ...  # existing ambient+diffuse+specular sum over lights
        reflected = self.reflected_color(comps, remaining)
        return surface_color + reflected

    def color_at(self, ray: Ray, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
        ...
        return self.shade_hit(comps, remaining)

    def reflected_color(self, comps: Computations, remaining: int = MAX_REFLECTION_DEPTH) -> Color:
        if remaining <= 0:
            return Color(0, 0, 0)
        if math.isclose(comps.object.material.reflective, 0, abs_tol=1e-9):
            return Color(0, 0, 0)

        reflect_ray = Ray(comps.over_point, comps.reflect_vector)
        color = self.color_at(reflect_ray, remaining - 1)
        return color * comps.object.material.reflective
```

## 5. API / Design Decisions

- **`remaining` decrements by exactly 1 per recursive call**, defaulting to
  `MAX_REFLECTION_DEPTH = 5` at the top level (`render()`'s calls into
  `world.color_at`). This is what guarantees termination between two
  mutually reflective surfaces — without it, `color_at -> shade_hit ->
  reflected_color -> color_at -> ...` between two facing mirrors would
  recurse forever.
- **`reflect_ray` originates from `comps.over_point`, not `comps.point`** —
  identical shadow-acne reasoning from Phase 6: a ray starting exactly on
  a surface risks immediately re-intersecting that same surface due to
  floating-point imprecision. The reflected ray needs the same
  epsilon-offset treatment as shadow rays.
- **Two independent early-outs in `reflected_color`**: `remaining <= 0`
  guarantees termination regardless of material; the
  `reflective ≈ 0` check is a pure performance optimization — the vast
  majority of materials in a typical scene are non-reflective, and this
  check avoids the cost of constructing and tracing a whole secondary ray
  (including its own potential further recursion) for each of them.
- **Defaulting `remaining` on every public method** (`shade_hit`,
  `color_at`, `reflected_color`) means every pre-Phase-12 call site and
  test continues to work unchanged — only tests specifically about
  reflection depth need to pass a smaller value explicitly.

## 6. Testing Strategy

- `reflected_color` returns black for a material with `reflective = 0`
  (the default) — confirms the early-out fires correctly for ordinary,
  non-reflective materials.
- `reflected_color` returns a non-black color for a genuinely reflective
  material with something to reflect.
- `reflected_color` returns black at `remaining = 0`, regardless of how
  reflective the material is — the depth-limit guard, tested in isolation
  from the reflectivity guard.
- `shade_hit`'s combined output, with reflection contributing, is at least
  as bright as the surface lighting alone (reflection only adds light,
  never subtracts).
- **The termination test**: two infinite mirror planes facing each other,
  with a ray fired between them — the only real assertion is that
  `color_at` returns *at all*, rather than raising `RecursionError`. This
  is the test that actually proves the depth limit works; a single-bounce
  reflection test cannot distinguish "reflection works" from "the
  recursion happens to terminate."

## 7. Common Mistakes to Watch For

No implementation bugs were found in this project's actual reflection
code during review — the risk areas worth being deliberately careful
about (based on patterns from earlier phases) are:

1. **Forgetting the `remaining` decrement** on the recursive `color_at`
   call inside `reflected_color` — passing `remaining` unchanged instead
   of `remaining - 1` would silently disable the depth limit entirely,
   surfacing only as a `RecursionError` on a scene with mutually
   reflective surfaces, not on simpler test scenes. This is exactly why
   the mutual-mirrors termination test matters as a permanent regression
   test, not just a one-off check.
2. **Using `comps.point` instead of `comps.over_point`** as the reflected
   ray's origin — would reintroduce shadow-acne-style self-intersection
   artifacts, but this time on reflective surfaces (a mirror developing
   visible speckling/noise rather than a clean reflection).
3. **Reflecting the wrong vector** — `ray.direction.reflect(normal)`, not
   `comps.eye_vector.reflect(normal)`. The eye vector is `-ray.direction`,
   so reflecting it would produce a vector pointing the opposite way from
   the correct reflected ray direction. Worth tracing through by hand once
   to confirm which one is correct, rather than assuming the "reflect
   something involving eye and normal" pattern from `lighting()` transfers
   directly — `lighting()`'s reflection was of the *light* direction
   (negated), a genuinely different vector than the ray's own direction.

## 8. What This Will Be Used For Later

- Phase 13 (refraction) reuses the exact same recursive-depth-limit
  pattern (`remaining`, decremented per call, defaulted everywhere) for
  `refracted_color` — the two mechanisms are structurally parallel.
- Phase 15 (path tracing) generalizes this single deterministic reflection
  ray into *many* randomly-sampled rays per bounce (Monte Carlo
  integration over the hemisphere of possible outgoing directions) — the
  core idea of "recursively trace a secondary ray and combine its color"
  carries over directly; what changes is how many secondary rays are
  traced and in which directions.
- Once Phase 14 (performance) is underway, reflection/refraction recursion
  is one of the first places render time balloons — a scene with several
  reflective surfaces facing each other multiplies the number of rays
  traced by up to `MAX_REFLECTION_DEPTH` per primary ray, making this a
  natural candidate to profile explicitly rather than assume is fine.
