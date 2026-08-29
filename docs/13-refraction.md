# Phase 13 — Refraction

## 1. Theory

Transparent materials (glass, water, clear plastic) bend light passing
through them rather than simply reflecting or absorbing it. This bending
depends on the **refractive index** (`n`) of each material involved — a
measure of how much that material slows light down relative to a vacuum
(vacuum/air ≈ 1.0, water ≈ 1.33, glass ≈ 1.5, diamond ≈ 2.4). The
relationship between the angle of incidence and the angle of refraction is
given by **Snell's law**:

```
n1 * sin(theta_i) = n2 * sin(theta_t)
```

where `n1` is the refractive index of the material the ray is currently
in, and `n2` is the material it's entering.

## 2. Geometric Intuition

Light entering a denser medium (higher `n`) at an angle bends *toward* the
normal; light exiting into a less dense medium bends *away* from the
normal — this is why a straw in a glass of water looks bent at the
surface. At a steep enough angle exiting into a less dense medium, the
bend becomes so extreme that no refraction angle exists at all — the light
cannot exit and instead reflects entirely back into the denser medium.
This is **total internal reflection**, and it's why looking up at the
water's surface from underwater, past a certain angle, shows a mirror-like
reflection of the water itself rather than a view of the world above.

## 3. Derivations

Given a unit incident ray direction `d`, a unit surface normal `N`
(oriented against `d`), and `n_ratio = n1/n2`:

```
cos(theta_i) = -d . N
sin^2(theta_t) = n_ratio^2 * (1 - cos^2(theta_i))
```

If `sin^2(theta_t) > 1`, **no real angle satisfies Snell's law** — this is
exactly the total-internal-reflection condition, and no refracted ray
should be cast at all (only a reflected one, handled separately by Phase
12's mechanism). Otherwise:

```
cos(theta_t) = sqrt(1 - sin^2(theta_t))
refracted_direction = n_ratio * d + (n_ratio * cos(theta_i) - cos(theta_t)) * N
```

## 4. The n1/n2 Tracking Problem

The genuinely fiddly part of this phase: at any single intersection,
determining *which* two refractive indices are involved (the medium being
left, `n1`, and the medium being entered, `n2`) requires knowing which
transparent objects the ray currently sits inside — not just the object
being hit. For nested or overlapping transparent shapes (e.g. a glass
marble suspended inside a glass sphere of water), naively using the hit
object's own refractive index for both `n1` and `n2` is wrong.

The correct approach replays **every intersection along the ray, in
order**, maintaining a stack of "currently entered" objects:

- Entering an object's surface (first time seen) pushes it onto the stack.
- Exiting an object's surface (already on the stack) pops it off.
- At the moment the intersection actually being shaded is reached: `n1` is
  read from the top of the stack **before** updating it (the medium being
  left), and `n2` is read **after** updating it (the medium being
  entered) — falling back to `1.0` (vacuum/air) if the stack is empty at
  either point.

This is a **stack-based simulation of ray traversal through nested
media**, not a lookup on the hit object alone — the classic worked example
is three overlapping glass spheres of differing refractive index, which
produces six intersections along one ray, each with a distinct, only-
correctly-derivable-via-the-stack `(n1, n2)` pair.

## 5. Python / NumPy Representation

`Material` gains `transparency` and `refractive_index`:

```python
class Material:
    __slots__ = (
        "color", "ambient", "diffuse", "specular", "shininess",
        "reflective", "transparency", "refractive_index",
    )
```

`Computations` gains `n1`, `n2`, and `under_point` (the refraction analog
of `over_point` — offset *into* the surface rather than away from it, so
a refracted ray starting just past the boundary it crossed doesn't
immediately re-intersect the same surface from the wrong side):

```python
def prepare_computations(intersection, ray, all_intersections=None):
    ...
    under_point = point - normal_vector * UNDER_EPSILON
    n1, n2 = _compute_refractive_indices(intersection, all_intersections)
    ...
```

The stack-replay logic:

```python
def _compute_refractive_indices(hit_intersection, all_intersections):
    containers = []
    n1 = n2 = 1.0

    for i in all_intersections:
        is_hit = i is hit_intersection

        if is_hit:
            n1 = containers[-1].material.refractive_index if containers else 1.0

        if i.object in containers:
            containers.remove(i.object)
        else:
            containers.append(i.object)

        if is_hit:
            n2 = containers[-1].material.refractive_index if containers else 1.0
            break

    return n1, n2
```

`World.refracted_color` mirrors `reflected_color`'s structure exactly,
substituting Snell's law for the reflection formula:

```python
def refracted_color(self, comps, remaining=MAX_REFLECTION_DEPTH):
    if remaining <= 0:
        return Color(0, 0, 0)
    if math.isclose(comps.object.material.transparency, 0, abs_tol=1e-9):
        return Color(0, 0, 0)

    n_ratio = comps.n1 / comps.n2
    cos_i = comps.eye_vector.dot(comps.normal_vector)
    sin2_t = (n_ratio**2) * (1 - cos_i**2)

    if sin2_t > 1:
        return Color(0, 0, 0)  # total internal reflection

    cos_t = math.sqrt(1.0 - sin2_t)
    direction = (comps.normal_vector * (n_ratio * cos_i - cos_t)) - (comps.eye_vector * n_ratio)

    refract_ray = Ray(comps.under_point, direction)
    color = self.color_at(refract_ray, remaining - 1)
    return color * comps.object.material.transparency
```

`color_at` must pass **all** intersections (not just the hit) into
`prepare_computations`, since `n1`/`n2` require the full stack-replay:

```python
def color_at(self, ray, remaining=MAX_REFLECTION_DEPTH):
    intersections = self.intersect(ray)
    intersection = hit(intersections)
    if intersection is None:
        return Color(0, 0, 0)
    comps = prepare_computations(intersection, ray, intersections)  # note: full list
    return self.shade_hit(comps, remaining)
```

## 6. API / Design Decisions

- **`refracted_color` structurally mirrors `reflected_color`** —
  same `remaining <= 0` guard, same near-zero-coefficient early-out
  (`transparency` instead of `reflective`), same pattern of building a
  secondary `Ray` and recursing into `color_at`. Deliberately kept
  parallel rather than differently structured, since both are the same
  underlying idea (trace a secondary ray, scale its contribution) applied
  to a different physical phenomenon.
- **The total-internal-reflection check (`sin2_t > 1`) returns black
  directly from `refracted_color`**, rather than attempting to redirect
  that energy into `reflected_color`. Physically, energy that can't
  refract *does* reflect instead — but this project's `reflected_color`
  is driven purely by the `reflective` material coefficient, independent
  of whether TIR occurred. A more physically complete renderer would
  route TIR energy into the reflection term explicitly; this
  implementation treats them as separate, coefficient-driven effects. Worth
  flagging as a simplification, not an oversight.
- **`_compute_refractive_indices` defaults to `1.0` (air/vacuum) at both
  empty-stack cases** — entering from open space, or exiting back into
  open space, both correctly fall back to the "not inside anything"
  refractive index.
- **`prepare_computations`'s `all_intersections` parameter defaults to
  `None`**, falling back to a single-element list containing just the hit
  intersection. This keeps every non-refraction call site (most existing
  tests) working unchanged, at the cost of `n1`/`n2` being potentially
  wrong (not accounting for containment) if a caller relies on the
  default in a scene that actually has nested transparent objects — a
  deliberate default-for-convenience tradeoff, not a universally safe one.

## 7. Testing Strategy

- The three-glass-spheres `n1`/`n2` table — six intersections, each with
  an independently-derivable expected `(n1, n2)` pair from the
  containment-stack logic. This is the test that actually exercises
  nesting; anything with a single transparent object cannot distinguish
  correct stack logic from a naive "always use the hit object's own
  index" shortcut.
- `under_point` is offset to the correct side of the surface (opposite
  direction from `over_point`).
- `refracted_color` returns black for an opaque material
  (`transparency = 0`).
- `refracted_color` returns black at `remaining = 0`.
- `refracted_color` returns black under total internal reflection — a ray
  exiting a sphere at a steep angle relative to the normal, with the
  angle chosen specifically to guarantee `sin2_t > 1` (not left to chance).
- `refracted_color` produces a genuinely non-black result for a properly
  refracting ray through a transparent object with something on the other
  side to see.
- Full `shade_hit` integration: a transparent floor over a colored object
  underneath it — critically, the object underneath must actually be
  present, or the refracted ray hits nothing regardless of whether
  refraction math is correct, and the test cannot distinguish "refraction
  works" from "refraction was never really exercised."

## 8. Common Mistakes Encountered (and Fixed)

1. **An initial refraction integration test used a transparent floor with
   nothing underneath it**, then asserted the shaded color differed from
   the non-refractive case. Since the refracted ray hit nothing either
   way (returning black either from `transparency=0` or from a ray into
   empty space), the assertion passed *without the refraction code path
   being meaningfully exercised at all* — a passing test that proved
   nothing. Fixed by adding a distinctly colored object beneath the floor
   for the refracted ray to actually intersect, and asserting against the
   exact expected color rather than a relative "differs from" comparison.
   This is a useful general lesson: a test that passes on both the
   correct and a plausible incorrect implementation isn't actually
   testing the behavior it claims to.

## 9. What This Will Be Used For Later

- Phase 15 (path tracing) generalizes both reflection and refraction into
  probabilistic sampling — rather than a single deterministic reflected
  ray and a single deterministic refracted ray, physically based
  rendering traces many randomly sampled rays per bounce, weighted by how
  likely each direction is given the material's properties (a mirror
  concentrates probability into one direction; a rough/diffuse surface
  spreads it across a hemisphere). The `n1`/`n2` stack-tracking machinery
  built here still applies unchanged to that setting.
- The pattern of "replay all intersections along a ray to track state"
  established in `_compute_refractive_indices` is a technique worth
  recognizing outside refraction too — anywhere ray traversal needs to
  know not just the closest hit but the full ordered sequence of what was
  crossed (e.g. volumetric effects like fog or subsurface scattering,
  should the project extend that far).
