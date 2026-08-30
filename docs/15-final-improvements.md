# Phase 15 — Path Tracing (Monte Carlo / Physically-Based Rendering)

## 1. Theory: The Rendering Equation

Every prior phase computed **direct** illumination (light source → surface
→ eye) plus a small, fixed number of **deterministic** secondary rays
(exactly one reflected ray, one refracted ray). Real light also bounces
off *other diffuse surfaces* before reaching the eye — this is **indirect
illumination**, and it's why a white wall next to a red carpet picks up a
faint red tint (light bounces off the carpet, picks up red, then bounces
onto the wall). The full physical description is the **rendering
equation**:

```
L_o(w_o) = L_e(w_o) + Integral[ f_r(w_i, w_o) * L_i(w_i) * cos(theta_i) dw_i ]
```

integrated over the entire hemisphere of incoming directions at a point.
This integral has no closed-form solution for arbitrary scenes — the only
way to evaluate it is to **estimate** it.

## 2. Monte Carlo Integration

Randomly sample directions `w_i`, evaluate the integrand at each, and
average. The estimate's error shrinks proportionally to `1/sqrt(N)` for
`N` samples — this is *why* path-traced renders look visibly noisy at low
sample counts and progressively clean up as more samples accumulate; the
noise is the direct, visible signature of Monte Carlo estimation error,
not a bug.

### Importance sampling: cosine-weighted hemisphere sampling

Sampling directions **uniformly** across the hemisphere wastes samples on
directions that barely matter (grazing angles contribute almost nothing,
via the `cos(theta)` term). **Importance sampling** biases sample
selection toward directions where the integrand is large — sampling with
probability density proportional to `cos(theta)` means more samples land
exactly where they contribute most, converging faster for the same sample
budget.

### The key simplification for Lambertian surfaces

For a perfectly diffuse (Lambertian) surface, the BRDF is a constant,
`f_r = albedo / pi`. Cosine-weighted sampling has probability density
`p(w) = cos(theta) / pi`. Substituting both into the single-sample Monte
Carlo estimator:

```
estimate = f_r * L_i * cos(theta) / p(w)
         = (albedo/pi) * L_i * cos(theta) / (cos(theta)/pi)
         = albedo * L_i
```

**Every term except `albedo * L_i` cancels exactly.** This is the reason
cosine-weighted sampling is the standard choice for diffuse surfaces in
practice — it isn't only about faster convergence, the estimator itself
collapses to something almost trivial to implement: multiply the
incoming (recursively traced) radiance by the surface's own albedo.

## 3. Russian Roulette

A hard recursion-depth cutoff (as used for reflection/refraction in
Phases 12-13) introduces **bias** — it silently discards all light that
would have arrived via paths longer than the cutoff, systematically
under-estimating brightness. **Russian roulette** instead terminates each
path *stochastically*: continue with probability `p` (tied here to the
surface's own reflectance/albedo — brighter, more-reflective surfaces are
more likely to matter, so they're kept alive longer), and when a path
does survive, divide its contribution by `p` to compensate. This keeps the
estimator **unbiased** — correct in expectation — while still bounding how
deep recursion tends to go in practice, which is a fundamentally different
guarantee than a hard depth limit provides.

## 4. Python Implementation

**`geometry/sampling.py`** — cosine-weighted hemisphere sampling via the
standard concentric-disk mapping, transformed into an orthonormal basis
built around the surface normal:

```python
def random_cosine_weighted_hemisphere(normal, rng):
    u1, u2 = rng.random(), rng.random()
    r = math.sqrt(u1)
    theta = 2 * math.pi * u2
    x, y = r * math.cos(theta), r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))

    tangent = arbitrary_vector.cross(normal).normalize()
    bitangent = normal.cross(tangent)
    direction = ((tangent * x) + (bitangent * y) + (normal * z)).normalize()

    pdf = z / math.pi  # z is cos(theta) in the local frame
    return direction, pdf
```

**`World.path_trace`** — an additive method alongside (not replacing)
`shade_hit`/`color_at`, implementing direct lighting exactly as before,
plus one stochastic indirect-diffuse bounce (gated by Russian roulette),
plus deterministic reflection/refraction (reusing the same recursive
structure, now calling `path_trace` instead of `color_at`):

```python
def path_trace(self, ray, depth, rng):
    ...
    direct = <same lighting() sum over all lights as shade_hit>

    if depth <= 0:
        return direct

    albedo = material.color * material.diffuse
    survival = min(max(albedo.r, albedo.g, albedo.b), 0.95)

    indirect = Color(0, 0, 0)
    if survival > 1e-6 and rng.random() < survival:
        direction, pdf = random_cosine_weighted_hemisphere(comps.normal_vector, rng)
        if pdf > 1e-6:
            incoming = self.path_trace(Ray(comps.over_point, direction), depth - 1, rng)
            indirect = albedo.hadamard(incoming) * (1.0 / survival)

    reflected = <deterministic, as in Phase 12, but recursing via path_trace>
    refracted = <deterministic, as in Phase 13, but recursing via path_trace>

    return direct + indirect + reflected + refracted
```

**`rendering/path_tracer.py`** — `render_path_traced`, a separate function
from `render()`, firing `samples_per_pixel` randomly jittered rays per
pixel and averaging:

```python
def render_path_traced(camera, world, samples_per_pixel=16, max_depth=5, seed=None, ...):
    rng = random.Random(seed)
    for y in rows:
        for x in range(camera.hsize):
            accumulated = Color(0, 0, 0)
            for _ in range(samples_per_pixel):
                jittered_x = x + rng.random() - 0.5
                jittered_y = y + rng.random() - 0.5
                ray = camera.ray_for_pixel(jittered_x, jittered_y)
                accumulated += world.path_trace(ray, max_depth, rng)
            image.write_pixel(x, y, accumulated * (1.0 / samples_per_pixel))
```

## 5. API / Design Decisions

- **Additive, not a replacement.** `path_trace`/`render_path_traced` are
  entirely new methods/functions, leaving `shade_hit`/`color_at`/`render`
  completely untouched. This was a deliberate choice to avoid risking any
  regression in the 193 tests already passing from prior phases — verified
  directly: the full pre-existing suite passed unchanged after adding
  path tracing.
- **Per-pixel jitter as a side effect gives free anti-aliasing.** Firing
  multiple randomly-offset rays per pixel and averaging naturally
  smooths jagged edges — a side benefit of the sampling approach, not a
  separately implemented feature.
- **A local, seeded `random.Random` instance, not the global `random`
  module** — makes an entire render fully reproducible given the same
  seed, which matters for comparing renders (e.g. verifying variance
  actually decreases with more samples) without sampling noise
  confounding the comparison.
- **Reflection/refraction stayed deterministic (one ray each), not
  stochastically sampled**, even inside `path_trace`. A more complete
  physically-based renderer would sample glossy/rough reflections
  stochastically too; this project's version keeps mirrors and glass
  exactly as precise as Phases 12-13 left them, and only the *diffuse*
  term becomes probabilistic. This is a real, documented simplification,
  not an oversight — extending reflection to sampled microfacet
  distributions would be a natural next step beyond this project's
  current scope.
- **Russian roulette survival probability tied to `max(albedo channels)`,
  clamped to 0.95** — a simple, standard heuristic (brighter surfaces are
  more likely to matter and are kept alive longer) rather than a more
  sophisticated adaptive scheme.

## 6. Testing Strategy

- Sampler-level checks: 2000+ sampled directions are unit length, stay on
  the correct hemisphere side (`direction.dot(normal) >= 0`), and have
  positive pdf, across several differently-oriented normals.
- **A statistical concentration check**: cosine-weighted samples should
  land disproportionately close to the normal compared to what uniform
  hemisphere sampling would produce — distinguishing this implementation
  from an easy-to-confuse uniform-sampling alternative that would compile
  and run without error but converge more slowly.
- **`path_trace(ray, depth=0)` must exactly equal `color_at(ray,
  remaining=0)`** — the single most important correctness test in this
  phase. It proves the direct-lighting computation inside `path_trace` is
  the *same* computation as the already-verified deterministic path, not
  a parallel reimplementation that merely looks similar and could harbor
  its own independent bugs.
- **Mutual-mirrors termination**, mirroring the Phase 12 test — confirms
  `path_trace`'s recursive reflection handling is still bounded and
  doesn't reintroduce infinite recursion.
- **Reproducibility**: identical seed produces identical output;
  different seeds produce at least one differing pixel — confirms
  randomness is genuinely being used and is genuinely controllable.
- **A convergence test comparing variance, not a single pixel's value at
  two sample counts.** Comparing one pixel's color at `samples=4` vs.
  `samples=64` cannot distinguish "more samples converged" from "ordinary
  Monte Carlo noise happened to differ" — the correct test computes the
  *variance across many independent renders* at each sample count, and
  confirms variance is lower (not necessarily zero, just lower) at the
  higher sample count. This is the properly statistical way to validate a
  Monte Carlo estimator's convergence behavior.

## 7. Common Mistakes to Watch For

1. **Comparing a single low-sample and high-sample pixel directly** and
   expecting the higher-sample one to be "more correct" in an absolute
   sense — both are unbiased estimates of the same true value; the
   higher-sample one merely has *lower expected variance*, which only
   shows up reliably when averaged across many independent trials, not
   guaranteed on any single comparison. An early draft of this project's
   test suite made exactly this mistake before being corrected to a
   proper variance-based statistical test.
2. **Forgetting to divide by the Russian-roulette survival probability**
   on a surviving indirect bounce — this would introduce systematic bias
   (the renderer would converge to a *wrong*, too-dark answer no matter
   how many samples are used), rather than merely being noisier.
3. **Reusing the global `random` module instead of a local
   `random.Random` instance** — would make renders non-reproducible and
   would make comparing "before vs. after a change" impossible to do
   cleanly, since two runs could never be guaranteed to sample identical
   directions.

## 8. What This Completes

This is the final phase of the originally planned roadmap. The project
now spans, in one continuous, tested, documented arc: 3D points/vectors
grounded in explicit dot/cross-product derivations, ray-object
intersection via both implicit (quadratic, for spheres) and explicit
(barycentric, for triangles) methods, the Phong reflection model derived
from first principles, shadow rays with an explained floating-point
workaround, 4x4 homogeneous transformation matrices derived from the
Point/Vector `w` distinction established on day one, a full camera and
scene abstraction, recursive reflection and refraction with correct
nested-medium tracking, a profiling-driven performance investigation that
found and fixed three genuinely non-obvious bottlenecks (each backed by
measured before/after evidence), a bounding-volume hierarchy validated
pixel-for-pixel against brute force before being trusted, and finally a
physically-motivated Monte Carlo path tracer built on the same
`World`/`Camera`/`Material` abstractions used since Phase 9 — with every
step's underlying mathematics derived, tested, and documented rather than
taken on faith.