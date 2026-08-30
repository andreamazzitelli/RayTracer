# RayTracer

A ray tracer built from scratch in Python, as a rigorous learning project
in computer graphics, linear algebra, geometry, numerical computation, and
rendering algorithms. The goal is not primarily a fast or feature-complete
renderer — it's a deep, tested, documented understanding of how ray tracing
actually works, mathematically and computationally, built incrementally
from first principles.

## Project Philosophy

- **Theory before implementation.** Every feature follows the cycle:
  theory → derivation → geometric intuition → design tests → implement →
  test → visualize → document.
- **NumPy as a foundation, not a crutch.** NumPy is used where it
  genuinely helps (bulk array operations, real linear algebra like matrix
  inversion/determinants); it is deliberately *not* used where its
  per-call overhead swamps tiny, high-frequency operations (see
  `docs/14-performance-and-profiling.md` for a concrete case study).
- **Test everything, including the math.** Property-based tests (e.g.
  `normalize().magnitude() ~= 1`, `cross(a,b).dot(a) ~= 0`) sit alongside
  hand-computed examples, since they validate the underlying relationship
  rather than one fixed input/output pair.
- **Profile before optimizing.** Performance work (Phase 14) is driven
  entirely by `cProfile` evidence — every optimization documented in this
  project is backed by a before/after measurement, not a guess.

## Project Structure

```
src/raytracer/
    geometry/       Point, Vector, Ray, Matrix, transforms, Intersection,
                     Computations, BoundingBox, sampling (Monte Carlo helpers)
    shapes/         Shape (base), Sphere, Plane, Triangle, Mesh (BVH-accelerated)
    rendering/      Camera, PointLight, Material, lighting(), renderer
                     (deterministic), path_tracer (Monte Carlo)
    scene/          World, OBJ mesh parser, procedural mesh generators

tests/
    unit/           Mirrors the src/ structure, one test file per module
    integration/    End-to-end render/shadow/transform tests

examples/
    generate_render.py         Configurable CLI: scene/camera/mode via arguments
                                (deterministic or path-traced, in one script)
    random_scene_profiler.py   Generates & profiles a batch of random scenes
    mesh_profile.py            Profiles a procedurally generated triangle mesh
    path_tracing_test.py       Validates path tracer convergence (sanity checks
                                + numeric + visual)

docs/
    01-15 phase-by-phase study notes: theory, derivations, design
    decisions, bugs found and fixed, and testing strategy for each phase.

renders/            Output directory for rendered .ppm files
```

## Setup

Requires Python 3.11+ (developed against 3.10 in practice; see notes
below if your environment differs).

```bash
git clone <this repo>
cd RayTracer

python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows

pip install -e . --break-system-packages
```

The `-e` (editable) install means changes to `src/raytracer/` take effect
immediately without reinstalling. If `pip install -e .` succeeds but
`import raytracer` still fails outside the `tests/` directory, check:

1. `which python` and `which pip` resolve to the **same** `.venv` — if
   they diverge, use `python -m pip install -e .` instead of a bare `pip`.
2. `pyproject.toml` has both a `[build-system]` table AND
   `[tool.setuptools.packages.find] where = ["src"]` — missing either
   causes a silently-broken or misdirected editable install even though
   `pip` reports success.
3. `grep -rn "\.transform\.inverse()"` in `src/raytracer/shapes/` and
   `src/raytracer/rendering/camera.py` should list only the base-class
   caching implementation and camera.py — if shape subclasses still call
   `.transform.inverse()` directly, that's an unrelated but real
   performance bug (see `docs/14-performance-and-profiling.md`), not an
   import issue.

## Running Tests

```bash
pytest                  # run the full suite
pytest -v                # verbose, one line per test
pytest tests/unit/geometry/   # a single module's tests
ruff check src/ tests/   # linting
mypy src/ --strict       # static type checking
```

The full suite (as of Phase 15) is 206 tests across geometry, shapes,
image, rendering, and scene modules, including property-based tests for
every major mathematical invariant (vector normalization, matrix
inverses, rotation periodicity, cross-product perpendicularity,
cosine-weighted-sample hemisphere membership, etc.), a pixel-perfect
BVH-vs-brute-force equivalence test, and a statistical variance-reduction
test for the path tracer.

## Generating a Render

A single configurable entry point handles both rendering modes:

```bash
# Deterministic (Phong + shadows + reflection + refraction), default scene/camera
python examples/generate_render.py

# Higher resolution, custom output path
python examples/generate_render.py --hsize 400 --vsize 200 --output renders/big.ppm

# Path-traced (Monte Carlo), reproducible via a fixed seed
python examples/generate_render.py --mode path-trace --samples 128 --max-depth 5 --seed 7

# Custom camera position, target, and field of view
python examples/generate_render.py --camera-from 0 2 -6 --camera-to 0 1 0 --fov 60

# Also save a PNG alongside the PPM (requires Pillow)
python examples/generate_render.py --png
```

Run `python examples/generate_render.py --help` for the full argument
list. Path-traced renders default to a scene with an emissive ceiling
(no `PointLight`) since path tracing needs something actually emitting
light to bounce off of; deterministic renders use a `PointLight`-lit
scene instead.

A progress bar (via `tqdm`) shows per-row rendering progress in either
mode; pass `--no-progress` when scripting many renders back-to-back,
where per-render progress bars would just add noise.

`.ppm` files are plain-text and not natively viewable in most image
viewers. Convert with `--png` above, or manually:

```bash
# via Pillow
python -c "from PIL import Image; Image.open('renders/my_scene.ppm').save('renders/my_scene.png')"

# or via ImageMagick, if installed
convert renders/my_scene.ppm renders/my_scene.png
```

## Bounding Volume Hierarchy (Mesh Acceleration)

Triangle meshes (whether loaded from OBJ or procedurally generated) can
be wrapped in a `Mesh`, which builds a bounding-volume hierarchy (BVH)
over the triangles and prunes entire subtrees a ray couldn't possibly hit,
instead of testing every triangle individually:

```python
from raytracer.shapes.mesh import Mesh
from raytracer.scene.mesh_generators import uv_sphere_triangles

triangles = uv_sphere_triangles(radius=1.5, latitude_segments=20, longitude_segments=20)
mesh = Mesh(triangles)
world = World(objects=[floor, mesh], lights=[light])  # add Mesh directly, like any shape
```

This was built and adopted only after profiling confirmed render time
scales *linearly* with triangle count under brute-force testing (see
`docs/14-performance-and-profiling.md`) — not assumed necessary up front.
A direct correctness check (BVH-accelerated render vs. brute-force render
of the same scene, pixel-by-pixel) showed **zero mismatches with a 44x
speedup** on an 800-triangle test mesh; that comparison is a permanent
regression test in `tests/unit/shapes/test_mesh.py`.

## Path Tracing (Physically Based Rendering)

`World.path_trace(ray, depth, rng)` implements Monte Carlo path tracing:
a stochastic estimator of the full rendering equation (direct lighting +
recursively sampled indirect lighting via cosine-weighted hemisphere
sampling), bounded by both a hard recursion-depth cap and Russian
roulette for unbiased early termination. A single call is one noisy
sample; `render_path_traced()` (in `rendering/path_tracer.py`) averages
many samples per pixel, converging toward a physically based image as
sample count increases — at the cost of significantly more render time
and visible noise at low sample counts.

Materials can now carry an `emissive` color, letting geometry itself act
as a light source (the physically-based convention), independent of the
existing `PointLight` mechanism used by deterministic rendering.

See `docs/15-path-tracing.md` for the full derivation (the rendering
equation, Monte Carlo integration, cosine-weighted importance sampling,
Russian roulette) and `examples/path_tracing_test.py` for the
convergence-validation methodology used to confirm it actually works
before trusting any render it produces.

## Profiling

Two profiling scripts exist for different purposes:

**`random_scene_profiler.py`** — generates a batch of randomized scenes
(random resolution from `AVAILABLE_SIZES`, random object count/placement/
materials, random lights, random camera position), profiles each with
`cProfile`, and produces a comparative summary (a table if `pandas` is
installed, plain text otherwise) plus a thumbnail grid of every render
(requires `matplotlib` and `pillow`). Useful for catching bottlenecks that
only show up with certain combinations of scene properties, and for
regression-testing a performance fix across many scenes at once via a
fixed `--seed`.

```bash
python examples/random_scene_profiler.py --num-scenes 15 --seed 7
```

**`mesh_profile.py`** — profiles a single scene containing a procedurally
generated triangle mesh (a UV-sphere, built in-memory to control exact
triangle count without needing an OBJ file), specifically for validating
mesh-heavy performance work (e.g. the bounding-volume-hierarchy work in
Phase 14). Triangle count scales via `rings`/`segments`.

```bash
python examples/mesh_profile.py --triangles 800
```

Both scripts run identically in a plain terminal or in a Google Colab
notebook cell (import the functions directly and call
`run_experiment(...)`/`summarize(...)` rather than running as `__main__`).

### Interpreting profiler output

Always compare **call counts**, not just wall-clock time, when validating
a performance fix — wall-clock time is sensitive to system load and can
mask or fake an improvement; a redundant computation being eliminated
shows up unambiguously as a drop in `ncalls` for the relevant function
regardless of what else the machine was doing during the run.

## Current Status

All 15 phases of the original roadmap are implemented and tested — from
Points and Vectors through Monte Carlo path tracing. The project's full
test suite (206 tests) passes, including property-based mathematical
invariant tests, a pixel-perfect BVH-vs-brute-force equivalence check, and
statistical convergence/variance tests for the stochastic path tracer.

Phase 14 (performance) in particular is documented as a real investigative
process, not a checklist: every optimization (cached transform inverses,
a hand-rolled cross product replacing `np.cross`, removing NumPy-array
backing from `Point`/`Vector`/`Color`, and finally the BVH itself) is
backed by concrete before/after `cProfile` evidence in
`docs/14-performance-and-profiling.md` — including two cases where an
initial fix attempt was verified *not* to have worked (via call-count
evidence, not just wall-clock time) and had to be corrected before being
trusted.

See `docs/` for the full phase-by-phase theory, derivations, design
decisions, and lessons learned (including real bugs found and fixed along
the way — kept in the docs deliberately, since understanding a bug's root
cause is often more instructive than the corrected formula alone).

## Roadmap

- [x] Phase 1 — Points and Vectors
- [x] Phase 2 — Rays
- [x] Phase 3 — Canvas and Image Output
- [x] Phase 4 — Spheres and Ray Intersection
- [x] Phase 5 — Normals and Shading (Phong model)
- [x] Phase 6 — Shadows
- [x] Phase 7 — Matrices and Transformations
- [x] Phase 8 — Camera
- [x] Phase 9 — World / Scene
- [x] Phase 10 — Planes and Triangles
- [x] Phase 11 — OBJ / Mesh Support
- [x] Phase 12 — Reflection
- [x] Phase 13 — Refraction
- [x] Phase 14 — Performance / Acceleration (profiling-driven fixes + BVH)
- [x] Phase 15 — Path Tracing (Monte Carlo / physically based rendering)

## Where to Go From Here

The roadmap is complete, but a project like this never runs out of
legitimate next steps, should you want to keep going:

- **Surface-area-heuristic (SAH) BVH construction** — the current BVH uses
  a simple largest-extent-axis median split; a proper SAH-based split
  would build a measurably better tree for irregular meshes.
- **Importance sampling toward lights** in the path tracer, rather than
  only cosine-weighted hemisphere sampling — would substantially reduce
  variance (and thus required sample count) for scenes with small,
  bright light sources.
- **Textures** — currently every material has a single flat color;
  UV-mapped image textures would be a natural extension of the OBJ/mesh
  work in Phase 11.
- **Multi-threading or multiprocessing** the per-pixel render loop —
  each pixel's computation is fully independent, making this an
  embarrassingly parallel workload that was left single-threaded
  throughout this project by design, to keep the profiling story in
  Phase 14 simple and attributable to algorithmic/data-representation
  choices rather than parallelism.