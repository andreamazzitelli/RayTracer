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
    geometry/       Point, Vector, Ray, Matrix, transforms, Intersection, Computations
    shapes/         Shape (base), Sphere, Plane, Triangle
    rendering/      Camera, PointLight, Material, lighting(), renderer
    scene/          World, OBJ mesh parser, procedural mesh generators
    image/          Color, Canvas, PPM serialization

tests/
    unit/           Mirrors the src/ structure, one test file per module
    integration/    End-to-end render/shadow/transform tests

examples/
    render_scene.py            A single hand-built scene, rendered to PPM
    random_scene_profiler.py   Generates & profiles a batch of random scenes
    mesh_profile.py            Profiles a procedurally generated triangle mesh

docs/
    01-14 phase-by-phase study notes: theory, derivations, design
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

The full suite (as of Phase 13) is 179 tests across geometry, shapes,
image, rendering, and scene modules, including property-based tests for
every major mathematical invariant (vector normalization, matrix
inverses, rotation periodicity, cross-product perpendicularity, etc.).

## Generating a Render

```bash
python examples/render_scene.py 200 100 renders/my_scene.ppm
```

Arguments are `hsize vsize output_path`, all optional (defaults: 200x100
to `renders/scene.ppm`). A progress bar (via `tqdm`) shows per-row
rendering progress; pass `show_progress=False` to `render()` directly if
scripting a batch of renders where per-render progress bars would just
add noise.

`.ppm` files are plain-text and not natively viewable in most image
viewers. Convert with:

```bash
# via Pillow
python -c "from PIL import Image; Image.open('renders/my_scene.ppm').save('renders/my_scene.png')"

# or via ImageMagick, if installed
convert renders/my_scene.ppm renders/my_scene.png
```

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

Phases 1 through 13 are implemented and tested (points/vectors through
reflection/refraction). Phase 14 (performance/acceleration) is in
progress: several profiling-driven optimizations have been made (cached
transform inverses, a hand-rolled cross product replacing `np.cross`, and
removing NumPy-array backing from `Point`/`Vector`/`Color` in favor of
plain Python floats — each backed by measured before/after profiling
evidence in `docs/14-performance-and-profiling.md`), and a bounding-volume
hierarchy is the next piece of work, justified by confirmed linear
scaling of render time with triangle count on unaccelerated meshes.

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
- [ ] Phase 14 — Performance / Acceleration (profiling done; BVH in progress)
- [ ] Phase 15 — Path Tracing (Monte Carlo / physically based rendering)