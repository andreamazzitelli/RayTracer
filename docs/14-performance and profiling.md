# Phase 14 — Performance / Acceleration (Part 1: Profiling-Driven Optimization)

## 1. Theory: Why Profile Before Optimizing

Ray tracing's cost structure is `O(pixels x objects x rays_per_pixel)` in
the worst case — every pixel fires a primary ray, which may spawn shadow
rays per light, plus recursive reflection/refraction rays up to a depth
limit, and each of those rays is tested against every object in the
`World`. Given that structure, it's tempting to *assume* the bottleneck is
"testing too many objects per ray" and jump straight to building a spatial
acceleration structure (a BVH). This project's own stated principle —
*use profiling to determine actual bottlenecks rather than optimizing
prematurely* — turned out to matter concretely: several real, measured
bottlenecks in this codebase had nothing to do with object count at all.

## 2. Profiling Setup

- **`cProfile`**, Python's built-in deterministic profiler, wrapped around
  `render()` calls, with `pstats.Stats(...).sort_stats("cumulative")` to
  rank functions by total time spent in them and everything they call.
- **A progress bar** (`tqdm`, wrapping the outer per-row loop in
  `render()`, one update per row rather than per pixel to avoid its own
  overhead distorting timing) — a UX addition, not a profiling tool, but
  necessary once renders started taking minutes and needed visible
  progress feedback.
- **A randomized scene generator/profiler batch script**: generates scenes
  with random resolution (from a fixed list), random object count,
  placement, and materials, random lights, and a random camera transform,
  all driven by a single seeded `random.Random` instance for full
  reproducibility across before/after comparisons. Profiles each scene,
  collects per-function `cumtime`/`tottime`/`ncalls`, and produces both a
  tabular summary and a thumbnail grid of the actual rendered images
  (loaded via Pillow, displayed via matplotlib) — the "visualize the
  result" step applied to a *batch* of renders rather than one at a time.
- **A synthetic mesh generator** (`generate_uv_sphere`): produces a
  triangle-mesh sphere entirely in memory (no OBJ file parsing needed),
  parametrized by triangle count, specifically to have a *triangle-heavy*
  scene to profile — necessary because the randomized sphere-based scenes
  never had enough objects to make `World.intersect`'s linear scan the
  dominant cost.

## 3. Bottleneck #1: Redundant `Matrix.inverse()` Calls Per Ray

### The problem

`Shape.transform` is fixed once a scene is built — it never changes
between rays during a render. But `Sphere._to_local_ray` (used by every
`intersect` call) and `Sphere.normal_at`/`Plane.normal_at`/
`Triangle.normal_at` were all calling `self.transform.inverse()` (and, for
normals, `.transpose()` too) **fresh, every single call** — meaning every
ray tested against a shape recomputed the same matrix inversion from
scratch.

### The evidence

A 200x100, 6-object scene profile showed:

```
matrix.py:inverse()        3,912,015 calls
_linalg.py:inv()           3,912,015 calls,  23.0s tottime
matrix.py:is_invertible()  3,912,015 calls,  35.4s cumtime
```

3.9 million inversions of matrices that never changed during the render.

### The fix

Cache the inverse (and inverse-transpose) on `Shape`, recomputed exactly
once, at the moment `transform` is *assigned* — via a property setter,
not on every read:

```python
class Shape(ABC):
    @property
    def transform(self) -> Matrix:
        return self._transform

    @transform.setter
    def transform(self, value: Matrix) -> None:
        self._transform = value
        self._inverse_transform = value.inverse()
        self._inverse_transpose_transform = self._inverse_transform.transpose()
```

Every call site that previously did `self.transform.inverse()` was updated
to use the cached `self._inverse_transform` (and
`self._inverse_transpose_transform` for normals) directly.

### Result

A direct call-count instrumentation (a temporary global counter inside
`Matrix.inverse()`) confirmed the fix's actual effect, rather than trusting
wall-clock time alone (which is noisy — subject to system load, thermal
throttling, etc.): **1,401,041 calls before full fix -> 20,006 calls
after**, on an identical seeded scene. The remaining ~20,000 calls were
traced to `Camera.ray_for_pixel`, called once per pixel — a legitimate,
much smaller cost, deliberately left uncached since its call volume was
never the bottleneck.

**Lesson**: call-count instrumentation is more reliable evidence than
wall-clock timing alone for confirming whether an optimization actually
changed program behavior, since wall-clock time on a shared or
variable-load machine can drop or rise for reasons unrelated to the code
change itself.

## 4. Bottleneck #2: `np.cross` Overhead on 3-Element Vectors

### The problem

`Vector.cross()` delegated to `np.cross()`, NumPy's general N-dimensional
cross product function. That generality — supporting broadcasting across
arbitrary-shaped batches of vectors along arbitrary axes — comes with
real, mostly-fixed per-call overhead (`np.moveaxis`,
`normalize_axis_tuple`) that has nothing to do with the actual arithmetic
of a single 3D cross product.

### The evidence

Profiling a 180-triangle mesh scene:

```
Vector.cross()            7,012,824 calls,  267.1s cumulative
np.cross() itself         7,012,824 calls,  244.6s cumulative (60%+ of total runtime)
  np.moveaxis             21,038,472 calls, 137.1s cumulative
  normalize_axis_tuple    42,076,944 calls,  71.3s cumulative
```

### The fix

Replace the `np.cross` call with the direct, hand-derived 3D cross-product
formula from Phase 1:

```python
def cross(self, other: "Vector") -> "Vector":
    return Vector(
        self.y * other.z - self.z * other.y,
        self.z * other.x - self.x * other.z,
        self.x * other.y - self.y * other.x,
    )
```

### Result

Same 180-triangle scene: **405.6s -> 152.5s** (a 62% reduction), with
`np.cross`/`moveaxis`/`normalize_axis_tuple` disappearing from the profile
entirely.

**Lesson, worth stating plainly**: NumPy's performance advantage comes
from vectorizing operations over *large arrays*, amortizing its own
overhead across many elements processed at once. Called millions of times
on individual 3-element vectors, that same generality becomes the
dominant cost rather than a benefit. "Use NumPy for numerical operations"
(this project's own original design principle) is not the same claim as
"NumPy is always faster" — the two can directly conflict at high call
volumes on tiny payloads, and only measurement reveals which applies.

## 5. Bottleneck #3: NumPy-Backed `Vector`/`Point` Constructors

### The problem

Even after fixing `cross`, every `Vector`/`Point` arithmetic operation
(`__add__`, `__sub__`, `dot`, construction itself) allocated a fresh
`np.array` internally — NumPy array construction carries real fixed
overhead (dtype checking, buffer allocation) that dwarfs the cost of
adding three floats, at the call volumes a ray tracer produces.

### The evidence

Same 180-triangle scene, post-`cross`-fix:

```
numpy.array (builtin)      40,262,511 calls,  18.0s tottime
Vector.__init__            19,863,892 calls,  14.8s cumulative
Matrix.apply_to_point       6,700,800 calls,  29.7s cumulative
Matrix.apply_to_vector      6,677,600 calls,  27.8s cumulative
```

### The fix — a genuine architectural reversal from Phase 1

`Vector`, `Point`, and `Color` were changed from NumPy-array-backed
storage to **plain Python `float` fields** in `__slots__`, with all
arithmetic implemented as direct Python float operations rather than
delegating to NumPy:

```python
class Vector:
    __slots__ = ("_x", "_y", "_z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    def dot(self, other: "Vector") -> float:
        return self._x * other._x + self._y * other._y + self._z * other._z
    # ... all other operations similarly reimplemented without NumPy
```

`Matrix` **remains** NumPy-backed — its genuinely `O(n^3)`-ish operations
(`@`, `.inverse()`, `.determinant()`) benefit from NumPy's optimized
linear algebra, since that work is substantial enough per call to amortize
NumPy's overhead. The dividing line established here: NumPy earns its
keep on operations with real computational weight per call; it costs more
than it saves on tiny, extremely-high-call-volume operations.

### Result

Same scene: **152.5s -> 89.8s** (a further 41% reduction). `numpy.array`
calls dropped out of the profile's top functions entirely.

## 6. Bottleneck #4: `Matrix.apply_to_point`/`apply_to_vector` NumPy Indexing Overhead

### The problem

Even with `Vector`/`Point` no longer NumPy-backed, `apply_to_point`/
`apply_to_vector` still indexed into `Matrix`'s NumPy-array storage
element-by-element (`self._data[0, 0]`, etc.) — 12+ individual NumPy
scalar-indexing operations per call, each carrying its own small fixed
overhead.

### The evidence

Scaling to a 3120-triangle mesh (confirming this scales with the problem
size, not just a one-off cost):

```
Matrix.apply_to_point   114,892,800 calls,  333.98s tottime (22% of total runtime)
Matrix.apply_to_vector  114,869,600 calls,  276.52s tottime (18% of total runtime)
```

Combined, ~40% of total render time.

### The fix

Cache the matrix's 16 values as a plain Python tuple of floats
(`self._flat`), computed once whenever `_data` is set (constructor and
`_from_ndarray`), and use that tuple — not the NumPy array — for the
per-point/per-vector hot-path multiplication:

```python
def apply_to_point(self, point: Point) -> Point:
    m = self._flat
    x, y, z = point.x, point.y, point.z
    return Point(
        m[0]*x + m[1]*y + m[2]*z + m[3],
        m[4]*x + m[5]*y + m[6]*z + m[7],
        m[8]*x + m[9]*y + m[10]*z + m[11],
    )
```

Same caching principle as Bottleneck #1: pay a conversion cost once at
construction, not on every one of millions of subsequent uses.

## 7. Confirming Linear Scaling — Evidence the BVH Is Actually Justified

With the above fixes in place, triangle-mesh scenes at two different
triangle counts were compared directly:

```
180 triangles:   6,624,000 Triangle.intersect calls,     89.8s total
3,120 triangles: 114,816,000 Triangle.intersect calls, 1503.1s total

Triangle count ratio:  17.33x
Call count ratio:      17.33x  (exact match)
Time ratio:            ~16.7x  (tracks closely)
```

This is the actual justification for building a bounding volume
hierarchy: cost scales **linearly** with triangle count, because every
ray tests every triangle unconditionally, with no remaining fixed
overhead distorting the measurement. Unlike Bottlenecks #1-4, there is no
"stop doing wasted work" fix available here — the work is genuinely
necessary for genuinely-tested triangles. The only way to reduce it is to
avoid testing triangles a ray provably cannot hit, which is exactly what
a bounding-volume hierarchy is for.

## 8. Methodological Lesson (the actual point of this phase)

Four real, substantial bottlenecks were found and fixed in this codebase,
and **none of them were "too many objects tested per ray"** — the
assumption that would have motivated jumping straight to a BVH. They were,
in order: redundant recomputation of an invariant (cached matrix
inverses), a general-purpose NumPy function's overhead dominating a
trivial fixed-size computation (`np.cross`), NumPy array allocation
overhead dominating trivial arithmetic (`Vector`/`Point`/`Color`
construction), and the same NumPy-indexing overhead one level deeper
(`Matrix.apply_to_*`). Only *after* all four were fixed did the profile
reveal genuinely linear, unavoidable-without-a-different-algorithm scaling
with triangle count — the actual, evidence-backed case for a BVH.

This is the concrete lesson behind the project brief's instruction to
profile before optimizing: intuition about where a ray tracer's time goes
(`"it must be all those intersection tests"`) was directionally right but
substantively wrong about *why* — the real costs were architectural
choices (NumPy-backing tiny fixed-size objects) that had nothing to do
with the ray-tracing algorithm itself, and would never have been found by
reasoning about complexity alone.

## 9. What This Will Be Used For Later

- Part 2 of this phase (bounding boxes / BVH) builds directly on the
  confirmed-linear-scaling evidence gathered here.
- The `generate_uv_sphere` synthetic mesh generator remains a reusable
  tool for stress-testing and benchmarking any future performance work
  (Phase 15's path tracing will multiply ray counts substantially further,
  and the same profiling methodology — measure, don't assume — applies
  directly).
- The NumPy-vs-plain-Python tradeoff surfaced here (amortized vectorized
  cost vs. per-call fixed overhead) is a lens worth reapplying to any
  future hot path before reflexively reaching for NumPy.