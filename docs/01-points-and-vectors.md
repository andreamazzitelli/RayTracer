# Phase 1 — Points and Vectors

## 1. Mathematical Theory

### Cartesian coordinates

A point in 3D space is located by three signed distances along three mutually
perpendicular axes: `(x, y, z)`. Everything in this ray tracer is expressed in
this coordinate system unless explicitly transformed into another one (camera
space, object/local space — see Phase 7).

### Points vs. Vectors

Both are triples of real numbers, but they answer different questions:

| Concept | Answers | Has a fixed location? |
|---|---|---|
| **Point** | "Where?" | Yes — an absolute position |
| **Vector** | "Which way, how far?" | No — a pure displacement |

This distinction is not cosmetic. It constrains which operations are
geometrically meaningful:

```
Point  - Point   -> Vector   (displacement between two locations)
Point  + Vector  -> Point    (move a location by a displacement)
Point  - Vector  -> Point
Vector + Vector  -> Vector
Vector - Vector  -> Vector
Point  + Point   -> undefined (what would "adding two locations" mean?)
```

Encoding this in the type system (two distinct classes, not one generic
`Tuple3`) turns a whole category of bugs — accidentally adding two positions,
or subtracting a location from a direction — into type errors instead of
silent, wrong renders.

### Homogeneous coordinates preview

The reason the Point/Vector split is *mathematically* justified, not just a
software-engineering preference, becomes concrete in Phase 7. Every point and
vector can be extended with a fourth coordinate `w`:

- Point: `w = 1`
- Vector: `w = 0`

Under this convention, `Point + Point` would produce `w = 2` — a signal that
something invalid happened — while `Point + Vector` produces `w = 1`, a valid
point. The algebra above literally falls out of this single extra coordinate
once 4×4 transformation matrices are introduced. Point/Vector are not an
arbitrary API choice; they are the 3D projection of a deeper 4D structure.

## 2. Geometric Intuition

- **Vector addition**: tip-to-tail placement, or equivalently the diagonal of
  the parallelogram spanned by the two vectors.
- **Scalar multiplication**: stretches, shrinks, or reverses (negative scalar)
  a vector along its own line, without changing its direction (except for
  sign flip).
- **Magnitude**: the length of the vector — a 3D extension of the Pythagorean
  theorem.
- **Normalization**: rescaling a vector to unit length (magnitude 1) while
  preserving direction. Ray directions, surface normals, and light directions
  all need to be unit vectors for later shading math (Phase 5) to be
  physically correct — the diffuse lighting term relies on a dot product
  being interpretable as a cosine, which only holds for unit vectors.
- **Dot product**: a measure of directional alignment between two vectors,
  scaled by their magnitudes.
- **Cross product**: produces a vector perpendicular to both inputs, with
  magnitude equal to the area of the parallelogram they span, and direction
  given by the right-hand rule.

## 3. Equations and Derivations

### Magnitude

```
|v| = sqrt(x^2 + y^2 + z^2)
```

Direct 3D extension of the Pythagorean theorem: first find the length of the
projection onto the XY-plane (`sqrt(x^2 + y^2)`), then treat that length and
`z` as the two legs of a right triangle whose hypotenuse is `|v|`.

### Normalization

```
v_hat = v / |v|
```

Undefined for the zero vector (division by zero) — see "Common Mistakes"
below for how this project handles that edge case.

### Dot product — algebraic and geometric forms

Algebraic:

```
a . b = a_x*b_x + a_y*b_y + a_z*b_z
```

Geometric identity:

```
a . b = |a| |b| cos(theta)
```

**Derivation of the equivalence** (law of cosines approach): consider the
triangle formed by vectors `a`, `b`, and `a - b`. The law of cosines gives:

```
|a - b|^2 = |a|^2 + |b|^2 - 2|a||b|cos(theta)
```

Expand the left side using the algebraic dot product identity
`|v|^2 = v . v`:

```
|a - b|^2 = (a - b).(a - b) = a.a - 2(a.b) + b.b = |a|^2 - 2(a.b) + |b|^2
```

Set the two expansions equal:

```
|a|^2 - 2(a.b) + |b|^2 = |a|^2 + |b|^2 - 2|a||b|cos(theta)
```

Cancel `|a|^2 + |b|^2` from both sides:

```
-2(a.b) = -2|a||b|cos(theta)
a.b = |a||b|cos(theta)
```

This identity is the workhorse of the entire renderer: whenever the code
needs "the cosine of the angle between two directions" (surface normal vs.
light direction for diffuse shading, reflection vector vs. eye vector for
specular shading, ray direction vs. surface normal for various geometric
tests), it is computed as a dot product between two **unit** vectors, since
`cos(theta) = a.b` exactly when `|a| = |b| = 1`.

### Cross product

```
a x b = (a_y*b_z - a_z*b_y,  a_z*b_x - a_x*b_z,  a_x*b_y - a_y*b_x)
```

Key properties used later:
- `(a x b) . a = 0` and `(a x b) . b = 0` — perpendicularity to both inputs.
- `a x b = -(b x a)` — anti-commutative.
- `|a x b| = |a||b|sin(theta)` — area of the spanned parallelogram.

This project doesn't need the cross product heavily until camera basis
vectors are built in Phase 8.

## 4. Python / NumPy Representation

`Point` and `Vector` are distinct classes, each internally backed by a
`np.ndarray` of shape `(3,)` and `dtype=np.float64`, stored in a single
`__slots__` field (`_data`). NumPy is used purely as **storage and low-level
arithmetic** (`np.dot`, `np.cross`) — never exposed directly to calling code.
All public operations go through named methods (`.dot()`, `.cross()`,
`.magnitude()`, `.normalize()`) or operator overloads (`+`, `-`, `*`, `/`,
unary `-`), so the rest of the codebase never manipulates raw arrays.

```python
class Vector:
    __slots__ = ("_data",)

    def __init__(self, x: float, y: float, z: float) -> None:
        self._data = np.array([x, y, z], dtype=np.float64)
```

Pinning `dtype=np.float64` explicitly in the constructor avoids a subtle
trap: `np.array([1, 2, 3])` without a dtype infers `int64`, which behaves
correctly under addition/subtraction but is a latent source of bugs (e.g.
unexpected integer division) that's easy to miss until it isn't.

## 5. API / Design Decisions

- **`x`, `y`, `z` exposed as read-only `@property`**, not raw attributes,
  keeping `_data` the single source of truth and preventing partial,
  inconsistent mutation of a vector's components.
- **`__eq__` uses tolerance-based comparison**, not `==`, via
  `math.isclose(..., abs_tol=1e-9)`. An explicit `abs_tol` is required —
  `math.isclose`'s default `abs_tol=0.0` makes it a *purely relative*
  tolerance, which fails badly near zero (see Common Mistakes).
- **`__eq__` type-checks its argument** (`isinstance(other, Vector)`) and
  returns `NotImplemented` for non-`Vector` comparisons, rather than
  crashing on `vector == None` or similar. This follows Python's documented
  `__eq__` protocol (`other: object`, not `other: Vector`).
- **Operator overloading is used deliberately** where it mirrors real vector
  algebra (`+`, `-`, unary `-`, scalar `*`/`/`) — not as a blanket "operator
  overload everything" pattern.
- **`from_np_array` static constructor**: used internally (e.g. by
  `cross()`, which computes via `np.cross` and needs to re-wrap the raw
  result as a `Vector`) with a defensive shape check (`raise ValueError` if
  not shape `(3,)`).

## 6. Testing Strategy

- Exact hand-computed examples for `+`, `-`, unary `-`, scalar `*`/`/`.
- `magnitude()` on an axis-aligned vector (trivial, e.g. `(1,0,0) -> 1`) and
  a hand-checkable non-trivial one (`(1,2,2) -> 3`, since `1+4+4=9`).
- **Property tests**, preferred over single hard-coded examples where
  possible:
  - `v.normalize().magnitude() ~= 1` for any nonzero `v`.
  - `dot(a, b) == dot(b, a)` (commutativity).
  - `cross(a, b).dot(a) ~= 0` and `cross(a, b).dot(b) ~= 0`
    (perpendicularity — tests the *geometric meaning*, not just the formula).
  - `cross(a, b) == -cross(b, a)` (anti-commutativity).
- Equality against a non-`Vector` (`None`, an int, a string) must not crash.
- `from_np_array` with a wrong-shape array must raise `ValueError`.
- Normalizing the zero vector: behavior must be explicitly decided and
  tested (this project returns the zero vector rather than raising or
  propagating `inf`/`nan` — see below).

## 7. Common Mistakes Encountered (and Fixed)

1. **Circular import.** `vector.py` imported `Point` from `point.py` purely
   for use in an (unused) type hint, while `point.py` imported `Vector` for
   real (`isinstance` checks at runtime). Because both modules imported each
   other at load time, this created a circular import that crashed at
   startup. Fix: only import a sibling type at module level if it is
   actually used at *runtime*; a bare type-hint dependency, combined with
   `from __future__ import annotations`, does not need a real import at all.

2. **`math.isclose` default tolerance fails near zero.** `math.isclose(a, b)`
   defaults to `rel_tol=1e-9, abs_tol=0.0` — a purely *relative* tolerance.
   `math.isclose(0.0, 1e-16)` is `False` under these defaults, because the
   allowed slack scales with the magnitude of the numbers being compared,
   shrinking to nothing near zero. This directly breaks exactly the kind of
   property test this project relies on — e.g. `cross(a,b).dot(a) ~= 0`,
   where the "expected" value is mathematically zero but the *computed*
   value is something like `1e-16` due to floating-point roundoff. Fix: pass
   an explicit `abs_tol` (this project uses `1e-9`) everywhere floating
   point equality is checked.

3. **`__eq__` typed as `other: Vector` instead of `other: object`.** Violates
   the `object.__eq__` contract and causes real runtime crashes:
   `Vector(1,2,3) == None` raised `AttributeError` instead of returning
   `False`, because the method assumed `other` always had `.x`/`.y`/`.z`.
   Fixed with an `isinstance` guard returning `NotImplemented` for
   non-matching types.

4. **Missing explicit `dtype` in the NumPy constructor** risked silent
   `int64` arrays for integer-valued constructor calls like `Vector(1,2,3)`.
   Fixed by always passing `dtype=np.float64`.

5. **Design decision, not a bug: normalizing the zero vector.** `magnitude()`
   of the zero vector is `0`, and `self / 0` would raise
   `ZeroDivisionError`/produce `inf`. This project chose to special-case
   `normalize()` on a near-zero magnitude (`math.isclose(magnitude, 0,
   abs_tol=1e-9)`) and return the zero vector rather than raising. This
   avoids crashes in downstream code (e.g. degenerate reflection or shading
   calculations) at the cost of silently producing a physically meaningless
   "direction." Documented here explicitly so the tradeoff is not forgotten.

## 8. What This Will Be Used For Later

- `Ray` (Phase 2) is defined entirely in terms of `Point` (origin) and
  `Vector` (direction).
- `Sphere.normal_at` (Phase 4) and later shape normals return `Vector`s.
- The dot product identity `a.b = |a||b|cos(theta)` is the mathematical
  foundation of Lambertian diffuse shading and specular highlights
  (Phase 5).
- The cross product becomes necessary for constructing camera basis vectors
  (Phase 8).
- The Point/Vector distinction, and specifically the `w=1`/`w=0` convention,
  is the exact mechanism that makes 4×4 homogeneous transformation matrices
  (Phase 7) behave correctly — translating points but leaving vectors
  unaffected.
