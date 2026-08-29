# Phase 7 — Matrices and Transformations

## 1. Mathematical Theory

Up to this point, changing an object's position meant literally mutating its
geometry (e.g. `Sphere.center`). That approach doesn't generalize: rotating
a sphere is meaningless (it's symmetric), but rotating a cube or triangle
is meaningful, and per-shape rotation logic would need to be duplicated
everywhere. The standard solution: represent transformations as **matrices**,
attach one to each object, and transform *rays* into the object's local
(untransformed) space before intersecting — so each shape's `intersect`
only ever needs to handle its simplest canonical form, and all positional/
rotational/scaling complexity lives in one matrix.

## 2. Why 4x4, not 3x3

A 3x3 matrix represents linear transformations (scaling, rotation) — always
fixing the origin in place. Translation (`p' = p + t`) is additive, not
expressible as `p' = Mp` for any 3x3 `M`. The fix is **homogeneous
coordinates**: extend every point/vector to 4 components with a `w`
coordinate — `Point` gets `w=1`, `Vector` gets `w=0` (established back in
Phase 1). A 4x4 translation matrix:

```
[1 0 0 tx]   [x]   [x + tx]
[0 1 0 ty] . [y] = [y + ty]
[0 0 1 tz]   [z]   [z + tz]
[0 0 0  1]   [w]   [w     ]
```

For a point (`w=1`), the translation column is picked up and added. For a
vector (`w=0`), that same column gets multiplied by zero and vanishes —
vectors are correctly unaffected by translation. This is the payoff of the
Point/Vector `w` distinction from Phase 1.

## 3. Equations and Derivations

### Scaling — diagonal matrix

```
[sx 0  0  0]
[0  sy 0  0]
[0  0  sz 0]
[0  0  0  1]
```

Each axis is scaled independently, with no cross-terms.

### Rotation about Z — derived from angle-addition identities

A point `(x, y)` at polar angle `phi` (so `x = r*cos(phi)`, `y = r*sin(phi)`)
rotated by `theta` moves to angle `phi + theta`:

```
x' = r*cos(phi + theta) = r*cos(phi)*cos(theta) - r*sin(phi)*sin(theta)
   = x*cos(theta) - y*sin(theta)

y' = r*sin(phi + theta) = r*sin(phi)*cos(theta) + r*cos(phi)*sin(theta)
   = x*sin(theta) + y*cos(theta)
```

giving the matrix:

```
[cos(t) -sin(t) 0 0]
[sin(t)  cos(t) 0 0]
[0       0      1 0]
[0       0      0 1]
```

Rotation about X and Y follow the identical derivation with axes permuted.
**Y is the case most likely to get a sign wrong** — the pattern does not
transfer directly by analogy from X/Z; deriving it explicitly (`x' =
x*cos(t) + z*sin(t)`, `z' = -x*sin(t) + z*cos(t)`) is worth doing rather than
guessing.

### Shearing

A mostly-identity matrix with off-diagonal terms making one coordinate
shift in proportion to another, e.g. parameter `xy` sets how much `x`
shifts per unit of `y`:

```
[1  xy xz 0]
[yx 1  yz 0]
[zx zy 1  0]
[0  0  0  1]
```

## 4. Determinant and Inverse

### Determinant — geometric meaning

The determinant measures how a transformation scales volume (3D) or area
(2D). Identity -> determinant 1. Uniform scale by `k` -> determinant `k^3`
(volume scaling). A negative determinant indicates a flip in orientation
(mirroring), on top of any scaling.

**2x2 base case**: `det = a*d - b*c` — derivable as the signed area of the
parallelogram spanned by the matrix's two row (or column) vectors.

**Cofactor expansion** (general n x n case):

```
det(M) = sum over j of M[0][j] * cofactor(M, 0, j)
cofactor(M, i, j) = (-1)^(i+j) * minor(M, i, j)
minor(M, i, j) = det(submatrix(M, i, j))     # delete row i, column j
```

Recursive: an n x n determinant needs n (n-1)x(n-1) determinants (cofactors
along a row), bottoming out at the 2x2 base case.

### Inverse

```
M_inverse[i][j] = cofactor(M, j, i) / det(M)
```

Note the **transposed indices** — the matrix of cofactors, transposed, is
called the **adjugate**; the inverse is the adjugate divided by the
determinant. A matrix is invertible if and only if `det(M) != 0` — a zero
determinant means the transformation collapses space into a lower
dimension, which is not reversible.

## 5. Python / NumPy Representation

`Matrix` is a general-purpose type backed by an `np.ndarray`, deliberately
kept independent of any "this is a rotation" domain knowledge:

```python
class Matrix:
    __slots__ = ("_data",)

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix._from_ndarray(self._data @ other._data)

    def determinant(self) -> float:
        return float(np.linalg.det(self._data))

    def inverse(self) -> Matrix:
        if not self.is_invertible():
            raise ValueError("Matrix is not invertible")
        return Matrix._from_ndarray(np.linalg.inv(self._data))
```

`np.linalg.det`/`np.linalg.inv` are used directly rather than the hand-rolled
cofactor-expansion recursion — a deliberate engineering tradeoff, documented
explicitly since it trades the "derive it by hand" learning goal for
concision. A hand-rolled `minor`/`cofactor` implementation was built first
as the reference/learning exercise and can be kept alongside as a
cross-check (`determinant_by_cofactor_expansion`) if desired.

Transformation constructors live in a separate module, `transform.py`, as
**free functions returning plain `Matrix` objects** — not a `Transformation`
subtype — since nothing downstream needs to know *how* a matrix was built,
which is exactly what makes composition (`translation(...) @ rotation_x(...)
@ scaling(...)`) trivial ordinary matrix multiplication.

## 6. API / Design Decisions

- **`Matrix._from_ndarray`**: an internal fast-path constructor bypassing
  the validated `list[list[float]]` constructor, used whenever a method
  already has a correctly-shaped NumPy result (matrix multiply, transpose,
  identity, every transform constructor). Safe only because every caller
  produces a guaranteed-rectangular array; any future method that could
  produce a ragged result must go through the validated constructor
  instead.
- **`apply_to_point` vs `apply_to_vector` as two separate methods**, not one
  polymorphic `apply` — mirrors the exact same overload problem hit with
  `Point.__sub__` back in Phase 1: the only thing distinguishing the two
  operations is the value of `w`, which isn't information either `Point`
  or `Vector` carries on its own.
- **Composition order matters and is not commutative.** `A @ B != B @ A`
  in general. Building a combined transform as `translation @ rotation @
  scaling` means, when applied to a point (`M @ p`), scaling happens first,
  then rotation, then translation — right-to-left application order. Getting
  this backwards is a common, easy-to-make bug once transforms are combined.

## 7. Testing Strategy

- Hand-computed matrix x matrix and matrix x point/vector products.
- **[property]** `identity(n) @ M == M`.
- **[property]** Non-commutativity: `A @ B != B @ A` for suitable `A`, `B`.
- **[property]** `M @ M.inverse() == identity(4)` for translation, scaling,
  and rotation matrices — each independently hand-verifiable (undo a
  translation by `(tx,ty,tz)` is obviously translate by `(-tx,-ty,-tz)`;
  undo a scale by `s` is obviously scale by `1/s`).
- `translation(...).apply_to_vector(v) == v` — vectors unaffected by
  translation (the test that actually proves the `w=0` mechanism works).
- Axis-aligned rotation test cases at `pi/2`, hand-computed via trig.
- **[property]** four successive `pi/2` rotations return to the original
  point (periodicity — catches sign errors that "sort of" work once but
  drift).
- Composition-order test: applying scale-then-rotate-then-translate as
  three sequential steps matches applying the single combined matrix.

## 8. Common Mistakes Encountered (and Fixed)

1. **`identity()` used `np.empty` instead of `np.zeros`.** `np.empty`
   leaves uninitialized memory in every cell; the loop only explicitly set
   the diagonal to 1, leaving off-diagonal entries as garbage rather than
   0. Fixed by starting from `np.zeros`.

2. **`shape` property indexed incorrectly and crashed on non-square
   matrices.** An initial draft computed `(len(self._data[0]),
   len(self._data[0][0]))` — the first term actually returns `num_cols`
   (mislabeled), and the second calls `len()` on a scalar (the top-left
   element), raising `TypeError`. Fixed by using `self._data.shape`
   directly, which NumPy arrays already expose correctly.

3. **`cofactor`, `determinant`, and `inverse` were drafted calling
   `np.linalg.det`/`np.linalg.inv` directly**, defeating the purpose of
   the hand-rolled recursive exercise this sub-phase was meant to be. Also,
   in an earlier draft, `cofactor` was implemented as `determinant *
   inverse.T` — not a meaningful operation, and backwards in terms of
   dependency direction (`determinant`/`inverse` should depend on
   `cofactor`, not the reverse).

4. **`is_invertible` returned the inverted boolean.**
   `np.isclose(determinant, 0)` is `True` exactly when the matrix is
   singular — the opposite of "invertible." This silently let `inverse()`
   proceed on non-invertible matrices (or reject valid ones).

5. **`inverse()`'s guard never fired**, independent of bug #4:
   `if not self.is_invertible:` referenced the bound *method object*
   itself (always truthy) rather than calling it
   (`self.is_invertible()`) — missing parentheses meant the guard was
   dead code regardless of its logic being correct or not.

6. **Unnecessary array-to-list-to-array round-trips.** Several methods
   (`__matmul__`, `transpose`, `identity`) produced a correct `np.ndarray`
   result, converted it to a Python list via `.tolist()`, and passed it
   through the list-validating constructor, which immediately converted it
   back to an `np.ndarray`. Fixed by routing internal-only construction
   through `_from_ndarray`, skipping the redundant round-trip — the
   `identity()` construction itself was also simplified from a manual
   nested loop to `np.eye(size)`.

## 9. What This Will Be Used For Later

- Every `Shape` subclass (`Sphere`, `Plane`, `Triangle`) carries a
  `transform: Matrix` attribute; `intersect`/`normal_at` transform
  incoming rays/points via `transform.inverse()` to work in the object's
  local space.
- Surface normals require the **inverse transpose** of the transform
  (not the transform itself) to remain correctly perpendicular under
  non-uniform scaling — introduced properly in Phase 9's `Sphere`
  refactor.
- The camera (Phase 8) is positioned and oriented via the same transform
  machinery — camera-space-to-world-space conversion uses
  `camera.transform.inverse()` exactly like an object's local-space
  conversion.
