# Phase 5 — Normals and Shading (the Phong Reflection Model)

## 1. Mathematical Theory

This phase is where the renderer starts producing something that looks like
a lit 3D object instead of a flat silhouette. The **Phong reflection
model** approximates how light interacts with a surface as the sum of three
independent components:

```
final_color = ambient + diffuse + specular
```

At a shaded point `P`, three unit vectors matter:

- `N` — the surface normal at `P` (from `Shape.normal_at`).
- `L` — the direction **toward the light**: `(light.position -
  P).normalize()`.
- `V` — the direction **toward the viewer**: `(-ray.direction).normalize()`
  (the ray traveled *in* `ray.direction` to reach `P`; "back toward the
  origin" is its negation).

## 2. Geometric Intuition

### Ambient

A constant term, independent of any angle — a crude stand-in for indirect
("bounced") light, which this renderer does not actually simulate until
Phase 15 (path tracing). Purely a modeling convenience: `material.color *
material.ambient`.

### Diffuse (Lambertian reflectance)

Models a **matte** surface — one that scatters incoming light roughly
equally in all directions, so its perceived brightness does not depend on
where the viewer stands, only on the angle between the surface and the
light.

**Intuition**: imagine a fixed-width beam of light hitting a surface. If the
surface directly faces the light (normal parallel to light direction), the
beam's energy lands on a small patch — concentrated, bright. If the surface
is tilted away from the light, the *same* beam spreads its energy over a
larger patch of surface, diluting it — each point looks dimmer. The
brightness is therefore proportional to how "head-on" the surface faces the
light, i.e. `cos(theta)` between `N` and `L`.

### Specular

Models the bright highlight on **glossy** (not matte) surfaces — genuinely
viewer-dependent, unlike diffuse. The idea: light reflects off the surface
like a mirror around the normal; the highlight is brightest exactly where
the viewer happens to be looking along that reflected direction, and falls
off sharply as the viewer moves away from it.

## 3. Equations and Derivations

### Diffuse — from the dot product identity (Phase 1)

Recall `N . L = |N||L|cos(theta)`. Since both `N` and `L` are unit vectors,
this collapses to:

```
N . L = cos(theta)
```

which is exactly the "how head-on does the surface face the light"
brightness factor described above. The diffuse term is:

```
diffuse = material.color * material.diffuse * (N . L)
```

**Edge case**: when the surface faces away from the light entirely (`theta
> 90 degrees`), `N . L` is negative. A negative diffuse contribution is
physically nonsensical — a light source cannot *subtract* brightness from a
surface it isn't illuminating — so the implementation clamps this
contribution to zero rather than allowing a negative term into the sum.

### Specular — deriving the reflection vector

Given the incoming light direction (surface-to-light reversed, i.e. `-L`,
pointing *from* the light *toward* the surface) and the normal `N`, the
reflected direction `R` is derived by decomposition:

1. Project the incoming vector onto `N`: since `N` is unit length, this
   projection is `(incoming . N) * N` — the component of `incoming` that
   lies *along* the normal.
2. The remaining component, `incoming - (incoming.N)*N`, lies entirely
   *perpendicular* to `N` (tangent to the surface).
3. Mirror reflection **flips the parallel component** and **keeps the
   perpendicular component unchanged**. The reflected vector is therefore:

```
incoming - 2 * (incoming . N) * N
```

i.e. subtract *twice* the parallel component — once to cancel it, once more
to flip it to the opposite side.

**Sanity check** (verified by hand): reflecting `Vector(1, -1, 0)` (heading
down-and-right) off `Vector(0, 1, 0)` (a horizontal surface, normal
straight up) should flip the vertical component and leave the horizontal
component untouched, giving `Vector(1, 1, 0)`. This concrete, axis-aligned
case is a useful permanent regression test, since it is easy to verify by
eye and catches a specific, easy-to-make sign error (see Common Mistakes).

The specular term compares the reflection vector `R` (of the *incoming*
light direction, i.e. `(-L).reflect(N)`) against the eye vector `V`, using
the same `cos(theta)` dot-product trick, raised to a `shininess` exponent
that controls how tight the highlight is:

```
specular = light.intensity * material.specular * (R . V)^shininess
```

Higher `shininess` -> smaller, sharper highlight (polished metal); lower ->
broader, softer highlight (satin/plastic). As with diffuse, a negative `R .
V` (reflection pointing away from the viewer) is clamped to zero rather than
contributing a negative or nonsensical (fractional power of a negative
number) term.

## 4. Python / NumPy Representation

```python
def lighting(material, light, point, eye_vector, normal_vector) -> Color:
    effective_color = material.color.hadamard(light.intensity)
    light_vector = (light.position - point).normalize()

    ambient = effective_color * material.ambient

    black = Color(0, 0, 0)
    diffuse = black
    specular = black

    light_dot_normal = light_vector.dot(normal_vector)
    if light_dot_normal >= 0:
        diffuse = effective_color * material.diffuse * light_dot_normal

        reflect_vector = (-light_vector).reflect(normal_vector)
        reflect_dot_eye = reflect_vector.dot(eye_vector)
        if reflect_dot_eye > 0:
            factor = reflect_dot_eye ** material.shininess
            specular = light.intensity * material.specular * factor

    return ambient + diffuse + specular
```

Notable details:

- `effective_color = material.color.hadamard(light.intensity)` — the
  surface color and the light's color combine via the **Hadamard**
  (component-wise) product established in Phase 3, not scalar
  multiplication: a red surface under a pure blue light should render
  black, which only the channel-by-channel product produces correctly.
- `reflect_vector = (-light_vector).reflect(normal_vector)` — note the
  negation. `light_vector` (`L`) points *from the surface toward the
  light*; the reflection formula conceptually reflects the *incoming* ray
  (light-to-surface, i.e. `-L`), not the outgoing one. Getting this sign
  backward is easy and was an actual bug encountered during development
  (see Common Mistakes).
- The two clamp checks (`light_dot_normal >= 0` and, nested inside it,
  `reflect_dot_eye > 0`) are independent and intentionally nested rather
  than combined: a surface can face the light (diffuse contributes) while
  its reflection still points away from the eye (specular should still be
  zero even though diffuse is not).

## 5. API / Design Decisions

Two small supporting types:

```python
class Material:
    __slots__ = ("color", "ambient", "diffuse", "specular", "shininess")
    # defaults: ambient=0.1, diffuse=0.9, specular=0.9, shininess=200.0

class PointLight:
    __slots__ = ("position", "intensity")
    # a light source with no physical size, emitting uniformly in all directions
```

`reflect` was implemented as a method on `Vector` (`vector.reflect(normal)
-> Vector`), consistent with the existing placement of `dot`/`cross` as
`Vector` methods rather than free functions — kept consistent with the
established style for operations that take one `Vector` and return another.

`lighting()` itself is a free function (not a method on `Material` or
`PointLight`), since it genuinely needs inputs from several independent
sources (material, light, geometric point, two vectors) — it doesn't
naturally "belong" to any single one of them.

## 6. Testing Strategy

- Eye directly between the light and the surface, both looking straight on
  — work out by hand whether specular should be at maximum in this
  configuration (it depends on where the *reflection* vector actually
  points relative to the eye — do not assume).
- Eye offset 45 degrees from the normal.
- Light positioned directly behind the eye.
- Light behind the **surface** (`N . L < 0`) — diffuse and specular both
  zero, only ambient remains.
- Eye positioned exactly in the path of the reflection vector — specular at
  its maximum.
- At least one specular case should be **hand-computed in full** before
  trusting the code's output, specifically because specular is the term
  most likely to hide a sign error (as it did during development).

## 7. Common Mistakes Encountered (and Fixed)

1. **`reflect` scaled the wrong vector.** An initial implementation was:

   ```python
   def reflect(self, normal: Vector) -> Vector:
       return self - (2 * self.dot(normal) * self)
   ```

   Compare against the derived formula, `R = L - 2(L.N)N`: the scalar `2 *
   (L.N)` must scale the **normal** `N` (the axis being reflected across),
   not `self` (`L`) again. The buggy version effectively subtracted a
   scaled copy of the *input* vector from itself rather than reflecting it
   across the normal — a plausible-looking but geometrically wrong
   formula, since it doesn't correspond to "flip the component along `N`."

   This was caught by hand-verifying a concrete, easy case: `Vector(1, -1,
   0).reflect(Vector(0, 1, 0))` should yield `Vector(1, 1, 0)` (flip the
   vertical component off a horizontal surface); the buggy version did not
   produce this. Fixed to:

   ```python
   def reflect(self, normal: Vector) -> Vector:
       return self - (normal * (2 * self.dot(normal)))
   ```

2. **A typo in `Material.__repr__`**: the `Specular:` field printed
   `self.diffuse` a second time instead of `self.specular`. Purely
   cosmetic (does not affect rendering), but worth fixing, since an
   incorrect `__repr__` actively misleads debugging sessions later —
   exactly the moment a wrong debug print is most costly.

## 8. What This Will Be Used For Later

- `lighting()` is called once per visible intersection, per light, by the
  renderer (Phase 9) to determine a pixel's final color.
- `Vector.reflect` is reused directly and unchanged in Phase 12
  (reflection), this time reflecting the *ray's* direction rather than the
  light's, to spawn secondary reflected rays.
- The ambient/diffuse/specular clamp-to-zero pattern (never allow negative
  light contributions) recurs conceptually in Phase 6 (shadows: a shadowed
  point receives *only* ambient light, with diffuse and specular both
  suppressed entirely) and Phase 13 (refraction, where analogous physical
  validity checks — like total internal reflection — gate whether a term
  contributes at all).
