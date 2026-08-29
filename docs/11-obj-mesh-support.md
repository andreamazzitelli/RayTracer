# Phase 11 — OBJ / Mesh Support

## 1. Purpose

This phase is primarily about parsing and data-structuring, not new
geometric math — it adds the ability to load real 3D models (Wavefront
`.obj` files) and convert them into collections of the `Triangle` shapes
built in Phase 10.

## 2. OBJ Format Essentials

A `.obj` file is plain text, line-oriented:

```
v 1.0 0.0 0.0        # vertex — a Point; indices are 1-based, not 0-based
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 2 3               # face — references vertices by 1-based index
```

Any line not starting with `v` or `f` (comments, normals `vn`, texture
coordinates `vt`, group markers `g`) is ignored by this minimal parser.

### Vertex indices are 1-based

This is a real, easy-to-miss detail of the OBJ spec — vertex `1` in a face
line refers to the *first* parsed vertex, not the second. Any parser storing
vertices in a plain 0-indexed list must subtract 1 at every face-parsing
site, or maintain a sentinel to avoid that arithmetic scattering through the
code (see Design Decisions).

### Fan triangulation

Faces with more than three vertices (polygons) must be split into
triangles. **Fan triangulation** always pairs the first vertex of the face
with each successive adjacent pair of the remaining vertices:

```
f 1 2 3 4 5   =>   triangles (1,2,3), (1,3,4), (1,4,5)
```

Geometrically, this fans out from a single shared vertex across the
polygon — correct for **convex** polygons; a concave polygon can produce
incorrect (self-overlapping or inverted) triangles under fan triangulation,
a known limitation not addressed by this minimal parser.

## 3. Python Representation

```python
def fan_triangulate(vertices: list[Point]) -> list[Triangle]:
    triangles = []
    for i in range(1, len(vertices) - 1):
        triangles.append(Triangle(vertices[0], vertices[i], vertices[i + 1]))
    return triangles


def parse_obj(source: str) -> ParseResult:
    result = ParseResult()
    result.vertices.append(Point(0, 0, 0))  # index 0 unused — keeps 1-based indexing direct

    for line in source.splitlines():
        tokens = line.split()
        if not tokens:
            result.ignored_lines += 1
            continue

        keyword, *args = tokens

        if keyword == "v":
            x, y, z = (float(v) for v in args[:3])
            result.vertices.append(Point(x, y, z))
        elif keyword == "f":
            face_indices = [int(idx) for idx in args]
            face_vertices = [result.vertices[i] for i in face_indices]
            result.triangles.extend(fan_triangulate(face_vertices))
        else:
            result.ignored_lines += 1

    return result
```

## 4. API / Design Decisions

- **Dummy `vertices[0]` sentinel.** Rather than subtracting 1 from every
  face index at every parsing site, a placeholder `Point(0,0,0)` occupies
  index 0, so a face's 1-based indices map directly onto the vertex list
  without arithmetic scattered through the code. A deliberate, documented
  tradeoff: an odd-looking unused sentinel value in exchange for
  eliminating a whole class of off-by-one bugs at every call site that
  reads a face index.
- **`ignored_lines` counter, not silent dropping.** Lines that aren't
  vertices or faces are still counted, not just skipped invisibly — this
  gives tests a way to positively assert that non-geometry lines were
  correctly recognized and skipped, rather than only inferring it from the
  absence of a crash.
- **`ParseResult` as a small dataclass** (`vertices`, `triangles`,
  `ignored_lines`) rather than returning a bare tuple — self-documenting
  field access (`result.triangles` vs. `result[1]`) at negligible cost.
- **Negative OBJ vertex indices are not handled.** The OBJ spec allows
  referencing vertices backward from the end of the current vertex list
  (e.g. `-1` meaning "the most recently defined vertex") — a real but less
  common part of the spec, explicitly out of scope for this minimal parser.
  Flagged here as a known, documented gap rather than a silent omission,
  in case a real-world `.obj` file using this feature is ever loaded and
  produces confusing results.

## 5. Testing Strategy

- Unrecognized lines are correctly counted via `ignored_lines`, not
  silently mishandled.
- Vertex records parse into the correct `Point` values at the correct
  (1-based) indices.
- Simple triangular faces (`f 1 2 3`) produce exactly one `Triangle` with
  vertices in the same order as the face line.
- Polygon faces (`f 1 2 3 4 5`) triangulate into the expected number of
  triangles (`n - 2` for an `n`-vertex face), each referencing the correct
  fan-triangulated vertex triple.

## 6. What This Will Be Used For Later

- Parsed `Triangle` collections are added directly to `World.objects` —
  no special handling is needed elsewhere in the renderer, since
  `Triangle` already integrates uniformly with `World.intersect`/
  `shade_hit` from Phase 9/10.
- Real-world meshes loaded this way (potentially thousands of triangles)
  are the primary motivating case for acceleration structures (Phase 14) —
  a linear `World.intersect` scan over every triangle becomes a genuine
  bottleneck exactly at the point meshes of nontrivial size are loaded.
- If per-vertex normals (`vn` lines, currently ignored) are added later for
  smooth (Phong-interpolated) shading across a mesh surface, that would
  extend this parser and `Triangle`'s normal handling — noted here as a
  natural, currently-unimplemented extension point.
