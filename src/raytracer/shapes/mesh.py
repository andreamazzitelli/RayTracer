from __future__ import annotations

from raytracer.geometry.bounding_box import BoundingBox
from raytracer.geometry.intersection import Intersection
from raytracer.geometry.ray import Ray
from raytracer.shapes.triangle import Triangle

DEFAULT_LEAF_SIZE = 4


class BVHNode:
    """A node in a bounding-volume hierarchy over triangles. Leaf nodes
    hold a small list of triangles directly; internal nodes hold two
    children, each with their own (tighter) bounding box."""

    __slots__ = ("bounding_box", "left", "right", "triangles")

    def __init__(
        self,
        bounding_box: BoundingBox,
        left: "BVHNode | None" = None,
        right: "BVHNode | None" = None,
        triangles: list[Triangle] | None = None,
    ) -> None:
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.triangles = triangles

    def is_leaf(self) -> bool:
        return self.triangles is not None


def _triangle_centroid(t: Triangle):
    return (
        (t.p1.x + t.p2.x + t.p3.x) / 3.0,
        (t.p1.y + t.p2.y + t.p3.y) / 3.0,
        (t.p1.z + t.p2.z + t.p3.z) / 3.0,
    )


def build_bvh(triangles: list[Triangle], leaf_size: int = DEFAULT_LEAF_SIZE) -> BVHNode:
    """Recursively partition `triangles` into a BVH. Split axis is
    chosen as whichever axis has the largest extent of the current
    node's bounding box (a simple, effective heuristic — not the most
    sophisticated split strategy, but a large improvement over no
    spatial structure at all). Triangles are sorted by centroid along
    that axis and split at the median, giving a reasonably balanced
    tree without needing a more elaborate cost model."""
    box = BoundingBox.empty()
    for t in triangles:
        box = box.merge(t.bounding_box)

    if len(triangles) <= leaf_size:
        return BVHNode(box, triangles=triangles)

    extents = (
        box.max_point.x - box.min_point.x,
        box.max_point.y - box.min_point.y,
        box.max_point.z - box.min_point.z,
    )
    axis = extents.index(max(extents))

    sorted_triangles = sorted(triangles, key=lambda t: _triangle_centroid(t)[axis])
    mid = len(sorted_triangles) // 2

    # Guard against a degenerate split (e.g. all centroids identical on
    # this axis) producing an empty half, which would recurse forever.
    if mid == 0 or mid == len(sorted_triangles):
        return BVHNode(box, triangles=triangles)

    left = build_bvh(sorted_triangles[:mid], leaf_size)
    right = build_bvh(sorted_triangles[mid:], leaf_size)
    return BVHNode(box, left=left, right=right)


def intersect_bvh(node: BVHNode, ray: Ray) -> list[Intersection]:
    """Traverse the BVH, pruning any subtree whose bounding box the ray
    misses entirely. Only leaf nodes actually test real triangle
    geometry."""
    if not node.bounding_box.intersects(ray):
        return []

    if node.is_leaf():
        results = []
        for triangle in node.triangles:
            results.extend(triangle.intersect(ray))
        return results

    return intersect_bvh(node.left, ray) + intersect_bvh(node.right, ray)


class Mesh:
    """A collection of triangles accelerated by a bounding-volume
    hierarchy. Exposes the same `intersect(ray)` interface World expects
    of any object, but is not itself a Shape — normal_at/material live on
    the individual Triangles, which is what Intersection.object points
    to after a hit, so shading works unchanged."""

    __slots__ = ("triangles", "root", "bounding_box")

    def __init__(self, triangles: list[Triangle], leaf_size: int = DEFAULT_LEAF_SIZE) -> None:
        self.triangles = triangles
        self.root = build_bvh(triangles, leaf_size)
        self.bounding_box = self.root.bounding_box

    def intersect(self, ray: Ray) -> list[Intersection]:
        return intersect_bvh(self.root, ray)