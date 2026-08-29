from __future__ import annotations

from typing import TYPE_CHECKING

from raytracer.geometry.point import Point
from raytracer.geometry.ray import Ray
from raytracer.geometry.vector import Vector

if TYPE_CHECKING:
    from raytracer.geometry.intersection import Intersection
    from raytracer.shapes.shape import Shape

SHADOW_EPSILON = 1e-5
UNDER_EPSILON = 1e-5


class Computations:
    """Precomputed geometric state for a single intersection, shared by
    shade_hit, reflected_color, and refracted_color — computed once so
    those consumers never need to re-derive point/normal/eye/n1/n2
    themselves."""

    __slots__ = (
        "t",
        "object",
        "point",
        "eye_vector",
        "normal_vector",
        "inside",
        "over_point",
        "under_point",
        "reflect_vector",
        "n1",
        "n2",
    )

    def __init__(
        self,
        t: float,
        obj: "Shape",
        point: Point,
        eye_vector: Vector,
        normal_vector: Vector,
        inside: bool,
        over_point: Point,
        under_point: Point,
        reflect_vector: Vector,
        n1: float,
        n2: float,
    ) -> None:
        self.t = t
        self.object = obj
        self.point = point
        self.eye_vector = eye_vector
        self.normal_vector = normal_vector
        self.inside = inside
        self.over_point = over_point
        self.under_point = under_point
        self.reflect_vector = reflect_vector
        self.n1 = n1
        self.n2 = n2


def prepare_computations(
    intersection: "Intersection",
    ray: Ray,
    all_intersections: list["Intersection"] | None = None,
) -> Computations:
    """Derive a Computations from a single Intersection and the Ray that
    produced it.

    `all_intersections` should be every intersection along `ray` (sorted
    ascending by t), used to correctly compute n1/n2 for refraction through
    nested/overlapping transparent objects. If omitted, defaults to a
    single-element list containing just `intersection` — correct for
    non-refractive scenes, but n1/n2 will not account for containment by
    other transparent objects.
    """
    if all_intersections is None:
        all_intersections = [intersection]

    point = ray.position_at(intersection.t)
    eye_vector = -ray.direction
    normal_vector = intersection.object.normal_at(point)

    if normal_vector.dot(eye_vector) < 0:
        inside = True
        normal_vector = -normal_vector
    else:
        inside = False

    over_point = point + normal_vector * SHADOW_EPSILON
    under_point = point - normal_vector * UNDER_EPSILON
    reflect_vector = ray.direction.reflect(normal_vector)

    n1, n2 = _compute_refractive_indices(intersection, all_intersections)

    return Computations(
        t=intersection.t,
        obj=intersection.object,
        point=point,
        eye_vector=eye_vector,
        normal_vector=normal_vector,
        inside=inside,
        over_point=over_point,
        under_point=under_point,
        reflect_vector=reflect_vector,
        n1=n1,
        n2=n2,
    )


def _compute_refractive_indices(
    hit_intersection: "Intersection",
    all_intersections: list["Intersection"],
) -> tuple[float, float]:
    """Walk every intersection along the ray, maintaining a stack of
    currently-entered ("containing") objects, to determine n1 (the
    refractive index of the material being left) and n2 (the material
    being entered) at `hit_intersection`. Handles nested/overlapping
    transparent shapes correctly, unlike naively using the hit object's
    own refractive_index for both."""
    containers: list["Shape"] = []
    n1 = 1.0
    n2 = 1.0

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