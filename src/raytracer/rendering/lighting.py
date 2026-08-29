from __future__ import annotations

from raytracer.geometry.point import Point
from raytracer.geometry.vector import Vector
from raytracer.image.color import Color
from raytracer.rendering.light import PointLight
from raytracer.rendering.material import Material


def lighting(
    material: Material,
    light: PointLight,
    point: Point,
    eye_vector: Vector,
    normal_vector: Vector,
    in_shadow: bool = False,
) -> Color:
    """Compute the Phong-shaded color at `point`: ambient + diffuse + specular.

    eye_vector: direction from `point` toward the viewer (V).
    normal_vector: surface normal at `point` (N).
    Light direction (L) is derived internally from light.position and point.
    in_shadow: if True, only ambient light contributes (diffuse and
    specular are suppressed, since the light can't reach `point`).
    """
    effective_color = material.color.hadamard(light.intensity)
    light_vector = (light.position - point).normalize()

    ambient = effective_color * material.ambient

    black = Color(0, 0, 0)
    diffuse = black
    specular = black

    if not in_shadow:
        light_dot_normal = light_vector.dot(normal_vector)

        if light_dot_normal >= 0:
            diffuse = effective_color * material.diffuse * light_dot_normal

            reflect_vector = (-light_vector).reflect(normal_vector)
            reflect_dot_eye = reflect_vector.dot(eye_vector)

            if reflect_dot_eye > 0:
                factor = reflect_dot_eye**material.shininess
                specular = light.intensity * material.specular * factor

    return ambient + diffuse + specular