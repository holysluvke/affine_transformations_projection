#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отсечение нелицевых граней (back-face culling).

Модуль намеренно независим от Z-буфера. Он классифицирует грани модели по
направлению внешней нормали и направлению на камеру. Полученный список
лицевых граней можно передать в существующий рендерер, не меняя его алгоритм.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

from geometry import Polyhedron

Vec3 = tuple[float, float, float]
ToView = Callable[[Sequence[float]], Vec3]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec3) -> Vec3:
    length = _length(a)
    if length <= 1e-15:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def face_cross_normal(points: Sequence[Sequence[float]]) -> Vec3:
    """Вернуть вектор нормали как сумму векторных произведений векторов грани.

    Для каждой треугольной части берутся два вектора, лежащие в плоскости
    грани: (p[i] - p[0]) x (p[i+1] - p[0]). Сумма позволяет работать с
    многоугольниками и не вырождается, если первые три вершины коллинеарны.
    """
    if len(points) < 3:
        return (0.0, 0.0, 0.0)
    origin = tuple(points[0])
    normal = [0.0, 0.0, 0.0]
    for i in range(1, len(points) - 1):
        a = _sub(tuple(points[i]), origin)
        b = _sub(tuple(points[i + 1]), origin)
        cross = _cross(a, b)
        for axis in range(3):
            normal[axis] += cross[axis]
    return (normal[0], normal[1], normal[2])


def _signed_volume6(model: Polyhedron) -> float:
    """Шесть ориентированных объёмов тетраэдров относительно опорной вершины."""
    if not model.vertices:
        return 0.0
    reference = model.vertices[0].coords()
    total = 0.0
    for face in model.faces:
        points = [
            tuple(point.coords()[axis] - reference[axis] for axis in range(3))
            for point in face.points
        ]
        if len(points) < 3:
            continue
        a = points[0]
        for i in range(1, len(points) - 1):
            b, c = points[i], points[i + 1]
            total += _dot(a, _cross(b, c))
    return total


def _is_closed(model: Polyhedron) -> bool:
    """Проверить, что каждое неориентированное ребро встречается дважды."""
    edge_counts: dict[tuple[int, int], int] = {}
    vertex_index = {id(point): i for i, point in enumerate(model.vertices)}
    for face in model.faces:
        ids = [vertex_index[id(point)] for point in face.points]
        for i, first in enumerate(ids):
            second = ids[(i + 1) % len(ids)]
            edge = (first, second) if first < second else (second, first)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return bool(edge_counts) and all(count == 2 for count in edge_counts.values())


def _outward_sign(model: Polyhedron) -> float:
    """Определить общий знак обхода граней, чтобы нормали смотрели наружу.

    Для замкнутой ориентированной сетки знак определяется знаком объёма.
    Для открытых поверхностей (например, вазы) используется суммарное
    направление нормалей относительно центра вершин.
    """
    volume6 = _signed_volume6(model)
    bounds = [
        max(point.coords()[axis] for point in model.vertices)
        - min(point.coords()[axis] for point in model.vertices)
        for axis in range(3)
    ] if model.vertices else [1.0, 1.0, 1.0]
    scale = max(max(bounds), 1e-6)
    volume_epsilon = max(scale ** 3 * 1e-12, 1e-18)
    if _is_closed(model) and abs(volume6) > volume_epsilon:
        return 1.0 if volume6 > 0 else -1.0

    if not model.vertices:
        return 1.0
    center = tuple(
        sum(point.coords()[axis] for point in model.vertices) / len(model.vertices)
        for axis in range(3)
    )
    score = 0.0
    for face in model.faces:
        points = [point.coords() for point in face.points]
        normal = face_cross_normal(points)
        face_center = tuple(
            sum(point[axis] for point in points) / len(points)
            for axis in range(3)
        )
        score += _dot(normal, _sub(face_center, center))
    return -1.0 if score < 0 else 1.0


@dataclass(frozen=True)
class FaceView:
    """Грань в координатах камеры после проверки направления нормали."""

    index: int
    points: tuple[Vec3, ...]
    centroid: Vec3
    normal: Vec3
    visible: bool


def classify_faces(
    model: Polyhedron,
    to_view: ToView,
    projection: str = "axon",
    camera_distance: float = 7.0,
) -> list[FaceView]:
    """Классифицировать все грани относительно камеры.

    Камера смотрит вдоль -Z в системе координат вида. Для параллельной
    проекции направление на камеру постоянно (0, 0, 1). Для перспективной
    проекции используется вектор от центра грани к центру проекции
    (0, 0, camera_distance).
    """
    if projection not in ("axon", "persp"):
        raise ValueError("Проекция должна быть 'axon' или 'persp'")

    outward_sign = _outward_sign(model)
    result: list[FaceView] = []
    for index, face in enumerate(model.faces):
        points = tuple(to_view(point.coords()) for point in face.points)
        if not points:
            continue
        raw_normal = face_cross_normal(points)
        normal = _unit(tuple(component * outward_sign for component in raw_normal))
        centroid = tuple(
            sum(point[axis] for point in points) / len(points)
            for axis in range(3)
        )

        if projection == "persp":
            to_camera = (-centroid[0], -centroid[1], camera_distance - centroid[2])
        else:
            to_camera = (0.0, 0.0, 1.0)

        normal_length = _length(normal)
        camera_length = _length(to_camera)
        facing = _dot(normal, to_camera)
        tolerance = 1e-10 * max(normal_length * camera_length, 1.0)
        visible = normal_length > 0 and facing > tolerance
        result.append(FaceView(index, points, centroid, normal, visible))
    return result


def visible_and_culled(face_views: Sequence[FaceView]) -> tuple[list[int], list[int]]:
    """Вернуть индексы лицевых и отсечённых граней в исходном порядке."""
    visible = [face.index for face in face_views if face.visible]
    culled = [face.index for face in face_views if not face.visible]
    return visible, culled


def filtered_model(model: Polyhedron, visible_indices: Sequence[int]) -> Polyhedron:
    """Создать оболочку модели только с лицевыми гранями, не копируя вершины."""
    result = Polyhedron(
        model.vertices,
        [model.faces[index] for index in visible_indices],
        name=model.name,
    )
    # 9 лаба
    # При сглаживании учитываются все смежные грани, в том числе отсечённые.
    result._vertex_normals = list(model.vertex_normals())
    return result
