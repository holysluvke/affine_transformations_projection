"""Отдельный модуль освещения Phong и toon-shading для OBJ-просмотрщика.

Модуль отделяет алгоритмы закраски от базового Z-буфера и интерфейса Tkinter.
Он содержит вспомогательные функции векторной математики, попиксельный
растеризатор Phong/toon и функцию сборки всей сцены.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from lighting import DEFAULT_LIGHT_POSITION, lambert_color
from zbuffer import PALETTE, ZBuffer, clip_near_varyings, project, triangulate

DEFAULT_TOON_THRESHOLDS = (0.2, 0.45, 0.7)
DEFAULT_AMBIENT = 0.18
DEFAULT_DIFFUSE = 0.82


def normalize_vector(vector: Sequence[float]) -> tuple[float, float, float]:
    """Нормализовать трёхмерный вектор; нулевой вектор оставить нулевым.

    Параметр ``vector`` должен содержать три конечных числа. Возвращается
    единичный вектор того же направления или ``(0, 0, 0)``, если длина
    исходного вектора практически равна нулю.
    """
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError("Вектор должен содержать три конечных числа")
    length = math.hypot(*vector)
    if length <= 1e-15:
        return (0.0, 0.0, 0.0)
    return tuple(value / length for value in vector)


def transform_normal_to_view(
    normal: Sequence[float], to_view: Callable[[Sequence[float]], Sequence[float]]
) -> tuple[float, float, float]:
    """Преобразовать мировую нормаль в систему координат камеры без переноса.

    ``to_view`` преобразует точки. Чтобы не применить к нормали перенос,
    функция вычитает образ нулевой точки из образа конца вектора, то есть
    оставляет линейную часть преобразования, затем нормализует результат.

    Аффинное преобразование самой модели выполняется раньше в
    ``Polyhedron.transform``: сохранённые OBJ-нормали обрабатываются
    обратной транспонированной матрицей A^(-T), вычисляемые пересчитываются.
    """
    origin = to_view((0.0, 0.0, 0.0))
    endpoint = to_view(tuple(float(value) for value in normal))
    direction = tuple(endpoint[i] - origin[i] for i in range(3))
    return normalize_vector(direction)


def toon_band(
    diffuse_factor: float,
    thresholds: Sequence[float] = DEFAULT_TOON_THRESHOLDS,
) -> float:
    """Перевести коэффициент ``max(0, N·L)`` в одну из четырёх ступеней.

    Три порога задают границы интервалов. Возвращаемые уровни равны
    ``0``, ``1/3``, ``2/3`` и ``1``. Пороги должны возрастать и лежать
    в диапазоне от 0 до 1.
    """
    if (len(thresholds) != 3
            or not all(math.isfinite(value) for value in thresholds)
            or not 0 <= thresholds[0] <= thresholds[1] <= thresholds[2] <= 1):
        raise ValueError("Нужны три возрастающих порога toon от 0 до 1")
    factor = min(1.0, max(0.0, float(diffuse_factor)))
    if factor < thresholds[0]:
        return 0.0
    if factor < thresholds[1]:
        return 1.0 / 3.0
    if factor < thresholds[2]:
        return 2.0 / 3.0
    return 1.0


def phong_toon_color(
    position: Sequence[float],
    normal: Sequence[float],
    light_position: Sequence[float],
    object_color: Sequence[float],
    *,
    ambient: float = DEFAULT_AMBIENT,
    diffuse: float = DEFAULT_DIFFUSE,
    thresholds: Sequence[float] = DEFAULT_TOON_THRESHOLDS,
) -> tuple[float, float, float]:
    """Рассчитать RGB одного пикселя по нормали и точечному источнику света.

    Все 3D-векторы должны быть в одной системе координат. Нормализуются
    нормаль ``N`` и направление ``L = light_position - position``. Затем
    вычисляется ``max(0, N·L)``, выбирается toon-ступень и применяется
    интенсивность ``clamp(ambient + diffuse * ступень, 0, 1)`` к RGB.

    Реализована попиксельная диффузная модель с toon-дискретизацией.
    Зеркальный блик (specular) и затухание света с расстоянием не входят
    в эту версию алгоритма.
    """
    vectors = (position, normal, light_position)
    if any(len(vector) != 3 or not all(math.isfinite(v) for v in vector)
           for vector in vectors):
        raise ValueError("Позиции и нормаль должны содержать три конечных числа")
    if (len(object_color) != 3
            or not all(math.isfinite(v) and 0 <= v <= 255 for v in object_color)):
        raise ValueError("Цвет объекта должен быть RGB в диапазоне 0..255")
    if not all(math.isfinite(v) and v >= 0 for v in (ambient, diffuse)):
        raise ValueError("Ambient и diffuse должны быть конечными и неотрицательными")

    unit_normal = normalize_vector(normal)
    light_direction = normalize_vector(tuple(
        light_position[i] - position[i] for i in range(3)
    ))
    ndotl = max(0.0, sum(unit_normal[i] * light_direction[i] for i in range(3)))
    band = toon_band(ndotl, thresholds)
    intensity = min(1.0, max(0.0, ambient + diffuse * band))
    return tuple(float(channel) * intensity for channel in object_color)


def triangle_phong_toon(
    buffer, p0, p1, p2, n0, n1, n2, light_position,
    object_color=(90, 170, 245), owner=0, *, positions=None,
    reciprocal_w=None, ambient=DEFAULT_AMBIENT, diffuse=DEFAULT_DIFFUSE,
    thresholds=DEFAULT_TOON_THRESHOLDS,
):
    """Попиксельный Phong с дискретной яркостью (toon shading).

    ``p0..p2`` — экранные координаты (x, y, глубина), ``n0..n2`` —
    нормали вершин, а ``positions`` — соответствующие 3D-позиции в той
    же системе координат, что и ``light_position``. Если positions не
    заданы, для автономного использования берутся первые три компонента
    экранных точек. В основном рендерере всегда передаются позиции вида.

    Нормали и 3D-позиция интерполируются барицентрически. При перспективе
    применяется поправка 1/w; интерполированная нормаль нормализуется
    отдельно для каждого пикселя перед вычислением N·L.
    """
    screen_points = (p0, p1, p2)
    normals = (n0, n1, n2)
    positions = positions or tuple(tuple(point[:3]) for point in screen_points)
    if len(positions) != 3:
        raise ValueError("Для Phong нужны три 3D-позиции вершин")
    vectors = (*normals, *positions, light_position)
    if any(len(vector) != 3 or not all(math.isfinite(v) for v in vector)
           for vector in vectors):
        raise ValueError("Позиции, нормали и свет должны содержать три конечных числа")
    if (len(object_color) != 3
            or not all(math.isfinite(v) and 0 <= v <= 255 for v in object_color)):
        raise ValueError("Цвет объекта должен содержать RGB-компоненты в диапазоне 0..255")
    if not all(math.isfinite(v) and v >= 0 for v in (ambient, diffuse)):
        raise ValueError("Коэффициенты освещения должны быть конечными и неотрицательными")
    if (len(thresholds) != 3
            or not all(math.isfinite(v) for v in thresholds)
            or not 0 <= thresholds[0] <= thresholds[1] <= thresholds[2] <= 1):
        raise ValueError("Пороги toon должны быть тремя числами от 0 до 1 по возрастанию")
    if reciprocal_w is not None and (
        len(reciprocal_w) != 3
        or not all(math.isfinite(v) and v > 0 for v in reciprocal_w)
    ):
        raise ValueError("Для перспективы нужны три положительных значения 1/w")
    if not all(math.isfinite(v) for point in screen_points for v in point):
        return

    ax, ay, az = p0
    bx, by, bz = p1
    cx, cy, cz = p2
    determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(determinant) < 1e-12:
        return
    left = max(0, math.ceil(min(ax, bx, cx) - 0.5))
    right = min(buffer.width - 1, math.floor(max(ax, bx, cx) - 0.5))
    top = max(0, math.ceil(min(ay, by, cy) - 0.5))
    bottom = min(buffer.height - 1, math.floor(max(ay, by, cy) - 0.5))
    if left > right or top > bottom:
        return

    ux, uy = (cy - ay) / determinant, (ax - cx) / determinant
    vx, vy = (ay - by) / determinant, (bx - ax) / determinant
    dzdx = ux * (bz - az) + vx * (cz - az)
    dzdy = uy * (bz - az) + vy * (cz - az)
    depth, pixels, owners = buffer.depth, buffer.rgb, buffer.owner
    thresholds = tuple(thresholds)
    base = tuple(float(c) for c in object_color)

    for y in range(top, bottom + 1):
        dx, dy = left + 0.5 - ax, y + 0.5 - ay
        u, v = ux * dx + uy * dy, vx * dx + vy * dy
        z = az + dzdx * dx + dzdy * dy
        index = y * buffer.width + left
        for _ in range(left, right + 1):
            if u >= -1e-10 and v >= -1e-10 and u + v <= 1 + 1e-10 and z < depth[index]:
                w0, w1, w2 = 1.0 - u - v, u, v
                if reciprocal_w is not None:
                    w0 *= reciprocal_w[0]
                    w1 *= reciprocal_w[1]
                    w2 *= reciprocal_w[2]
                    total = w0 + w1 + w2
                    if total > 1e-15:
                        w0, w1, w2 = w0 / total, w1 / total, w2 / total
                    else:
                        w0, w1, w2 = 1.0, 0.0, 0.0

                normal = tuple(
                    w0 * n0[axis] + w1 * n1[axis] + w2 * n2[axis]
                    for axis in range(3)
                )
                position = tuple(
                    w0 * positions[0][axis]
                    + w1 * positions[1][axis]
                    + w2 * positions[2][axis]
                    for axis in range(3)
                )
                pixel_color = phong_toon_color(
                    position, normal, light_position, base,
                    ambient=ambient, diffuse=diffuse, thresholds=thresholds,
                )

                depth[index] = z
                owners[index] = owner
                offset = index * 3
                for channel in range(3):
                    pixels[offset + channel] = max(0, min(255, round(pixel_color[channel])))
            u += ux
            v += vx
            z += dzdx
            index += 1



def render_scene(models, to_view, width, height, zoom, mode, distance, filled=True, *,
                 shading="flat", light_position=DEFAULT_LIGHT_POSITION,
                 object_color=None, object_colors=None, ambient=0.15, diffuse=0.85):
    """Нарисовать сцену с плоской, Гуро или попиксельной toon-закраской.

    ``light_position`` задаётся в мировых координатах. ``object_colors`` —
    необязательный список цветов по индексам моделей; элемент None оставляет
    цвет из PALETTE. Режим texture пока использует базовый цвет модели, так как
    загрузчик проекта ещё не хранит UV-координаты и изображения материалов.
    """
    if shading not in ("flat", "gouraud", "toon", "texture"):
        raise ValueError("Режим должен быть flat, gouraud, toon или texture")
    buffer = ZBuffer(width, height)
    edges = []
    owner = 0

    light_view = to_view(light_position)

    def screen(p):
        return project(p, width, height, zoom, mode, distance)

    for object_index, model in enumerate(models):
        vertices = [to_view(p) for p in model.vertex_array()]
        requested_color = (object_colors[object_index]
                           if object_colors is not None and object_index < len(object_colors)
                           else object_color)
        base = requested_color if requested_color is not None else PALETTE[object_index % len(PALETTE)]
        for face_index, indices in enumerate(model.faces_as_indices()):
            owner += 1
            points = [vertices[i] for i in indices]
            corner_normals = model.face_vertex_normals(face_index)
            normals = [transform_normal_to_view(n, to_view) for n in corner_normals]

            # Flat — одна яркость на грань; texture — безопасный базовый цвет,
            # пока в геометрии не реализованы OBJ vt/mtl.
            face_normal = [0.0, 0.0, 0.0]
            for a, b in zip(points, points[1:] + points[:1]):
                face_normal[0] += (a[1] - b[1]) * (a[2] + b[2])
                face_normal[1] += (a[2] - b[2]) * (a[0] + b[0])
                face_normal[2] += (a[0] - b[0]) * (a[1] + b[1])
            centroid = tuple(sum(p[i] for p in points) / len(points) for i in range(3))
            face_color = lambert_color(centroid, face_normal, light_view, base,
                                       ambient=ambient, diffuse=diffuse)
            color = (tuple(max(0, min(255, round(v))) for v in face_color)
                     if filled and shading in ("flat", "texture") else (255, 255, 255))

            for triangle in triangulate(points):
                tri_points = [points[i] for i in triangle]
                tri_normals = [normals[i] for i in triangle]
                if mode == "persp":
                    tri_points, tri_normals = clip_near_varyings(
                        tri_points, tri_normals, distance
                    )
                if len(tri_points) < 3:
                    continue
                for i in range(1, len(tri_points) - 1):
                    positions = (tri_points[0], tri_points[i], tri_points[i + 1])
                    projected = [screen(p) for p in positions]
                    reciprocal_w = (
                        tuple(1.0 / (distance - p[2]) for p in positions)
                        if mode == "persp" else None
                    )
                    normals_triangle = (tri_normals[0], tri_normals[i], tri_normals[i + 1])
                    if not filled:
                        buffer.triangle(*projected, (255, 255, 255), owner)
                    elif shading == "gouraud":
                        vertex_colors = tuple(
                            lambert_color(position, normal, light_view, base,
                                          ambient=ambient, diffuse=diffuse)
                            for position, normal in zip(positions, normals_triangle)
                        )
                        buffer.triangle_gouraud(
                            *projected, *vertex_colors, owner, reciprocal_w=reciprocal_w
                        )
                    elif shading == "toon":
                        buffer.triangle_phong_toon(
                            *projected, *normals_triangle, light_view, base, owner,
                            positions=positions, reciprocal_w=reciprocal_w,
                            ambient=ambient, diffuse=diffuse,
                        )
                    else:
                        buffer.triangle(*projected, color, owner)
            for a, b in zip(points, points[1:] + points[:1]):
                if mode == "persp":
                    # Обрезаем ребро по той же ближней плоскости, что и треугольники.
                    limit = distance - 0.05
                    if a[2] > limit and b[2] > limit:
                        continue
                    if (a[2] > limit) != (b[2] > limit):
                        t = (limit - a[2]) / (b[2] - a[2])
                        intersection = tuple(a[j] + t * (b[j] - a[j]) for j in range(3))
                        if a[2] > limit:
                            a = intersection
                        else:
                            b = intersection
                edges.append((screen(a), screen(b), owner))
    for a, b, owner in edges:
        buffer.line(a, b, owner=owner)
    return buffer
