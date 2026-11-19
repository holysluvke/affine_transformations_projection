"""Отдельный модуль освещения Phong и toon-shading для OBJ-просмотрщика.

Модуль отделяет алгоритмы закраски от базового Z-буфера и интерфейса Tkinter.
Он содержит вспомогательные функции векторной математики, попиксельный
растеризатор Phong/toon и функцию сборки всей сцены.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

try:
    import numpy as np
except ImportError:  # NumPy ускоряет растеризацию, но не является обязательной зависимостью.
    np = None

from lighting import DEFAULT_LIGHT_POSITION, lambert_color
from zbuffer import PALETTE, ZBuffer, clip_near_varyings, project, triangulate

DEFAULT_TOON_AMBIENT_BIAS = 0.2
DEFAULT_TOON_THRESHOLDS = (0.4, 0.7)
DEFAULT_TOON_LEVELS = (0.3, 1.0, 1.3)


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
    *,
    ambient_bias: float = DEFAULT_TOON_AMBIENT_BIAS,
) -> float:
    """Вернуть множитель цвета по примеру Cel/toon shading из лекции (с. 43).

    На вход подаётся ``max(0, N·L)``. Перед квантованием к нему добавляется
    базовая составляющая ``ambient_bias``: ``diff = ambient_bias + max(0, N·L)``.
    При ``diff < 0.4`` возвращается 0.3, при ``diff < 0.7`` — 1.0,
    иначе — 1.3. Это ровно три дискретных уровня из учебного примера.
    """
    if (len(thresholds) != 2
            or not all(math.isfinite(value) for value in thresholds)
            or not 0 <= thresholds[0] <= thresholds[1] <= 2):
        raise ValueError("Нужны два возрастающих порога toon в диапазоне 0..2")
    if not math.isfinite(ambient_bias) or ambient_bias < 0:
        raise ValueError("Базовая составляющая toon должна быть конечной и неотрицательной")
    if not math.isfinite(diffuse_factor):
        raise ValueError("Коэффициент N·L должен быть конечным числом")
    factor = min(1.0, max(0.0, float(diffuse_factor)))
    diff = ambient_bias + factor
    if diff < thresholds[0]:
        return DEFAULT_TOON_LEVELS[0]
    if diff < thresholds[1]:
        return DEFAULT_TOON_LEVELS[1]
    return DEFAULT_TOON_LEVELS[2]


def phong_toon_color(
    position: Sequence[float],
    normal: Sequence[float],
    light_position: Sequence[float],
    object_color: Sequence[float],
    *,
    ambient_bias: float = DEFAULT_TOON_AMBIENT_BIAS,
    thresholds: Sequence[float] = DEFAULT_TOON_THRESHOLDS,
) -> tuple[float, float, float]:
    """Рассчитать RGB пикселя по диффузной модели Cel/toon shading из лекции.

    Нормализуются нормаль ``N`` и направление ``L = light_position - position``.
    Затем вычисляется ``diff = 0.2 + max(dot(N, L), 0.0)`` и выбирается один
    из трёх множителей цвета: 0.3, 1.0 или 1.3. Зеркальная составляющая
    здесь намеренно отсутствует: в презентации toon shading показан только
    для диффузного освещения (страницы 42–43).
    """
    vectors = (position, normal, light_position)
    if any(len(vector) != 3 or not all(math.isfinite(v) for v in vector)
           for vector in vectors):
        raise ValueError("Позиции и нормаль должны содержать три конечных числа")
    if (len(object_color) != 3
            or not all(math.isfinite(v) and 0 <= v <= 255 for v in object_color)):
        raise ValueError("Цвет объекта должен быть RGB в диапазоне 0..255")
    if not math.isfinite(ambient_bias) or ambient_bias < 0:
        raise ValueError("Базовая составляющая toon должна быть конечной и неотрицательной")

    unit_normal = normalize_vector(normal)
    light_direction = normalize_vector(tuple(
        light_position[i] - position[i] for i in range(3)
    ))
    ndotl = max(0.0, sum(unit_normal[i] * light_direction[i] for i in range(3)))
    multiplier = toon_band(ndotl, thresholds, ambient_bias=ambient_bias)
    return tuple(min(255.0, float(channel) * multiplier) for channel in object_color)


def phong_color(
    position: Sequence[float],
    normal: Sequence[float],
    light_position: Sequence[float],
    view_position: Sequence[float] | None,
    object_color: Sequence[float],
    *,
    ambient: float = 0.15,
    diffuse: float = 0.7,
    specular: float = 0.25,
    shininess: float = 32.0,
) -> tuple[float, float, float]:
    """Рассчитать непрерывное попиксельное освещение по модели Фонга.

    Используются ambient, diffuse и specular составляющие. Зеркальный блик
    белый; ``view_position`` задаёт положение камеры в той же системе
    координат, что и точка поверхности. Для ортографической проекции передайте
    ``None`` — тогда направление на наблюдателя считается равным (0, 0, 1).
    В отличие от toon-режима, коэффициент N·L не квантуется на ступени.
    """
    vectors = (position, normal, light_position)
    if any(len(vector) != 3 or not all(math.isfinite(v) for v in vector)
           for vector in vectors):
        raise ValueError("Позиции и нормаль должны содержать три конечных числа")
    if view_position is not None and (
        len(view_position) != 3 or not all(math.isfinite(v) for v in view_position)
    ):
        raise ValueError("Положение камеры должно содержать три конечных числа")
    if (len(object_color) != 3
            or not all(math.isfinite(v) and 0 <= v <= 255 for v in object_color)):
        raise ValueError("Цвет объекта должен быть RGB в диапазоне 0..255")
    if not all(math.isfinite(v) and v >= 0 for v in (ambient, diffuse, specular)):
        raise ValueError("Коэффициенты освещения должны быть конечными и неотрицательными")
    if not math.isfinite(shininess) or shininess <= 0:
        raise ValueError("Показатель блеска shininess должен быть положительным")

    unit_normal = normalize_vector(normal)
    light_direction = normalize_vector(tuple(
        light_position[i] - position[i] for i in range(3)
    ))
    if view_position is None:
        view_direction = (0.0, 0.0, 1.0)
    else:
        view_direction = normalize_vector(tuple(
            view_position[i] - position[i] for i in range(3)
        ))

    ndotl = max(0.0, sum(unit_normal[i] * light_direction[i] for i in range(3)))
    specular_factor = 0.0
    if ndotl > 0.0 and specular > 0.0:
        # Вектор отражённого луча R = 2(N·L)N − L.
        reflection = tuple(
            2.0 * ndotl * unit_normal[i] - light_direction[i]
            for i in range(3)
        )
        r_dot_v = max(0.0, sum(reflection[i] * view_direction[i] for i in range(3)))
        specular_factor = specular * (r_dot_v ** shininess)

    diffuse_intensity = max(0.0, ambient + diffuse * ndotl)
    return tuple(
        min(255.0, max(0.0, float(channel) * diffuse_intensity + 255.0 * specular_factor))
        for channel in object_color
    )


def _triangle_per_pixel(
    buffer, p0, p1, p2, n0, n1, n2, light_position,
    object_color, owner, *, positions=None, reciprocal_w=None, shade_pixel,
):
    """Общий цикл растеризации для попиксельных моделей освещения."""
    screen_points = (p0, p1, p2)
    normals = (n0, n1, n2)
    positions = positions or tuple(tuple(point[:3]) for point in screen_points)
    if len(positions) != 3:
        raise ValueError("Для попиксельного освещения нужны три 3D-позиции вершин")
    vectors = (*normals, *positions, light_position)
    if any(len(vector) != 3 or not all(math.isfinite(v) for v in vector)
           for vector in vectors):
        raise ValueError("Позиции, нормали и свет должны содержать три конечных числа")
    if (len(object_color) != 3
            or not all(math.isfinite(v) and 0 <= v <= 255 for v in object_color)):
        raise ValueError("Цвет объекта должен содержать RGB-компоненты в диапазоне 0..255")
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

                # Ключевое требование: нормаль интерполируется, а затем
                # нормализуется shade-функцией отдельно для каждого пикселя.
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
                pixel_color = shade_pixel(position, normal, base)

                depth[index] = z
                owners[index] = owner
                offset = index * 3
                for channel in range(3):
                    pixels[offset + channel] = max(0, min(255, round(pixel_color[channel])))
            u += ux
            v += vx
            z += dzdx
            index += 1



def _triangle_per_pixel_numpy(
    buffer, p0, p1, p2, n0, n1, n2, light_position,
    object_color, owner, *, positions=None, reciprocal_w=None, shade_batch,
):
    """Векторизованный растеризатор: целый прямоугольник обрабатывается пакетом."""
    if np is None:
        return False
    screen_points = (p0, p1, p2)
    positions = positions or tuple(tuple(point[:3]) for point in screen_points)
    if len(positions) != 3:
        raise ValueError("Для попиксельного освещения нужны три 3D-позиции вершин")
    vectors = (*((n0, n1, n2)), *positions, light_position)
    if any(len(vector) != 3 or not all(math.isfinite(v) for v in vector)
           for vector in vectors):
        raise ValueError("Позиции, нормали и свет должны содержать три конечных числа")
    if (len(object_color) != 3
            or not all(math.isfinite(v) and 0 <= v <= 255 for v in object_color)):
        raise ValueError("Цвет объекта должен содержать RGB-компоненты в диапазоне 0..255")
    if reciprocal_w is not None and (
        len(reciprocal_w) != 3
        or not all(math.isfinite(v) and v > 0 for v in reciprocal_w)
    ):
        raise ValueError("Для перспективы нужны три положительных значения 1/w")
    if not all(math.isfinite(v) for point in screen_points for v in point):
        return True

    ax, ay, az = p0
    bx, by, bz = p1
    cx, cy, cz = p2
    determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(determinant) < 1e-12:
        return True
    left = max(0, math.ceil(min(ax, bx, cx) - 0.5))
    right = min(buffer.width - 1, math.floor(max(ax, bx, cx) - 0.5))
    top = max(0, math.ceil(min(ay, by, cy) - 0.5))
    bottom = min(buffer.height - 1, math.floor(max(ay, by, cy) - 0.5))
    if left > right or top > bottom:
        return True

    ux, uy = (cy - ay) / determinant, (ax - cx) / determinant
    vx, vy = (ay - by) / determinant, (bx - ax) / determinant
    dzdx = ux * (bz - az) + vx * (cz - az)
    dzdy = uy * (bz - az) + vy * (cz - az)

    x_count, y_count = right - left + 1, bottom - top + 1
    x_delta = np.tile(np.arange(left, right + 1, dtype=np.float64) + 0.5 - ax, y_count)
    y_delta = np.repeat(np.arange(top, bottom + 1, dtype=np.float64) + 0.5 - ay, x_count)
    u = ux * x_delta + uy * y_delta
    v = vx * x_delta + vy * y_delta
    z = az + dzdx * x_delta + dzdy * y_delta
    coverage = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1.0 + 1e-10)
    candidate_positions = np.flatnonzero(coverage)
    if candidate_positions.size == 0:
        return True

    x_index = np.tile(np.arange(left, right + 1, dtype=np.int64), y_count)
    y_index = np.repeat(np.arange(top, bottom + 1, dtype=np.int64), x_count)
    pixel_indices = y_index[candidate_positions] * buffer.width + x_index[candidate_positions]
    candidate_depth = z[candidate_positions]
    if isinstance(buffer.depth, np.ndarray):
        old_depth = buffer.depth[pixel_indices]
    else:
        old_depth = np.fromiter(
            (buffer.depth[int(index)] for index in pixel_indices),
            dtype=np.float64, count=len(pixel_indices),
        )
    depth_pass = candidate_depth < old_depth
    if not np.any(depth_pass):
        return True

    selected_positions = candidate_positions[depth_pass]
    pixel_indices = pixel_indices[depth_pass]
    selected_z = candidate_depth[depth_pass]
    u_selected, v_selected = u[selected_positions], v[selected_positions]
    w0, w1, w2 = 1.0 - u_selected - v_selected, u_selected, v_selected
    if reciprocal_w is not None:
        inv_w = np.asarray(reciprocal_w, dtype=np.float64)
        w0, w1, w2 = w0 * inv_w[0], w1 * inv_w[1], w2 * inv_w[2]
        total = w0 + w1 + w2
        valid_total = total > 1e-15
        safe_total = np.where(valid_total, total, 1.0)
        w0, w1, w2 = w0 / safe_total, w1 / safe_total, w2 / safe_total
        w0 = np.where(valid_total, w0, 1.0)
        w1 = np.where(valid_total, w1, 0.0)
        w2 = np.where(valid_total, w2, 0.0)

    vertex_normals = np.asarray((n0, n1, n2), dtype=np.float64)
    vertex_positions = np.asarray(positions, dtype=np.float64)
    interpolated_normals = (
        w0[:, None] * vertex_normals[0]
        + w1[:, None] * vertex_normals[1]
        + w2[:, None] * vertex_normals[2]
    )
    interpolated_positions = (
        w0[:, None] * vertex_positions[0]
        + w1[:, None] * vertex_positions[1]
        + w2[:, None] * vertex_positions[2]
    )
    normal_lengths = np.linalg.norm(interpolated_normals, axis=1)
    normals_unit = np.divide(
        interpolated_normals, normal_lengths[:, None],
        out=np.zeros_like(interpolated_normals),
        where=normal_lengths[:, None] > 1e-15,
    )
    colors = shade_batch(
        interpolated_positions, normals_unit,
        np.asarray(object_color, dtype=np.float64),
    )
    colors_u8 = np.clip(np.rint(colors), 0, 255).astype(np.uint8)

    if isinstance(buffer.depth, np.ndarray):
        buffer.depth[pixel_indices] = selected_z
        buffer.owner[pixel_indices] = owner
    else:
        for index, z_value in zip(pixel_indices, selected_z):
            buffer.depth[int(index)] = float(z_value)
            buffer.owner[int(index)] = owner
    rgb = np.frombuffer(buffer.rgb, dtype=np.uint8).reshape(-1, 3)
    rgb[pixel_indices] = colors_u8
    return True


def _shade_toon_batch(position, normal, base, light_position,
                      ambient_bias, thresholds):
    light_vector = np.asarray(light_position, dtype=np.float64) - position
    light_length = np.linalg.norm(light_vector, axis=1)
    light_direction = np.divide(
        light_vector, light_length[:, None], out=np.zeros_like(light_vector),
        where=light_length[:, None] > 1e-15,
    )
    ndotl = np.clip(np.einsum("ij,ij->i", normal, light_direction), 0.0, 1.0)
    diff = ambient_bias + ndotl
    multiplier = np.where(
        diff < thresholds[0], DEFAULT_TOON_LEVELS[0],
        np.where(diff < thresholds[1], DEFAULT_TOON_LEVELS[1], DEFAULT_TOON_LEVELS[2]),
    )
    return multiplier[:, None] * base[None, :]


def _shade_phong_batch(position, normal, base, light_position,
                       view_position, ambient, diffuse, specular, shininess):
    light_vector = np.asarray(light_position, dtype=np.float64) - position
    light_length = np.linalg.norm(light_vector, axis=1)
    light_direction = np.divide(
        light_vector, light_length[:, None], out=np.zeros_like(light_vector),
        where=light_length[:, None] > 1e-15,
    )
    if view_position is None:
        view_direction = np.zeros_like(position)
        view_direction[:, 2] = 1.0
    else:
        view_vector = np.asarray(view_position, dtype=np.float64) - position
        view_length = np.linalg.norm(view_vector, axis=1)
        view_direction = np.divide(
            view_vector, view_length[:, None], out=np.zeros_like(view_vector),
            where=view_length[:, None] > 1e-15,
        )
    ndotl = np.maximum(0.0, np.einsum("ij,ij->i", normal, light_direction))
    diffuse_intensity = np.maximum(0.0, ambient + diffuse * ndotl)
    result = diffuse_intensity[:, None] * base[None, :]
    if specular > 0.0:
        reflection = 2.0 * ndotl[:, None] * normal - light_direction
        r_dot_v = np.maximum(0.0, np.einsum("ij,ij->i", reflection, view_direction))
        specular_factor = specular * np.power(r_dot_v, shininess)
        specular_factor = np.where(ndotl > 0.0, specular_factor, 0.0)
        result = result + (255.0 * specular_factor)[:, None]
    return np.clip(result, 0.0, 255.0)


def triangle_phong_toon(
    buffer, p0, p1, p2, n0, n1, n2, light_position,
    object_color=(90, 170, 245), owner=0, *, positions=None,
    reciprocal_w=None, ambient_bias=DEFAULT_TOON_AMBIENT_BIAS,
    thresholds=DEFAULT_TOON_THRESHOLDS,
):
    """Попиксельный Cel/toon shading по примеру из лекции (с. 43).

    Нормали и 3D-позиции интерполируются по барицентрическим весам; в
    перспективе используется поправка 1/w. Нормаль нормализуется отдельно
    для каждого пикселя, после чего применяется формула
    ``diff = ambient_bias + max(0, N·L)`` и один из множителей 0.3/1.0/1.3.
    """
    if not math.isfinite(ambient_bias) or ambient_bias < 0:
        raise ValueError("Базовая составляющая toon должна быть конечной и неотрицательной")
    if (len(thresholds) != 2
            or not all(math.isfinite(v) for v in thresholds)
            or not 0 <= thresholds[0] <= thresholds[1] <= 2):
        raise ValueError("Пороги toon должны быть двумя числами в диапазоне 0..2 по возрастанию")
    thresholds = tuple(float(value) for value in thresholds)

    def shade_batch(position, normal, base):
        return _shade_toon_batch(
            position, normal, base, light_position, ambient_bias, thresholds,
        )

    if not _triangle_per_pixel_numpy(
        buffer, p0, p1, p2, n0, n1, n2, light_position, object_color, owner,
        positions=positions, reciprocal_w=reciprocal_w, shade_batch=shade_batch,
    ):
        def shade(position, normal, base):
            return phong_toon_color(
                position, normal, light_position, base,
                ambient_bias=ambient_bias, thresholds=thresholds,
            )
        _triangle_per_pixel(
            buffer, p0, p1, p2, n0, n1, n2, light_position, object_color, owner,
            positions=positions, reciprocal_w=reciprocal_w, shade_pixel=shade,
        )


def triangle_phong(
    buffer, p0, p1, p2, n0, n1, n2, light_position,
    object_color=(90, 170, 245), owner=0, *, positions=None,
    reciprocal_w=None, view_position=None, ambient=0.15, diffuse=0.7,
    specular=0.25, shininess=32.0,
):
    """Попиксельная непрерывная модель освещения Фонга для одного треугольника."""
    def shade_batch(position, normal, base):
        return _shade_phong_batch(
            position, normal, base, light_position, view_position,
            ambient, diffuse, specular, shininess,
        )

    if not _triangle_per_pixel_numpy(
        buffer, p0, p1, p2, n0, n1, n2, light_position, object_color, owner,
        positions=positions, reciprocal_w=reciprocal_w, shade_batch=shade_batch,
    ):
        def shade(position, normal, base):
            return phong_color(
                position, normal, light_position, view_position, base,
                ambient=ambient, diffuse=diffuse, specular=specular,
                shininess=shininess,
            )
        _triangle_per_pixel(
            buffer, p0, p1, p2, n0, n1, n2, light_position, object_color, owner,
            positions=positions, reciprocal_w=reciprocal_w, shade_pixel=shade,
        )


def render_scene(models, to_view, width, height, zoom, mode, distance, filled=True, *,
                 shading="flat", light_position=DEFAULT_LIGHT_POSITION,
                 object_color=None, object_colors=None, ambient=0.15, diffuse=0.85,
                 specular=0.25, shininess=32.0):
    """Нарисовать сцену с плоской, Гуро, Фонг- или toon-закраской.

    ``light_position`` задаётся в мировых координатах. ``object_colors`` —
    необязательный список цветов по индексам моделей; элемент None оставляет
    цвет из PALETTE. Режим texture пока использует базовый цвет модели, так как
    загрузчик проекта ещё не хранит UV-координаты и изображения материалов.
    """
    if shading not in ("flat", "gouraud", "phong", "toon", "texture"):
        raise ValueError("Режим должен быть flat, gouraud, phong, toon или texture")
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
                    elif shading == "phong":
                        buffer.triangle_phong(
                            *projected, *normals_triangle, light_view, base, owner,
                            positions=positions, reciprocal_w=reciprocal_w,
                            view_position=(0.0, 0.0, distance) if mode == "persp" else None,
                            ambient=ambient, diffuse=diffuse,
                            specular=specular, shininess=shininess,
                        )
                    elif shading == "toon":
                        buffer.triangle_phong_toon(
                            *projected, *normals_triangle, light_view, base, owner,
                            positions=positions, reciprocal_w=reciprocal_w,
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
