#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Построение фигуры вращения (задание 2).

Фигура вращения задаётся:
  * образующей — ломаной из точек (x, y, z);
  * осью вращения — одной из координатных осей X, Y или Z (через начало координат);
  * числом разбиений n — угол одного шага равен 360° / n.

Каждая точка образующей поворачивается n раз на угол k * 360° / n (k = 0..n-1)
матрицей поворота из transforms.py. Соседние «кольца» точек соединяются гранями:
четырёхугольниками, а если точка лежит на оси (полюс) — треугольниками.
Результат — обычный Polyhedron (вершины + грани), который можно показать в
просмотрщике, преобразовать аффинными матрицами и сохранить в Wavefront OBJ.

Запуск без интерфейса (создаёт OBJ-файл):
    python revolution.py -a y -n 24 -o vase.obj --points "0 0; 1 0; 1.2 0.5; 0.6 1.5"
    python revolution.py -a y -n 16 -o torus.obj --preset Тор
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import transforms
from geometry import Point, Polygon, Polyhedron

AXES = ("x", "y", "z")
MIN_SEGMENTS = 3
MAX_SEGMENTS = 360

_ROTATIONS = {
    "x": transforms.rotation_x,
    "y": transforms.rotation_y,
    "z": transforms.rotation_z,
}


# ---------------------------------------------------------------- образующая
def parse_generatrix(text: str) -> List[Tuple[float, float, float]]:
    """Разобрать образующую из текста.

    Одна точка в строке (или через «;»): ``x y z``; координату z можно
    опустить (тогда z = 0). Разделители — пробелы, запятые или табуляция,
    десятичный разделитель — точка. Строки после «#» — комментарий.
    """
    points: List[Tuple[float, float, float]] = []
    lines = text.replace(";", "\n").splitlines()
    for number, raw in enumerate(lines, 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        fields = [f for f in re.split(r"[\s,]+", line) if f]
        if len(fields) not in (2, 3):
            raise ValueError(
                f"Точка {len(points) + 1} («{line}»): нужно 2 или 3 числа (x y [z])"
            )
        try:
            values = [float(f) for f in fields]
        except ValueError:
            raise ValueError(f"Точка {len(points) + 1} («{line}»): не число") from None
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Точка {len(points) + 1}: координаты должны быть конечными")
        if len(values) == 2:
            values.append(0.0)
        points.append((values[0], values[1], values[2]))
    return points


# ---------------------------------------------------------------- построение
def _signed_volume(vertices: Sequence[Point], faces: Sequence[Sequence[int]]) -> float:
    total = 0.0
    for face in faces:
        a = vertices[face[0]]
        for i in range(1, len(face) - 1):
            b, c = vertices[face[i]], vertices[face[i + 1]]
            total += (
                a.x * (b.y * c.z - b.z * c.y)
                - a.y * (b.x * c.z - b.z * c.x)
                + a.z * (b.x * c.y - b.y * c.x)
            )
    return total / 6.0


def _is_closed_surface(faces: Sequence[Sequence[int]]) -> bool:
    """Каждое ребро принадлежит ровно двум граням (поверхность без краёв)."""
    count: Dict[Tuple[int, int], int] = {}
    for face in faces:
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            key = (a, b) if a < b else (b, a)
            count[key] = count.get(key, 0) + 1
    return all(v == 2 for v in count.values())


def build_revolution(
    generatrix: Sequence[Sequence[float]],
    axis: str = "y",
    segments: int = 16,
    closed: bool = False,
    name: str | None = None,
) -> Polyhedron:
    """Построить фигуру вращения.

    generatrix — точки образующей (x, y, z) по порядку обхода;
    axis       — ось вращения 'x' | 'y' | 'z';
    segments   — число разбиений (угол шага = 360 / segments);
    closed     — соединить последнюю точку образующей с первой (например, тор).
    """
    axis = str(axis).strip().lower()
    if axis not in AXES:
        raise ValueError(f"Неизвестная ось {axis!r}: нужно x, y или z")
    if isinstance(segments, bool) or int(segments) != segments:
        raise ValueError("Число разбиений должно быть целым")
    segments = int(segments)
    if not MIN_SEGMENTS <= segments <= MAX_SEGMENTS:
        raise ValueError(
            f"Число разбиений должно быть от {MIN_SEGMENTS} до {MAX_SEGMENTS}"
        )

    pts = [tuple(float(c) for c in p) for p in generatrix]
    if any(len(p) != 3 or not all(math.isfinite(c) for c in p) for p in pts):
        raise ValueError("Каждая точка образующей — три конечных числа (x, y, z)")
    scale = max((abs(c) for p in pts for c in p), default=0.0) or 1.0
    eps = 1e-9 * scale

    # Убираем подряд идущие совпадающие точки.
    cleaned: List[Tuple[float, float, float]] = []
    for p in pts:
        if not cleaned or math.dist(p, cleaned[-1]) > eps:
            cleaned.append(p)
    if closed and len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) <= eps:
        cleaned.pop()
    pts = cleaned
    if len(pts) < (3 if closed else 2):
        raise ValueError(
            "Для замкнутой образующей нужно минимум 3 различные точки"
            if closed else "Образующая должна содержать минимум 2 различные точки"
        )

    k = AXES.index(axis)
    others = [i for i in range(3) if i != k]
    on_axis = [math.hypot(p[others[0]], p[others[1]]) <= eps for p in pts]
    if all(on_axis):
        raise ValueError("Все точки образующей лежат на оси вращения — фигура вырождена")

    # Матрицы поворота на углы j * 360° / n.
    step = 360.0 / segments
    matrices = [_ROTATIONS[axis](j * step) for j in range(segments)]

    vertices: List[Point] = []
    rings: List[List[int]] = []
    for p, pole in zip(pts, on_axis):
        if pole:
            # Точка на оси при вращении не движется — одна общая вершина.
            snapped = list(p)
            for i in others:
                snapped[i] = 0.0
            vertices.append(Point(*snapped))
            rings.append([len(vertices) - 1] * segments)
        else:
            ring = []
            for j in range(segments):
                vertices.append(Point(*p) if j == 0 else Point(*p).transformed(matrices[j]))
                ring.append(len(vertices) - 1)
            rings.append(ring)

    pairs = [(i, i + 1) for i in range(len(pts) - 1)]
    if closed:
        pairs.append((len(pts) - 1, 0))

    face_indices: List[List[int]] = []
    for a, b in pairs:
        for j in range(segments):
            j2 = (j + 1) % segments
            quad = [rings[a][j], rings[b][j], rings[b][j2], rings[a][j2]]
            face: List[int] = []
            for index in quad:  # схлопываем повторы у полюсов
                if not face or face[-1] != index:
                    face.append(index)
            if len(face) > 1 and face[0] == face[-1]:
                face.pop()
            if len(face) >= 3:
                face_indices.append(face)

    # Для замкнутой поверхности направляем нормали наружу.
    if _is_closed_surface(face_indices) and _signed_volume(vertices, face_indices) < 0:
        face_indices = [list(reversed(f)) for f in face_indices]

    faces = [Polygon([vertices[i] for i in f]) for f in face_indices]
    return Polyhedron(vertices, faces, name or f"revolution_{axis}_{segments}")


# ---------------------------------------------------------------- заготовки
def _circle_points(cx: float, cy: float, r: float, n: int) -> str:
    return "\n".join(
        "%.4f %.4f" % (cx + r * math.cos(2 * math.pi * i / n),
                       cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    )


def _sphere_profile(n: int = 8) -> str:
    rows = []
    for i in range(n + 1):
        t = math.pi * i / n
        rows.append("%.4f %.4f" % (math.sin(t) if 0 < i < n else 0.0, math.cos(t)))
    return "\n".join(rows)


# Все заготовки рассчитаны на ось Y: x — расстояние до оси, y — высота.
PRESETS: Dict[str, Tuple[str, bool]] = {
    "Ваза": ("0 0\n1 0\n1.3 0.6\n0.8 1.4\n0.55 2.0\n0.9 2.6\n1.0 2.9", False),
    "Цилиндр": ("0 1\n1 1\n1 -1\n0 -1", False),
    "Конус": ("0 1.5\n1 -1\n0 -1", False),
    "Сфера": (_sphere_profile(8), False),
    "Тор": (_circle_points(2.0, 0.0, 0.7, 8), True),
    "Песочные часы": ("0 0\n1 1.5\n0.25 0.15\n0.25 -0.15\n1 -1.5\n0 0", False),
}


# ---------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить фигуру вращения и сохранить её в OBJ."
    )
    parser.add_argument("-a", "--axis", default="y", choices=AXES, help="ось вращения")
    parser.add_argument("-n", "--segments", type=int, default=16, help="число разбиений")
    parser.add_argument("-o", "--output", required=True, help="выходной .obj")
    parser.add_argument("--closed", action="store_true", help="замкнуть образующую")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--points", help='точки: "x y; x y; ..." (z необязателен)')
    src.add_argument("--file", help="текстовый файл с точками образующей")
    src.add_argument("--preset", choices=list(PRESETS), help="готовая образующая")
    args = parser.parse_args()

    closed = args.closed
    if args.preset:
        text, preset_closed = PRESETS[args.preset]
        closed = closed or preset_closed
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8-sig")
    else:
        text = args.points

    poly = build_revolution(parse_generatrix(text), args.axis, args.segments, closed)
    from viewer import save_obj  # только здесь, чтобы CLI не зависел от интерфейса

    save_obj(poly, args.output)
    print(poly.info())
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
