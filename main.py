#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — демонстрация и самопроверка Части 1.

Запуск:
    python main.py

Скрипт:
  * строит все пять правильных многогранников,
  * проверяет формулу Эйлера и равенство длин рёбер,
  * показывает работу матрицы (то, что будет делать человек 2),
  * демонстрирует однородные координаты.
"""

import math

from geometry import (
    POLYHEDRA_KEYS,
    Point,
    create_hexahedron,
    create_polyhedron,
)


def check_polyhedra() -> None:
    print("Проверка построения правильных многогранников")
    print("=" * 78)
    for key in POLYHEDRA_KEYS:
        poly = create_polyhedron(key, radius=1.0)
        verts = poly.vertex_array()
        lengths = {round(math.dist(verts[a], verts[b]), 6) for a, b in poly.edges()}
        print(f"{poly.info():<62} | рёбер равной длины: {len(lengths)}")
        assert poly.euler_check() == 2, f"{key}: не выполняется формула Эйлера"
        assert len(lengths) == 1, f"{key}: рёбра разной длины {lengths}"


def demo_matrix_application() -> None:
    # Матрица сдвига - то же самое будет возвращать модуль человека 2
    shift = [[1, 0, 0, 2.0],
             [0, 1, 0, -1.0],
             [0, 0, 1, 0.5],
             [0, 0, 0, 1.0]]

    cube = create_hexahedron(radius=1.0)
    print("\nДо преобразования:  ", cube.vertices[0])
    cube.transform(shift)
    print("После сдвига:       ", cube.vertices[0])
    print("Новый центр куба:   ",
          tuple(round(c, 4) for c in cube.center()))


def demo_homogeneous() -> None:
    p = Point(1.0, 2.0, 3.0)
    print("\nОднородные координаты точки:", p.homogeneous())


if __name__ == "__main__":
    check_polyhedra()
    demo_matrix_application()
    demo_homogeneous()
    print("\nВсё в порядке: классы и многогранники готовы к интеграции.")