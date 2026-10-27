#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py — ЧАСТЬ 1: геометрические классы и правильные многогранники.

==============================================================================
пояснение для команды
==============================================================================
1. Точка хранится в декартовых координатах (x, y, z) — класс Point.
2. Однородные координаты: Point.homogeneous() -> [x, y, z, 1.0].
3. Матрица преобразования — список из 4 списков по 4 float (4x4),
   умножение СЛЕВА:  v' = M @ v,  где v = (x, y, z, 1)^T.
   Пример: m[1][3] — это сдвиг по Y.
4. Применение матрицы к точке:
       p.apply_matrix(M)   — изменяет точку на месте (in-place), возвращает self
       p.transformed(M)    — возвращает НОВУЮ точку
   (внутри учитывается деление на w — это нужно для перспективной проекции)
5. Применение матрицы ко всему многограннику:
       poly.transform(M)      — in-place (рекомендуется, см. п.6)
       poly.transformed(M)    — на копии
6. Polygon хранит ССЫЛКИ на те же объекты Point, что и Polyhedron.vertices.
   Поэтому преобразование вершин автоматически преобразует грани.
   ВАЖНО: не пересоздавайте точки вершин новыми объектами — меняйте их
   на месте (apply_matrix / transform) либо берите poly.copy().
7. Многогранник всегда строится с центром в начале координат (0,0,0)
   и радиусом описанной сферы = radius (по умолчанию 1.0).
   poly.center() возвращает текущий центр (среднее вершин) —
   он нужен человеку 2 для масштабирования/вращения относительно центра.
==============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

# Тип матрицы 4x4 (для подсказок человеку 2)
Matrix = List[List[float]]
Vec3 = Tuple[float, float, float]


# ============================================================================
#  Вспомогательные векторные функции (работают с кортежами из 3 float)
# ============================================================================
def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_mul(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _v_len(a: Vec3) -> float:
    return math.sqrt(_v_dot(a, a))


def _v_norm(a: Vec3) -> Vec3:
    length = _v_len(a)
    if length < 1e-15:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _centroid(pts: Sequence[Vec3]) -> Vec3:
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n)



#  КЛАСС 1: ТОЧКА

class Point:
    """Точка (она же вектор) в трёхмерном пространстве."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ---------- координаты -------------------------------------------------
    def coords(self) -> Vec3:
        """Возвращает (x, y, z) кортежем."""
        return (self.x, self.y, self.z)

    def homogeneous(self) -> List[float]:
        """Однородные координаты [x, y, z, 1] — для человека 2 и 3."""
        return [self.x, self.y, self.z, 1.0]

    @staticmethod
    def from_homogeneous(v: Sequence[float]) -> "Point":
        """Собирает точку из однородного вектора с делением на w."""
        w = v[3]
        if abs(w) < 1e-12:          # защита от деления на 0 (точка в бесконечности)
            w = 1.0
        return Point(v[0] / w, v[1] / w, v[2] / w)

    # ---------- применение матрицы 4x4 -------------------------------------
    def apply_matrix(self, m: Matrix) -> "Point":
        """v' = M @ (x, y, z, 1)^T.  Меняет точку НА МЕСТЕ, возвращает self."""
        x, y, z = self.x, self.y, self.z
        nx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
        ny = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
        nz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
        nw = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3]
        if abs(nw) < 1e-12:
            nw = 1.0
        self.x, self.y, self.z = nx / nw, ny / nw, nz / nw
        return self

    def transformed(self, m: Matrix) -> "Point":
        """Возвращает НОВУЮ точку — результат применения матрицы."""
        return Point(self.x, self.y, self.z).apply_matrix(m)

   
    def copy(self) -> "Point":
        return Point(self.x, self.y, self.z)

    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2 +
                         (self.z - other.z) ** 2)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"Point({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"



#  КЛАСС 2: МНОГОУГОЛЬНИК (ГРАНЬ)

class Polygon:
    """Грань многогранника — замкнутая ломаная из точек (ссылки на Point)."""

    def __init__(self, points: Sequence[Point]) -> None:
        pts = list(points)
        if len(pts) < 3:
            raise ValueError("В грани должно быть не меньше 3 точек")
        self.points: List[Point] = pts

    # ---------- геометрия ---------------------------------------------------
    def centroid(self) -> Vec3:
        """Центр грани (среднее вершин)."""
        return _centroid([p.coords() for p in self.points])

    def normal(self) -> Vec3:
        """
        Ненормированная нормаль методом Ньюэлла.
        Работает и для неплоских многоугольников, и для n > 3.
        Для грани, обойдённой против часовой стрелки, если смотреть
        снаружи, нормаль направлена НАРУЖУ.
        """
        nx = ny = nz = 0.0
        n = len(self.points)
        for i in range(n):
            a = self.points[i]
            b = self.points[(i + 1) % n]
            nx += (a.y - b.y) * (a.z + b.z)
            ny += (a.z - b.z) * (a.x + b.x)
            nz += (a.x - b.x) * (a.y + b.y)
        return (nx, ny, nz)

    def unit_normal(self) -> Vec3:
        """Единичная внешняя нормаль — пригодится человеку 3."""
        return _v_norm(self.normal())

    # ---------- утилиты -----------------------------------------------------
    def copy(self) -> "Polygon":
        """Глубокая копия (с собственными точками)."""
        return Polygon([p.copy() for p in self.points])

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, i: int) -> Point:
        return self.points[i]

    def __repr__(self) -> str:
        return f"Polygon({len(self.points)} вершин)"


#  КЛАСС 3: МНОГОГРАННИК

class Polyhedron:
    """Многогранник: список вершин + список граней (Polygon)."""

    def __init__(self,
                 vertices: Sequence[Point],
                 faces: Sequence[Polygon],
                 name: str = "Многогранник") -> None:
        self.vertices: List[Point] = list(vertices)
        self.faces: List[Polygon] = list(faces)
        self.name = name

    # ---------- геометрия ---------------------------------------------------
    def center(self) -> Vec3:
        """Центр многогранника = среднее его вершин."""
        return _centroid([p.coords() for p in self.vertices])

    def center_point(self) -> Point:
        c = self.center()
        return Point(*c)

    def radius(self) -> float:
        """Радиус описанной сферы (макс. расстояние от центра до вершины)."""
        c = self.center()
        return max(math.sqrt((p.x - c[0]) ** 2 + (p.y - c[1]) ** 2 + (p.z - c[2]) ** 2)
                   for p in self.vertices)

    def edges(self) -> List[Tuple[int, int]]:
        """Список рёбер как пар индексов вершин (a < b) — для каркаса."""
        index = {id(p): i for i, p in enumerate(self.vertices)}
        seen, result = set(), []
        for face in self.faces:
            ids = [index[id(p)] for p in face.points]
            n = len(ids)
            for i in range(n):
                a, b = ids[i], ids[(i + 1) % n]
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result

    def faces_as_indices(self) -> List[List[int]]:
        """Грани в виде списков индексов вершин — удобно для рендера."""
        index = {id(p): i for i, p in enumerate(self.vertices)}
        return [[index[id(p)] for p in f.points] for f in self.faces]

    def vertex_array(self) -> List[Vec3]:
        """Все вершины кортежами — быстрый доступ для человека 3."""
        return [p.coords() for p in self.vertices]

    def face_normals(self) -> List[Vec3]:
        """Единичные внешние нормали всех граней."""
        return [f.unit_normal() for f in self.faces]

    def bounding_box(self) -> Tuple[Vec3, Vec3]:
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        zs = [p.z for p in self.vertices]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    # ---------- преобразования (точка входа для человека 2) -----------------
    def transform(self, m: Matrix) -> "Polyhedron":
        """Применяет матрицу 4x4 ко ВСЕМ вершинам на месте. Возвращает self."""
        for p in self.vertices:
            p.apply_matrix(m)
        return self

    def transformed(self, m: Matrix) -> "Polyhedron":
        """Возвращает НОВЫЙ многогранник — копию, к которой применена матрица."""
        return self.copy().transform(m)

    def copy(self) -> "Polyhedron":
        """Глубокая копия: новые точки, грани ссылаются на новые точки."""
        mapping: Dict[int, Point] = {id(p): Point(p.x, p.y, p.z) for p in self.vertices}
        new_vertices = [mapping[id(p)] for p in self.vertices]
        new_faces = [Polygon([mapping[id(p)] for p in f.points]) for f in self.faces]
        return Polyhedron(new_vertices, new_faces, self.name)

    # ---------- служебное ---------------------------------------------------
    def euler_check(self) -> int:
        """V - E + F. Для любого выпуклого многогранника должно быть равно 2."""
        return len(self.vertices) - len(self.edges()) + len(self.faces)

    def info(self) -> str:
        return (f"{self.name}: V={len(self.vertices)}, "
                f"E={len(self.edges())}, F={len(self.faces)}, "
                f"R={self.radius():.4f}, V-E+F={self.euler_check()}")

    def __repr__(self) -> str:
        return f"Polyhedron('{self.name}', V={len(self.vertices)}, F={len(self.faces)})"



#  ПОСТРОЕНИЕ ПРАВИЛЬНЫХ МНОГОГРАННИКОВ

PHI = (1.0 + math.sqrt(5.0)) / 2.0      # золотое сечение ≈ 1.618


def _orient_face(points: List[Point], inner: Vec3) -> List[Point]:
    """Разворачивает обход грани так, чтобы её нормаль смотрела НАРУЖУ."""
    n = len(points)
    nx = ny = nz = 0.0
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        nx += (a.y - b.y) * (a.z + b.z)
        ny += (a.z - b.z) * (a.x + b.x)
        nz += (a.x - b.x) * (a.y + b.y)

    c = _centroid([p.coords() for p in points])
    outward = _v_sub(c, inner)
    if _v_dot((nx, ny, nz), outward) < 0.0:
        points.reverse()
    return points


def _build(name: str,
           raw_vertices: Sequence[Vec3],
           raw_faces: Sequence[Sequence[int]],
           radius: float) -> Polyhedron:
    """Собирает Polyhedron из «сырых» координат с нормировкой радиуса."""
    max_r = max(_v_len(v) for v in raw_vertices)
    k = radius / max_r
    points = [Point(v[0] * k, v[1] * k, v[2] * k) for v in raw_vertices]

    faces: List[Polygon] = []
    for idx in raw_faces:
        pts = [points[i] for i in idx]
        faces.append(Polygon(_orient_face(pts, (0.0, 0.0, 0.0))))

    return Polyhedron(points, faces, name)


# ---------------------------------------------------------------- тетраэдр
def create_tetrahedron(radius: float = 1.0) -> Polyhedron:
    """Правильный тетраэдр: 4 вершины, 4 треугольника."""
    verts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return _build("Тетраэдр", verts, faces, radius)


# ---------------------------------------------------------------- гексаэдр
def create_hexahedron(radius: float = 1.0) -> Polyhedron:
    """Гексаэдр (куб): 8 вершин, 6 квадратов."""
    verts = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    faces = [(0, 3, 2, 1),      # низ  z = -1
             (4, 5, 6, 7),      # верх z = +1
             (0, 1, 5, 4),      # передний  y = -1
             (3, 7, 6, 2),      # задний    y = +1
             (0, 4, 7, 3),      # левый     x = -1
             (1, 2, 6, 5)]      # правый    x = +1
    return _build("Гексаэдр (куб)", verts, faces, radius)


# ---------------------------------------------------------------- октаэдр
def create_octahedron(radius: float = 1.0) -> Polyhedron:
    """Октаэдр: 6 вершин, 8 треугольников."""
    verts = [(1, 0, 0), (-1, 0, 0),
             (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
             (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)]
    return _build("Октаэдр", verts, faces, radius)


# ---------------------------------------------------------------- икосаэдр
def _faces_of_icosahedron(verts: Sequence[Vec3]) -> List[Tuple[int, int, int]]:
    """Автоматический поиск 20 треугольных граней по минимальной длине ребра."""
    n = len(verts)

    def dist2(i: int, j: int) -> float:
        return sum((verts[i][k] - verts[j][k]) ** 2 for k in range(3))

    edge2 = min(dist2(i, j) for i in range(n) for j in range(i + 1, n))
    tol = edge2 * 1e-6

    faces: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dist2(i, j) - edge2) > tol:
                continue
            for k in range(j + 1, n):
                if abs(dist2(i, k) - edge2) <= tol and abs(dist2(j, k) - edge2) <= tol:
                    faces.append((i, j, k))
    return faces


def create_icosahedron(radius: float = 1.0) -> Polyhedron:
    """Икосаэдр (*бонус): 12 вершин, 20 треугольников."""
    verts: List[Vec3] = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts.append((0.0, s1 * 1.0, s2 * PHI))
            verts.append((s1 * 1.0, s2 * PHI, 0.0))
            verts.append((s2 * PHI, 0.0, s1 * 1.0))

    faces = _faces_of_icosahedron(verts)
    assert len(verts) == 12 and len(faces) == 20, "Ошибка построения икосаэдра"
    return _build("Икосаэдр", verts, faces, radius)


# ---------------------------------------------------------------- додекаэдр
def _order_cycle(verts: Sequence[Vec3],
                 indices: Sequence[int],
                 axis: Vec3) -> List[int]:
    """Упорядочивает вершины грани против часовой стрелки, глядя с конца axis."""
    axis = _v_norm(axis)
    pts = [verts[i] for i in indices]
    c = _centroid(pts)

    u: Vec3 = (0.0, 0.0, 0.0)
    for p in pts:
        v = _v_sub(p, c)
        v = _v_sub(v, _v_mul(axis, _v_dot(v, axis)))
        if _v_len(v) > 1e-9:
            u = _v_norm(v)
            break
    w = _v_cross(axis, u)

    def angle(p: Vec3) -> float:
        v = _v_sub(p, c)
        return math.atan2(_v_dot(v, w), _v_dot(v, u))

    order = sorted(range(len(indices)), key=lambda i: angle(pts[i]))
    return [indices[i] for i in order]


def _faces_of_dodecahedron(verts: Sequence[Vec3]) -> List[List[int]]:
    """12 пятиугольных граней; центры граней — направления вершин икосаэдра."""
    base = [(1.0, 0.0, PHI), (0.0, PHI, 1.0), (PHI, 1.0, 0.0)]
    directions: List[Vec3] = []
    for b in base:
        nz = [i for i in range(3) if abs(b[i]) > 1e-12]
        for s0 in (1, -1):
            for s1 in (1, -1):
                v = [0.0, 0.0, 0.0]
                v[nz[0]] = s0 * b[nz[0]]
                v[nz[1]] = s1 * b[nz[1]]
                directions.append((v[0], v[1], v[2]))

    faces: List[List[int]] = []
    for d in directions:
        dn = _v_norm(d)
        proj = [_v_dot(v, dn) for v in verts]
        mx = max(proj)
        tol = 1e-6 * max(1.0, abs(mx))
        sel = [i for i, p in enumerate(proj) if abs(p - mx) <= tol]
        if len(sel) != 5:
            raise RuntimeError(f"Додекаэдр: у грани {sel} не 5 вершин")
        faces.append(_order_cycle(verts, sel, dn))
    return faces


def create_dodecahedron(radius: float = 1.0) -> Polyhedron:
    """Додекаэдр (*бонус): 20 вершин, 12 пятиугольников."""
    verts: List[Vec3] = []

    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                verts.append((sx * 1.0, sy * 1.0, sz * 1.0))

    for s1 in (1, -1):
        for s2 in (1, -1):
            verts.append((0.0, s1 / PHI, s2 * PHI))
            verts.append((s1 / PHI, s2 * PHI, 0.0))
            verts.append((s2 * PHI, 0.0, s1 / PHI))

    faces = _faces_of_dodecahedron(verts)
    assert len(verts) == 20 and len(faces) == 12, "Ошибка построения додекаэдра"
    return _build("Додекаэдр", verts, faces, radius)



#  ФАБРИКА - то, чем будет пользоваться Миша

POLYHEDRA: Dict[str, callable] = {
    "Тетраэдр": create_tetrahedron,
    "Гексаэдр (куб)": create_hexahedron,
    "Октаэдр": create_octahedron,
    "Икосаэдр": create_icosahedron,
    "Додекаэдр": create_dodecahedron,
}

POLYHEDRA_KEYS: List[str] = list(POLYHEDRA.keys())


def create_polyhedron(name: str, radius: float = 1.0) -> Polyhedron:
    """Создаёт многогранник по имени из POLYHEDRA."""
    if name not in POLYHEDRA:
        raise KeyError(f"Неизвестный многогранник: {name!r}. "
                       f"Доступно: {POLYHEDRA_KEYS}")
    return POLYHEDRA[name](radius)