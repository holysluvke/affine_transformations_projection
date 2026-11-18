#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Загрузка, преобразование и сохранение многогранников Wavefront OBJ.

Запуск:
    python obj_viewer.py
    python obj_viewer.py path/to/model.obj

Используются geometry.py, transforms.py и интерфейс viewer.py. Зависимости —
только стандартная библиотека Python, включая tkinter.
Читаются вершины v и многоугольные грани f (v, v/vt, v//vn, v/vt/vn),
положительные и отрицательные индексы. Нормали vn сохраняются отдельно для
каждого угла грани; отсутствующие нормали вычисляются по смежным граням.
Текстуры, материалы, линии и свободные поверхности не импортируются. При сохранении записываются
геометрия v/f и текущие координаты после преобразований. Подбор вида меняет
только камеру, координаты модели при этом остаются исходными.
Нелицевые грани предварительно отсекаются по внешним нормалям и вектору
обзора; оставшиеся поверхности обрабатываются общим программным Z-буфером.
Демонстрация из трёх кубов позволяет преобразовывать каждый объект независимо.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from geometry import Point, Polygon, Polyhedron, create_polyhedron
import transforms
from zbuffer import render_scene
from backface_culling import classify_faces, filtered_model, visible_and_culled
from revolution import (
    AXES, MAX_SEGMENTS, MIN_SEGMENTS, PRESETS, build_revolution, parse_generatrix,
)
from surface import (
    MAX_SEGMENTS as SURF_MAX, MIN_SEGMENTS as SURF_MIN, PRESETS as SURF_PRESETS,
    build_surface,
)
from viewer import (
    App, mat_apply, mat_mul, projection_matrix,
    save_obj, screen_matrix, view_matrix,
)


def _obj_lines(stream):
    """Логические строки OBJ с учётом комментариев и продолжений через \\."""
    pending = ""
    start = 1
    for number, raw in enumerate(stream, 1):
        line = raw.partition("#")[0].strip()
        if not pending:
            start = number
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        yield start, pending + line
        pending = ""
    if pending:
        raise ValueError(f"Строка {start}: незавершённое продолжение строки")


def load_obj(path: str | Path) -> Polyhedron:
    """Прочитать полигональную геометрию OBJ; ошибки содержат номер строки."""
    path = Path(path)
    vertices = []
    # 9 лаба
    # Нормали OBJ хранятся отдельно от координат вершин.
    normals = []
    face_indices = []
    with path.open(encoding="utf-8-sig", errors="replace") as stream:
        for number, line in _obj_lines(stream):
            fields = line.split()
            if not fields:
                continue
            kind, values = fields[0], fields[1:]
            try:
                if kind == "v":
                    if len(values) < 3:
                        raise ValueError("у вершины нужны координаты x, y, z")
                    xyz = [float(value) for value in values[:3]]
                    if len(values) == 4:
                        weight = float(values[3])
                        if not math.isfinite(weight) or weight == 0:
                            raise ValueError("однородная координата w должна быть конечной и ненулевой")
                        xyz = [value / weight for value in xyz]
                    if not all(math.isfinite(value) for value in xyz):
                        raise ValueError("координаты должны быть конечными числами")
                    vertices.append(Point(*xyz))
                # 9 лаба
                # Читаем и нормализуем нормаль из записи vn.
                elif kind == "vn":
                    if len(values) != 3:
                        raise ValueError("у нормали нужны три координаты")
                    normal = tuple(float(value) for value in values)
                    if not all(math.isfinite(c) for c in normal):
                        raise ValueError("нормаль должна содержать конечные числа")
                    length = math.hypot(*normal)
                    if length == 0:
                        raise ValueError("нормаль не может быть нулевой")
                    normals.append(tuple(c / length for c in normal))
                elif kind == "f":
                    if len(values) < 3:
                        raise ValueError("грань должна содержать минимум три вершины")
                    indices = []
                    # 9 лаба
                    normal_indices = []
                    for value in values:
                        # 9 лаба
                        # В ссылках v/vt/vn и v//vn нормаль задаётся третьим индексом.
                        parts = value.split("/")
                        if len(parts) > 3:
                            raise ValueError(f"неверная ссылка вершины: {value}")
                        index = int(parts[0])
                        if index == 0:
                            raise ValueError("индексы OBJ начинаются с 1, индекс 0 недопустим")
                        resolved = index - 1 if index > 0 else len(vertices) + index
                        if index < 0 and not 0 <= resolved < len(vertices):
                            raise ValueError(f"вершина {index} отсутствует")
                        indices.append(resolved)
                        # 9 лаба
                        # Отрицательный индекс отсчитывается от уже прочитанных vn.
                        normal_index = None
                        if len(parts) == 3 and parts[2]:
                            raw = int(parts[2])
                            if raw == 0:
                                raise ValueError("индексы нормалей OBJ начинаются с 1")
                            normal_index = raw - 1 if raw > 0 else len(normals) + raw
                            if raw < 0 and not 0 <= normal_index < len(normals):
                                raise ValueError(f"нормаль {raw} отсутствует")
                        normal_indices.append(normal_index)
                    # 9 лаба
                    face_indices.append((number, indices, normal_indices))
            except (ValueError, OverflowError) as error:
                raise ValueError(f"Строка {number}: {error}") from error

    if not vertices or not face_indices:
        raise ValueError("В OBJ должны быть вершины v и многоугольные грани f")
    faces = []
    for number, indices, normal_indices in face_indices:  # 9 лаба
        if any(index >= len(vertices) for index in indices):
            raise ValueError(f"Строка {number}: индекс вершины выходит за границы модели")
        # Грани ссылаются на общие Point, поэтому transform() изменяет их вместе.
        # 9 лаба
        # Привязываем нормаль к углу грани, сохраняя общие позиции вершин.
        if any(i is not None and i >= len(normals) for i in normal_indices):
            raise ValueError(f"Строка {number}: индекс нормали выходит за границы модели")
        faces.append(Polygon([vertices[index] for index in indices],
                             [normals[i] if i is not None else None for i in normal_indices]))
    return Polyhedron(vertices, faces, name=path.stem)


class RevolutionDialog:
    """Окно задания образующей, оси и числа разбиений для фигуры вращения."""

    HINT = (
        "Одна точка в строке: x y z (z можно не писать).\n"
        "Ось проходит через начало координат. Для оси Y точка (x, y)\n"
        "означает: x — расстояние до оси, y — высота. Для оси X\n"
        "радиус задаёт y, для оси Z — x. Точки на оси — полюса."
    )

    def __init__(self, app):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Фигура вращения")
        self.window.transient(app.root)
        frame = ttk.Frame(self.window, padding=8)
        frame.pack(fill="both", expand=True)

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="Заготовка:").pack(side="left")
        self.preset = tk.StringVar(value="Ваза")
        box = ttk.Combobox(row, textvariable=self.preset, values=list(PRESETS),
                           state="readonly", width=18)
        box.pack(side="left", padx=6)
        box.bind("<<ComboboxSelected>>", lambda event: self.load_preset())

        ttk.Label(frame, text="Образующая (точки):").pack(anchor="w", pady=(8, 0))
        self.text = tk.Text(frame, width=34, height=11, font=("Courier", 10))
        self.text.pack(fill="both", expand=True)

        opts = ttk.Frame(frame)
        opts.pack(fill="x", pady=(8, 0))
        ttk.Label(opts, text="Ось:").grid(row=0, column=0, sticky="w")
        self.axis = tk.StringVar(value="y")
        for i, axis in enumerate(AXES):
            ttk.Radiobutton(opts, text=axis.upper(), value=axis,
                            variable=self.axis).grid(row=0, column=1 + i)
        ttk.Label(opts, text="Разбиений:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.segments = tk.StringVar(value="24")
        self.segments.trace_add("write", lambda *args: self.update_angle())
        ttk.Entry(opts, textvariable=self.segments, width=6).grid(
            row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))
        self.angle_label = ttk.Label(opts, text="")
        self.angle_label.grid(row=1, column=3, columnspan=3, sticky="w", pady=(6, 0))
        self.closed = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Замкнуть образующую (последняя точка → первая)",
                        variable=self.closed).grid(row=2, column=0, columnspan=6,
                                                   sticky="w", pady=(6, 0))

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Построить", command=self.build).pack(
            side="left", expand=True, fill="x")
        ttk.Button(buttons, text="Построить и сохранить в OBJ…",
                   command=self.build_and_save).pack(side="left", expand=True,
                                                     fill="x", padx=(6, 0))
        ttk.Label(frame, text=self.HINT, foreground="#555", justify="left").pack(
            anchor="w", pady=(8, 0))

        self.load_preset()
        self.update_angle()

    def load_preset(self):
        text, closed = PRESETS[self.preset.get()]
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.axis.set("y")  # заготовки рассчитаны на ось Y
        self.closed.set(closed)

    def update_angle(self):
        try:
            n = int(self.segments.get())
            self.angle_label.configure(text=f"угол шага = {360 / n:.4g}°")
        except (ValueError, ZeroDivisionError):
            self.angle_label.configure(text="")

    def build(self):
        """Построить модель и показать её в главном окне. Возвращает её или None."""
        try:
            segments = int(self.segments.get().strip())
        except ValueError:
            messagebox.showerror("Фигура вращения", "Число разбиений — целое число "
                                 f"от {MIN_SEGMENTS} до {MAX_SEGMENTS}", parent=self.window)
            return None
        try:
            poly = build_revolution(parse_generatrix(self.text.get("1.0", "end")),
                                    self.axis.get(), segments, self.closed.get())
        except ValueError as error:
            messagebox.showerror("Фигура вращения", str(error), parent=self.window)
            return None
        self.app.show_model(poly, "Фигура вращения (построена в программе)")
        return poly

    def build_and_save(self):
        if self.build() is not None:
            self.app.save_as()


class SurfaceDialog:
    """Окно задания функции z = f(x, y), диапазонов и числа разбиений."""

    HINT = (
        "Формула от x и y: + - * / ** (или ^), скобки, константы pi, e,\n"
        "функции sin, cos, tan, exp, log, sqrt, abs и др. из math.\n"
        "Десятичный разделитель — точка. Где функция не определена,\n"
        "в поверхности остаётся дыра."
    )

    def __init__(self, app):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("График функции двух переменных")
        self.window.transient(app.root)
        frame = ttk.Frame(self.window, padding=8)
        frame.pack(fill="both", expand=True)

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="Заготовка:").pack(side="left")
        self.preset = tk.StringVar(value="Волна")
        box = ttk.Combobox(row, textvariable=self.preset, values=list(SURF_PRESETS),
                           state="readonly", width=20)
        box.pack(side="left", padx=6)
        box.bind("<<ComboboxSelected>>", lambda event: self.load_preset())

        ttk.Label(frame, text="z = f(x, y) =").pack(anchor="w", pady=(8, 0))
        self.formula = tk.StringVar()
        ttk.Entry(frame, textvariable=self.formula, width=40).pack(fill="x")

        grid = ttk.Frame(frame)
        grid.pack(fill="x", pady=(8, 0))
        self.x0, self.x1 = tk.StringVar(), tk.StringVar()
        self.y0, self.y1 = tk.StringVar(), tk.StringVar()
        self.nx, self.ny = tk.StringVar(), tk.StringVar()
        ttk.Label(grid, text="x от").grid(row=0, column=0, sticky="e")
        ttk.Entry(grid, textvariable=self.x0, width=8).grid(row=0, column=1)
        ttk.Label(grid, text="до").grid(row=0, column=2)
        ttk.Entry(grid, textvariable=self.x1, width=8).grid(row=0, column=3)
        ttk.Label(grid, text="y от").grid(row=1, column=0, sticky="e", pady=(4, 0))
        self.y0_entry = ttk.Entry(grid, textvariable=self.y0, width=8)
        self.y0_entry.grid(row=1, column=1, pady=(4, 0))
        ttk.Label(grid, text="до").grid(row=1, column=2, pady=(4, 0))
        self.y1_entry = ttk.Entry(grid, textvariable=self.y1, width=8)
        self.y1_entry.grid(row=1, column=3, pady=(4, 0))
        ttk.Label(grid, text="разбиений X").grid(row=2, column=0, sticky="e", pady=(4, 0))
        ttk.Entry(grid, textvariable=self.nx, width=8).grid(row=2, column=1, pady=(4, 0))
        ttk.Label(grid, text="Y").grid(row=2, column=2, pady=(4, 0))
        self.ny_entry = ttk.Entry(grid, textvariable=self.ny, width=8)
        self.ny_entry.grid(row=2, column=3, pady=(4, 0))

        self.same = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Диапазон и разбиения по Y как по X",
                        variable=self.same, command=self.sync).pack(anchor="w", pady=(6, 0))
        for var in (self.x0, self.x1, self.nx):
            var.trace_add("write", lambda *args: self.sync())

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Построить", command=self.build).pack(
            side="left", expand=True, fill="x")
        ttk.Button(buttons, text="Построить и сохранить в OBJ…",
                   command=self.build_and_save).pack(side="left", expand=True,
                                                     fill="x", padx=(6, 0))
        ttk.Label(frame, text=self.HINT, foreground="#555", justify="left").pack(
            anchor="w", pady=(8, 0))
        self.load_preset()

    def sync(self):
        """При включённой галочке поля Y повторяют поля X и блокируются."""
        if self.same.get():
            self.y0.set(self.x0.get())
            self.y1.set(self.x1.get())
            self.ny.set(self.nx.get())
        state = "disabled" if self.same.get() else "normal"
        for entry in (self.y0_entry, self.y1_entry, self.ny_entry):
            entry.configure(state=state)

    def load_preset(self):
        formula, x0, x1, y0, y1, n = SURF_PRESETS[self.preset.get()]
        self.formula.set(formula)
        for var, value in ((self.x0, x0), (self.x1, x1), (self.y0, y0),
                           (self.y1, y1), (self.nx, n), (self.ny, n)):
            var.set(str(value))
        self.same.set(x0 == y0 and x1 == y1)
        self.sync()

    @staticmethod
    def _number(var, name):
        try:
            return float(var.get().strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"Поле «{name}»: введите число") from None

    @staticmethod
    def _integer(var, name):
        try:
            return int(var.get().strip())
        except ValueError:
            raise ValueError(f"Поле «{name}»: целое число от {SURF_MIN} до {SURF_MAX}") from None

    def build(self):
        """Построить поверхность и показать её в главном окне. Возвращает её или None."""
        try:
            poly = build_surface(
                self.formula.get(),
                (self._number(self.x0, "x от"), self._number(self.x1, "x до")),
                (self._number(self.y0, "y от"), self._number(self.y1, "y до")),
                self._integer(self.nx, "разбиений X"),
                self._integer(self.ny, "разбиений Y"),
            )
        except (ValueError, OverflowError) as error:
            messagebox.showerror("График функции", str(error), parent=self.window)
            return None
        self.app.show_model(poly, "График функции (построен в программе)")
        return poly

    def build_and_save(self):
        if self.build() is not None:
            self.app.save_as()


class OBJApp(App):
    """Просмотрщик лабораторной №6 с файловыми операциями для OBJ."""

    def __init__(self, root):
        self.models = []
        self.originals = []
        self.active_index = 0
        self.view_azimuth = tk.DoubleVar(root, value=30.0)
        self.view_elevation = tk.DoubleVar(root, value=20.0)
        self.backface_enabled = tk.BooleanVar(root, value=True)
        self.show_culled_faces = tk.BooleanVar(root, value=False)
        self.show_face_normals = tk.BooleanVar(root, value=False)
        self._render_image = None
        self.source_path = None
        self.source_note = "Встроенная модель"
        self.revolution_dialog = None
        self.surface_dialog = None
        self.original = None
        self.view_center = (0.0, 0.0, 0.0)
        self.view_radius = 1.0
        super().__init__(root)
        root.title("Wavefront OBJ — отсечение граней, Z-буфер и преобразования")
        self.show_model(self.poly, "Встроенная модель")

    def _build_ui(self):
        super()._build_ui()
        def bind_reset(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton" and child.cget("text") == "Сбросить фигуру":
                    child.configure(command=self.restore_model)
                bind_reset(child)
        bind_reset(self.root)
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Открыть OBJ…", accelerator="Ctrl+O", command=self.open_obj)
        file_menu.add_command(label="Сохранить OBJ как…", accelerator="Ctrl+Shift+S", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Фигура вращения…", accelerator="Ctrl+R", command=self.open_revolution)
        file_menu.add_command(label="График функции z=f(x,y)…", accelerator="Ctrl+G", command=self.open_surface)
        file_menu.add_command(label="Демонстрация Z-буфера: три куба", command=self.load_zbuffer_demo)
        file_menu.add_separator()
        file_menu.add_command(label="Подобрать вид", accelerator="F", command=self.fit_view)
        file_menu.add_command(label="Вернуть исходную модель", command=self.restore_model)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.destroy)
        menu.add_cascade(label="Файл", menu=file_menu)
        self.root.config(menu=menu)
        self.root.bind("<Control-o>", lambda event: self.open_obj())
        self.root.bind("<Control-r>", lambda event: self.open_revolution())
        self.root.bind("<Control-g>", lambda event: self.open_surface())
        ttk.Button(self.top_panel, text="Открыть OBJ…", command=self.open_obj).grid(
            row=6, column=0, sticky="we", pady=(4, 0))
        ttk.Button(self.top_panel, text="Фигура вращения…", command=self.open_revolution).grid(
            row=6, column=1, sticky="we", pady=(4, 0))
        ttk.Button(self.top_panel, text="График z=f(x,y)…", command=self.open_surface).grid(
            row=7, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Button(self.top_panel, text="Z-буфер: три куба", command=self.load_zbuffer_demo).grid(
            row=8, column=0, sticky="we", pady=(4, 0))
        self.object_box = ttk.Combobox(self.top_panel, state="readonly", width=22)
        self.object_box.grid(row=8, column=1, sticky="we", pady=(4, 0))
        self.object_box.bind("<<ComboboxSelected>>", self.select_object)

        ttk.Label(self.top_panel, text="Вектор обзора · азимут (°)").grid(
            row=9, column=0, sticky="w", pady=(7, 0))
        ttk.Scale(
            self.top_panel, from_=-180, to=180, variable=self.view_azimuth,
            command=self._on_view_vector_change,
        ).grid(row=9, column=1, sticky="we", pady=(7, 0))
        ttk.Label(self.top_panel, text="Вектор обзора · наклон (°)").grid(
            row=10, column=0, sticky="w")
        ttk.Scale(
            self.top_panel, from_=-89, to=89, variable=self.view_elevation,
            command=self._on_view_vector_change,
        ).grid(row=10, column=1, sticky="we")
        ttk.Checkbutton(
            self.top_panel, text="Отсечение нелицевых граней",
            variable=self.backface_enabled, command=self.redraw,
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            self.top_panel, text="Показать удалённые грани (красный пунктир)",
            variable=self.show_culled_faces, command=self.redraw,
        ).grid(row=12, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            self.top_panel, text="Показать нормали лицевых граней",
            variable=self.show_face_normals, command=self.redraw,
        ).grid(row=13, column=0, columnspan=2, sticky="w")
        self.root.bind("<Control-O>", lambda event: self.open_obj())
        self.root.bind("<Control-Shift-S>", lambda event: self.save_as())
        # Клавиша работает на холсте и не перехватывает ввод в полях параметров.
        self.canvas.bind("<Key-f>", lambda event: self.fit_view())
        self.canvas.bind("<Key-F>", lambda event: self.fit_view())

    def _on_press(self, event):
        self.canvas.focus_set()
        super()._on_press(event)

    def _sync_view_controls(self):
        """Синхронизировать слайдеры с текущими углами камеры."""
        azimuth = (math.degrees(self.yaw) + 180.0) % 360.0 - 180.0
        self.yaw = math.radians(azimuth)
        self.view_azimuth.set(azimuth)
        self.view_elevation.set(math.degrees(self.pitch))

    def _on_view_vector_change(self, _value=None):
        """Изменить направление взгляда по двум углам камеры."""
        self.yaw = math.radians(float(self.view_azimuth.get()))
        self.pitch = math.radians(max(-89.0, min(89.0, float(self.view_elevation.get()))))
        self.redraw()

    def reset_view(self):
        yaw, pitch = self.PRESETS[self.mode.get()]
        self.yaw, self.pitch = math.radians(yaw), math.radians(pitch)
        self._sync_view_controls()
        self.redraw()

    def _on_drag(self, event):
        """Перетаскивание мышью по-прежнему вращает камеру и обновляет слайдеры."""
        if self._drag is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self.yaw += dx * 0.01
        limit = math.radians(89.0)
        self.pitch = max(-limit, min(limit, self.pitch + dy * 0.01))
        self._sync_view_controls()
        self.redraw()

    @staticmethod
    def _clip_segment_to_near(a, b, distance):
        """Обрезать отрезок в координатах камеры той же ближней плоскостью."""
        limit = distance - 0.05
        a_behind, b_behind = a[2] > limit, b[2] > limit
        if a_behind and b_behind:
            return None
        if a_behind != b_behind:
            t = (limit - a[2]) / (b[2] - a[2])
            intersection = tuple(a[i] + t * (b[i] - a[i]) for i in range(3))
            if a_behind:
                a = intersection
            else:
                b = intersection
        return a, b

    def _draw_culled_faces(self, face_views, to_screen, mode, distance):
        """Показать удалённые из рендера грани пунктирными красными рёбрами."""
        for item in face_views:
            if item.visible:
                continue
            points = item.points
            for index, point_a in enumerate(points):
                point_b = points[(index + 1) % len(points)]
                if mode == "persp":
                    clipped = self._clip_segment_to_near(point_a, point_b, distance)
                    if clipped is None:
                        continue
                    point_a, point_b = clipped
                screen_a, screen_b = to_screen(point_a), to_screen(point_b)
                if screen_a is None or screen_b is None:
                    continue
                self.canvas.create_line(
                    *screen_a, *screen_b, fill="#d62728", width=2,
                    dash=(5, 3),
                )

    def _draw_face_normals(self, face_views, to_screen, view_radius):
        """Нарисовать синие стрелки нормалей для оставленных граней выбранного объекта."""
        length = max(0.035, 0.16 * self.poly.radius() / max(view_radius, 1e-12))
        for item in face_views:
            if not item.visible:
                continue
            tip = tuple(item.centroid[i] + item.normal[i] * length for i in range(3))
            start_screen, end_screen = to_screen(item.centroid), to_screen(tip)
            if start_screen is None or end_screen is None:
                continue
            self.canvas.create_line(
                *start_screen, *end_screen, fill="#0877bd", width=2,
                arrow=tk.LAST,
            )

    def show_model(self, poly, note):
        """Показать новую модель (загруженную или построенную) как исходную."""
        self.poly = poly
        self.original = poly.copy()
        self.models = [poly]
        self.originals = [self.original]
        self.active_index = 0
        self._update_object_box()
        self.source_path = None
        self.source_note = note
        self.fit_view()

    def _update_object_box(self):
        self.object_box.configure(values=[f"{i + 1}: {p.name}" for i, p in enumerate(self.models)])
        self.object_box.current(self.active_index)

    def select_object(self, event=None):
        """Все элементы управления преобразованиями действуют только на выбранный объект."""
        self.active_index = self.object_box.current()
        self.poly = self.models[self.active_index]
        self.original = self.originals[self.active_index]
        self.redraw()

    def load_zbuffer_demo(self):
        """Для пересекающихся выпуклых объектов нужна общая попиксельная проверка глубины."""
        poses = [((-0.55, 0.15, 0.15), (15, 25, 10)),
                 ((0.55, 0.15, -0.15), (-20, -30, -15)),
                 ((0.0, -0.65, 0.3), (35, 10, 30))]
        self.models = []
        for i, (position, angles) in enumerate(poses):
            poly = create_polyhedron("Гексаэдр (куб)", radius=1.25)
            poly.name = ("Синий куб", "Оранжевый куб", "Зелёный куб")[i]
            for rotate, angle in zip((transforms.rotation_x, transforms.rotation_y,
                                      transforms.rotation_z), angles):
                poly.transform(rotate(angle))
            poly.transform(transforms.translation(*position))
            self.models.append(poly)
        self.originals = [p.copy() for p in self.models]
        self.active_index = 0
        self.poly, self.original = self.models[0], self.originals[0]
        self.source_path = None
        self.source_note = "Три пересекающихся куба. Преобразования — к выбранному объекту."
        self.filled.set(True)
        self._update_object_box()
        self.fit_view()

    def open_revolution(self):
        if self.revolution_dialog is not None and self.revolution_dialog.window.winfo_exists():
            self.revolution_dialog.window.lift()
            return
        self.revolution_dialog = RevolutionDialog(self)

    def open_surface(self):
        if self.surface_dialog is not None and self.surface_dialog.window.winfo_exists():
            self.surface_dialog.window.lift()
            return
        self.surface_dialog = SurfaceDialog(self)

    def open_obj(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.root, title="Открыть модель Wavefront OBJ",
                filetypes=[("Wavefront OBJ", "*.obj"), ("Все файлы", "*.*")],
            )
        if not path:
            return
        try:
            poly = load_obj(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка загрузки OBJ", str(error), parent=self.root)
            return
        self.show_model(poly, str(Path(path)))
        self.source_path = Path(path)

    def save_as(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Сохранить текущую модель Wavefront OBJ",
            defaultextension=".obj",
            initialfile=(self.source_path.stem if self.source_path else self.poly.name) + "_transformed.obj",
            filetypes=[("Wavefront OBJ", "*.obj")],
        )
        if not path:
            return
        try:
            save_obj(self.poly, path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка сохранения OBJ", str(error), parent=self.root)
            return
        self.status.set(self.status.get() + f"\nСохранено: {path}")

    def load_polyhedron(self):
        # Выбор встроенной фигуры в списке начинает новую модель.
        self.show_model(create_polyhedron(self.poly_name.get()), "Встроенная модель")

    def restore_model(self):
        if self.original is not None:
            self.poly = self.original.copy()
            self.models[self.active_index] = self.poly
            self.fit_view()

    @staticmethod
    def _num(var, name="значение"):
        value = App._num(var, name)
        if not math.isfinite(value):
            raise ValueError(f"Поле «{name}»: введите конечное число")
        return value

    def _apply(self, build_matrix):
        # Проверяем результат на копии, чтобы ошибка не портила текущую модель.
        try:
            poly = self.poly.transformed(build_matrix())
            if not all(math.isfinite(c) for p in poly.vertices for c in p.coords()):
                raise ValueError("Результат преобразования выходит за диапазон чисел")
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            messagebox.showerror("Ошибка преобразования", str(error), parent=self.root)
            return
        self.poly = poly
        self.models[self.active_index] = poly
        self.redraw()

    def fit_view(self):
        vertices = [p.coords() for model in (self.models or [self.poly]) for p in model.vertices]
        lo = [min(p[i] for p in vertices) for i in range(3)]
        hi = [max(p[i] for p in vertices) for i in range(3)]
        self.view_center = tuple((lo[i] + hi[i]) / 2 for i in range(3))
        self.view_radius = max(max(math.dist(p, self.view_center) for p in vertices), 1e-12)
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 10 or height < 10:
            width = height = 600
        self.zoom = min(width, height) * 0.3
        self.redraw()

    def redraw(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 10 or height < 10:
            return
        mode, dist = self.mode.get(), self.dist.get()
        rotation = view_matrix(self.yaw, self.pitch)
        projection = mat_mul(
            screen_matrix(self.zoom, width / 2, height / 2),
            projection_matrix(mode, dist),
        )

        def to_view(point):
            normalized = tuple((point[i] - self.view_center[i]) / self.view_radius for i in range(3))
            return mat_apply(rotation, normalized)[:3]

        def to_screen(point):
            x, y, z, w = mat_apply(projection, point)
            if mode == "persp" and w * dist < 0.05:
                return None
            return x / w, y / w

        scene_models = self.models or [self.poly]
        render_models = []
        face_views_by_model = []
        counts = []
        for model in scene_models:
            face_views = classify_faces(model, to_view, mode, dist)
            visible_indices, culled_indices = visible_and_culled(face_views)
            face_views_by_model.append(face_views)
            counts.append((len(visible_indices), len(culled_indices), len(model.faces)))
            render_models.append(
                filtered_model(model, visible_indices)
                if self.backface_enabled.get() else model
            )

        # Важно: Z-буфер получает только отфильтрованные модели. Его реализация
        # и проверка глубины остаются без изменений.
        buffer = render_scene(render_models, to_view, width, height,
                              self.zoom, mode, dist, self.filled.get())
        self._render_image = tk.PhotoImage(master=self.root, data=buffer.ppm(), format="PPM")
        canvas.create_image(0, 0, anchor="nw", image=self._render_image)
        if self.show_axes.get():
            # Координатные оси — вспомогательная разметка поверх изображения сцены.
            self._draw_axes(to_view, to_screen)
        if self.backface_enabled.get() and self.show_culled_faces.get():
            for face_views in face_views_by_model:
                self._draw_culled_faces(face_views, to_screen, mode, dist)
            canvas.create_text(
                width - 12, 14, anchor="ne",
                text="Красный пунктир — отсечённые грани",
                fill="#d62728", font=("Arial", 10, "bold"),
            )
        if self.show_face_normals.get():
            # Стрелки показываются только для выбранного объекта, чтобы не
            # перегружать сцену при демонстрации нескольких тел.
            self._draw_face_normals(
                face_views_by_model[self.active_index], to_screen, self.view_radius
            )
        source = self.source_note
        visible_total = sum(item[0] for item in counts)
        culled_total = sum(item[1] for item in counts)
        if self.backface_enabled.get():
            culling_note = f"Отсечение: вкл. · оставлено {visible_total}, удалено {culled_total}"
        else:
            culling_note = f"Отсечение: выкл. · было бы удалено {culled_total}"
        azimuth, elevation = math.radians(self.view_azimuth.get()), math.radians(self.view_elevation.get())
        view_vector = (
            math.sin(azimuth) * math.cos(elevation),
            -math.sin(elevation),
            -math.cos(azimuth) * math.cos(elevation),
        )
        self.status.set(
            f"Z-буфер · Объектов: {len(self.models) or 1}\n"
            + f"Выбран: {self.poly.name}: V={len(self.poly.vertices)}, F={len(self.poly.faces)}"
            + "\nЦентр: (%.3g, %.3g, %.3g)" % self.poly.center()
            + f"\n{culling_note}"
            + "\nВектор обзора: (%.3f, %.3f, %.3f)" % view_vector
            + f"\n{source}\nЛКМ — камера; колесо — масштаб.\nФайл → Подобрать вид: показать всю модель."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="путь к модели Wavefront OBJ")
    args = parser.parse_args()
    root = tk.Tk()
    app = OBJApp(root)
    if args.model:
        root.after_idle(lambda: app.open_obj(args.model))
    root.mainloop()


if __name__ == "__main__":
    main()
