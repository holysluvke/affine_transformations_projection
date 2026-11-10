#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Загрузка, преобразование и сохранение многогранников Wavefront OBJ.

Запуск из корня рабочего каталога:
    python affine_transformations_projection/obj_viewer.py
    python affine_transformations_projection/obj_viewer.py path/to/model.obj

Используются geometry.py, transforms.py и интерфейс viewer.py. Зависимости —
только стандартная библиотека Python, включая tkinter.
Читаются вершины v и многоугольные грани f (v, v/vt, v//vn, v/vt/vn),
положительные и отрицательные индексы. Текстуры, материалы, заданные нормали,
линии и свободные поверхности не импортируются. При сохранении записываются
геометрия v/f и текущие координаты после преобразований. Подбор вида меняет
только камеру, координаты модели при этом остаются исходными.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from geometry import Point, Polygon, Polyhedron, create_polyhedron
from revolution import (
    AXES, MAX_SEGMENTS, MIN_SEGMENTS, PRESETS, build_revolution, parse_generatrix,
)
from viewer import (
    App, dot, mat_apply, mat_mul, newell_normal, projection_matrix,
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
                elif kind == "f":
                    if len(values) < 3:
                        raise ValueError("грань должна содержать минимум три вершины")
                    indices = []
                    for value in values:
                        index = int(value.split("/", 1)[0])
                        if index == 0:
                            raise ValueError("индексы OBJ начинаются с 1, индекс 0 недопустим")
                        resolved = index - 1 if index > 0 else len(vertices) + index
                        if index < 0 and not 0 <= resolved < len(vertices):
                            raise ValueError(f"вершина {index} отсутствует")
                        indices.append(resolved)
                    face_indices.append((number, indices))
            except (ValueError, OverflowError) as error:
                raise ValueError(f"Строка {number}: {error}") from error

    if not vertices or not face_indices:
        raise ValueError("В OBJ должны быть вершины v и многоугольные грани f")
    faces = []
    for number, indices in face_indices:
        if any(index >= len(vertices) for index in indices):
            raise ValueError(f"Строка {number}: индекс вершины выходит за границы модели")
        # Грани ссылаются на общие Point, поэтому transform() изменяет их вместе.
        faces.append(Polygon([vertices[index] for index in indices]))
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


class OBJApp(App):
    """Просмотрщик лабораторной №6 с файловыми операциями для OBJ."""

    def __init__(self, root):
        self.source_path = None
        self.source_note = "Встроенная модель"
        self.revolution_dialog = None
        self.original = None
        self.view_center = (0.0, 0.0, 0.0)
        self.view_radius = 1.0
        super().__init__(root)
        root.title("Wavefront OBJ — загрузка, преобразования и сохранение")
        self.original = self.poly.copy()
        self.fit_view()

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
        file_menu.add_separator()
        file_menu.add_command(label="Подобрать вид", accelerator="F", command=self.fit_view)
        file_menu.add_command(label="Вернуть исходную модель", command=self.restore_model)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.destroy)
        menu.add_cascade(label="Файл", menu=file_menu)
        self.root.config(menu=menu)
        self.root.bind("<Control-o>", lambda event: self.open_obj())
        self.root.bind("<Control-r>", lambda event: self.open_revolution())
        ttk.Button(self.top_panel, text="Открыть OBJ…", command=self.open_obj).grid(
            row=6, column=0, sticky="we", pady=(4, 0))
        ttk.Button(self.top_panel, text="Фигура вращения…", command=self.open_revolution).grid(
            row=6, column=1, sticky="we", pady=(4, 0))
        self.root.bind("<Control-O>", lambda event: self.open_obj())
        self.root.bind("<Control-Shift-S>", lambda event: self.save_as())
        # Клавиша работает на холсте и не перехватывает ввод в полях параметров.
        self.canvas.bind("<Key-f>", lambda event: self.fit_view())
        self.canvas.bind("<Key-F>", lambda event: self.fit_view())

    def _on_press(self, event):
        self.canvas.focus_set()
        super()._on_press(event)

    def show_model(self, poly, note):
        """Показать новую модель (загруженную или построенную) как исходную."""
        self.poly = poly
        self.original = poly.copy()
        self.source_path = None
        self.source_note = note
        self.fit_view()

    def open_revolution(self):
        if self.revolution_dialog is not None and self.revolution_dialog.window.winfo_exists():
            self.revolution_dialog.window.lift()
            return
        self.revolution_dialog = RevolutionDialog(self)

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
        self.source_path = Path(path)
        self.show_model(poly, str(self.source_path))

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
        self.poly = create_polyhedron(self.poly_name.get())
        self.original = self.poly.copy()
        self.source_path = None
        self.source_note = "Встроенная модель"
        self.fit_view()

    def restore_model(self):
        if self.original is not None:
            self.poly = self.original.copy()
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
        self.redraw()

    def fit_view(self):
        self.view_center = self.poly.center()
        self.view_radius = max(self.poly.radius(), 1e-12)
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

        if self.show_axes.get():
            self._draw_axes(to_view, to_screen)
        view = [to_view(point) for point in self.poly.vertex_array()]
        light_length = math.sqrt(1.34)
        light = (0.3 / light_length, 0.5 / light_length, 1 / light_length)
        faces = []
        for indices in self.poly.faces_as_indices():
            points = [view[index] for index in indices]
            screen = [to_screen(point) for point in points]
            if any(point is None for point in screen):
                continue
            normal = newell_normal(points)
            length = math.sqrt(dot(normal, normal))
            # Двусторонние грани: применимо к невыпуклым, открытым моделям
            # и отражениям. Не определяем нормаль по общему центру модели.
            shade = 0.3 + 0.7 * abs(dot(normal, light)) / length if length > 1e-12 else 0.3
            depth = sum(point[2] for point in points) / len(points)
            faces.append((depth, screen, shade))
        faces.sort(key=lambda face: face[0])
        for _, screen, shade in faces:
            color = "#%02x%02x%02x" % (int(70 * shade + 20), int(140 * shade + 30), int(225 * shade + 20))
            canvas.create_polygon(
                [coordinate for point in screen for coordinate in point],
                fill=color if self.filled.get() else "", outline="#1b2a41", width=1,
            )
        source = self.source_note
        self.status.set(
            f"{self.poly.name}: V={len(self.poly.vertices)}, F={len(self.poly.faces)}"
            + "\nЦентр: (%.3g, %.3g, %.3g)" % self.poly.center()
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
