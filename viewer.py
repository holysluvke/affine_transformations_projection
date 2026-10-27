#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer.py — ЧАСТЬ 3: отображение, проекции и интерфейс.

Запуск:  python viewer.py
Нужны рядом: geometry.py (человек 1) и transforms.py (человек 2).
Если transforms.py ещё нет — используются заглушки внутри этого файла (единичная матрица).

Какие функции ожидаются от transforms.py (углы в градусах, матрицы 4x4):
    translation(dx, dy, dz)
    rotation_x(a), rotation_y(a), rotation_z(a)
    scaling(sx, sy, sz)
    reflection(plane)                         # 'XY' | 'XZ' | 'YZ'
    scale_about_center(center, sx, sy, sz)    # center = poly.center()
    rotate_about_center_axis(center, axis, a) # axis = 'x' | 'y' | 'z'
    rotate_about_line(p1, p2, a)              # p1, p2 — кортежи (x, y, z)
Если у человека 2 имена другие — поправьте блок «АДАПТЕР» ниже.

Управление мышью: ЛКМ + движение — вращение камеры, колесо — масштаб.
"""

import math
import tkinter as tk
from tkinter import messagebox, ttk

from geometry import POLYHEDRA_KEYS, create_polyhedron

# ============================================================================
#  АДАПТЕР к модулю человека 2
# ============================================================================
try:
    import transforms as _tr

    _TR_SOURCE = "transforms.py"
except ImportError:
    _tr = None
    _TR_SOURCE = "ЗАГЛУШКИ (transforms.py не найден)"


def _stub(*args, **kwargs):
    """Заглушка: возвращает единичную матрицу (фигура не меняется)."""
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


# Заглушки для всех функций, которые должен дать человек 2.
# Когда появится transforms.py — функции берутся из него автоматически.
translation = _stub  # (dx, dy, dz)
rotation_x = _stub  # (angle_deg)
rotation_y = _stub
rotation_z = _stub
scaling = _stub  # (sx, sy, sz)
reflection = _stub  # (plane: 'XY' | 'XZ' | 'YZ')
scale_about_center = _stub  # (center, sx, sy, sz)
rotate_about_center_axis = _stub  # (center, axis, angle_deg)
rotate_about_line = _stub  # (p1, p2, angle_deg)


def _fn(name):
    """Функция из transforms.py, а если её там нет — заглушка."""
    if _tr is not None and hasattr(_tr, name):
        return getattr(_tr, name)
    return globals()[name]


# ============================================================================
#  Проекции
# ============================================================================
def mat_mul(a, b):
    """Произведение матриц 4x4: a @ b (сначала применяется b)."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
    ]


def mat_apply(m, p):
    """M @ (x, y, z, 1)^T. Возвращает однородный вектор (x, y, z, w) БЕЗ деления на w."""
    x, y, z = p[0], p[1], p[2]
    return tuple(m[i][0] * x + m[i][1] * y + m[i][2] * z + m[i][3] for i in range(4))


def view_matrix(yaw, pitch):
    """
    Матрица вида (поворот камеры): V = Rx(pitch) @ Ry(yaw).
    Сначала поворот вокруг Y (yaw), затем вокруг X (pitch). Радианы.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]]
    rx = [[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]]
    return mat_mul(rx, ry)


def projection_matrix(mode, dist):
    """
    Матрица проекции 4x4 (камера смотрит вдоль -Z).
      axon  — ортогональная (аксонометрия): обнуляем Z;
      persp — перспективная, центр проекции (0, 0, d), плоскость z = 0:
              w' = 1 - z/d, после деления на w: x' = x*d/(d - z).
    """
    if mode == "axon":
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, -1.0 / dist, 1]]


def screen_matrix(zoom, cx, cy):
    """Матрица перехода к экрану: масштаб, переворот оси Y, сдвиг в центр окна."""
    return [[zoom, 0, 0, cx], [0, -zoom, 0, cy], [0, 0, 1, 0], [0, 0, 0, 1]]


def newell_normal(pts):
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return (nx, ny, nz)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ============================================================================
#  Приложение
# ============================================================================
class App:
    PRESETS = {  # (yaw°, pitch°) по умолчанию
        "axon": (45.0, 35.264),  # изометрия
        "persp": (30.0, 20.0),
    }

    def __init__(self, root):
        self.root = root
        root.title("Многогранники: преобразования и проекции")
        root.geometry("1180x780")

        self.mode = tk.StringVar(value="persp")
        self.poly_name = tk.StringVar(value=POLYHEDRA_KEYS[0])
        self.filled = tk.BooleanVar(value=True)
        self.show_axes = tk.BooleanVar(value=True)
        self.dist = tk.DoubleVar(value=7.0)
        self.status = tk.StringVar()

        self.zoom = 110.0
        self.yaw, self.pitch = (math.radians(a) for a in self.PRESETS["persp"])
        self._drag = None

        self.poly = create_polyhedron(self.poly_name.get())
        self._build_ui()
        self.redraw()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        left = ttk.Frame(self.root, padding=6)
        left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
        self.canvas.pack(side="right", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind(
            "<MouseWheel>", lambda e: self._on_wheel(1 if e.delta > 0 else -1)
        )
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel(1))
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel(-1))

        # --- выбор многогранника и проекции
        top = ttk.LabelFrame(left, text="Многогранник и проекция", padding=6)
        top.pack(fill="x", pady=3)
        cb = ttk.Combobox(
            top,
            textvariable=self.poly_name,
            values=POLYHEDRA_KEYS,
            state="readonly",
            width=22,
        )
        cb.grid(row=0, column=0, columnspan=2, sticky="we")
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_polyhedron())
        ttk.Radiobutton(
            top,
            text="Перспективная",
            value="persp",
            variable=self.mode,
            command=self.on_mode,
        ).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(
            top,
            text="Аксонометрическая",
            value="axon",
            variable=self.mode,
            command=self.on_mode,
        ).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(
            top, text="Заливка граней", variable=self.filled, command=self.redraw
        ).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(
            top, text="Оси координат", variable=self.show_axes, command=self.redraw
        ).grid(row=2, column=1, sticky="w")
        ttk.Label(top, text="Камера (перспектива):").grid(row=3, column=0, sticky="w")
        ttk.Scale(
            top, from_=3.0, to=20.0, variable=self.dist, command=lambda v: self.redraw()
        ).grid(row=3, column=1, sticky="we")
        ttk.Button(top, text="Сбросить вид", command=self.reset_view).grid(
            row=4, column=0, sticky="we", pady=(4, 0)
        )
        ttk.Button(top, text="Сбросить фигуру", command=self.load_polyhedron).grid(
            row=4, column=1, sticky="we", pady=(4, 0)
        )

        # --- сдвиг
        self.v_move = self._section(
            left, "Перемещение", ["dx", "dy", "dz"], ["0", "0", "0"], self.apply_move
        )
        # --- поворот
        self.v_rot = self._section(
            left,
            "Поворот вокруг осей X, Y, Z (градусы)",
            ["X°", "Y°", "Z°"],
            ["0", "0", "0"],
            self.apply_rotate,
        )
        # --- масштаб
        self.v_scale = self._section(
            left,
            "Масштабирование",
            ["sx", "sy", "sz"],
            ["1", "1", "1"],
            self.apply_scale,
        )
        self.scale_center = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._last_frame, text="относительно центра", variable=self.scale_center
        ).grid(row=2, column=0, columnspan=8, sticky="w")
        # --- отражение
        refl = ttk.LabelFrame(left, text="Отражение относительно плоскости", padding=6)
        refl.pack(fill="x", pady=3)
        for i, pl in enumerate(("XY", "XZ", "YZ")):
            ttk.Button(
                refl, text=pl, width=8, command=lambda p=pl: self.apply_reflect(p)
            ).grid(row=0, column=i, padx=2)
        # --- вращение вокруг оси через центр
        ax = ttk.LabelFrame(
            left, text="Вращение вокруг оси через центр фигуры", padding=6
        )
        ax.pack(fill="x", pady=3)
        self.axis_var = tk.StringVar(value="y")
        for i, a in enumerate(("x", "y", "z")):
            ttk.Radiobutton(ax, text=a.upper(), value=a, variable=self.axis_var).grid(
                row=0, column=i
            )
        self.axis_angle = tk.StringVar(value="30")
        ttk.Label(ax, text="угол°").grid(row=0, column=3, padx=(8, 0))
        ttk.Entry(ax, textvariable=self.axis_angle, width=6).grid(row=0, column=4)
        ttk.Button(ax, text="Применить", command=self.apply_center_axis).grid(
            row=0, column=5, padx=6
        )
        # --- вращение вокруг произвольной прямой
        ln = ttk.LabelFrame(left, text="Вращение вокруг прямой (2 точки)", padding=6)
        ln.pack(fill="x", pady=3)
        self.line_vars = []
        for r, (name, defs) in enumerate(
            (("A", ("0", "0", "0")), ("B", ("1", "1", "1")))
        ):
            ttk.Label(ln, text=name).grid(row=r, column=0)
            for c, (lab, d) in enumerate(zip("xyz", defs)):
                v = tk.StringVar(value=d)
                self.line_vars.append(v)
                ttk.Label(ln, text=lab).grid(row=r, column=1 + 2 * c)
                ttk.Entry(ln, textvariable=v, width=5).grid(row=r, column=2 + 2 * c)
        self.line_angle = tk.StringVar(value="30")
        ttk.Label(ln, text="угол°").grid(row=2, column=0, columnspan=2, pady=(4, 0))
        ttk.Entry(ln, textvariable=self.line_angle, width=6).grid(
            row=2, column=2, pady=(4, 0)
        )
        ttk.Button(ln, text="Применить", command=self.apply_line).grid(
            row=2, column=3, columnspan=3, pady=(4, 0)
        )

        ttk.Label(
            left,
            textvariable=self.status,
            wraplength=300,
            foreground="#444",
            justify="left",
        ).pack(fill="x", pady=(8, 0))
        ttk.Label(left, text=f"Матрицы: {_TR_SOURCE}", foreground="#888").pack(
            anchor="w"
        )

    def _section(self, parent, title, labels, defaults, command):
        """Секция: подписи + поля ввода в строку и кнопка «Применить»."""
        fr = ttk.LabelFrame(parent, text=title, padding=6)
        fr.pack(fill="x", pady=3)
        vars_ = []
        for i, (lab, d) in enumerate(zip(labels, defaults)):
            v = tk.StringVar(value=d)
            vars_.append(v)
            ttk.Label(fr, text=lab).grid(row=0, column=2 * i, padx=(4 if i else 0, 1))
            ttk.Entry(fr, textvariable=v, width=6).grid(row=0, column=2 * i + 1)
        ttk.Button(fr, text="Применить", command=command).grid(
            row=1, column=0, columnspan=6, sticky="we", pady=(4, 0)
        )
        self._last_frame = fr
        return vars_

    # ------------------------------------------------------- ввод параметров
    @staticmethod
    def _num(var, name="значение"):
        try:
            return float(var.get().strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"Поле «{name}»: введите число")

    def _nums(self, vars_, names):
        return [self._num(v, n) for v, n in zip(vars_, names)]

    def _apply(self, build_matrix):
        """Строит матрицу (через transforms) и применяет к многограннику."""
        try:
            m = build_matrix()
        except (ValueError, ZeroDivisionError) as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        self.poly.transform(m)
        self.redraw()

    # --------------------------------------------------- преобразования
    def apply_move(self):
        self._apply(
            lambda: _fn("translation")(*self._nums(self.v_move, ["dx", "dy", "dz"]))
        )

    def apply_rotate(self):
        def build():
            ax, ay, az = self._nums(self.v_rot, ["X°", "Y°", "Z°"])
            # последовательно: X, затем Y, затем Z
            m = mat_mul(_fn("rotation_y")(ay), _fn("rotation_x")(ax))
            return mat_mul(_fn("rotation_z")(az), m)

        self._apply(build)

    def apply_scale(self):
        def build():
            sx, sy, sz = self._nums(self.v_scale, ["sx", "sy", "sz"])
            if self.scale_center.get():
                return _fn("scale_about_center")(self.poly.center(), sx, sy, sz)
            return _fn("scaling")(sx, sy, sz)

        self._apply(build)

    def apply_reflect(self, plane):
        self._apply(lambda: _fn("reflection")(plane))

    def apply_center_axis(self):
        self._apply(
            lambda: _fn("rotate_about_center_axis")(
                self.poly.center(),
                self.axis_var.get(),
                self._num(self.axis_angle, "угол"),
            )
        )

    def apply_line(self):
        def build():
            v = self._nums(self.line_vars, ["Ax", "Ay", "Az", "Bx", "By", "Bz"])
            return _fn("rotate_about_line")(
                tuple(v[:3]), tuple(v[3:]), self._num(self.line_angle, "угол")
            )

        self._apply(build)

    # ---------------------------------------------------- состояние / камера
    def load_polyhedron(self):
        self.poly = create_polyhedron(self.poly_name.get())
        self.redraw()

    def on_mode(self):
        self.reset_view()

    def reset_view(self):
        yaw, pitch = self.PRESETS[self.mode.get()]
        self.yaw, self.pitch = math.radians(yaw), math.radians(pitch)
        self.redraw()

    def _on_press(self, e):
        self._drag = (e.x, e.y)

    def _on_drag(self, e):
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.yaw += dx * 0.01
        self.pitch = max(-math.pi / 2, min(math.pi / 2, self.pitch + dy * 0.01))
        self.redraw()

    def _on_wheel(self, direction):
        self.zoom = max(
            20.0, min(600.0, self.zoom * (1.1 if direction > 0 else 1 / 1.1))
        )
        self.redraw()

    # ------------------------------------------------------------ рисование
    def redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        cx, cy = w / 2, h / 2
        mode = self.mode.get()
        dist = self.dist.get()

        # Конвейер целиком на матрицах: экран = Screen @ Proj @ View
        vmat = view_matrix(self.yaw, self.pitch)
        psmat = mat_mul(screen_matrix(self.zoom, cx, cy), projection_matrix(mode, dist))

        def to_view(p):
            return mat_apply(vmat, p)[:3]

        def to_screen(pt):
            x, y, z, w = mat_apply(psmat, pt)
            if mode == "persp" and w * dist < 0.05:  # точка за камерой / у камеры
                return None
            return (x / w, y / w)

        if self.show_axes.get():
            self._draw_axes(to_view, to_screen)

        poly = self.poly
        view = [to_view(p) for p in poly.vertex_array()]
        center_v = to_view(poly.center())
        cam = (0.0, 0.0, dist)
        light = (0.3, 0.5, 1.0)
        ln = math.sqrt(dot(light, light))
        light = (light[0] / ln, light[1] / ln, light[2] / ln)

        faces_draw = []
        for idx in poly.faces_as_indices():
            pts = [view[i] for i in idx]
            n = newell_normal(pts)
            nl = math.sqrt(dot(n, n))
            if nl < 1e-12:
                continue
            n = (n[0] / nl, n[1] / nl, n[2] / nl)
            fc = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
                sum(p[2] for p in pts) / len(pts),
            )
            # нормаль наружу (на случай отражений, меняющих порядок обхода)
            if (
                dot(n, (fc[0] - center_v[0], fc[1] - center_v[1], fc[2] - center_v[2]))
                < 0
            ):
                n = (-n[0], -n[1], -n[2])
            to_cam = (
                (0.0, 0.0, 1.0)
                if mode == "axon"
                else (cam[0] - fc[0], cam[1] - fc[1], cam[2] - fc[2])
            )
            if dot(n, to_cam) <= 0:  # нелицевая грань
                continue
            scr = [to_screen(p) for p in pts]
            if any(s is None for s in scr):
                continue
            shade = 0.3 + 0.7 * max(0.0, dot(n, light))
            faces_draw.append((fc[2], scr, shade))

        faces_draw.sort(key=lambda t: t[0])  # от дальних к ближним
        for _, scr, shade in faces_draw:
            flat = [v for s in scr for v in s]
            if self.filled.get():
                col = "#%02x%02x%02x" % (
                    int(70 * shade + 20),
                    int(140 * shade + 30),
                    int(225 * shade + 20),
                )
                c.create_polygon(flat, fill=col, outline="#1b2a41", width=1.5)
            else:
                c.create_polygon(flat, fill="", outline="#1b2a41", width=2)

        self.status.set(poly.info() + "\nЦентр: (%.2f, %.2f, %.2f)" % poly.center())

    def _draw_axes(self, to_view, to_screen):
        L = 2.6
        for end, color, label in (
            ((L, 0, 0), "#d33", "X"),
            ((0, L, 0), "#2a2", "Y"),
            ((0, 0, L), "#33d", "Z"),
        ):
            a = to_screen(to_view((0, 0, 0)))
            b = to_screen(to_view(end))
            if a is None or b is None:
                continue
            self.canvas.create_line(*a, *b, fill=color, width=1, dash=(4, 3))
            self.canvas.create_text(
                b[0] + 8, b[1] - 8, text=label, fill=color, font=("Arial", 11, "bold")
            )


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
