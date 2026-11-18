import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from camera import Camera, render_camera
from obj_viewer import OBJApp

MIN_DISTANCE, MAX_DISTANCE = 0.2, 15.0  # расстояние камеры, в радиусах сцены
ORBIT_DELAY_MS = 10  # пауза между кадрами облёта


def _clamp(value, low, high):
    return max(low, min(high, value))


def _wrap_degrees(angle):
    """Привести угол к диапазону [-180, 180)"""
    return (angle + 180.0) % 360.0 - 180.0


class CameraApp(OBJApp):
    def __init__(self, root):
        # конструктор родителя вызывает redraw() сам.
        self.camera = Camera(fov=50.0)
        self.camera_mode = tk.BooleanVar(root, value=True)
        self.cam_azimuth = tk.DoubleVar(root, value=30.0)
        self.cam_elevation = tk.DoubleVar(root, value=20.0)
        self.cam_distance = tk.DoubleVar(root, value=3.0)  # в радиусах сцены
        self.cam_fov = tk.DoubleVar(root, value=50.0)
        self.cam_speed = tk.DoubleVar(root, value=3.0)  # градусов за кадр
        self._suspend_redraw = False
        self._orbiting = False
        self._orbit_job = None
        self.camera_info = tk.StringVar(root)
        super().__init__(root)
        root.title("Камера — облёт вокруг объекта")

    def _build_ui(self):
        super()._build_ui()  # сначала строим родительское окно
        panel = ttk.LabelFrame(self.top_panel.master, text="Камера (орбита вокруг объекта)",
                               padding=6)
        panel.pack(fill="x", pady=3, after=self.top_panel)  # сразу под верхней панелью
        panel.columnconfigure(1, weight=1)
        ttk.Checkbutton(panel, text="Режим камеры (перспектива)", variable=self.camera_mode,
                        command=self.redraw).grid(row=0, column=0, columnspan=2, sticky="w")
        sliders = (
            ("Азимут (°)", self.cam_azimuth, -180, 180),
            ("Высота (°)", self.cam_elevation, -89, 89),
            ("Расстояние (R)", self.cam_distance, MIN_DISTANCE, MAX_DISTANCE),
            ("Угол обзора FOV (°)", self.cam_fov, 20, 120),
        )
        for row, (label, variable, low, high) in enumerate(sliders, start=1):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w")
            ttk.Scale(panel, from_=low, to=high, variable=variable,
                      command=lambda _v: self.redraw()).grid(row=row, column=1, sticky="we")
        self.orbit_button = ttk.Button(panel, text="Облёт", command=self.toggle_orbit)
        self.orbit_button.grid(row=5, column=0, sticky="we", pady=(4, 0))
        speed = ttk.Frame(panel)
        speed.grid(row=5, column=1, sticky="we", pady=(4, 0))
        ttk.Label(speed, text="°/кадр").pack(side="left", padx=(6, 2))
        ttk.Scale(speed, from_=0.5, to=10.0, variable=self.cam_speed).pack(
            side="left", fill="x", expand=True)
        ttk.Button(panel, text="Сбросить камеру", command=self.reset_view).grid(
            row=6, column=0, sticky="we", pady=(4, 0))
        ttk.Button(panel, text="Сохранить кадр…", command=self.save_frame).grid(
            row=6, column=1, sticky="we", pady=(4, 0), padx=(4, 0))
        ttk.Label(panel, textvariable=self.camera_info, font=("Courier", 8),
                  justify="left").grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _reset_camera(self):
        """Вернуть камеру на (30°, 20°) и на расстояние, где вся сцена в кадре"""
        self.cam_azimuth.set(30.0)
        self.cam_elevation.set(20.0)
        self.camera.fov = float(self.cam_fov.get())
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        aspect = w / h if w >= 10 and h >= 10 else 1.0
        fit = self.camera.distance_to_fit(1.0, aspect) * 1.15  # 15% запаса по краям
        self.cam_distance.set(_clamp(fit, MIN_DISTANCE, MAX_DISTANCE))

    def fit_view(self):
        # OBJApp.fit_view сам вызывает redraw()
        self._suspend_redraw = True
        try:
            super().fit_view()  # считает view_center и view_radius сцены
        finally:
            self._suspend_redraw = False
        self._reset_camera()
        self.redraw()

    def reset_view(self):
        if not self.camera_mode.get():
            super().reset_view()
            return
        self._reset_camera()
        self.redraw()

    def _place_camera(self):
        """Параметры камеры из ползунков, объект стоит, двигается камера"""
        radius = self.view_radius
        distance = self.cam_distance.get() * radius
        self.camera.fov = float(self.cam_fov.get())
        self.camera.near = 0.01 * radius  # почти вплотную, но не ноль
        self.camera.far = distance + 10.0 * radius  # дальше любой точки сцены
        self.camera.orbit_around(self.view_center, self.cam_azimuth.get(),
                                 self.cam_elevation.get(), distance)

    def _on_drag(self, event):
        if not self.camera_mode.get():
            super()._on_drag(event)
            return
        if self._drag is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        # тянем вправо и объект едет вправо, значит камера уходит влево
        self.cam_azimuth.set(_wrap_degrees(self.cam_azimuth.get() - dx * 0.4))
        self.cam_elevation.set(_clamp(self.cam_elevation.get() + dy * 0.4, -89.0, 89.0))
        self.redraw()

    def _on_wheel(self, direction):
        if not self.camera_mode.get():
            super()._on_wheel(direction)
            return
        factor = 1 / 1.1 if direction > 0 else 1.1  # колесо от себя: приближение
        self.cam_distance.set(_clamp(self.cam_distance.get() * factor,
                                     MIN_DISTANCE, MAX_DISTANCE))
        self.redraw()

    # ---------- облёт ----------
    def toggle_orbit(self):
        if self._orbiting:
            self._orbiting = False
            if self._orbit_job is not None:
                self.root.after_cancel(self._orbit_job)
                self._orbit_job = None
            self.orbit_button.configure(text="Облёт start")
            return
        self.camera_mode.set(True)
        self._orbiting = True
        self.orbit_button.configure(text="Облёт stop")
        self._orbit_step()

    def _orbit_step(self):
        self._orbit_job = None
        if not self._orbiting:
            return
        self.cam_azimuth.set(_wrap_degrees(self.cam_azimuth.get() + self.cam_speed.get()))
        self.redraw()
        # кадры не копятся, если рендер медленный
        self._orbit_job = self.root.after(ORBIT_DELAY_MS, self._orbit_step)


    def save_frame(self):
        """Сохранить чистый кадр камеры (без осей и стрелок поверх)."""
        if self._render_image is None or not self.camera_mode.get():
            messagebox.showinfo("Сохранить кадр", "Включите режим камеры.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Сохранить кадр камеры", defaultextension=".png",
            initialfile=f"{self.poly.name}_camera.png",
            filetypes=[("PNG", "*.png"), ("PPM", "*.ppm")])
        if not path:
            return
        try:
            self._render_image.write(path, format="ppm" if path.lower().endswith(".ppm") else "png")
        except (tk.TclError, OSError) as error:
            messagebox.showerror("Ошибка сохранения кадра", str(error), parent=self.root)

    def _segment(self, a, b, width, height):
        """Отрезок в координатах камеры -> ((x0, y0), (x1, y1)) или None."""
        projected = self.camera.project_view_segment(a, b, width, height)
        if projected is None:
            return None
        return projected[0][:2], projected[1][:2]

    def _draw_camera_axes(self, width, height):
        length = 2.6 * self.view_radius
        origin = self.camera.to_view((0.0, 0.0, 0.0))
        for end, color, label in (((length, 0, 0), "#d33", "X"),
                                  ((0, length, 0), "#2a2", "Y"),
                                  ((0, 0, length), "#33d", "Z")):
            segment = self._segment(origin, self.camera.to_view(end), width, height)
            if segment is None:
                continue
            (x0, y0), (x1, y1) = segment
            self.canvas.create_line(x0, y0, x1, y1, fill=color, dash=(4, 3))
            self.canvas.create_text(x1 + 8, y1 - 8, text=label, fill=color,
                                    font=("Arial", 11, "bold"))

    def _draw_camera_culled(self, face_views, width, height):
        for item in face_views:
            if item.visible:
                continue
            points = item.points
            for a, b in zip(points, points[1:] + points[:1]):
                segment = self._segment(a, b, width, height)
                if segment is not None:
                    (x0, y0), (x1, y1) = segment
                    self.canvas.create_line(x0, y0, x1, y1, fill="#d62728", width=2,
                                            dash=(5, 3))

    def _draw_camera_normals(self, face_views, width, height):
        length = 0.16 * max(self.poly.radius(), 1e-12)       # длина стрелки в мировых единицах
        for item in face_views:
            if not item.visible:
                continue
            tip = tuple(item.centroid[i] + item.normal[i] * length for i in range(3))
            segment = self._segment(item.centroid, tip, width, height)
            if segment is not None:
                (x0, y0), (x1, y1) = segment
                self.canvas.create_line(x0, y0, x1, y1, fill="#0877bd", width=2, arrow=tk.LAST)
    def _update_status(self, result, aspect):
        cam = self.camera
        if self.backface_enabled.get():
            culling = f"Отсечение: вкл. · оставлено {result.visible}, удалено {result.culled}"
        else:
            culling = f"Отсечение: выкл. · было бы удалено {result.culled}"
        self.status.set(f"Камера · Z-буфер · Выбран: {self.poly.name}\n{culling}\n"
                        "ЛКМ — орбита; колесо — приближение.")
        rows = "\n".join("[" + " ".join(f"{v:7.3f}" for v in row) + "]"
                         for row in cam.projection_matrix(aspect))
        self.camera_info.set(
            "Положение: (%.2f, %.2f, %.2f)\n" % cam.position
            + "Направление: (%.3f, %.3f, %.3f)\n" % cam.direction
            + f"FOV {cam.fov:.0f}° · near {cam.near:.3g} · far {cam.far:.3g}\n"
            + "Матрица проекции P:\n" + rows)


    def save_frame(self):
        """Сохранить чистый кадр камеры (без осей и стрелок поверх)."""
        if self._render_image is None or not self.camera_mode.get():
            messagebox.showinfo("Сохранить кадр", "Включите режим камеры.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Сохранить кадр камеры", defaultextension=".png",
            initialfile=f"{self.poly.name}_camera.png",
            filetypes=[("PNG", "*.png"), ("PPM", "*.ppm")])
        if not path:
            return
        try:
            self._render_image.write(path, format="ppm" if path.lower().endswith(".ppm") else "png")
        except (tk.TclError, OSError) as error:
            messagebox.showerror("Ошибка сохранения кадра", str(error), parent=self.root)

    def _segment(self, a, b, width, height):
        """Отрезок в координатах камеры -> ((x0, y0), (x1, y1)) или None."""
        projected = self.camera.project_view_segment(a, b, width, height)
        if projected is None:
            return None
        return projected[0][:2], projected[1][:2]

    def _draw_camera_axes(self, width, height):
        length = 2.6 * self.view_radius
        origin = self.camera.to_view((0.0, 0.0, 0.0))
        for end, color, label in (((length, 0, 0), "#d33", "X"),
                                  ((0, length, 0), "#2a2", "Y"),
                                  ((0, 0, length), "#33d", "Z")):
            segment = self._segment(origin, self.camera.to_view(end), width, height)
            if segment is None:
                continue
            (x0, y0), (x1, y1) = segment
            self.canvas.create_line(x0, y0, x1, y1, fill=color, dash=(4, 3))
            self.canvas.create_text(x1 + 8, y1 - 8, text=label, fill=color,
                                    font=("Arial", 11, "bold"))

    def _draw_camera_culled(self, face_views, width, height):
        for item in face_views:
            if item.visible:
                continue
            points = item.points
            for a, b in zip(points, points[1:] + points[:1]):
                segment = self._segment(a, b, width, height)
                if segment is not None:
                    (x0, y0), (x1, y1) = segment
                    self.canvas.create_line(x0, y0, x1, y1, fill="#d62728", width=2,
                                            dash=(5, 3))

    def _draw_camera_normals(self, face_views, width, height):
        length = 0.16 * max(self.poly.radius(), 1e-12)       # длина стрелки в мировых единицах
        for item in face_views:
            if not item.visible:
                continue
            tip = tuple(item.centroid[i] + item.normal[i] * length for i in range(3))
            segment = self._segment(item.centroid, tip, width, height)
            if segment is not None:
                (x0, y0), (x1, y1) = segment
                self.canvas.create_line(x0, y0, x1, y1, fill="#0877bd", width=2, arrow=tk.LAST)

    def _update_status(self, result, aspect):
        cam = self.camera
        if self.backface_enabled.get():
            culling = f"Отсечение: вкл. · оставлено {result.visible}, удалено {result.culled}"
        else:
            culling = f"Отсечение: выкл. · было бы удалено {result.culled}"
        self.status.set(f"Камера · Z-буфер · Выбран: {self.poly.name}\n{culling}\n"
                        "ЛКМ — орбита; колесо — приближение.")
        rows = "\n".join("[" + " ".join(f"{v:7.3f}" for v in row) + "]"
                         for row in cam.projection_matrix(aspect))
        self.camera_info.set(
            "Положение: (%.2f, %.2f, %.2f)\n" % cam.position
            + "Направление: (%.3f, %.3f, %.3f)\n" % cam.direction
            + f"FOV {cam.fov:.0f}° · near {cam.near:.3g} · far {cam.far:.3g}\n"
            + "Матрица проекции P:\n" + rows)

    def redraw(self):
        if self._suspend_redraw:
            return
        if not self.camera_mode.get():
            super().redraw()
            return
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 10 or height < 10:
            return
        try:
            self._place_camera()
            result = render_camera(self.models or [self.poly], self.camera, width, height,
                                   self.filled.get(), self.backface_enabled.get())
        except ValueError as error:
            self.status.set(f"Ошибка камеры: {error}")
            return
        self._render_image = tk.PhotoImage(master=self.root, data=result.buffer.ppm(),
                                           format="PPM")
        canvas.create_image(0, 0, anchor="nw", image=self._render_image)
        if self.show_axes.get():
            self._draw_camera_axes(width, height)
        if self.backface_enabled.get() and self.show_culled_faces.get():
            for face_views in result.face_views:
                self._draw_camera_culled(face_views, width, height)
        if self.show_face_normals.get() and result.face_views:
            self._draw_camera_normals(result.face_views[self.active_index], width, height)
        self._update_status(result, width / height)



def main():
    root = tk.Tk()
    CameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()