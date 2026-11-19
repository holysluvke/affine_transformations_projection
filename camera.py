import math
from geometry import (Matrix,
                      _v_sub as _sub,
                      _v_dot as _dot,
                      _v_cross as _cross,
                      _v_norm as _unit)

from dataclasses import dataclass, field

from backface_culling import classify_faces
from zbuffer import PALETTE, ZBuffer, triangulate
# 9 лаба
from lighting import DEFAULT_LIGHT_POSITION, lambert_color

MAX_ELEVATION = 89.0

def look_at_matrix(eye, direction, up=(0.0, 1.0, 0.0)) -> Matrix:
    """Мир -> координаты камеры (камера в нуле, смотрит вдоль -Z)"""
    forward = _unit(direction)
    if forward == (0.0, 0.0, 0.0):
        raise ValueError("направление обзора не может быть нулевым")
    right = _cross(forward, up)
    if math.sqrt(_dot(right, right)) < 1e-9:
        raise ValueError("направление обзора параллельно вектору «вверх»")
    right = _unit(right)
    true_up = _cross(right, forward)
    return [
        [right[0],    right[1],    right[2],    -_dot(right, eye)],
        [true_up[0],  true_up[1],  true_up[2],  -_dot(true_up, eye)],
        [-forward[0], -forward[1], -forward[2],  _dot(forward, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def perspective_matrix(fov_deg, aspect, near, far) -> Matrix:
    """Перспектива. fov — вертикальный угол обзора, aspect = ширина/высота"""
    if not 0.0 < fov_deg < 180.0:
        raise ValueError("FOV должен быть в диапазоне (0°, 180°)")
    if aspect <= 0.0:
        raise ValueError("aspect должен быть положительным")
    if not 0.0 < near < far:
        raise ValueError("нужно 0 < near < far")
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def _apply(m, p):
    """Матрица 4x4 * точку (x, y, z, 1) -> (x, y, z, w), без деления на w"""
    x, y, z = p[0], p[1], p[2]
    return tuple(m[i][0] * x + m[i][1] * y + m[i][2] * z + m[i][3]
                 for i in range(4))


class Camera:
    """Положение + направление обзора + параметры проекции"""
    def __init__(self, position=(0.0, 0.0, 5.0), direction=(0.0, 0.0, -1.0),
                 up=(0.0, 1.0, 0.0), fov=60.0, near=0.1, far=100.0):
        self.position = tuple(float(c) for c in position)
        self.direction = _unit(direction)
        self.up = tuple(float(c) for c in up)
        self.fov, self.near, self.far = float(fov), float(near), float(far)

    def view_matrix(self):
        return look_at_matrix(self.position, self.direction, self.up)

    def projection_matrix(self, aspect):
        # aspect это отношение ширины и высоты кадра
        return perspective_matrix(self.fov, aspect, self.near, self.far)

    def to_view(self, point):
        """Мировая точка -> координаты камеры (камера в нуле, смотрит в -Z)"""
        return _apply(self.view_matrix(), point)[:3]

    # --- управление
    def look_at(self, target):
        """Повернуть камеру на точку target, не двигая её"""
        self.direction = _unit(_sub(target, self.position))

    def orbit_around(self, center, azimuth_deg, elevation_deg, distance):
        """Поставить камеру на сферу вокруг центра и повернуть лицом к центру"""
        if distance <= 0.0:
            raise ValueError("расстояние до центра должно быть положительным")
        az = math.radians(azimuth_deg)
        el = math.radians(max(-MAX_ELEVATION, min(MAX_ELEVATION, elevation_deg)))
        offset = (distance * math.cos(el) * math.sin(az),
                  distance * math.sin(el),
                  distance * math.cos(el) * math.cos(az))
        self.position = (center[0] + offset[0],
                         center[1] + offset[1],
                         center[2] + offset[2])
        self.direction = _unit((-offset[0], -offset[1], -offset[2]))

    def distance_to_fit(self, radius, aspect):
        """Расстояние, на котором сфера радиуса radius целиком влезает в кадр"""
        half_v = math.radians(self.fov) / 2.0  # половина вертикального угла
        half_h = math.atan(math.tan(half_v) * aspect)  # половина горизонтального
        return radius / math.sin(min(half_v, half_h))  # узкий угол ограничивает

    def project_view_segment(self, a, b, width, height):
        """Отрезок в координатах камеры -> два экранных (x, y, depth) или None"""
        proj = self.projection_matrix(width / height)
        clipped = clip_segment(_apply(proj, a), _apply(proj, b))
        if clipped is None:
            return None
        return _viewport(clipped[0], width, height), _viewport(clipped[1], width, height)


# расстояние со знаком до плоскости в clip-пространстве: >= 0 значит внутри
_CLIP_PLANES = (
    lambda c: c[3] + c[2],  # ближняя: z >= -w
    lambda c: c[3] - c[2],  # дальняя: z <= w
)

def _lerp4(a, b, t):
    """Точка на отрезке a-b: при t=0 это a, при t=1 это b (все 4 координаты)"""
    return tuple(a[i] + t * (b[i] - a[i]) for i in range(4))

def clip_segment(a, b):
    """Обрезать отрезок a-b (clip-координаты).
       Вернуть (a, b) или None, если он снаружи"""
    for plane in _CLIP_PLANES:
        da, db = plane(a), plane(b)
        if da < 0.0 and db < 0.0:  # оба конца снаружи
            return None
        if da < 0.0 or db < 0.0:  # отрезок пересекает плоскость
            t = da / (da - db)  # где расстояние падает до нуля
            point = _lerp4(a, b, t)
            if da < 0.0:
                a = point  # заменяем тот конец, что снаружи
            else:
                b = point
    return a, b


def clip_polygon(polygon):
    """Сазерленд–Ходжман: обрезать многоугольник (список clip-точек) по near и far"""
    result = list(polygon)
    for plane in _CLIP_PLANES:
        if not result:
            break
        clipped = []
        previous = result[-1]  # начинаем с последней вершины
        previous_d = plane(previous)
        for current in result:
            current_d = plane(current)
            if (previous_d >= 0.0) != (current_d >= 0.0):  # ребро пересекло плоскость
                t = previous_d / (previous_d - current_d)
                clipped.append(_lerp4(previous, current, t))
            if current_d >= 0.0:
                clipped.append(current)  # внутренние вершины остаются
            previous, previous_d = current, current_d
        result = clipped
    return result


# 9 лаба
def clip_polygon_varyings(polygon, varyings):
    """Обрезать позиции и атрибуты вершин с одинаковым параметром вдоль ребра.

    Атрибуты — произвольные кортежи, для Гуро это компоненты RGB. Они
    интерполируются до деления на w; затем растеризатор учитывает 1/w.
    """
    if len(polygon) != len(varyings):
        raise ValueError("Каждой clip-вершине должен соответствовать набор атрибутов")
    if varyings and any(len(values) != len(varyings[0]) for values in varyings):
        raise ValueError("Число атрибутов должно совпадать у всех вершин")
    result = list(zip(polygon, varyings))
    for plane in _CLIP_PLANES:
        if not result:
            break
        clipped = []
        previous, previous_values = result[-1]
        previous_d = plane(previous)
        for current, current_values in result:
            current_d = plane(current)
            if (previous_d >= 0.0) != (current_d >= 0.0):
                t = previous_d / (previous_d - current_d)
                values = tuple(a + t * (b - a) for a, b in zip(previous_values, current_values))
                clipped.append((_lerp4(previous, current, t), values))
            if current_d >= 0.0:
                clipped.append((current, current_values))
            previous, previous_values, previous_d = current, current_values, current_d
        result = clipped
    return [p for p, _ in result], [v for _, v in result]


def _viewport(c, width, height):
    """clip -> пиксели: делим на w, растягиваем NDC [-1,1] на окно, переворачиваем Y"""
    w = c[3]
    return (width / 2.0 + c[0] / w * width / 2.0,
            height / 2.0 - c[1] / w * height / 2.0,  # в экране Y растёт вниз
            c[2] / w)  # глубина для Z-буфера



LIGHT = (0.3, 0.5, 1.0)  # свет в МИРОВЫХ координатах, не едет за камерой

@dataclass
class RenderResult:
    buffer: ZBuffer
    visible: int = 0  # сколько граней лицевых
    culled: int = 0  # сколько нелицевых
    face_views: list = field(default_factory=list)


# 9 лаба
def render_camera(models, camera, width, height, filled=True, backface=True, *,
                  shading="gouraud", light_position=DEFAULT_LIGHT_POSITION,
                  object_color=None, ambient=0.15, diffuse=0.85, draw_edges=None):
    """Вернуть RenderResult с плоской закраской или шейдингом Гуро по Ламберту.

    Именованные параметры предназначены для подключения интерфейса участником 2.
    Гуро включён по умолчанию; его можно показать без новых элементов управления.
    Положение света задаётся в мировых координатах, цвет — компонентами RGB.
    При object_color=None сохраняется палитра объектов. В режиме заливки Гуро
    рёбра по умолчанию скрыты, чтобы были видны плавные переходы цвета.
    """
    # 9 лаба
    if shading not in ("flat", "gouraud"):
        raise ValueError("Поддерживаются режимы flat и gouraud")
    if draw_edges is None:
        draw_edges = not filled or shading == "flat"
    buffer = ZBuffer(width, height)
    view = camera.view_matrix()  # один раз на кадр
    projection = camera.projection_matrix(width / height)

    def to_view(point):
        return _apply(view, point)[:3]

    # свет переводим в систему камеры (вектор: сдвиг не нужен, берём 3x3-часть)
    light = _unit(tuple(sum(view[i][k] * LIGHT[k] for k in range(3)) for i in range(3)))

    result = RenderResult(buffer)
    edges = []
    owner = 0  # уникальный номер грани

    for object_index, model in enumerate(models):
        # 9 лаба
        base = object_color if object_color is not None else PALETTE[object_index % len(PALETTE)]
        face_views = classify_faces(model, to_view, "persp", 0.0)
        result.face_views.append(face_views)

        for face in face_views:
            if face.visible:
                result.visible += 1
            else:
                result.culled += 1
                if backface:
                    continue  # отсечение: грань не рисуем
            owner += 1
            points = list(face.points)  # уже в системе камеры
            shade = 0.3 + 0.7 * abs(_dot(face.normal, light))  # 0.3 — «фоновый» свет
            color = (tuple(min(255, int(v * shade)) for v in base)
                     if filled else (255, 255, 255))
            # 9 лаба
            # Цвет каждой вершины вычисляется по Ламберту в мировых координатах.
            vertex_colors = None
            if filled and shading == "gouraud":
                world_face = model.faces[face.index]
                normals = model.face_vertex_normals(face.index)
                vertex_colors = [lambert_color(p.coords(), n, light_position, base, ambient, diffuse)
                                 for p, n in zip(world_face.points, normals)]

            for triangle in triangulate(points):
                # 9 лаба
                # При обрезании треугольника переносим цвета в новые вершины.
                projected = [_apply(projection, points[i]) for i in triangle]
                if vertex_colors is not None:
                    clip, colors = clip_polygon_varyings(projected, [vertex_colors[i] for i in triangle])
                else:
                    clip = clip_polygon(projected)
                if len(clip) < 3:
                    continue  # треугольник целиком вне
                screen = [_viewport(c, width, height) for c in clip]
                for i in range(1, len(screen) - 1):  # веер треугольников
                    # 9 лаба
                    # Передаём 1/w для интерполяции цвета с поправкой на перспективу.
                    if vertex_colors is not None:
                        buffer.triangle_gouraud(
                            screen[0], screen[i], screen[i + 1],
                            colors[0], colors[i], colors[i + 1], owner,
                            reciprocal_w=(1 / clip[0][3], 1 / clip[i][3], 1 / clip[i + 1][3]))
                    else:
                        buffer.triangle(screen[0], screen[i], screen[i + 1], color, owner)

            # 9 лаба
            # Видимость рёбер задаётся параметром draw_edges.
            for a, b in zip(points, points[1:] + points[:1]) if draw_edges else ():
                segment = clip_segment(_apply(projection, a), _apply(projection, b))
                if segment is not None:
                    edges.append((_viewport(segment[0], width, height),
                                  _viewport(segment[1], width, height), owner))

    for a, b, edge_owner in edges:  # рёбра: после всех граней
        buffer.line(a, b, owner=edge_owner)
    return result
