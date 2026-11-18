"""Программный Z-буфер без зависимости от Tkinter и сторонних библиотек.

Экранные вершины задаются тройками (x, y, глубина). Меньшая глубина ближе.
В параллельной проекции глубина равна -z, в прежней перспективной проекции —
-1 / (расстояние_камеры - z). Обратная глубина линейна в экранных координатах,
в отличие от самого расстояния до камеры.
"""

import math


def triangulate(points):
    """Триангуляция отсечением ушей в основной плоскости грани OBJ.

    Поддерживаются выпуклые и вогнутые грани. Возвращаются индексы вершин;
    исходная граница многоугольника сохраняется для отрисовки рёбер.
    Неплоские четырёхугольники трактуются как поверхность из треугольников.
    """
    if len(points) < 3:
        return []
    normal = [0.0, 0.0, 0.0]
    for a, b in zip(points, points[1:] + points[:1]):
        normal[0] += (a[1] - b[1]) * (a[2] + b[2])
        normal[1] += (a[2] - b[2]) * (a[0] + b[0])
        normal[2] += (a[0] - b[0]) * (a[1] + b[1])
    drop = max(range(3), key=lambda i: abs(normal[i]))
    axes = [i for i in range(3) if i != drop]
    flat = [(p[axes[0]], p[axes[1]]) for p in points]

    def cross(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    area = sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(flat, flat[1:] + flat[:1]))
    span = max(max(p[i] for p in flat) - min(p[i] for p in flat) for i in (0, 1))
    eps = max(span * span * 1e-12, 1e-30)
    if abs(area) <= eps:
        return []
    sign = 1 if area > 0 else -1
    remaining = list(range(len(points)))
    result = []
    while len(remaining) > 3:
        for k, b in enumerate(remaining):
            a, c = remaining[k - 1], remaining[(k + 1) % len(remaining)]
            turn = sign * cross(flat[a], flat[b], flat[c])
            if abs(turn) <= eps:
                remaining.pop(k)
                break
            if turn < 0:
                continue
            if any(sign * cross(flat[a], flat[b], flat[p]) >= -eps
                   and sign * cross(flat[b], flat[c], flat[p]) >= -eps
                   and sign * cross(flat[c], flat[a], flat[p]) >= -eps
                   for p in remaining if p not in (a, b, c)):
                continue
            result.append((a, b, c))
            remaining.pop(k)
            break
        else:
            # У некорректных и самопересекающихся граней нет однозначной внутренней области.
            return []
    if len(remaining) == 3:
        result.append(tuple(remaining))
    return result


def clip_near(points, distance, near=0.05):
    """Обрезать многоугольник в координатах вида по z <= distance - near."""
    result = []
    if not points:
        return result
    limit = distance - near
    previous = points[-1]
    for current in points:
        if (previous[2] <= limit) != (current[2] <= limit):
            t = (limit - previous[2]) / (current[2] - previous[2])
            result.append(tuple(previous[i] + t * (current[i] - previous[i])
                                for i in range(3)))
        if current[2] <= limit:
            result.append(current)
        previous = current
    return result


def project(point, width, height, zoom, mode, distance):
    x, y, z = point
    if mode == "persp":
        reciprocal = 1.0 / (distance - z)
        return (width / 2 + zoom * x * distance * reciprocal,
                height / 2 - zoom * y * distance * reciprocal, -reciprocal)
    return width / 2 + zoom * x, height / 2 - zoom * y, -z


class ZBuffer:
    """Одна глубина и один RGB-цвет на пиксель; сортировка граней не нужна."""

    def __init__(self, width, height, background=(255, 255, 255)):
        self.width, self.height = width, height
        self.depth = [math.inf] * (width * height)
        self.owner = [-1] * (width * height)
        self.rgb = bytearray(background) * (width * height)

    def triangle(self, a, b, c, color, owner=0):
        """Барицентрическая растеризация по центрам пикселей при любом обходе вершин."""
        # 9 лаба
        self._triangle(a, b, c, color, owner)

    # 9 лаба
    def triangle_gouraud(self, p0, p1, p2, c0, c1, c2, owner=0, *, reciprocal_w=None):
        """Интерполяция RGB вершин по барицентрическим весам с проверкой глубины.

        reciprocal_w=(1/w0, 1/w1, 1/w2) включает поправку на перспективу.
        Для параллельной проекции и обычной экранной интерполяции параметр
        можно опустить. Спроецированная глубина интерполируется линейно.
        """
        colors = (c0, c1, c2)
        if any(len(c) != 3 or not all(math.isfinite(v) for v in c) for c in colors):
            raise ValueError("Цвет вершины должен содержать три конечных компонента")
        if reciprocal_w is not None:
            if len(reciprocal_w) != 3 or not all(math.isfinite(v) and v > 0 for v in reciprocal_w):
                raise ValueError("Для перспективы нужны три положительных значения 1/w")
        self._triangle(p0, p1, p2, None, owner, colors, reciprocal_w)

    # 9 лаба
    def _triangle(self, a, b, c, color, owner, colors=None, reciprocal_w=None):
        """Общий расчёт покрытия и глубины для плоской закраски и шейдинга Гуро."""
        if not all(math.isfinite(v) for p in (a, b, c) for v in p):
            return
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = c
        determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(determinant) < 1e-12:
            return
        left = max(0, math.ceil(min(ax, bx, cx) - 0.5))
        right = min(self.width - 1, math.floor(max(ax, bx, cx) - 0.5))
        top = max(0, math.ceil(min(ay, by, cy) - 0.5))
        bottom = min(self.height - 1, math.floor(max(ay, by, cy) - 0.5))
        if left > right or top > bottom:
            return
        ux, uy = (cy - ay) / determinant, (ax - cx) / determinant
        vx, vy = (ay - by) / determinant, (bx - ax) / determinant
        dzdx, dzdy = ux * (bz - az) + vx * (cz - az), uy * (bz - az) + vy * (cz - az)
        # 9 лаба
        rgb = bytes(color) if colors is None else None
        depth, pixels, owners = self.depth, self.rgb, self.owner
        for y in range(top, bottom + 1):
            dx, dy = left + 0.5 - ax, y + 0.5 - ay
            u, v = ux * dx + uy * dy, vx * dx + vy * dy
            z = az + dzdx * dx + dzdy * dy
            index = y * self.width + left
            for _ in range(left, right + 1):
                if u >= -1e-10 and v >= -1e-10 and u + v <= 1 + 1e-10:
                    if z < depth[index]:
                        depth[index] = z
                        owners[index] = owner
                        offset = index * 3
                        # 9 лаба
                        # Для Гуро смешиваем цвета вершин после проверки глубины.
                        if colors is None:
                            pixels[offset:offset + 3] = rgb
                        else:
                            w0, w1, w2 = 1 - u - v, u, v
                            if reciprocal_w is not None:
                                w0 *= reciprocal_w[0]
                                w1 *= reciprocal_w[1]
                                w2 *= reciprocal_w[2]
                                total = w0 + w1 + w2
                                w0, w1, w2 = w0 / total, w1 / total, w2 / total
                            for channel in range(3):
                                value = (w0 * colors[0][channel] + w1 * colors[1][channel]
                                         + w2 * colors[2][channel])
                                pixels[offset + channel] = max(0, min(255, round(value)))
                u += ux
                v += vx
                z += dzdx
                index += 1

    def line(self, a, b, color=(27, 42, 65), owner=0):
        """Нарисовать границы граней с проверкой по заполненному буферу глубины."""
        # Сначала обрезаем по окну, чтобы большой масштаб не увеличивал число шагов.
        dx, dy, dz = (b[i] - a[i] for i in range(3))
        lo, hi = 0.0, 1.0
        for start, delta, maximum in ((a[0], dx, self.width - 1),
                                      (a[1], dy, self.height - 1)):
            if abs(delta) < 1e-15:
                if not 0 <= start <= maximum:
                    return
            else:
                t0, t1 = -start / delta, (maximum - start) / delta
                lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
                if lo > hi:
                    return
        steps = max(1, math.ceil(max(abs(dx), abs(dy)) * (hi - lo)))
        rgb = bytes(color)
        for k in range(steps + 1):
            t = lo + (hi - lo) * k / steps
            x, y = int(a[0] + dx * t), int(a[1] + dy * t)
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            index = y * self.width + x
            z = a[2] + dz * t
            if self.owner[index] == owner or z <= self.depth[index] + 1e-9:
                offset = index * 3
                self.rgb[offset:offset + 3] = rgb

    def ppm(self):
        """Двоичный формат PPM напрямую поддерживается tkinter.PhotoImage."""
        return f"P6\n{self.width} {self.height}\n255\n".encode("ascii") + self.rgb


PALETTE = ((90, 170, 245), (240, 120, 70), (95, 195, 110))


def render_scene(models, to_view, width, height, zoom, mode, distance, filled=True):
    """Нарисовать все объекты, включая обратные грани, в общий буфер глубины."""
    buffer = ZBuffer(width, height)
    edges = []
    owner = 0
    light = (0.3, 0.5, 1.0)
    light_length = math.sqrt(sum(v * v for v in light))

    def screen(p):
        return project(p, width, height, zoom, mode, distance)

    for object_index, model in enumerate(models):
        vertices = [to_view(p) for p in model.vertex_array()]
        base = PALETTE[object_index % len(PALETTE)]
        for indices in model.faces_as_indices():
            owner += 1
            points = [vertices[i] for i in indices]
            normal = [0.0, 0.0, 0.0]
            for a, b in zip(points, points[1:] + points[:1]):
                normal[0] += (a[1] - b[1]) * (a[2] + b[2])
                normal[1] += (a[2] - b[2]) * (a[0] + b[0])
                normal[2] += (a[0] - b[0]) * (a[1] + b[1])
            length = math.sqrt(sum(v * v for v in normal))
            # Сохраняем прежнее двустороннее освещение; нормаль не отсекает грань.
            shade = (0.3 + 0.7 * abs(sum(a * b for a, b in zip(normal, light)))
                     / (length * light_length)) if length > 1e-12 else 0.3
            color = tuple(min(255, int(v * shade)) for v in base) if filled else (255, 255, 255)
            for triangle in triangulate(points):
                polygon = [points[i] for i in triangle]
                if mode == "persp":
                    polygon = clip_near(polygon, distance)
                projected = [screen(p) for p in polygon]
                for i in range(1, len(projected) - 1):
                    buffer.triangle(projected[0], projected[i], projected[i + 1], color, owner)
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
