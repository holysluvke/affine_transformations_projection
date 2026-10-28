import math

from geometry import Matrix


def identity() -> Matrix:
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]

def translation(tx: float, ty: float, tz: float) -> Matrix:
    m = identity()
    m[0][3] = tx
    m[1][3] = ty
    m[2][3] = tz
    return m

def scaling(sx: float, sy: float, sz: float) -> Matrix:
    m = identity()
    m[0][0] = sx
    m[1][1] = sy
    m[2][2] = sz
    return m

def rotation_z(angle_deg: float) -> Matrix:
    a = math.radians(angle_deg)
    m = identity()
    m[0][0], m[0][1] = math.cos(a), -math.sin(a)
    m[1][0], m[1][1] = math.sin(a), math.cos(a)
    return m

def rotation_x(angle_deg: float) -> Matrix:
    a = math.radians(angle_deg)
    m = identity()
    m[1][1], m[1][2] = math.cos(a), -math.sin(a)
    m[2][1], m[2][2] = math.sin(a), math.cos(a)
    return m

def rotation_y(angle_deg: float) -> Matrix:
    a = math.radians(angle_deg)
    m = identity()
    m[0][0], m[0][2] = math.cos(a), math.sin(a)
    m[2][0], m[2][2] = -math.sin(a), math.cos(a)
    return m


def reflection_xy() -> Matrix:
    """z меняет знак"""
    return scaling(1.0, 1.0, -1.0)


def reflection_xz() -> Matrix:
    """y меняет знак"""
    return scaling(1.0, -1.0, 1.0)


def reflection_yz() -> Matrix:
    """x меняет знак"""
    return scaling(-1.0, 1.0, 1.0)

def multiply(a: Matrix, b: Matrix) -> Matrix:
    """умножение 4x4: a @ b, при столбцах первой действует b, потом a"""
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def about_point(c, m: Matrix) -> Matrix:
    """применить m так, будто начало координат находится в точке c
    читается справа налево по столбцам:
        1) сдвиг на -c (c уезжает в начало координат)
        2) m (делаем нужное преобразование)
        3) сдвиг на +c (возвращаем обратно)
    """
    to_origin = translation(-c[0], -c[1], -c[2])
    back = translation(c[0], c[1], c[2])
    return multiply(back, multiply(m, to_origin))


def scaling_about_point(c, sx: float, sy: float, sz: float) -> Matrix:
    return about_point(c, scaling(sx, sy, sz))


_AXIS_ROTATION = {"x": rotation_x, "y": rotation_y, "z": rotation_z}


def rotation_about_axis_through_point(c, axis: str, angle_deg: float) -> Matrix:
    """вращение вокруг прямой через точку c, параллельной оси x y и z"""
    return about_point(c, _AXIS_ROTATION[axis](angle_deg))


def rotation_about_axis(axis, angle_deg: float) -> Matrix:
    """поворот вокруг оси через начало координат с единичным направлением axis."""
    l, m, n = axis
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    return [[cos + l * l * (1.0 - cos),         l * m * (1.0 - cos) - n * sin,      l * n * (1.0 - cos) + m * sin,  0.0],
            [m * l * (1.0 - cos) + n * sin,     cos + m * m * (1.0 - cos),          m * n * (1.0 - cos) - l * sin,  0.0],
            [n * l * (1.0 - cos) - m * sin,     n * m * (1.0 - cos) + l * sin,      cos + n * n * (1.0 - cos),      0.0],
            [0.0,                               0.0,                                0.0,                            1.0]]


def rotation_about_line(p1, p2, angle_deg: float) -> Matrix:
    """поворот вокруг прямой, заданной двумя точками."""
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        raise ValueError("точки совпадают, тк прямая не определена")
    axis = (dx / length, dy / length, dz / length)
    return about_point(p1, rotation_about_axis(axis, angle_deg))


_PLANES = {"XY": reflection_xy, "XZ": reflection_xz, "YZ": reflection_yz}


def reflection(plane: str) -> Matrix:
    """отражение относительно координатной плоскости XY, XZ, YZ"""
    key = str(plane).strip().upper()
    if key not in _PLANES:
        raise ValueError(f"неизвестная плоскость: {plane!r} (нужно XY, XZ или YZ)")
    return _PLANES[key]()


def scale_about_center(center, sx: float, sy: float, sz: float) -> Matrix:
    return scaling_about_point(center, sx, sy, sz)

def rotate_about_center_axis(center, axis: str, angle_deg: float) -> Matrix:
    key = str(axis).strip().lower()
    if key not in _AXIS_ROTATION:
        raise ValueError(f"неизвестная ось: {axis!r} (нужно x, y или z)")
    return rotation_about_axis_through_point(center, key, angle_deg)

def rotate_about_line(p1, p2, angle_deg: float) -> Matrix:
    return rotation_about_line(p1, p2, angle_deg)