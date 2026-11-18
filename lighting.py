# 9 лаба
"""Освещение Ламберта от точечного источника для шейдинга Гуро.

Все входные векторы задаются в мировых координатах. Компоненты RGB и
возвращаемые вещественные цвета лежат в диапазоне [0, 255]. Округление
выполняется только после интерполяции цвета в растеризаторе.
"""

import math

DEFAULT_LIGHT_POSITION = (4.0, 6.0, 8.0)


# 9 лаба
def lambert_color(position, normal, light_position=DEFAULT_LIGHT_POSITION,
                  object_color=(90, 170, 245), ambient=0.15, diffuse=0.85):
    """Цвет = цвет объекта * ограничение(ambient + diffuse * max(0, N·L), 0, 1).

    L направлен от вершины к точечному источнику света. Блики и затухание
    с расстоянием не учитываются; модуль скалярного произведения не берётся.
    При нулевой нормали или совпадении вершины со светом остаётся только
    фоновое освещение. Для чистого диффузного Ламберта задайте ambient=0.
    """
    for vector in (position, normal, light_position, object_color):
        if len(vector) != 3 or not all(math.isfinite(c) for c in vector):
            raise ValueError("Позиции, нормаль и RGB должны содержать три конечных числа")
    if any(c < 0 or c > 255 for c in object_color):
        raise ValueError("Компоненты RGB должны быть в диапазоне 0..255")
    if not all(math.isfinite(c) and c >= 0 for c in (ambient, diffuse)):
        raise ValueError("Коэффициенты освещения должны быть конечными и неотрицательными")
    light = tuple(light_position[i] - position[i] for i in range(3))
    nl, ll = math.hypot(*normal), math.hypot(*light)
    cosine = sum((normal[i] / nl) * (light[i] / ll) for i in range(3)) if nl and ll else 0.0
    intensity = min(1.0, max(0.0, ambient + diffuse * max(0.0, cosine)))
    return tuple(c * intensity for c in object_color)
