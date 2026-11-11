#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import math
import operator
from typing import Callable, Dict, List, Optional, Tuple

from geometry import Point, Polygon, Polyhedron

MIN_SEGMENTS = 1
MAX_SEGMENTS = 150
MAX_ABS_Z = 1e9


# ---------------------------------------------------------------- формулы
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS: Dict[str, Callable[..., float]] = {
    name: getattr(math, name)
    for name in ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
                 "cosh", "tanh", "exp", "log", "log10", "log2", "sqrt", "hypot",
                 "floor", "ceil", "fabs", "radians", "degrees", "pow")
}
_FUNCTIONS["abs"] = abs
_FUNCTIONS["min"] = min
_FUNCTIONS["max"] = max
_FUNCTIONS["sign"] = lambda v: (v > 0) - (v < 0)
_FUNCTIONS["ln"] = math.log
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_MAX_POWER = 1000.0


def compile_formula(text: str) -> Callable[[float, float], float]:
    source = str(text).strip().replace("^", "**")
    if not source:
        raise ValueError("Введите формулу f(x, y)")
    if source.lower().startswith("z") and "=" in source:
        source = source.split("=", 1)[1]
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        raise ValueError(f"Не удалось разобрать формулу «{text.strip()}»") from None

    def check(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            check(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("В формуле допустимы только числа")
        elif isinstance(node, ast.Name):
            if node.id not in ("x", "y") and node.id not in _CONSTANTS:
                raise ValueError(f"Неизвестное имя «{node.id}» (есть x, y, pi, e, tau)")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OPS:
                raise ValueError("Недопустимая операция в формуле")
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ValueError("Недопустимая операция в формуле")
            check(node.operand)
        elif isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS
                    or node.keywords):
                name = node.func.id if isinstance(node.func, ast.Name) else "?"
                raise ValueError(f"Неизвестная функция «{name}»")
            for arg in node.args:
                check(arg)
        else:
            raise ValueError("Недопустимая конструкция в формуле")

    check(tree)

    def evaluate(node: ast.AST, x: float, y: float) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body, x, y)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return x if node.id == "x" else y if node.id == "y" else _CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](evaluate(node.operand, x, y))
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left, x, y), evaluate(node.right, x, y)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
                raise OverflowError("слишком большая степень")
            return _BIN_OPS[type(node.op)](left, right)
        return _FUNCTIONS[node.func.id](*(evaluate(a, x, y) for a in node.args))

    def f(x: float, y: float) -> float:
        return float(evaluate(tree, x, y))

    return f


#  построение
def _check_range(name: str, lo: float, hi: float) -> Tuple[float, float]:
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"Диапазон {name}: границы должны быть конечными числами")
    if not lo < hi:
        raise ValueError(f"Диапазон {name}: нужно {name}0 < {name}1")
    return lo, hi


def _check_segments(name: str, value) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"Число разбиений по {name} должно быть целым")
    value = int(value)
    if not MIN_SEGMENTS <= value <= MAX_SEGMENTS:
        raise ValueError(
            f"Число разбиений по {name} должно быть от {MIN_SEGMENTS} до {MAX_SEGMENTS}")
    return value


def build_surface(
    func: "str | Callable[[float, float], float]",
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Optional[Tuple[float, float]] = None,
    nx: int = 20,
    ny: Optional[int] = None,
    name: Optional[str] = None,
) -> Polyhedron:
    """Построить сегмент поверхности z = f(x, y).

    func     — формула-строка или функция f(x, y);
    x_range  — [x0, x1]; y_range — [y0, y1] (по умолчанию такой же, как для X);
    nx, ny   — число разбиений по осям (ny по умолчанию равно nx).
    Вершины в точках, где f не определена, пропускаются вместе с гранями.
    """
    label = func.strip() if isinstance(func, str) else getattr(func, "__name__", "f")
    f = compile_formula(func) if isinstance(func, str) else func
    x0, x1 = _check_range("x", *x_range)
    y0, y1 = _check_range("y", *(y_range if y_range is not None else x_range))
    nx = _check_segments("x", nx)
    ny = _check_segments("y", nx if ny is None else ny)

    xs = [x0 + (x1 - x0) * i / nx for i in range(nx + 1)]
    ys = [y0 + (y1 - y0) * j / ny for j in range(ny + 1)]

    values: List[List[Optional[float]]] = []
    for x in xs:
        column: List[Optional[float]] = []
        for y in ys:
            try:
                z = f(x, y)
                ok = isinstance(z, (int, float)) and math.isfinite(z) and abs(z) <= MAX_ABS_Z
            except (ArithmeticError, ValueError, TypeError):
                ok = False
            column.append(float(z) if ok else None)
        values.append(column)

    cells: List[Tuple[int, int, int, int]] = []
    used = set()
    for i in range(nx):
        for j in range(ny):
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            if all(values[a][b] is not None for a, b in corners):
                cells.append(corners)
                used.update(corners)
    if not cells:
        raise ValueError("Функция не определена ни на одной клетке выбранного диапазона")

    vertices: List[Point] = []
    index: Dict[Tuple[int, int], int] = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            if (i, j) in used:
                index[(i, j)] = len(vertices)
                vertices.append(Point(xs[i], ys[j], values[i][j]))
    faces = [Polygon([vertices[index[c]] for c in cell]) for cell in cells]
    return Polyhedron(vertices, faces, name or f"surface: {label}")


#  заготовки
# (формула, x0, x1, y0, y1, разбиений)
PRESETS: Dict[str, Tuple[str, float, float, float, float, int]] = {
    "Параболоид":     ("x**2 + y**2", -2, 2, -2, 2, 20),
    "Седло":          ("x**2 - y**2", -2, 2, -2, 2, 20),
    "Волна":          ("sin(x) * cos(y)", -3.14159, 3.14159, -3.14159, 3.14159, 30),
    "Рябь":           ("sin(sqrt(x**2 + y**2)) * 2", -8, 8, -8, 8, 50),
    "Полусфера":      ("sqrt(4 - x**2 - y**2)", -2, 2, -2, 2, 30),
    "Гауссиан":       ("3 * exp(-(x**2 + y**2))", -3, 3, -3, 3, 30),
    "Плоскость":      ("0.5 * x + 0.3 * y", -2, 2, -2, 2, 4),
    "Обезьянье седло": ("x**3 - 3 * x * y**2", -1.5, 1.5, -1.5, 1.5, 30),
}


#  CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить график z = f(x, y) и сохранить его в OBJ.")
    parser.add_argument("-o", "--output", required=True, help="выходной .obj")
    parser.add_argument("-f", "--function", help='формула, например "sin(x)*cos(y)"')
    parser.add_argument("--preset", choices=list(PRESETS), help="готовая функция")
    parser.add_argument("--x", nargs=2, type=float, metavar=("X0", "X1"))
    parser.add_argument("--y", nargs=2, type=float, metavar=("Y0", "Y1"))
    parser.add_argument("-n", "--segments", type=int, help="разбиений по X и Y")
    parser.add_argument("--ny", type=int, help="разбиений по Y (если отличается)")
    args = parser.parse_args()

    if bool(args.function) == bool(args.preset):
        parser.error("укажите ровно одно из: --function или --preset")
    if args.preset:
        formula, px0, px1, py0, py1, pn = PRESETS[args.preset]
    else:
        formula, (px0, px1, py0, py1), pn = args.function, (-3, 3, -3, 3), 20
    x_range = tuple(args.x) if args.x else (px0, px1)
    y_range = tuple(args.y) if args.y else (py0, py1)
    nx = args.segments or pn
    poly = build_surface(formula, x_range, y_range, nx, args.ny or nx)

    from viewer import save_obj

    save_obj(poly, args.output)
    print(poly.info())
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
