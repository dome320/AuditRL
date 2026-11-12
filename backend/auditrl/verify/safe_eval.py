"""Evaluate LLM-proposed arithmetic without `eval`, propagating rounding intervals."""

import ast
import itertools
import operator
from collections.abc import Mapping

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "min": min, "max": max}
MAX_OPERANDS = 8


class UnsafeExpression(ValueError):
    pass


def _check(node: ast.AST, names: set[str]) -> None:
    match node:
        case ast.Expression(body=body):
            _check(body, names)
        case ast.BinOp(left=l, op=op, right=r) if type(op) in _BINOPS:
            _check(l, names)
            _check(r, names)
        case ast.UnaryOp(op=op, operand=o) if type(op) in _UNARY:
            _check(o, names)
        case ast.Constant(value=v) if isinstance(v, int | float) and not isinstance(v, bool):
            pass
        case ast.Name(id=n) if n in names:
            pass
        case ast.Call(func=ast.Name(id=f), args=args, keywords=[]) if f in _FUNCS:
            for a in args:
                _check(a, names)
        case _:
            raise UnsafeExpression(f"disallowed syntax: {ast.dump(node)[:60]}")


def _eval(node: ast.AST, env: Mapping[str, float]) -> float:
    match node:
        case ast.Expression(body=body):
            return _eval(body, env)
        case ast.BinOp(left=l, op=op, right=r):
            right = _eval(r, env)
            if isinstance(op, ast.Pow) and abs(right) > 100:
                raise UnsafeExpression("exponent too large")
            return _BINOPS[type(op)](_eval(l, env), right)
        case ast.UnaryOp(op=op, operand=o):
            return _UNARY[type(op)](_eval(o, env))
        case ast.Constant(value=v):
            return float(v)
        case ast.Name(id=n):
            return env[n]
        case ast.Call(func=ast.Name(id=f), args=args):
            return float(_FUNCS[f](*(_eval(a, env) for a in args)))
    raise UnsafeExpression("unreachable")


def compile_expression(expression: str, names: set[str]) -> ast.Expression:
    if len(expression) > 300:
        raise UnsafeExpression("expression too long")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    _check(tree, names)
    return tree


def evaluate(expression: str, values: Mapping[str, float]) -> float:
    return _eval(compile_expression(expression, set(values)), values)


def evaluate_interval(
    expression: str, intervals: Mapping[str, tuple[float, float]]
) -> tuple[float, float, float]:
    """Return (low, high, point) over all corners of the operand rounding intervals."""
    if len(intervals) > MAX_OPERANDS:
        raise UnsafeExpression("too many operands")
    tree = compile_expression(expression, set(intervals))
    names = list(intervals)
    point = _eval(tree, {n: (lo + hi) / 2 for n, (lo, hi) in intervals.items()})
    results = [point]
    for corner in itertools.product(*(intervals[n] for n in names)):
        try:
            results.append(_eval(tree, dict(zip(names, corner, strict=True))))
        except ZeroDivisionError:
            continue
    return min(results), max(results), point
