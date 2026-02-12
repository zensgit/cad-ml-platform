from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Mapping, Optional

BinaryOp = Callable[[Any, Any], Any]
UnaryOp = Callable[[Any], Any]
CompareOp = Callable[[Any, Any], bool]

_BIN_OPS: dict[type[ast.operator], BinaryOp] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], UnaryOp] = {
    ast.Not: operator.not_,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CMP_OPS: dict[type[ast.cmpop], CompareOp] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
}
_BOOL_NAMES = {
    "true": True,
    "false": False,
    "null": None,
    "none": None,
}


def safe_eval(expression: str, names: Mapping[str, Any]) -> Any:
    """Evaluate a simple expression with a restricted AST."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body, names)


def _eval_node(node: ast.AST, names: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id.lower() in _BOOL_NAMES:
            return _BOOL_NAMES[node.id.lower()]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(value, names) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise ValueError("Unsupported boolean operator")
    if isinstance(node, ast.BinOp):
        bin_op = _BIN_OPS.get(type(node.op))
        if not bin_op:
            raise ValueError("Unsupported binary operator")
        return bin_op(_eval_node(node.left, names), _eval_node(node.right, names))
    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPS.get(type(node.op))
        if not unary_op:
            raise ValueError("Unsupported unary operator")
        return unary_op(_eval_node(node.operand, names))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, names)
        for op_node, comparator in zip(node.ops, node.comparators):
            cmp_op = _CMP_OPS.get(type(op_node))
            if not cmp_op:
                raise ValueError("Unsupported comparison operator")
            right = _eval_node(comparator, names)
            if not cmp_op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Attribute):
        value = _eval_node(node.value, names)
        attr = node.attr
        if attr.startswith("_"):
            raise ValueError("Private attributes are not allowed")
        if isinstance(value, dict) and attr in value:
            return value[attr]
        return getattr(value, attr)
    if isinstance(node, ast.Subscript):
        value = _eval_node(node.value, names)
        key = _eval_slice(node.slice, names)
        return value[key]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(elt, names) for elt in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise ValueError("Dict unpacking is not supported")
            result[_eval_node(key_node, names)] = _eval_node(value_node, names)
        return result
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def _eval_slice(node: ast.AST, names: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Slice):
        lower = _eval_optional(node.lower, names)
        upper = _eval_optional(node.upper, names)
        step = _eval_optional(node.step, names)
        return slice(lower, upper, step)
    return _eval_node(node, names)


def _eval_optional(node: Optional[ast.AST], names: Mapping[str, Any]) -> Optional[Any]:
    if node is None:
        return None
    return _eval_node(node, names)
