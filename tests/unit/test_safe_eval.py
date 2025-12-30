from __future__ import annotations

from src.utils.safe_eval import safe_eval


def test_safe_eval_arithmetic() -> None:
    assert safe_eval("1 + 2 * 3", {}) == 7


def test_safe_eval_comparison_with_names() -> None:
    assert safe_eval("a > 3 and b == 2", {"a": 4, "b": 2}) is True


def test_safe_eval_attribute_dict_access() -> None:
    ctx = {"results": {"task1": {"success": True}}}
    assert safe_eval("results.task1.success == true", ctx) is True


def test_safe_eval_subscript_access() -> None:
    ctx = {"results": {"task1": {"success": False}}}
    assert safe_eval("results['task1']['success'] == false", ctx) is True
