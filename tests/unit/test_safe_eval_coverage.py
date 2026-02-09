"""Tests for safe_eval to improve coverage.

Targets uncovered code paths in src/utils/safe_eval.py:
- All binary operations (Add, Sub, Mult, Div, Mod)
- All unary operations (Not, UAdd, USub)
- All comparison operations (Eq, NotEq, Gt, GtE, Lt, LtE)
- Boolean operations (And, Or)
- Name resolution (known names, boolean names, unknown names)
- Attribute access (dict, object, private attribute error)
- Subscript (index, slice)
- List, Tuple, Dict literals
- Error paths (unsupported operators, expressions)
"""

from __future__ import annotations

import pytest

from src.utils.safe_eval import safe_eval


class TestBinaryOperations:
    """Tests for binary operators."""

    def test_add(self):
        """Addition operator."""
        assert safe_eval("a + b", {"a": 3, "b": 5}) == 8

    def test_sub(self):
        """Subtraction operator."""
        assert safe_eval("a - b", {"a": 10, "b": 3}) == 7

    def test_mult(self):
        """Multiplication operator."""
        assert safe_eval("a * b", {"a": 4, "b": 5}) == 20

    def test_div(self):
        """Division operator."""
        assert safe_eval("a / b", {"a": 10, "b": 2}) == 5.0

    def test_mod(self):
        """Modulo operator."""
        assert safe_eval("a % b", {"a": 17, "b": 5}) == 2

    def test_unsupported_binary_op(self):
        """Unsupported binary operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported binary operator"):
            safe_eval("a ** b", {"a": 2, "b": 3})  # Power not supported


class TestUnaryOperations:
    """Tests for unary operators."""

    def test_not(self):
        """Not operator."""
        assert safe_eval("not a", {"a": True}) is False
        assert safe_eval("not a", {"a": False}) is True

    def test_uadd(self):
        """Unary positive operator."""
        assert safe_eval("+a", {"a": 5}) == 5
        assert safe_eval("+a", {"a": -5}) == -5

    def test_usub(self):
        """Unary negative operator."""
        assert safe_eval("-a", {"a": 5}) == -5
        assert safe_eval("-a", {"a": -5}) == 5

    def test_unsupported_unary_op(self):
        """Unsupported unary operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported unary operator"):
            safe_eval("~a", {"a": 5})  # Bitwise not not supported


class TestComparisonOperations:
    """Tests for comparison operators."""

    def test_eq(self):
        """Equality operator."""
        assert safe_eval("a == b", {"a": 5, "b": 5}) is True
        assert safe_eval("a == b", {"a": 5, "b": 6}) is False

    def test_not_eq(self):
        """Not equal operator."""
        assert safe_eval("a != b", {"a": 5, "b": 6}) is True
        assert safe_eval("a != b", {"a": 5, "b": 5}) is False

    def test_gt(self):
        """Greater than operator."""
        assert safe_eval("a > b", {"a": 6, "b": 5}) is True
        assert safe_eval("a > b", {"a": 5, "b": 5}) is False

    def test_gte(self):
        """Greater than or equal operator."""
        assert safe_eval("a >= b", {"a": 5, "b": 5}) is True
        assert safe_eval("a >= b", {"a": 6, "b": 5}) is True
        assert safe_eval("a >= b", {"a": 4, "b": 5}) is False

    def test_lt(self):
        """Less than operator."""
        assert safe_eval("a < b", {"a": 4, "b": 5}) is True
        assert safe_eval("a < b", {"a": 5, "b": 5}) is False

    def test_lte(self):
        """Less than or equal operator."""
        assert safe_eval("a <= b", {"a": 5, "b": 5}) is True
        assert safe_eval("a <= b", {"a": 4, "b": 5}) is True
        assert safe_eval("a <= b", {"a": 6, "b": 5}) is False

    def test_chained_comparison(self):
        """Chained comparison (e.g., a < b < c)."""
        assert safe_eval("a < b < c", {"a": 1, "b": 2, "c": 3}) is True
        assert safe_eval("a < b < c", {"a": 1, "b": 3, "c": 2}) is False

    def test_unsupported_comparison_op(self):
        """Unsupported comparison operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported comparison operator"):
            safe_eval("a in b", {"a": 1, "b": [1, 2, 3]})  # in not supported


class TestBooleanOperations:
    """Tests for boolean operations."""

    def test_and(self):
        """And operator."""
        assert safe_eval("a and b", {"a": True, "b": True}) is True
        assert safe_eval("a and b", {"a": True, "b": False}) is False
        assert safe_eval("a and b", {"a": False, "b": True}) is False

    def test_or(self):
        """Or operator."""
        assert safe_eval("a or b", {"a": True, "b": False}) is True
        assert safe_eval("a or b", {"a": False, "b": True}) is True
        assert safe_eval("a or b", {"a": False, "b": False}) is False

    def test_multiple_and(self):
        """Multiple and operations."""
        assert safe_eval("a and b and c", {"a": True, "b": True, "c": True}) is True
        assert safe_eval("a and b and c", {"a": True, "b": False, "c": True}) is False

    def test_multiple_or(self):
        """Multiple or operations."""
        assert safe_eval("a or b or c", {"a": False, "b": False, "c": True}) is True
        assert safe_eval("a or b or c", {"a": False, "b": False, "c": False}) is False


class TestNameResolution:
    """Tests for name resolution."""

    def test_known_name(self):
        """Known name from names dict."""
        assert safe_eval("x", {"x": 42}) == 42

    def test_bool_name_true(self):
        """Boolean name 'true' (case insensitive)."""
        assert safe_eval("true", {}) is True
        assert safe_eval("True", {}) is True
        assert safe_eval("TRUE", {}) is True

    def test_bool_name_false(self):
        """Boolean name 'false' (case insensitive)."""
        assert safe_eval("false", {}) is False
        assert safe_eval("False", {}) is False

    def test_bool_name_null(self):
        """Boolean name 'null' (case insensitive)."""
        assert safe_eval("null", {}) is None
        assert safe_eval("Null", {}) is None

    def test_bool_name_none(self):
        """Boolean name 'none' (case insensitive)."""
        assert safe_eval("none", {}) is None
        assert safe_eval("None", {}) is None

    def test_unknown_name(self):
        """Unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown name: x"):
            safe_eval("x", {})


class TestConstants:
    """Tests for constant values."""

    def test_int_constant(self):
        """Integer constant."""
        assert safe_eval("42", {}) == 42

    def test_float_constant(self):
        """Float constant."""
        assert safe_eval("3.14", {}) == 3.14

    def test_string_constant(self):
        """String constant."""
        assert safe_eval("'hello'", {}) == "hello"
        assert safe_eval('"world"', {}) == "world"


class TestAttributeAccess:
    """Tests for attribute access."""

    def test_dict_attribute(self):
        """Attribute access on dict returns dict value."""
        obj = {"name": "test", "value": 123}
        assert safe_eval("obj.name", {"obj": obj}) == "test"
        assert safe_eval("obj.value", {"obj": obj}) == 123

    def test_object_attribute(self):
        """Attribute access on object returns attribute."""
        class Obj:
            name = "test"
            value = 123

        obj = Obj()
        assert safe_eval("obj.name", {"obj": obj}) == "test"
        assert safe_eval("obj.value", {"obj": obj}) == 123

    def test_private_attribute_error(self):
        """Private attribute access raises ValueError."""
        obj = {"_private": "secret"}
        with pytest.raises(ValueError, match="Private attributes are not allowed"):
            safe_eval("obj._private", {"obj": obj})


class TestSubscript:
    """Tests for subscript access."""

    def test_list_index(self):
        """List index access."""
        lst = [10, 20, 30]
        assert safe_eval("lst[0]", {"lst": lst}) == 10
        assert safe_eval("lst[1]", {"lst": lst}) == 20
        assert safe_eval("lst[-1]", {"lst": lst}) == 30

    def test_dict_key(self):
        """Dict key access."""
        d = {"a": 1, "b": 2}
        assert safe_eval('d["a"]', {"d": d}) == 1
        assert safe_eval('d["b"]', {"d": d}) == 2

    def test_string_index(self):
        """String index access."""
        s = "hello"
        assert safe_eval("s[0]", {"s": s}) == "h"
        assert safe_eval("s[-1]", {"s": s}) == "o"

    def test_slice(self):
        """Slice access."""
        lst = [1, 2, 3, 4, 5]
        assert safe_eval("lst[1:3]", {"lst": lst}) == [2, 3]
        assert safe_eval("lst[:2]", {"lst": lst}) == [1, 2]
        assert safe_eval("lst[2:]", {"lst": lst}) == [3, 4, 5]
        assert safe_eval("lst[::2]", {"lst": lst}) == [1, 3, 5]

    def test_slice_with_step(self):
        """Slice with step."""
        lst = [1, 2, 3, 4, 5, 6]
        assert safe_eval("lst[1:5:2]", {"lst": lst}) == [2, 4]


class TestContainerLiterals:
    """Tests for container literals."""

    def test_list_literal(self):
        """List literal."""
        assert safe_eval("[1, 2, 3]", {}) == [1, 2, 3]
        assert safe_eval("[a, b]", {"a": 1, "b": 2}) == [1, 2]

    def test_tuple_literal(self):
        """Tuple literal."""
        assert safe_eval("(1, 2, 3)", {}) == [1, 2, 3]  # Returns list for tuple AST

    def test_dict_literal(self):
        """Dict literal."""
        result = safe_eval('{"a": 1, "b": 2}', {})
        assert result == {"a": 1, "b": 2}

    def test_dict_literal_with_vars(self):
        """Dict literal with variables."""
        result = safe_eval('{"key": x}', {"x": 42})
        assert result == {"key": 42}

    def test_empty_list(self):
        """Empty list literal."""
        assert safe_eval("[]", {}) == []

    def test_empty_dict(self):
        """Empty dict literal."""
        assert safe_eval("{}", {}) == {}


class TestComplexExpressions:
    """Tests for complex expressions."""

    def test_nested_binary_ops(self):
        """Nested binary operations."""
        assert safe_eval("a + b * c", {"a": 2, "b": 3, "c": 4}) == 14
        assert safe_eval("(a + b) * c", {"a": 2, "b": 3, "c": 4}) == 20

    def test_comparison_with_arithmetic(self):
        """Comparison with arithmetic."""
        assert safe_eval("a + b > c", {"a": 3, "b": 4, "c": 5}) is True

    def test_boolean_with_comparison(self):
        """Boolean operation with comparison."""
        assert safe_eval("a > 0 and b > 0", {"a": 1, "b": 2}) is True
        assert safe_eval("a > 0 or b > 0", {"a": -1, "b": 2}) is True

    def test_nested_attribute(self):
        """Nested attribute access."""
        obj = {"inner": {"value": 42}}
        assert safe_eval("obj.inner.value", {"obj": obj}) == 42

    def test_subscript_with_expression(self):
        """Subscript with computed index."""
        lst = [10, 20, 30]
        assert safe_eval("lst[a]", {"lst": lst, "a": 1}) == 20


class TestErrorCases:
    """Tests for error cases."""

    def test_unsupported_expression(self):
        """Unsupported expression raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported expression"):
            safe_eval("x if True else y", {"x": 1, "y": 2})  # IfExp not supported

    def test_function_call_not_supported(self):
        """Function call not supported."""
        with pytest.raises(ValueError, match="Unsupported expression"):
            safe_eval("len(x)", {"x": [1, 2, 3]})

    def test_invalid_syntax(self):
        """Invalid syntax raises SyntaxError."""
        with pytest.raises(SyntaxError):
            safe_eval("a +", {})

    def test_dict_unpacking_not_supported(self):
        """Dict unpacking is not supported."""
        with pytest.raises(ValueError, match="Dict unpacking is not supported"):
            safe_eval("{**d}", {"d": {"a": 1}})
