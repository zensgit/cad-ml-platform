import pytest

from src.core.knowledge.tolerance import get_fundamental_deviation


@pytest.mark.parametrize("symbol,size", [("H", 10.0), ("g", 10.0)])
def test_get_fundamental_deviation_returns_value(symbol, size):
    value = get_fundamental_deviation(symbol, size)
    assert value is not None


def test_get_fundamental_deviation_rejects_empty():
    assert get_fundamental_deviation("", 10.0) is None
    assert get_fundamental_deviation(" ", 10.0) is None
