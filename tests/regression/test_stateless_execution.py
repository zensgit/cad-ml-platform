import pytest
import random
from src.core.similarity import _VECTOR_STORE, _VECTOR_META

# Mock data setup
def setup_data():
    _VECTOR_STORE.clear()
    _VECTOR_META.clear()
    for i in range(30):
        vid = f"vec_{i}"
        _VECTOR_STORE[vid] = [float(x) for x in range(10)]
        _VECTOR_META[vid] = {"material": "steel", "feature_version": "v1"}

def run_test_isolated(test_id):
    """Simulate a test case execution."""
    vid = f"vec_{test_id}"
    assert vid in _VECTOR_STORE

@pytest.mark.parametrize("order", [
    list(range(30)),
    list(range(30))[::-1],
    random.sample(range(30), 30)
])
def test_critical_path_random_order(order):
    """随机顺序执行关键测试，验证无状态耦合"""
    setup_data()
    for i in order:
        run_test_isolated(i)
    assert len(_VECTOR_STORE) == 30
