from src.core.similarity import _VECTOR_STORE, register_vector  # type: ignore


def test_vector_dimension_enforcement():
    initial_count = len(_VECTOR_STORE)
    ok = register_vector("vec_dim_ok", [0.1] * 7, meta={"material": "steel"})
    assert ok is True
    assert "vec_dim_ok" in _VECTOR_STORE
    bad = register_vector("vec_dim_bad", [0.2] * 5, meta={"material": "steel"})
    assert bad is False
    assert "vec_dim_bad" not in _VECTOR_STORE
    assert len(_VECTOR_STORE) == initial_count + 1


def test_vector_dimension_enforcement_disabled(disable_dimension_enforcement):
    # With enforcement disabled vectors of different dims should be accepted
    ok1 = register_vector("vec_a", [0.1, 0.2], meta={"material": "steel"})
    ok2 = register_vector("vec_b", [0.3] * 5, meta={"material": "aluminum"})
    assert ok1 is True and ok2 is True
    assert "vec_a" in _VECTOR_STORE and "vec_b" in _VECTOR_STORE
