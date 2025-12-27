from src.core.similarity import _VECTOR_STORE, last_vector_error, register_vector  # type: ignore


def test_vector_register_dimension_error_structured():
    # establish baseline vector to set expected dimension
    ok = register_vector("baseline_dim", [0.1] * 4)
    assert ok is True
    bad = register_vector("bad_dim_vec", [0.2] * 3)
    assert bad is False
    err = last_vector_error()
    assert err is not None
    assert err.get("code") == "DIMENSION_MISMATCH"
    assert err.get("stage") == "vector_register"
    assert err.get("id") == "bad_dim_vec"
    assert err.get("expected") == "4"
    assert err.get("found") == "3"
    assert "bad_dim_vec" not in _VECTOR_STORE
