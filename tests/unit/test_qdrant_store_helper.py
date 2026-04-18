from unittest.mock import patch

from src.core.qdrant_store_helper import get_qdrant_store_or_none


def test_get_qdrant_store_or_none_returns_none_when_backend_is_not_qdrant():
    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "memory"}):
        assert get_qdrant_store_or_none() is None


def test_get_qdrant_store_or_none_returns_store_when_qdrant_backend_enabled():
    sentinel = object()

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.core.vector_stores.get_vector_store",
        return_value=sentinel,
    ) as mocked:
        assert get_qdrant_store_or_none() is sentinel

    mocked.assert_called_once_with("qdrant")


def test_get_qdrant_store_or_none_swallows_factory_errors():
    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.core.vector_stores.get_vector_store",
        side_effect=RuntimeError("boom"),
    ):
        assert get_qdrant_store_or_none() is None
