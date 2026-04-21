from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.errors_extended import ErrorCode
from src.core.vector_register_pipeline import run_vector_register_pipeline


class _Response(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.mark.asyncio
async def test_run_vector_register_pipeline_uses_qdrant_branch():
    payload = SimpleNamespace(id="vec-q", vector=[0.1] * 7, meta={"material": "steel"})

    class _Store:
        def __init__(self) -> None:
            self.calls = []

        async def register_vector(self, vector_id, vector, metadata=None):  # noqa: ANN001, ANN201
            self.calls.append((vector_id, vector, metadata))
            return True

    store = _Store()
    result = await run_vector_register_pipeline(
        payload=payload,
        response_cls=_Response,
        error_code_cls=ErrorCode,
        build_error_fn=lambda code, **kwargs: {"code": code.value, **kwargs},
        get_qdrant_store_fn=lambda: store,
    )

    assert result["status"] == "accepted"
    assert result["dimension"] == 7
    _, _, metadata = store.calls[0]
    assert metadata["material"] == "steel"
    assert metadata["total_dim"] == "7"

