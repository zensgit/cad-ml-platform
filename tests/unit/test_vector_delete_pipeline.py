from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.core.errors_extended import ErrorCode
from src.core.vector_delete_pipeline import run_vector_delete_pipeline


class _Response(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _Payload:
    def __init__(self, vector_id: str) -> None:
        self.id = vector_id


@pytest.mark.asyncio
async def test_run_vector_delete_pipeline_uses_qdrant_branch():
    class _Store:
        async def get_vector(self, vector_id):  # noqa: ANN001, ANN201
            return {"id": vector_id}

        async def delete_vector(self, vector_id):  # noqa: ANN001, ANN201
            return vector_id == "vec-q"

    result = await run_vector_delete_pipeline(
        payload=_Payload("vec-q"),
        response_cls=_Response,
        error_code_cls=ErrorCode,
        build_error_fn=lambda code, **kwargs: {"code": code.value, **kwargs},
        get_qdrant_store_fn=lambda: _Store(),
        get_client_fn=lambda: None,
    )

    assert result["id"] == "vec-q"
    assert result["status"] == "deleted"


@pytest.mark.asyncio
async def test_run_vector_delete_pipeline_qdrant_not_found_raises_404():
    class _Store:
        async def get_vector(self, _vector_id):  # noqa: ANN001, ANN201
            return None

        async def delete_vector(self, _vector_id):  # noqa: ANN001, ANN201
            return False

    with pytest.raises(HTTPException) as exc_info:
        await run_vector_delete_pipeline(
            payload=_Payload("missing"),
            response_cls=_Response,
            error_code_cls=ErrorCode,
            build_error_fn=lambda code, **kwargs: {"code": code.value, **kwargs},
            get_qdrant_store_fn=lambda: _Store(),
            get_client_fn=lambda: None,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["code"] == ErrorCode.DATA_NOT_FOUND.value

