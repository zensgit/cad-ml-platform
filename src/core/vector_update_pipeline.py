from __future__ import annotations

import os
from typing import Any, Optional, Protocol

from fastapi import HTTPException

from src.core.errors_extended import ErrorCode, build_error


class VectorUpdatePayload(Protocol):
    id: str
    replace: Optional[list[float]]
    append: Optional[list[float]]
    material: Optional[str]
    complexity: Optional[str]
    format: Optional[str]


async def run_vector_update_pipeline(
    *,
    payload: VectorUpdatePayload,
    qdrant_store=None,
) -> dict[str, Any]:
    from src.utils.analysis_metrics import analysis_error_code_total

    enforce = os.getenv("ANALYSIS_VECTOR_DIM_CHECK", "0") == "1"

    if qdrant_store is not None:
        current = await qdrant_store.get_vector(payload.id)
        if current is None:
            err = build_error(
                ErrorCode.DATA_NOT_FOUND,
                stage="vector_update",
                message="Vector not found",
                id=payload.id,
            )
            analysis_error_code_total.labels(code=ErrorCode.DATA_NOT_FOUND.value).inc()
            return {"id": payload.id, "status": "not_found", "error": err}

        vec = list(current.vector or [])
        meta = dict(current.metadata or {})
        original_dim = len(vec)

        try:
            vec = _apply_vector_update(
                payload=payload,
                vec=vec,
                original_dim=original_dim,
                enforce=enforce,
            )
            _update_vector_meta(meta, payload)
            meta["total_dim"] = str(len(vec))
            await qdrant_store.register_vector(payload.id, vec, metadata=meta)
            return {
                "id": payload.id,
                "status": "updated",
                "dimension": len(vec),
                "feature_version": meta.get("feature_version"),
            }
        except HTTPException:
            raise
        except ValueError as exc:
            if exc.args and exc.args[0] == "dimension_mismatch":
                _, found = exc.args
                return _build_dimension_mismatch_response(
                    payload_id=payload.id,
                    original_dim=original_dim,
                    found=found,
                )
            raise
        except Exception as exc:
            err = build_error(
                ErrorCode.INTERNAL_ERROR,
                stage="vector_update",
                message=str(exc),
                id=payload.id,
            )
            analysis_error_code_total.labels(code=ErrorCode.INTERNAL_ERROR.value).inc()
            return {"id": payload.id, "status": "error", "error": err}

    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    if payload.id not in _VECTOR_STORE:
        err = build_error(
            ErrorCode.DATA_NOT_FOUND,
            stage="vector_update",
            message="Vector not found",
            id=payload.id,
        )
        analysis_error_code_total.labels(code=ErrorCode.DATA_NOT_FOUND.value).inc()
        return {"id": payload.id, "status": "not_found", "error": err}

    vec = _VECTOR_STORE[payload.id]
    original_dim = len(vec)
    try:
        updated = _apply_vector_update(
            payload=payload,
            vec=list(vec),
            original_dim=original_dim,
            enforce=enforce,
        )
        _VECTOR_STORE[payload.id] = updated
        meta = _VECTOR_META.get(payload.id, {})
        _update_vector_meta(meta, payload)
        _VECTOR_META[payload.id] = meta
        return {
            "id": payload.id,
            "status": "updated",
            "dimension": len(_VECTOR_STORE[payload.id]),
            "feature_version": _VECTOR_META.get(payload.id, {}).get("feature_version"),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        if exc.args and exc.args[0] == "dimension_mismatch":
            _, found = exc.args
            return _build_dimension_mismatch_response(
                payload_id=payload.id,
                original_dim=original_dim,
                found=found,
            )
        raise
    except Exception as exc:
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="vector_update",
            message=str(exc),
            id=payload.id,
        )
        analysis_error_code_total.labels(code=ErrorCode.INTERNAL_ERROR.value).inc()
        return {"id": payload.id, "status": "error", "error": err}


def _apply_vector_update(
    *,
    payload: VectorUpdatePayload,
    vec: list[float],
    original_dim: int,
    enforce: bool,
) -> list[float]:
    from src.utils.analysis_metrics import (
        analysis_error_code_total,
        vector_dimension_rejections_total,
    )

    if payload.replace is not None:
        if len(payload.replace) != original_dim:
            if enforce:
                err = build_error(
                    ErrorCode.DIMENSION_MISMATCH,
                    stage="vector_update",
                    message=f"Expected {original_dim}, got {len(payload.replace)}",
                    id=payload.id,
                    expected=original_dim,
                    found=len(payload.replace),
                )
                analysis_error_code_total.labels(
                    code=ErrorCode.DIMENSION_MISMATCH.value
                ).inc()
                vector_dimension_rejections_total.labels(
                    reason="dimension_mismatch_replace"
                ).inc()
                raise HTTPException(status_code=409, detail=err)
            raise ValueError("dimension_mismatch", len(payload.replace))
        return [float(x) for x in payload.replace]

    if payload.append is not None:
        if enforce and original_dim != 0:
            new_dim = original_dim + len(payload.append)
            if new_dim != original_dim:
                err = build_error(
                    ErrorCode.DIMENSION_MISMATCH,
                    stage="vector_update",
                    message=f"Append changes dimension {original_dim}->{new_dim}",
                    id=payload.id,
                    expected=original_dim,
                    found=new_dim,
                )
                analysis_error_code_total.labels(
                    code=ErrorCode.DIMENSION_MISMATCH.value
                ).inc()
                vector_dimension_rejections_total.labels(
                    reason="dimension_mismatch_append"
                ).inc()
                raise HTTPException(status_code=409, detail=err)
        return vec + [float(float(x)) for x in payload.append]

    return vec


def _update_vector_meta(meta: dict[str, Any], payload: VectorUpdatePayload) -> None:
    if payload.material is not None:
        meta["material"] = payload.material
    if payload.complexity is not None:
        meta["complexity"] = payload.complexity
    if payload.format is not None:
        meta["format"] = payload.format


def _build_dimension_mismatch_response(
    *,
    payload_id: str,
    original_dim: int,
    found: int,
) -> dict[str, Any]:
    return {
        "id": payload_id,
        "status": "dimension_mismatch",
        "dimension": original_dim,
        "error": {
            "code": ErrorCode.DIMENSION_MISMATCH.value,
            "expected": original_dim,
            "found": found,
            "id": payload_id,
        },
    }


__all__ = ["run_vector_update_pipeline"]
