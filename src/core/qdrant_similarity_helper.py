"""Shared Qdrant similarity helper for vector pipeline callers."""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.similarity import _cosine


async def compute_qdrant_cosine_similarity(
    qdrant_store: Any,
    reference_id: str,
    candidate_vector: List[float],
) -> Dict[str, Any]:
    """Compute cosine similarity against a stored Qdrant reference vector."""
    reference = await qdrant_store.get_vector(reference_id)
    if reference is None:
        return {
            "reference_id": reference_id,
            "status": "reference_not_found",
            "score": 0.0,
        }

    reference_vector = list(reference.vector or [])
    if len(reference_vector) != len(candidate_vector):
        return {
            "reference_id": reference_id,
            "status": "dimension_mismatch",
            "score": 0.0,
            "method": "cosine",
            "dimension": min(len(reference_vector), len(candidate_vector)),
        }

    score = _cosine(reference_vector, candidate_vector)
    return {
        "reference_id": reference_id,
        "score": round(score, 4),
        "method": "cosine",
        "dimension": len(candidate_vector),
    }


__all__ = ["compute_qdrant_cosine_similarity"]
