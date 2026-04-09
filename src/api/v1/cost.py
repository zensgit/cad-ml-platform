"""Manufacturing cost estimation API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.ml.cost.estimator import CostEstimator
from src.ml.cost.models import CostEstimateRequest, CostEstimateResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cost"])

# Singleton estimator (loaded once at import time).
_estimator: CostEstimator | None = None


def _get_estimator() -> CostEstimator:
    """Lazy-initialise the shared CostEstimator instance."""
    global _estimator  # noqa: PLW0603
    if _estimator is None:
        _estimator = CostEstimator()
    return _estimator


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@router.post(
    "/estimate",
    response_model=CostEstimateResponse,
    summary="Estimate manufacturing cost for a single part",
)
async def estimate_cost(request: CostEstimateRequest) -> CostEstimateResponse:
    """Return an itemised cost estimate with optimistic / pessimistic bounds.

    The estimate is driven by the cost model configuration in
    ``config/cost_model.yaml`` and a set of heuristic rules that translate
    geometric features into machining time and material consumption.
    """
    try:
        estimator = _get_estimator()
        return estimator.estimate(request)
    except Exception as exc:
        logger.exception("Cost estimation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/batch-estimate",
    response_model=List[CostEstimateResponse],
    summary="Estimate manufacturing cost for multiple parts",
)
async def batch_estimate_cost(
    requests: List[CostEstimateRequest],
) -> List[CostEstimateResponse]:
    """Return cost estimates for a batch of parts in a single call."""
    if not requests:
        return []
    try:
        estimator = _get_estimator()
        return [estimator.estimate(req) for req in requests]
    except Exception as exc:
        logger.exception("Batch cost estimation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/materials",
    response_model=Dict[str, Any],
    summary="List available materials and their cost-model properties",
)
async def list_materials() -> Dict[str, Any]:
    """Return the material catalogue loaded from the cost model configuration."""
    estimator = _get_estimator()
    return {"materials": estimator.materials}
