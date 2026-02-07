"""Tolerance and fits query API endpoints (ISO 286 / GB/T 1800)."""

from __future__ import annotations

import logging
import re
from typing import Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.knowledge.tolerance import get_fit_deviations, get_limit_deviations, get_tolerance_value

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_fit_code(fit_code: str) -> str:
    """Best-effort normalization for fit codes like `h7/G6` -> `H7/g6`."""
    raw = (fit_code or "").strip()
    if not raw or "/" not in raw:
        return raw

    hole_raw, shaft_raw = [p.strip() for p in raw.split("/", 1)]

    def _split_symbol_grade(spec: str) -> Tuple[str, str]:
        m = re.match(r"^([A-Za-z]{1,3})(\d{1,2})$", spec.strip())
        if not m:
            return spec.strip(), ""
        return m.group(1), m.group(2)

    hole_symbol, hole_grade = _split_symbol_grade(hole_raw)
    shaft_symbol, shaft_grade = _split_symbol_grade(shaft_raw)
    if not hole_grade or not shaft_grade:
        return raw

    return f"{hole_symbol.upper()}{hole_grade}/{shaft_symbol.lower()}{shaft_grade}"


class ITToleranceResponse(BaseModel):
    nominal_size_mm: float = Field(..., description="Nominal size / diameter in mm")
    grade: str = Field(..., description="IT grade (e.g., IT7)")
    tolerance_um: float = Field(..., description="Tolerance in micrometers (um)")
    source: str = Field(
        default="ISO 286-1 / GB/T 1800.1",
        description="Reference standard/source",
    )


@router.get("/it", response_model=ITToleranceResponse)
async def it_tolerance(
    diameter_mm: float = Query(..., gt=0.0, description="Nominal size / diameter in mm"),
    grade: str = Query(..., min_length=2, description="IT grade, e.g. IT7"),
    api_key: str = Depends(get_api_key),
) -> ITToleranceResponse:
    _ = api_key
    value = get_tolerance_value(diameter_mm, grade.strip().upper())
    if value is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported size/grade. Expect 0 < diameter_mm <= 3150 and grade like IT7.",
        )
    return ITToleranceResponse(
        nominal_size_mm=float(diameter_mm),
        grade=grade.strip().upper(),
        tolerance_um=float(value),
    )


class LimitDeviationsResponse(BaseModel):
    nominal_size_mm: float = Field(..., description="Nominal size / diameter in mm")
    symbol: str = Field(..., description="Fundamental deviation symbol (e.g., H, h, g, JS)")
    grade: int = Field(..., description="Tolerance grade number (e.g., 7 for H7)")
    lower_deviation_um: float = Field(..., description="Lower deviation in micrometers (um)")
    upper_deviation_um: float = Field(..., description="Upper deviation in micrometers (um)")
    label: str = Field(..., description="Normalized label (e.g., H7, g6)")
    source: str = Field(
        default="ISO 286-2 table (data/knowledge/iso286_deviations.json)",
        description="Reference data source",
    )


@router.get("/limit-deviations", response_model=LimitDeviationsResponse)
async def limit_deviations(
    symbol: str = Query(..., min_length=1, description="Symbol, e.g. H, h, g, JS"),
    grade: int = Query(..., ge=1, le=18, description="Grade number, e.g. 7"),
    diameter_mm: float = Query(..., gt=0.0, description="Nominal size / diameter in mm"),
    api_key: str = Depends(get_api_key),
) -> LimitDeviationsResponse:
    _ = api_key
    parsed = get_limit_deviations(symbol, grade, diameter_mm)
    if parsed is None:
        raise HTTPException(status_code=404, detail="Limit deviations not found for given symbol/grade/size.")
    lower, upper = parsed
    symbol_clean = symbol.strip()
    # `get_limit_deviations` normalizes hole symbols to uppercase and shafts to lowercase.
    label = (
        f"{symbol_clean.upper()}{grade}"
        if symbol_clean.isupper()
        else f"{symbol_clean.lower()}{grade}"
    )
    return LimitDeviationsResponse(
        nominal_size_mm=float(diameter_mm),
        symbol=symbol_clean,
        grade=int(grade),
        lower_deviation_um=float(lower),
        upper_deviation_um=float(upper),
        label=label,
    )


class FitDeviationsResponse(BaseModel):
    fit_code: str = Field(..., description="Fit code (e.g., H7/g6)")
    nominal_size_mm: float = Field(..., description="Nominal size / diameter in mm")
    fit_type: str = Field(..., description="Fit type: clearance/transition/interference")
    hole_upper_deviation_um: float
    hole_lower_deviation_um: float
    shaft_upper_deviation_um: float
    shaft_lower_deviation_um: float
    max_clearance_um: float
    min_clearance_um: float
    source: str = Field(
        default="ISO 286-1/2 (computed from IT grades + fundamental deviations)",
        description="Computation basis",
    )


@router.get("/fit", response_model=FitDeviationsResponse)
async def fit_deviations(
    fit_code: str = Query(..., min_length=3, description="Fit code, e.g. H7/g6"),
    diameter_mm: float = Query(..., gt=0.0, description="Nominal size / diameter in mm"),
    api_key: str = Depends(get_api_key),
) -> FitDeviationsResponse:
    _ = api_key
    normalized = _normalize_fit_code(fit_code)
    result = get_fit_deviations(normalized, diameter_mm)
    if result is None:
        raise HTTPException(status_code=404, detail="Fit code not supported or out of range.")
    return FitDeviationsResponse(
        fit_code=result.fit_code,
        nominal_size_mm=float(result.nominal_size_mm),
        fit_type=str(result.fit_type.value),
        hole_upper_deviation_um=float(result.hole_upper_deviation_um),
        hole_lower_deviation_um=float(result.hole_lower_deviation_um),
        shaft_upper_deviation_um=float(result.shaft_upper_deviation_um),
        shaft_lower_deviation_um=float(result.shaft_lower_deviation_um),
        max_clearance_um=float(result.max_clearance_um),
        min_clearance_um=float(result.min_clearance_um),
    )


__all__ = ["router"]
