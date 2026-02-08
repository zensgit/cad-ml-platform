"""Standards library query API endpoints.

Exposes a small, stable HTTP surface for standard-part knowledge that is already
embedded in `src/core/knowledge/standards`.

Scopes (MVP):
- Metric threads (ISO 261/262)
- Rolling bearings (ISO 15)
- O-rings (ISO 3601)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.knowledge.standards import (
    get_bearing_by_bore,
    get_bearing_spec,
    get_oring_by_id,
    get_oring_spec,
    get_tap_drill_size,
    get_thread_spec,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ThreadSpecResponse(BaseModel):
    designation: str
    nominal_diameter_mm: float
    pitch_mm: float
    thread_type: str
    pitch_diameter_mm: float
    minor_diameter_ext_mm: float
    minor_diameter_int_mm: float
    tap_drill_mm: float
    source: str = Field(default="ISO 261/262 (built-in)")


@router.get("/thread", response_model=ThreadSpecResponse)
async def thread_spec(
    designation: str = Query(
        ..., min_length=2, description="Thread designation, e.g. M10 or M10x1.25"
    ),
    api_key: str = Depends(get_api_key),
) -> ThreadSpecResponse:
    _ = api_key
    spec = get_thread_spec(designation)
    if spec is None:
        raise HTTPException(status_code=404, detail="Thread spec not found")
    tap = get_tap_drill_size(spec.designation) or spec.tap_drill_size
    return ThreadSpecResponse(
        designation=spec.designation,
        nominal_diameter_mm=float(spec.nominal_diameter),
        pitch_mm=float(spec.pitch),
        thread_type=str(spec.thread_type.value),
        pitch_diameter_mm=float(spec.pitch_diameter),
        minor_diameter_ext_mm=float(spec.minor_diameter_ext),
        minor_diameter_int_mm=float(spec.minor_diameter_int),
        tap_drill_mm=float(tap),
    )


class BearingSpecResponse(BaseModel):
    designation: str
    bearing_type: str
    series: str
    bore_mm: float
    outer_d_mm: float
    width_mm: float
    dynamic_load_kn: float
    static_load_kn: float
    limiting_speed_grease_rpm: int
    limiting_speed_oil_rpm: int
    weight_kg: float
    source: str = Field(default="ISO 15 (built-in)")


@router.get("/bearing", response_model=BearingSpecResponse)
async def bearing_spec(
    designation: str = Query(
        ..., min_length=3, description="Bearing designation, e.g. 6205 or 6205-2RS"
    ),
    api_key: str = Depends(get_api_key),
) -> BearingSpecResponse:
    _ = api_key
    spec = get_bearing_spec(designation)
    if spec is None:
        raise HTTPException(status_code=404, detail="Bearing spec not found")
    return BearingSpecResponse(
        designation=spec.designation,
        bearing_type=str(spec.bearing_type.value),
        series=str(spec.series.value),
        bore_mm=float(spec.bore_d),
        outer_d_mm=float(spec.outer_d),
        width_mm=float(spec.width_b),
        dynamic_load_kn=float(spec.dynamic_load_c),
        static_load_kn=float(spec.static_load_c0),
        limiting_speed_grease_rpm=int(spec.limiting_speed_grease),
        limiting_speed_oil_rpm=int(spec.limiting_speed_oil),
        weight_kg=float(spec.weight),
    )


class BearingByBoreItem(BaseModel):
    designation: str
    series: str
    bore_mm: float
    outer_d_mm: float
    width_mm: float


class BearingByBoreResponse(BaseModel):
    bore_mm: float
    results: List[BearingByBoreItem]
    total: int


@router.get("/bearing/by-bore", response_model=BearingByBoreResponse)
async def bearings_by_bore(
    bore_mm: float = Query(
        ..., gt=0.0, description="Bearing bore (inner diameter) in mm"
    ),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
    api_key: str = Depends(get_api_key),
) -> BearingByBoreResponse:
    _ = api_key
    items = get_bearing_by_bore(float(bore_mm)) or []
    items = items[: int(limit)]
    return BearingByBoreResponse(
        bore_mm=float(bore_mm),
        results=[
            BearingByBoreItem(
                designation=spec.designation,
                series=str(spec.series.value),
                bore_mm=float(spec.bore_d),
                outer_d_mm=float(spec.outer_d),
                width_mm=float(spec.width_b),
            )
            for spec in items
        ],
        total=len(items),
    )


class ORingSpecResponse(BaseModel):
    designation: str
    inner_diameter_mm: float
    cross_section_mm: float
    id_tolerance_plus_mm: float
    id_tolerance_minus_mm: float
    cs_tolerance_plus_mm: float
    cs_tolerance_minus_mm: float
    groove_width_static_mm: float
    groove_width_dynamic_mm: float
    groove_depth_static_mm: float
    groove_depth_dynamic_mm: float
    standard: str
    source: str = Field(default="ISO 3601 (built-in)")


@router.get("/oring", response_model=ORingSpecResponse)
async def oring_spec(
    designation: str = Query(
        ..., min_length=3, description="O-ring designation, e.g. 20x3"
    ),
    api_key: str = Depends(get_api_key),
) -> ORingSpecResponse:
    _ = api_key
    spec = get_oring_spec(designation)
    if spec is None:
        raise HTTPException(status_code=404, detail="O-ring spec not found")
    return ORingSpecResponse(
        designation=spec.designation,
        inner_diameter_mm=float(spec.inner_diameter),
        cross_section_mm=float(spec.cross_section),
        id_tolerance_plus_mm=float(spec.id_tolerance_plus),
        id_tolerance_minus_mm=float(spec.id_tolerance_minus),
        cs_tolerance_plus_mm=float(spec.cs_tolerance_plus),
        cs_tolerance_minus_mm=float(spec.cs_tolerance_minus),
        groove_width_static_mm=float(spec.groove_width_static),
        groove_width_dynamic_mm=float(spec.groove_width_dynamic),
        groove_depth_static_mm=float(spec.groove_depth_static),
        groove_depth_dynamic_mm=float(spec.groove_depth_dynamic),
        standard=str(spec.standard),
    )


class ORingByIdItem(BaseModel):
    designation: str
    inner_diameter_mm: float
    cross_section_mm: float
    standard: str


class ORingByIdResponse(BaseModel):
    inner_diameter_mm: float
    results: List[ORingByIdItem]
    total: int


@router.get("/oring/by-id", response_model=ORingByIdResponse)
async def orings_by_id(
    inner_diameter_mm: float = Query(..., gt=0.0, description="O-ring ID in mm"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
    api_key: str = Depends(get_api_key),
) -> ORingByIdResponse:
    _ = api_key
    items = get_oring_by_id(float(inner_diameter_mm)) or []
    items = items[: int(limit)]
    return ORingByIdResponse(
        inner_diameter_mm=float(inner_diameter_mm),
        results=[
            ORingByIdItem(
                designation=spec.designation,
                inner_diameter_mm=float(spec.inner_diameter),
                cross_section_mm=float(spec.cross_section),
                standard=str(spec.standard),
            )
            for spec in items
        ],
        total=len(items),
    )


class StandardsStatusResponse(BaseModel):
    status: str
    counts: Dict[str, int]
    examples: Dict[str, str]


@router.get("/status", response_model=StandardsStatusResponse)
async def standards_status(
    api_key: str = Depends(get_api_key),
) -> StandardsStatusResponse:
    _ = api_key

    # Keep this endpoint lightweight; do not dump full databases.
    from src.core.knowledge.standards import (
        BEARING_DATABASE,
        METRIC_THREADS,
        ORING_DATABASE,
    )

    return StandardsStatusResponse(
        status="ok",
        counts={
            "threads": len(METRIC_THREADS),
            "bearings": len(BEARING_DATABASE),
            "orings": len(ORING_DATABASE),
        },
        examples={
            "thread": "M10",
            "bearing": "6205",
            "oring": "20x3",
        },
    )


__all__ = ["router"]
