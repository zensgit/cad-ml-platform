"""Design standards query API endpoints.

Exposes deterministic design-guideline knowledge embedded in
`src/core/knowledge/design_standards`.

Scopes (MVP):
- Surface finish (ISO 1302) lookup by grade/application
- General tolerances (ISO 2768-1) lookup for linear/angular
- Preferred diameter selection and basic design feature helpers
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.knowledge.design_standards import (
    GeneralToleranceClass,
    SurfaceFinishGrade,
    get_angular_tolerance,
    get_general_tolerance_table,
    get_linear_tolerance,
    get_standard_chamfer,
    get_standard_fillet,
    get_preferred_diameter,
    get_ra_value,
    get_surface_finish_for_application,
    LINEAR_TOLERANCE_TABLE,
    PREFERRED_DIAMETERS,
    STANDARD_CHAMFERS,
    STANDARD_FILLETS,
    SURFACE_FINISH_TABLE,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _parse_surface_finish_grade(raw: str) -> SurfaceFinishGrade:
    token = (raw or "").strip().upper()
    if not token:
        raise HTTPException(status_code=400, detail="Missing surface finish grade")
    try:
        return SurfaceFinishGrade(token)
    except Exception:
        raise HTTPException(status_code=404, detail="Surface finish grade not found")


def _parse_general_tolerance_class(raw: str) -> GeneralToleranceClass:
    token = (raw or "").strip().lower()
    if not token:
        raise HTTPException(status_code=400, detail="Missing tolerance class")
    try:
        return GeneralToleranceClass(token)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tolerance class (expect f/m/c/v)")


class DesignStandardsStatusResponse(BaseModel):
    status: str
    counts: Dict[str, int]
    examples: Dict[str, str]


@router.get("/status", response_model=DesignStandardsStatusResponse)
async def design_standards_status(
    api_key: str = Depends(get_api_key),
) -> DesignStandardsStatusResponse:
    _ = api_key
    return DesignStandardsStatusResponse(
        status="ok",
        counts={
            "surface_finish_grades": len(SURFACE_FINISH_TABLE),
            "linear_tolerance_ranges": len(LINEAR_TOLERANCE_TABLE),
            "preferred_diameters": len(PREFERRED_DIAMETERS),
            "standard_chamfers": len(STANDARD_CHAMFERS),
            "standard_fillets": len(STANDARD_FILLETS),
        },
        examples={
            "surface_finish_grade": "N7",
            "surface_finish_application": "bearing_journal",
            "linear_tolerance": "dimension_mm=50&tolerance_class=m",
            "angular_tolerance": "length_mm=80&tolerance_class=m",
            "preferred_diameter": "target_mm=23&direction=nearest",
        },
    )


class SurfaceFinishGradeResponse(BaseModel):
    grade: str
    ra_um: float
    rz_um: float
    process_zh: str
    process_en: str
    source: str = Field(default="ISO 1302 (built-in)")


@router.get("/surface-finish/grade", response_model=SurfaceFinishGradeResponse)
async def surface_finish_grade(
    grade: str = Query(..., min_length=2, description="Surface finish grade, e.g. N7"),
    api_key: str = Depends(get_api_key),
) -> SurfaceFinishGradeResponse:
    _ = api_key
    parsed = _parse_surface_finish_grade(grade)
    data = SURFACE_FINISH_TABLE.get(parsed)
    if not data:
        raise HTTPException(status_code=404, detail="Surface finish grade not found")
    ra, rz, process_zh, process_en = data
    return SurfaceFinishGradeResponse(
        grade=parsed.value,
        ra_um=float(ra),
        rz_um=float(rz),
        process_zh=str(process_zh),
        process_en=str(process_en),
    )


class SurfaceFinishGradesListResponse(BaseModel):
    grades: list[SurfaceFinishGradeResponse]
    total: int
    source: str = Field(default="ISO 1302 (built-in)")


@router.get("/surface-finish/grades", response_model=SurfaceFinishGradesListResponse)
async def surface_finish_grades(
    api_key: str = Depends(get_api_key),
) -> SurfaceFinishGradesListResponse:
    _ = api_key
    items: list[SurfaceFinishGradeResponse] = []
    for grade in SurfaceFinishGrade:
        data = SURFACE_FINISH_TABLE.get(grade)
        if not data:
            continue
        ra, rz, process_zh, process_en = data
        items.append(
            SurfaceFinishGradeResponse(
                grade=grade.value,
                ra_um=float(ra),
                rz_um=float(rz),
                process_zh=str(process_zh),
                process_en=str(process_en),
            )
        )
    return SurfaceFinishGradesListResponse(grades=items, total=len(items))


class SurfaceFinishApplicationResponse(BaseModel):
    application: str
    grade: str
    ra_value_um: float
    ra_range_min_um: float
    ra_range_max_um: float
    process_zh: str
    process_en: str
    description_zh: str
    description_en: str
    source: str = Field(default="ISO 1302 (built-in)")


@router.get("/surface-finish/application", response_model=SurfaceFinishApplicationResponse)
async def surface_finish_application(
    application: str = Query(..., min_length=3, description="Application key, e.g. bearing_journal"),
    api_key: str = Depends(get_api_key),
) -> SurfaceFinishApplicationResponse:
    _ = api_key
    result = get_surface_finish_for_application(application)
    if result is None:
        raise HTTPException(status_code=404, detail="Surface finish application not found")
    ra_min, ra_max = result.get("ra_range") or (None, None)
    try:
        ra_min_f = float(ra_min)
        ra_max_f = float(ra_max)
    except Exception:
        ra_min_f = 0.0
        ra_max_f = 0.0
    return SurfaceFinishApplicationResponse(
        application=str(result.get("application", application)),
        grade=str(result.get("grade", "")),
        ra_value_um=float(result.get("ra_value") or 0.0),
        ra_range_min_um=ra_min_f,
        ra_range_max_um=ra_max_f,
        process_zh=str(result.get("process_zh") or ""),
        process_en=str(result.get("process_en") or ""),
        description_zh=str(result.get("description_zh") or ""),
        description_en=str(result.get("description_en") or ""),
    )


class LinearToleranceResponse(BaseModel):
    dimension_mm: float
    tolerance_class: str
    tolerance_plus_mm: float
    tolerance_minus_mm: float
    source: str = Field(default="ISO 2768-1 / GB/T 1804 (built-in)")


@router.get("/general-tolerances/linear", response_model=LinearToleranceResponse)
async def general_linear_tolerance(
    dimension_mm: float = Query(..., gt=0.0, description="Nominal linear dimension in mm"),
    tolerance_class: str = Query("m", min_length=1, description="Tolerance class: f/m/c/v"),
    api_key: str = Depends(get_api_key),
) -> LinearToleranceResponse:
    _ = api_key
    cls = _parse_general_tolerance_class(tolerance_class)
    tol = get_linear_tolerance(float(dimension_mm), cls)
    if tol is None:
        raise HTTPException(status_code=404, detail="Linear tolerance not found for given size/class")
    return LinearToleranceResponse(
        dimension_mm=float(dimension_mm),
        tolerance_class=str(cls.value),
        tolerance_plus_mm=float(tol),
        tolerance_minus_mm=float(-tol),
    )


class AngularToleranceResponse(BaseModel):
    length_mm: float
    tolerance_class: str
    tolerance: str
    source: str = Field(default="ISO 2768-1 / GB/T 1804 (built-in)")


@router.get("/general-tolerances/angular", response_model=AngularToleranceResponse)
async def general_angular_tolerance(
    length_mm: float = Query(..., gt=0.0, description="Shorter side length of angle in mm"),
    tolerance_class: str = Query("m", min_length=1, description="Tolerance class: f/m/c/v"),
    api_key: str = Depends(get_api_key),
) -> AngularToleranceResponse:
    _ = api_key
    cls = _parse_general_tolerance_class(tolerance_class)
    tol = get_angular_tolerance(float(length_mm), cls)
    if tol is None:
        raise HTTPException(status_code=404, detail="Angular tolerance not found for given length/class")
    return AngularToleranceResponse(
        length_mm=float(length_mm),
        tolerance_class=str(cls.value),
        tolerance=str(tol),
    )


class GeneralToleranceTableResponse(BaseModel):
    tolerance_class: str
    linear_table: list[Dict[str, Any]]
    source: str = Field(default="ISO 2768-1 / GB/T 1804 (built-in)")


@router.get("/general-tolerances/table", response_model=GeneralToleranceTableResponse)
async def general_tolerance_table(
    tolerance_class: str = Query("m", min_length=1, description="Tolerance class: f/m/c/v"),
    api_key: str = Depends(get_api_key),
) -> GeneralToleranceTableResponse:
    _ = api_key
    cls = _parse_general_tolerance_class(tolerance_class)
    table = get_general_tolerance_table(cls)
    return GeneralToleranceTableResponse(
        tolerance_class=str(cls.value),
        linear_table=table,
    )


class PreferredDiameterResponse(BaseModel):
    target_mm: float
    direction: str
    preferred_mm: float
    source: str = Field(default="ISO preferred diameters (built-in)")


@router.get("/preferred-diameter", response_model=PreferredDiameterResponse)
async def preferred_diameter(
    target_mm: float = Query(..., gt=0.0, description="Target diameter in mm"),
    direction: str = Query("nearest", description="up | down | nearest"),
    api_key: str = Depends(get_api_key),
) -> PreferredDiameterResponse:
    _ = api_key
    direction_norm = (direction or "").strip().lower() or "nearest"
    if direction_norm not in {"up", "down", "nearest"}:
        raise HTTPException(status_code=400, detail="Invalid direction (expect up/down/nearest)")
    value = get_preferred_diameter(float(target_mm), direction_norm)
    if value is None:
        raise HTTPException(status_code=404, detail="Preferred diameter not found")
    return PreferredDiameterResponse(
        target_mm=float(target_mm),
        direction=direction_norm,
        preferred_mm=float(value),
    )


class PreferredDiametersListResponse(BaseModel):
    min_mm: float
    max_mm: float
    diameters_mm: list[float]
    total: int
    source: str = Field(default="ISO preferred diameters (built-in)")


@router.get(
    "/design-features/preferred-diameters",
    response_model=PreferredDiametersListResponse,
)
async def preferred_diameters_list(
    min_mm: float = Query(0.0, ge=0.0, description="Min diameter (mm), inclusive"),
    max_mm: float = Query(200.0, gt=0.0, description="Max diameter (mm), inclusive"),
    api_key: str = Depends(get_api_key),
) -> PreferredDiametersListResponse:
    _ = api_key
    if max_mm < min_mm:
        raise HTTPException(status_code=400, detail="Invalid range (max_mm < min_mm)")

    from src.core.knowledge.design_standards.design_features import list_preferred_diameters

    values = list_preferred_diameters(min_d=float(min_mm), max_d=float(max_mm))
    return PreferredDiametersListResponse(
        min_mm=float(min_mm),
        max_mm=float(max_mm),
        diameters_mm=[float(v) for v in values],
        total=len(values),
    )


class StandardChamferResponse(BaseModel):
    target_size_mm: float
    designation: str
    size_mm: float
    range_min_mm: float
    range_max_mm: float
    application_zh: str
    application_en: str
    source: str = Field(default="Built-in standard chamfers (guideline)")


@router.get("/design-features/chamfer", response_model=StandardChamferResponse)
async def standard_chamfer(
    target_size_mm: float = Query(..., gt=0.0, description="Target chamfer size in mm"),
    api_key: str = Depends(get_api_key),
) -> StandardChamferResponse:
    _ = api_key
    result = get_standard_chamfer(float(target_size_mm))
    if result is None:
        raise HTTPException(status_code=404, detail="Standard chamfer not found")
    rmin, rmax = result.get("range") or (0.0, 0.0)
    return StandardChamferResponse(
        target_size_mm=float(target_size_mm),
        designation=str(result.get("designation") or ""),
        size_mm=float(result.get("size") or 0.0),
        range_min_mm=float(rmin),
        range_max_mm=float(rmax),
        application_zh=str(result.get("application_zh") or ""),
        application_en=str(result.get("application_en") or ""),
    )


class StandardFilletResponse(BaseModel):
    target_radius_mm: float
    designation: str
    size_mm: float
    range_min_mm: float
    range_max_mm: float
    application_zh: str
    application_en: str
    source: str = Field(default="Built-in standard fillets (guideline)")


@router.get("/design-features/fillet", response_model=StandardFilletResponse)
async def standard_fillet(
    target_radius_mm: float = Query(..., gt=0.0, description="Target fillet radius in mm"),
    api_key: str = Depends(get_api_key),
) -> StandardFilletResponse:
    _ = api_key
    result = get_standard_fillet(float(target_radius_mm))
    if result is None:
        raise HTTPException(status_code=404, detail="Standard fillet not found")
    rmin, rmax = result.get("range") or (0.0, 0.0)
    return StandardFilletResponse(
        target_radius_mm=float(target_radius_mm),
        designation=str(result.get("designation") or ""),
        size_mm=float(result.get("size") or 0.0),
        range_min_mm=float(rmin),
        range_max_mm=float(rmax),
        application_zh=str(result.get("application_zh") or ""),
        application_en=str(result.get("application_en") or ""),
    )


class SurfaceFinishSuggestResponse(BaseModel):
    target_ra_um: float
    grade: str
    ra_um: float
    source: str = Field(default="ISO 1302 (built-in)")


@router.get("/surface-finish/suggest", response_model=SurfaceFinishSuggestResponse)
async def surface_finish_suggest(
    target_ra_um: float = Query(..., gt=0.0, description="Target Ra in micrometers (um)"),
    api_key: str = Depends(get_api_key),
) -> SurfaceFinishSuggestResponse:
    _ = api_key
    from src.core.knowledge.design_standards.surface_finish import suggest_surface_finish

    grade = suggest_surface_finish(float(target_ra_um))
    return SurfaceFinishSuggestResponse(
        target_ra_um=float(target_ra_um),
        grade=grade.value,
        ra_um=float(get_ra_value(grade)),
    )


__all__ = ["router"]
