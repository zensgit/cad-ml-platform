"""Drawing version diff API endpoints.

Provides REST endpoints for comparing two DXF drawing revisions, generating
diff reports, and producing Engineering Change Notices.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.diff.annotation_diff import AnnotationDiff
from src.core.diff.geometry_diff import GeometryDiff
from src.core.diff.models import DiffResult, EntityChange
from src.core.diff.report import DiffReportGenerator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["diff"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------


class EntityChangeResponse(BaseModel):
    entity_type: str
    change_type: str
    location: List[float]
    details: Dict[str, Any] = Field(default_factory=dict)


class DiffResultResponse(BaseModel):
    added: List[EntityChangeResponse] = Field(default_factory=list)
    removed: List[EntityChangeResponse] = Field(default_factory=list)
    modified: List[EntityChangeResponse] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    change_regions: List[Dict[str, Any]] = Field(default_factory=list)


class ReportResponse(BaseModel):
    markdown: str


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _diff_result_to_response(result: DiffResult) -> DiffResultResponse:
    """Convert internal DiffResult dataclass to Pydantic response model."""

    def _convert(change: EntityChange) -> EntityChangeResponse:
        return EntityChangeResponse(
            entity_type=change.entity_type,
            change_type=change.change_type,
            location=list(change.location),
            details=change.details,
        )

    return DiffResultResponse(
        added=[_convert(c) for c in result.added],
        removed=[_convert(c) for c in result.removed],
        modified=[_convert(c) for c in result.modified],
        summary=result.summary,
        change_regions=result.change_regions,
    )


async def _save_upload(upload: UploadFile, suffix: str = ".dxf") -> str:
    """Persist an uploaded file to a temp path and return the path."""
    content = await upload.read()
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return path


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/compare", response_model=DiffResultResponse)
async def compare_files(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
) -> DiffResultResponse:
    """Upload two DXF files and return full geometry diff."""
    path_a = await _save_upload(file_a)
    path_b = await _save_upload(file_b)
    try:
        differ = GeometryDiff()
        result = differ.compare(path_a, path_b)
        return _diff_result_to_response(result)
    except Exception:
        logger.exception("Geometry diff failed")
        raise HTTPException(status_code=500, detail="Diff comparison failed")
    finally:
        for p in (path_a, path_b):
            try:
                os.unlink(p)
            except OSError:
                pass


@router.post("/annotations", response_model=DiffResultResponse)
async def compare_annotations(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
) -> DiffResultResponse:
    """Upload two DXF files and return annotation-only diff."""
    path_a = await _save_upload(file_a)
    path_b = await _save_upload(file_b)
    try:
        differ = AnnotationDiff()
        result = differ.compare(path_a, path_b)
        return _diff_result_to_response(result)
    except Exception:
        logger.exception("Annotation diff failed")
        raise HTTPException(status_code=500, detail="Annotation diff failed")
    finally:
        for p in (path_a, path_b):
            try:
                os.unlink(p)
            except OSError:
                pass


@router.post("/report", response_model=ReportResponse)
async def generate_report(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    part_number: str = Form(default="UNKNOWN"),
    api_key: str = Depends(get_api_key),
) -> ReportResponse:
    """Upload two DXF files and return a Markdown diff report."""
    path_a = await _save_upload(file_a)
    path_b = await _save_upload(file_b)
    try:
        differ = GeometryDiff()
        result = differ.compare(path_a, path_b)
        generator = DiffReportGenerator()
        name_a = file_a.filename or "file_a.dxf"
        name_b = file_b.filename or "file_b.dxf"
        markdown = generator.generate_markdown(result, name_a, name_b)
        return ReportResponse(markdown=markdown)
    except Exception:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail="Report generation failed")
    finally:
        for p in (path_a, path_b):
            try:
                os.unlink(p)
            except OSError:
                pass


@router.post("/ecn", response_model=ReportResponse)
async def generate_ecn(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    part_number: str = Form(default="UNKNOWN"),
    revision: str = Form(default="A"),
    api_key: str = Depends(get_api_key),
) -> ReportResponse:
    """Upload two DXF files and return an Engineering Change Notice."""
    path_a = await _save_upload(file_a)
    path_b = await _save_upload(file_b)
    try:
        differ = GeometryDiff()
        result = differ.compare(path_a, path_b)
        generator = DiffReportGenerator()
        markdown = generator.generate_ecn(result, part_number, revision)
        return ReportResponse(markdown=markdown)
    except Exception:
        logger.exception("ECN generation failed")
        raise HTTPException(status_code=500, detail="ECN generation failed")
    finally:
        for p in (path_a, path_b):
            try:
                os.unlink(p)
            except OSError:
                pass
