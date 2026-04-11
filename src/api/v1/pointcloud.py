"""Point Cloud Analysis API endpoints.

Provides:
- POST /api/v1/pointcloud/classify  - Classify a 3D file
- POST /api/v1/pointcloud/features  - Extract feature vector from a 3D file
- POST /api/v1/pointcloud/similar   - Find similar parts
- GET  /api/v1/pointcloud/formats   - List supported file formats
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.ml.pointnet.inference import PointNet3DAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pointcloud"])

# ------------------------------------------------------------------
# Singleton analyser instance
# ------------------------------------------------------------------

_analyzer: Optional[PointNet3DAnalyzer] = None


def _get_analyzer() -> PointNet3DAnalyzer:
    global _analyzer
    if _analyzer is None:
        model_path = os.getenv("POINTNET_MODEL_PATH", None)
        _analyzer = PointNet3DAnalyzer(model_path=model_path)
    return _analyzer


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Confidence score [0, 1]")
    probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Per-class probabilities"
    )
    status: str = Field(..., description="ok or model_unavailable")


class FeaturesResponse(BaseModel):
    vector: List[float] = Field(..., description="Feature vector")
    dimension: int = Field(..., description="Vector dimensionality")
    status: str = Field(..., description="ok or model_unavailable")


class SimilarResult(BaseModel):
    id: str = Field(..., description="Part identifier")
    score: float = Field(..., description="Similarity score")


class SimilarResponse(BaseModel):
    query_vector: List[float] = Field(..., description="Query feature vector")
    dimension: int = Field(..., description="Vector dimensionality")
    top_k: int = Field(..., description="Requested number of results")
    results: List[SimilarResult] = Field(
        default_factory=list, description="Similar parts"
    )
    status: str = Field(..., description="ok or model_unavailable")


class FormatsResponse(BaseModel):
    formats: List[str] = Field(..., description="Supported file extensions")


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


async def _save_upload(upload: UploadFile) -> str:
    """Persist an uploaded file to a temporary path and return that path."""
    suffix = os.path.splitext(upload.filename or "file.stl")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await upload.read()
        tmp.write(content)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/classify", response_model=ClassifyResponse)
async def classify_pointcloud(file: UploadFile = File(...)):
    """Upload a 3D file and return its predicted class."""
    tmp_path = await _save_upload(file)
    try:
        result = _get_analyzer().classify(tmp_path)
        return ClassifyResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)


@router.post("/features", response_model=FeaturesResponse)
async def extract_features(file: UploadFile = File(...)):
    """Upload a 3D file and return its feature vector."""
    tmp_path = await _save_upload(file)
    try:
        result = _get_analyzer().extract_features(tmp_path)
        return FeaturesResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Feature extraction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)


@router.post("/similar", response_model=SimilarResponse)
async def find_similar(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=100, description="Number of similar parts"),
):
    """Upload a 3D file and find similar parts."""
    tmp_path = await _save_upload(file)
    try:
        result = _get_analyzer().find_similar(tmp_path, top_k=top_k)
        return SimilarResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Similarity search failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)


@router.get("/formats", response_model=FormatsResponse)
async def get_supported_formats():
    """Return supported 3D file formats."""
    return FormatsResponse(formats=PointNet3DAnalyzer.supported_formats())
