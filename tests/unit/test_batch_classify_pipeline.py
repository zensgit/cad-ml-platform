from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from src.core.classification.decision_service import DECISION_CONTRACT_VERSION
from src.core.classification.batch_classify_pipeline import (
    build_batch_classify_item,
    run_batch_classify_pipeline,
)


def _upload(filename: str, content: bytes = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n"):
    return UploadFile(filename=filename, file=io.BytesIO(content))


def test_build_batch_classify_item_exposes_fine_coarse_contract() -> None:
    item = build_batch_classify_item(
        file_name="part.dxf",
        category="人孔",
        confidence=0.91,
        probabilities={"人孔": 0.91},
        classifier="ml_v6",
    )

    assert item["category"] == "人孔"
    assert item["fine_category"] == "人孔"
    assert item["coarse_category"] == "开孔件"
    assert item["is_coarse_label"] is False
    assert item["fine_part_type"] == "人孔"
    assert item["coarse_part_type"] == "开孔件"
    assert item["decision_source"] == "ml_v6"
    assert item["contract_version"] == DECISION_CONTRACT_VERSION
    assert item["decision_contract"]["contract_version"] == DECISION_CONTRACT_VERSION
    assert item["decision_contract"]["fine_part_type"] == "人孔"
    assert item["evidence"][0]["source"] == "baseline"
    assert item["evidence"][1]["source"] == "part_classifier"


@pytest.mark.asyncio
async def test_run_batch_classify_pipeline_keeps_invalid_prefix_aligned_in_v6_fallback():
    valid_result = SimpleNamespace(
        category="人孔",
        confidence=0.91,
        probabilities={"人孔": 0.91, "法兰": 0.09},
        needs_review=True,
        review_reason="edge_case",
        top2_category="法兰",
        top2_confidence=0.09,
    )
    ml_classifier = MagicMock()
    ml_classifier.predict.return_value = valid_result

    result = await run_batch_classify_pipeline(
        files=[_upload("bad.pdf", b"pdf"), _upload("good.dxf")],
        max_workers=None,
        logger=MagicMock(),
        get_v16_classifier=lambda: None,
        get_ml_classifier=lambda: ml_classifier,
    )

    assert result["total"] == 2
    assert result["success"] == 1
    assert result["failed"] == 1
    assert "unsupported" in result["results"][0]["error"].lower()
    assert result["results"][1]["file_name"] == "good.dxf"
    assert result["results"][1]["coarse_category"] == "开孔件"
    assert result["results"][1]["is_coarse_label"] is False
    assert result["results"][1]["decision_contract"]["coarse_part_type"] == "开孔件"


@pytest.mark.asyncio
async def test_run_batch_classify_pipeline_keeps_invalid_prefix_aligned_in_v16_batch():
    batch_result = SimpleNamespace(
        category="壳体类",
        confidence=0.88,
        probabilities={"壳体类": 0.88},
        needs_review=False,
        review_reason=None,
        top2_category=None,
        top2_confidence=None,
        model_version="v16",
    )
    classifier = MagicMock()
    classifier.predict_batch.return_value = [batch_result]

    result = await run_batch_classify_pipeline(
        files=[_upload("bad.txt", b"txt"), _upload("good.dxf")],
        max_workers=2,
        logger=MagicMock(),
        get_v16_classifier=lambda: classifier,
        get_ml_classifier=lambda: None,
    )

    assert result["total"] == 2
    assert result["success"] == 1
    assert result["failed"] == 1
    assert "unsupported" in result["results"][0]["error"].lower()
    assert result["results"][1]["file_name"] == "good.dxf"
    assert result["results"][1]["fine_category"] == "壳体类"
    assert result["results"][1]["contract_version"] == DECISION_CONTRACT_VERSION
    classifier.predict_batch.assert_called_once()
