from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.classification.baseline_policy import (
    build_baseline_classification_payload,
)


def _build_doc(*, file_name: str = "sample.dxf", text: str | None = None):
    metadata = {"meta": {"drawing_number": "DWG-001"}}
    if text is not None:
        metadata["text"] = text
    return SimpleNamespace(
        file_name=file_name,
        metadata=metadata,
        entities=[
            SimpleNamespace(kind="LINE"),
            SimpleNamespace(kind="LINE"),
            SimpleNamespace(kind="CIRCLE"),
        ],
    )


@pytest.mark.asyncio
async def test_build_baseline_classification_payload_uses_l3_fusion():
    doc = _build_doc()
    features = {"geometric": [0.1], "semantic": [0.2]}
    features_3d = {"faces": 6, "embedding_vector": [0.5, 0.6]}

    class DummyFusion:
        def classify(self, **_: object) -> dict[str, object]:
            return {
                "type": "impeller",
                "confidence": 0.93,
                "alternatives": ["shaft"],
                "fusion_breakdown": {"3d": 0.7, "2d": 0.23},
            }

    async def _classify_part(*_: object) -> dict[str, object]:
        raise AssertionError("L1 classifier should not run when L3 fusion succeeds")

    result = await build_baseline_classification_payload(
        doc=doc,
        features=features,
        features_3d=features_3d,
        classify_part=_classify_part,
        fusion_classifier_factory=DummyFusion,
        build_l2_features_fn=lambda _doc: {"holes": 2},
    )

    assert result["part_type"] == "impeller"
    assert result["confidence"] == 0.93
    assert result["rule_version"] == "L3-Fusion-v1"
    assert result["confidence_source"] == "fusion"
    assert result["confidence_breakdown"] == {"3d": 0.7, "2d": 0.23}


@pytest.mark.asyncio
async def test_build_baseline_classification_payload_falls_back_to_l1_when_l3_fusion_errors():
    doc = _build_doc()
    features = {"geometric": [], "semantic": []}

    async def _classify_part(*_: object) -> dict[str, object]:
        return {
            "type": "simple_plate",
            "confidence": 0.31,
            "rule_version": "v1",
            "alternatives": [],
        }

    class FailingFusion:
        def classify(self, **_: object) -> dict[str, object]:
            raise RuntimeError("fusion unavailable")

    result = await build_baseline_classification_payload(
        doc=doc,
        features=features,
        features_3d={"faces": 2},
        classify_part=_classify_part,
        fusion_classifier_factory=FailingFusion,
        build_l2_features_fn=lambda _doc: {"holes": 1},
    )

    assert result["part_type"] == "simple_plate"
    assert result["confidence"] == 0.31
    assert result["rule_version"] == "v1"
    assert result["confidence_source"] == "rules"


@pytest.mark.asyncio
async def test_build_baseline_classification_payload_adopts_l2_fusion_for_2d_docs():
    doc = _build_doc(file_name="manhole.dxf", text="inspection opening")
    features = {"geometric": [], "semantic": []}

    async def _classify_part(*_: object) -> dict[str, object]:
        return {
            "type": "unknown",
            "confidence": 0.12,
            "rule_version": "v1",
            "alternatives": [],
        }

    class DummyFusion:
        def classify(self, **_: object) -> dict[str, object]:
            return {
                "type": "人孔",
                "confidence": 0.89,
                "alternatives": ["法兰"],
                "fusion_breakdown": {"text": 0.6, "geom": 0.29},
            }

    result = await build_baseline_classification_payload(
        doc=doc,
        features=features,
        features_3d={},
        classify_part=_classify_part,
        fusion_classifier_factory=DummyFusion,
        build_l2_features_fn=lambda _doc: {"holes": 3},
    )

    assert result["part_type"] == "人孔"
    assert result["confidence"] == 0.89
    assert result["rule_version"] == "L2-Fusion-v1"
    assert result["confidence_source"] == "fusion"


@pytest.mark.asyncio
async def test_build_baseline_classification_payload_keeps_l1_when_l2_fusion_is_unknown():
    doc = _build_doc(file_name="plate.dxf")
    features = {"geometric": [], "semantic": []}

    async def _classify_part(*_: object) -> dict[str, object]:
        return {
            "type": "simple_plate",
            "confidence": 0.42,
            "rule_version": "v1",
            "alternatives": ["other"],
        }

    class DummyFusion:
        def classify(self, **_: object) -> dict[str, object]:
            return {
                "type": "unknown",
                "confidence": 0.0,
                "alternatives": [],
                "fusion_breakdown": {"text": 0.0},
            }

    result = await build_baseline_classification_payload(
        doc=doc,
        features=features,
        features_3d=None,
        classify_part=_classify_part,
        fusion_classifier_factory=DummyFusion,
        build_l2_features_fn=lambda _doc: {"holes": 0},
    )

    assert result["part_type"] == "simple_plate"
    assert result["confidence"] == 0.42
    assert result["rule_version"] == "v1"
    assert result["confidence_source"] == "rules"
