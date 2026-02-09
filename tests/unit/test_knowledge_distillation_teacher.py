from __future__ import annotations

from typing import Any, List

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from src.ml.knowledge_distillation import TeacherModel


def test_titleblock_teacher_uses_file_bytes(monkeypatch) -> None:
    label_map = {"人孔": 0, "other": 1}
    teacher = TeacherModel(teacher_type="titleblock", label_to_idx=label_map, num_classes=2)

    def fake_read_entities(data: bytes) -> List[Any]:
        assert data == b"dxf-bytes"
        return ["ENTITY"]

    def fake_predict(self, entities: List[Any], bbox=None):  # noqa: ANN001
        assert entities == ["ENTITY"]
        return {"label": "人孔", "confidence": 0.85, "status": "matched"}

    monkeypatch.setattr(
        "src.utils.dxf_io.read_dxf_entities_from_bytes", fake_read_entities, raising=True
    )
    monkeypatch.setattr(
        "src.ml.titleblock_extractor.TitleBlockClassifier.predict", fake_predict, raising=True
    )

    logits = teacher.generate_soft_labels(["masked.dxf"], file_bytes_list=[b"dxf-bytes"])
    assert tuple(logits.shape) == (1, 2)
    assert logits[0, 0] > logits[0, 1]


def test_titleblock_teacher_unknown_label_falls_back_to_other(monkeypatch) -> None:
    label_map = {"other": 0}
    teacher = TeacherModel(teacher_type="titleblock", label_to_idx=label_map, num_classes=1)

    monkeypatch.setattr(
        "src.utils.dxf_io.read_dxf_entities_from_bytes",
        lambda _data: ["ENTITY"],
        raising=True,
    )
    monkeypatch.setattr(
        "src.ml.titleblock_extractor.TitleBlockClassifier.predict",
        lambda _self, _entities, bbox=None: {"label": "不存在的标签", "confidence": 0.9},
        raising=True,
    )

    logits = teacher.generate_soft_labels(["masked.dxf"], file_bytes_list=[b"dxf-bytes"])
    assert tuple(logits.shape) == (1, 1)
    assert float(logits[0, 0].item()) > 0.0

