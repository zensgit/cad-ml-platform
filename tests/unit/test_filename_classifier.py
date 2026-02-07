import json
from pathlib import Path

import pytest

from src.ml.filename_classifier import FilenameClassifier
from src.ml.hybrid_classifier import DecisionSource, HybridClassifier


def _write_synonyms(path: Path) -> None:
    data = {
        "人孔": ["人孔"],
        "出料凸缘": ["出料凸缘"],
        "调节螺栓": ["调节螺栓"],
        "拖车": ["拖车"],
        "传动件": ["传动件"],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def synonyms_file(tmp_path: Path) -> Path:
    path = tmp_path / "synonyms.json"
    _write_synonyms(path)
    return path


def test_extract_part_name_patterns(synonyms_file: Path) -> None:
    classifier = FilenameClassifier(synonyms_path=str(synonyms_file))
    assert classifier.extract_part_name("J2925001-01人孔v2.dxf") == "人孔"
    assert classifier.extract_part_name("BTJ01239901522-00拖轮组件v1.dxf") == "拖轮组件"
    assert classifier.extract_part_name("BTJ01231201522-00拖车DN1500v1.dxf") == "拖车"
    assert (
        classifier.extract_part_name("比较_LTJ012306102-0084调节螺栓v1 vs v2.dxf")
        == "调节螺栓"
    )


def test_match_label_exact_and_partial(synonyms_file: Path) -> None:
    classifier = FilenameClassifier(synonyms_path=str(synonyms_file))

    label, conf, match_type = classifier.match_label("人孔")
    assert label == "人孔"
    assert conf == pytest.approx(classifier.exact_match_conf)
    assert match_type == "exact"

    label, conf, match_type = classifier.match_label("人孔盖板")
    assert label == "人孔"
    assert conf == pytest.approx(classifier.fuzzy_match_conf)
    assert match_type in {"partial_high", "partial_low"}

    # Suffix normalization is done at extraction time; `predict()` should hit exact match.
    result = classifier.predict("BTJ01231201522-00拖车DN1500v1.dxf")
    assert result["extracted_name"] == "拖车"
    assert result["label"] == "拖车"
    assert result["confidence"] == pytest.approx(classifier.exact_match_conf)
    assert result["match_type"] == "exact"


class _DummyFilenameClassifier:
    def __init__(self, label: str, confidence: float):
        self._label = label
        self._confidence = confidence

    def predict(self, filename: str):
        return {
            "label": self._label,
            "confidence": self._confidence,
            "match_type": "exact",
            "extracted_name": self._label,
            "status": "matched",
        }


def test_hybrid_prefers_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FILENAME_CLASSIFIER_ENABLED", raising=False)
    monkeypatch.delenv("GRAPH2D_ENABLED", raising=False)

    classifier = HybridClassifier(filename_min_conf=0.8, graph2d_min_conf=0.5)
    classifier._filename_classifier = _DummyFilenameClassifier("人孔", 0.9)

    result = classifier.classify(
        filename="J2925001-01人孔v2.dxf",
        graph2d_result={"label": "传动件", "confidence": 0.95},
    )

    assert result.label == "人孔"
    assert result.source == DecisionSource.FILENAME


def test_hybrid_prefers_graph2d_when_filename_low(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FILENAME_CLASSIFIER_ENABLED", raising=False)
    monkeypatch.delenv("GRAPH2D_ENABLED", raising=False)

    classifier = HybridClassifier(filename_min_conf=0.8, graph2d_min_conf=0.5)
    classifier._filename_classifier = _DummyFilenameClassifier("人孔", 0.3)

    result = classifier.classify(
        filename="J2925001-01人孔v2.dxf",
        graph2d_result={"label": "传动件", "confidence": 0.9},
    )

    assert result.label == "传动件"
    assert result.source == DecisionSource.GRAPH2D


def test_hybrid_fusion_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FILENAME_CLASSIFIER_ENABLED", raising=False)
    monkeypatch.delenv("GRAPH2D_ENABLED", raising=False)

    classifier = HybridClassifier(filename_min_conf=0.8, graph2d_min_conf=0.9)
    classifier._filename_classifier = _DummyFilenameClassifier("人孔", 0.6)

    result = classifier.classify(
        filename="J2925001-01人孔v2.dxf",
        graph2d_result={"label": "传动件", "confidence": 0.8},
    )

    assert result.label == "人孔"
    assert result.source in {DecisionSource.FILENAME, DecisionSource.FUSION}
