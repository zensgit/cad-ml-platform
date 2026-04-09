from src.core.classification.coarse_labels import (
    labels_conflict,
    normalize_coarse_label,
)


def test_normalize_coarse_label_maps_fine_labels() -> None:
    assert normalize_coarse_label("人孔") == "开孔件"
    assert normalize_coarse_label("搅拌轴组件") == "传动件"
    assert normalize_coarse_label("传动件") == "传动件"


def test_labels_conflict_compares_after_normalization() -> None:
    assert labels_conflict("人孔", "传动件") is True
    assert labels_conflict("人孔", "捕集口") is False
    assert labels_conflict("", "传动件") is False
