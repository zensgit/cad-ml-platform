from __future__ import annotations


def test_normalize_dxf_label_maps_known_label() -> None:
    from src.ml.label_normalization import normalize_dxf_label

    assert normalize_dxf_label("对接法兰") == "法兰"


def test_normalize_dxf_label_passthrough_unknown_label() -> None:
    from src.ml.label_normalization import normalize_dxf_label

    assert normalize_dxf_label("不存在的标签") == "不存在的标签"


def test_normalize_dxf_label_returns_default_for_unknown() -> None:
    from src.ml.label_normalization import normalize_dxf_label

    assert normalize_dxf_label("不存在的标签", default="other") == "other"
