from __future__ import annotations

import pytest

from src.core.analyzer import CADAnalyzer
from src.models.cad_document import CadDocument


@pytest.mark.asyncio
async def test_classify_with_v16_skips_model_load_without_file_path(monkeypatch) -> None:
    called = {"v16": 0}

    def _boom(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        called["v16"] += 1
        raise AssertionError("_get_v16_classifier should not be called without file_path")

    import src.core.analyzer as analyzer_mod

    monkeypatch.setattr(analyzer_mod, "_get_v16_classifier", _boom)

    analyzer = CADAnalyzer()
    doc = CadDocument(file_name="x.dxf", format="dxf")
    out = await analyzer._classify_with_v16(doc)
    assert out is None
    assert called["v16"] == 0


@pytest.mark.asyncio
async def test_classify_with_ml_skips_model_load_without_file_path(monkeypatch) -> None:
    called = {"v6": 0}

    def _boom(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        called["v6"] += 1
        raise AssertionError("_get_ml_classifier should not be called without file_path")

    import src.core.analyzer as analyzer_mod

    monkeypatch.setattr(analyzer_mod, "_get_ml_classifier", _boom)

    analyzer = CADAnalyzer()
    doc = CadDocument(file_name="x.dxf", format="dxf")
    out = await analyzer._classify_with_ml(doc)
    assert out is None
    assert called["v6"] == 0

