"""Unit tests for OCRManager provider strategy normalization."""

from __future__ import annotations

from src.core.ocr.manager import OcrManager


def test_select_provider_deepseek_alias_maps_to_deepseek_hf() -> None:
    manager = OcrManager(providers={"deepseek_hf": object()})
    assert manager._select_provider("deepseek") == "deepseek_hf"
    assert manager._select_provider("DEEPSEEK") == "deepseek_hf"
    assert manager._select_provider("deepseek-hf") == "deepseek_hf"


def test_select_provider_case_folds_explicit_names() -> None:
    manager = OcrManager(providers={"paddle": object()})
    assert manager._select_provider("Paddle") == "paddle"


def test_select_provider_auto_prefers_paddle_then_deepseek_hf() -> None:
    manager = OcrManager(providers={"deepseek_hf": object(), "paddle": object()})
    assert manager._select_provider("auto") == "paddle"
    assert manager._select_provider("") == "paddle"
