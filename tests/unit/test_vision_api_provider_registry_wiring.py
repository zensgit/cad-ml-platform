"""Vision API should wire OCRManager providers from the core ProviderRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_get_vision_manager_registers_ocr_providers_from_registry() -> None:
    from src.api.v1 import vision

    vision.reset_vision_manager()

    fake_paddle = object()
    fake_deepseek = object()

    with patch("src.api.v1.vision.create_vision_provider") as mock_create:
        with patch(
            "src.api.v1.vision.bootstrap_core_provider_registry"
        ) as mock_bootstrap:
            with patch("src.api.v1.vision.ProviderRegistry.get") as mock_get:
                with patch("src.core.ocr.manager.OcrManager") as mock_ocr_cls:
                    mock_create.return_value = MagicMock()
                    mock_ocr = MagicMock()
                    mock_ocr_cls.return_value = mock_ocr

                    def _get(domain: str, provider_name: str):
                        assert domain == "ocr"
                        if provider_name == "paddle":
                            return fake_paddle
                        if provider_name == "deepseek_hf":
                            return fake_deepseek
                        raise AssertionError(f"unexpected provider: {provider_name}")

                    mock_get.side_effect = _get

                    manager = vision.get_vision_manager(provider_type="stub")
                    assert manager is not None

                    mock_bootstrap.assert_called_once()
                    mock_ocr.register_provider.assert_any_call("paddle", fake_paddle)
                    mock_ocr.register_provider.assert_any_call(
                        "deepseek_hf", fake_deepseek
                    )
