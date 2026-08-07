"""Honesty: missing model assets must not invent success (Track 2 / L2).

Also: do not permanently pollute ``sys.modules`` — use ``monkeypatch.setitem``
so later tests in the same pytest session keep a clean import graph.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from unittest.mock import MagicMock


def test_paddle_missing_client_does_not_fabricate_pad001_text():
    """Real extract() with _ocr=None must not invent PAD-001 / Φ20 dimensions."""
    from src.core.ocr.providers.paddle import PaddleOcrProvider

    provider = PaddleOcrProvider(enable_preprocess=False)
    provider._ocr = None
    tiny = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    result = asyncio.run(provider.extract(tiny))
    text = getattr(result, "text", "") or ""
    assert "PAD-001" not in text
    assert "Φ20" not in text
    assert text.strip() == ""
    dims = getattr(result, "dimensions", None) or []
    assert list(dims) == []


def test_pointnet_source_refuses_missing_extractor_state_dict():
    """Structural: shipped loader must raise when extractor_state_dict absent."""
    from pathlib import Path

    src = Path("src/ml/pointnet/inference.py").read_text(encoding="utf-8")
    assert 'if "extractor_state_dict" not in checkpoint:' in src
    assert "refusing randomly-initialized feature extractor" in src


def test_pointnet_missing_extractor_state_dict_does_not_report_loaded(monkeypatch):
    """Drive real _try_load_model with gateway bytes + incomplete checkpoint.

    Uses monkeypatch.setitem for any sys.modules stubs so session isolation holds.
    """
    # Minimal torch stub so HAS_TORCH path is reachable without installing torch.
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch_mod.load = MagicMock(return_value={"classifier_state_dict": {"w": 1}})
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    class DummyMod:
        def load_state_dict(self, *a, **k):
            return None

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

    fake_model = types.ModuleType("src.ml.pointnet.model")
    fake_model.PointNetClassifier = lambda **k: DummyMod()
    fake_model.PointNetFeatureExtractor = lambda **k: DummyMod()
    monkeypatch.setitem(sys.modules, "src.ml.pointnet.model", fake_model)

    import src.ml.pointnet.inference as inf

    # Reload under stubbed torch; monkeypatch undoes module dict + attribute sets.
    monkeypatch.setattr(inf, "HAS_TORCH", True, raising=False)
    monkeypatch.setattr(inf, "torch", torch_mod, raising=False)
    monkeypatch.setattr(inf, "activate_file", lambda *a, **k: b"fake-ckpt-bytes")

    # Re-bind model symbols used inside _try_load_model's local import path:
    # the function does `from src.ml.pointnet.model import ...` which hits our stub.
    analyzer = inf.PointNet3DAnalyzer(model_path="ignored.pt")

    assert analyzer._model_loaded is False
    assert analyzer._feature_extractor is None
    assert analyzer._load_error is not None
    assert "extractor_state_dict" in analyzer._load_error
