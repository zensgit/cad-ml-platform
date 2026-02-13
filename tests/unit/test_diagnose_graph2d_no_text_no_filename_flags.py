from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def test_diagnose_graph2d_supports_strip_text_and_mask_filename(
    monkeypatch, tmp_path: Path
) -> None:
    dxf_dir = tmp_path / "dxfs"
    dxf_dir.mkdir(parents=True, exist_ok=True)
    (dxf_dir / "sample.dxf").write_bytes(b"dummy-dxf-bytes")

    calls: List[Tuple[bytes, str]] = []
    strip_calls: List[bytes] = []

    def fake_strip(
        data: bytes, strip_blocks: bool = True
    ) -> bytes:  # noqa: FBT001, FBT002
        assert strip_blocks is True
        strip_calls.append(data)
        return b"stripped-bytes"

    monkeypatch.setattr(
        "src.utils.dxf_io.strip_dxf_text_entities_from_bytes",
        fake_strip,
        raising=True,
    )

    class FakeGraph2DClassifier:  # noqa: D401
        """Small stub to avoid torch/model dependencies in script unit tests."""

        last_instance: Optional["FakeGraph2DClassifier"] = None

        def __init__(self, model_path: Optional[str] = None) -> None:
            FakeGraph2DClassifier.last_instance = self
            self.model_path = model_path
            self.label_map: Dict[str, int] = {"X": 0}

        def predict_from_bytes(self, data: bytes, file_name: str) -> Dict[str, Any]:
            calls.append((data, file_name))
            return {
                "status": "ok",
                "label": "X",
                "confidence": 0.9,
                "temperature": 1.0,
                "temperature_source": "none",
                "label_map_size": len(self.label_map),
            }

    monkeypatch.setattr(
        "src.ml.vision_2d.Graph2DClassifier", FakeGraph2DClassifier, raising=True
    )

    out_dir = tmp_path / "out"
    from scripts.diagnose_graph2d_on_dxf_dir import main

    rc = main(
        [
            "--dxf-dir",
            str(dxf_dir),
            "--model-path",
            "models/graph2d_dummy.pth",
            "--max-files",
            "1",
            "--output-dir",
            str(out_dir),
            "--strip-text-entities",
            "--mask-filename",
        ]
    )
    assert rc == 0

    assert strip_calls == [b"dummy-dxf-bytes"]
    assert calls == [(b"stripped-bytes", "masked.dxf")]

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_options"]["strip_text_entities"] is True
    assert summary["eval_options"]["mask_filename"] is True
