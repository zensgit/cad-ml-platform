from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.ml.hybrid_classifier import HybridClassifier


def _build_synthetic_dxf_bytes(titleblock_texts: List[str]) -> bytes:
    # ezdxf may try to write to ~/.cache; point it at a writable directory.
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

    import ezdxf

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    msp.add_line((0, 0), (1000, 0))
    msp.add_line((1000, 0), (1000, 1000))
    msp.add_line((1000, 1000), (0, 1000))
    msp.add_line((0, 1000), (0, 0))

    if titleblock_texts:
        x = 800
        y0 = 200
        for idx, text in enumerate(titleblock_texts):
            msp.add_text(
                str(text),
                dxfattribs={
                    "height": 10,
                    "insert": (x, y0 - (idx * 20)),
                },
            )

    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def _load_cases() -> List[Dict[str, Any]]:
    path = Path("tests/golden/golden_dxf_hybrid_cases.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def test_golden_hybrid_cases() -> None:
    env = {
        "HYBRID_CLASSIFIER_ENABLED": "true",
        "FILENAME_CLASSIFIER_ENABLED": "true",
        "GRAPH2D_ENABLED": "false",
        "PROCESS_FEATURES_ENABLED": "false",
        "TITLEBLOCK_ENABLED": "true",
        "TITLEBLOCK_OVERRIDE_ENABLED": "true",
        "TITLEBLOCK_MIN_CONF": "0.6",
        "XDG_CACHE_HOME": "/tmp/xdg-cache",
    }

    with pytest.MonkeyPatch.context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)

        clf = HybridClassifier()
        for case in _load_cases():
            filename = str(case["filename"])
            expected_label = str(case["expected_label"])
            expected_source = case.get("expected_source")
            expected_source = str(expected_source) if expected_source else None
            titleblock_texts = list(case.get("titleblock_texts") or [])
            graph2d_result = case.get("graph2d_result")

            file_bytes = _build_synthetic_dxf_bytes(titleblock_texts)
            result = clf.classify(
                filename=filename,
                file_bytes=file_bytes,
                graph2d_result=graph2d_result,
            ).to_dict()

            assert result.get("label") == expected_label
            if expected_source is not None:
                assert result.get("source") == expected_source
