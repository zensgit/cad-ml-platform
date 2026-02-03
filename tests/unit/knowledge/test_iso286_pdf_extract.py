"""Tests for ISO 286 PDF extraction helpers."""
from pathlib import Path
import importlib
import sys
import types

import pytest


def test_build_deviations_requires_pdfplumber(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=object))
    sys.modules.pop("scripts.extract_iso286_hole_deviations_from_pdf", None)
    extractor = importlib.import_module("scripts.extract_iso286_hole_deviations_from_pdf")
    monkeypatch.setattr(extractor, "_has_pdfplumber", lambda: False)
    with pytest.raises(SystemExit, match="pdfplumber is required"):
        extractor._build_deviations(Path("missing.pdf"), 6, allow_partial=False)
