from __future__ import annotations

from pathlib import Path
import re


DATASET2D_IMPORT = "src.ml.train.dataset_2d"


def _has_torch_guard(text: str) -> bool:
    if re.search(r"pytest\.importorskip\(\s*[\"']torch[\"']\s*\)", text):
        return True
    if "pytest.mark.skipif" in text and "find_spec" in text and "torch" in text:
        return True
    return False


def test_dataset2d_unit_tests_require_torch_guard() -> None:
    files = sorted(Path("tests/unit").glob("test_*.py"))
    checked = 0
    missing: list[str] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        if DATASET2D_IMPORT not in text:
            continue
        checked += 1
        if not _has_torch_guard(text):
            missing.append(str(path))

    assert checked > 0, "no dataset_2d unit tests found"
    assert not missing, (
        "dataset_2d tests must guard torch availability with importorskip/skipif: "
        + ", ".join(missing)
    )
