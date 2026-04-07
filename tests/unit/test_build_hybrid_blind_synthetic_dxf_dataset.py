from __future__ import annotations

import json
from pathlib import Path


def test_build_hybrid_blind_synthetic_dataset_generates_files(tmp_path: Path) -> None:
    from scripts.ci import build_hybrid_blind_synthetic_dxf_dataset as mod

    manifest = tmp_path / "manifest.json"
    output_dir = tmp_path / "out"
    manifest.write_text(
        json.dumps(
            [
                {"filename": "A-01人孔v1.dxf"},
                {"filename": "B-02调节螺栓v2.dxf"},
                {"filename": "noext_name"},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--max-files",
            "2",
        ]
    )
    assert rc == 0
    files = sorted(output_dir.glob("*.dxf"))
    assert len(files) == 2
    assert all(path.stat().st_size > 0 for path in files)
