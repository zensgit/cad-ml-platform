from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze_graph2d_baseline import compute_sha256, freeze_baseline


def test_freeze_graph2d_baseline_writes_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint-bytes")

    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "label_cn,total,correct,accuracy,precision,recall,f1,top2_accuracy,share,macro_f1,weighted_f1",
                "__overall__,10,9,0.900,,,,0.950,1.000,0.880,0.900",
            ]
        ),
        encoding="utf-8",
    )

    result = freeze_baseline(
        checkpoint=checkpoint,
        output_dir=tmp_path / "frozen",
        baseline_name="unit",
        manifest="manifest.csv",
        notes="unit-test",
        metrics_csv=metrics,
    )

    metadata_path = Path(result["metadata"])
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["source_sha256"] == compute_sha256(checkpoint)
    assert payload["frozen_sha256"] == compute_sha256(Path(result["checkpoint"]))
    assert payload["metrics_summary"]["accuracy"] == 0.9
    assert payload["manifest"] == "manifest.csv"
