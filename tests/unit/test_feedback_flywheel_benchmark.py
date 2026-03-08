from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.feedback_flywheel import (
    build_feedback_flywheel_status,
    feedback_flywheel_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "export_feedback_flywheel_benchmark.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_feedback_flywheel_status_closed_loop_ready() -> None:
    summary = build_feedback_flywheel_status(
        {
            "total": 12,
            "correction_count": 8,
            "coarse_correction_count": 6,
            "average_rating": 4.2,
            "by_review_outcome": {"updated": 8},
            "by_review_reason": {"low_confidence": 5},
        },
        {
            "sample_count": 8,
            "vector_count": 8,
            "label_distribution": {"人孔": 4},
            "coarse_label_distribution": {"开孔件": 4},
        },
        {
            "triplet_count": 6,
            "unique_anchor_count": 6,
            "anchor_label_distribution": {"人孔": 3},
            "negative_label_distribution": {"法兰": 3},
        },
    )

    assert summary["status"] == "closed_loop_ready"
    assert summary["feedback_total"] == 12
    assert summary["metric_triplet_count"] == 6
    assert summary["finetune_sample_count"] == 8


def test_feedback_flywheel_recommendations_flags_feedback_gap() -> None:
    items = feedback_flywheel_recommendations({"status": "feedback_collected"})
    assert any("metric-training" in item for item in items)


def test_export_feedback_flywheel_benchmark_outputs_files(tmp_path: Path) -> None:
    feedback = _write_json(
        tmp_path / "feedback.json",
        {
            "total": 10,
            "correction_count": 5,
            "coarse_correction_count": 4,
            "average_rating": 4.0,
            "by_review_outcome": {"updated": 5},
            "by_review_reason": {"branch_conflict": 2},
        },
    )
    finetune = _write_json(
        tmp_path / "finetune.json",
        {
            "sample_count": 5,
            "vector_count": 5,
            "label_distribution": {"人孔": 3},
            "coarse_label_distribution": {"开孔件": 3},
        },
    )
    metric_train = _write_json(
        tmp_path / "metric_train.json",
        {
            "triplet_count": 4,
            "unique_anchor_count": 4,
            "anchor_label_distribution": {"人孔": 2},
            "negative_label_distribution": {"法兰": 2},
        },
    )
    output_json = tmp_path / "flywheel.json"
    output_md = tmp_path / "flywheel.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "Feedback Flywheel",
            "--feedback-summary",
            str(feedback),
            "--finetune-summary",
            str(finetune),
            "--metric-train-summary",
            str(metric_train),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["feedback_flywheel"]["status"] == "closed_loop_ready"
    assert payload["feedback_flywheel"]["coarse_correction_count"] == 4
    assert output_json.exists()
    assert output_md.exists()
    markdown = output_md.read_text(encoding="utf-8")
    assert "Feedback Flywheel" in markdown
    assert "closed_loop_ready" in markdown
