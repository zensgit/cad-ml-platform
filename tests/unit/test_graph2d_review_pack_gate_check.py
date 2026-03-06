from __future__ import annotations

import json
from pathlib import Path


def test_evaluate_review_pack_gate_passes_when_rates_within_threshold() -> None:
    from scripts.ci.check_graph2d_review_pack_gate import (
        DEFAULT_THRESHOLDS,
        evaluate_review_pack_gate,
    )

    summary = {
        "total_rows": 100,
        "candidate_rows": 40,
        "hybrid_rejected_count": 20,
        "conflict_count": 15,
        "low_confidence_count": 35,
    }
    report = evaluate_review_pack_gate(summary=summary, thresholds=DEFAULT_THRESHOLDS)
    assert report["status"] == "passed"
    assert report["failures"] == []


def test_evaluate_review_pack_gate_fails_when_candidate_rate_too_high() -> None:
    from scripts.ci.check_graph2d_review_pack_gate import (
        DEFAULT_THRESHOLDS,
        evaluate_review_pack_gate,
    )

    summary = {
        "total_rows": 20,
        "candidate_rows": 19,
        "hybrid_rejected_count": 3,
        "conflict_count": 2,
        "low_confidence_count": 5,
    }
    report = evaluate_review_pack_gate(summary=summary, thresholds=DEFAULT_THRESHOLDS)
    assert report["status"] == "failed"
    assert any("candidate_rate" in item for item in report["failures"])


def test_review_pack_gate_main_uses_config_file(tmp_path: Path) -> None:
    from scripts.ci.check_graph2d_review_pack_gate import main

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "total_rows": 10,
                "candidate_rows": 8,
                "hybrid_rejected_count": 0,
                "conflict_count": 0,
                "low_confidence_count": 0,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(
        "\n".join(
            [
                "graph2d_review_pack_gate:",
                "  max_candidate_rate: 0.75",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--summary-json",
            str(summary_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 1
    assert output_path.exists()
