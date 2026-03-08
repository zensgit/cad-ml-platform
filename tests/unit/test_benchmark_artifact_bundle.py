from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import export_benchmark_artifact_bundle as module

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "export_benchmark_artifact_bundle.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_build_bundle_prefers_operational_summary() -> None:
    payload = module.build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_feedback_gap",
            "components": {"hybrid": {"status": "ready"}},
        },
        benchmark_operational_summary={
            "overall_status": "attention_required",
            "component_statuses": {
                "feedback_flywheel": "feedback_collected",
                "assistant_explainability": "evidence_partial",
                "review_queue": "managed_backlog",
                "ocr_review": "review_heavy",
            },
            "blockers": ["feedback backlog"],
            "recommendations": ["Close the review queue."],
        },
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_operational_summary": "operational.json"},
    )
    assert payload["overall_status"] == "attention_required"
    assert payload["available_artifact_count"] == 2
    assert payload["component_statuses"]["feedback_flywheel"] == "feedback_collected"
    assert payload["blockers"] == ["feedback backlog"]
    assert payload["recommendations"] == ["Close the review queue."]


def test_build_bundle_falls_back_to_scorecard() -> None:
    payload = module.build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "components": {
                "hybrid": {"status": "ready"},
                "assistant_explainability": {"status": "evidence_ready"},
            },
            "recommendations": ["Freeze this run."],
        },
        benchmark_operational_summary={},
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_scorecard": "scorecard.json"},
    )
    assert payload["overall_status"] == "benchmark_ready_with_multisignal_evidence"
    assert payload["component_statuses"]["assistant_explainability"] == "evidence_ready"
    assert payload["recommendations"] == ["Freeze this run."]


def test_export_benchmark_artifact_bundle_outputs_files(tmp_path: Path) -> None:
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        {
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "components": {"hybrid": {"status": "ready"}},
            "recommendations": ["Freeze this run."],
        },
    )
    operational = _write_json(
        tmp_path / "operational.json",
        {
            "overall_status": "attention_required",
            "component_statuses": {"review_queue": "managed_backlog"},
            "blockers": ["review queue backlog"],
            "recommendations": ["Drain critical samples."],
        },
    )
    output_json = tmp_path / "bundle.json"
    output_md = tmp_path / "bundle.md"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--benchmark-scorecard",
            str(scorecard),
            "--benchmark-operational-summary",
            str(operational),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "attention_required"
    assert payload["artifacts"]["benchmark_scorecard"]["present"] is True
    assert payload["artifacts"]["benchmark_operational_summary"]["present"] is True
    assert "review queue backlog" in output_md.read_text(encoding="utf-8")
