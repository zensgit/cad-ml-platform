import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_companion_summary import (
    build_companion_summary,
    render_markdown,
)


def test_build_companion_summary_prefers_bundle_and_flags_attention() -> None:
    payload = build_companion_summary(
        title="Benchmark Companion",
        benchmark_scorecard={
            "overall_status": "gap_detected",
            "components": {
                "hybrid": {"status": "healthy"},
                "review_queue": {"status": "healthy"},
            },
            "recommendations": ["improve scorecard coverage"],
        },
        benchmark_operational_summary={
            "overall_status": "attention_required",
            "component_statuses": {
                "assistant_explainability": "partial_coverage",
                "review_queue": "managed_backlog",
            },
            "blockers": ["assistant_explainability:partial_coverage"],
            "recommendations": ["raise assistant evidence coverage"],
        },
        benchmark_artifact_bundle={
            "overall_status": "attention_required",
            "component_statuses": {
                "assistant_explainability": "partial_coverage",
                "review_queue": "managed_backlog",
                "ocr_review": "managed_review",
            },
            "blockers": ["review_queue:managed_backlog"],
            "recommendations": ["reduce review queue backlog"],
        },
        artifact_paths={
            "benchmark_scorecard": "scorecard.json",
            "benchmark_operational_summary": "operational.json",
            "benchmark_artifact_bundle": "bundle.json",
        },
    )

    assert payload["overall_status"] == "attention_required"
    assert payload["review_surface"] == "attention_required"
    assert payload["primary_gap"] == "review_queue:managed_backlog"
    assert payload["component_statuses"]["assistant_explainability"] == "partial_coverage"
    assert payload["recommended_actions"] == ["reduce review queue backlog"]
    assert payload["artifacts"]["benchmark_artifact_bundle"]["present"] is True


def test_render_markdown_includes_sections() -> None:
    payload = {
        "title": "Benchmark Companion",
        "overall_status": "healthy",
        "review_surface": "ready",
        "primary_gap": "none",
        "component_statuses": {"hybrid": "healthy"},
        "recommended_actions": ["keep monitoring"],
        "blockers": [],
        "artifacts": {
            "benchmark_artifact_bundle": {
                "present": True,
                "path": "bundle.json",
            }
        },
    }

    rendered = render_markdown(payload)
    assert "# Benchmark Companion" in rendered
    assert "`review_surface`: `ready`" in rendered
    assert "## Recommended Actions" in rendered
    assert "bundle.json" in rendered


def test_cli_writes_outputs(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    operational = tmp_path / "operational.json"
    bundle = tmp_path / "bundle.json"
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    scorecard.write_text(
        json.dumps(
            {
                "overall_status": "healthy",
                "components": {"hybrid": {"status": "healthy"}},
                "recommendations": ["keep monitoring"],
            }
        ),
        encoding="utf-8",
    )
    operational.write_text(
        json.dumps(
            {
                "overall_status": "healthy",
                "component_statuses": {"review_queue": "healthy"},
            }
        ),
        encoding="utf-8",
    )
    bundle.write_text(
        json.dumps(
            {
                "overall_status": "healthy",
                "component_statuses": {
                    "assistant_explainability": "explainability_ready",
                    "review_queue": "healthy",
                    "ocr_review": "ocr_ready",
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_benchmark_companion_summary.py",
            "--benchmark-scorecard",
            str(scorecard),
            "--benchmark-operational-summary",
            str(operational),
            "--benchmark-artifact-bundle",
            str(bundle),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "healthy"
    assert payload["review_surface"] == "ready"
    assert output_md.exists()
