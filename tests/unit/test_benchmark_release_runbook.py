import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_release_runbook import (
    build_release_runbook,
    render_markdown,
)


def test_build_release_runbook_requires_blocker_resolution() -> None:
    payload = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision={
            "release_status": "blocked",
            "automation_ready": False,
            "primary_signal_source": "benchmark_companion_summary",
            "blocking_signals": ["review_queue:critical_backlog"],
            "artifacts": {
                "benchmark_scorecard": {"path": "scorecard.json"},
                "benchmark_operational_summary": {"path": "operational.json"},
            },
        },
        benchmark_companion_summary={
            "blockers": ["review_queue:critical_backlog"],
        },
        benchmark_artifact_bundle={},
        artifact_paths={"benchmark_release_decision": "release.json"},
    )

    assert payload["release_status"] == "blocked"
    assert payload["next_action"] == "collect_artifacts"
    assert "benchmark_artifact_bundle" in payload["missing_artifacts"]
    assert payload["operator_steps"][1]["key"] == "resolve_blockers"
    assert payload["operator_steps"][1]["status"] == "required"


def test_build_release_runbook_freezes_when_ready() -> None:
    payload = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision={
            "release_status": "ready",
            "automation_ready": True,
            "primary_signal_source": "benchmark_release_decision",
            "artifacts": {
                "benchmark_scorecard": {"path": "scorecard.json"},
                "benchmark_operational_summary": {"path": "operational.json"},
                "benchmark_companion_summary": {"path": "companion.json"},
                "benchmark_artifact_bundle": {"path": "bundle.json"},
            },
        },
        benchmark_companion_summary={"overall_status": "healthy"},
        benchmark_artifact_bundle={"overall_status": "healthy"},
        artifact_paths={
            "benchmark_release_decision": "release.json",
            "benchmark_companion_summary": "companion.json",
            "benchmark_artifact_bundle": "bundle.json",
        },
    )

    assert payload["ready_to_freeze_baseline"] is True
    assert payload["next_action"] == "freeze_release_baseline"
    assert payload["operator_steps"][-1]["status"] == "ready"


def test_render_markdown_and_cli_outputs(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    companion = tmp_path / "companion.json"
    bundle = tmp_path / "bundle.json"
    output_json = tmp_path / "runbook.json"
    output_md = tmp_path / "runbook.md"

    release.write_text(
        json.dumps(
            {
                "release_status": "review_required",
                "automation_ready": False,
                "primary_signal_source": "benchmark_companion_summary",
                "review_signals": ["Drain review queue"],
                "artifacts": {
                    "benchmark_scorecard": {"path": "scorecard.json"},
                    "benchmark_operational_summary": {"path": "operational.json"},
                    "benchmark_companion_summary": {"path": "companion.json"},
                    "benchmark_artifact_bundle": {"path": "bundle.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    companion.write_text(
        json.dumps({"recommended_actions": ["Drain review queue"]}),
        encoding="utf-8",
    )
    bundle.write_text(json.dumps({"overall_status": "attention_required"}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/export_benchmark_release_runbook.py",
            "--benchmark-release-decision",
            str(release),
            "--benchmark-companion-summary",
            str(companion),
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
    assert payload["release_status"] == "review_required"
    assert payload["next_action"] == "review_signals"
    assert output_md.exists()

    rendered = render_markdown(payload)
    assert "# Benchmark Release Runbook" in rendered
    assert "`next_action`: `review_signals`" in rendered
