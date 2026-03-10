from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_knowledge_domain_release_surface_alignment import (
    build_payload,
)
from src.core.benchmark import build_knowledge_domain_release_surface_alignment


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_domain_release_surface_alignment.py"
)


def _release_decision() -> dict:
    return {
        "knowledge_domain_control_plane_status": "knowledge_domain_control_plane_blocked",
        "knowledge_domain_control_plane_drift_status": "regressed",
        "knowledge_domain_control_plane_release_blockers": ["gdt"],
        "knowledge_domain_control_plane_domains": {
            "tolerance": {"status": "ready"},
            "standards": {"status": "partial"},
            "gdt": {"status": "blocked"},
        },
    }


def _release_runbook() -> dict:
    return {
        "knowledge_domain_control_plane_status": "knowledge_domain_control_plane_blocked",
        "knowledge_domain_control_plane_drift_status": "regressed",
        "knowledge_domain_control_plane_release_blockers": ["gdt"],
        "knowledge_domain_control_plane_domains": {
            "tolerance": {"status": "ready"},
            "standards": {"status": "partial"},
            "gdt": {"status": "blocked"},
        },
    }


def test_build_knowledge_domain_release_surface_alignment_aligned() -> None:
    component = build_knowledge_domain_release_surface_alignment(
        benchmark_release_decision=_release_decision(),
        benchmark_release_runbook=_release_runbook(),
    )

    assert component["status"] == "aligned"
    assert component["mismatch_count"] == 0
    assert component["mismatches"] == []


def test_build_knowledge_domain_release_surface_alignment_diverged() -> None:
    runbook = _release_runbook()
    runbook["knowledge_domain_control_plane_drift_status"] = "stable"
    runbook["knowledge_domain_control_plane_domains"]["standards"]["status"] = "blocked"

    component = build_knowledge_domain_release_surface_alignment(
        benchmark_release_decision=_release_decision(),
        benchmark_release_runbook=runbook,
    )

    assert component["status"] == "diverged"
    assert "control_plane_drift_status:regressed->stable" in component["mismatches"]
    assert "standards:partial->blocked" in component["domain_mismatches"]


def test_export_benchmark_knowledge_domain_release_surface_alignment_cli(
    tmp_path: Path,
) -> None:
    decision = tmp_path / "release_decision.json"
    runbook = tmp_path / "release_runbook.json"
    output_json = tmp_path / "alignment.json"
    output_md = tmp_path / "alignment.md"
    decision.write_text(
        json.dumps(_release_decision(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runbook.write_text(
        json.dumps(_release_runbook(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-release-decision",
            str(decision),
            "--benchmark-release-runbook",
            str(runbook),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )

    assert result.returncode == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["knowledge_domain_release_surface_alignment"]["status"] == "aligned"
    rendered = output_md.read_text(encoding="utf-8")
    assert "# Benchmark Knowledge Domain Release Surface Alignment" in rendered
    assert "## Recommendations" in rendered


def test_build_payload_includes_recommendations() -> None:
    payload = build_payload(
        title="Benchmark Knowledge Domain Release Surface Alignment",
        benchmark_release_decision=_release_decision(),
        benchmark_release_runbook=_release_runbook(),
        artifact_paths={
            "benchmark_release_decision": "release_decision.json",
            "benchmark_release_runbook": "release_runbook.json",
        },
    )
    assert payload["knowledge_domain_release_surface_alignment"]["status"] == "aligned"
    assert payload["recommendations"] == [
        "Keep knowledge-domain control-plane fields aligned between release decision "
        "and release runbook."
    ]
