from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_knowledge_subdomain_surface_matrix import build_summary
from src.core.benchmark import (
    build_knowledge_subdomain_surface_matrix,
    render_knowledge_subdomain_surface_matrix_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_subdomain_surface_matrix.py"
)


def test_build_knowledge_subdomain_surface_matrix() -> None:
    payload = build_knowledge_subdomain_surface_matrix()
    assert payload["status"].startswith("knowledge_subdomain_surface_matrix_")
    assert "standards.threads" in payload["subdomains"]
    assert "tolerance.it_grades" in payload["subdomains"]


def test_export_benchmark_knowledge_subdomain_surface_matrix_cli(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "knowledge-subdomain-surface-matrix.json"
    output_md = tmp_path / "knowledge-subdomain-surface-matrix.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert "knowledge_subdomain_surface_matrix" in payload
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Recommendations" in rendered


def test_render_knowledge_subdomain_surface_matrix_markdown() -> None:
    rendered = render_knowledge_subdomain_surface_matrix_markdown(
        build_summary(
            title="Benchmark Knowledge Subdomain Surface Matrix",
            artifact_paths={},
        ),
        "Benchmark Knowledge Subdomain Surface Matrix",
    )
    assert "# Benchmark Knowledge Subdomain Surface Matrix" in rendered
    assert "## Priority Subdomains" in rendered
