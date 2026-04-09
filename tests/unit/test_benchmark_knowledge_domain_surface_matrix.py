from __future__ import annotations

from scripts.export_benchmark_knowledge_domain_surface_matrix import build_summary
from src.core.benchmark.knowledge_domain_surface_matrix import (
    build_knowledge_domain_surface_matrix,
    knowledge_domain_surface_matrix_recommendations,
    render_knowledge_domain_surface_matrix_markdown,
)


def test_build_surface_matrix_detects_public_surface_gaps() -> None:
    component = build_knowledge_domain_surface_matrix()

    assert component["status"] in {
        "knowledge_domain_surface_matrix_partial",
        "knowledge_domain_surface_matrix_blocked",
    }
    assert component["domains"]["tolerance"]["ready_subcapability_count"] >= 3
    assert "gdt" in component["public_surface_gap_domains"]
    assert component["domains"]["gdt"]["subcapabilities"]["gdt_public_api"][
        "present_route_count"
    ] == 0


def test_surface_matrix_recommendations_and_markdown_highlight_gdt_gap() -> None:
    payload = build_summary(title="Knowledge Domain Surface Matrix")
    component = payload["knowledge_domain_surface_matrix"]
    recommendations = knowledge_domain_surface_matrix_recommendations(component)
    rendered = render_knowledge_domain_surface_matrix_markdown(
        payload,
        "Knowledge Domain Surface Matrix",
    )

    assert recommendations
    assert any("GD&T" in item or "gdt" in item.lower() for item in recommendations)
    assert "# Knowledge Domain Surface Matrix" in rendered
    assert "## Domains" in rendered
    assert "`gdt`" in rendered

