from src.core.benchmark.knowledge_domain_release_readiness_matrix import (
    build_knowledge_domain_release_readiness_matrix,
    knowledge_domain_release_readiness_matrix_recommendations,
    render_knowledge_domain_release_readiness_matrix_markdown,
)


def test_build_release_readiness_matrix_summarizes_ready_partial_blocked() -> None:
    payload = build_knowledge_domain_release_readiness_matrix(
        benchmark_knowledge_domain_validation_matrix={
            "knowledge_domain_validation_matrix": {
                "domains": {
                    "standards": {"status": "ready"},
                    "tolerance": {"status": "partial"},
                    "gdt": {"status": "blocked"},
                }
            }
        },
        benchmark_knowledge_domain_release_gate={
            "knowledge_domain_release_gate": {
                "gate_open": True,
                "releasable_domains": ["standards"],
                "priority_domains": ["tolerance"],
                "blocked_domains": ["gdt"],
            }
        },
        benchmark_knowledge_reference_inventory={
            "knowledge_reference_inventory": {
                "domains": {
                    "standards": {"status": "ready"},
                    "tolerance": {"status": "partial"},
                    "gdt": {"status": "missing"},
                }
            }
        },
        benchmark_knowledge_domain_release_surface_alignment={
            "knowledge_domain_release_surface_alignment": {
                "domain_mismatches": ["tolerance:release_decision->partial"],
            }
        },
    )

    assert payload["status"] == "knowledge_domain_release_readiness_blocked"
    assert payload["ready_domain_count"] == 1
    assert payload["partial_domain_count"] == 1
    assert payload["blocked_domain_count"] == 1
    assert payload["releasable_domains"] == ["standards"]
    assert sorted(payload["priority_domains"]) == ["gdt", "tolerance"]
    assert payload["domains"]["standards"]["status"] == "ready"
    assert payload["domains"]["tolerance"]["status"] == "partial"
    assert payload["domains"]["tolerance"]["alignment_warning"] is True
    assert payload["domains"]["gdt"]["status"] == "blocked"
    assert payload["domains"]["gdt"]["blocking_reasons"] == [
        "validation:blocked",
        "inventory:missing",
        "release_gate:blocked",
    ]


def test_release_readiness_matrix_recommendations_prefer_focus_actions() -> None:
    component = {
        "status": "knowledge_domain_release_readiness_partial",
        "focus_areas_detail": [
            {"domain": "tolerance", "action": "Unblock tolerance release readiness."},
            {"domain": "gdt", "action": "Unblock gdt release readiness."},
        ],
    }

    assert knowledge_domain_release_readiness_matrix_recommendations(component) == [
        "Unblock tolerance release readiness.",
        "Unblock gdt release readiness.",
    ]


def test_render_release_readiness_matrix_markdown_includes_domain_details() -> None:
    component = {
        "knowledge_domain_release_readiness_matrix": {
            "status": "knowledge_domain_release_readiness_partial",
            "summary": "ready=1; partial=1; blocked=0",
            "ready_domain_count": 1,
            "partial_domain_count": 1,
            "blocked_domain_count": 0,
            "domains": {
                "tolerance": {
                    "label": "Tolerance & Fits",
                    "status": "partial",
                    "validation_status": "partial",
                    "inventory_status": "ready",
                    "release_gate_status": "partial",
                    "alignment_warning": True,
                    "blocking_reasons": [],
                    "warning_reasons": ["release_surface_alignment:mismatch"],
                }
            },
        },
        "recommendations": ["Unblock tolerance release readiness."],
    }

    markdown = render_knowledge_domain_release_readiness_matrix_markdown(
        component,
        "Knowledge Domain Release Readiness Matrix",
    )

    assert "# Knowledge Domain Release Readiness Matrix" in markdown
    assert "### Tolerance & Fits" in markdown
    assert "`alignment_warning`: `True`" in markdown
    assert "Unblock tolerance release readiness." in markdown
