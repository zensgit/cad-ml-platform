from __future__ import annotations


def test_build_annotation_lines_contains_warning_and_notice() -> None:
    from scripts.ci.emit_graph2d_review_pack_gate_annotations import build_annotation_lines

    lines = build_annotation_lines(
        {
            "status": "failed",
            "failures": ["candidate_rate 0.9 > 0.7"],
            "warnings": ["total_rows 8 < min_total_rows 10"],
            "metrics": {
                "candidate_rate": 0.9,
                "hybrid_rejected_rate": 0.3,
                "conflict_rate": 0.1,
                "low_confidence_rate": 0.5,
            },
        }
    )
    assert any("Graph2D Review Gate Failure" in line for line in lines)
    assert any("Graph2D Review Gate Warning" in line for line in lines)
    assert any("Graph2D Review Gate Metrics" in line for line in lines)
