from __future__ import annotations


def test_build_annotation_line_for_warn_severity() -> None:
    from scripts.ci.emit_graph2d_context_drift_index_annotations import build_annotation_line

    line = build_annotation_line(
        {
            "overview": {
                "severity": "warn",
                "status": "clear",
                "alert_count": 0,
                "severity_reason": "context drift observed below alert threshold",
            }
        }
    )
    assert line.startswith("::warning")
    assert "severity=warn" in line


def test_build_annotation_line_for_failed_severity() -> None:
    from scripts.ci.emit_graph2d_context_drift_index_annotations import build_annotation_line

    line = build_annotation_line(
        {
            "overview": {
                "severity": "failed",
                "status": "failed",
                "alert_count": 2,
            }
        }
    )
    assert line.startswith("::error")
    assert "alert_count=2" in line
