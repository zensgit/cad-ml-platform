from __future__ import annotations


def test_build_warning_lines_from_alert_report() -> None:
    from scripts.ci.emit_graph2d_context_drift_warnings import build_warning_lines

    report = {
        "alerts": [
            {"message": "context drift key 'max_samples' count 2 >= threshold 2"},
            {"message": "context drift key 'seeds' count 3 >= threshold 3"},
        ]
    }
    lines = build_warning_lines(report)
    assert len(lines) == 2
    assert lines[0].startswith("::warning title=Graph2D Context Drift::")
    assert "max_samples" in lines[0]
    assert "seeds" in lines[1]


def test_build_warning_lines_handles_empty_alerts() -> None:
    from scripts.ci.emit_graph2d_context_drift_warnings import build_warning_lines

    assert build_warning_lines({"alerts": []}) == []
    assert build_warning_lines({}) == []
