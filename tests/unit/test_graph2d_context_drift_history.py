from __future__ import annotations


def test_update_history_appends_and_trims() -> None:
    from scripts.ci.update_graph2d_context_drift_history import update_history

    history = [
        {"run_id": "1", "timestamp": "2026-02-16T00:00:00+00:00"},
        {"run_id": "2", "timestamp": "2026-02-16T00:01:00+00:00"},
    ]
    snapshot = {"run_id": "3", "timestamp": "2026-02-16T00:02:00+00:00"}
    out = update_history(history_payload=history, snapshot=snapshot, max_runs=2)
    assert len(out) == 2
    assert out[0]["run_id"] == "2"
    assert out[1]["run_id"] == "3"


def test_update_history_replaces_same_run_id() -> None:
    from scripts.ci.update_graph2d_context_drift_history import update_history

    history = [
        {"run_id": "100", "timestamp": "2026-02-16T00:00:00+00:00", "status": "failed"},
    ]
    snapshot = {"run_id": "100", "timestamp": "2026-02-16T00:01:00+00:00", "status": "passed"}
    out = update_history(history_payload=history, snapshot=snapshot, max_runs=10)
    assert len(out) == 1
    assert out[0]["status"] == "passed"


def test_build_history_markdown_contains_recent_totals() -> None:
    from scripts.ci.render_graph2d_context_drift_history import build_markdown

    history = [
        {
            "run_number": "101",
            "status": "passed",
            "warning_count": 0,
            "failure_count": 0,
            "drift_key_counts": {},
        },
        {
            "run_number": "102",
            "status": "passed_with_warnings",
            "warning_count": 1,
            "failure_count": 0,
            "drift_key_counts": {"max_samples": 1, "seeds": 2},
        },
    ]
    text = build_markdown(history=history, title="Context Drift History")
    assert "Context Drift History" in text
    assert "`#101`" in text
    assert "`#102`" in text
    assert "`seeds` | 2" in text
    assert "`max_samples` | 1" in text


def test_build_history_markdown_handles_empty() -> None:
    from scripts.ci.render_graph2d_context_drift_history import build_markdown

    text = build_markdown(history=[], title="Context Drift History")
    assert "No context drift history found." in text
