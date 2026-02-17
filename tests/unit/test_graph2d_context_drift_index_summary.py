from __future__ import annotations


def test_build_summary_contains_overview_and_artifacts() -> None:
    from scripts.ci.summarize_graph2d_context_drift_index import build_summary

    text = build_summary(
        {
            "overview": {
                "status": "alerted",
                "alert_count": 1,
                "history_entries": 5,
                "recent_window": 5,
                "drift_key_count": 2,
                "top_drift_key": {"key": "max_samples", "count": 3},
            },
            "artifacts": {
                "alerts_report": {"exists": True, "path": "/tmp/a.json"},
                "history_summary": {"exists": True, "path": "/tmp/b.json"},
                "key_counts_summary": {"exists": False, "path": "/tmp/c.json"},
            },
            "summaries": {
                "alerts": {
                    "rows": [{"key": "max_samples", "count": 3, "threshold": 2}],
                }
            },
        },
        "Context Drift Index",
    )
    assert "Context Drift Index" in text
    assert "| Alert count | ❌ | `1` |" in text
    assert "max_samples:3" in text
    assert "`key_counts_summary`" in text
    assert "max_samples: 3 >= 2" in text


def test_build_summary_handles_empty_payload_sections() -> None:
    from scripts.ci.summarize_graph2d_context_drift_index import build_summary

    text = build_summary({}, "Context Drift Index")
    assert "Context Drift Index" in text
    assert "| Status | ✅ | `clear` |" in text
    assert "| Artifact coverage | ✅ | `0/0` |" in text
