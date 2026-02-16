from __future__ import annotations


def test_build_markdown_counts_context_diff_keys() -> None:
    from scripts.ci.render_graph2d_context_drift_key_counts import build_markdown

    reports = [
        (
            "/tmp/report-a.json",
            {
                "channel": "standard",
                "status": "passed_with_warnings",
                "warnings": ["context: mismatch on keys [training_profile,max_samples]"],
                "failures": [],
                "thresholds": {"context_mismatch_mode": "warn"},
                "baseline_metadata": {
                    "context_match": False,
                    "context_diff": {
                        "training_profile": {"baseline": "a", "current": "b"},
                        "max_samples": {"baseline": 120, "current": 80},
                    },
                },
            },
        ),
        (
            "/tmp/report-b.json",
            {
                "channel": "strict",
                "status": "passed_with_warnings",
                "warnings": ["context: mismatch on keys [training_profile]"],
                "failures": [],
                "thresholds": {"context_mismatch_mode": "warn"},
                "baseline_metadata": {
                    "context_match": False,
                    "context_diff": {
                        "training_profile": {"baseline": "a", "current": "c"},
                    },
                },
            },
        ),
    ]

    text = build_markdown(reports=reports, title="Context Drift Keys")
    assert "Context Drift Keys" in text
    assert "`training_profile` | 2" in text
    assert "`max_samples` | 1" in text


def test_build_markdown_handles_empty_report_list() -> None:
    from scripts.ci.render_graph2d_context_drift_key_counts import build_markdown

    text = build_markdown(reports=[], title="Context Drift Keys")
    assert "No regression reports found." in text


def test_build_markdown_handles_no_context_drift_keys() -> None:
    from scripts.ci.render_graph2d_context_drift_key_counts import build_markdown

    reports = [
        (
            "/tmp/report-a.json",
            {
                "channel": "strict",
                "status": "passed",
                "warnings": [],
                "failures": [],
                "thresholds": {"context_mismatch_mode": "warn"},
                "baseline_metadata": {"context_match": True, "context_diff": {}},
            },
        )
    ]
    text = build_markdown(reports=reports, title="Context Drift Keys")
    assert "Context drift key counts: none." in text
