from __future__ import annotations


def test_evaluate_policy_breach_when_severity_too_high() -> None:
    from scripts.ci.check_graph2d_context_drift_index_policy import evaluate_policy

    report = evaluate_policy(
        {"overview": {"severity": "failed"}},
        {"max_allowed_severity": "alerted", "fail_on_breach": False},
    )
    assert report["status"] == "breached"
    assert report["breached"] is True


def test_evaluate_policy_pass_when_severity_within_threshold() -> None:
    from scripts.ci.check_graph2d_context_drift_index_policy import evaluate_policy

    report = evaluate_policy(
        {"overview": {"severity": "warn"}},
        {"max_allowed_severity": "alerted", "fail_on_breach": False},
    )
    assert report["status"] == "pass"
    assert report["breached"] is False


def test_build_markdown_contains_policy_source() -> None:
    from scripts.ci.check_graph2d_context_drift_index_policy import build_markdown

    text = build_markdown(
        {
            "status": "pass",
            "current_severity": "warn",
            "max_allowed_severity": "alerted",
            "breached": False,
            "reason": "ok",
            "policy_source": {
                "config": "config/graph2d_context_drift_index_policy.yaml",
                "config_loaded": True,
                "resolved_policy": {
                    "max_allowed_severity": "alerted",
                    "fail_on_breach": False,
                },
            },
        },
        "Index Policy",
    )
    assert "Index Policy" in text
    assert "config/graph2d_context_drift_index_policy.yaml" in text
    assert "resolved_max_allowed=alerted" in text
