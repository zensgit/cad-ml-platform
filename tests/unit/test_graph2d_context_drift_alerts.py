from __future__ import annotations


def test_evaluate_alerts_hits_default_threshold() -> None:
    from scripts.ci.check_graph2d_context_drift_alerts import evaluate_alerts

    history = [
        {"run_number": "1", "drift_key_counts": {"max_samples": 1}},
        {"run_number": "2", "drift_key_counts": {"max_samples": 1}},
        {"run_number": "3", "drift_key_counts": {"max_samples": 1}},
    ]
    report = evaluate_alerts(
        history=history,
        recent_runs=3,
        default_key_threshold=3,
        key_thresholds={},
    )
    assert report["status"] == "alerted"
    assert report["alerts"]
    assert report["alerts"][0]["key"] == "max_samples"


def test_evaluate_alerts_respects_per_key_override() -> None:
    from scripts.ci.check_graph2d_context_drift_alerts import evaluate_alerts

    history = [
        {"run_number": "1", "drift_key_counts": {"seeds": 1}},
        {"run_number": "2", "drift_key_counts": {"seeds": 1}},
    ]
    report = evaluate_alerts(
        history=history,
        recent_runs=2,
        default_key_threshold=3,
        key_thresholds={"seeds": 2},
    )
    assert report["status"] == "alerted"
    assert report["alerts"][0]["key"] == "seeds"
    assert report["alerts"][0]["threshold"] == 2


def test_evaluate_alerts_clear_when_below_threshold() -> None:
    from scripts.ci.check_graph2d_context_drift_alerts import evaluate_alerts

    history = [
        {"run_number": "1", "drift_key_counts": {"seeds": 1}},
        {"run_number": "2", "drift_key_counts": {"seeds": 1}},
    ]
    report = evaluate_alerts(
        history=history,
        recent_runs=2,
        default_key_threshold=3,
        key_thresholds={},
    )
    assert report["status"] == "clear"
    assert report["alerts"] == []


def test_build_markdown_contains_alert_lines() -> None:
    from scripts.ci.check_graph2d_context_drift_alerts import build_markdown

    report = {
        "status": "alerted",
        "history_size": 5,
        "recent_runs": 3,
        "default_key_threshold": 3,
        "key_totals": {"max_samples": 4},
        "policy_source": {
            "config": "config/graph2d_context_drift_alerts.yaml",
            "config_loaded": True,
            "resolved_policy": {"key_thresholds": {"max_samples": 2}},
        },
        "alerts": [
            {
                "message": "context drift key 'max_samples' count 4 >= threshold 3",
            }
        ],
    }
    text = build_markdown(report, "Context Drift Alerts")
    assert "Context Drift Alerts" in text
    assert "| Status | `alerted` |" in text
    assert "max_samples" in text
    assert "count 4 >= threshold 3" in text
    assert "config/graph2d_context_drift_alerts.yaml" in text


def test_resolve_alert_policy_prefers_cli_over_config() -> None:
    from scripts.ci.check_graph2d_context_drift_alerts import _resolve_alert_policy

    policy = _resolve_alert_policy(
        config_payload={
            "recent_runs": 9,
            "default_key_threshold": 4,
            "fail_on_alert": True,
            "key_thresholds": {"max_samples": 5, "seeds": 7},
        },
        cli_overrides={
            "recent_runs": 3,
            "default_key_threshold": 2,
            "key_threshold": ["max_samples=2"],
            "fail_on_alert": "false",
        },
    )
    assert policy["recent_runs"] == 3
    assert policy["default_key_threshold"] == 2
    assert policy["fail_on_alert"] is False
    assert policy["key_thresholds"]["max_samples"] == 2
    assert policy["key_thresholds"]["seeds"] == 7
