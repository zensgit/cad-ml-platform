from __future__ import annotations


def test_graph2d_seed_gate_summary_includes_pass_status() -> None:
    from scripts.ci.summarize_graph2d_seed_gate import build_summary

    summary = {
        "config": "config/graph2d_seed_gate.yaml",
        "training_profile": "none",
        "manifest_label_mode": "parent_dir",
        "num_runs": 2,
        "num_success_runs": 2,
        "num_error_runs": 0,
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_accuracy_max": 0.43,
        "gate": {
            "enabled": True,
            "passed": True,
            "failures": [],
        },
    }
    text = build_summary(summary, "Graph2D Seed Gate")
    assert "Graph2D Seed Gate" in text
    assert "Seed gate passed" in text
    assert "`True`" in text
    assert "0.360000 / 0.290000 / 0.430000" in text


def test_graph2d_seed_gate_summary_shows_failures() -> None:
    from scripts.ci.summarize_graph2d_seed_gate import build_summary

    summary = {
        "num_runs": 2,
        "num_success_runs": 1,
        "num_error_runs": 1,
        "strict_accuracy_mean": 0.10,
        "strict_accuracy_min": 0.01,
        "strict_accuracy_max": 0.20,
        "gate": {
            "enabled": True,
            "passed": False,
            "failures": ["strict_accuracy_mean: 0.100000 < 0.250000"],
        },
    }
    text = build_summary(summary, "Graph2D Seed Gate")
    assert "Gate failures:" in text
    assert "strict_accuracy_mean: 0.100000 < 0.250000" in text
