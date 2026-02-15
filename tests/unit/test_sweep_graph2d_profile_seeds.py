from __future__ import annotations

import subprocess

def test_parse_seeds_comma_delimited() -> None:
    from scripts.sweep_graph2d_profile_seeds import _parse_seeds

    assert _parse_seeds("7,21,42") == [7, 21, 42]


def test_parse_seeds_strips_whitespace() -> None:
    from scripts.sweep_graph2d_profile_seeds import _parse_seeds

    assert _parse_seeds(" 7,  13 ,42 ") == [7, 13, 42]


def test_evaluate_gate_passes_when_disabled() -> None:
    from scripts.sweep_graph2d_profile_seeds import _evaluate_gate

    gate = _evaluate_gate(
        rows=[{"status": "ok"}],
        strict_accuracy_mean=0.31,
        strict_accuracy_min=0.27,
        min_strict_accuracy_mean=-1.0,
        min_strict_accuracy_min=-1.0,
        min_manifest_distinct_labels=-1,
        max_strict_top_pred_ratio=-1.0,
        max_strict_low_conf_ratio=-1.0,
        require_all_ok=False,
    )
    assert gate["enabled"] is False
    assert gate["passed"] is True
    assert gate["failures"] == []


def test_evaluate_gate_fails_when_mean_below_threshold() -> None:
    from scripts.sweep_graph2d_profile_seeds import _evaluate_gate

    gate = _evaluate_gate(
        rows=[{"status": "ok"}, {"status": "ok"}],
        strict_accuracy_mean=0.29,
        strict_accuracy_min=0.27,
        min_strict_accuracy_mean=0.30,
        min_strict_accuracy_min=-1.0,
        min_manifest_distinct_labels=-1,
        max_strict_top_pred_ratio=-1.0,
        max_strict_low_conf_ratio=-1.0,
        require_all_ok=False,
    )
    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert any("strict_accuracy_mean" in failure for failure in gate["failures"])


def test_evaluate_gate_fails_when_require_all_ok_and_errors_exist() -> None:
    from scripts.sweep_graph2d_profile_seeds import _evaluate_gate

    gate = _evaluate_gate(
        rows=[{"status": "ok"}, {"status": "error"}],
        strict_accuracy_mean=0.35,
        strict_accuracy_min=0.30,
        min_strict_accuracy_mean=-1.0,
        min_strict_accuracy_min=-1.0,
        min_manifest_distinct_labels=-1,
        max_strict_top_pred_ratio=-1.0,
        max_strict_low_conf_ratio=-1.0,
        require_all_ok=True,
    )
    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert gate["num_error_runs"] == 1
    assert any("require_all_ok" in failure for failure in gate["failures"])


def test_load_yaml_defaults_reads_section_and_normalizes_keys(tmp_path) -> None:
    from scripts.sweep_graph2d_profile_seeds import _load_yaml_defaults

    cfg = tmp_path / "seed_gate.yaml"
    cfg.write_text(
        "graph2d_seed_sweep:\n"
        "  seeds: \"7,21\"\n"
        "  manifest-label-mode: parent_dir\n"
        "  force-normalize-labels: false\n"
        "  force-clean-min-count: 0\n"
        "  retry-failures: 2\n"
        "  min-strict-accuracy-mean: 0.3\n",
        encoding="utf-8",
    )
    defaults = _load_yaml_defaults(str(cfg), "graph2d_seed_sweep")
    assert defaults["seeds"] == "7,21"
    assert defaults["manifest_label_mode"] == "parent_dir"
    assert bool(defaults["force_normalize_labels"]) is False
    assert int(defaults["force_clean_min_count"]) == 0
    assert int(defaults["retry_failures"]) == 2
    assert float(defaults["min_strict_accuracy_mean"]) == 0.3


def test_run_with_retries_succeeds_after_one_retry(monkeypatch) -> None:
    import scripts.sweep_graph2d_profile_seeds as sweep

    calls = {"count": 0}

    def _fake_run(_cmd, dry_run=False):
        _ = dry_run
        calls["count"] += 1
        if calls["count"] == 1:
            raise subprocess.CalledProcessError(9, ["fake"])

    monkeypatch.setattr(sweep, "_run", _fake_run)
    out = sweep._run_with_retries(
        ["fake"],
        dry_run=False,
        retry_failures=1,
        retry_backoff_seconds=0.0,
    )
    assert out["status"] == "ok"
    assert int(out["attempts"]) == 2
    assert int(out["return_code"]) == 0


def test_run_with_retries_returns_error_after_limit(monkeypatch) -> None:
    import scripts.sweep_graph2d_profile_seeds as sweep

    def _always_fail(_cmd, dry_run=False):
        _ = dry_run
        raise subprocess.CalledProcessError(5, ["fake"])

    monkeypatch.setattr(sweep, "_run", _always_fail)
    out = sweep._run_with_retries(
        ["fake"],
        dry_run=False,
        retry_failures=1,
        retry_backoff_seconds=0.0,
    )
    assert out["status"] == "error"
    assert int(out["attempts"]) == 2
    assert int(out["return_code"]) == 5


def test_evaluate_gate_fails_when_manifest_distinct_labels_below_threshold() -> None:
    from scripts.sweep_graph2d_profile_seeds import _evaluate_gate

    gate = _evaluate_gate(
        rows=[
            {"status": "ok", "seed": 7, "manifest_distinct_labels": 5},
            {"status": "ok", "seed": 21, "manifest_distinct_labels": 1},
        ],
        strict_accuracy_mean=0.60,
        strict_accuracy_min=0.50,
        min_strict_accuracy_mean=-1.0,
        min_strict_accuracy_min=-1.0,
        min_manifest_distinct_labels=3,
        max_strict_top_pred_ratio=-1.0,
        max_strict_low_conf_ratio=-1.0,
        require_all_ok=False,
    )
    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert any("manifest_distinct_labels" in failure for failure in gate["failures"])


def test_evaluate_gate_fails_when_top_pred_ratio_above_threshold() -> None:
    from scripts.sweep_graph2d_profile_seeds import _evaluate_gate

    gate = _evaluate_gate(
        rows=[
            {"status": "ok", "seed": 7, "strict_top_pred_ratio": 0.45},
            {"status": "ok", "seed": 21, "strict_top_pred_ratio": 0.95},
        ],
        strict_accuracy_mean=0.60,
        strict_accuracy_min=0.50,
        min_strict_accuracy_mean=-1.0,
        min_strict_accuracy_min=-1.0,
        min_manifest_distinct_labels=-1,
        max_strict_top_pred_ratio=0.90,
        max_strict_low_conf_ratio=-1.0,
        require_all_ok=False,
    )
    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert any("strict_top_pred_ratio" in failure for failure in gate["failures"])


def test_evaluate_gate_fails_when_low_conf_ratio_above_threshold() -> None:
    from scripts.sweep_graph2d_profile_seeds import _evaluate_gate

    gate = _evaluate_gate(
        rows=[
            {"status": "ok", "seed": 7, "strict_low_conf_ratio": 0.08},
            {"status": "ok", "seed": 21, "strict_low_conf_ratio": 0.41},
        ],
        strict_accuracy_mean=0.60,
        strict_accuracy_min=0.50,
        min_strict_accuracy_mean=-1.0,
        min_strict_accuracy_min=-1.0,
        min_manifest_distinct_labels=-1,
        max_strict_top_pred_ratio=-1.0,
        max_strict_low_conf_ratio=0.20,
        require_all_ok=False,
    )
    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert any("strict_low_conf_ratio" in failure for failure in gate["failures"])
