from __future__ import annotations


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
        require_all_ok=True,
    )
    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert gate["num_error_runs"] == 1
    assert any("require_all_ok" in failure for failure in gate["failures"])
