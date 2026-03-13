from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _valid_superpass_payload() -> dict:
    return {
        "status": "passed",
        "headline": "ok",
        "thresholds": {
            "min_hybrid_accuracy": 0.6,
            "min_hybrid_gain_vs_graph2d": 0.0,
            "max_calibration_ece": 0.08,
        },
        "checks": [{"name": "hybrid_accuracy", "passed": True}],
        "failures": [],
        "warnings": [],
    }


def _valid_gate_payload() -> dict:
    return {"metrics": {"hybrid_accuracy": 0.71, "hybrid_gain_vs_graph2d": 0.09}}


def _valid_calibration_payload() -> dict:
    return {"metrics_after": {"ece": 0.041}}


def test_validate_superpass_reports_ok(tmp_path: Path, capsys) -> None:
    from scripts.ci import validate_hybrid_superpass_reports as mod

    superpass = tmp_path / "superpass.json"
    gate = tmp_path / "gate.json"
    calibration = tmp_path / "calibration.json"
    _write_json(superpass, _valid_superpass_payload())
    _write_json(gate, _valid_gate_payload())
    _write_json(calibration, _valid_calibration_payload())

    rc = mod.main(
        [
            "--superpass-json",
            str(superpass),
            "--hybrid-blind-gate-report",
            str(gate),
            "--hybrid-calibration-json",
            str(calibration),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "ok"
    assert payload["errors"] == []
    assert payload["warnings"] == []


def test_validate_superpass_reports_error_on_missing_required_field(
    tmp_path: Path, capsys
) -> None:
    from scripts.ci import validate_hybrid_superpass_reports as mod

    superpass = tmp_path / "superpass.json"
    payload = _valid_superpass_payload()
    del payload["headline"]
    _write_json(superpass, payload)

    rc = mod.main(["--superpass-json", str(superpass)])
    assert rc == 1
    output = json.loads(capsys.readouterr().out.strip())
    assert output["status"] == "error"
    assert any("superpass.headline is required" in err for err in output["errors"])


def test_validate_superpass_reports_warn_and_strict_fail_on_warning(
    tmp_path: Path, capsys
) -> None:
    from scripts.ci import validate_hybrid_superpass_reports as mod

    superpass = tmp_path / "superpass.json"
    _write_json(superpass, _valid_superpass_payload())

    rc = mod.main(["--superpass-json", str(superpass)])
    assert rc == 0
    non_strict = json.loads(capsys.readouterr().out.strip())
    assert non_strict["status"] == "warn"
    assert any("hybrid_blind_gate report missing" in msg for msg in non_strict["warnings"])
    assert any("hybrid calibration report missing" in msg for msg in non_strict["warnings"])

    rc_strict = mod.main(["--superpass-json", str(superpass), "--strict"])
    assert rc_strict == 1
    strict_payload = json.loads(capsys.readouterr().out.strip())
    assert strict_payload["status"] == "warn"


def test_validate_superpass_reports_output_json_written(tmp_path: Path) -> None:
    from scripts.ci import validate_hybrid_superpass_reports as mod

    superpass = tmp_path / "superpass.json"
    output = tmp_path / "validation.json"
    _write_json(superpass, _valid_superpass_payload())

    rc = mod.main(
        [
            "--superpass-json",
            str(superpass),
            "--output-json",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "warn"
    assert payload["overall_exit_code"] == 0
