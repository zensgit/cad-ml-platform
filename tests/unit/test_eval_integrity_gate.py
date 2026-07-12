"""Unit tests for the fail-closed evaluation-integrity gate (L3).

Covers the three fail-closed modes named in the spec (missing / invalid / version-mismatch) plus
the required-fields and reproducible contract, and the valid pass. See PRODUCT_STRATEGY.md §5.2, §8.1.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import eval_integrity_gate as gate


def _valid_artifact() -> dict:
    return {
        "schema_version": "evaluation-integrity-v2",
        "split_strategy": "content-hash+normalized-family",
        "holdout": {"type": "customer-family", "families": 12},
        "metrics": {"per_class": {"gear": 0.93}, "macro_f1": 0.91, "calibration_ece": 0.03,
                    "false_duplicate_rate": 0.02, "missed_reuse_rate": 0.05},
        "label_authority": "human-verified",
        "reproducible": True,
    }


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
    return path


def test_valid_artifact_passes(tmp_path: Path) -> None:
    p = _write(tmp_path / "art.json", _valid_artifact())
    data = gate.validate_artifact(str(p))
    assert data["schema_version"] == "evaluation-integrity-v2"


def test_missing_artifact_is_failclosed(tmp_path: Path) -> None:
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(tmp_path / "does_not_exist.json"))
    assert exc.value.kind == "missing"


def test_unparseable_artifact_is_invalid(tmp_path: Path) -> None:
    p = _write(tmp_path / "art.json", "{ not json")
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


def test_non_object_artifact_is_invalid(tmp_path: Path) -> None:
    p = _write(tmp_path / "art.json", "[1, 2, 3]")
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


def test_version_mismatch_is_failclosed(tmp_path: Path) -> None:
    art = _valid_artifact()
    art["schema_version"] = "evaluation-integrity-v1"
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "version-mismatch"


@pytest.mark.parametrize("field", ["split_strategy", "holdout", "metrics", "label_authority"])
def test_missing_required_field_is_invalid(tmp_path: Path, field: str) -> None:
    art = _valid_artifact()
    del art[field]
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"
    assert field in exc.value.detail


@pytest.mark.parametrize("field", ["split_strategy", "holdout", "metrics"])
def test_empty_required_field_is_invalid(tmp_path: Path, field: str) -> None:
    art = _valid_artifact()
    art[field] = {} if field != "split_strategy" else ""
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


def test_all_boolean_fields_are_invalid(tmp_path: Path) -> None:
    # Type-confusion / Track E schema-bug guard: every required field present but set to `true`
    # (correct schema_version) must NOT open the training path.
    art = {
        "schema_version": "evaluation-integrity-v2",
        "split_strategy": True,
        "holdout": True,
        "metrics": True,
        "label_authority": True,
        "reproducible": True,
    }
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


def test_split_strategy_wrong_value_is_invalid(tmp_path: Path) -> None:
    art = _valid_artifact()
    art["split_strategy"] = "path-only"
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


@pytest.mark.parametrize("field", ["holdout", "metrics"])
def test_object_fields_must_be_objects(tmp_path: Path, field: str) -> None:
    art = _valid_artifact()
    art[field] = True
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


@pytest.mark.parametrize("family", list(gate.REQUIRED_METRIC_KEYS))
def test_metrics_missing_a_family_is_invalid(tmp_path: Path, family: str) -> None:
    art = _valid_artifact()
    del art["metrics"][family]
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"
    assert family in exc.value.detail


@pytest.mark.parametrize("bad", [True, 1, "", "   ", {}, []])
def test_label_authority_must_be_nonempty_string_or_object(tmp_path: Path, bad) -> None:
    art = _valid_artifact()
    art["label_authority"] = bad
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


def test_reproducible_must_be_true(tmp_path: Path) -> None:
    art = _valid_artifact()
    art["reproducible"] = False
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


def test_reproducible_truthy_string_is_not_accepted(tmp_path: Path) -> None:
    # A truthy non-True value (e.g. the string "true") must NOT satisfy the reproducible gate.
    art = _valid_artifact()
    art["reproducible"] = "true"
    p = _write(tmp_path / "art.json", art)
    with pytest.raises(gate.GateError) as exc:
        gate.validate_artifact(str(p))
    assert exc.value.kind == "invalid"


# --- CLI exit codes (stdlib-only, so this runs without the ML stack) ----------------------------
def _run_cli(artifact: str) -> subprocess.CompletedProcess:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "eval_integrity_gate.py"),
         "--artifact", artifact, "--require-version", "evaluation-integrity-v2"],
        capture_output=True, text=True,
    )


def test_cli_exits_nonzero_and_points_to_strategy_on_missing(tmp_path: Path) -> None:
    r = _run_cli(str(tmp_path / "nope.json"))
    assert r.returncode == 1
    assert "§5.2" in r.stderr and "§8.1" in r.stderr
    assert "environment toggle" in r.stderr


def test_cli_exits_zero_on_valid(tmp_path: Path) -> None:
    p = _write(tmp_path / "art.json", _valid_artifact())
    r = _run_cli(str(p))
    assert r.returncode == 0
    assert "PASS" in r.stdout
