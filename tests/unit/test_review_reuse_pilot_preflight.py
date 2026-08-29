"""Unit tests for review_reuse_pilot_preflight exit codes (monkeypatch env)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest

from scripts.review_reuse_pilot_preflight import (
    check_dangerous_combos,
    main,
    read_posture,
    run_preflight,
)


def test_script_entrypoint_resolves_repo_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts/review_reuse_pilot_preflight.py"), "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def _clear_pilot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "REVIEW_REUSE_DECISIONS_ENABLED",
        "REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER",
        "REVIEW_REUSE_LIVE_DEDUP",
        "REVIEW_REUSE_STORE",
        "REVIEW_REUSE_STORE_DIR",
        "INTEGRATION_AUTH_MODE",
        "INTEGRATION_JWT_SECRET",
        "INTEGRATION_JWT_AUDIENCE",
        "INTEGRATION_JWT_ISSUER",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_env_advisory_exit_0(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pilot_env(monkeypatch)
    code, text = run_preflight(os.environ)
    assert code == 0
    assert "posture:              A" in text
    assert "decisions:            False" in text
    assert "warnings:             none" in text


def test_posture_b_evidence_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_LIVE_DEDUP", "true")
    monkeypatch.setenv("REVIEW_REUSE_STORE", "filesystem")
    posture = read_posture(os.environ)
    assert posture["posture"] == "B"
    assert posture["decisions"] is False
    code, _ = run_preflight(os.environ)
    assert code == 0


def test_safe_posture_c_exit_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decisions on + complete required auth → advisory OK."""
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", "true")
    monkeypatch.setenv("INTEGRATION_AUTH_MODE", "required")
    monkeypatch.setenv("INTEGRATION_JWT_SECRET", "secret")
    monkeypatch.setenv("INTEGRATION_JWT_AUDIENCE", "audience")
    monkeypatch.setenv("INTEGRATION_JWT_ISSUER", "issuer")
    monkeypatch.setenv("REVIEW_REUSE_STORE", "filesystem")
    code, text = run_preflight(os.environ)
    assert code == 0
    assert "posture:              C" in text
    assert "decisions:            True" in text
    assert "require_validated:    True" in text
    assert "integration_auth_mode:required" in text
    assert "warnings:             none" in text


def test_legacy_reviewer_flag_cannot_weaken_identity_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", "false")
    monkeypatch.setenv("INTEGRATION_AUTH_MODE", "required")
    monkeypatch.setenv("INTEGRATION_JWT_SECRET", "secret")
    monkeypatch.setenv("INTEGRATION_JWT_AUDIENCE", "audience")
    monkeypatch.setenv("INTEGRATION_JWT_ISSUER", "issuer")
    code, text = run_preflight(os.environ)
    assert code == 0
    assert "require_validated:    False" in text
    assert "warnings:             none" in text


def test_decisions_without_auth_required_exit_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", "true")
    monkeypatch.setenv("INTEGRATION_AUTH_MODE", "optional")
    monkeypatch.setenv("INTEGRATION_JWT_SECRET", "secret")
    monkeypatch.setenv("INTEGRATION_JWT_AUDIENCE", "audience")
    monkeypatch.setenv("INTEGRATION_JWT_ISSUER", "issuer")
    code, text = run_preflight(os.environ)
    assert code == 2
    assert "INTEGRATION_AUTH_MODE" in text
    assert "DANGER" in text


def test_decisions_with_incomplete_jwt_identity_exit_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("INTEGRATION_AUTH_MODE", "required")
    monkeypatch.setenv("INTEGRATION_JWT_SECRET", "secret")
    code, text = run_preflight(os.environ)
    assert code == 2
    assert "JWT secret, audience, or issuer is missing" in text


def test_decisions_with_padded_issuer_exit_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("INTEGRATION_AUTH_MODE", "required")
    monkeypatch.setenv("INTEGRATION_JWT_SECRET", "secret")
    monkeypatch.setenv("INTEGRATION_JWT_AUDIENCE", "audience")
    monkeypatch.setenv("INTEGRATION_JWT_ISSUER", " issuer ")

    code, text = run_preflight(os.environ)

    assert code == 2
    assert "issuer" in text.lower()


def test_decisions_both_dangers_exit_2(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "1")
    monkeypatch.setenv("INTEGRATION_AUTH_MODE", "disabled")
    # require_validated unset → false
    code, text = run_preflight(os.environ)
    assert code == 2
    warnings = check_dangerous_combos(read_posture(os.environ))
    assert len(warnings) == 1
    assert "DANGER" in text


def test_main_does_not_enable_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_pilot_env(monkeypatch)
    assert os.getenv("REVIEW_REUSE_DECISIONS_ENABLED") is None
    rc = main([])
    assert rc == 0
    assert os.getenv("REVIEW_REUSE_DECISIONS_ENABLED") is None


def test_main_exit_2_on_danger(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clear_pilot_env(monkeypatch)
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "yes")
    monkeypatch.setenv("INTEGRATION_AUTH_MODE", "disabled")
    rc = main([])
    assert rc == 2
    captured = capsys.readouterr()
    assert "DANGER" in captured.out
    assert "FAIL" in captured.err


def test_read_posture_from_mapping_not_os_environ() -> None:
    env: Dict[str, str] = {
        "REVIEW_REUSE_DECISIONS_ENABLED": "true",
        "REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER": "true",
        "INTEGRATION_AUTH_MODE": "required",
        "INTEGRATION_JWT_SECRET": "secret",
        "INTEGRATION_JWT_AUDIENCE": "audience",
        "INTEGRATION_JWT_ISSUER": "issuer",
        "REVIEW_REUSE_STORE": "filesystem",
        "REVIEW_REUSE_STORE_DIR": "/tmp/rr",
        "REVIEW_REUSE_LIVE_DEDUP": "on",
    }
    posture = read_posture(env)
    assert posture["decisions"] is True
    assert posture["require_validated"] is True
    assert posture["live_dedup"] is True
    assert posture["store"] == "filesystem"
    assert posture["store_dir"] == "/tmp/rr"
    assert posture["integration_auth_mode"] == "required"
    assert posture["posture"] == "C"
    code, _ = run_preflight(env)
    assert code == 0


def test_corrupt_filesystem_store_fails_preflight(
    tmp_path: Path,
) -> None:
    tenant_dir = tmp_path / ("tenant-v1-" + ("a" * 64))
    tenant_dir.mkdir(parents=True)
    env = {
        "REVIEW_REUSE_STORE": "filesystem",
        "REVIEW_REUSE_STORE_DIR": str(tmp_path),
    }

    code, text = run_preflight(env)
    assert code == 2
    assert "store integrity check failed" in text
    assert "store_record_corrupt" in text
