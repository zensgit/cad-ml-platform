"""L3: prove `scripts/auto_retrain.sh` is fail-closed at the evaluation-integrity gate.

The security-critical property: a missing / invalid / version-mismatched (or unset-env-defaulted)
artifact makes the script exit NON-ZERO **before any manifest write, preprocessing, or training**,
producing no manifest/model/cache side-effects and never reaching Step 1. A valid artifact lets the
script proceed past the gate. The gate cannot be turned off by an environment toggle. See §5.2, §8.1.

Hermetic: runs in a temp cwd containing only the two scripts under test, so the fail path needs no
ML stack and no side-effect can touch the real tree.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STEP1_MARKER = "Step 1: Reviewed samples"


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


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    """A temp cwd with only scripts/auto_retrain.sh + scripts/eval_integrity_gate.py."""
    (tmp_path / "scripts").mkdir()
    for name in ("auto_retrain.sh", "eval_integrity_gate.py"):
        shutil.copy(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    return tmp_path


def _run(sandbox: Path, *, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": __import__("os").environ.get("PATH", "")}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", "scripts/auto_retrain.sh"],
        cwd=str(sandbox), env=env, capture_output=True, text=True, timeout=120,
    )


def _assert_no_side_effects(sandbox: Path) -> None:
    manifests = list((sandbox / "data" / "manifests").glob("manifest_*.csv"))
    models = list((sandbox / "models").glob("graph2d_v*.pth"))
    cache = sandbox / "data" / "graph_cache"
    assert manifests == [], f"manifest side-effect leaked: {manifests}"
    assert models == [], f"model side-effect leaked: {models}"
    assert not cache.exists(), "graph_cache side-effect directory was created"


def test_missing_artifact_blocks_before_any_side_effect(sandbox: Path) -> None:
    r = _run(sandbox, env_extra={"EVAL_INTEGRITY_ARTIFACT": str(sandbox / "nope.json")})
    out = r.stdout + r.stderr
    assert r.returncode != 0, out                      # (1) fail-closed exit
    assert STEP1_MARKER not in out, out                # (2) died at the gate, not downstream
    _assert_no_side_effects(sandbox)                   # (3) no manifest/model/cache written
    assert "§5.2" in out and "§8.1" in out


def test_unset_env_defaults_and_still_blocks(sandbox: Path) -> None:
    # No EVAL_INTEGRITY_ARTIFACT set -> default path (absent) -> STILL checked (no fail-open).
    r = _run(sandbox)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert STEP1_MARKER not in out, out
    _assert_no_side_effects(sandbox)


def test_invalid_artifact_blocks(sandbox: Path) -> None:
    (sandbox / "bad.json").write_text("{ not json", encoding="utf-8")
    r = _run(sandbox, env_extra={"EVAL_INTEGRITY_ARTIFACT": str(sandbox / "bad.json")})
    assert r.returncode != 0
    assert STEP1_MARKER not in (r.stdout + r.stderr)
    _assert_no_side_effects(sandbox)


def test_version_mismatch_blocks(sandbox: Path) -> None:
    art = _valid_artifact()
    art["schema_version"] = "evaluation-integrity-v1"
    (sandbox / "old.json").write_text(json.dumps(art), encoding="utf-8")
    r = _run(sandbox, env_extra={"EVAL_INTEGRITY_ARTIFACT": str(sandbox / "old.json")})
    assert r.returncode != 0
    assert STEP1_MARKER not in (r.stdout + r.stderr)
    _assert_no_side_effects(sandbox)


def test_type_confused_all_boolean_artifact_blocks(sandbox: Path) -> None:
    # A structurally-present but type-confused artifact (every field == true) must NOT open the
    # pipeline — guards against a Track E producer schema bug, not just a hand-forged file.
    art = {
        "schema_version": "evaluation-integrity-v2",
        "split_strategy": True, "holdout": True, "metrics": True,
        "label_authority": True, "reproducible": True,
    }
    (sandbox / "boolish.json").write_text(json.dumps(art), encoding="utf-8")
    r = _run(sandbox, env_extra={"EVAL_INTEGRITY_ARTIFACT": str(sandbox / "boolish.json")})
    assert r.returncode != 0
    assert STEP1_MARKER not in (r.stdout + r.stderr)
    _assert_no_side_effects(sandbox)


def test_environment_toggles_cannot_bypass(sandbox: Path) -> None:
    # Plausible "skip" toggles must NOT open the gate (§8: not an environment toggle).
    r = _run(
        sandbox,
        env_extra={
            "EVAL_INTEGRITY_ARTIFACT": str(sandbox / "nope.json"),
            "SKIP_EVAL_GATE": "1", "FORCE": "1", "EVAL_GATE": "off",
            "EVALUATION_MODE": "soft", "DISABLE_GATE": "true",
        },
    )
    assert r.returncode != 0
    assert STEP1_MARKER not in (r.stdout + r.stderr)
    _assert_no_side_effects(sandbox)


def test_valid_artifact_proceeds_past_the_gate(sandbox: Path) -> None:
    # A valid artifact must let the script move PAST the gate. In the sandbox `src.ml` is not
    # importable, so Step 1 counts 0 reviewed samples and exits 0 with no side-effects — which is
    # exactly enough to prove the gate did not block (Step 1 marker present, exit 0).
    (sandbox / "ok.json").write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    r = _run(sandbox, env_extra={"EVAL_INTEGRITY_ARTIFACT": str(sandbox / "ok.json")})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert STEP1_MARKER in out, out                    # proves it passed the gate
    _assert_no_side_effects(sandbox)
