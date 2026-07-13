"""Phase-A bypass seals: every retrain/PROMOTION sink is fail-closed; ROLLBACK is not harmed.

The unconditional L3 gate (#509) guards scripts/auto_retrain.sh — but two sinks bypassed it:
  1. scripts/finetune_from_feedback.py — trains AND promotes (reload_model force=True), gate-free.
  2. POST /api/v1/model/reload — loads an ARBITRARY model path into the serving process.

These tests prove both are sealed BEFORE any side effect (no feedback export, no training, no
save, no reload), and that the emergency rollback path (auto_remediation, in-process, a KNOWN
previous model) still works — a seal that killed rollback would trade one risk for another.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


# --- 1. finetune_from_feedback.py: gate fires before ANY side effect ----------------------------
def _load_finetune_module():
    import importlib.util

    path = ROOT / "scripts" / "finetune_from_feedback.py"
    spec = importlib.util.spec_from_file_location("finetune_from_feedback_seal_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finetune_from_feedback_blocks_before_export(monkeypatch, capsys) -> None:
    mod = _load_finetune_module()

    # Spy: get_active_learner is the FIRST side-effectful step (feedback export follows it).
    called = {"learner": 0}

    def _spy_learner(*a, **k):
        called["learner"] += 1
        raise AssertionError("get_active_learner must never be reached (gate must fire first)")

    monkeypatch.setattr(mod, "get_active_learner", _spy_learner)
    monkeypatch.setattr(sys, "argv", ["finetune_from_feedback.py", "--force"])

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1                       # fail-closed exit
    assert called["learner"] == 0                    # zero side effects: no export/train/save/reload


def test_finetune_from_feedback_distrusts_a_subverted_gate(monkeypatch) -> None:
    # Even if the gate is swapped for a stub that RETURNS (no raise), the script must still refuse:
    # the gate has no pass path by construction, so a return can only mean subversion.
    mod = _load_finetune_module()
    monkeypatch.setattr(mod, "get_active_learner", lambda *a, **k: pytest.fail("reached learner"))
    monkeypatch.setattr(sys, "argv", ["finetune_from_feedback.py", "--force"])

    import eval_integrity_gate as gate_mod  # scripts/ dir is on sys.path via the module under test

    monkeypatch.setattr(gate_mod, "check", lambda: None)   # subverted: returns instead of raising
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


# --- 2. POST /api/v1/model/reload: sealed, reload_model never invoked ---------------------------
@pytest.fixture()
def api_client(monkeypatch):
    monkeypatch.setenv("X_API_KEY", "test")
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from fastapi.testclient import TestClient

    from src.main import app

    return TestClient(app)


def test_model_reload_route_is_sealed_403_and_never_loads(api_client, monkeypatch, tmp_path) -> None:
    import src.ml.classifier as classifier

    calls = {"reload": 0}

    def _spy_reload(*a, **k):
        calls["reload"] += 1
        return {"status": "success"}

    monkeypatch.setattr(classifier, "reload_model", _spy_reload)

    model = tmp_path / "candidate.pkl"
    model.write_bytes(b"\x80\x05N.")
    resp = api_client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
        json={"path": str(model), "force": True},
    )
    assert resp.status_code == 403                    # sealed
    assert "fail-closed" in resp.json()["detail"]
    assert calls["reload"] == 0                       # the loader is NEVER invoked via the API


def test_model_reload_seal_has_no_payload_bypass(api_client, monkeypatch) -> None:
    import src.ml.classifier as classifier

    monkeypatch.setattr(
        classifier, "reload_model", lambda *a, **k: pytest.fail("reload_model reached via API")
    )
    for payload in (
        {"path": None, "force": True},
        {"path": "", "expected_version": "v1"},
        {"path": "/tmp/x.pkl", "force": False},
    ):
        resp = api_client.post(
            "/api/v1/model/reload",
            headers={"X-API-Key": "test", "X-Admin-Token": "secret"},
            json=payload,
        )
        assert resp.status_code == 403, payload       # no payload shape opens the seal


# --- 3. Emergency ROLLBACK (in-process, known previous model) is NOT harmed ----------------------
def test_auto_remediation_rollback_still_works(monkeypatch, tmp_path) -> None:
    import src.ml.classifier as classifier
    from src.ml.monitoring.auto_remediation import AutoRemediation

    prev = tmp_path / "model_prev.pkl"
    prev.write_bytes(b"\x80\x05N.")
    calls = {"reload": []}
    monkeypatch.setattr(
        classifier, "reload_model", lambda *a, **k: (calls["reload"].append((a, k)) or {"status": "success"})
    )
    monkeypatch.setattr(classifier, "_MODEL_PREV_PATH", prev, raising=False)

    remediation = AutoRemediation()
    anomaly = types.SimpleNamespace(metric_name="accuracy_drop", severity="critical")
    result = remediation._action_rollback_model(anomaly)

    assert calls["reload"], "rollback must still reach reload_model (in-process, not via the API)"
    assert result["action"] == "rollback_model"
    assert result["previous_path"] == str(prev)


def test_model_reload_unauthenticated_is_rejected_before_the_seal(monkeypatch) -> None:
    # No credentials -> the auth dependencies reject (401/403) without ever reaching the handler;
    # with the seal behind them, there is no request shape that loads a model.
    monkeypatch.setenv("X_API_KEY", "test")
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from fastapi.testclient import TestClient

    from src.main import app

    client = TestClient(app)
    resp = client.post("/api/v1/model/reload", json={"path": "/tmp/x.pkl"})  # no auth headers
    assert resp.status_code in (401, 403)
