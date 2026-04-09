from __future__ import annotations

from typing import Any


def test_apply_hybrid_superpass_builds_expected_var_map() -> None:
    from scripts.ci import apply_hybrid_superpass_gh_vars as mod

    var_map = mod._build_var_map("config/hybrid_superpass_targets.yaml")
    assert var_map["HYBRID_SUPERPASS_ENABLE"] == "true"
    assert var_map["HYBRID_SUPERPASS_MISSING_MODE"] == "fail"
    assert var_map["HYBRID_SUPERPASS_FAIL_ON_FAILED"] == "true"
    assert var_map["HYBRID_SUPERPASS_CONFIG"] == "config/hybrid_superpass_targets.yaml"


def test_apply_hybrid_superpass_main_print_only() -> None:
    from scripts.ci import apply_hybrid_superpass_gh_vars as mod

    rc = mod.main(["--repo", "zensgit/cad-ml-platform"])
    assert rc == 0


def test_apply_hybrid_superpass_main_apply_mode(monkeypatch: Any) -> None:
    from scripts.ci import apply_hybrid_superpass_gh_vars as mod

    calls = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = mod.main(["--repo", "zensgit/cad-ml-platform", "--apply"])
    assert rc == 0
    assert len(calls) == 4
    assert calls[0][:3] == ["gh", "variable", "set"]


def test_apply_hybrid_superpass_apply_fails_on_any_error(monkeypatch: Any) -> None:
    from scripts.ci import apply_hybrid_superpass_gh_vars as mod

    idx = {"n": 0}

    def _fake_run(cmd: list[str], **kwargs: Any) -> Any:
        idx["n"] += 1

        class _Result:
            returncode = 1 if idx["n"] == 2 else 0
            stdout = ""
            stderr = "boom" if idx["n"] == 2 else ""

        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = mod.main(["--repo", "zensgit/cad-ml-platform", "--apply"])
    assert rc == 1
