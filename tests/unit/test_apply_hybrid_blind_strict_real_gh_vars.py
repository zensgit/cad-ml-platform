from __future__ import annotations

from typing import Any


def test_apply_hybrid_blind_strict_real_builds_expected_var_map() -> None:
    from scripts.ci import apply_hybrid_blind_strict_real_gh_vars as mod

    var_map = mod._build_var_map("/data/dxf-real")
    assert var_map["HYBRID_BLIND_ENABLE"] == "true"
    assert var_map["HYBRID_BLIND_FAIL_ON_GATE_FAILED"] == "true"
    assert var_map["HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA"] == "true"
    assert var_map["HYBRID_BLIND_DXF_DIR"] == "/data/dxf-real"
    assert var_map["HYBRID_BLIND_DRIFT_ALERT_ENABLE"] == "true"


def test_apply_hybrid_blind_strict_real_main_print_only() -> None:
    from scripts.ci import apply_hybrid_blind_strict_real_gh_vars as mod

    rc = mod.main(["--repo", "zensgit/cad-ml-platform", "--dxf-dir", "/data/dxf-real"])
    assert rc == 0


def test_apply_hybrid_blind_strict_real_main_apply_mode(monkeypatch: Any) -> None:
    from scripts.ci import apply_hybrid_blind_strict_real_gh_vars as mod

    calls = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--dxf-dir",
            "/data/dxf-real",
            "--apply",
        ]
    )
    assert rc == 0
    assert len(calls) == 5
    assert calls[0][:3] == ["gh", "variable", "set"]


def test_apply_hybrid_blind_strict_real_apply_fails_on_any_error(
    monkeypatch: Any,
) -> None:
    from scripts.ci import apply_hybrid_blind_strict_real_gh_vars as mod

    idx = {"n": 0}

    def _fake_run(cmd: list[str], **kwargs: Any) -> Any:
        idx["n"] += 1

        class _Result:
            returncode = 1 if idx["n"] == 3 else 0
            stdout = ""
            stderr = "boom" if idx["n"] == 3 else ""

        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--dxf-dir",
            "/data/dxf-real",
            "--apply",
        ]
    )
    assert rc == 1
