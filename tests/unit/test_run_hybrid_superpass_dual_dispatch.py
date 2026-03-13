from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_main_print_only_outputs_commands_and_writes_summary(
    capsys: Any,
    tmp_path: Path,
) -> None:
    from scripts.ci import run_hybrid_superpass_dual_dispatch as mod

    fail_json = tmp_path / "out" / "fail.json"
    success_json = tmp_path / "out" / "success.json"
    compare_json = tmp_path / "out" / "compare.json"
    compare_md = tmp_path / "out" / "compare.md"
    summary_json = tmp_path / "out" / "summary.json"

    rc = mod.main(
        [
            "--print-only",
            "--dispatch-trace-prefix",
            "dsp-fixed",
            "--fail-output-json",
            str(fail_json),
            "--success-output-json",
            str(success_json),
            "--compare-output-json",
            str(compare_json),
            "--compare-output-md",
            str(compare_md),
            "--output-json",
            str(summary_json),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "fail_dispatch_command=" in out
    assert "success_dispatch_command=" in out
    assert "compare_command=" in out
    assert "dsp-fixed-fail" in out
    assert "dsp-fixed-success" in out

    payload = _read_json(compare_json)
    assert payload["mode"] == "print_only"
    assert payload["overall_exit_code"] == 0
    assert payload["fail_dispatch_exit_code"] is None
    assert payload["success_dispatch_exit_code"] is None
    assert payload["compare_exit_code"] is None
    summary_payload = _read_json(summary_json)
    assert summary_payload == payload


def test_main_success_when_dispatch_and_compare_all_zero(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from scripts.ci import run_hybrid_superpass_dual_dispatch as mod

    fail_json = tmp_path / "out" / "fail.json"
    success_json = tmp_path / "out" / "success.json"
    compare_json = tmp_path / "out" / "compare.json"
    compare_md = tmp_path / "out" / "compare.md"
    summary_json = tmp_path / "out" / "summary.json"

    popen_calls: list[list[str]] = []
    popen_exit_codes = [0, 0]

    class _FakeProcess:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.wait_called = False

        def wait(self) -> int:
            self.wait_called = True
            return self.returncode

    fake_processes: list[_FakeProcess] = []

    def _fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProcess:
        del args, kwargs
        popen_calls.append(cmd)
        proc = _FakeProcess(returncode=popen_exit_codes[len(popen_calls) - 1])
        fake_processes.append(proc)
        return proc

    run_calls: list[list[str]] = []

    def _fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        run_calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            "--dispatch-trace-prefix",
            "dsp-ok",
            "--fail-output-json",
            str(fail_json),
            "--success-output-json",
            str(success_json),
            "--compare-output-json",
            str(compare_json),
            "--compare-output-md",
            str(compare_md),
            "--output-json",
            str(summary_json),
        ]
    )

    assert rc == 0
    assert len(popen_calls) == 2
    assert len(run_calls) == 1
    assert all(proc.wait_called for proc in fake_processes)

    payload = _read_json(compare_json)
    assert payload["fail_dispatch_exit_code"] == 0
    assert payload["success_dispatch_exit_code"] == 0
    assert payload["compare_exit_code"] == 0
    assert payload["overall_exit_code"] == 0
    assert _read_json(summary_json) == payload


def test_main_returns_one_when_dispatch_fails(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from scripts.ci import run_hybrid_superpass_dual_dispatch as mod

    fail_json = tmp_path / "out" / "fail.json"
    success_json = tmp_path / "out" / "success.json"
    compare_json = tmp_path / "out" / "compare.json"
    compare_md = tmp_path / "out" / "compare.md"

    popen_calls: list[list[str]] = []
    popen_exit_codes = [1, 0]

    class _FakeProcess:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

        def wait(self) -> int:
            return self.returncode

    def _fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProcess:
        del args, kwargs
        popen_calls.append(cmd)
        return _FakeProcess(returncode=popen_exit_codes[len(popen_calls) - 1])

    run_calls: list[list[str]] = []

    def _fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        run_calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            "--dispatch-trace-prefix",
            "dsp-fail",
            "--fail-output-json",
            str(fail_json),
            "--success-output-json",
            str(success_json),
            "--compare-output-json",
            str(compare_json),
            "--compare-output-md",
            str(compare_md),
        ]
    )

    assert rc == 1
    assert len(run_calls) == 1

    payload = _read_json(compare_json)
    assert payload["fail_dispatch_exit_code"] == 1
    assert payload["success_dispatch_exit_code"] == 0
    assert payload["compare_exit_code"] == 0
    assert payload["overall_exit_code"] == 1


def test_main_returns_one_when_compare_fails(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from scripts.ci import run_hybrid_superpass_dual_dispatch as mod

    fail_json = tmp_path / "out" / "fail.json"
    success_json = tmp_path / "out" / "success.json"
    compare_json = tmp_path / "out" / "compare.json"
    compare_md = tmp_path / "out" / "compare.md"

    popen_calls: list[list[str]] = []

    class _FakeProcess:
        def wait(self) -> int:
            return 0

    def _fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProcess:
        del args, kwargs
        popen_calls.append(cmd)
        return _FakeProcess()

    run_calls: list[list[str]] = []

    def _fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        run_calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=2)

    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            "--strict",
            "--dispatch-trace-prefix",
            "dsp-compare-fail",
            "--fail-output-json",
            str(fail_json),
            "--success-output-json",
            str(success_json),
            "--compare-output-json",
            str(compare_json),
            "--compare-output-md",
            str(compare_md),
        ]
    )

    assert rc == 1
    assert len(popen_calls) == 2
    assert len(run_calls) == 1
    assert "--strict" in run_calls[0]

    payload = _read_json(compare_json)
    assert payload["fail_dispatch_exit_code"] == 0
    assert payload["success_dispatch_exit_code"] == 0
    assert payload["compare_exit_code"] == 2
    assert payload["overall_exit_code"] == 1
