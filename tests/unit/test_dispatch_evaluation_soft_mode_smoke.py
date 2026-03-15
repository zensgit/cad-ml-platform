from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def test_find_variable_value_handles_list_payload() -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    found, value = mod._find_variable_value(
        [
            {"name": "A", "value": "1"},
            {"name": "EVALUATION_STRICT_FAIL_MODE", "value": "hard"},
        ],
        "EVALUATION_STRICT_FAIL_MODE",
    )
    assert found is True
    assert value == "hard"


def test_main_restores_existing_variable(monkeypatch: Any, tmp_path: Path) -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    commands: list[list[str]] = []

    def _fake_run(
        command: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["gh", "variable", "list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    [{"name": "EVALUATION_STRICT_FAIL_MODE", "value": "hard"}]
                ),
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9001"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="Resolved strict fail mode: soft",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.dispatcher, "check_gh_ready", lambda: (True, ""))

    def _fake_dispatch_main(argv: list[str]) -> int:
        output_json = Path(argv[argv.index("--output-json") + 1])
        output_json.write_text(
            json.dumps(
                {
                    "run_id": 9001,
                    "run_url": "https://example.com/r/9001",
                    "conclusion": "success",
                    "overall_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.dispatcher, "main", _fake_dispatch_main)

    output_json = tmp_path / "soft_smoke.json"
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--ref",
            "feat/hybrid-blind-drift-autotune-e2e",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["soft_marker_ok"] is True
    assert payload["restore_ok"] is True
    assert payload["variable_found_before"] is True
    assert payload["variable_value_before"] == "hard"

    set_commands = [
        cmd for cmd in commands if cmd[:4] == ["gh", "variable", "set", "EVALUATION_STRICT_FAIL_MODE"]
    ]
    assert len(set_commands) == 2
    assert set_commands[0][-1] == "soft"
    assert set_commands[1][-1] == "hard"


def test_main_deletes_variable_when_originally_missing(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    commands: list[list[str]] = []

    def _fake_run(
        command: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["gh", "variable", "list"]:
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="[]", stderr=""
            )
        if command[:4] == ["gh", "run", "view", "9011"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="Resolved strict fail mode: soft",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.dispatcher, "check_gh_ready", lambda: (True, ""))

    def _fake_dispatch_main(argv: list[str]) -> int:
        output_json = Path(argv[argv.index("--output-json") + 1])
        output_json.write_text(
            json.dumps(
                {
                    "run_id": 9011,
                    "run_url": "https://example.com/r/9011",
                    "conclusion": "success",
                    "overall_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.dispatcher, "main", _fake_dispatch_main)

    output_json = tmp_path / "soft_smoke_missing.json"
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["variable_found_before"] is False
    assert payload["restore_ok"] is True

    delete_commands = [
        cmd for cmd in commands if cmd[:4] == ["gh", "variable", "delete", "EVALUATION_STRICT_FAIL_MODE"]
    ]
    assert len(delete_commands) == 1


def test_main_fails_when_soft_marker_missing(monkeypatch: Any, tmp_path: Path) -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    def _fake_run(
        command: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["gh", "variable", "list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    [{"name": "EVALUATION_STRICT_FAIL_MODE", "value": "hard"}]
                ),
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9022"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="no strict marker in log",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.dispatcher, "check_gh_ready", lambda: (True, ""))

    def _fake_dispatch_main(argv: list[str]) -> int:
        output_json = Path(argv[argv.index("--output-json") + 1])
        output_json.write_text(
            json.dumps(
                {
                    "run_id": 9022,
                    "run_url": "https://example.com/r/9022",
                    "conclusion": "success",
                    "overall_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.dispatcher, "main", _fake_dispatch_main)

    output_json = tmp_path / "soft_smoke_marker_missing.json"
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["soft_marker_ok"] is False
    assert "missing marker" in payload["soft_marker_message"]
    assert payload["restore_ok"] is True


def test_main_retries_until_marker_passes(monkeypatch: Any, tmp_path: Path) -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    commands: list[list[str]] = []
    sleep_calls: list[int] = []

    def _fake_run(
        command: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["gh", "variable", "list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    [{"name": "EVALUATION_STRICT_FAIL_MODE", "value": "hard"}]
                ),
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9101"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="missing soft marker",
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9102"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="Resolved strict fail mode: soft",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.dispatcher, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleep_calls.append(int(seconds)))

    dispatch_calls = {"count": 0}

    def _fake_dispatch_main(argv: list[str]) -> int:
        dispatch_calls["count"] += 1
        run_id = 9101 if dispatch_calls["count"] == 1 else 9102
        output_json = Path(argv[argv.index("--output-json") + 1])
        output_json.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_url": f"https://example.com/r/{run_id}",
                    "conclusion": "success",
                    "overall_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.dispatcher, "main", _fake_dispatch_main)

    output_json = tmp_path / "soft_smoke_retry.json"
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--max-dispatch-attempts",
            "2",
            "--retry-sleep-seconds",
            "7",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["max_dispatch_attempts"] == 2
    assert payload["retry_sleep_seconds"] == 7
    assert payload["soft_marker_ok"] is True
    assert payload["overall_exit_code"] == 0
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["soft_marker_ok"] is False
    assert payload["attempts"][1]["soft_marker_ok"] is True
    assert sleep_calls == [7]

    set_commands = [
        cmd for cmd in commands if cmd[:4] == ["gh", "variable", "set", "EVALUATION_STRICT_FAIL_MODE"]
    ]
    assert len(set_commands) == 2


def test_main_retry_exhausted_returns_failure(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    def _fake_run(
        command: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["gh", "variable", "list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    [{"name": "EVALUATION_STRICT_FAIL_MODE", "value": "hard"}]
                ),
                stderr="",
            )
        if command[:3] == ["gh", "run", "view"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="still missing marker",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.dispatcher, "check_gh_ready", lambda: (True, ""))

    dispatch_calls = {"count": 0}

    def _fake_dispatch_main(argv: list[str]) -> int:
        dispatch_calls["count"] += 1
        run_id = 9200 + dispatch_calls["count"]
        output_json = Path(argv[argv.index("--output-json") + 1])
        output_json.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_url": f"https://example.com/r/{run_id}",
                    "conclusion": "success",
                    "overall_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.dispatcher, "main", _fake_dispatch_main)

    output_json = tmp_path / "soft_smoke_retry_exhausted.json"
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--max-dispatch-attempts",
            "2",
            "--retry-sleep-seconds",
            "0",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 1

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dispatch_exit_code"] == 0
    assert payload["soft_marker_ok"] is False
    assert payload["overall_exit_code"] == 1
    assert len(payload["attempts"]) == 2
    assert all(item["soft_marker_ok"] is False for item in payload["attempts"])
    assert payload["restore_ok"] is True


def test_main_retries_until_marker_check_passes(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    sleep_calls: list[int] = []

    def _fake_sleep(seconds: int) -> None:
        sleep_calls.append(int(seconds))

    def _fake_run(
        command: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["gh", "variable", "list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    [{"name": "EVALUATION_STRICT_FAIL_MODE", "value": "hard"}]
                ),
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9101"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="marker missing",
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9102"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="Resolved strict fail mode: soft",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    dispatch_calls = {"count": 0}

    def _fake_dispatch_main(argv: list[str]) -> int:
        dispatch_calls["count"] += 1
        output_json = Path(argv[argv.index("--output-json") + 1])
        run_id = 9100 + dispatch_calls["count"]
        output_json.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_url": f"https://example.com/r/{run_id}",
                    "conclusion": "success",
                    "overall_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.time, "sleep", _fake_sleep)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.dispatcher, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod.dispatcher, "main", _fake_dispatch_main)

    output_json = tmp_path / "soft_smoke_retry_success.json"
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--output-json",
            str(output_json),
            "--max-dispatch-attempts",
            "3",
            "--retry-sleep-seconds",
            "7",
        ]
    )
    assert rc == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["max_dispatch_attempts"] == 3
    assert payload["dispatch"]["run_id"] == 9102
    assert payload["soft_marker_ok"] is True
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["soft_marker_ok"] is False
    assert payload["attempts"][1]["soft_marker_ok"] is True
    assert sleep_calls == [7]


def test_main_retry_exhausted_keeps_last_attempt_payload(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from scripts.ci import dispatch_evaluation_soft_mode_smoke as mod

    sleep_calls: list[int] = []

    def _fake_sleep(seconds: int) -> None:
        sleep_calls.append(int(seconds))

    def _fake_run(
        command: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["gh", "variable", "list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    [{"name": "EVALUATION_STRICT_FAIL_MODE", "value": "hard"}]
                ),
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9201"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="marker missing attempt 1",
                stderr="",
            )
        if command[:4] == ["gh", "run", "view", "9202"] and "--log" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="marker missing attempt 2",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    dispatch_calls = {"count": 0}

    def _fake_dispatch_main(argv: list[str]) -> int:
        dispatch_calls["count"] += 1
        output_json = Path(argv[argv.index("--output-json") + 1])
        run_id = 9200 + dispatch_calls["count"]
        output_json.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_url": f"https://example.com/r/{run_id}",
                    "conclusion": "success",
                    "overall_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.time, "sleep", _fake_sleep)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.dispatcher, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod.dispatcher, "main", _fake_dispatch_main)

    output_json = tmp_path / "soft_smoke_retry_fail.json"
    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--output-json",
            str(output_json),
            "--max-dispatch-attempts",
            "2",
            "--retry-sleep-seconds",
            "5",
        ]
    )
    assert rc == 1

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dispatch"]["run_id"] == 9202
    assert payload["soft_marker_ok"] is False
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["dispatch"]["run_id"] == 9201
    assert payload["attempts"][1]["dispatch"]["run_id"] == 9202
    assert payload["overall_exit_code"] == 1
    assert sleep_calls == [5]
