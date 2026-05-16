from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_SCRIPT = REPO_ROOT / "scripts" / "ci" / "build_brep_golden_manifest_optional.sh"


def _wrapper_env(**values: str) -> dict[str, str]:
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    return {
        **os.environ,
        "PATH": f"{python_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        **values,
    }


def _read_outputs(path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if not path.exists():
        return outputs
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        outputs[key] = value
    return outputs


def _write_case_file(root: Path, rel_path: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")


def _case(case_id: str, path: str) -> dict:
    return {
        "id": case_id,
        "path": path,
        "format": "step",
        "source_type": "real_world",
        "release_eligible": True,
        "part_family": "block",
        "license": "internal",
        "expected_behavior": "parse_success",
        "expected_topology": {
            "faces_min": 1,
            "edges_min": 0,
            "solids_min": 0,
            "graph_nodes_min": 1,
        },
    }


def _release_ready_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "parts"
    cases = []
    for index in range(2):
        rel_path = f"part_{index}.step"
        _write_case_file(root, rel_path)
        cases.append(_case(f"part_{index}", rel_path))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "brep_golden_manifest.v1",
                "name": "release ready smoke manifest",
                "root": str(root),
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_brep_golden_manifest_wrapper_skips_when_disabled(tmp_path: Path) -> None:
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        BREP_GOLDEN_MANIFEST_ENABLE="false",
        BREP_GOLDEN_EVAL_ENABLE="false",
    )

    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "skip" in result.stdout
    outputs = _read_outputs(github_output)
    assert outputs["enabled"] == "false"
    assert outputs["eval_enabled"] == "false"


def test_brep_golden_manifest_wrapper_reports_insufficient_example(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "validation.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        BREP_GOLDEN_MANIFEST_ENABLE="true",
        BREP_GOLDEN_MANIFEST_JSON="config/brep_golden_manifest.example.json",
        BREP_GOLDEN_MANIFEST_OUTPUT_JSON=str(output_json),
        BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY="false",
    )

    subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    outputs = _read_outputs(github_output)
    assert payload["status"] == "insufficient_release_samples"
    assert outputs["enabled"] == "true"
    assert outputs["validation_status"] == "insufficient_release_samples"
    assert outputs["ready_for_release"] == "false"
    assert outputs["eval_enabled"] == "false"


def test_brep_golden_manifest_wrapper_fails_when_release_gate_enabled(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "validation.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        BREP_GOLDEN_MANIFEST_ENABLE="true",
        BREP_GOLDEN_MANIFEST_JSON="config/brep_golden_manifest.example.json",
        BREP_GOLDEN_MANIFEST_OUTPUT_JSON=str(output_json),
        BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY="true",
    )

    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(output_json.read_text(encoding="utf-8"))["ready_for_release"] is False
    outputs = _read_outputs(github_output)
    assert outputs["enabled"] == "true"
    assert outputs["validation_status"] == "insufficient_release_samples"


def test_brep_golden_manifest_wrapper_accepts_release_ready_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _release_ready_manifest(tmp_path)
    output_json = tmp_path / "validation.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        BREP_GOLDEN_MANIFEST_ENABLE="true",
        BREP_GOLDEN_MANIFEST_JSON=str(manifest_path),
        BREP_GOLDEN_MANIFEST_OUTPUT_JSON=str(output_json),
        BREP_GOLDEN_MANIFEST_MIN_RELEASE_SAMPLES="2",
        BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY="true",
    )

    subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    outputs = _read_outputs(github_output)
    assert payload["status"] == "release_ready"
    assert outputs["validation_status"] == "release_ready"
    assert outputs["ready_for_release"] == "true"
    assert outputs["release_eligible_count"] == "2"
