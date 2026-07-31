from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from src.core.cad.dwg.converter import (
    ConversionStatus,
    ConverterConfig,
    DWGConverter,
)
from src.core.dedupcad_precision import cad_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dwg_converter_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DWG_CONVERTER", raising=False)

    assert cad_pipeline.resolve_dwg_converter() == "disabled"


def test_invalid_dwg_converter_mode_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DWG_CONVERTER", "typo")

    with pytest.raises(RuntimeError, match="Unsupported DWG_CONVERTER"):
        cad_pipeline.resolve_dwg_converter()


def test_disabled_generic_converter_ignores_explicit_path_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    monkeypatch.setenv("ODA_FILE_CONVERTER", "/installed/ODAFileConverter")

    watched = {
        "/configured/ODAFileConverter",
        "/installed/ODAFileConverter",
    }
    probed: list[str] = []
    original_exists = Path.exists

    def track_exists(path: Path) -> bool:
        if str(path) in watched:
            probed.append(str(path))
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", track_exists)

    converter = DWGConverter(ConverterConfig(oda_path="/configured/ODAFileConverter"))

    assert converter.is_available is False
    assert probed == []


def test_disabled_precision_converter_ignores_explicit_path_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    monkeypatch.setenv("ODA_FILE_CONVERTER_EXE", "/installed/ODAFileConverter")

    watched = {"/installed/ODAFileConverter"}
    probed: list[str] = []
    original_exists = Path.exists

    def track_exists(path: Path) -> bool:
        if str(path) in watched:
            probed.append(str(path))
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", track_exists)

    assert cad_pipeline.resolve_oda_exe_from_env() is None
    assert probed == []


def test_disabled_mode_does_not_scan_installed_default_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    monkeypatch.delenv("ODA_FILE_CONVERTER", raising=False)
    monkeypatch.delenv("ODA_FILE_CONVERTER_EXE", raising=False)
    generic_candidate = (
        "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter"
    )
    probed: list[str] = []
    original_exists = Path.exists

    def track_exists(path: Path) -> bool:
        if str(path) == generic_candidate:
            probed.append(str(path))
            return True
        return original_exists(path)

    precision_candidates = Mock(
        side_effect=AssertionError(
            "disabled mode must not enumerate default candidates"
        )
    )
    monkeypatch.setattr(Path, "exists", track_exists)
    monkeypatch.setattr("src.core.cad.dwg.converter.platform.system", lambda: "Darwin")
    monkeypatch.setattr(cad_pipeline, "_default_oda_candidates", precision_candidates)

    assert DWGConverter().is_available is False
    assert cad_pipeline.resolve_oda_exe_from_env() is None
    assert probed == []
    precision_candidates.assert_not_called()


def test_disabled_generic_converter_never_starts_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter_path = tmp_path / "ODAFileConverter"
    converter_path.write_text("converter", encoding="utf-8")
    dwg_path = tmp_path / "sample.dwg"
    dwg_path.write_bytes(b"AC1032")
    run = Mock(side_effect=AssertionError("disabled mode must not start ODA"))

    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    monkeypatch.setattr(subprocess, "run", run)

    result = DWGConverter(ConverterConfig(oda_path=str(converter_path))).convert(
        dwg_path
    )

    assert result.status is ConversionStatus.CONVERTER_NOT_FOUND
    assert result.error_message == "DWG conversion disabled by DWG_CONVERTER"
    run.assert_not_called()


@pytest.mark.parametrize("mode", ["disabled", "cmd"])
def test_generic_oda_execution_sink_rejects_non_oda_modes(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = DWGConverter.__new__(DWGConverter)
    converter._config = ConverterConfig()
    converter._oda_path = "/installed/ODAFileConverter"
    run = Mock(side_effect=AssertionError("unauthorized ODA mode must not execute"))
    monkeypatch.setenv("DWG_CONVERTER", mode)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(RuntimeError, match="ODA conversion requires"):
        converter._run_oda_conversion(
            tmp_path / "sample.dwg",
            tmp_path / "sample.dxf",
            converter._config.output_version,
        )

    run.assert_not_called()


@pytest.mark.parametrize("helper", ["oda", "cmd"])
def test_disabled_direct_helpers_never_start_subprocess(
    helper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter_path = tmp_path / "ODAFileConverter"
    converter_path.write_text("converter", encoding="utf-8")
    dwg_path = tmp_path / "sample.dwg"
    dwg_path.write_bytes(b"AC1032")
    output_path = tmp_path / "sample.dxf"
    run = Mock(side_effect=AssertionError("disabled mode must not start a helper"))

    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    monkeypatch.setattr(cad_pipeline.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="conversion requires DWG_CONVERTER"):
        if helper == "oda":
            cad_pipeline.convert_dwg_to_dxf_oda(
                dwg_path,
                output_path,
                cfg=cad_pipeline.OdaConverterConfig(exe_path=converter_path),
            )
        else:
            cad_pipeline.convert_dwg_to_dxf_cmd(
                dwg_path,
                output_path,
                cmd_template="converter {input} {output}",
            )

    run.assert_not_called()


def test_cmd_mode_cannot_start_oda_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter_path = tmp_path / "ODAFileConverter"
    converter_path.write_text("converter", encoding="utf-8")
    dwg_path = tmp_path / "sample.dwg"
    dwg_path.write_bytes(b"AC1032")
    run = Mock(side_effect=AssertionError("cmd mode must not start ODA"))
    monkeypatch.setenv("DWG_CONVERTER", "cmd")
    monkeypatch.setattr(cad_pipeline.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="ODA conversion requires"):
        cad_pipeline.convert_dwg_to_dxf_oda(
            dwg_path,
            tmp_path / "sample.dxf",
            cfg=cad_pipeline.OdaConverterConfig(exe_path=converter_path),
        )

    run.assert_not_called()


def test_disabled_worker_entry_never_resolves_or_runs_a_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dwg_path = tmp_path / "sample.dwg"
    dwg_path.write_bytes(b"AC1032")
    output_path = tmp_path / "sample.dxf"
    resolver = Mock(side_effect=AssertionError("disabled worker must not resolve ODA"))
    run = Mock(side_effect=AssertionError("disabled worker must not start a converter"))

    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    monkeypatch.setattr(cad_pipeline, "resolve_oda_exe_from_env", resolver)
    monkeypatch.setattr(cad_pipeline.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="DWG conversion disabled by DWG_CONVERTER"):
        cad_pipeline.convert_dwg_to_dxf(dwg_path, output_path)

    resolver.assert_not_called()
    run.assert_not_called()


def test_explicit_auto_mode_preserves_both_oda_path_integrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic_path = tmp_path / "generic-ODAFileConverter"
    generic_path.write_text("converter", encoding="utf-8")
    precision_path = tmp_path / "precision-ODAFileConverter"
    precision_path.write_text("converter", encoding="utf-8")

    monkeypatch.setenv("DWG_CONVERTER", "auto")
    monkeypatch.setenv("ODA_FILE_CONVERTER_EXE", str(precision_path))

    assert DWGConverter(ConverterConfig(oda_path=str(generic_path))).oda_path == str(
        generic_path
    )
    assert cad_pipeline.resolve_oda_exe_from_env() == precision_path


def test_explicit_cmd_mode_preserves_direct_command_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dwg_path = tmp_path / "sample.dwg"
    dwg_path.write_bytes(b"AC1032")
    output_path = tmp_path / "sample.dxf"

    def run_command(_cmd: list[str], *, check: bool) -> None:
        assert check is True
        output_path.write_text("DXF", encoding="utf-8")

    monkeypatch.setenv("DWG_CONVERTER", "cmd")
    monkeypatch.setattr(cad_pipeline.subprocess, "run", run_command)

    cad_pipeline.convert_dwg_to_dxf_cmd(
        dwg_path,
        output_path,
        cmd_template="converter {input} {output}",
    )

    assert output_path.read_text(encoding="utf-8") == "DXF"


def test_tracked_runtime_templates_pin_dwg_conversion_disabled() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    render_server = (REPO_ROOT / "scripts/run_cad_render_server.sh").read_text(
        encoding="utf-8"
    )
    compose = yaml.safe_load(
        (REPO_ROOT / "deployments/docker/docker-compose.yml").read_text(
            encoding="utf-8"
        )
    )
    root_compose = yaml.safe_load(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    helm_values = yaml.safe_load(
        (REPO_ROOT / "charts/cad-ml-platform/values.yaml").read_text(encoding="utf-8")
    )
    helm_prod = yaml.safe_load(
        (REPO_ROOT / "charts/cad-ml-platform/values-prod.yaml").read_text(
            encoding="utf-8"
        )
    )
    helm_worker_template = (
        REPO_ROOT / "charts/cad-ml-platform/templates/dedup2d-configmap.yaml"
    ).read_text(encoding="utf-8")
    kustomize_config = yaml.safe_load(
        (REPO_ROOT / "k8s/kustomize/base/configmap.yaml").read_text(encoding="utf-8")
    )

    assert "DWG_CONVERTER=disabled" in env_example
    assert 'DWG_CONVERTER="${DWG_CONVERTER:-disabled}"' in render_server
    assert "ODA_FILE_CONVERTER_EXE=" not in render_server
    assert (
        "DWG_CONVERTER=${DWG_CONVERTER:-disabled}"
        in compose["services"]["cad-ml-api"]["environment"]
    )
    assert (
        "DWG_CONVERTER=${DWG_CONVERTER:-disabled}"
        in compose["services"]["dedup2d-worker"]["environment"]
    )
    assert (
        "DWG_CONVERTER=${DWG_CONVERTER:-disabled}"
        in root_compose["services"]["cad-ml-platform"]["environment"]
    )
    assert helm_values["env"]["DWG_CONVERTER"] == "disabled"
    assert helm_prod["env"]["DWG_CONVERTER"] == "disabled"
    assert (
        'DWG_CONVERTER: {{ .Values.env.DWG_CONVERTER | default "disabled" | quote }}'
        in (helm_worker_template)
    )
    assert kustomize_config["data"]["DWG_CONVERTER"] == "disabled"


def test_required_ci_runs_dwg_conversion_policy_gate() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci-tiered-tests.yml").read_text(
        encoding="utf-8"
    )

    assert "Run DWG conversion disabled policy gate" in workflow
    assert "pytest -q tests/unit/test_dwg_conversion_disabled_policy.py" in workflow
