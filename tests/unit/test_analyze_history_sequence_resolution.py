from __future__ import annotations

from pathlib import Path

from src.api.v1.analyze import AnalysisOptions, _resolve_history_sequence_file_path


def test_resolve_history_path_from_options(tmp_path: Path) -> None:
    history = tmp_path / "opt.h5"
    history.write_bytes(b"")
    options = AnalysisOptions(history_file_path=str(history))

    path, source = _resolve_history_sequence_file_path(
        file_name="A001.dxf",
        file_format="dxf",
        analysis_options=options,
    )
    assert path == str(history)
    assert source == "options"


def test_resolve_history_path_from_env(monkeypatch, tmp_path: Path) -> None:
    history = tmp_path / "env.h5"
    history.write_bytes(b"")
    monkeypatch.setenv("HISTORY_SEQUENCE_FILE_PATH", str(history))

    path, source = _resolve_history_sequence_file_path(
        file_name="A001.dxf",
        file_format="dxf",
        analysis_options=AnalysisOptions(),
    )
    assert path == str(history)
    assert source == "env"


def test_resolve_history_path_from_sidecar_dir(monkeypatch, tmp_path: Path) -> None:
    history = tmp_path / "Bolt_M6x20.h5"
    history.write_bytes(b"")
    monkeypatch.setenv("HISTORY_SEQUENCE_SIDECAR_DIR", str(tmp_path))
    monkeypatch.delenv("HISTORY_SEQUENCE_FILE_PATH", raising=False)

    path, source = _resolve_history_sequence_file_path(
        file_name="Bolt_M6x20.dxf",
        file_format="dxf",
        analysis_options=AnalysisOptions(),
    )
    assert path == str(history)
    assert source == "sidecar_exact"


def test_resolve_history_path_non_dxf_returns_none(monkeypatch, tmp_path: Path) -> None:
    history = tmp_path / "part.h5"
    history.write_bytes(b"")
    monkeypatch.setenv("HISTORY_SEQUENCE_SIDECAR_DIR", str(tmp_path))
    monkeypatch.delenv("HISTORY_SEQUENCE_FILE_PATH", raising=False)

    path, source = _resolve_history_sequence_file_path(
        file_name="part.step",
        file_format="step",
        analysis_options=AnalysisOptions(),
    )
    assert path is None
    assert source is None


def test_resolve_history_path_rejects_non_h5_explicit_path(tmp_path: Path) -> None:
    invalid = tmp_path / "sample.txt"
    invalid.write_text("x", encoding="utf-8")
    options = AnalysisOptions(history_file_path=str(invalid))

    path, source = _resolve_history_sequence_file_path(
        file_name="A001.dxf",
        file_format="dxf",
        analysis_options=options,
    )
    assert path is None
    assert source is None


def test_resolve_history_path_respects_allowed_root(
    monkeypatch, tmp_path: Path
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.h5"
    outside.write_bytes(b"")
    inside = allowed / "inside.h5"
    inside.write_bytes(b"")

    monkeypatch.setenv("HISTORY_SEQUENCE_ALLOWED_ROOT", str(allowed))

    path_out, source_out = _resolve_history_sequence_file_path(
        file_name="A001.dxf",
        file_format="dxf",
        analysis_options=AnalysisOptions(history_file_path=str(outside)),
    )
    assert path_out is None
    assert source_out is None

    path_in, source_in = _resolve_history_sequence_file_path(
        file_name="A001.dxf",
        file_format="dxf",
        analysis_options=AnalysisOptions(history_file_path=str(inside)),
    )
    assert path_in == str(inside.resolve())
    assert source_in == "options"
