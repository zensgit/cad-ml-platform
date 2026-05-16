import json
from pathlib import Path

from scripts.eval_brep_step_dir import (
    StepCase,
    _build_graph_qa_report,
    _build_ok_row,
    _evaluate_cases,
    _load_manifest_cases,
    _load_step_cases,
    _summarize_rows,
)


def test_load_step_cases_collects_unique_patterns(tmp_path: Path) -> None:
    (tmp_path / "a.step").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.stp").write_text("x", encoding="utf-8")
    (nested / "c.STEP").write_text("x", encoding="utf-8")

    cases = _load_step_cases(tmp_path, ["*.step", "*.stp", "*.STEP"])

    assert [case.relative_path for case in cases] == [
        "a.step",
        "nested/b.stp",
        "nested/c.STEP",
    ]


def test_load_step_cases_default_patterns_include_iges(tmp_path: Path) -> None:
    (tmp_path / "a.step").write_text("x", encoding="utf-8")
    (tmp_path / "b.iges").write_text("x", encoding="utf-8")

    cases = _load_step_cases(tmp_path, ["*.step", "*.iges"])

    assert [case.relative_path for case in cases] == ["a.step", "b.iges"]


def test_load_manifest_cases_resolves_paths_relative_to_manifest_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "part.step").write_text("x", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "brep_golden_manifest.v1",
                "root": "dataset",
                "cases": [{"id": "part", "path": "part.step"}],
            }
        ),
        encoding="utf-8",
    )

    cases = _load_manifest_cases(manifest)

    assert cases[0].file_path == (root / "part.step").resolve()
    assert cases[0].relative_path == "part.step"


def test_build_ok_row_includes_brep_summary_fields(tmp_path: Path) -> None:
    case = StepCase(file_path=tmp_path / "cube_hole.step", relative_path="cube_hole.step")
    features = {
        "valid_3d": True,
        "faces": 7,
        "edges": 30,
        "vertices": 60,
        "solids": 1,
        "shells": 1,
        "volume": 12.5,
        "surface_area": 23.5,
        "surface_types": {"plane": 6, "cylinder": 1},
        "bbox": {"diag": 62.217},
        "is_assembly": False,
    }
    graph = {
        "graph_schema_version": "v2",
        "node_count": 7,
        "edge_count": 28,
        "graph_metadata": {"undirected_edge_count": 14},
    }

    row = _build_ok_row(case, features, graph)

    assert row["status"] == "ok"
    assert row["parse_success"] is True
    assert row["brep_valid_3d"] is True
    assert row["graph_valid"] is True
    assert row["graph_schema_version"] == "v2"
    assert row["node_count"] == 7
    assert row["edge_count"] == 28
    assert row["primary_surface_type"] == "plane"
    assert row["top_hint_label"] == "block"
    assert json.loads(row["feature_hints"]) == {"block": 0.7, "plate": 0.5}


def test_summarize_rows_counts_surface_types_and_hints() -> None:
    summary = _summarize_rows(
        [
            {
                "status": "ok",
                "shape_loaded": True,
                "brep_valid_3d": True,
                "faces": 7,
                "node_count": 7,
                "edge_count": 28,
                "is_assembly": False,
                "primary_surface_type": "plane",
                "top_hint_label": "block",
                "graph_schema_version": "v2",
                "parse_success": True,
                "graph_valid": True,
                "surface_types": '{"plane": 6}',
                "extraction_latency_ms": 1.5,
            },
            {
                "status": "ok",
                "shape_loaded": True,
                "brep_valid_3d": True,
                "parse_success": True,
                "graph_valid": True,
                "faces": 6,
                "edges": 12,
                "solids": 1,
                "node_count": 6,
                "edge_count": 24,
                "is_assembly": True,
                "primary_surface_type": "plane",
                "top_hint_label": "block",
                "graph_schema_version": "v2",
                "surface_types": '{"plane": 5, "cylinder": 1}',
                "extraction_latency_ms": 2.5,
            },
            {
                "status": "load_failed",
                "shape_loaded": False,
                "brep_valid_3d": False,
                "parse_success": False,
                "graph_valid": False,
                "failure_reason": "step_parse_failed",
                "faces": 0,
                "edges": 0,
                "solids": 0,
                "node_count": 0,
                "edge_count": 0,
                "is_assembly": False,
                "primary_surface_type": "",
                "top_hint_label": "",
                "graph_schema_version": "",
                "surface_types": "",
                "extraction_latency_ms": 3.5,
            },
        ]
    )

    assert summary["sample_size"] == 3
    assert summary["status_counts"] == {"ok": 2, "load_failed": 1}
    assert summary["failure_reason_counts"] == {"step_parse_failed": 1}
    assert summary["parse_success_count"] == 2
    assert summary["shape_loaded_count"] == 2
    assert summary["valid_3d_count"] == 2
    assert summary["graph_valid_count"] == 2
    assert summary["hint_coverage_count"] == 2
    assert summary["assembly_count"] == 1
    assert summary["face_count_total"] == 13
    assert summary["edge_count_total"] == 12
    assert summary["solid_count_total"] == 1
    assert summary["avg_faces_ok"] == 6.5
    assert summary["avg_nodes_ok"] == 6.5
    assert summary["avg_edges_ok"] == 26.0
    assert summary["avg_extraction_latency_ms"] == 2.5
    assert summary["surface_type_histogram"] == {"plane": 11, "cylinder": 1}
    assert summary["primary_surface_type_counts"] == {"plane": 2}
    assert summary["top_hint_label_counts"] == {"block": 2}
    assert summary["graph_schema_version_counts"] == {"v2": 2}


def test_evaluate_cases_uses_geometry_engine(tmp_path: Path) -> None:
    step_file = tmp_path / "sample.step"
    step_file.write_text("fake-step", encoding="utf-8")

    class FakeEngine:
        def load_step(self, content: bytes, file_name: str):
            assert file_name == "sample.step"
            assert content == b"fake-step"
            return object()

        def extract_brep_features(self, shape):
            return {
                "valid_3d": True,
                "faces": 7,
                "edges": 10,
                "vertices": 12,
                "solids": 1,
                "shells": 1,
                "volume": 4.2,
                "surface_area": 8.4,
                "surface_types": {"plane": 6, "cylinder": 1},
                "bbox": {"diag": 3.14},
            }

        def extract_brep_graph(self, shape):
            return {
                "graph_schema_version": "v2",
                "node_count": 7,
                "edge_count": 28,
                "graph_metadata": {"undirected_edge_count": 14},
            }

    rows = _evaluate_cases(
        [StepCase(file_path=step_file, relative_path="sample.step")],
        has_occ=True,
        engine=FakeEngine(),
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["parse_success"] is True
    assert rows[0]["graph_valid"] is True
    assert rows[0]["node_count"] == 7


def test_evaluate_cases_strict_marks_invalid_brep_as_failure(tmp_path: Path) -> None:
    step_file = tmp_path / "invalid.step"
    step_file.write_text("fake-step", encoding="utf-8")

    class FakeEngine:
        def load_step(self, content: bytes, file_name: str):
            return object()

        def extract_brep_features(self, shape):
            return {
                "valid_3d": False,
                "faces": 0,
                "edges": 0,
                "vertices": 0,
                "solids": 0,
                "shells": 0,
                "surface_types": {},
                "bbox": {},
            }

        def extract_brep_graph(self, shape):
            return {"valid_3d": True, "graph_schema_version": "v2", "node_count": 0}

    rows = _evaluate_cases(
        [StepCase(file_path=step_file, relative_path="invalid.step")],
        has_occ=True,
        engine=FakeEngine(),
        strict=True,
    )

    assert rows[0]["status"] == "invalid_brep"
    assert rows[0]["failure_reason"] == "brep_features_invalid"
    assert rows[0]["parse_success"] is True


def test_evaluate_cases_strict_rejects_synthetic_geometry_unless_demo_allowed(
    tmp_path: Path,
) -> None:
    step_file = tmp_path / "demo.step"
    step_file.write_text("fake-step", encoding="utf-8")

    class FakeEngine:
        def load_step(self, content: bytes, file_name: str):
            return object()

        def extract_brep_features(self, shape):
            return {
                "valid_3d": True,
                "synthetic_geometry": True,
                "faces": 6,
                "edges": 12,
                "vertices": 8,
                "solids": 1,
                "shells": 1,
                "surface_types": {"plane": 6},
                "bbox": {"diag": 1.0},
            }

        def extract_brep_graph(self, shape):
            return {
                "valid_3d": True,
                "graph_schema_version": "v2",
                "node_count": 6,
                "edge_count": 12,
            }

    strict_rows = _evaluate_cases(
        [StepCase(file_path=step_file, relative_path="demo.step")],
        has_occ=True,
        engine=FakeEngine(),
        strict=True,
    )
    demo_rows = _evaluate_cases(
        [StepCase(file_path=step_file, relative_path="demo.step")],
        has_occ=True,
        engine=FakeEngine(),
        strict=True,
        demo_geometry_allowed=True,
    )

    assert strict_rows[0]["status"] == "demo_geometry_rejected"
    assert strict_rows[0]["failure_reason"] == "synthetic_geometry_not_allowed"
    assert demo_rows[0]["status"] == "ok"
    assert demo_rows[0]["demo_geometry_allowed"] is True


def test_evaluate_cases_uses_iges_loader_when_available(tmp_path: Path) -> None:
    iges_file = tmp_path / "sample.igs"
    iges_file.write_text("fake-iges", encoding="utf-8")

    class FakeEngine:
        def load_iges(self, content: bytes, file_name: str):
            assert file_name == "sample.igs"
            assert content == b"fake-iges"
            return object()

        def extract_brep_features(self, shape):
            return {
                "valid_3d": True,
                "faces": 2,
                "edges": 1,
                "vertices": 2,
                "solids": 0,
                "shells": 1,
                "surface_types": {"plane": 2},
                "bbox": {"diag": 1.0},
            }

        def extract_brep_graph(self, shape):
            return {"valid_3d": True, "graph_schema_version": "v2", "node_count": 2}

    rows = _evaluate_cases(
        [StepCase(file_path=iges_file, relative_path="sample.igs")],
        has_occ=True,
        engine=FakeEngine(),
        strict=True,
    )

    assert rows[0]["status"] == "ok"
    assert rows[0]["file_format"] == "iges"


def test_evaluate_cases_marks_iges_missing_loader_as_unsupported(tmp_path: Path) -> None:
    iges_file = tmp_path / "sample.iges"
    iges_file.write_text("fake-iges", encoding="utf-8")

    class FakeEngine:
        def load_step(self, content: bytes, file_name: str):
            raise AssertionError("IGES should not use the STEP loader")

    rows = _evaluate_cases(
        [StepCase(file_path=iges_file, relative_path="sample.iges")],
        has_occ=True,
        engine=FakeEngine(),
        strict=True,
    )

    assert rows[0]["status"] == "unsupported_format"
    assert rows[0]["failure_reason"] == "iges_loader_missing"
    assert rows[0]["parse_success"] is False


def test_build_graph_qa_report_lists_invalid_graph_rows() -> None:
    rows = [
        {
            "relative_path": "ok.step",
            "status": "ok",
            "failure_reason": "",
            "parse_success": True,
            "brep_valid_3d": True,
            "graph_valid": True,
            "faces": 6,
            "node_count": 6,
            "edge_count": 12,
        },
        {
            "relative_path": "bad.step",
            "status": "graph_invalid",
            "failure_reason": "brep_graph_invalid",
            "parse_success": True,
            "brep_valid_3d": True,
            "graph_valid": False,
            "faces": 6,
            "node_count": 0,
            "edge_count": 0,
        },
    ]
    summary = {
        "sample_size": 2,
        "parse_success_count": 2,
        "graph_valid_count": 1,
        "failure_reason_counts": {"brep_graph_invalid": 1},
    }

    report = _build_graph_qa_report(rows, summary)

    assert report["status"] == "failed"
    assert report["graph_valid_ratio"] == 0.5
    assert report["invalid_graph_rows"][0]["relative_path"] == "bad.step"


def test_evaluate_cases_requires_occ(tmp_path: Path) -> None:
    step_file = tmp_path / "sample.step"
    step_file.write_text("fake-step", encoding="utf-8")

    try:
        _evaluate_cases(
            [StepCase(file_path=step_file, relative_path="sample.step")],
            has_occ=False,
            engine=object(),
        )
    except RuntimeError as exc:
        assert "pythonocc-core" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError when OCC is unavailable")
