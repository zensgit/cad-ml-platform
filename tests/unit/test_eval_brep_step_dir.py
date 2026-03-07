import json
from pathlib import Path

from scripts.eval_brep_step_dir import (
    StepCase,
    _build_ok_row,
    _evaluate_cases,
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
    assert row["brep_valid_3d"] is True
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
                "brep_valid_3d": True,
                "primary_surface_type": "plane",
                "top_hint_label": "block",
                "graph_schema_version": "v2",
            },
            {
                "status": "ok",
                "brep_valid_3d": True,
                "primary_surface_type": "plane",
                "top_hint_label": "block",
                "graph_schema_version": "v2",
            },
            {
                "status": "load_failed",
                "brep_valid_3d": False,
                "primary_surface_type": "",
                "top_hint_label": "",
                "graph_schema_version": "",
            },
        ]
    )

    assert summary["sample_size"] == 3
    assert summary["status_counts"] == {"ok": 2, "load_failed": 1}
    assert summary["valid_3d_count"] == 2
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
    assert rows[0]["node_count"] == 7


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
