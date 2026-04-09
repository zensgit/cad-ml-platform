from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_archive_hybrid_blind_eval_history_writes_snapshot(tmp_path: Path) -> None:
    from scripts.ci import archive_hybrid_blind_eval_history as mod

    summary_json = tmp_path / "summary.json"
    gate_json = tmp_path / "gate.json"
    out_dir = tmp_path / "eval_history"
    _write_json(
        summary_json,
        {
            "sample_size": 50,
            "weak_labels": {
                "covered_rate": 0.8,
                "accuracy": {
                    "hybrid_label": {"accuracy": 0.42},
                    "graph2d_label": {"accuracy": 0.31},
                },
            },
        },
    )
    _write_json(gate_json, {"status": "passed", "metrics": {"hybrid_gain_vs_graph2d": 0.11}})

    rc = mod.main(
        [
            "--summary-json",
            str(summary_json),
            "--gate-report-json",
            str(gate_json),
            "--output-dir",
            str(out_dir),
            "--branch",
            "main",
            "--commit",
            "abc1234",
            "--runner",
            "ci",
            "--machine",
            "test-host",
            "--os-info",
            "Linux 6.0",
            "--python-version",
            "3.11.9",
            "--ci-job-id",
            "1001",
            "--ci-workflow",
            "Evaluation Report",
        ]
    )
    assert rc == 0
    files = list(out_dir.glob("*_hybrid_blind.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["type"] == "hybrid_blind"
    assert payload["metrics"]["sample_size"] == 50
    assert payload["metrics"]["hybrid_accuracy"] == 0.42
    assert payload["metrics"]["graph2d_accuracy"] == 0.31
    assert payload["metrics"]["hybrid_gain_vs_graph2d"] == 0.11
    assert payload["metrics"]["gate_status"] == "passed"
    assert payload["metrics"]["label_slices"] == []
    assert payload["metrics"]["family_slices"] == []


def test_archive_hybrid_blind_eval_history_includes_label_slices(tmp_path: Path) -> None:
    from scripts.ci import archive_hybrid_blind_eval_history as mod

    summary_json = tmp_path / "summary.json"
    out_dir = tmp_path / "eval_history"
    _write_json(
        summary_json,
        {
            "sample_size": 40,
            "weak_labels": {
                "covered_rate": 0.9,
                "label_counts": {"人孔": 10, "捕集口": 8, "传动件": 1},
                "accuracy": {
                    "hybrid_label": {"accuracy": 0.5},
                    "graph2d_label": {"accuracy": 0.2},
                },
                "by_true_label_accuracy": {
                    "hybrid_label": {
                        "人孔": {"evaluated": 10, "correct": 9, "accuracy": 0.9},
                        "捕集口": {"evaluated": 8, "correct": 6, "accuracy": 0.75},
                    },
                    "graph2d_label": {
                        "人孔": {"evaluated": 10, "correct": 3, "accuracy": 0.3},
                        "捕集口": {"evaluated": 8, "correct": 2, "accuracy": 0.25},
                    },
                },
            },
        },
    )

    rc = mod.main(
        [
            "--summary-json",
            str(summary_json),
            "--output-dir",
            str(out_dir),
            "--label-slice-min-support",
            "2",
            "--label-slice-max-slices",
            "5",
        ]
    )
    assert rc == 0
    files = list(out_dir.glob("*_hybrid_blind.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    slices = payload["metrics"]["label_slices"]
    assert len(slices) == 2
    assert slices[0]["label"] == "人孔"
    assert slices[0]["support"] == 10
    assert slices[0]["hybrid_accuracy"] == 0.9
    assert slices[0]["graph2d_accuracy"] == 0.3
    assert slices[0]["hybrid_gain_vs_graph2d"] == 0.6000000000000001
    assert payload["metrics"]["label_slice_meta"]["slice_count"] == 2
    family_slices = payload["metrics"]["family_slices"]
    assert len(family_slices) == 2
    assert family_slices[0]["family"] == "人孔"
    assert family_slices[0]["support"] == 10
    assert family_slices[1]["family"] == "捕集"
    assert payload["metrics"]["family_slice_meta"]["slice_count"] == 2


def test_archive_hybrid_blind_eval_history_redacts_non_hash_commit(tmp_path: Path) -> None:
    from scripts.ci import archive_hybrid_blind_eval_history as mod

    summary_json = tmp_path / "summary.json"
    out_dir = tmp_path / "eval_history"
    _write_json(
        summary_json,
        {
            "sample_size": 10,
            "weak_labels": {
                "covered_rate": 1.0,
                "accuracy": {
                    "hybrid_label": {"accuracy": 0.7},
                    "graph2d_label": {"accuracy": 0.4},
                },
            },
        },
    )

    rc = mod.main(
        [
            "--summary-json",
            str(summary_json),
            "--output-dir",
            str(out_dir),
            "--commit",
            "bootstrap",
        ]
    )
    assert rc == 0
    files = list(out_dir.glob("*_hybrid_blind.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["commit"] == "[redacted]"
