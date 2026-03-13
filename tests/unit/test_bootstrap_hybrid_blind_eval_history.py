from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_bootstrap_hybrid_blind_eval_history_writes_multiple_snapshots(tmp_path: Path) -> None:
    from scripts.ci import bootstrap_hybrid_blind_eval_history as mod

    summary_json = tmp_path / "summary.json"
    gate_json = tmp_path / "gate.json"
    output_dir = tmp_path / "eval_history"

    _write_json(
        summary_json,
        {
            "sample_size": 30,
            "weak_labels": {
                "covered_rate": 0.9,
                "label_counts": {"人孔": 10, "捕集口": 8},
                "accuracy": {
                    "hybrid_label": {"accuracy": 0.5},
                    "graph2d_label": {"accuracy": 0.3},
                },
                "by_true_label_accuracy": {
                    "hybrid_label": {
                        "人孔": {"evaluated": 10, "correct": 7, "accuracy": 0.7},
                        "捕集口": {"evaluated": 8, "correct": 4, "accuracy": 0.5},
                    },
                    "graph2d_label": {
                        "人孔": {"evaluated": 10, "correct": 3, "accuracy": 0.3},
                        "捕集口": {"evaluated": 8, "correct": 2, "accuracy": 0.25},
                    },
                },
            },
        },
    )
    _write_json(gate_json, {"status": "passed"})

    rc = mod.main(
        [
            "--summary-json",
            str(summary_json),
            "--gate-report-json",
            str(gate_json),
            "--output-dir",
            str(output_dir),
            "--branch",
            "main",
            "--commit",
            "abc1234",
            "--count",
            "3",
            "--hours-step",
            "12",
            "--hybrid-accuracy-deltas",
            "0,-0.05,-0.10",
            "--graph2d-accuracy-deltas",
            "0,0,0",
            "--coverage-deltas",
            "0,-0.02,-0.04",
        ]
    )
    assert rc == 0

    files = sorted(output_dir.glob("*_hybrid_blind.json"))
    assert len(files) == 3
    first = json.loads(files[0].read_text(encoding="utf-8"))
    last = json.loads(files[-1].read_text(encoding="utf-8"))

    assert first["metrics"]["hybrid_accuracy"] == 0.5
    assert last["metrics"]["hybrid_accuracy"] == 0.4
    assert first["metrics"]["hybrid_gain_vs_graph2d"] == 0.2
    assert round(last["metrics"]["hybrid_gain_vs_graph2d"], 6) == 0.1
    assert last["metrics"]["weak_label_coverage"] == 0.86
    assert isinstance(last["metrics"].get("label_slices"), list)
    assert isinstance(last["metrics"].get("family_slices"), list)


def test_bootstrap_hybrid_blind_eval_history_direct_script_execution(tmp_path: Path) -> None:
    summary_json = tmp_path / "summary.json"
    output_dir = tmp_path / "eval_history"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "ci"
        / "bootstrap_hybrid_blind_eval_history.py"
    )

    _write_json(
        summary_json,
        {
            "sample_size": 10,
            "weak_labels": {
                "covered_rate": 1.0,
                "label_counts": {"人孔": 10},
                "accuracy": {
                    "hybrid_label": {"accuracy": 0.8},
                    "graph2d_label": {"accuracy": 0.4},
                },
                "by_true_label_accuracy": {
                    "hybrid_label": {"人孔": {"evaluated": 10, "correct": 8, "accuracy": 0.8}},
                    "graph2d_label": {"人孔": {"evaluated": 10, "correct": 4, "accuracy": 0.4}},
                },
            },
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-json",
            str(summary_json),
            "--output-dir",
            str(output_dir),
            "--count",
            "2",
            "--hours-step",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    files = sorted(output_dir.glob("*_hybrid_blind.json"))
    assert len(files) == 2


def test_bootstrap_hybrid_blind_eval_history_redacts_default_commit(tmp_path: Path) -> None:
    from scripts.ci import bootstrap_hybrid_blind_eval_history as mod

    summary_json = tmp_path / "summary.json"
    output_dir = tmp_path / "eval_history"

    _write_json(
        summary_json,
        {
            "sample_size": 10,
            "weak_labels": {
                "covered_rate": 1.0,
                "label_counts": {"人孔": 10},
                "accuracy": {
                    "hybrid_label": {"accuracy": 0.8},
                    "graph2d_label": {"accuracy": 0.4},
                },
                "by_true_label_accuracy": {
                    "hybrid_label": {"人孔": {"evaluated": 10, "correct": 8, "accuracy": 0.8}},
                    "graph2d_label": {"人孔": {"evaluated": 10, "correct": 4, "accuracy": 0.4}},
                },
            },
        },
    )

    rc = mod.main(
        [
            "--summary-json",
            str(summary_json),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
        ]
    )
    assert rc == 0
    files = sorted(output_dir.glob("*_hybrid_blind.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["commit"] == "[redacted]"
