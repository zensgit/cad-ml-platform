from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import export_benchmark_knowledge_reference_inventory as module

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_reference_inventory.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_build_summary_exposes_inventory_fields() -> None:
    payload = module.build_summary(
        title="Benchmark Knowledge Reference Inventory",
        snapshot={
            "tolerance": {
                "source_tables": {
                    "iso_limits": 12,
                    "fit_classes": 8,
                }
            },
            "standards": {
                "source_tables": {
                    "thread_series": 0,
                    "fasteners": 5,
                }
            },
            "gdt": {
                "source_tables": {
                    "datum_symbols": 0,
                }
            },
        },
        artifact_paths={"snapshot_json": "snapshot.json"},
    )

    component = payload["knowledge_reference_inventory"]
    assert component["status"] == "knowledge_reference_inventory_blocked"
    assert component["ready_domain_count"] == 1
    assert component["partial_domain_count"] == 1
    assert component["blocked_domain_count"] == 1
    assert component["priority_domains"] == ["standards", "gdt"]
    assert component["total_reference_items"] == 25
    assert payload["recommendations"] == [
        "Backfill standards reference tables: thread_series",
        "Backfill gdt reference tables: datum_symbols",
    ]
    markdown = module.render_knowledge_reference_inventory_markdown(
        payload,
        payload["title"],
    )
    assert "## Domains" in markdown
    assert "Backfill standards reference tables: thread_series" in markdown


def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    snapshot = _write_json(
        tmp_path / "snapshot.json",
        {
            "tolerance": {
                "source_tables": {
                    "iso_limits": 12,
                    "fit_classes": 8,
                }
            },
            "standards": {
                "source_tables": {
                    "thread_series": 7,
                    "fasteners": 5,
                }
            },
            "gdt": {
                "source_tables": {
                    "datum_symbols": 6,
                }
            },
        },
    )
    output_json = tmp_path / "knowledge_reference_inventory.json"
    output_md = tmp_path / "knowledge_reference_inventory.md"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--snapshot-json",
            str(snapshot),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    component = payload["knowledge_reference_inventory"]
    assert component["status"] == "knowledge_reference_inventory_ready"
    assert component["total_reference_items"] == 38
    assert payload["artifacts"]["snapshot_json"] == str(snapshot)
    assert payload["recommendations"] == [
        "Promote standards/tolerance/GD&T reference inventory into benchmark "
        "control-plane reviews."
    ]
    markdown = output_md.read_text(encoding="utf-8")
    assert "# Benchmark Knowledge Reference Inventory" in markdown
    assert "## Domains" in markdown
    assert "## Recommendations" in markdown
