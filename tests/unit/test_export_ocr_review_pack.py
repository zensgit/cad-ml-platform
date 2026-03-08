from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_ocr_review_pack_filters_review_candidates(tmp_path: Path) -> None:
    from scripts.export_ocr_review_pack import main

    input_json = tmp_path / "ocr_batch.json"
    output_csv = tmp_path / "ocr_review_pack.csv"
    output_json = tmp_path / "ocr_review_pack.json"
    input_json.write_text(
        json.dumps(
            [
                {
                    "provider": "paddle",
                    "success": True,
                    "confidence": 0.61,
                    "title_block": {
                        "drawing_number": "DWG-001",
                        "part_name": "Bracket",
                        "material": "Steel",
                    },
                    "identifiers": [{"identifier_type": "drawing_number", "value": "DWG-001"}],
                    "field_coverage": {
                        "recognized_count": 3,
                        "total_fields": 10,
                        "coverage_ratio": 0.3,
                        "recognized_keys": ["drawing_number", "part_name", "material"],
                        "missing_keys": ["revision", "company"],
                    },
                    "engineering_signals": {
                        "dimension_count": 0,
                        "symbol_count": 1,
                        "symbol_types": ["surface_roughness"],
                        "gdt_symbol_types": [],
                        "has_surface_finish": True,
                        "has_gdt": False,
                        "process_requirement_counts": {
                            "heat_treatments": 0,
                            "surface_treatments": 0,
                            "welding": 0,
                            "general_notes": 1,
                        },
                        "materials_detected": ["Steel"],
                        "standards_candidates": ["GB/T1804-M"],
                    },
                    "review_hints": {
                        "critical_fields": [
                            "drawing_number",
                            "part_name",
                            "revision",
                            "material",
                        ],
                        "present_critical_fields": [
                            "drawing_number",
                            "part_name",
                            "material",
                        ],
                        "missing_critical_fields": ["revision"],
                        "has_identifiers": True,
                        "has_dimensions": False,
                        "has_symbols": True,
                        "has_process_requirements": True,
                        "has_standards_candidates": True,
                        "review_recommended": True,
                        "review_reasons": ["missing_critical_fields", "no_dimensions"],
                        "primary_gap": "missing_critical_fields",
                        "review_priority": "high",
                        "automation_ready": False,
                        "recommended_actions": [
                            "fill_critical_title_block_fields",
                            "verify_dimensions",
                        ],
                        "readiness_score": 0.44,
                        "readiness_band": "low",
                    },
                },
                {
                    "provider": "paddle",
                    "success": True,
                    "confidence": 0.93,
                    "fields": [{"key": "drawing_number", "value": "DWG-002"}],
                    "field_confidence": {"drawing_number": 0.93},
                    "title_block": {
                        "drawing_number": "DWG-002",
                        "part_name": "Plate",
                        "revision": "A",
                        "material": "Aluminum",
                    },
                    "identifiers": [{"identifier_type": "drawing_number", "value": "DWG-002"}],
                    "field_coverage": {
                        "recognized_count": 4,
                        "total_fields": 10,
                        "coverage_ratio": 0.4,
                        "recognized_keys": [
                            "drawing_number",
                            "part_name",
                            "revision",
                            "material",
                        ],
                        "missing_keys": ["company"],
                    },
                    "engineering_signals": {
                        "dimension_count": 1,
                        "symbol_count": 2,
                        "symbol_types": ["surface_roughness", "position"],
                        "gdt_symbol_types": ["position"],
                        "has_surface_finish": True,
                        "has_gdt": True,
                        "process_requirement_counts": {
                            "heat_treatments": 0,
                            "surface_treatments": 1,
                            "welding": 0,
                            "general_notes": 1,
                        },
                        "materials_detected": ["Aluminum"],
                        "standards_candidates": ["GB/T13912"],
                    },
                    "review_hints": {
                        "critical_fields": [
                            "drawing_number",
                            "part_name",
                            "revision",
                            "material",
                        ],
                        "present_critical_fields": [
                            "drawing_number",
                            "part_name",
                            "revision",
                            "material",
                        ],
                        "missing_critical_fields": [],
                        "has_identifiers": True,
                        "has_dimensions": True,
                        "has_symbols": True,
                        "has_process_requirements": True,
                        "has_standards_candidates": True,
                        "review_recommended": False,
                        "review_reasons": [],
                        "primary_gap": "ready",
                        "review_priority": "low",
                        "automation_ready": True,
                        "recommended_actions": [],
                        "readiness_score": 0.88,
                        "readiness_band": "high",
                    },
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(input_json),
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ]
    )
    assert exit_code == 0

    rows = _read_csv_rows(output_csv)
    assert len(rows) == 1
    assert rows[0]["document_ref"] == "DWG-001"
    assert rows[0]["surface"] == "ocr"
    assert rows[0]["review_priority"] == "high"
    assert rows[0]["primary_gap"] == "missing_critical_fields"
    assert rows[0]["review_reasons"] == "missing_critical_fields;no_dimensions"
    assert rows[0]["recommended_actions"] == (
        "fill_critical_title_block_fields;verify_dimensions"
    )
    assert rows[0]["missing_critical_fields"] == "revision"
    assert rows[0]["standards_candidates"] == "GB/T1804-M"
    assert float(rows[0]["review_priority_score"]) > 300.0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["total_input_records"] == 2
    assert payload["summary"]["normalized_records"] == 2
    assert payload["summary"]["exported_records"] == 1
    assert payload["summary"]["review_candidate_count"] == 1
    assert payload["summary"]["surface_counts"] == [{"name": "ocr", "count": 1}]
    assert payload["summary"]["review_priority_counts"] == [{"name": "high", "count": 1}]
    assert payload["summary"]["top_missing_critical_fields"] == [
        {"name": "revision", "count": 1}
    ]
    assert payload["records"][0]["review_hints"]["review_priority"] == "high"
    assert payload["records"][0]["engineering_signals"]["has_surface_finish"] is True


def test_export_ocr_review_pack_include_ready_and_directory_input(tmp_path: Path) -> None:
    from scripts.export_ocr_review_pack import main

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "drawing.json").write_text(
        json.dumps(
            {
                "provider": "auto",
                "success": True,
                "confidence": 0.95,
                "fields": [{"key": "drawing_number", "value": "DWG-900"}],
                "title_block": {
                    "drawing_number": "DWG-900",
                    "part_name": "Housing",
                    "revision": "B",
                    "material": "Cast Iron",
                },
                "identifiers": [{"identifier_type": "drawing_number", "value": "DWG-900"}],
                "field_coverage": {
                    "recognized_count": 4,
                    "total_fields": 10,
                    "coverage_ratio": 0.4,
                    "recognized_keys": [
                        "drawing_number",
                        "part_name",
                        "revision",
                        "material",
                    ],
                    "missing_keys": ["company"],
                },
                "engineering_signals": {
                    "dimension_count": 2,
                    "symbol_count": 2,
                    "symbol_types": ["surface_roughness", "position"],
                    "gdt_symbol_types": ["position"],
                    "has_surface_finish": True,
                    "has_gdt": True,
                    "process_requirement_counts": {
                        "heat_treatments": 0,
                        "surface_treatments": 1,
                        "welding": 0,
                        "general_notes": 1,
                    },
                    "materials_detected": ["Cast Iron"],
                    "standards_candidates": ["GB/T13912"],
                },
                "review_hints": {
                    "critical_fields": [
                        "drawing_number",
                        "part_name",
                        "revision",
                        "material",
                    ],
                    "present_critical_fields": [
                        "drawing_number",
                        "part_name",
                        "revision",
                        "material",
                    ],
                    "missing_critical_fields": [],
                    "has_identifiers": True,
                    "has_dimensions": True,
                    "has_symbols": True,
                    "has_process_requirements": True,
                    "has_standards_candidates": True,
                    "review_recommended": False,
                    "review_reasons": [],
                    "primary_gap": "ready",
                    "review_priority": "low",
                    "automation_ready": True,
                    "recommended_actions": [],
                    "readiness_score": 0.9,
                    "readiness_band": "high",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (input_dir / "ocr.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "paddle",
                        "success": True,
                        "confidence": 0.51,
                        "title_block": {
                            "drawing_number": "DWG-700",
                            "part_name": "Bracket",
                        },
                        "field_coverage": {
                            "recognized_count": 2,
                            "total_fields": 10,
                            "coverage_ratio": 0.2,
                            "recognized_keys": ["drawing_number", "part_name"],
                            "missing_keys": ["revision", "material", "company"],
                        },
                        "engineering_signals": {
                            "dimension_count": 0,
                            "symbol_count": 0,
                            "symbol_types": [],
                            "gdt_symbol_types": [],
                            "has_surface_finish": False,
                            "has_gdt": False,
                            "process_requirement_counts": {
                                "heat_treatments": 0,
                                "surface_treatments": 0,
                                "welding": 0,
                                "general_notes": 0,
                            },
                            "materials_detected": [],
                            "standards_candidates": [],
                        },
                        "review_hints": {
                            "critical_fields": [
                                "drawing_number",
                                "part_name",
                                "revision",
                                "material",
                            ],
                            "present_critical_fields": ["drawing_number", "part_name"],
                            "missing_critical_fields": ["revision", "material"],
                            "has_identifiers": False,
                            "has_dimensions": False,
                            "has_symbols": False,
                            "has_process_requirements": False,
                            "has_standards_candidates": False,
                            "review_recommended": True,
                            "review_reasons": [
                                "missing_critical_fields",
                                "no_identifiers",
                                "no_dimensions",
                                "limited_engineering_context",
                            ],
                            "primary_gap": "missing_critical_fields",
                            "review_priority": "high",
                            "automation_ready": False,
                            "recommended_actions": [
                                "fill_critical_title_block_fields",
                                "verify_or_extract_identifiers",
                                "verify_dimensions",
                            ],
                            "readiness_score": 0.19,
                            "readiness_band": "low",
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"provider": "paddle", "success": True}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    output_csv = tmp_path / "ocr_review_pack.csv"
    output_json = tmp_path / "ocr_review_pack.json"
    exit_code = main(
        [
            "--input",
            str(input_dir),
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
            "--include-ready",
            "--top-k",
            "2",
        ]
    )
    assert exit_code == 0

    rows = _read_csv_rows(output_csv)
    assert len(rows) == 2
    assert rows[0]["document_ref"] == "DWG-700"
    assert rows[0]["review_priority"] == "high"
    assert rows[0]["surface"] == "ocr"
    assert rows[1]["document_ref"] == "DWG-900"
    assert rows[1]["surface"] == "drawing"
    assert rows[1]["automation_ready"] == "True"

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["total_input_records"] == 3
    assert payload["summary"]["normalized_records"] == 2
    assert payload["summary"]["skipped_records"] == 1
    assert payload["summary"]["exported_records"] == 2
    assert payload["summary"]["include_ready"] is True
    assert payload["summary"]["top_k"] == 2
    assert payload["summary"]["surface_counts"] == [
        {"name": "ocr", "count": 1},
        {"name": "drawing", "count": 1},
    ]
    assert payload["summary"]["review_priority_counts"] == [
        {"name": "high", "count": 1},
        {"name": "low", "count": 1},
    ]
    assert payload["summary"]["top_recommended_actions"][0] == {
        "name": "fill_critical_title_block_fields",
        "count": 1,
    }
    assert payload["summary"]["sample_records"][0]["document_ref"] == "DWG-700"
