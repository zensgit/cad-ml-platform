import csv
import json
from pathlib import Path

from scripts.build_manufacturing_review_manifest import (
    apply_reviewer_template_rows,
    build_review_assignment_markdown,
    build_review_batch_rows,
    build_review_batch_template_rows,
    build_review_context_rows,
    build_review_gap_rows,
    build_review_handoff_markdown,
    build_review_manifest_merge_audit_rows,
    build_review_progress_markdown,
    build_review_rows,
    build_reviewer_template_rows,
    build_reviewer_template_apply_audit_rows,
    build_reviewer_template_preflight_gap_rows,
    build_reviewer_template_preflight_markdown,
    main,
    merge_approved_review_rows,
    validate_review_manifest_rows,
    validate_reviewer_template_rows,
)


def test_build_review_rows_extracts_suggestions_without_claiming_review() -> None:
    evidence = [
        {
            "source": "dfm",
            "kind": "manufacturability_check",
            "status": "manufacturable",
            "details": {"mode": "rule"},
        },
        {
            "source": "manufacturing_cost",
            "label": "CNY",
            "details": {"cost_range": {"low": 90.0}},
        },
        {
            "source": "manufacturing_decision",
            "status": "manufacturable",
            "details": {"risks_count": 0},
        },
    ]

    rows = build_review_rows(
        [
            {
                "file_name": "shaft.dxf",
                "true_label": "轴类",
                "relative_path": "release/shaft.dxf",
                "manufacturing_evidence": json.dumps(evidence, ensure_ascii=False),
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["file_name"] == "shaft.dxf"
    assert row["label_cn"] == "轴类"
    assert row["relative_path"] == "release/shaft.dxf"
    assert row["reviewed_manufacturing_evidence_sources"] == ""
    assert row["reviewed_manufacturing_evidence_payload_json"] == ""
    assert row["review_status"] == "needs_human_review"
    assert row["reviewer"] == ""
    assert row["reviewed_at"] == ""
    assert row["suggested_manufacturing_evidence_sources"] == (
        "dfm;manufacturing_cost;manufacturing_decision"
    )

    payloads = json.loads(row["suggested_manufacturing_evidence_payload_json"])
    assert payloads["dfm"] == {
        "details.mode": "rule",
        "kind": "manufacturability_check",
        "status": "manufacturable",
    }
    assert payloads["manufacturing_cost"]["details.cost_range.low"] == "90.0"
    assert payloads["manufacturing_decision"]["details.risks_count"] == "0"


def test_validate_review_manifest_counts_source_payload_and_detail_labels() -> None:
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "reviewed_manufacturing_evidence_sources": "dfm;process",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {
                    "dfm": {
                        "status": "manufacturable",
                        "details": {"mode": "rule"},
                    },
                    "process": {
                        "label": "milling",
                        "details.rule_version": "process.v1",
                    },
                }
            ),
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "expected_manufacturing_evidence_sources": "dfm;cost;decision",
            "expected_dfm_status": "manufacturable",
            "expected_dfm_detail_mode": "rule",
            "expected_cost_label": "CNY",
            "expected_cost_detail_cost_range__low": "90.0",
            "expected_decision_status": "manufacturable",
            "expected_decision_detail_risks_count": "0",
        },
    ]

    summary = validate_review_manifest_rows(rows, min_reviewed_samples=2)

    assert summary["status"] == "release_label_ready"
    assert summary["source_reviewed_sample_count"] == 2
    assert summary["payload_reviewed_sample_count"] == 2
    assert summary["payload_detail_reviewed_sample_count"] == 2
    assert summary["payload_expected_field_total"] == 10
    assert summary["payload_detail_expected_field_total"] == 5
    assert summary["blocking_reasons"] == []


def test_validate_review_manifest_requires_approved_status_when_column_exists() -> None:
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        }
    ]

    summary = validate_review_manifest_rows(rows, min_reviewed_samples=1)

    assert summary["status"] == "blocked"
    assert summary["unapproved_review_sample_count"] == 1
    assert summary["approved_review_sample_count"] == 0
    assert summary["source_reviewed_sample_count"] == 0
    assert "source_reviewed_sample_count_below_minimum" in summary["blocking_reasons"]


def test_validate_review_manifest_can_require_reviewer_metadata() -> None:
    row = {
        "file_name": "shaft.dxf",
        "label_cn": "轴类",
        "review_status": "approved",
        "reviewed_manufacturing_evidence_sources": "dfm",
        "reviewed_manufacturing_evidence_payload_json": json.dumps(
            {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
        ),
    }

    blocked = validate_review_manifest_rows(
        [row],
        min_reviewed_samples=1,
        require_reviewer_metadata=True,
    )
    ready = validate_review_manifest_rows(
        [{**row, "reviewer": "manufacturing-reviewer", "reviewed_at": "2026-05-13"}],
        min_reviewed_samples=1,
        require_reviewer_metadata=True,
    )

    assert blocked["status"] == "blocked"
    assert blocked["reviewer_metadata_missing_sample_count"] == 1
    assert "reviewer_metadata_missing" in blocked["blocking_reasons"]
    assert ready["status"] == "release_label_ready"
    assert ready["approved_review_sample_count"] == 1
    assert ready["reviewer_metadata_missing_sample_count"] == 0


def test_build_review_progress_markdown_lists_counts_and_gap_rows() -> None:
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-13",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
        },
    ]
    summary = validate_review_manifest_rows(
        rows,
        min_reviewed_samples=2,
        require_reviewer_metadata=True,
    )

    markdown = build_review_progress_markdown(
        rows,
        summary,
        require_reviewer_metadata=True,
    )

    assert "Source labels | 1 | 2 | 1" in markdown
    assert "Payload labels | 1 | 2 | 1" in markdown
    assert "Detail labels | 1 | 2 | 1" in markdown
    assert "release/flange.dxf" in markdown
    assert "set approved review_status" in markdown
    assert "fill reviewer and reviewed_at" in markdown
    assert "add details.* payload labels" in markdown


def test_build_review_gap_rows_lists_actionable_review_backlog() -> None:
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-13",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "suggested_manufacturing_evidence_sources": "dfm;process",
            "suggested_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details.mode": "rule"}}
            ),
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
            "review_notes": "needs detail labels",
        },
    ]

    gap_rows = build_review_gap_rows(rows, require_reviewer_metadata=True)

    assert len(gap_rows) == 1
    gap = gap_rows[0]
    assert gap["row_id"] == "release/flange.dxf"
    assert gap["label_cn"] == "法兰"
    assert "set approved review_status" in gap["gap_reasons"]
    assert "fill reviewer and reviewed_at" in gap["gap_reasons"]
    assert "add details.* payload labels" in gap["gap_reasons"]
    assert gap["source_ready"] == "false"
    assert gap["payload_ready"] == "false"
    assert gap["detail_ready"] == "false"
    assert gap["suggested_manufacturing_evidence_sources"] == "dfm;process"
    assert gap["review_notes"] == "needs detail labels"


def test_build_review_context_rows_summarizes_suggested_and_actual_evidence() -> None:
    actual_evidence = [
        {
            "source": "dfm",
            "kind": "manufacturability_check",
            "status": "manufacturable",
            "details": {"mode": "rule", "rule_version": "dfm.v1"},
        },
        {
            "source": "manufacturing_process",
            "label": "milling",
            "details": {"operation": "roughing"},
        },
    ]
    rows = [
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "suggested_manufacturing_evidence_sources": "dfm;process",
            "suggested_manufacturing_evidence_payload_json": json.dumps(
                {
                    "dfm": {
                        "status": "manufacturable",
                        "details": {"mode": "rule"},
                    },
                    "process": {"label": "milling", "details.operation": "roughing"},
                }
            ),
            "actual_manufacturing_evidence": json.dumps(actual_evidence),
            "review_notes": "needs detail labels",
        }
    ]

    context_rows = build_review_context_rows(rows, require_reviewer_metadata=True)

    assert len(context_rows) == 1
    context = context_rows[0]
    assert context["row_id"] == "release/flange.dxf"
    assert context["gap_reason_count"] == "2"
    assert context["source_ready"] == "false"
    assert context["suggested_manufacturing_evidence_sources"] == "dfm;manufacturing_process"
    assert context["suggested_source_count"] == "2"
    assert context["suggested_payload_field_count"] == "4"
    assert context["suggested_detail_field_count"] == "2"
    assert "dfm.details.mode" in context["suggested_payload_fields"]
    assert "manufacturing_process.details.operation" in context[
        "suggested_payload_fields"
    ]
    assert context["actual_evidence_item_count"] == "2"
    assert context["actual_evidence_sources"] == "dfm;manufacturing_process"
    assert "source=dfm" in context["actual_evidence_summary"]
    assert "status=manufacturable" in context["actual_evidence_summary"]
    assert "label=milling" in context["actual_evidence_summary"]
    assert "details.rule_version" in context["actual_evidence_detail_keys"]
    assert context["review_notes"] == "needs detail labels"


def test_build_review_batch_rows_balances_label_review_work() -> None:
    rows = [
        {
            "file_name": "flange-a.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange-a.dxf",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
            "actual_manufacturing_evidence": json.dumps(
                [{"source": "dfm", "status": "manufacturable"}]
            ),
        },
        {
            "file_name": "flange-b.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange-b.dxf",
            "review_status": "needs_human_review",
        },
        {
            "file_name": "plate.dxf",
            "label_cn": "板类",
            "relative_path": "release/plate.dxf",
            "review_status": "needs_human_review",
            "suggested_manufacturing_evidence_sources": "dfm;process",
        },
    ]

    batch_rows = build_review_batch_rows(
        rows,
        max_rows_per_label=1,
        require_reviewer_metadata=True,
    )

    assert len(batch_rows) == 2
    assert [row["review_batch"] for row in batch_rows] == [
        "batch_001",
        "batch_002",
    ]
    assert batch_rows[0]["label_cn"] == "法兰"
    assert batch_rows[0]["label_gap_row_count"] == "2"
    assert batch_rows[0]["row_id"] == "release/flange-a.dxf"
    assert batch_rows[0]["detail_gap"] == "true"
    assert batch_rows[0]["approval_gap"] == "true"
    assert batch_rows[0]["metadata_gap"] == "true"
    assert batch_rows[0]["actual_evidence_sources"] == "dfm"
    assert batch_rows[1]["label_cn"] == "板类"
    assert batch_rows[1]["label_gap_row_count"] == "1"
    assert batch_rows[1]["suggested_manufacturing_evidence_sources"] == (
        "dfm;manufacturing_process"
    )


def test_build_review_batch_template_rows_limits_editable_template() -> None:
    rows = [
        {
            "file_name": "flange-a.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange-a.dxf",
            "review_status": "needs_human_review",
            "suggested_manufacturing_evidence_sources": "dfm;process",
            "suggested_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details.mode": "rule"}}
            ),
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
            "review_notes": "needs detail labels",
        },
        {
            "file_name": "flange-b.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange-b.dxf",
            "review_status": "needs_human_review",
        },
        {
            "file_name": "plate.dxf",
            "label_cn": "板类",
            "relative_path": "release/plate.dxf",
            "review_status": "needs_human_review",
        },
    ]

    template_rows = build_review_batch_template_rows(
        rows,
        max_rows_per_label=1,
        require_reviewer_metadata=True,
    )

    assert len(template_rows) == 2
    assert template_rows[0]["review_batch"] == "batch_001"
    assert template_rows[0]["batch_rank"] == "1"
    assert template_rows[0]["label_gap_row_count"] == "2"
    assert template_rows[0]["row_id"] == "release/flange-a.dxf"
    assert template_rows[0]["reviewed_manufacturing_evidence_sources"] == "dfm"
    assert "details.mode" in template_rows[0][
        "suggested_manufacturing_evidence_payload_json"
    ]
    assert "add details.* payload labels" in template_rows[0]["gap_reasons"]
    assert template_rows[1]["review_batch"] == "batch_002"
    assert template_rows[1]["row_id"] == "release/plate.dxf"


def test_build_review_assignment_markdown_groups_gaps_by_label() -> None:
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-13",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "suggested_manufacturing_evidence_sources": "dfm;process",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
        },
        {
            "file_name": "plate.dxf",
            "label_cn": "板类",
            "relative_path": "release/plate.dxf",
            "review_status": "needs_human_review",
        },
    ]
    summary = validate_review_manifest_rows(
        rows,
        min_reviewed_samples=2,
        require_reviewer_metadata=True,
    )

    markdown = build_review_assignment_markdown(
        rows,
        summary,
        max_rows_per_label=1,
        require_reviewer_metadata=True,
    )

    assert "Manufacturing Review Assignment Plan" in markdown
    assert "| 法兰 | 1 | 0 | 0 | 1 | 1 | 1 |" in markdown
    assert "| 板类 | 1 | 1 | 1 | 0 | 0 | 0 |" in markdown
    assert "release/flange.dxf" in markdown
    assert "dfm;process" in markdown
    assert "release/plate.dxf" in markdown


def test_build_reviewer_template_rows_preserves_editable_review_fields() -> None:
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "suggested_manufacturing_evidence_sources": "dfm;process",
            "suggested_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details.mode": "rule"}}
            ),
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
            "review_notes": "needs detail labels",
        },
    ]

    template_rows = build_reviewer_template_rows(rows)

    assert len(template_rows) == 1
    template = template_rows[0]
    assert template["row_id"] == "release/flange.dxf"
    assert template["review_status"] == "needs_human_review"
    assert template["reviewed_manufacturing_evidence_sources"] == "dfm"
    assert "details.mode" in template["suggested_manufacturing_evidence_payload_json"]
    assert "add details.* payload labels" in template["gap_reasons"]


def test_merge_approved_review_rows_can_require_reviewer_metadata() -> None:
    base_rows = [{"file_name": "shaft.dxf", "label_cn": "轴类"}]
    review_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        }
    ]

    blocked_rows, blocked = merge_approved_review_rows(
        base_rows,
        review_rows,
        require_reviewer_metadata=True,
    )
    ready_rows, ready = merge_approved_review_rows(
        base_rows,
        [
            {
                **review_rows[0],
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-13",
            }
        ],
        require_reviewer_metadata=True,
    )

    assert blocked["status"] == "blocked"
    assert blocked["approved_review_row_count"] == 1
    assert blocked["skipped_missing_metadata_row_count"] == 1
    assert blocked["merged_row_count"] == 0
    assert "no_approved_review_rows_merged" in blocked["blocking_reasons"]
    assert "reviewed_manufacturing_evidence_sources" not in blocked_rows[0]
    assert ready["status"] == "merged"
    assert ready["merged_row_count"] == 1
    assert ready["skipped_missing_metadata_row_count"] == 0
    assert ready_rows[0]["reviewed_manufacturing_evidence_sources"] == "dfm"


def test_merge_approved_review_rows_blocks_duplicate_base_manifest() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {
            "file_name": "shaft-a.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
        },
        {
            "file_name": "shaft-b.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
        },
    ]
    review_rows = [
        {
            "file_name": "shaft-a.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    merged_rows, summary = merge_approved_review_rows(base_rows, review_rows)

    assert summary["status"] == "blocked"
    assert summary["merged_row_count"] == 0
    assert summary["approved_review_row_count"] == 1
    assert summary["base_manifest_duplicate_identity_count"] == 1
    assert summary["base_manifest_duplicate_identifiers"] == ["release/shaft.dxf"]
    assert "base_manifest_duplicate_rows" in summary["blocking_reasons"]
    assert merged_rows == base_rows


def test_merge_approved_review_rows_blocks_ambiguous_file_name_fallback() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/a/shaft.dxf",
        },
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/b/shaft.dxf",
        },
    ]
    review_row = {
        "file_name": "shaft.dxf",
        "label_cn": "轴类",
        "review_status": "approved",
        "reviewer": "manufacturing-reviewer",
        "reviewed_at": "2026-05-14",
        "reviewed_manufacturing_evidence_sources": "dfm",
        "reviewed_manufacturing_evidence_payload_json": payload_json,
    }

    ambiguous_rows, ambiguous = merge_approved_review_rows(base_rows, [review_row])
    precise_rows, precise = merge_approved_review_rows(
        base_rows,
        [{**review_row, "relative_path": "release/a/shaft.dxf"}],
    )
    mixed_rows, mixed = merge_approved_review_rows(
        base_rows,
        [
            review_row,
            {**review_row, "relative_path": "release/a/shaft.dxf"},
        ],
    )

    assert ambiguous["status"] == "blocked"
    assert ambiguous["merged_row_count"] == 0
    assert ambiguous["ambiguous_file_name_match_row_count"] == 1
    assert "ambiguous_file_name_match_rows" in ambiguous["blocking_reasons"]
    assert ambiguous_rows == base_rows
    assert precise["status"] == "merged"
    assert precise["ambiguous_file_name_match_row_count"] == 0
    assert precise_rows[0]["reviewed_manufacturing_evidence_sources"] == "dfm"
    assert "reviewed_manufacturing_evidence_sources" not in precise_rows[1]
    assert mixed["status"] == "blocked"
    assert mixed["merged_row_count"] == 1
    assert mixed["ambiguous_file_name_match_row_count"] == 1
    assert "ambiguous_file_name_match_rows" in mixed["blocking_reasons"]
    assert mixed_rows[0]["reviewed_manufacturing_evidence_sources"] == "dfm"
    assert "reviewed_manufacturing_evidence_sources" not in mixed_rows[1]


def test_build_review_manifest_merge_audit_rows_lists_merge_outcomes() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
        }
    ]
    review_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
        {
            "file_name": "nut.dxf",
            "label_cn": "螺母",
            "relative_path": "release/nut.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
    ]

    audit_rows = build_review_manifest_merge_audit_rows(
        base_rows,
        review_rows,
        require_reviewer_metadata=True,
    )

    assert [row["merge_status"] for row in audit_rows] == [
        "merged",
        "unmatched_review_row",
        "skipped_unapproved_review",
    ]
    assert audit_rows[0]["matched_base_row"] == "true"
    assert audit_rows[0]["detail_ready"] == "true"
    assert audit_rows[1]["merge_reasons"] == "match row_id to base benchmark manifest"
    assert audit_rows[2]["merge_reasons"] == "set approved review_status"


def test_build_review_manifest_merge_audit_rows_blocks_duplicate_base() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {"file_name": "shaft-a.dxf", "relative_path": "release/shaft.dxf"},
        {"file_name": "shaft-b.dxf", "relative_path": "release/shaft.dxf"},
    ]
    review_rows = [
        {
            "file_name": "shaft-a.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    audit_rows = build_review_manifest_merge_audit_rows(
        base_rows,
        review_rows,
    )

    assert audit_rows[0]["merge_status"] == "blocked_duplicate_base_manifest"
    assert audit_rows[0]["matched_base_row"] == "false"
    assert audit_rows[0]["merge_reasons"] == "deduplicate base benchmark manifest"


def test_build_review_manifest_merge_audit_rows_reports_ambiguous_file_name() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {"file_name": "shaft.dxf", "relative_path": "release/a/shaft.dxf"},
        {"file_name": "shaft.dxf", "relative_path": "release/b/shaft.dxf"},
    ]
    review_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    audit_rows = build_review_manifest_merge_audit_rows(
        base_rows,
        review_rows,
    )

    assert audit_rows[0]["merge_status"] == "ambiguous_file_name_match"
    assert audit_rows[0]["matched_base_row"] == "false"
    assert audit_rows[0]["merge_reasons"] == (
        "add relative_path to disambiguate duplicate file_name"
    )


def test_apply_reviewer_template_rows_updates_full_review_manifest() -> None:
    manifest_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-13",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
        },
    ]
    template_rows = [
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm;process",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {
                    "dfm": {
                        "status": "manufacturable",
                        "details": {"mode": "rule"},
                    }
                }
            ),
        },
        {
            "file_name": "plate.dxf",
            "label_cn": "板类",
            "relative_path": "release/plate.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        },
    ]

    applied_rows, summary = apply_reviewer_template_rows(
        manifest_rows,
        template_rows,
        require_reviewer_metadata=True,
    )

    assert summary["status"] == "applied"
    assert summary["applied_row_count"] == 1
    assert summary["unmatched_template_row_count"] == 1
    assert applied_rows[1]["review_status"] == "approved"
    assert applied_rows[1]["reviewed_at"] == "2026-05-14"
    assert applied_rows[1]["reviewed_manufacturing_evidence_sources"] == "dfm;process"


def test_apply_reviewer_template_rows_blocks_duplicate_base_manifest() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    manifest_rows = [
        {
            "file_name": "flange-a.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
        },
        {
            "file_name": "flange-b.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
        },
    ]
    template_rows = [
        {
            "file_name": "flange-a.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    applied_rows, summary = apply_reviewer_template_rows(
        manifest_rows,
        template_rows,
    )

    assert summary["status"] == "blocked"
    assert summary["applied_row_count"] == 0
    assert summary["approved_template_row_count"] == 1
    assert summary["base_manifest_duplicate_identity_count"] == 1
    assert summary["base_manifest_duplicate_identifiers"] == ["release/flange.dxf"]
    assert "base_manifest_duplicate_rows" in summary["blocking_reasons"]
    assert applied_rows == manifest_rows


def test_apply_reviewer_template_rows_blocks_ambiguous_file_name_fallback() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    manifest_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/a/shaft.dxf",
            "review_status": "needs_human_review",
        },
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/b/shaft.dxf",
            "review_status": "needs_human_review",
        },
    ]
    template_row = {
        "file_name": "shaft.dxf",
        "label_cn": "轴类",
        "review_status": "approved",
        "reviewer": "manufacturing-reviewer",
        "reviewed_at": "2026-05-14",
        "reviewed_manufacturing_evidence_sources": "dfm",
        "reviewed_manufacturing_evidence_payload_json": payload_json,
    }

    ambiguous_rows, ambiguous = apply_reviewer_template_rows(
        manifest_rows,
        [template_row],
    )
    precise_rows, precise = apply_reviewer_template_rows(
        manifest_rows,
        [{**template_row, "relative_path": "release/a/shaft.dxf"}],
    )
    typo_path_rows, typo_path = apply_reviewer_template_rows(
        manifest_rows,
        [{**template_row, "relative_path": "release/missing/shaft.dxf"}],
    )
    mixed_rows, mixed = apply_reviewer_template_rows(
        manifest_rows,
        [
            template_row,
            {**template_row, "relative_path": "release/a/shaft.dxf"},
        ],
    )

    assert ambiguous["status"] == "blocked"
    assert ambiguous["applied_row_count"] == 0
    assert ambiguous["ambiguous_file_name_match_row_count"] == 1
    assert "ambiguous_file_name_match_rows" in ambiguous["blocking_reasons"]
    assert ambiguous_rows == manifest_rows

    assert precise["status"] == "applied"
    assert precise["applied_row_count"] == 1
    assert precise["ambiguous_file_name_match_row_count"] == 0
    assert precise_rows[0]["review_status"] == "approved"
    assert precise_rows[1]["review_status"] == "needs_human_review"

    assert typo_path["status"] == "blocked"
    assert typo_path["applied_row_count"] == 0
    assert typo_path["ambiguous_file_name_match_row_count"] == 1
    assert typo_path["unmatched_template_row_count"] == 0
    assert typo_path_rows == manifest_rows

    assert mixed["status"] == "blocked"
    assert mixed["applied_row_count"] == 1
    assert mixed["ambiguous_file_name_match_row_count"] == 1
    assert "ambiguous_file_name_match_rows" in mixed["blocking_reasons"]
    assert mixed_rows[0]["review_status"] == "approved"
    assert mixed_rows[1]["review_status"] == "needs_human_review"


def test_build_reviewer_template_apply_audit_rows_lists_apply_outcomes() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    manifest_rows = [
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
        }
    ]
    template_rows = [
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
        {
            "file_name": "plate.dxf",
            "label_cn": "板类",
            "relative_path": "release/plate.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
    ]

    audit_rows = build_reviewer_template_apply_audit_rows(
        manifest_rows,
        template_rows,
        require_reviewer_metadata=True,
    )

    assert [row["apply_status"] for row in audit_rows] == [
        "applied",
        "unmatched_template_row",
        "skipped_unapproved_template",
    ]
    assert audit_rows[0]["matched_manifest_row"] == "true"
    assert audit_rows[0]["detail_ready"] == "true"
    assert audit_rows[1]["apply_reasons"] == "match row_id to review manifest"
    assert audit_rows[2]["apply_reasons"] == "set approved review_status"


def test_build_reviewer_template_apply_audit_rows_blocks_duplicate_base() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    manifest_rows = [
        {"file_name": "flange-a.dxf", "relative_path": "release/flange.dxf"},
        {"file_name": "flange-b.dxf", "relative_path": "release/flange.dxf"},
    ]
    template_rows = [
        {
            "file_name": "flange-a.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    audit_rows = build_reviewer_template_apply_audit_rows(
        manifest_rows,
        template_rows,
    )

    assert audit_rows[0]["apply_status"] == "blocked_duplicate_base_manifest"
    assert audit_rows[0]["matched_manifest_row"] == "false"
    assert audit_rows[0]["apply_reasons"] == "deduplicate base review manifest"


def test_build_reviewer_template_apply_audit_rows_reports_ambiguous_file_name() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    manifest_rows = [
        {"file_name": "shaft.dxf", "relative_path": "release/a/shaft.dxf"},
        {"file_name": "shaft.dxf", "relative_path": "release/b/shaft.dxf"},
    ]
    template_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    audit_rows = build_reviewer_template_apply_audit_rows(
        manifest_rows,
        template_rows,
    )

    assert audit_rows[0]["apply_status"] == "ambiguous_file_name_match"
    assert audit_rows[0]["matched_manifest_row"] == "false"
    assert audit_rows[0]["apply_reasons"] == (
        "add relative_path to disambiguate duplicate file_name"
    )


def test_validate_reviewer_template_rows_reports_preflight_blockers() -> None:
    ready_payload = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": ready_payload,
        },
        {
            "file_name": "shaft-duplicate.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": ready_payload,
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
        },
        {
            "file_name": "plate.dxf",
            "label_cn": "板类",
            "relative_path": "release/plate.dxf",
            "review_status": "approved",
        },
    ]

    summary = validate_reviewer_template_rows(
        rows,
        min_ready_rows=2,
        require_reviewer_metadata=True,
    )

    assert summary["status"] == "blocked"
    assert summary["ready_template_row_count"] == 1
    assert summary["base_manifest_match_required"] is False
    assert summary["base_manifest_duplicate_identity_count"] == 0
    assert summary["unmatched_template_row_count"] == 0
    assert summary["duplicate_template_row_count"] == 1
    assert summary["unapproved_template_row_count"] == 1
    assert summary["reviewer_metadata_missing_row_count"] == 1
    assert summary["no_review_content_row_count"] == 1
    assert summary["payload_detail_missing_row_count"] == 1
    assert "ready_template_row_count_below_minimum" in summary["blocking_reasons"]
    assert "duplicate_template_rows" in summary["blocking_reasons"]


def test_validate_reviewer_template_rows_blocks_unmatched_manifest_rows() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
        }
    ]
    template_rows = [
        {
            "file_name": "missing.dxf",
            "label_cn": "轴类",
            "relative_path": "release/missing.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    summary = validate_reviewer_template_rows(
        template_rows,
        min_ready_rows=1,
        base_rows=base_rows,
    )

    assert summary["status"] == "blocked"
    assert summary["base_manifest_match_required"] is True
    assert summary["base_manifest_row_count"] == 1
    assert summary["ready_template_row_count"] == 0
    assert summary["unmatched_template_row_count"] == 1
    assert "unmatched_template_rows" in summary["blocking_reasons"]


def test_validate_reviewer_template_rows_reports_empty_base_manifest() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    template_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    summary = validate_reviewer_template_rows(
        template_rows,
        min_ready_rows=1,
        base_rows=[],
    )

    assert summary["status"] == "blocked"
    assert summary["base_manifest_match_required"] is True
    assert summary["base_manifest_row_count"] == 0
    assert summary["ready_template_row_count"] == 0
    assert summary["unmatched_template_row_count"] == 1
    assert "base_manifest_empty" in summary["blocking_reasons"]
    assert "unmatched_template_rows" in summary["blocking_reasons"]


def test_validate_reviewer_template_rows_reports_duplicate_base_manifest_rows() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {
            "file_name": "shaft-a.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
        },
        {
            "file_name": "shaft-b.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
        },
    ]
    template_rows = [
        {
            "file_name": "shaft-a.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    summary = validate_reviewer_template_rows(
        template_rows,
        min_ready_rows=1,
        base_rows=base_rows,
    )

    assert summary["status"] == "blocked"
    assert summary["base_manifest_match_required"] is True
    assert summary["base_manifest_row_count"] == 2
    assert summary["base_manifest_duplicate_identity_count"] == 1
    assert summary["base_manifest_duplicate_identifiers"] == ["release/shaft.dxf"]
    assert summary["ready_template_row_count"] == 1
    assert summary["unmatched_template_row_count"] == 0
    assert "base_manifest_duplicate_rows" in summary["blocking_reasons"]


def test_validate_reviewer_template_rows_blocks_ambiguous_file_name_match() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/a/shaft.dxf",
        },
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/b/shaft.dxf",
        },
    ]
    template_row = {
        "file_name": "shaft.dxf",
        "label_cn": "轴类",
        "review_status": "approved",
        "reviewed_manufacturing_evidence_sources": "dfm",
        "reviewed_manufacturing_evidence_payload_json": payload_json,
    }

    ambiguous = validate_reviewer_template_rows(
        [template_row],
        min_ready_rows=1,
        base_rows=base_rows,
    )
    precise = validate_reviewer_template_rows(
        [{**template_row, "relative_path": "release/a/shaft.dxf"}],
        min_ready_rows=1,
        base_rows=base_rows,
    )
    typo_path = validate_reviewer_template_rows(
        [{**template_row, "relative_path": "release/missing/shaft.dxf"}],
        min_ready_rows=1,
        base_rows=base_rows,
    )

    assert ambiguous["status"] == "blocked"
    assert ambiguous["ready_template_row_count"] == 0
    assert ambiguous["unmatched_template_row_count"] == 0
    assert ambiguous["ambiguous_file_name_match_row_count"] == 1
    assert "ambiguous_file_name_match_rows" in ambiguous["blocking_reasons"]

    assert precise["status"] == "ready"
    assert precise["ready_template_row_count"] == 1
    assert precise["ambiguous_file_name_match_row_count"] == 0

    assert typo_path["status"] == "blocked"
    assert typo_path["ready_template_row_count"] == 0
    assert typo_path["unmatched_template_row_count"] == 0
    assert typo_path["ambiguous_file_name_match_row_count"] == 1
    assert "ambiguous_file_name_match_rows" in typo_path["blocking_reasons"]


def test_build_reviewer_template_preflight_markdown_lists_blocking_rows() -> None:
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
            ),
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
        },
    ]
    summary = validate_reviewer_template_rows(
        rows,
        min_ready_rows=2,
        require_reviewer_metadata=True,
    )

    markdown = build_reviewer_template_preflight_markdown(
        rows,
        summary,
        require_reviewer_metadata=True,
    )

    assert "Manufacturing Reviewer Template Preflight" in markdown
    assert "ready_template_row_count: `1`" in markdown
    assert "payload_detail_labels_missing" in markdown
    assert "release/flange.dxf" in markdown
    assert "add details.* payload labels" in markdown
    assert "set approved review_status" in markdown


def test_build_reviewer_template_preflight_markdown_lists_unmatched_rows() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [{"file_name": "shaft.dxf", "relative_path": "release/shaft.dxf"}]
    template_rows = [
        {
            "file_name": "missing.dxf",
            "label_cn": "轴类",
            "relative_path": "release/missing.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]
    summary = validate_reviewer_template_rows(
        template_rows,
        min_ready_rows=1,
        base_rows=base_rows,
    )

    markdown = build_reviewer_template_preflight_markdown(
        template_rows,
        summary,
        base_rows=base_rows,
    )

    assert "base_manifest_match_required: `true`" in markdown
    assert "| Unmatched manifest rows | 1 |" in markdown
    assert "unmatched_template_rows" in markdown
    assert "release/missing.dxf" in markdown
    assert "match row identity to review manifest" in markdown


def test_build_reviewer_template_preflight_markdown_reports_empty_base() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    template_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]
    summary = validate_reviewer_template_rows(
        template_rows,
        min_ready_rows=1,
        base_rows=[],
    )

    markdown = build_reviewer_template_preflight_markdown(
        template_rows,
        summary,
        base_rows=[],
    )

    assert "base_manifest_match_required: `true`" in markdown
    assert "base_manifest_row_count: `0`" in markdown
    assert "base_manifest_empty" in markdown
    assert "match row identity to review manifest" in markdown


def test_build_reviewer_template_preflight_markdown_reports_duplicate_base() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {"file_name": "shaft-a.dxf", "relative_path": "release/shaft.dxf"},
        {"file_name": "shaft-b.dxf", "relative_path": "release/shaft.dxf"},
    ]
    template_rows = [
        {
            "file_name": "shaft-a.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]
    summary = validate_reviewer_template_rows(
        template_rows,
        min_ready_rows=1,
        base_rows=base_rows,
    )

    markdown = build_reviewer_template_preflight_markdown(
        template_rows,
        summary,
        base_rows=base_rows,
    )

    assert "base_manifest_duplicate_identity_count: `1`" in markdown
    assert "| Duplicate base manifest row identities | 1 |" in markdown
    assert "base_manifest_duplicate_rows" in markdown
    assert "Duplicate Base Manifest Row IDs" in markdown
    assert "release/shaft.dxf" in markdown


def test_build_reviewer_template_preflight_markdown_reports_ambiguous_file_name() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {"file_name": "shaft.dxf", "relative_path": "release/a/shaft.dxf"},
        {"file_name": "shaft.dxf", "relative_path": "release/b/shaft.dxf"},
    ]
    template_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]
    summary = validate_reviewer_template_rows(
        template_rows,
        min_ready_rows=1,
        base_rows=base_rows,
    )

    markdown = build_reviewer_template_preflight_markdown(
        template_rows,
        summary,
        base_rows=base_rows,
    )

    assert "| Ambiguous file-name fallback rows | 1 |" in markdown
    assert "ambiguous_file_name_match_rows" in markdown
    assert "shaft.dxf" in markdown
    assert "add relative_path to disambiguate duplicate file_name" in markdown


def test_build_reviewer_template_preflight_gap_rows_lists_blockers() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "relative_path": "release/shaft.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
        {
            "file_name": "flange.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "needs_human_review",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": json.dumps(
                {"dfm": {"status": "manufacturable"}}
            ),
        },
        {
            "file_name": "flange-copy.dxf",
            "label_cn": "法兰",
            "relative_path": "release/flange.dxf",
            "review_status": "approved",
            "reviewer": "manufacturing-reviewer",
            "reviewed_at": "2026-05-14",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        },
    ]

    gap_rows = build_reviewer_template_preflight_gap_rows(
        rows,
        require_reviewer_metadata=True,
    )

    assert len(gap_rows) == 2
    assert gap_rows[0]["row_id"] == "release/flange.dxf"
    assert gap_rows[0]["duplicate_row"] == "true"
    assert gap_rows[0]["matched_manifest_row"] == "not_checked"
    assert gap_rows[0]["detail_ready"] == "false"
    assert "add details.* payload labels" in gap_rows[0]["preflight_reasons"]
    assert "set approved review_status" in gap_rows[0]["preflight_reasons"]
    assert "fill reviewer and reviewed_at" in gap_rows[0]["preflight_reasons"]
    assert gap_rows[1]["row_id"] == "release/flange.dxf"
    assert gap_rows[1]["duplicate_row"] == "true"
    assert gap_rows[1]["matched_manifest_row"] == "not_checked"
    assert gap_rows[1]["preflight_reasons"] == "deduplicate row_id"


def test_build_reviewer_template_preflight_gap_rows_lists_unmatched_rows() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [{"file_name": "shaft.dxf", "relative_path": "release/shaft.dxf"}]
    template_rows = [
        {
            "file_name": "missing.dxf",
            "label_cn": "轴类",
            "relative_path": "release/missing.dxf",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    gap_rows = build_reviewer_template_preflight_gap_rows(
        template_rows,
        base_rows=base_rows,
    )

    assert len(gap_rows) == 1
    assert gap_rows[0]["row_id"] == "release/missing.dxf"
    assert gap_rows[0]["matched_manifest_row"] == "false"
    assert "match row identity to review manifest" in gap_rows[0]["preflight_reasons"]


def test_build_reviewer_template_preflight_gap_rows_lists_ambiguous_file_name() -> None:
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    base_rows = [
        {"file_name": "shaft.dxf", "relative_path": "release/a/shaft.dxf"},
        {"file_name": "shaft.dxf", "relative_path": "release/b/shaft.dxf"},
    ]
    template_rows = [
        {
            "file_name": "shaft.dxf",
            "label_cn": "轴类",
            "review_status": "approved",
            "reviewed_manufacturing_evidence_sources": "dfm",
            "reviewed_manufacturing_evidence_payload_json": payload_json,
        }
    ]

    gap_rows = build_reviewer_template_preflight_gap_rows(
        template_rows,
        base_rows=base_rows,
    )

    assert len(gap_rows) == 1
    assert gap_rows[0]["row_id"] == "shaft.dxf"
    assert gap_rows[0]["matched_manifest_row"] == "false"
    assert gap_rows[0]["ambiguous_file_name_match"] == "true"
    assert gap_rows[0]["preflight_reasons"] == (
        "add relative_path to disambiguate duplicate file_name"
    )


def test_build_review_handoff_markdown_lists_artifacts_and_commands() -> None:
    summary = {
        "status": "blocked",
        "row_count": 3,
        "min_reviewed_samples": 2,
        "source_reviewed_sample_count": 1,
        "payload_reviewed_sample_count": 1,
        "payload_detail_reviewed_sample_count": 0,
        "source_reviewed_ready": False,
        "payload_reviewed_ready": False,
        "payload_detail_reviewed_ready": False,
        "require_reviewer_metadata": True,
        "blocking_reasons": [
            "source_reviewed_sample_count_below_minimum",
            "payload_detail_reviewed_sample_count_below_minimum",
        ],
    }

    markdown = build_review_handoff_markdown(
        summary,
        manifest_path="reports/review.csv",
        summary_json_path="reports/review-summary.json",
        progress_md_path="reports/review-progress.md",
        gap_csv_path="reports/review-gaps.csv",
        context_csv_path="reports/review-context.csv",
        batch_csv_path="reports/review-batch.csv",
        batch_template_csv_path="reports/review-batch-template.csv",
        assignment_md_path="reports/review-assignment.md",
        reviewer_template_csv_path="reports/reviewer-template.csv",
        reviewer_template_preflight_md_path="reports/preflight.md",
        reviewer_template_preflight_gap_csv_path="reports/preflight-gaps.csv",
        reviewer_template_preflight_min_ready_rows=1,
    )

    assert "Manufacturing Review Handoff" in markdown
    assert "| Source labels | 1 | 2 | 1 | false |" in markdown
    assert "reports/reviewer-template.csv" in markdown
    assert "reports/review-context.csv" in markdown
    assert "reports/review-batch.csv" in markdown
    assert "reports/review-batch-template.csv" in markdown
    assert "reports/preflight-gaps.csv" in markdown
    assert "--validate-reviewer-template reports/review-batch-template.csv" in markdown
    assert "--apply-reviewer-template reports/review-batch-template.csv" in markdown
    assert "--base-manifest reports/review.csv" in markdown
    assert "--reviewer-template-preflight-md reports/preflight.md" in markdown
    assert "--reviewer-template-preflight-gap-csv reports/preflight-gaps.csv" in markdown
    assert "--min-reviewed-samples 1" in markdown
    assert "--min-reviewed-samples 2" in markdown
    assert "--require-reviewer-metadata" in markdown
    assert "Do not copy suggestions into reviewed fields without human review" in markdown
    assert "payload_detail_reviewed_sample_count_below_minimum" in markdown

    default_markdown = build_review_handoff_markdown(summary)
    assert default_markdown.count("--min-reviewed-samples 2") == 2

    full_template_markdown = build_review_handoff_markdown(
        summary,
        reviewer_template_csv_path="reports/reviewer-template.csv",
    )
    assert (
        "--validate-reviewer-template reports/reviewer-template.csv"
        in full_template_markdown
    )


def test_main_build_and_validate_manifest(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    output_csv = tmp_path / "review.csv"
    summary_json = tmp_path / "summary.json"
    progress_md = tmp_path / "progress.md"
    gap_csv = tmp_path / "gaps.csv"
    context_csv = tmp_path / "context.csv"
    batch_csv = tmp_path / "batch.csv"
    batch_template_csv = tmp_path / "batch-template.csv"
    assignment_md = tmp_path / "assignment.md"
    reviewer_template_csv = tmp_path / "reviewer-template.csv"
    handoff_md = tmp_path / "handoff.md"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "true_label", "manufacturing_evidence"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "true_label": "轴类",
                "manufacturing_evidence": json.dumps(
                    [
                        {
                            "source": "dfm",
                            "status": "manufacturable",
                            "details": {"mode": "rule"},
                        }
                    ]
                ),
            }
        )

    assert main(
        [
            "--from-results-csv",
            str(results_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--progress-md",
            str(progress_md),
            "--gap-csv",
            str(gap_csv),
            "--review-context-csv",
            str(context_csv),
            "--review-batch-csv",
            str(batch_csv),
            "--review-batch-template-csv",
            str(batch_template_csv),
            "--assignment-md",
            str(assignment_md),
            "--reviewer-template-csv",
            str(reviewer_template_csv),
            "--handoff-md",
            str(handoff_md),
            "--prefill-reviewed-from-suggestions",
            "--min-reviewed-samples",
            "1",
            "--fail-under-minimum",
        ]
    ) == 1

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["unapproved_review_sample_count"] == 1
    assert summary["source_reviewed_sample_count"] == 0
    assert summary["payload_detail_reviewed_sample_count"] == 0
    assert output_csv.exists()
    assert "Manufacturing Evidence Review Progress" in progress_md.read_text(
        encoding="utf-8"
    )
    with gap_csv.open("r", encoding="utf-8", newline="") as handle:
        gap_rows = list(csv.DictReader(handle))
    assert len(gap_rows) == 1
    assert "set approved review_status" in gap_rows[0]["gap_reasons"]
    with context_csv.open("r", encoding="utf-8", newline="") as handle:
        context_rows = list(csv.DictReader(handle))
    assert len(context_rows) == 1
    assert context_rows[0]["actual_evidence_sources"] == "dfm"
    assert "details.mode" in context_rows[0]["actual_evidence_detail_keys"]
    with batch_csv.open("r", encoding="utf-8", newline="") as handle:
        batch_rows = list(csv.DictReader(handle))
    assert len(batch_rows) == 1
    assert batch_rows[0]["review_batch"] == "batch_001"
    assert batch_rows[0]["source_gap"] == "false"
    assert batch_rows[0]["approval_gap"] == "true"
    with batch_template_csv.open("r", encoding="utf-8", newline="") as handle:
        batch_template_rows = list(csv.DictReader(handle))
    assert len(batch_template_rows) == 1
    assert batch_template_rows[0]["review_batch"] == "batch_001"
    assert batch_template_rows[0]["row_id"] == "shaft.dxf"
    assert batch_template_rows[0]["review_status"] == "needs_human_review"
    assert "Manufacturing Review Assignment Plan" in assignment_md.read_text(
        encoding="utf-8"
    )
    with reviewer_template_csv.open("r", encoding="utf-8", newline="") as handle:
        template_rows = list(csv.DictReader(handle))
    assert len(template_rows) == 1
    assert template_rows[0]["review_status"] == "needs_human_review"
    handoff = handoff_md.read_text(encoding="utf-8")
    assert "Manufacturing Review Handoff" in handoff
    assert str(reviewer_template_csv) in handoff
    assert "--validate-reviewer-template" in handoff


def test_main_merge_approved_review_manifest_writes_only_approved_rows(
    tmp_path: Path,
) -> None:
    base_manifest = tmp_path / "base.csv"
    review_manifest = tmp_path / "review.csv"
    output_csv = tmp_path / "merged.csv"
    summary_json = tmp_path / "merge-summary.json"
    merge_audit_csv = tmp_path / "merge-audit.csv"
    with base_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "label_cn", "relative_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/shaft.dxf",
            }
        )
        writer.writerow(
            {
                "file_name": "flange.dxf",
                "label_cn": "法兰",
                "relative_path": "release/flange.dxf",
            }
        )

    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    with review_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "label_cn",
                "relative_path",
                "review_status",
                "reviewer",
                "reviewed_at",
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_evidence_payload_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/shaft.dxf",
                "review_status": "approved",
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-13",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": payload_json,
            }
        )
        writer.writerow(
            {
                "file_name": "flange.dxf",
                "label_cn": "法兰",
                "relative_path": "release/flange.dxf",
                "review_status": "needs_human_review",
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-13",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": payload_json,
            }
        )
        writer.writerow(
            {
                "file_name": "nut.dxf",
                "label_cn": "螺母",
                "relative_path": "release/nut.dxf",
                "review_status": "approved",
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-13",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": payload_json,
            }
        )

    assert main(
        [
            "--merge-approved-review-manifest",
            str(review_manifest),
            "--base-manifest",
            str(base_manifest),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--review-manifest-merge-audit-csv",
            str(merge_audit_csv),
            "--require-reviewer-metadata",
            "--fail-under-minimum",
        ]
    ) == 0

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["status"] == "merged"
    assert summary["approved_review_row_count"] == 2
    assert summary["merged_row_count"] == 1
    assert summary["skipped_unapproved_review_row_count"] == 1
    assert summary["unmatched_review_row_count"] == 1

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert output_rows[0]["reviewed_manufacturing_evidence_sources"] == "dfm"
    assert output_rows[0]["reviewed_manufacturing_evidence_payload_json"] == payload_json
    assert output_rows[0]["review_status"] == "approved"
    assert output_rows[1]["reviewed_manufacturing_evidence_sources"] == ""
    assert output_rows[1]["review_status"] == ""
    with merge_audit_csv.open("r", encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert [row["merge_status"] for row in audit_rows] == [
        "merged",
        "skipped_unapproved_review",
        "unmatched_review_row",
    ]


def test_main_apply_reviewer_template_updates_review_manifest(tmp_path: Path) -> None:
    base_manifest = tmp_path / "review-base.csv"
    reviewer_template = tmp_path / "reviewer-template.csv"
    output_csv = tmp_path / "review-updated.csv"
    summary_json = tmp_path / "apply-summary.json"
    apply_audit_csv = tmp_path / "apply-audit.csv"
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    with base_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "label_cn",
                "relative_path",
                "review_status",
                "reviewer",
                "reviewed_at",
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_evidence_payload_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/shaft.dxf",
                "review_status": "approved",
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-13",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": payload_json,
            }
        )
        writer.writerow(
            {
                "file_name": "flange.dxf",
                "label_cn": "法兰",
                "relative_path": "release/flange.dxf",
                "review_status": "needs_human_review",
            }
        )
    with reviewer_template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "label_cn",
                "relative_path",
                "review_status",
                "reviewer",
                "reviewed_at",
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_evidence_payload_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "flange.dxf",
                "label_cn": "法兰",
                "relative_path": "release/flange.dxf",
                "review_status": "approved",
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-14",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": payload_json,
            }
        )

    assert main(
        [
            "--apply-reviewer-template",
            str(reviewer_template),
            "--base-manifest",
            str(base_manifest),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--reviewer-template-apply-audit-csv",
            str(apply_audit_csv),
            "--min-reviewed-samples",
            "2",
            "--require-reviewer-metadata",
            "--fail-under-minimum",
        ]
    ) == 0

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["status"] == "applied"
    assert summary["applied_row_count"] == 1
    assert summary["post_apply_validation"]["status"] == "release_label_ready"
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert output_rows[1]["review_status"] == "approved"
    assert output_rows[1]["reviewed_at"] == "2026-05-14"
    with apply_audit_csv.open("r", encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert audit_rows[0]["apply_status"] == "applied"
    assert audit_rows[0]["matched_manifest_row"] == "true"


def test_main_validate_reviewer_template_writes_summary(tmp_path: Path) -> None:
    base_manifest = tmp_path / "review-manifest.csv"
    reviewer_template = tmp_path / "reviewer-template.csv"
    summary_json = tmp_path / "template-summary.json"
    preflight_md = tmp_path / "template-preflight.md"
    preflight_gap_csv = tmp_path / "template-preflight-gaps.csv"
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    with base_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "label_cn", "relative_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/shaft.dxf",
            }
        )
    with reviewer_template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "label_cn",
                "relative_path",
                "review_status",
                "reviewer",
                "reviewed_at",
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_evidence_payload_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/shaft.dxf",
                "review_status": "approved",
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-14",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": payload_json,
            }
        )

    assert main(
        [
            "--validate-reviewer-template",
            str(reviewer_template),
            "--base-manifest",
            str(base_manifest),
            "--summary-json",
            str(summary_json),
            "--reviewer-template-preflight-md",
            str(preflight_md),
            "--reviewer-template-preflight-gap-csv",
            str(preflight_gap_csv),
            "--min-reviewed-samples",
            "1",
            "--require-reviewer-metadata",
            "--fail-under-minimum",
        ]
    ) == 0

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["mode"] == "validate_reviewer_template"
    assert summary["base_manifest"] == str(base_manifest)
    assert summary["base_manifest_match_required"] is True
    assert summary["unmatched_template_row_count"] == 0
    assert summary["status"] == "ready"
    assert summary["ready_template_row_count"] == 1
    assert summary["blocking_reasons"] == []
    assert "Manufacturing Reviewer Template Preflight" in preflight_md.read_text(
        encoding="utf-8"
    )
    with preflight_gap_csv.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_main_validate_reviewer_template_reports_ambiguous_file_name(
    tmp_path: Path,
) -> None:
    base_manifest = tmp_path / "review-manifest.csv"
    reviewer_template = tmp_path / "reviewer-template.csv"
    summary_json = tmp_path / "template-summary.json"
    preflight_md = tmp_path / "template-preflight.md"
    preflight_gap_csv = tmp_path / "template-preflight-gaps.csv"
    payload_json = json.dumps(
        {"dfm": {"status": "manufacturable", "details": {"mode": "rule"}}}
    )
    with base_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "label_cn", "relative_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/a/shaft.dxf",
            }
        )
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/b/shaft.dxf",
            }
        )
    with reviewer_template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "label_cn",
                "relative_path",
                "review_status",
                "reviewer",
                "reviewed_at",
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_evidence_payload_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "",
                "review_status": "approved",
                "reviewer": "manufacturing-reviewer",
                "reviewed_at": "2026-05-14",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": payload_json,
            }
        )

    assert (
        main(
            [
                "--validate-reviewer-template",
                str(reviewer_template),
                "--base-manifest",
                str(base_manifest),
                "--summary-json",
                str(summary_json),
                "--reviewer-template-preflight-md",
                str(preflight_md),
                "--reviewer-template-preflight-gap-csv",
                str(preflight_gap_csv),
                "--min-reviewed-samples",
                "1",
                "--require-reviewer-metadata",
                "--fail-under-minimum",
            ]
        )
        == 1
    )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["mode"] == "validate_reviewer_template"
    assert summary["reviewer_template"] == str(reviewer_template)
    assert summary["base_manifest"] == str(base_manifest)
    assert summary["status"] == "blocked"
    assert summary["template_row_count"] == 1
    assert summary["base_manifest_row_count"] == 2
    assert summary["ready_template_row_count"] == 0
    assert summary["unmatched_template_row_count"] == 0
    assert summary["ambiguous_file_name_match_row_count"] == 1
    assert set(summary["blocking_reasons"]) == {
        "ready_template_row_count_below_minimum",
        "ambiguous_file_name_match_rows",
    }

    markdown = preflight_md.read_text(encoding="utf-8")
    assert "Manufacturing Reviewer Template Preflight" in markdown
    assert "ready_template_row_count: `0`" in markdown
    assert "| Unmatched manifest rows | 0 |" in markdown
    assert "| Ambiguous file-name fallback rows | 1 |" in markdown
    assert "shaft.dxf" in markdown
    assert "add relative_path to disambiguate duplicate file_name" in markdown

    with preflight_gap_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "relative_path" in reader.fieldnames
        assert "ambiguous_file_name_match" in reader.fieldnames
        gap_rows = list(reader)
    assert len(gap_rows) == 1
    assert gap_rows[0]["row_id"] == "shaft.dxf"
    assert gap_rows[0]["relative_path"] == ""
    assert gap_rows[0]["matched_manifest_row"] == "false"
    assert gap_rows[0]["ambiguous_file_name_match"] == "true"
    assert gap_rows[0]["preflight_reasons"] == (
        "add relative_path to disambiguate duplicate file_name"
    )

    no_fail_summary_json = tmp_path / "template-summary-no-fail.json"
    assert (
        main(
            [
                "--validate-reviewer-template",
                str(reviewer_template),
                "--base-manifest",
                str(base_manifest),
                "--summary-json",
                str(no_fail_summary_json),
                "--min-reviewed-samples",
                "1",
                "--require-reviewer-metadata",
            ]
        )
        == 0
    )
    no_fail_summary = json.loads(no_fail_summary_json.read_text(encoding="utf-8"))
    assert no_fail_summary["status"] == "blocked"
    assert no_fail_summary["ambiguous_file_name_match_row_count"] == 1
