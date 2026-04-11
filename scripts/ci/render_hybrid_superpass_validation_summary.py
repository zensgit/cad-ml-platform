#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

try:
    from scripts.ci.summary_render_utils import (
        append_markdown_section,
        is_zeroish,
        read_json_object,
        render_inline_items,
        top_nonempty,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.ci.summary_render_utils import (
        append_markdown_section,
        is_zeroish,
        read_json_object,
        render_inline_items,
        top_nonempty,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    summary_obj = summary if isinstance(summary, dict) else {}
    inputs = payload.get("inputs")
    inputs_obj = inputs if isinstance(inputs, dict) else {}
    errors = payload.get("errors")
    errors_list = errors if isinstance(errors, list) else []
    warnings = payload.get("warnings")
    warnings_list = warnings if isinstance(warnings, list) else []
    status_text = str(payload.get("status", "n/a")).strip().lower()
    verdict = (
        "ok"
        if is_zeroish(payload.get("overall_exit_code", 1))
        and status_text in {"ok", "pass", "passed", "warn", "warning"}
        and len(errors_list) == 0
        else "attention_required"
    )
    top_errors = top_nonempty(errors_list)
    top_warnings = top_nonempty(warnings_list)

    lines = ["## Hybrid Superpass Validation"]
    append_markdown_section(
        lines,
        "Validation Verdict",
        [
            ("verdict", verdict),
            ("status", payload.get("status", "n/a")),
            ("strict", payload.get("strict", "n/a")),
            ("schema_mode", payload.get("schema_mode", "n/a")),
            ("overall_exit_code", payload.get("overall_exit_code", "n/a")),
            ("top_errors", render_inline_items(top_errors)),
            ("top_warnings", render_inline_items(top_warnings)),
        ],
    )
    append_markdown_section(
        lines,
        "Validation Snapshot",
        [
            ("inputs.superpass_json", inputs_obj.get("superpass_json", "")),
            ("inputs.hybrid_blind_gate_report", inputs_obj.get("hybrid_blind_gate_report", "")),
            ("inputs.hybrid_calibration_json", inputs_obj.get("hybrid_calibration_json", "")),
            ("summary.superpass_status", summary_obj.get("superpass_status", "n/a")),
            ("summary.superpass_check_count", summary_obj.get("superpass_check_count", "n/a")),
            ("summary.superpass_failure_count", summary_obj.get("superpass_failure_count", "n/a")),
            ("summary.superpass_warning_count", summary_obj.get("superpass_warning_count", "n/a")),
            ("summary.gate_hybrid_accuracy", summary_obj.get("gate_hybrid_accuracy", "n/a")),
            (
                "summary.gate_hybrid_gain_vs_graph2d",
                summary_obj.get("gate_hybrid_gain_vs_graph2d", "n/a"),
            ),
            ("summary.calibration_ece", summary_obj.get("calibration_ece", "n/a")),
            ("errors_total", len(errors_list)),
            ("warnings_total", len(warnings_list)),
        ],
    )
    if errors_list:
        lines.extend(["", "### Errors"])
        for item in errors_list:
            lines.append(f"- {item}")
    if warnings_list:
        lines.extend(["", "### Warnings"])
        for item in warnings_list:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render markdown summary from hybrid superpass validation json."
    )
    parser.add_argument("--validation-json", required=True, help="Validation json path")
    parser.add_argument("--output-md", default="", help="Optional markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    validation_path = Path(str(args.validation_json)).expanduser()
    if not validation_path.exists():
        print(f"validation json does not exist: {validation_path}")
        return 1

    try:
        payload = read_json_object(validation_path, "validation")
        markdown = render_markdown(payload)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    output_md = str(args.output_md or "").strip()
    if output_md:
        output_path = Path(output_md).expanduser()
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
