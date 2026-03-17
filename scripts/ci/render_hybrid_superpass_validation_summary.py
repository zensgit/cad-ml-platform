#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _read_validation(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read validation json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse validation json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("validation json must be an object")
    return payload


def _is_zeroish(value: Any) -> bool:
    return str(value if value is not None else "").strip() == "0"


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
        if _is_zeroish(payload.get("overall_exit_code", 1))
        and status_text in {"ok", "pass", "passed", "warn", "warning"}
        and len(errors_list) == 0
        else "attention_required"
    )
    top_errors = [str(item).strip() for item in errors_list if str(item).strip()][:3]
    top_warnings = [str(item).strip() for item in warnings_list if str(item).strip()][:3]

    lines = [
        "## Hybrid Superpass Validation",
        "",
        "## Validation Verdict",
        "",
        f"- verdict: {verdict}",
        f"- status: {payload.get('status', 'n/a')}",
        f"- strict: {payload.get('strict', 'n/a')}",
        f"- schema_mode: {payload.get('schema_mode', 'n/a')}",
        f"- overall_exit_code: {payload.get('overall_exit_code', 'n/a')}",
        f"- top_errors: {', '.join(top_errors) if top_errors else '(none)'}",
        f"- top_warnings: {', '.join(top_warnings) if top_warnings else '(none)'}",
        "",
        "## Validation Snapshot",
        "",
        f"- inputs.superpass_json: {inputs_obj.get('superpass_json', '')}",
        f"- inputs.hybrid_blind_gate_report: {inputs_obj.get('hybrid_blind_gate_report', '')}",
        f"- inputs.hybrid_calibration_json: {inputs_obj.get('hybrid_calibration_json', '')}",
        f"- summary.superpass_status: {summary_obj.get('superpass_status', 'n/a')}",
        f"- summary.superpass_check_count: {summary_obj.get('superpass_check_count', 'n/a')}",
        f"- summary.superpass_failure_count: {summary_obj.get('superpass_failure_count', 'n/a')}",
        f"- summary.superpass_warning_count: {summary_obj.get('superpass_warning_count', 'n/a')}",
        f"- summary.gate_hybrid_accuracy: {summary_obj.get('gate_hybrid_accuracy', 'n/a')}",
        f"- summary.gate_hybrid_gain_vs_graph2d: {summary_obj.get('gate_hybrid_gain_vs_graph2d', 'n/a')}",
        f"- summary.calibration_ece: {summary_obj.get('calibration_ece', 'n/a')}",
        f"- errors_total: {len(errors_list)}",
        f"- warnings_total: {len(warnings_list)}",
    ]
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
        payload = _read_validation(validation_path)
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
