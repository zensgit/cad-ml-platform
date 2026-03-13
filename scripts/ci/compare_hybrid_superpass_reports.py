#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_conclusion(value: Any) -> str:
    return _normalize_text(value).lower()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    token = _normalize_text(value).lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - error path exercised through main
        raise ValueError(f"failed to read json: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"json root is not object: {path}")
    return payload


def _extract_case(
    payload: Dict[str, Any],
    *,
    scenario: str,
    default_expected_conclusion: str,
) -> Dict[str, Any]:
    conclusion = _normalize_conclusion(payload.get("conclusion"))
    expected_conclusion = _normalize_conclusion(
        payload.get("expected_conclusion") or default_expected_conclusion
    )
    matched_from_payload = payload.get("matched_expectation")
    matched_expectation = _coerce_bool(
        matched_from_payload,
        default=(conclusion == expected_conclusion),
    )

    return {
        "scenario": scenario,
        "run_id": _normalize_text(payload.get("run_id")),
        "conclusion": conclusion,
        "expected_conclusion": expected_conclusion,
        "matched_expectation": bool(matched_expectation),
        "dispatch_trace_id": _normalize_text(payload.get("dispatch_trace_id")),
        "run_url": _normalize_text(payload.get("run_url")),
    }


def _markdown_cell(value: Any) -> str:
    token = _normalize_text(value)
    if not token:
        return "-"
    return token.replace("|", "\\|")


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def build_summary(
    fail_payload: Dict[str, Any],
    success_payload: Dict[str, Any],
) -> Dict[str, Any]:
    fail_case = _extract_case(
        fail_payload,
        scenario="fail",
        default_expected_conclusion="failure",
    )
    success_case = _extract_case(
        success_payload,
        scenario="success",
        default_expected_conclusion="success",
    )

    fail_run_id = _normalize_text(fail_case.get("run_id"))
    success_run_id = _normalize_text(success_case.get("run_id"))

    run_id_is_different = bool(
        fail_run_id and success_run_id and fail_run_id != success_run_id
    )
    fail_expected_failure = bool(
        fail_case["matched_expectation"]
        and fail_case["expected_conclusion"] == "failure"
    )
    success_expected_success = bool(
        success_case["matched_expectation"]
        and success_case["expected_conclusion"] == "success"
    )

    warnings: list[str] = []
    if not fail_run_id or not success_run_id:
        warnings.append(
            "missing run_id in one or both scenarios; cannot fully validate isolation"
        )
    elif not run_id_is_different:
        warnings.append(
            "fail/success scenarios share the same run_id; possible parallel dispatch mix-up"
        )

    return {
        "fail": fail_case,
        "success": success_case,
        "run_id_is_different": run_id_is_different,
        "checks": {
            "fail_expected_failure": fail_expected_failure,
            "success_expected_success": success_expected_success,
        },
        "warnings": warnings,
    }


def build_markdown(summary: Dict[str, Any], *, strict: bool, exit_code: int) -> str:
    fail_case = summary.get("fail", {})
    success_case = summary.get("success", {})
    checks = summary.get("checks", {})
    warnings = summary.get("warnings", [])

    lines: list[str] = [
        "# Hybrid Superpass Dispatch Comparison",
        "",
        "## Comparison Table",
        "",
        "| Scenario | run_id | conclusion | expected_conclusion | "
        "matched_expectation | dispatch_trace_id | run_url |",
        "|---|---|---|---|---|---|---|",
        "| fail | "
        f"`{_markdown_cell(fail_case.get('run_id'))}` | "
        f"`{_markdown_cell(fail_case.get('conclusion'))}` | "
        f"`{_markdown_cell(fail_case.get('expected_conclusion'))}` | "
        f"`{_yes_no(_coerce_bool(fail_case.get('matched_expectation')))}` | "
        f"`{_markdown_cell(fail_case.get('dispatch_trace_id'))}` | "
        f"{_markdown_cell(fail_case.get('run_url'))} |",
        "| success | "
        f"`{_markdown_cell(success_case.get('run_id'))}` | "
        f"`{_markdown_cell(success_case.get('conclusion'))}` | "
        f"`{_markdown_cell(success_case.get('expected_conclusion'))}` | "
        f"`{_yes_no(_coerce_bool(success_case.get('matched_expectation')))}` | "
        f"`{_markdown_cell(success_case.get('dispatch_trace_id'))}` | "
        f"{_markdown_cell(success_case.get('run_url'))} |",
        "",
        "## Key Conclusions",
        "",
        "- Runs are different (parallel run isolation check): "
        f"**{_yes_no(_coerce_bool(summary.get('run_id_is_different')))}**",
        "- Fail scenario failed as expected: "
        f"**{_yes_no(_coerce_bool(checks.get('fail_expected_failure')))}**",
        "- Success scenario succeeded as expected: "
        f"**{_yes_no(_coerce_bool(checks.get('success_expected_success')))}**",
        "- Strict mode result: "
        + (
            "**FAILED (exit 1)**"
            if strict and exit_code != 0
            else ("**PASSED (exit 0)**" if strict else "not enabled")
        ),
        "",
    ]

    if isinstance(warnings, list) and warnings:
        lines.append("## Warnings")
        lines.append("")
        for item in warnings:
            lines.append(f"- {_normalize_text(item)}")
        lines.append("")

    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fail/success hybrid superpass dispatch outputs and generate "
            "machine/json + markdown summaries."
        )
    )
    parser.add_argument(
        "--fail-json",
        required=True,
        help="Path to fail scenario dispatch JSON",
    )
    parser.add_argument(
        "--success-json",
        required=True,
        help="Path to success scenario dispatch JSON",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output machine summary JSON",
    )
    parser.add_argument("--output-md", required=True, help="Output markdown report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if fail/success scenario does not match expectation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        fail_payload = _read_json_object(Path(str(args.fail_json)).expanduser())
        success_payload = _read_json_object(Path(str(args.success_json)).expanduser())
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    summary = build_summary(fail_payload, success_payload)

    fail_matched = _coerce_bool(summary.get("fail", {}).get("matched_expectation"))
    success_matched = _coerce_bool(summary.get("success", {}).get("matched_expectation"))
    strict_failed = bool(args.strict) and (not fail_matched or not success_matched)
    exit_code = 1 if strict_failed else 0

    output_payload: Dict[str, Any] = {
        **summary,
        "strict_mode": bool(args.strict),
        "strict_failed": strict_failed,
        "overall_exit_code": exit_code,
    }

    markdown = build_markdown(output_payload, strict=bool(args.strict), exit_code=exit_code)
    _write_json(Path(str(args.output_json)).expanduser(), output_payload)
    _write_text(Path(str(args.output_md)).expanduser(), markdown)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
