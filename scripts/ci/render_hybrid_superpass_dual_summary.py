#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_TITLE = "Hybrid Superpass Dual Dispatch Summary"


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _format_value(value: Any) -> str:
    if not _has_value(value):
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().replace("|", "\\|")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to read json: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"json root is not object: {path}")
    return payload


def _read_optional_json_object(path_value: str) -> dict[str, Any]:
    token = _normalize_text(path_value)
    if not token:
        return {}

    path = Path(token).expanduser()
    if not path.exists():
        return {}

    try:
        return _read_json_object(path)
    except ValueError:
        return {}


def _read_optional_text(path_value: str) -> str | None:
    token = _normalize_text(path_value)
    if not token:
        return None

    path = Path(token).expanduser()
    if not path.exists():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _lookup(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _pick(primary: Mapping[str, Any], secondary: Mapping[str, Any], *keys: str) -> Any:
    value = _lookup(primary, *keys)
    if _has_value(value):
        return value
    value = _lookup(secondary, *keys)
    if _has_value(value):
        return value
    return None


def build_markdown(
    dual_summary: Mapping[str, Any],
    compare_summary: Mapping[str, Any],
    *,
    title: str,
    compare_markdown: str | None,
) -> str:
    overall_exit_code = _format_value(
        _pick(dual_summary, compare_summary, "overall_exit_code")
    )
    fail_dispatch_exit_code = _format_value(
        _pick(dual_summary, compare_summary, "fail_dispatch_exit_code")
    )
    success_dispatch_exit_code = _format_value(
        _pick(dual_summary, compare_summary, "success_dispatch_exit_code")
    )
    compare_exit_code = _format_value(
        _pick(dual_summary, compare_summary, "compare_exit_code")
    )
    run_id_is_different = _format_value(
        _pick(dual_summary, compare_summary, "run_id_is_different")
    )
    fail_expected_failure = _format_value(
        _pick(dual_summary, compare_summary, "checks", "fail_expected_failure")
    )
    success_expected_success = _format_value(
        _pick(dual_summary, compare_summary, "checks", "success_expected_success")
    )
    trace_pair_consistent = _format_value(
        _pick(dual_summary, compare_summary, "checks", "trace_pair_consistent")
    )
    strict_mode = _format_value(_pick(dual_summary, compare_summary, "strict_mode"))
    strict_distinct = _format_value(
        _pick(dual_summary, compare_summary, "strict_require_distinct_run_ids")
    )
    strict_trace_pair = _format_value(
        _pick(dual_summary, compare_summary, "strict_require_trace_pair")
    )
    strict_failed = _format_value(_pick(dual_summary, compare_summary, "strict_failed"))

    lines = [
        f"# {title}",
        "",
        "## Exit Codes",
        "",
        f"- overall: `{overall_exit_code}`",
        f"- fail dispatch: `{fail_dispatch_exit_code}`",
        f"- success dispatch: `{success_dispatch_exit_code}`",
        f"- compare: `{compare_exit_code}`",
        "",
        "## Key Checks",
        "",
        f"- run_id_is_different: `{run_id_is_different}`",
        f"- fail_expected_failure: `{fail_expected_failure}`",
        f"- success_expected_success: `{success_expected_success}`",
        f"- trace_pair_consistent: `{trace_pair_consistent}`",
        "- strict_flags: "
        f"`strict_mode={strict_mode}, "
        f"strict_require_distinct_run_ids={strict_distinct}, "
        f"strict_require_trace_pair={strict_trace_pair}`",
        f"- strict_failed: `{strict_failed}`",
        "",
        "## Run Info",
        "",
    ]

    for scenario in ("fail", "success"):
        run_id = _format_value(_pick(dual_summary, compare_summary, scenario, "run_id"))
        run_url = _format_value(_pick(dual_summary, compare_summary, scenario, "run_url"))
        dispatch_trace_id = _format_value(
            _pick(dual_summary, compare_summary, scenario, "dispatch_trace_id")
        )
        lines.extend(
            [
                f"### {scenario}",
                f"- run_id: `{run_id}`",
                f"- run_url: `{run_url}`",
                f"- dispatch_trace_id: `{dispatch_trace_id}`",
                "",
            ]
        )

    if compare_markdown is not None:
        lines.extend(["## Compare Markdown", ""])
        rendered_compare_md = compare_markdown.rstrip("\n")
        if rendered_compare_md:
            lines.append(rendered_compare_md)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_text(path: Path, content: str) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render markdown report from hybrid superpass dual dispatch summary JSON."
    )
    parser.add_argument("--dual-summary-json", required=True)
    parser.add_argument("--compare-json", default="")
    parser.add_argument("--compare-md", default="")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dual_path = Path(str(args.dual_summary_json)).expanduser()
    try:
        dual_summary = _read_json_object(dual_path)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    compare_summary = _read_optional_json_object(str(args.compare_json))
    compare_markdown = _read_optional_text(str(args.compare_md))

    markdown = build_markdown(
        dual_summary,
        compare_summary,
        title=str(args.title),
        compare_markdown=compare_markdown,
    )

    try:
        _write_text(Path(str(args.output_md)).expanduser(), markdown)
    except OSError as exc:
        print(f"error: failed to write markdown: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
