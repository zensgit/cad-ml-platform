#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


SEVERITY_ORDER: Dict[str, int] = {
    "clear": 0,
    "warn": 1,
    "alerted": 2,
    "failed": 3,
}

DEFAULT_POLICY: Dict[str, Any] = {
    "max_allowed_severity": "alerted",
    "fail_on_breach": False,
}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_severity(value: Any, default: str = "clear") -> str:
    token = str(value).strip().lower()
    if token in SEVERITY_ORDER:
        return token
    return str(default).strip().lower()


def _load_yaml_defaults(config_path: str, section: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    data = payload.get(section)
    return data if isinstance(data, dict) else {}


def _resolve_policy(
    *,
    config_payload: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    max_allowed_severity = _normalize_severity(
        (
            cli_overrides.get("max_allowed_severity")
            if cli_overrides.get("max_allowed_severity") is not None
            else config_payload.get("max_allowed_severity")
        ),
        _normalize_severity(DEFAULT_POLICY.get("max_allowed_severity"), "alerted"),
    )
    fail_on_breach = _safe_bool(
        config_payload.get("fail_on_breach"),
        _safe_bool(DEFAULT_POLICY.get("fail_on_breach"), False),
    )
    fail_on_breach_cli = str(cli_overrides.get("fail_on_breach", "auto")).strip().lower()
    if fail_on_breach_cli in {"true", "false"}:
        fail_on_breach = fail_on_breach_cli == "true"
    return {
        "max_allowed_severity": max_allowed_severity,
        "fail_on_breach": bool(fail_on_breach),
    }


def evaluate_policy(index_payload: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    overview = _as_dict(index_payload.get("overview"))
    current = _normalize_severity(overview.get("severity"), "clear")
    allowed = _normalize_severity(policy.get("max_allowed_severity"), "alerted")
    breached = SEVERITY_ORDER.get(current, 0) > SEVERITY_ORDER.get(allowed, 2)
    return {
        "status": "breached" if breached else "pass",
        "current_severity": current,
        "max_allowed_severity": allowed,
        "breached": bool(breached),
        "reason": (
            f"current severity '{current}' exceeds allowed '{allowed}'"
            if breached
            else f"current severity '{current}' within allowed '{allowed}'"
        ),
    }


def build_markdown(report: Dict[str, Any], title: str) -> str:
    policy_source = _as_dict(report.get("policy_source"))
    resolved = _as_dict(policy_source.get("resolved_policy"))
    out: List[str] = []
    out.append(f"### {title}")
    out.append("")
    out.append("| Check | Value |")
    out.append("|---|---|")
    out.append(f"| Status | `{str(report.get('status', 'pass'))}` |")
    out.append(
        f"| Current severity | `{str(report.get('current_severity', 'clear'))}` |"
    )
    out.append(
        f"| Max allowed severity | `{str(report.get('max_allowed_severity', 'alerted'))}` |"
    )
    out.append(f"| Breached | `{bool(report.get('breached', False))}` |")
    out.append(
        "| Policy source | "
        f"`config={policy_source.get('config', '')}, "
        f"loaded={bool(policy_source.get('config_loaded', False))}, "
        f"resolved_max_allowed={resolved.get('max_allowed_severity', '')}, "
        f"fail_on_breach={bool(resolved.get('fail_on_breach', False))}` |"
    )
    out.append("")
    out.append("```text")
    out.append(str(report.get("reason", "")))
    out.append("```")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Graph2D context drift index severity against policy."
    )
    parser.add_argument("--index-json", required=True, help="Path to index json.")
    parser.add_argument(
        "--config",
        default="config/graph2d_context_drift_index_policy.yaml",
        help="YAML policy config path.",
    )
    parser.add_argument(
        "--config-section",
        default="graph2d_context_drift_index_policy",
        help="Config section name in yaml.",
    )
    parser.add_argument(
        "--max-allowed-severity",
        choices=["clear", "warn", "alerted", "failed"],
        default=None,
        help="Override max allowed severity.",
    )
    parser.add_argument(
        "--fail-on-breach",
        choices=["auto", "true", "false"],
        default="auto",
        help="Exit non-zero when breached (auto uses config).",
    )
    parser.add_argument("--title", default="Graph2D Context Drift Index Policy")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    index_payload = _as_dict(_read_json(Path(str(args.index_json))))
    config_payload = _load_yaml_defaults(str(args.config), str(args.config_section))
    policy = _resolve_policy(
        config_payload=config_payload,
        cli_overrides={
            "max_allowed_severity": args.max_allowed_severity,
            "fail_on_breach": args.fail_on_breach,
        },
    )
    report = evaluate_policy(index_payload, policy)
    report["policy_source"] = {
        "config": str(args.config),
        "config_section": str(args.config_section),
        "config_loaded": bool(config_payload),
        "resolved_policy": policy,
        "cli_overrides": {
            key: value
            for key, value in {
                "max_allowed_severity": args.max_allowed_severity,
                "fail_on_breach": (
                    args.fail_on_breach if str(args.fail_on_breach) != "auto" else None
                ),
            }.items()
            if value is not None
        },
    }

    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    md_text = build_markdown(report, str(args.title))
    if str(args.output_json).strip():
        out_json = Path(str(args.output_json))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json_text + "\n", encoding="utf-8")
    if str(args.output_md).strip():
        out_md = Path(str(args.output_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_text + "\n", encoding="utf-8")

    print(json_text)
    if bool(policy.get("fail_on_breach", False)) and bool(report.get("breached", False)):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
