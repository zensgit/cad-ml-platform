#!/usr/bin/env python3
"""Fast CI guardrails for training data governance invariants."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_GOLDEN_TRAIN = "data/manifests/golden_train_set.csv"
DEFAULT_GOLDEN_VAL = "data/manifests/golden_val_set.csv"
DEFAULT_AUTO_RETRAIN = "scripts/auto_retrain.sh"
DEFAULT_APPEND_REVIEWED = "scripts/append_reviewed_to_manifest.py"
DEFAULT_FINETUNE_AUGMENTED = "scripts/finetune_graph2d_v2_augmented.py"
DEFAULT_FINETUNE_PRETRAINED = "scripts/finetune_graph2d_from_pretrained.py"
DEFAULT_ACTIVE_LEARNING_API = "src/api/v1/active_learning.py"
DEFAULT_ACTIVE_LEARNING_CORE = "src/core/active_learning.py"
DEFAULT_BACKFILL_HELPER = "scripts/backfill_manifest_cache_paths.py"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip()


def _collect_manifest_keys(
    rows: list[dict[str, str]],
    *,
    field_name: str,
) -> set[str]:
    return {
        token
        for row in rows
        if (token := _normalize_token(row.get(field_name)))
    }


def _check_required_substrings(
    *,
    path: Path,
    required_substrings: list[str],
) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "status": "error",
            "path": str(path),
            "missing_tokens": required_substrings,
            "reason": "file_missing",
        }

    text = _read_text(path)
    missing = [token for token in required_substrings if token not in text]
    return {
        "status": "ok" if not missing else "error",
        "path": str(path),
        "missing_tokens": missing,
        "reason": "" if not missing else "missing_required_tokens",
    }


def _check_regex(
    *,
    path: Path,
    pattern: str,
    label: str,
) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "status": "error",
            "path": str(path),
            "label": label,
            "reason": "file_missing",
        }
    text = _read_text(path)
    matched = re.search(pattern, text, flags=re.MULTILINE) is not None
    return {
        "status": "ok" if matched else "error",
        "path": str(path),
        "label": label,
        "reason": "" if matched else "pattern_not_found",
        "pattern": pattern,
    }


def check_training_data_governance(
    *,
    golden_train_manifest: Path,
    golden_val_manifest: Path,
    auto_retrain_script: Path,
    append_reviewed_script: Path,
    finetune_augmented_script: Path,
    finetune_pretrained_script: Path,
    active_learning_api: Path,
    active_learning_core: Path,
    backfill_helper: Path,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    manifests_ok = True
    golden_train_rows: list[dict[str, str]] = []
    golden_val_rows: list[dict[str, str]] = []
    for label, path in (
        ("golden_train_manifest", golden_train_manifest),
        ("golden_val_manifest", golden_val_manifest),
    ):
        check = {
            "label": label,
            "path": str(path),
            "status": "ok",
            "reason": "",
            "rows": 0,
        }
        if not path.exists() or not path.is_file():
            check["status"] = "error"
            check["reason"] = "manifest_missing"
            manifests_ok = False
        else:
            rows = _read_manifest_rows(path)
            check["rows"] = len(rows)
            if not rows:
                check["status"] = "error"
                check["reason"] = "manifest_empty"
                manifests_ok = False
            if label == "golden_train_manifest":
                golden_train_rows = rows
            else:
                golden_val_rows = rows
        checks.append(check)
        if check["status"] != "ok":
            violations.append(check)

    if manifests_ok:
        train_file_paths = _collect_manifest_keys(golden_train_rows, field_name="file_path")
        val_file_paths = _collect_manifest_keys(golden_val_rows, field_name="file_path")
        train_cache_paths = _collect_manifest_keys(golden_train_rows, field_name="cache_path")
        val_cache_paths = _collect_manifest_keys(golden_val_rows, field_name="cache_path")

        overlap_file_paths = sorted(train_file_paths & val_file_paths)
        overlap_cache_paths = sorted(train_cache_paths & val_cache_paths)
        overlap_check = {
            "label": "golden_train_val_overlap",
            "status": "ok" if not overlap_file_paths and not overlap_cache_paths else "error",
            "file_path_overlap_count": len(overlap_file_paths),
            "cache_path_overlap_count": len(overlap_cache_paths),
            "sample_file_path_overlap": overlap_file_paths[:5],
            "sample_cache_path_overlap": overlap_cache_paths[:5],
        }
        checks.append(overlap_check)
        if overlap_check["status"] != "ok":
            violations.append(overlap_check)

    substring_checks = [
        (
            "auto_retrain_script",
            auto_retrain_script,
            [
                "human_verified",
                "eligible_for_training",
                "--val-manifest",
                "backfill_manifest_cache_paths.py",
                "exit 1",
            ],
        ),
        (
            "append_reviewed_script",
            append_reviewed_script,
            [
                "human_verified",
                "--include-unverified",
                "cache_path",
                "will need preprocess",
            ],
        ),
        (
            "finetune_augmented_script",
            finetune_augmented_script,
            [
                "val_paths",
                "cp not in val_paths",
                "Leakage prevention: removed",
            ],
        ),
        (
            "finetune_pretrained_script",
            finetune_pretrained_script,
            [
                "val_paths",
                "cp not in val_paths",
                "Leakage prevention: removed",
            ],
        ),
        (
            "active_learning_core",
            active_learning_core,
            [
                "eligible_for_training",
                "eligible_count",
                "human_feedback",
            ],
        ),
        (
            "backfill_helper",
            backfill_helper,
            [
                "cache_manifest.csv",
                "missing cache_path after backfill",
                "hashlib.md5",
            ],
        ),
    ]
    for label, path, required_substrings in substring_checks:
        check = {"label": label, **_check_required_substrings(path=path, required_substrings=required_substrings)}
        checks.append(check)
        if check["status"] != "ok":
            violations.append(check)

    regex_checks = [
        (
            "active_learning_api_default_label_source",
            active_learning_api,
            r'label_source:\s*Optional\[str\]\s*=\s*Field\(default="human_feedback"',
        ),
    ]
    for label, path, pattern in regex_checks:
        check = {"label": label, **_check_regex(path=path, pattern=pattern, label=label)}
        checks.append(check)
        if check["status"] != "ok":
            violations.append(check)

    return {
        "status": "ok" if not violations else "error",
        "checks_count": len(checks),
        "violations_count": len(violations),
        "checks": checks,
        "violations": violations,
        "paths": {
            "golden_train_manifest": str(golden_train_manifest),
            "golden_val_manifest": str(golden_val_manifest),
            "auto_retrain_script": str(auto_retrain_script),
            "append_reviewed_script": str(append_reviewed_script),
            "finetune_augmented_script": str(finetune_augmented_script),
            "finetune_pretrained_script": str(finetune_pretrained_script),
            "active_learning_api": str(active_learning_api),
            "active_learning_core": str(active_learning_core),
            "backfill_helper": str(backfill_helper),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check training data governance CI invariants."
    )
    parser.add_argument("--golden-train-manifest", default=DEFAULT_GOLDEN_TRAIN)
    parser.add_argument("--golden-val-manifest", default=DEFAULT_GOLDEN_VAL)
    parser.add_argument("--auto-retrain-script", default=DEFAULT_AUTO_RETRAIN)
    parser.add_argument("--append-reviewed-script", default=DEFAULT_APPEND_REVIEWED)
    parser.add_argument("--finetune-augmented-script", default=DEFAULT_FINETUNE_AUGMENTED)
    parser.add_argument("--finetune-pretrained-script", default=DEFAULT_FINETUNE_PRETRAINED)
    parser.add_argument("--active-learning-api", default=DEFAULT_ACTIVE_LEARNING_API)
    parser.add_argument("--active-learning-core", default=DEFAULT_ACTIVE_LEARNING_CORE)
    parser.add_argument("--backfill-helper", default=DEFAULT_BACKFILL_HELPER)
    parser.add_argument("--output-json", default="")
    return parser


def _write_output_json(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = check_training_data_governance(
        golden_train_manifest=Path(str(args.golden_train_manifest)).expanduser(),
        golden_val_manifest=Path(str(args.golden_val_manifest)).expanduser(),
        auto_retrain_script=Path(str(args.auto_retrain_script)).expanduser(),
        append_reviewed_script=Path(str(args.append_reviewed_script)).expanduser(),
        finetune_augmented_script=Path(str(args.finetune_augmented_script)).expanduser(),
        finetune_pretrained_script=Path(str(args.finetune_pretrained_script)).expanduser(),
        active_learning_api=Path(str(args.active_learning_api)).expanduser(),
        active_learning_core=Path(str(args.active_learning_core)).expanduser(),
        backfill_helper=Path(str(args.backfill_helper)).expanduser(),
    )

    output_json = str(args.output_json or "").strip()
    if output_json:
        _write_output_json(output_json, report)

    print(
        json.dumps(
            {
                "status": report["status"],
                "checks_count": report["checks_count"],
                "violations_count": report["violations_count"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if report["violations"]:
        for violation in report["violations"]:
            print(
                json.dumps(violation, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
