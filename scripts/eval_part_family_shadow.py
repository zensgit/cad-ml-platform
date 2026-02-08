#!/usr/bin/env python3
"""Evaluate PartClassifier shadow-only `part_family*` fields via local TestClient.

This script is intentionally lightweight and does not require a running server.
It calls `/api/v1/analyze/` in-process and writes a CSV suitable for offline review.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root on sys.path for TestClient usage
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep ezdxf cache out of $HOME by default (helpful for sandboxed environments).
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

# Reduce noisy multipart parser warnings when TestClient uploads files.
logging.getLogger("python_multipart.multipart").setLevel(logging.ERROR)
logging.getLogger("python_multipart").setLevel(logging.ERROR)

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"FastAPI TestClient import failed: {exc}")

try:
    from src.main import app
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import app: {exc}")


def _collect_files(root: Path, suffixes: List[str]) -> List[Path]:
    wanted = {s.lower().lstrip(".") for s in suffixes if s}
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower().lstrip(".")
        if suf in wanted:
            out.append(p)
    return out


def _safe_get(d: Dict[str, Any], key: str) -> Any:
    v = d.get(key)
    return v if v is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PartClassifier shadow-only part_family fields (local TestClient)"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing DXF/DWG files")
    parser.add_argument("--suffixes", default="dxf", help="Comma-separated file suffixes (default: dxf)")
    parser.add_argument("--output-csv", default="/tmp/part_family_shadow.csv", help="Output CSV path")
    parser.add_argument("--max-files", type=int, default=200, help="Max files to evaluate")
    parser.add_argument("--seed", type=int, default=22, help="Shuffle seed")
    parser.add_argument("--provider-name", default=os.getenv("PART_CLASSIFIER_PROVIDER_NAME", "v16"))
    parser.add_argument("--timeout-seconds", default=os.getenv("PART_CLASSIFIER_PROVIDER_TIMEOUT_SECONDS", "2.0"))
    parser.add_argument("--max-mb", default=os.getenv("PART_CLASSIFIER_PROVIDER_MAX_MB", "10.0"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    suffixes = [s.strip() for s in (args.suffixes or "").split(",") if s.strip()]
    files = _collect_files(input_dir, suffixes)
    if not files:
        raise SystemExit(f"No files found under {input_dir} (suffixes={suffixes})")

    random.seed(args.seed)
    random.shuffle(files)
    files = files[: args.max_files]

    # Enable shadow-only provider integration for this run (safe; does not override part_type).
    os.environ["PART_CLASSIFIER_PROVIDER_ENABLED"] = "true"
    os.environ["PART_CLASSIFIER_PROVIDER_NAME"] = str(args.provider_name).strip() or "v16"
    os.environ["PART_CLASSIFIER_PROVIDER_TIMEOUT_SECONDS"] = str(args.timeout_seconds)
    os.environ["PART_CLASSIFIER_PROVIDER_MAX_MB"] = str(args.max_mb)

    client = TestClient(app)
    options = {"extract_features": True, "classify_parts": True}

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    family_error_counts: Counter[str] = Counter()

    for item in files:
        payload = item.read_bytes()

        resp = client.post(
            "/api/v1/analyze/",
            files={"file": (item.name, payload, f"application/{item.suffix.lstrip('.')}")},
            data={"options": json.dumps(options)},
            headers={"x-api-key": os.getenv("API_KEY", "test")},
        )
        if resp.status_code != 200:
            rows.append(
                {
                    "file": item.name,
                    "http_status": resp.status_code,
                    "error": resp.text,
                }
            )
            status_counts["http_error"] += 1
            continue

        data = resp.json()
        classification = data.get("results", {}).get("classification", {}) or {}

        pred = classification.get("part_classifier_prediction", {}) or {}
        pred_status = str(pred.get("status") or "")
        status_counts[pred_status or "missing"] += 1

        family_error = classification.get("part_family_error", {}) or {}
        family_error_code = str(family_error.get("code") or "")
        if family_error_code:
            family_error_counts[family_error_code] += 1

        rows.append(
            {
                "file": item.name,
                "part_type": _safe_get(classification, "part_type"),
                "confidence": _safe_get(classification, "confidence"),
                "confidence_source": _safe_get(classification, "confidence_source"),
                "rule_version": _safe_get(classification, "rule_version"),
                "fine_part_type": _safe_get(classification, "fine_part_type"),
                "fine_confidence": _safe_get(classification, "fine_confidence"),
                "fine_source": _safe_get(classification, "fine_source"),
                "part_classifier_status": _safe_get(pred, "status"),
                "part_classifier_label": _safe_get(pred, "label"),
                "part_classifier_confidence": _safe_get(pred, "confidence"),
                "part_classifier_provider": _safe_get(pred, "provider"),
                "part_classifier_error": _safe_get(pred, "error"),
                "part_family": _safe_get(classification, "part_family"),
                "part_family_confidence": _safe_get(classification, "part_family_confidence"),
                "part_family_source": _safe_get(classification, "part_family_source"),
                "part_family_model_version": _safe_get(classification, "part_family_model_version"),
                "part_family_needs_review": _safe_get(classification, "part_family_needs_review"),
                "part_family_review_reason": _safe_get(classification, "part_family_review_reason"),
                "part_family_top2": json.dumps(
                    classification.get("part_family_top2", None),
                    ensure_ascii=False,
                ),
                "part_family_error_code": family_error_code,
                "part_family_error_message": str(family_error.get("message") or ""),
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            raise SystemExit("No rows produced")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_path}")
    print("part_classifier_status_counts:", dict(status_counts))
    print("part_family_error_code_counts:", dict(family_error_counts))


if __name__ == "__main__":
    main()

