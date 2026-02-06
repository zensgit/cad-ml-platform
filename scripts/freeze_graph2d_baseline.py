#!/usr/bin/env python3
"""Freeze a Graph2D checkpoint into a versioned baseline bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def try_git_sha() -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or None
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_eval_summary(metrics_csv: Optional[Path]) -> Dict[str, Any]:
    if metrics_csv is None or not metrics_csv.exists():
        return {}
    try:
        with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    overall = None
    for row in rows:
        if (row.get("label_cn") or "").strip() == "__overall__":
            overall = row
            break
    if overall is None:
        return {}
    return {
        "accuracy": _to_float(overall.get("accuracy")),
        "top2_accuracy": _to_float(overall.get("top2_accuracy")),
        "macro_f1": _to_float(overall.get("macro_f1")),
        "weighted_f1": _to_float(overall.get("weighted_f1")),
        "sample_count": _to_float(overall.get("total")),
    }


def freeze_baseline(
    checkpoint: Path,
    output_dir: Path,
    baseline_name: str,
    manifest: str = "",
    notes: str = "",
    metrics_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    baseline_id = f"{baseline_name}_{timestamp}"
    baseline_dir = output_dir / baseline_id
    baseline_dir.mkdir(parents=True, exist_ok=False)

    source_sha = compute_sha256(checkpoint)
    target_checkpoint = baseline_dir / checkpoint.name
    shutil.copy2(checkpoint, target_checkpoint)
    frozen_sha = compute_sha256(target_checkpoint)

    summary = load_eval_summary(metrics_csv)
    metadata = {
        "baseline_id": baseline_id,
        "created_at_utc": timestamp,
        "checkpoint_file": target_checkpoint.name,
        "source_checkpoint": str(checkpoint),
        "source_sha256": source_sha,
        "frozen_sha256": frozen_sha,
        "file_size_bytes": target_checkpoint.stat().st_size,
        "manifest": manifest,
        "metrics_csv": str(metrics_csv) if metrics_csv else "",
        "metrics_summary": summary,
        "notes": notes,
        "git_commit": try_git_sha(),
    }
    metadata_path = baseline_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "baseline_id": baseline_id,
        "baseline_dir": str(baseline_dir),
        "checkpoint": str(target_checkpoint),
        "metadata": str(metadata_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Graph2D baseline checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="models/graph2d_merged_latest.pth",
        help="Source Graph2D checkpoint path.",
    )
    parser.add_argument(
        "--name",
        default="graph2d_baseline",
        help="Baseline name prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/baselines/graph2d",
        help="Directory where baseline bundles are stored.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Training/eval manifest path for traceability.",
    )
    parser.add_argument(
        "--metrics-csv",
        default="",
        help="Optional eval metrics CSV; __overall__ row will be captured.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional operator notes saved into metadata.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else None
    result = freeze_baseline(
        checkpoint=checkpoint,
        output_dir=output_dir,
        baseline_name=args.name,
        manifest=args.manifest,
        notes=args.notes,
        metrics_csv=metrics_csv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
