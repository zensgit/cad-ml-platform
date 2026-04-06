#!/usr/bin/env python3
"""Shared raw data-loading helpers for evaluation reports."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


def load_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _scan_typed_history(history_dir: Path) -> Dict[str, list[Dict[str, Any]]]:
    if not history_dir.exists():
        return {"combined": [], "ocr": [], "hybrid_blind": []}

    buckets: Dict[str, list[Dict[str, Any]]] = {"combined": [], "ocr": [], "hybrid_blind": []}
    for path in sorted(history_dir.glob("*.json"), reverse=True):
        payload = load_json_dict(path)
        if not payload:
            continue
        report_type = str(payload.get("type") or "").strip()
        is_combined = (
            report_type == "combined"
            or "combined" in payload
            or path.name.endswith("_combined.json")
        )
        if is_combined:
            payload["_file"] = path.name
            buckets["combined"].append(payload)
            continue

        is_legacy_ocr = report_type == "" and "metrics" in payload and "history_metrics" not in payload
        if report_type == "ocr" or is_legacy_ocr:
            payload["_file"] = path.name
            buckets["ocr"].append(payload)
            continue

        if report_type == "hybrid_blind":
            payload["_file"] = path.name
            buckets["hybrid_blind"].append(payload)
    return buckets


def load_combined_history(history_dir: Path) -> list[Dict[str, Any]]:
    return list(_scan_typed_history(history_dir)["combined"])


def load_ocr_history(history_dir: Path) -> list[Dict[str, Any]]:
    return list(_scan_typed_history(history_dir)["ocr"])


def load_hybrid_blind_history(history_dir: Path) -> list[Dict[str, Any]]:
    return list(_scan_typed_history(history_dir)["hybrid_blind"])


def encode_image_base64(image_path: Path) -> Optional[str]:
    if not image_path.exists():
        return None

    try:
        data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except Exception:
        return None

    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(suffix, "image/png")
    return f"data:{mime_type};base64,{data}"


def load_plot_base64_assets(
    plots_dir: Path,
    plot_map: Mapping[str, str],
) -> Dict[str, str]:
    if not plots_dir.exists():
        return {}

    assets: MutableMapping[str, str] = {}
    for key, filename in plot_map.items():
        encoded = encode_image_base64(plots_dir / filename)
        if encoded:
            assets[key] = encoded
    return dict(assets)
