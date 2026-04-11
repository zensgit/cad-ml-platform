#!/usr/bin/env python3
"""Shared helpers for top-level eval reporting bundle consumers.

This module only handles discovery and loading — it must NOT own
summary aggregation, metrics computation, HTML rendering, trend
plotting, or weekly markdown generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from scripts.eval_report_data_helpers import load_json_dict
except ImportError:
    from eval_report_data_helpers import load_json_dict  # type: ignore


def load_eval_reporting_bundle(
    history_dir: Path,
    *,
    bundle_json_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load the top-level eval reporting bundle manifest.

    Returns the parsed manifest dict, or None if the file is missing or invalid.
    """
    return load_json_dict(
        bundle_json_path or (history_dir / "eval_reporting_bundle.json")
    )


def load_eval_reporting_assets(
    history_dir: Path,
    *,
    bundle_json_path: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load the top-level bundle and both sub-bundle manifests.

    Returns ``(top_level_bundle, eval_signal_bundle, history_sequence_bundle)``.
    When the top-level bundle exists its sub-bundle path pointers are preferred
    over the default paths.
    """
    top = load_eval_reporting_bundle(history_dir, bundle_json_path=bundle_json_path)

    eval_signal_bundle: Optional[Dict[str, Any]] = None
    history_sequence_bundle: Optional[Dict[str, Any]] = None

    if isinstance(top, dict):
        es_path = str(top.get("eval_signal_bundle_json") or "").strip()
        if es_path:
            eval_signal_bundle = load_json_dict(Path(es_path))

        hs_path = str(top.get("history_sequence_bundle_json") or "").strip()
        if hs_path:
            history_sequence_bundle = load_json_dict(Path(hs_path))

    if eval_signal_bundle is None:
        eval_signal_bundle = load_json_dict(
            history_dir / "eval_signal_reporting_bundle.json"
        )
    if history_sequence_bundle is None:
        history_sequence_bundle = load_json_dict(
            history_dir / "history_sequence_reporting_bundle.json"
        )

    return top, eval_signal_bundle, history_sequence_bundle


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def build_eval_reporting_discovery_context(
    top_level_bundle: Optional[Dict[str, Any]],
    eval_signal_bundle: Optional[Dict[str, Any]],
    history_sequence_bundle: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a discovery context dict from the top-level and sub-bundle manifests.

    This is a pure read-through — no metrics computation occurs here.
    """
    top = top_level_bundle if isinstance(top_level_bundle, dict) else {}
    es = eval_signal_bundle if isinstance(eval_signal_bundle, dict) else {}
    hs = history_sequence_bundle if isinstance(history_sequence_bundle, dict) else {}

    return {
        "available": bool(top),
        "generated_at": _safe_str(top.get("generated_at")),
        "eval_history_dir": _safe_str(top.get("eval_history_dir")),
        "static_report_html": _safe_str(top.get("static_report_html")),
        "interactive_report_html": _safe_str(top.get("interactive_report_html")),
        "plots_dir": _safe_str(top.get("plots_dir")),
        "eval_signal_bundle_json": _safe_str(top.get("eval_signal_bundle_json")),
        "history_sequence_bundle_json": _safe_str(top.get("history_sequence_bundle_json")),
        "eval_signal_report_count": int(es.get("report_count", 0) or 0),
        "eval_signal_surface_kind": _safe_str(es.get("surface_kind")),
        "history_sequence_report_count": int(hs.get("report_count", 0) or 0),
        "history_sequence_surface_kind": _safe_str(hs.get("surface_kind")),
    }
