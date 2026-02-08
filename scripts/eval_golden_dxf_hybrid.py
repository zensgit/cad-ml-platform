#!/usr/bin/env python3
"""Evaluate HybridClassifier behavior on a small golden DXF manifest.

This is intentionally model-light:
- Uses synthetic DXF bytes generated via ezdxf (no committed *.dxf files).
- Focuses on FilenameClassifier + TitleBlockClassifier behavior and fusion rules.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_ezdxf_cache_dir() -> None:
    # ezdxf may try to write to ~/.cache; in sandboxed environments this can fail.
    # Point it at a writable temp directory.
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")


def _build_synthetic_dxf_bytes(titleblock_texts: List[str]) -> bytes:
    _ensure_ezdxf_cache_dir()
    import ezdxf  # noqa: WPS433

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Border rectangle defines a stable bbox (0..1000), so titleblock extraction
    # can use right-bottom region heuristics.
    msp.add_line((0, 0), (1000, 0))
    msp.add_line((1000, 0), (1000, 1000))
    msp.add_line((1000, 1000), (0, 1000))
    msp.add_line((0, 1000), (0, 0))

    if titleblock_texts:
        x = 800
        y0 = 200
        for idx, text in enumerate(titleblock_texts):
            msp.add_text(
                str(text),
                dxfattribs={
                    "height": 10,
                    "insert": (x, y0 - (idx * 20)),
                },
            )

    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def _macro_f1(expected: List[str], predicted: List[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    if not labels:
        return 0.0

    f1s: List[float] = []
    for label in labels:
        tp = sum(1 for e, p in zip(expected, predicted) if e == label and p == label)
        fp = sum(1 for e, p in zip(expected, predicted) if e != label and p == label)
        fn = sum(1 for e, p in zip(expected, predicted) if e == label and p != label)
        denom = (2 * tp) + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


@dataclass
class CaseResult:
    case_id: str
    filename: str
    expected_label: str
    expected_source: Optional[str]
    predicted_label: Optional[str]
    predicted_source: Optional[str]
    confidence: float
    ok: bool
    detail: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "id": self.case_id,
            "filename": self.filename,
            "expected_label": self.expected_label,
            "expected_source": self.expected_source,
            "predicted_label": self.predicted_label,
            "predicted_source": self.predicted_source,
            "confidence": round(float(self.confidence or 0.0), 6),
            "ok": "Y" if self.ok else "N",
            "detail": self.detail or "",
        }


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    return payload


def _default_output_dir() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("reports") / "experiments" / date_str / "golden_dxf_hybrid_eval"


def _apply_eval_env_defaults() -> None:
    # Defaults chosen to keep evaluation deterministic and dependency-light.
    os.environ.setdefault("HYBRID_CLASSIFIER_ENABLED", "true")
    os.environ.setdefault("FILENAME_CLASSIFIER_ENABLED", "true")
    os.environ.setdefault("GRAPH2D_ENABLED", "false")
    os.environ.setdefault("PROCESS_FEATURES_ENABLED", "false")
    os.environ.setdefault("TITLEBLOCK_ENABLED", "true")
    os.environ.setdefault("TITLEBLOCK_OVERRIDE_ENABLED", "true")
    os.environ.setdefault("TITLEBLOCK_MIN_CONF", "0.6")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="tests/golden/golden_dxf_hybrid_cases.json",
        help="Path to golden manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for CSV/JSON reports (default: reports/experiments/<date>/golden_dxf_hybrid_eval)",
    )
    parser.add_argument(
        "--low-conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for low-confidence rate",
    )
    args = parser.parse_args(argv)

    _apply_eval_env_defaults()

    manifest_path = Path(args.manifest)
    cases = _load_manifest(manifest_path)

    try:
        from src.ml.hybrid_classifier import HybridClassifier
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to import HybridClassifier: {exc}", file=sys.stderr)
        return 2

    clf = HybridClassifier()

    started_at = time.perf_counter()
    results: List[CaseResult] = []
    expected_labels: List[str] = []
    predicted_labels: List[str] = []
    low_conf_count = 0

    for case in cases:
        case_id = str(case.get("id") or "")
        filename = str(case.get("filename") or "")
        expected_label = str(case.get("expected_label") or "")
        expected_source = case.get("expected_source")
        expected_source = str(expected_source) if expected_source else None
        titleblock_texts = case.get("titleblock_texts") or []

        file_bytes = _build_synthetic_dxf_bytes(list(titleblock_texts))
        payload = clf.classify(filename=filename, file_bytes=file_bytes).to_dict()

        pred_label = payload.get("label")
        pred_source = payload.get("source")
        conf = float(payload.get("confidence", 0.0) or 0.0)
        ok = (pred_label == expected_label) and (
            expected_source is None or pred_source == expected_source
        )
        detail = None
        if not ok:
            detail = json.dumps(
                {
                    "decision_path": payload.get("decision_path"),
                    "filename_pred": payload.get("filename_prediction"),
                    "titleblock_pred": payload.get("titleblock_prediction"),
                    "graph2d_pred": payload.get("graph2d_prediction"),
                },
                ensure_ascii=False,
            )
        results.append(
            CaseResult(
                case_id=case_id,
                filename=filename,
                expected_label=expected_label,
                expected_source=expected_source,
                predicted_label=pred_label,
                predicted_source=pred_source,
                confidence=conf,
                ok=ok,
                detail=detail,
            )
        )
        expected_labels.append(expected_label)
        predicted_labels.append(str(pred_label or ""))
        if conf < float(args.low_conf_threshold):
            low_conf_count += 1

    duration_s = time.perf_counter() - started_at
    passed = sum(1 for r in results if r.ok)
    total = len(results)
    accuracy = (passed / total) if total else 0.0
    macro_f1 = _macro_f1(expected_labels, predicted_labels)
    low_conf_rate = (low_conf_count / total) if total else 0.0

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "filename",
                "expected_label",
                "expected_source",
                "predicted_label",
                "predicted_source",
                "confidence",
                "ok",
                "detail",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())

    summary = {
        "manifest": str(manifest_path),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "low_conf_threshold": float(args.low_conf_threshold),
        "low_conf_rate": round(low_conf_rate, 6),
        "duration_seconds": round(duration_s, 3),
        "env": {
            "TITLEBLOCK_ENABLED": os.getenv("TITLEBLOCK_ENABLED"),
            "TITLEBLOCK_OVERRIDE_ENABLED": os.getenv("TITLEBLOCK_OVERRIDE_ENABLED"),
            "TITLEBLOCK_MIN_CONF": os.getenv("TITLEBLOCK_MIN_CONF"),
            "GRAPH2D_ENABLED": os.getenv("GRAPH2D_ENABLED"),
            "PROCESS_FEATURES_ENABLED": os.getenv("PROCESS_FEATURES_ENABLED"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
