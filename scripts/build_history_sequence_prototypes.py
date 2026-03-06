#!/usr/bin/env python3
"""Build history-sequence prototype weights from labeled `.h5` files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.filename_classifier import FilenameClassifier  # noqa: E402
from src.ml.history_sequence_tools import (  # noqa: E402
    build_prototype_payload,
    iter_h5_files,
    load_h5_label_pairs_from_manifest,
    read_command_tokens_from_h5,
)

_SIDE_SUFFIX_RE = re.compile(r"_\d+$")


def _normalize_h5_stem_for_filename(stem: str) -> str:
    text = str(stem or "").strip()
    if not text:
        return text
    return _SIDE_SUFFIX_RE.sub("", text)


def _resolve_label_from_h5_name(
    path: Path,
    *,
    classifier: FilenameClassifier,
    min_confidence: float,
) -> Optional[str]:
    stem = _normalize_h5_stem_for_filename(path.stem)
    pseudo_name = f"{stem}.dxf"
    prediction = classifier.predict(pseudo_name)
    if prediction.get("label") and float(prediction.get("confidence", 0.0)) >= float(
        min_confidence
    ):
        return str(prediction["label"])
    return None


def _collect_labeled_h5(
    *,
    manifest_path: Optional[Path],
    h5_dir: Optional[Path],
    label_source: str,
    manifest_h5_col: str,
    manifest_label_col: str,
    synonyms_path: Optional[str],
    filename_min_conf: float,
    max_files: int,
    recursive: bool,
) -> List[Tuple[Path, str]]:
    if label_source == "manifest":
        if manifest_path is None:
            raise ValueError("--label-source=manifest requires --manifest")
        pairs = load_h5_label_pairs_from_manifest(
            manifest_path,
            h5_col=manifest_h5_col,
            label_col=manifest_label_col,
        )
    elif label_source == "filename":
        if h5_dir is None:
            raise ValueError("--label-source=filename requires --h5-dir")
        classifier = FilenameClassifier(synonyms_path=synonyms_path)
        pairs = []
        for h5_path in iter_h5_files(h5_dir, recursive=recursive):
            label = _resolve_label_from_h5_name(
                h5_path,
                classifier=classifier,
                min_confidence=filename_min_conf,
            )
            if label:
                pairs.append((h5_path, label))
    else:  # pragma: no cover
        raise ValueError(f"Unsupported label_source: {label_source}")

    return pairs[:max_files] if max_files > 0 else pairs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="", help="JSON/CSV manifest path")
    parser.add_argument("--h5-dir", default="", help="Directory containing .h5 files")
    parser.add_argument(
        "--label-source",
        choices=["manifest", "filename"],
        default="manifest",
        help="How to get labels for each .h5 sample",
    )
    parser.add_argument("--manifest-h5-col", default="h5_path")
    parser.add_argument("--manifest-label-col", default="label")
    parser.add_argument("--synonyms-path", default="")
    parser.add_argument("--filename-min-conf", type=float, default=0.8)
    parser.add_argument("--vec-key", default="vec")
    parser.add_argument("--command-col", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-samples-per-label", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument(
        "--output",
        default="data/knowledge/history_sequence_prototypes_template.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_path = Path(args.manifest).expanduser() if args.manifest else None
    h5_dir = Path(args.h5_dir).expanduser() if args.h5_dir else None
    output_path = Path(args.output).expanduser()

    pairs = _collect_labeled_h5(
        manifest_path=manifest_path,
        h5_dir=h5_dir,
        label_source=str(args.label_source),
        manifest_h5_col=str(args.manifest_h5_col),
        manifest_label_col=str(args.manifest_label_col),
        synonyms_path=(str(args.synonyms_path).strip() or None),
        filename_min_conf=float(args.filename_min_conf),
        max_files=max(0, int(args.max_files)),
        recursive=not bool(args.no_recursive),
    )

    labeled_sequences: List[Tuple[str, List[int]]] = []
    skipped_missing = 0
    skipped_read_error = 0
    skipped_empty = 0

    for h5_path, label in pairs:
        if not h5_path.exists():
            skipped_missing += 1
            continue
        try:
            tokens = read_command_tokens_from_h5(
                h5_path,
                vec_key=str(args.vec_key),
                command_col=int(args.command_col),
            )
        except Exception:
            skipped_read_error += 1
            continue
        if not tokens:
            skipped_empty += 1
            continue
        labeled_sequences.append((str(label), tokens))

    payload = build_prototype_payload(
        labeled_sequences,
        top_k=max(1, int(args.top_k)),
        min_samples_per_label=max(1, int(args.min_samples_per_label)),
    )
    payload["meta"] = dict(payload.get("meta") or {})
    payload["meta"].update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "label_source": str(args.label_source),
            "input_pairs": len(pairs),
            "used_samples": len(labeled_sequences),
            "skipped_missing": int(skipped_missing),
            "skipped_read_error": int(skipped_read_error),
            "skipped_empty": int(skipped_empty),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    label_counter = Counter(label for label, _ in labeled_sequences)
    summary = {
        "output": str(output_path),
        "input_pairs": len(pairs),
        "used_samples": len(labeled_sequences),
        "label_count": len(label_counter),
        "top_labels": label_counter.most_common(10),
        "skipped_missing": skipped_missing,
        "skipped_read_error": skipped_read_error,
        "skipped_empty": skipped_empty,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if labeled_sequences else 2


if __name__ == "__main__":
    raise SystemExit(main())
