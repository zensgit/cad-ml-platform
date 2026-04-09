"""Utilities for history-based CAD command sequences.

These helpers provide a thin, reusable layer around HPSketch/DeepCAD-style
`.h5` command vectors so training code, inference code, and diagnostics all
share the same sequence parsing rules.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import h5py

    HAS_H5PY = True
except Exception:  # pragma: no cover - optional dependency
    h5py = None
    HAS_H5PY = False


def load_h5_sequence_array(file_path: str, vec_key: str = "vec") -> Any:
    """Load a raw sequence array from an H5 file."""
    if not HAS_H5PY or h5py is None:
        raise RuntimeError("h5py not available")

    path = Path(file_path).expanduser()
    with h5py.File(path, "r") as handle:
        if vec_key not in handle:
            raise KeyError(f"dataset key not found: {vec_key}")
        return handle[vec_key][()]


def extract_command_tokens(
    sequence_array: Any,
    *,
    command_col: int = 0,
    min_token: int = 0,
) -> List[int]:
    """Extract integer command tokens from a raw H5 vector array."""
    raw_tokens: List[Any]
    if getattr(sequence_array, "ndim", 0) == 1:
        raw_tokens = list(sequence_array.tolist())
    elif getattr(sequence_array, "ndim", 0) >= 2:
        if sequence_array.shape[1] <= int(command_col):
            raise IndexError(
                f"command_col={command_col} out of bounds for vec "
                f"shape={tuple(sequence_array.shape)}"
            )
        raw_tokens = list(sequence_array[:, int(command_col)].tolist())
    else:
        raw_tokens = []

    tokens: List[int] = []
    for value in raw_tokens:
        try:
            token = int(value)
        except Exception:
            continue
        if token < int(min_token):
            continue
        tokens.append(token)
    return tokens


def load_command_tokens_from_h5(
    file_path: str,
    *,
    vec_key: str = "vec",
    command_col: int = 0,
    min_token: int = 0,
) -> List[int]:
    """Load and normalize command tokens from an H5 file."""
    array = load_h5_sequence_array(file_path, vec_key=vec_key)
    return extract_command_tokens(
        array,
        command_col=command_col,
        min_token=min_token,
    )


def truncate_sequence(tokens: Sequence[int], max_length: int) -> List[int]:
    """Keep the most recent commands, matching CAD history semantics."""
    max_length = max(1, int(max_length))
    if len(tokens) <= max_length:
        return list(tokens)
    return list(tokens[-max_length:])


def build_bigram_counts(tokens: Sequence[int]) -> Counter[Tuple[int, int]]:
    """Build adjacent command-pair statistics."""
    if len(tokens) < 2:
        return Counter()
    return Counter(zip(tokens[:-1], tokens[1:]))


def sequence_statistics(tokens: Sequence[int], *, top_k: int = 5) -> Dict[str, Any]:
    """Summarize a command sequence for diagnostics and review."""
    normalized = [int(token) for token in tokens if int(token) >= 0]
    token_counts = Counter(normalized)
    bigram_counts = build_bigram_counts(normalized)
    return {
        "length": len(normalized),
        "unique_commands": len(token_counts),
        "top_commands": token_counts.most_common(max(0, int(top_k))),
        "top_bigrams": bigram_counts.most_common(max(0, int(top_k))),
    }


def discover_h5_files(root_dir: str) -> List[str]:
    """Recursively discover `.h5` files under a root directory."""
    root = Path(root_dir).expanduser()
    if not root.exists():
        return []
    return sorted(str(path) for path in root.rglob("*.h5") if path.is_file())


def build_label_map(labels: Iterable[str]) -> Dict[str, int]:
    """Build a stable label map from an iterable of labels."""
    normalized = sorted({str(label).strip() for label in labels if str(label).strip()})
    return {label: idx for idx, label in enumerate(normalized)}


def iter_h5_files(root_dir: Path | str, recursive: bool = True) -> List[Path]:
    """Collect `.h5` files as ``Path`` objects for compatibility helpers."""
    root = Path(root_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    pattern = "**/*.h5" if recursive else "*.h5"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def load_h5_label_pairs_from_manifest(
    manifest_path: Path | str,
    *,
    h5_col: str = "h5_path",
    label_col: str = "label",
) -> List[Tuple[Path, str]]:
    """Load ``(h5_path, label)`` pairs from a JSON/CSV manifest."""
    path = Path(manifest_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")

    base_dir = path.parent
    pairs: List[Tuple[Path, str]] = []

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list")
        rows = payload
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported manifest suffix: {path.suffix}")

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_h5 = str(row.get(h5_col) or "").strip()
        label = str(row.get(label_col) or "").strip()
        if not raw_h5 or not label:
            continue
        h5_path = Path(raw_h5)
        if not h5_path.is_absolute():
            h5_path = (base_dir / h5_path).resolve()
        pairs.append((h5_path, label))

    return pairs


def read_command_tokens_from_h5(
    file_path: Path | str,
    *,
    vec_key: str = "vec",
    command_col: int = 0,
) -> List[int]:
    """Backward-compatible alias around ``load_command_tokens_from_h5``."""
    return load_command_tokens_from_h5(
        str(Path(file_path).expanduser()),
        vec_key=vec_key,
        command_col=command_col,
    )


def build_prototype_payload(
    samples: Sequence[Tuple[str, Sequence[int]]],
    *,
    top_k: int = 32,
    min_samples_per_label: int = 2,
    smoothing: float = 1e-6,
) -> Dict[str, Any]:
    """Build prototype weights for ``HistorySequenceClassifier``."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if min_samples_per_label <= 0:
        raise ValueError("min_samples_per_label must be > 0")

    per_label_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    per_label_bigram_counts: Dict[str, Counter[Tuple[int, int]]] = defaultdict(Counter)
    per_label_samples: Dict[str, int] = Counter()
    global_counts: Counter[int] = Counter()
    global_bigram_counts: Counter[Tuple[int, int]] = Counter()
    total_tokens = 0
    total_bigrams = 0

    for label, seq in samples:
        label_text = str(label or "").strip()
        if not label_text:
            continue
        tokens: List[int] = []
        for token in seq:
            try:
                token_id = int(token)
            except Exception:
                continue
            if token_id >= 0:
                tokens.append(token_id)
        if not tokens:
            continue
        per_label_samples[label_text] += 1
        token_counts = Counter(tokens)
        per_label_counts[label_text].update(token_counts)
        global_counts.update(token_counts)
        total_tokens += len(tokens)

        if len(tokens) >= 2:
            bigrams = list(zip(tokens[:-1], tokens[1:]))
            bigram_counts = Counter(bigrams)
            per_label_bigram_counts[label_text].update(bigram_counts)
            global_bigram_counts.update(bigram_counts)
            total_bigrams += len(bigrams)

    valid_labels = sorted(
        label
        for label, count in per_label_samples.items()
        if int(count) >= min_samples_per_label
    )

    if total_tokens <= 0 or not valid_labels:
        return {
            "version": "0.2.0",
            "notes": "No valid labeled samples for prototype generation.",
            "labels": {},
            "meta": {
                "total_samples": int(sum(per_label_samples.values())),
                "total_tokens": int(total_tokens),
                "total_bigrams": int(total_bigrams),
                "min_samples_per_label": int(min_samples_per_label),
                "top_k": int(top_k),
            },
        }

    label_payload: Dict[str, Any] = {}
    total_samples = float(sum(per_label_samples[label] for label in valid_labels))
    eps = max(float(smoothing), 1e-12)

    for label in valid_labels:
        label_counter = per_label_counts[label]
        label_bigram_counter = per_label_bigram_counts[label]
        label_token_total = float(sum(label_counter.values())) or 1.0
        label_bigram_total = float(sum(label_bigram_counter.values())) or 1.0
        label_prior = float(per_label_samples[label]) / total_samples
        token_scores: List[Tuple[int, float]] = []
        bigram_scores: List[Tuple[Tuple[int, int], float]] = []

        for token, token_count in label_counter.items():
            p_label = float(token_count) / label_token_total
            p_global = float(global_counts[token]) / float(total_tokens)
            score = math.log((p_label + eps) / (p_global + eps))
            token_scores.append((int(token), float(score)))

        ranked_tokens = [item for item in token_scores if item[1] > 0] or token_scores
        ranked_tokens.sort(key=lambda item: item[1], reverse=True)
        ranked_tokens = ranked_tokens[:top_k]

        for bigram, bigram_count in label_bigram_counter.items():
            p_label = float(bigram_count) / label_bigram_total
            p_global = float(global_bigram_counts[bigram]) / float(
                max(1, total_bigrams)
            )
            score = math.log((p_label + eps) / (p_global + eps))
            bigram_scores.append((bigram, float(score)))

        ranked_bigrams = [item for item in bigram_scores if item[1] > 0] or bigram_scores
        ranked_bigrams.sort(key=lambda item: item[1], reverse=True)
        ranked_bigrams = ranked_bigrams[:top_k]

        label_payload[label] = {
            "bias": round(math.log(label_prior + eps), 6),
            "sample_count": int(per_label_samples[label]),
            "token_weights": {
                str(int(token)): round(float(score), 6)
                for token, score in ranked_tokens
            },
            "bigram_weights": {
                f"{int(pair[0])},{int(pair[1])}": round(float(score), 6)
                for pair, score in ranked_bigrams
            },
        }

    return {
        "version": "0.2.0",
        "notes": "Auto-generated history sequence prototypes from labeled .h5 samples.",
        "labels": label_payload,
        "meta": {
            "total_samples": int(sum(per_label_samples.values())),
            "valid_labels": int(len(valid_labels)),
            "unique_tokens": int(len(global_counts)),
            "total_tokens": int(total_tokens),
            "unique_bigrams": int(len(global_bigram_counts)),
            "total_bigrams": int(total_bigrams),
            "min_samples_per_label": int(min_samples_per_label),
            "top_k": int(top_k),
        },
    }


def macro_f1(expected: Iterable[str], predicted: Iterable[str]) -> float:
    """Compute macro-F1 from expected and predicted labels."""
    expected_list = [str(item) for item in expected]
    predicted_list = [str(item) for item in predicted]
    labels = sorted(set(expected_list) | set(predicted_list))
    if not labels:
        return 0.0

    f1_sum = 0.0
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for exp, pred in zip(expected_list, predicted_list):
            if exp == label and pred == label:
                tp += 1
            elif exp != label and pred == label:
                fp += 1
            elif exp == label and pred != label:
                fn += 1
        denom = (2 * tp) + fp + fn
        f1_sum += (2.0 * tp / denom) if denom else 0.0
    return f1_sum / float(len(labels))
