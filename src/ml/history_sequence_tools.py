"""Utilities for history-based CAD command sequences.

These helpers provide a thin, reusable layer around HPSketch/DeepCAD-style
`.h5` command vectors so training code, inference code, and diagnostics all
share the same sequence parsing rules.
"""

from __future__ import annotations

from collections import Counter
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
