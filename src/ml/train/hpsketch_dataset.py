"""Dataset helpers for HPSketch-style command sequences."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.ml.history_sequence_tools import (
    HAS_H5PY,
    build_label_map,
    discover_h5_files,
    iter_h5_files,
    load_command_tokens_from_h5,
    truncate_sequence,
)


@dataclass(frozen=True)
class HPSketchSequenceRecord:
    file_path: str
    label: Optional[str] = None


class HPSketchDataset(Dataset):
    """Dataset that loads command sequences from HPSketch `.h5` files."""

    def __init__(
        self,
        root_dir: str,
        vec_key: str = "vec",
        command_col: int = 0,
        seq_max_len: int = 0,
        min_seq_len: int = 1,
        recursive: bool = True,
    ) -> None:
        if not HAS_H5PY:
            raise RuntimeError("h5py is required for HPSketchDataset")
        if min_seq_len < 1:
            raise ValueError("min_seq_len must be >= 1")
        if command_col < 0:
            raise ValueError("command_col must be >= 0")
        if seq_max_len < 0:
            raise ValueError("seq_max_len must be >= 0")

        self.root_dir = str(root_dir)
        self.vec_key = str(vec_key)
        self.command_col = int(command_col)
        self.seq_max_len = int(seq_max_len)
        self.min_seq_len = int(min_seq_len)
        self.recursive = bool(recursive)

        root = Path(self.root_dir).expanduser()
        if not root.exists():
            self.files: List[Path] = []
            return

        self.files = iter_h5_files(root, recursive=self.recursive)

    def __len__(self) -> int:
        return len(self.files)

    def _read_tokens(self, path: Path) -> List[int]:
        tokens = load_command_tokens_from_h5(
            str(path),
            vec_key=self.vec_key,
            command_col=self.command_col,
        )
        if self.seq_max_len > 0:
            tokens = truncate_sequence(tokens, self.seq_max_len)
        return tokens

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        tokens = self._read_tokens(path)
        actual_len = len(tokens)
        if actual_len == 0:
            tokens = [0]
            actual_len = 1
        if len(tokens) < self.min_seq_len:
            tokens = tokens[:] + ([0] * (self.min_seq_len - len(tokens)))

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "length": actual_len,
            "file_path": str(path),
        }


class HPSketchNextTokenDataset(Dataset):
    """Self-supervised dataset for next-command prediction."""

    def __init__(self, base_dataset: HPSketchDataset) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        sample = self.base[idx]
        tokens: torch.Tensor = sample["tokens"]
        sample_len = int(sample.get("length", int(tokens.numel())))
        valid_len = max(1, min(sample_len, int(tokens.numel())))
        valid_tokens = tokens[:valid_len]
        if valid_tokens.numel() <= 1:
            input_tokens = valid_tokens[:1]
            target = valid_tokens[0] if valid_tokens.numel() > 0 else torch.tensor(0)
        else:
            input_tokens = valid_tokens[:-1]
            target = valid_tokens[-1]

        features = {
            "tokens": input_tokens.clone(),
            "length": int(input_tokens.numel()),
            "file_path": sample.get("file_path", ""),
        }
        return features, target.long()


def collate_hpsketch_sequences(
    batch: List[Dict[str, Any]],
    pad_token: int = 0,
) -> Dict[str, Any]:
    """Pad variable-length HPSketch token sequences."""
    batch_size = len(batch)
    raw_lengths = [int(item.get("length", 0)) for item in batch]
    max_len = max(max(raw_lengths, default=0), 1)
    tokens = torch.full((batch_size, max_len), int(pad_token), dtype=torch.long)
    lengths: List[int] = []
    file_paths: List[str] = []

    for idx, item in enumerate(batch):
        seq = item.get("tokens", torch.zeros(0, dtype=torch.long)).view(-1).long()
        declared_len = int(item.get("length", int(seq.numel())))
        keep = min(max(0, declared_len), int(seq.numel()), max_len)
        if keep > 0:
            tokens[idx, :keep] = seq[:keep]
        lengths.append(keep)
        file_paths.append(str(item.get("file_path", "")))

    return {
        "tokens": tokens,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "file_paths": file_paths,
    }


def collate_hpsketch_next_token_batch(
    batch: List[Tuple[Dict[str, Any], torch.Tensor]],
    pad_token: int = 0,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Collate function for next-token training."""
    features = [item[0] for item in batch]
    targets = torch.stack([item[1].long() for item in batch], dim=0)
    return collate_hpsketch_sequences(features, pad_token=pad_token), targets


def _resolve_path(base_dir: Path, value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _load_manifest_records(
    manifest_path: str,
    *,
    path_column: str = "file_path",
    label_column: str = "label",
) -> List[HPSketchSequenceRecord]:
    path = Path(manifest_path).expanduser().resolve()
    base_dir = path.parent
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            records = [
                HPSketchSequenceRecord(
                    file_path=_resolve_path(base_dir, row.get(path_column, "")),
                    label=str(row.get(label_column, "")).strip() or None,
                )
                for row in reader
            ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])
        records = [
            HPSketchSequenceRecord(
                file_path=_resolve_path(base_dir, row.get(path_column, "")),
                label=str(row.get(label_column, "")).strip() or None,
            )
            for row in rows
            if isinstance(row, dict)
        ]
    elif suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        records = [
            HPSketchSequenceRecord(
                file_path=_resolve_path(base_dir, row.get(path_column, "")),
                label=str(row.get(label_column, "")).strip() or None,
            )
            for row in rows
            if isinstance(row, dict)
        ]
    else:
        raise ValueError(f"Unsupported manifest suffix: {path.suffix}")

    return [record for record in records if record.file_path]


class HPSketchSequenceDataset(Dataset):
    """Load HPSketch-style `.h5` command vectors for training."""

    def __init__(
        self,
        *,
        manifest_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        vec_key: str = "vec",
        command_col: int = 0,
        max_sequence_length: int = 512,
        padding_idx: int = 0,
        label_map: Optional[Dict[str, int]] = None,
        path_column: str = "file_path",
        label_column: str = "label",
    ) -> None:
        if not manifest_path and not root_dir:
            raise ValueError("manifest_path or root_dir is required")

        self.manifest_path = manifest_path
        self.root_dir = root_dir
        self.vec_key = str(vec_key)
        self.command_col = max(0, int(command_col))
        self.max_sequence_length = max(1, int(max_sequence_length))
        self.padding_idx = max(0, int(padding_idx))

        if manifest_path:
            self.records = _load_manifest_records(
                manifest_path,
                path_column=path_column,
                label_column=label_column,
            )
        else:
            assert root_dir is not None
            self.records = [
                HPSketchSequenceRecord(file_path=path)
                for path in discover_h5_files(root_dir)
            ]

        labels = [record.label for record in self.records if record.label]
        self.label_map = dict(label_map or build_label_map(labels))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        record = self.records[index]
        tokens = load_command_tokens_from_h5(
            record.file_path,
            vec_key=self.vec_key,
            command_col=self.command_col,
        )
        truncated = truncate_sequence(tokens, self.max_sequence_length)
        sequence = torch.tensor(truncated, dtype=torch.long)
        length = int(sequence.numel())

        label_value = -1
        if record.label and record.label in self.label_map:
            label_value = int(self.label_map[record.label])

        sample = {
            "input_ids": sequence,
            "length": length,
            "file_path": record.file_path,
            "label_name": record.label,
        }
        return sample, torch.tensor(label_value, dtype=torch.long)

    def num_classes(self) -> int:
        return len(self.label_map)

    @staticmethod
    def collate_fn(
        batch: Sequence[Tuple[Dict[str, Any], torch.Tensor]],
        *,
        padding_idx: int = 0,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        if not batch:
            empty = {
                "input_ids": torch.zeros((0, 0), dtype=torch.long),
                "lengths": torch.zeros(0, dtype=torch.long),
                "file_path": [],
                "label_name": [],
            }
            return empty, torch.zeros(0, dtype=torch.long)

        samples, labels = zip(*batch)
        lengths = torch.tensor(
            [int(sample["length"]) for sample in samples],
            dtype=torch.long,
        )
        max_length = int(lengths.max().item()) if len(lengths) else 0
        input_ids = torch.full(
            (len(samples), max_length),
            fill_value=int(padding_idx),
            dtype=torch.long,
        )
        for idx, sample in enumerate(samples):
            seq = sample["input_ids"]
            if seq.numel():
                input_ids[idx, : seq.numel()] = seq

        merged = {
            "input_ids": input_ids,
            "lengths": lengths,
            "file_path": [sample["file_path"] for sample in samples],
            "label_name": [sample.get("label_name") for sample in samples],
        }
        return merged, torch.stack(list(labels))
