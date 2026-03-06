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
    build_label_map,
    discover_h5_files,
    load_command_tokens_from_h5,
    truncate_sequence,
)


@dataclass(frozen=True)
class HPSketchSequenceRecord:
    file_path: str
    label: Optional[str] = None


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
