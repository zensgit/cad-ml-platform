from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_json_object(path: Path, kind_label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read {kind_label} json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse {kind_label} json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{kind_label} json must be an object")
    return payload


def is_zeroish(value: Any) -> bool:
    return str(value if value is not None else "").strip() == "0"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def top_nonempty(values: Iterable[Any], limit: int = 3) -> list[str]:
    rows: list[str] = []
    for value in values:
        text = str(value if value is not None else "").strip()
        if not text:
            continue
        rows.append(text)
        if len(rows) >= limit:
            break
    return rows
