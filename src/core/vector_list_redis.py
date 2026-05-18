from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Optional

from src.core.similarity import extract_vector_label_contract


async def list_vectors_redis(
    client: Any,
    offset: int,
    limit: int,
    scan_limit: int,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
    *,
    item_cls: type[Any],
    response_cls: type[Any],
    matches_label_filters_fn: Callable[..., bool],
    extract_label_contract_fn: Callable[[dict[str, Any]], dict[str, Any]] = (
        extract_vector_label_contract
    ),
    json_loads_fn: Callable[[Any], Any] = json.loads,
) -> Any:
    items: list[Any] = []
    matched_total = 0
    scanned = 0
    cursor = 0
    while True:
        cursor, batch = await client.scan(
            cursor=cursor,
            match="vector:*",
            count=500,
        )
        for key in batch:
            scanned += 1
            if scan_limit > 0 and scanned > scan_limit:
                cursor = 0
                break
            data = await client.hgetall(key)
            raw_vec = data.get("v") or data.get(b"v")
            if not raw_vec:
                continue
            raw_meta = data.get("m") or data.get(b"m")
            meta: dict[str, Any] = {}
            if raw_meta:
                try:
                    loaded_meta = json_loads_fn(raw_meta)
                    meta = loaded_meta if isinstance(loaded_meta, dict) else {}
                except Exception:
                    meta = {}
            vec_dim = len([p for p in str(raw_vec).split(",") if p])
            key_str = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
            vid = key_str.split("vector:", 1)[1] if "vector:" in key_str else key_str
            label_contract = extract_label_contract_fn(meta)
            if not matches_label_filters_fn(
                material_filter=material_filter,
                complexity_filter=complexity_filter,
                fine_part_type_filter=fine_part_type_filter,
                coarse_part_type_filter=coarse_part_type_filter,
                decision_source_filter=decision_source_filter,
                is_coarse_label_filter=is_coarse_label_filter,
                meta=meta,
                label_contract=label_contract,
            ):
                continue
            matched_total += 1
            if matched_total <= offset:
                continue
            if len(items) < limit:
                items.append(
                    item_cls(
                        id=vid,
                        dimension=vec_dim,
                        material=meta.get("material"),
                        complexity=meta.get("complexity"),
                        format=meta.get("format"),
                        part_type=label_contract.get("part_type"),
                        fine_part_type=label_contract.get("fine_part_type"),
                        coarse_part_type=label_contract.get("coarse_part_type"),
                        decision_source=label_contract.get("decision_source"),
                        is_coarse_label=label_contract.get("is_coarse_label"),
                    )
                )
        if cursor == 0:
            break
    return response_cls(total=matched_total, vectors=items)


__all__ = ["list_vectors_redis"]
