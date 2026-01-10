#!/usr/bin/env python3
import hashlib
import os
import time
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, UploadFile


app = FastAPI(title="dedupcad-vision-stub", version="0.1")


def _hash_bytes(content: bytes) -> str:
    if not content:
        return "0" * 32
    return hashlib.md5(content).hexdigest()  # nosec: non-crypto hash ok for stub


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "indexes": {"l1": "ready", "l2": "ready"},
    }


@app.post("/api/v2/search")
async def search(
    file: UploadFile = File(...),
    mode: str = Form("balanced"),
    max_results: int = Form(5),
    compute_diff: bool = Form(False),
    enable_ml: bool = Form(False),
    enable_geometric: bool = Form(False),
) -> Dict[str, Any]:
    started = time.time()
    content = await file.read()
    file_hash = _hash_bytes(content)

    duplicates: List[Dict[str, Any]] = []
    if max_results and max_results > 0:
        duplicates.append(
            {
                "drawing_id": "stub-0",
                "file_hash": file_hash,
                "file_name": file.filename or "unknown",
                "similarity": 0.99,
                "confidence": 0.99,
                "match_level": 1,
                "verdict": "duplicate",
                "levels": {
                    "mode": mode,
                    "enable_ml": enable_ml,
                    "enable_geometric": enable_geometric,
                },
            }
        )

    elapsed_ms = int((time.time() - started) * 1000)
    timing = {
        "total_ms": elapsed_ms,
        "l1_ms": max(elapsed_ms - 2, 0),
        "l2_ms": 1,
        "l3_ms": 1,
        "l4_ms": 0,
    }

    level_stats = {
        "l1": {"passed": len(duplicates), "filtered": 0, "time_ms": timing["l1_ms"]},
        "l2": {"passed": 0, "filtered": 0, "time_ms": timing["l2_ms"]},
        "l3": {"passed": 0, "filtered": 0, "time_ms": timing["l3_ms"]},
        "l4": {"passed": 0, "filtered": 0, "time_ms": timing["l4_ms"]},
    }
    return {
        "success": True,
        "total_matches": len(duplicates),
        "duplicates": duplicates,
        "similar": [],
        "final_level": 1,
        "timing": timing,
        "level_stats": level_stats,
        "warnings": [],
        "error": None,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("DEDUPCAD_VISION_HOST", "0.0.0.0")
    port = int(os.getenv("DEDUPCAD_VISION_PORT", "58001"))
    uvicorn.run(app, host=host, port=port)
