"""Process rules audit endpoints extracted from analyze.py for modularity."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.utils.analysis_metrics import process_rules_audit_requests_total

router = APIRouter()


class ProcessRulesAuditResponse(BaseModel):
    version: str = Field(description="规则版本")
    source: str = Field(description="规则文件来源")
    hash: str | None = Field(default=None, description="文件内容哈希前16位")
    materials: list[str]
    complexities: Dict[str, list[str]]
    raw: Dict[str, Any]


@router.get("/process/rules/audit", response_model=ProcessRulesAuditResponse)
async def process_rules_audit(raw: bool = True, api_key: str = Depends(get_api_key)):
    from src.core.process_rules import load_rules

    path = os.getenv("PROCESS_RULES_FILE", "config/process_rules.yaml")
    rules = load_rules(force_reload=True)
    version = rules.get("__meta__", {}).get("version", "v1")
    materials = sorted([m for m in rules.keys() if not m.startswith("__")])
    complexities: Dict[str, list[str]] = {}
    for m in materials:
        cm = rules.get(m, {})
        if isinstance(cm, dict):
            complexities[m] = sorted([c for c in cm.keys() if isinstance(cm.get(c), list)])
    file_hash: str | None = None
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        file_hash = None
    try:
        resp = ProcessRulesAuditResponse(
        version=version,
        source=path if os.path.exists(path) else "embedded-defaults",
        hash=file_hash,
        materials=materials,
        complexities=complexities,
        raw=rules if raw else {},
        )
        process_rules_audit_requests_total.labels(status="ok").inc()
        return resp
    except Exception:
        process_rules_audit_requests_total.labels(status="error").inc()
        raise


__all__ = ["router"]
