"""
Model management API endpoints
模型管理相关的API端点 - 包含模型重载、版本管理等功能
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies import get_admin_token, get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class ModelReloadRequest(BaseModel):
    """模型重载请求"""

    path: Optional[str] = Field(None, description="模型文件路径")
    expected_version: Optional[str] = Field(None, description="期望的模型版本")
    force: bool = Field(False, description="强制重载即使版本不匹配")


class ModelReloadResponse(BaseModel):
    """模型重载响应"""

    status: str = Field(
        ...,
        description=(
            "状态: success/not_found/version_mismatch/size_exceeded/rollback/"
            "magic_invalid/hash_mismatch/opcode_blocked/opcode_scan_error/error"
        ),
    )
    model_version: Optional[str] = Field(None, description="加载的模型版本")
    hash: Optional[str] = Field(None, description="模型文件哈希")
    error: Optional[Dict[str, Any]] = Field(None, description="错误信息")
    opcode_audit: Optional[Dict[str, Any]] = Field(None, description="Opcode 审计信息 (仅当 audit 模式返回)")

    model_config = ConfigDict(protected_namespaces=())


@router.post(
    "/reload",
    status_code=403,  # SEALED — no success status exists; the contract is 403 (+422 validation)
    responses={403: {"description": "Sealed (L3 Phase A): model reload via the API is fail-closed."}},
)
async def model_reload(
    payload: ModelReloadRequest,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    """SEALED (L3 Phase A) — always 403; no model is ever loaded through this route.

    This route hot-reloaded an ARBITRARY caller-supplied `path` into the serving process
    (`reload_model(payload.path, force=...)` → `classifier.py:535 pickle.loads`, which deserializes
    BEFORE the whitelist/hash check, with the hash truncated to 16 hex — a reproduced RCE), guarded
    only by `api_key`/`admin_token` that both default to the literal `"test"`. It is the highest-risk
    external model-activation surface (L3 design-lock §1.A/§3.2).

    Per the design-lock, this route may be re-enabled ONLY when BOTH hold: (1) the production-identity
    gate (no default `test`, unforgeable authenticated subject), and (2) the proof membrane
    (`verify_and_load` — a server-owned artifact id, one immutable read hash-and-load, a signed
    out-of-band proof binding model_hash+family+split+evaluator+thresholds+env, expiry/revocation).
    Neither exists yet, so the membrane default here is #509's: **fail closed, unconditionally.**
    Emergency rollback runs in-process via auto-remediation and does not use this route.

    Do NOT log the caller-supplied path (untrusted input).
    """
    logger.warning(
        "model reload via API refused (L3 Phase-A seal): path_provided=%s force=%s",
        bool(payload.path), bool(payload.force),
    )
    raise HTTPException(
        status_code=403,
        detail=(
            "Model reload via the API is disabled (fail-closed). The retrain/activation path is "
            "sealed until the L3 production-identity gate AND the model-activation proof membrane "
            "both hold (design-lock §3.2). Emergency rollback runs in-process via auto-remediation."
        ),
    )
    # NOTE (post-membrane): reintroduce loading behind verify_and_load, accepting only a server-owned
    # artifact id (never a caller path) whose bytes' SHA-256 matches a signed, unexpired, unrevoked
    # release proof for the server-owned (family, environment). The pre-seal status envelope
    # (success / not_found / version_mismatch / size_exceeded / magic_invalid / hash_mismatch /
    # opcode_blocked / opcode_scan_error / rollback) was removed with the seal, not left as dead code.


@router.get("/version")
async def get_model_version(api_key: str = Depends(get_api_key)):
    """
    获取当前模型版本信息

    Returns:
        模型版本详情
    """
    from src.ml.classifier import get_model_info

    info = get_model_info()
    return {
        "model_version": info.get("version"),
        "model_hash": info.get("hash"),
        "loaded_at": info.get("loaded_at"),
        "path": info.get("path"),
    }


@router.get("/opcode-audit")
async def get_opcode_audit(
    api_key: str = Depends(get_api_key), admin_token: str = Depends(get_admin_token)
):
    """获取已审计的 pickle opcode 使用情况 (仅当启用扫描时)."""
    from src.ml.classifier import get_opcode_audit_snapshot

    return get_opcode_audit_snapshot()
