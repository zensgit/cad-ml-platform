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


@router.post("/reload", response_model=ModelReloadResponse)
async def model_reload(
    payload: ModelReloadRequest,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    """
    重载机器学习模型

    支持以下功能：
    - 指定路径加载模型
    - 版本验证
    - 强制重载
    - 自动回滚机制

    Args:
        payload: 模型重载请求参数
        api_key: API密钥

    Returns:
        模型重载结果
    """
    # L3 seal (Phase A): this route loads an ARBITRARY model path into the serving process — a
    # promotion sink that bypassed the unconditional evaluation-integrity gate guarding
    # auto_retrain.sh. Until the Phase-B two-stage release gate exists (which will accept only a
    # candidate whose SHA-256 matches an approved evaluation artifact), the route is sealed
    # fail-closed: no request payload, credential, or flag opens it. Emergency rollback is NOT
    # affected — auto_remediation._action_rollback_model calls reload_model(prev_path) in-process
    # and never traverses this route. See docs/PRODUCT_STRATEGY.md §5.2 and §8.1.
    logger.warning(
        "model reload via API refused (Phase-A seal): path=%s expected_version=%s force=%s",
        payload.path, payload.expected_version, payload.force,
    )
    raise HTTPException(
        status_code=403,
        detail=(
            "Model reload via the API is disabled: the retrain/promotion path is fail-closed "
            "until the Track E two-stage release gate binds an approved evaluation artifact to "
            "the candidate model hash (PRODUCT_STRATEGY.md §5.2, §8.1). Emergency rollback runs "
            "in-process via auto-remediation and is unaffected."
        ),
    )
    # NOTE (Phase B): the pre-seal implementation (reload_model + the structured status envelope —
    # success / not_found / version_mismatch / size_exceeded / magic_invalid / hash_mismatch /
    # opcode_blocked / opcode_scan_error / rollback) was removed with the seal rather than left as
    # dead code. Reintroduce it behind the two-stage gate, accepting ONLY a candidate whose SHA-256
    # matches the approved evaluation artifact. reload_model itself (and its security validation)
    # is unchanged and still covered by the direct-call unit tests.


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
