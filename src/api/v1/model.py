"""
Model management API endpoints
模型管理相关的API端点 - 包含模型重载、版本管理等功能
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key, get_admin_token
from src.core.errors_extended import ErrorCode, create_extended_error

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


@router.post("/reload", response_model=ModelReloadResponse)
async def model_reload(
    payload: ModelReloadRequest,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token)
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
    from src.ml.classifier import reload_model

    # Call the reload function
    result = reload_model(
        payload.path,
        expected_version=payload.expected_version,
        force=payload.force
    )

    status = result.get("status")

    # Handle different status cases
    if status == "success":
        logger.info(
            f"Model reloaded successfully: version={result.get('model_version')}, "
            f"hash={result.get('hash')}"
        )
        # When audit mode is active, include audit snapshot
        from src.ml.classifier import get_opcode_audit_snapshot  # type: ignore
        audit = get_opcode_audit_snapshot()
        return ModelReloadResponse(
            status="success",
            model_version=result.get("model_version"),
            hash=result.get("hash")
        )

    if status == "not_found":
        logger.error(f"Model file not found: {payload.path}")
        return ModelReloadResponse(status="not_found", error=result.get("error"))

    if status == "version_mismatch":
        logger.warning(
            f"Model version mismatch: expected={payload.expected_version}, actual={result.get('actual_version')}"
        )
        return ModelReloadResponse(status="version_mismatch", error=result.get("error"))

    if status == "size_exceeded":
        logger.error(
            f"Model size exceeded: {result.get('error', {}).get('context', {}).get('size_mb')}MB > {result.get('error', {}).get('context', {}).get('max_mb')}MB"
        )
        return ModelReloadResponse(status="size_exceeded", error=result.get("error"))

    # Pass-through structured security/validation errors
    if status in {"magic_invalid", "hash_mismatch", "opcode_blocked", "opcode_scan_error"}:
        logger.warning(
            "Model reload failed",
            extra={
                "status": status,
                "error": result.get("error"),
            },
        )
        return ModelReloadResponse(status=status, error=result.get("error"))

    if status == "rollback":
        logger.warning(f"Model rolled back to version {result.get('rollback_version')}")
        return ModelReloadResponse(
            status="rollback",
            model_version=result.get("rollback_version"),
            hash=result.get("rollback_hash"),
            error=result.get("error")
        )

    # Unknown status
    logger.error(f"Unknown model reload status: {status}")
    return ModelReloadResponse(status="error", error=result.get("error"))


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
        "path": info.get("path")
    }


@router.get("/opcode-audit")
async def get_opcode_audit(api_key: str = Depends(get_api_key), admin_token: str = Depends(get_admin_token)):
    """获取已审计的 pickle opcode 使用情况 (仅当启用扫描时)."""
    from src.ml.classifier import get_opcode_audit_snapshot
    return get_opcode_audit_snapshot()
