"""
Model management API endpoints
模型管理相关的API端点 - 包含模型重载、版本管理等功能
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.errors_extended import ErrorCode, create_extended_error

logger = logging.getLogger(__name__)
router = APIRouter()


class ModelReloadRequest(BaseModel):
    """模型重载请求"""
    path: str | None = Field(None, description="模型文件路径")
    expected_version: str | None = Field(None, description="期望的模型版本")
    force: bool = Field(False, description="强制重载即使版本不匹配")


class ModelReloadResponse(BaseModel):
    """模型重载响应"""
    status: str = Field(..., description="状态: success/not_found/version_mismatch/size_exceeded/rollback")
    model_version: str | None = Field(None, description="加载的模型版本")
    hash: str | None = Field(None, description="模型文件哈希")
    error: Dict[str, Any] | None = Field(None, description="错误信息")


@router.post("/reload", response_model=ModelReloadResponse)
async def model_reload(
    payload: ModelReloadRequest,
    api_key: str = Depends(get_api_key)
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
        return ModelReloadResponse(
            status="success",
            model_version=result.get("model_version"),
            hash=result.get("hash")
        )

    if status == "not_found":
        ext = create_extended_error(
            ErrorCode.DATA_NOT_FOUND,
            "Model file not found",
            stage="model_reload",
            context={"path": payload.path}
        )
        logger.error(f"Model file not found: {payload.path}")
        return ModelReloadResponse(
            status="not_found",
            error=ext.to_dict()
        )

    if status == "version_mismatch":
        ext = create_extended_error(
            ErrorCode.VALIDATION_FAILED,
            "Version mismatch",
            stage="model_reload",
            context={
                "expected": payload.expected_version,
                "actual": result.get("actual_version")
            }
        )
        logger.warning(
            f"Model version mismatch: expected={payload.expected_version}, "
            f"actual={result.get('actual_version')}"
        )
        return ModelReloadResponse(
            status="version_mismatch",
            error=ext.to_dict()
        )

    if status == "size_exceeded":
        ext = create_extended_error(
            ErrorCode.MODEL_SIZE_EXCEEDED,
            "Model file size exceeded limit",
            stage="model_reload",
            context={
                "size_mb": result.get("size_mb"),
                "max_mb": result.get("max_mb")
            }
        )
        logger.error(
            f"Model size exceeded: {result.get('size_mb')}MB > "
            f"{result.get('max_mb')}MB"
        )
        return ModelReloadResponse(
            status="size_exceeded",
            error=ext.to_dict()
        )

    if status == "rollback":
        ext = create_extended_error(
            ErrorCode.MODEL_ROLLBACK,
            "Model loading failed, rolled back to previous version",
            stage="model_reload",
            context={
                "rollback_version": result.get("rollback_version"),
                "rollback_hash": result.get("rollback_hash")
            }
        )
        logger.warning(
            f"Model rolled back to version {result.get('rollback_version')}"
        )
        return ModelReloadResponse(
            status="rollback",
            model_version=result.get("rollback_version"),
            hash=result.get("rollback_hash"),
            error=ext.to_dict()
        )

    # Unknown status
    ext = create_extended_error(
        ErrorCode.INTERNAL_ERROR,
        f"Unknown status: {status}",
        stage="model_reload"
    )
    logger.error(f"Unknown model reload status: {status}")
    return ModelReloadResponse(
        status="error",
        error=ext.to_dict()
    )


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