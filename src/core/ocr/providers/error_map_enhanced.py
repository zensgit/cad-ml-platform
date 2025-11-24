"""
Enhanced Error Mapping for OCR Providers
增强的错误映射系统，支持细粒度错误分类和智能映射
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.core.errors_extended import (
    ErrorCode,
    ErrorSource,
    ErrorSeverity,
    ExtendedError,
)

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """错误模式定义"""
    pattern: re.Pattern
    error_code: ErrorCode
    source: ErrorSource
    severity: ErrorSeverity


# 错误消息模式匹配规则
ERROR_PATTERNS = [
    # 网络相关
    ErrorPattern(
        re.compile(r"connection\s+(refused|reset|aborted)", re.IGNORECASE),
        ErrorCode.CONNECTION_REFUSED,
        ErrorSource.NETWORK,
        ErrorSeverity.ERROR
    ),
    ErrorPattern(
        re.compile(r"(dns|hostname|resolve)\s+.*\s+(fail|error)", re.IGNORECASE),
        ErrorCode.DNS_RESOLUTION_FAILED,
        ErrorSource.NETWORK,
        ErrorSeverity.ERROR
    ),
    ErrorPattern(
        re.compile(r"ssl|tls|certificate", re.IGNORECASE),
        ErrorCode.SSL_ERROR,
        ErrorSource.NETWORK,
        ErrorSeverity.ERROR
    ),

    # 超时相关
    ErrorPattern(
        re.compile(r"read\s+timeout", re.IGNORECASE),
        ErrorCode.READ_TIMEOUT,
        ErrorSource.PROVIDER,
        ErrorSeverity.WARNING
    ),
    ErrorPattern(
        re.compile(r"write\s+timeout", re.IGNORECASE),
        ErrorCode.WRITE_TIMEOUT,
        ErrorSource.PROVIDER,
        ErrorSeverity.WARNING
    ),
    ErrorPattern(
        re.compile(r"connect\s+timeout", re.IGNORECASE),
        ErrorCode.CONNECT_TIMEOUT,
        ErrorSource.NETWORK,
        ErrorSeverity.WARNING
    ),

    # 限流相关
    ErrorPattern(
        re.compile(r"(rate|quota)\s+limit", re.IGNORECASE),
        ErrorCode.UPSTREAM_RATE_LIMIT,
        ErrorSource.PROVIDER,
        ErrorSeverity.WARNING
    ),
    ErrorPattern(
        re.compile(r"too\s+many\s+requests", re.IGNORECASE),
        ErrorCode.UPSTREAM_RATE_LIMIT,
        ErrorSource.PROVIDER,
        ErrorSeverity.WARNING
    ),
    ErrorPattern(
        re.compile(r"daily\s+limit", re.IGNORECASE),
        ErrorCode.DAILY_LIMIT_EXCEEDED,
        ErrorSource.PROVIDER,
        ErrorSeverity.WARNING
    ),

    # 认证相关
    ErrorPattern(
        re.compile(r"api\s+key\s+(invalid|expired)", re.IGNORECASE),
        ErrorCode.API_KEY_INVALID,
        ErrorSource.SECURITY,
        ErrorSeverity.ERROR
    ),
    ErrorPattern(
        re.compile(r"(unauthorized|forbidden|401|403)", re.IGNORECASE),
        ErrorCode.UNAUTHORIZED,
        ErrorSource.SECURITY,
        ErrorSeverity.ERROR
    ),
    ErrorPattern(
        re.compile(r"permission\s+denied", re.IGNORECASE),
        ErrorCode.PERMISSION_DENIED,
        ErrorSource.SECURITY,
        ErrorSeverity.ERROR
    ),

    # 资源相关
    ErrorPattern(
        re.compile(r"out\s+of\s+memory|oom", re.IGNORECASE),
        ErrorCode.MEMORY_ERROR,
        ErrorSource.RESOURCE,
        ErrorSeverity.CRITICAL
    ),
    ErrorPattern(
        re.compile(r"disk\s+(full|space)", re.IGNORECASE),
        ErrorCode.DISK_FULL,
        ErrorSource.RESOURCE,
        ErrorSeverity.CRITICAL
    ),
    ErrorPattern(
        re.compile(r"cpu\s+limit", re.IGNORECASE),
        ErrorCode.CPU_LIMIT_EXCEEDED,
        ErrorSource.RESOURCE,
        ErrorSeverity.ERROR
    ),

    # 解析相关
    ErrorPattern(
        re.compile(r"json\s+(parse|decode|invalid)", re.IGNORECASE),
        ErrorCode.JSON_PARSE_ERROR,
        ErrorSource.SYSTEM,
        ErrorSeverity.ERROR
    ),
    ErrorPattern(
        re.compile(r"(malformed|corrupt)\s+(response|data)", re.IGNORECASE),
        ErrorCode.RESPONSE_MALFORMED,
        ErrorSource.PROVIDER,
        ErrorSeverity.ERROR
    ),

    # 模型相关
    ErrorPattern(
        re.compile(r"model\s+not\s+found", re.IGNORECASE),
        ErrorCode.MODEL_NOT_FOUND,
        ErrorSource.PROVIDER,
        ErrorSeverity.ERROR
    ),
    ErrorPattern(
        re.compile(r"model\s+(load|init)", re.IGNORECASE),
        ErrorCode.MODEL_LOAD_ERROR,
        ErrorSource.PROVIDER,
        ErrorSeverity.ERROR
    ),
]


# 异常类型到错误码的映射
EXCEPTION_TYPE_MAPPING: Dict[type, ErrorCode] = {
    # Python 内置异常
    MemoryError: ErrorCode.MEMORY_ERROR,
    TimeoutError: ErrorCode.TIMEOUT,
    ConnectionError: ErrorCode.CONNECTION_REFUSED,
    ConnectionRefusedError: ErrorCode.CONNECTION_REFUSED,
    ConnectionAbortedError: ErrorCode.CONNECTION_REFUSED,
    PermissionError: ErrorCode.PERMISSION_DENIED,
    FileNotFoundError: ErrorCode.DATA_NOT_FOUND,
    ValueError: ErrorCode.INPUT_VALIDATION_FAILED,
    TypeError: ErrorCode.INPUT_ERROR,
    KeyError: ErrorCode.MISSING_REQUIRED_FIELD,
    AttributeError: ErrorCode.PARSE_FAILED,
    OSError: ErrorCode.SYSTEM,
    IOError: ErrorCode.NETWORK_ERROR,

    # Asyncio 异常
    asyncio.TimeoutError: ErrorCode.TIMEOUT,
    asyncio.CancelledError: ErrorCode.INTERNAL_ERROR,
}


class EnhancedErrorMapper:
    """增强的错误映射器"""

    def __init__(self):
        self.custom_mappings: Dict[str, ErrorCode] = {}
        self.provider_specific_patterns: Dict[str, list] = {}

    def map_exception(
        self,
        exc: Exception,
        provider: Optional[str] = None,
        stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExtendedError:
        """
        将异常映射到扩展错误

        Args:
            exc: 异常实例
            provider: 提供商名称
            stage: 处理阶段
            context: 上下文信息

        Returns:
            ExtendedError 实例
        """
        # 1. 尝试异常类型映射
        error_code = self._map_by_exception_type(exc)

        # 2. 如果没有找到，尝试消息模式匹配
        source: ErrorSource = ErrorSource.SYSTEM
        severity: ErrorSeverity = ErrorSeverity.ERROR
        if error_code == ErrorCode.INTERNAL_ERROR:
            mapped_code, mapped_source, mapped_severity = self._map_by_message_pattern(str(exc))
            error_code = mapped_code
            source = mapped_source
            severity = mapped_severity

        # 3. 检查提供商特定映射
        if provider and provider in self.provider_specific_patterns:
            provider_code = self._check_provider_patterns(
                provider, exc, str(exc)
            )
            if provider_code:
                error_code = provider_code

        # 4. 创建扩展错误 (直接使用上面确定的 source/severity)
        extended_error = ExtendedError(
            code=error_code,
            source=source,
            severity=severity,
            message=str(exc),
            provider=provider,
            stage=stage,
            context=context or {}
        )

        # 5. 添加额外信息并记录
        self._enrich_error_info(extended_error, exc)
        self._log_error(extended_error, exc)

        return extended_error

    def _map_by_exception_type(self, exc: Exception) -> ErrorCode:
        """根据异常类型映射"""
        # 精确类型匹配
        exc_type = type(exc)
        if exc_type in EXCEPTION_TYPE_MAPPING:
            return EXCEPTION_TYPE_MAPPING[exc_type]

        # 继承链匹配
        for base_type, error_code in EXCEPTION_TYPE_MAPPING.items():
            if isinstance(exc, base_type):
                return error_code

        # 异常类名匹配
        exc_name = exc_type.__name__
        if "Timeout" in exc_name:
            return ErrorCode.TIMEOUT
        elif "Connection" in exc_name:
            return ErrorCode.NETWORK_ERROR
        elif "Auth" in exc_name:
            return ErrorCode.AUTH_FAILED
        elif "Permission" in exc_name:
            return ErrorCode.PERMISSION_DENIED
        elif "Parse" in exc_name:
            return ErrorCode.PARSE_FAILED
        elif "Quota" in exc_name or "RateLimit" in exc_name:
            return ErrorCode.QUOTA_EXCEEDED
        elif "Model" in exc_name:
            return ErrorCode.MODEL_LOAD_ERROR

        return ErrorCode.INTERNAL_ERROR

    def _map_by_message_pattern(
        self,
        error_message: str
    ) -> Tuple[ErrorCode, ErrorSource, ErrorSeverity]:
        """根据错误消息模式映射"""
        for pattern in ERROR_PATTERNS:
            if pattern.pattern.search(error_message):
                return pattern.error_code, pattern.source, pattern.severity

        # 默认值
        return ErrorCode.INTERNAL_ERROR, ErrorSource.SYSTEM, ErrorSeverity.ERROR

    def _check_provider_patterns(
        self,
        provider: str,
        exc: Exception,
        error_message: str
    ) -> Optional[ErrorCode]:
        """检查提供商特定模式"""
        if provider not in self.provider_specific_patterns:
            return None

        for pattern, error_code in self.provider_specific_patterns[provider]:
            if isinstance(pattern, type) and isinstance(exc, pattern):
                return error_code
            elif isinstance(pattern, re.Pattern) and pattern.search(error_message):
                return error_code

        return None

    def _enrich_error_info(self, error: ExtendedError, exc: Exception):
        """丰富错误信息"""
        # 添加重试建议
        if error.code in [
            ErrorCode.PROVIDER_TIMEOUT,
            ErrorCode.NETWORK_ERROR,
            ErrorCode.CONNECTION_REFUSED
        ]:
            error.retry_after = 5  # 5秒后重试
            error.fallback_available = True

        elif error.code in [
            ErrorCode.UPSTREAM_RATE_LIMIT,
            ErrorCode.QUOTA_EXCEEDED
        ]:
            error.retry_after = 60  # 60秒后重试
            error.fallback_available = False

        # 添加异常链信息
        if hasattr(exc, "__cause__") and exc.__cause__:
            if not error.context:
                error.context = {}
            error.context["cause"] = str(exc.__cause__)

    def _log_error(self, error: ExtendedError, exc: Exception):
        """记录错误日志"""
        log_message = (
            f"Error mapped: {error.code.value} | "
            f"Source: {error.source.value} | "
            f"Severity: {error.severity.value} | "
            f"Provider: {error.provider} | "
            f"Stage: {error.stage} | "
            f"Exception: {type(exc).__name__}: {exc}"
        )

        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def register_provider_pattern(
        self,
        provider: str,
        pattern: Any,
        error_code: ErrorCode
    ):
        """注册提供商特定模式"""
        if provider not in self.provider_specific_patterns:
            self.provider_specific_patterns[provider] = []

        self.provider_specific_patterns[provider].append((pattern, error_code))

    def register_custom_mapping(self, key: str, error_code: ErrorCode):
        """注册自定义映射"""
        self.custom_mappings[key] = error_code


# 全局错误映射器实例
error_mapper = EnhancedErrorMapper()


# 便利函数
def map_exception_enhanced(
    exc: Exception,
    provider: Optional[str] = None,
    stage: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> ExtendedError:
    """映射异常到扩展错误（便利函数）"""
    return error_mapper.map_exception(exc, provider, stage, context)


def handle_provider_error(
    exc: Exception,
    provider: str,
    stage: str = "unknown"
) -> ExtendedError:
    """处理提供商错误"""
    return map_exception_enhanced(
        exc,
        provider=provider,
        stage=stage,
        context={"provider_error": True}
    )


# 注册提供商特定模式示例
def register_deepseek_patterns():
    """注册 DeepSeek 特定错误模式"""
    error_mapper.register_provider_pattern(
        "deepseek",
        re.compile(r"deepseek.*rate.*limit", re.IGNORECASE),
        ErrorCode.UPSTREAM_RATE_LIMIT
    )
    error_mapper.register_provider_pattern(
        "deepseek",
        re.compile(r"deepseek.*model.*not.*available", re.IGNORECASE),
        ErrorCode.MODEL_NOT_FOUND
    )


def register_openai_patterns():
    """注册 OpenAI 特定错误模式"""
    error_mapper.register_provider_pattern(
        "openai",
        re.compile(r"insufficient_quota", re.IGNORECASE),
        ErrorCode.QUOTA_EXCEEDED
    )
    error_mapper.register_provider_pattern(
        "openai",
        re.compile(r"invalid_api_key", re.IGNORECASE),
        ErrorCode.API_KEY_INVALID
    )


# 初始化时注册已知模式
register_deepseek_patterns()
register_openai_patterns()


__all__ = [
    "EnhancedErrorMapper",
    "error_mapper",
    "map_exception_enhanced",
    "handle_provider_error",
]
