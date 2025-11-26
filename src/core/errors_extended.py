"""
Extended Error Codes and Error Source Classification
扩展的错误码系统，支持更细粒度的错误分类和来源追踪
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ErrorCode(str, Enum):
    """扩展的错误码枚举 - 细粒度错误分类"""

    # ========== 输入相关错误 ==========
    INPUT_ERROR = "INPUT_ERROR"                          # 通用输入错误
    INPUT_VALIDATION_FAILED = "INPUT_VALIDATION_FAILED"  # 输入验证失败
    INPUT_SIZE_EXCEEDED = "INPUT_SIZE_EXCEEDED"          # 输入大小超限
    INPUT_FORMAT_INVALID = "INPUT_FORMAT_INVALID"        # 输入格式无效
    INPUT_CORRUPTED = "INPUT_CORRUPTED"                  # 输入数据损坏
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"            # 不支持的格式
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"    # 缺少必填字段

    # ========== 网络相关错误 ==========
    NETWORK_ERROR = "NETWORK_ERROR"                      # 通用网络错误
    CONNECTION_REFUSED = "CONNECTION_REFUSED"            # 连接被拒绝
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"            # 连接超时
    DNS_RESOLUTION_FAILED = "DNS_RESOLUTION_FAILED"      # DNS解析失败
    SSL_ERROR = "SSL_ERROR"                              # SSL/TLS错误
    PROXY_ERROR = "PROXY_ERROR"                          # 代理错误

    # ========== 超时相关错误 ==========
    TIMEOUT = "TIMEOUT"                                  # 通用超时
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"                # 提供商API超时
    READ_TIMEOUT = "READ_TIMEOUT"                        # 读取超时
    WRITE_TIMEOUT = "WRITE_TIMEOUT"                      # 写入超时
    CONNECT_TIMEOUT = "CONNECT_TIMEOUT"                  # 连接超时

    # ========== 限流相关错误 ==========
    RATE_LIMIT = "RATE_LIMIT"                            # 通用限流
    UPSTREAM_RATE_LIMIT = "UPSTREAM_RATE_LIMIT"          # 上游服务限流
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"                    # 配额超限
    BURST_LIMIT_EXCEEDED = "BURST_LIMIT_EXCEEDED"        # 突发流量超限
    DAILY_LIMIT_EXCEEDED = "DAILY_LIMIT_EXCEEDED"        # 日限额超限

    # ========== 认证授权错误 ==========
    AUTH_FAILED = "AUTH_FAILED"                          # 通用认证失败
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"        # 授权失败（admin token无效）
    API_KEY_INVALID = "API_KEY_INVALID"                  # API密钥无效
    API_KEY_EXPIRED = "API_KEY_EXPIRED"                  # API密钥过期
    PERMISSION_DENIED = "PERMISSION_DENIED"              # 权限被拒绝
    TOKEN_EXPIRED = "TOKEN_EXPIRED"                      # 令牌过期
    UNAUTHORIZED = "UNAUTHORIZED"                        # 未授权

    # ========== 解析处理错误 ==========
    PARSE_FAILED = "PARSE_FAILED"                        # 通用解析失败
    JSON_PARSE_ERROR = "JSON_PARSE_ERROR"                # JSON解析错误
    XML_PARSE_ERROR = "XML_PARSE_ERROR"                  # XML解析错误
    DECODE_FAILURE = "DECODE_FAILURE"                    # 解码失败
    ENCODE_FAILURE = "ENCODE_FAILURE"                    # 编码失败
    RESPONSE_MALFORMED = "RESPONSE_MALFORMED"            # 响应格式错误

    # ========== 资源相关错误 ==========
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"            # 资源耗尽
    MEMORY_ERROR = "MEMORY_ERROR"                        # 内存错误
    DISK_FULL = "DISK_FULL"                              # 磁盘满
    CPU_LIMIT_EXCEEDED = "CPU_LIMIT_EXCEEDED"            # CPU限制超出
    THREAD_POOL_EXHAUSTED = "THREAD_POOL_EXHAUSTED"      # 线程池耗尽

    # ========== 提供商相关错误 ==========
    PROVIDER_DOWN = "PROVIDER_DOWN"                      # 提供商宕机
    PROVIDER_ERROR = "PROVIDER_ERROR"                    # 提供商错误
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"        # 提供商不可用
    PROVIDER_DEGRADED = "PROVIDER_DEGRADED"              # 提供商性能降级
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"                  # 模型未找到
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"                # 模型加载错误
    MODEL_SIZE_EXCEEDED = "MODEL_SIZE_EXCEEDED"          # 模型大小超限
    MODEL_ROLLBACK = "MODEL_ROLLBACK"                    # 模型回滚执行

    # ========== 系统相关错误 ==========
    INTERNAL_ERROR = "INTERNAL_ERROR"                    # 内部错误
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"          # 配置错误
    INITIALIZATION_ERROR = "INITIALIZATION_ERROR"        # 初始化错误
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"                # 依赖错误
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"          # 服务不可用
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"            # 向量维度不一致

    # ========== 弹性机制错误 ==========
    CIRCUIT_OPEN = "CIRCUIT_OPEN"                        # 熔断器开启
    BULKHEAD_FULL = "BULKHEAD_FULL"                      # 隔板已满
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"                  # 重试耗尽
    FALLBACK_FAILED = "FALLBACK_FAILED"                  # 降级失败

    # ========== 验证相关错误 ==========
    VALIDATION_FAILED = "VALIDATION_FAILED"              # 验证失败
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED" # Schema验证失败
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"  # 业务规则违反
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"        # 约束违反

    # ========== 数据相关错误 ==========
    DATA_NOT_FOUND = "DATA_NOT_FOUND"                    # 数据未找到
    DATA_CORRUPTION = "DATA_CORRUPTION"                  # 数据损坏
    DATA_INCONSISTENCY = "DATA_INCONSISTENCY"            # 数据不一致
    DUPLICATE_DATA = "DUPLICATE_DATA"                    # 数据重复

    # ========== 路由相关错误 ==========
    GONE = "RESOURCE_GONE"                               # 资源已移除(410)
    MOVED_PERMANENTLY = "RESOURCE_MOVED"                 # 资源永久移动(301)


class ErrorSource(str, Enum):
    """错误来源分类"""
    PROVIDER = "provider"      # 提供商/上游服务
    NETWORK = "network"        # 网络层
    INPUT = "input"            # 用户输入
    SYSTEM = "system"          # 系统内部
    CONFIGURATION = "config"   # 配置问题
    RESOURCE = "resource"      # 资源问题
    SECURITY = "security"      # 安全相关


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    CRITICAL = "critical"      # 严重 - 服务中断
    ERROR = "error"            # 错误 - 功能失败
    WARNING = "warning"        # 警告 - 性能降级
    INFO = "info"              # 信息 - 可忽略


@dataclass
class ExtendedError:
    """扩展错误信息"""
    code: ErrorCode
    source: ErrorSource
    severity: ErrorSeverity
    message: str
    provider: Optional[str] = None
    stage: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None  # 建议重试时间（秒）
    fallback_available: bool = False   # 是否有降级方案

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "code": self.code.value,
            "source": self.source.value,
            "severity": self.severity.value,
            "message": self.message
        }

        if self.provider:
            result["provider"] = self.provider
        if self.stage:
            result["stage"] = self.stage
        if self.context:
            result["context"] = self.context
        if self.retry_after:
            result["retry_after"] = self.retry_after
        if self.fallback_available:
            result["fallback_available"] = self.fallback_available

        return result


# 错误码到来源的映射
ERROR_SOURCE_MAPPING: Dict[ErrorCode, ErrorSource] = {
    # 输入错误
    ErrorCode.INPUT_ERROR: ErrorSource.INPUT,
    ErrorCode.INPUT_VALIDATION_FAILED: ErrorSource.INPUT,
    ErrorCode.INPUT_SIZE_EXCEEDED: ErrorSource.INPUT,
    ErrorCode.INPUT_FORMAT_INVALID: ErrorSource.INPUT,
    ErrorCode.INPUT_CORRUPTED: ErrorSource.INPUT,
    ErrorCode.UNSUPPORTED_FORMAT: ErrorSource.INPUT,
    ErrorCode.MISSING_REQUIRED_FIELD: ErrorSource.INPUT,
    ErrorCode.JSON_PARSE_ERROR: ErrorSource.INPUT,

    # 网络错误
    ErrorCode.NETWORK_ERROR: ErrorSource.NETWORK,
    ErrorCode.CONNECTION_REFUSED: ErrorSource.NETWORK,
    ErrorCode.CONNECTION_TIMEOUT: ErrorSource.NETWORK,
    ErrorCode.DNS_RESOLUTION_FAILED: ErrorSource.NETWORK,
    ErrorCode.SSL_ERROR: ErrorSource.NETWORK,
    ErrorCode.PROXY_ERROR: ErrorSource.NETWORK,

    # 提供商错误
    ErrorCode.PROVIDER_DOWN: ErrorSource.PROVIDER,
    ErrorCode.PROVIDER_ERROR: ErrorSource.PROVIDER,
    ErrorCode.PROVIDER_UNAVAILABLE: ErrorSource.PROVIDER,
    ErrorCode.PROVIDER_DEGRADED: ErrorSource.PROVIDER,
    ErrorCode.PROVIDER_TIMEOUT: ErrorSource.PROVIDER,
    ErrorCode.MODEL_NOT_FOUND: ErrorSource.PROVIDER,
    ErrorCode.MODEL_LOAD_ERROR: ErrorSource.PROVIDER,
    ErrorCode.MODEL_SIZE_EXCEEDED: ErrorSource.PROVIDER,
    ErrorCode.MODEL_ROLLBACK: ErrorSource.PROVIDER,
    ErrorCode.UPSTREAM_RATE_LIMIT: ErrorSource.PROVIDER,

    # 系统错误
    ErrorCode.INTERNAL_ERROR: ErrorSource.SYSTEM,
    ErrorCode.CONFIGURATION_ERROR: ErrorSource.CONFIGURATION,
    ErrorCode.INITIALIZATION_ERROR: ErrorSource.SYSTEM,
    ErrorCode.DEPENDENCY_ERROR: ErrorSource.SYSTEM,
    ErrorCode.SERVICE_UNAVAILABLE: ErrorSource.SYSTEM,
    ErrorCode.CIRCUIT_OPEN: ErrorSource.SYSTEM,
    ErrorCode.BULKHEAD_FULL: ErrorSource.SYSTEM,
    ErrorCode.RETRY_EXHAUSTED: ErrorSource.SYSTEM,
    ErrorCode.FALLBACK_FAILED: ErrorSource.SYSTEM,

    # 资源错误
    ErrorCode.RESOURCE_EXHAUSTED: ErrorSource.RESOURCE,
    ErrorCode.MEMORY_ERROR: ErrorSource.RESOURCE,
    ErrorCode.DISK_FULL: ErrorSource.RESOURCE,
    ErrorCode.CPU_LIMIT_EXCEEDED: ErrorSource.RESOURCE,
    ErrorCode.THREAD_POOL_EXHAUSTED: ErrorSource.RESOURCE,

    # 安全错误
    ErrorCode.AUTH_FAILED: ErrorSource.SECURITY,
    ErrorCode.API_KEY_INVALID: ErrorSource.SECURITY,
    ErrorCode.API_KEY_EXPIRED: ErrorSource.SECURITY,
    ErrorCode.PERMISSION_DENIED: ErrorSource.SECURITY,
    ErrorCode.TOKEN_EXPIRED: ErrorSource.SECURITY,
    ErrorCode.UNAUTHORIZED: ErrorSource.SECURITY,

    # 路由错误
    ErrorCode.GONE: ErrorSource.SYSTEM,
    ErrorCode.MOVED_PERMANENTLY: ErrorSource.SYSTEM,
}


# 错误码到严重程度的映射
ERROR_SEVERITY_MAPPING: Dict[ErrorCode, ErrorSeverity] = {
    # 严重错误
    ErrorCode.PROVIDER_DOWN: ErrorSeverity.CRITICAL,
    ErrorCode.SERVICE_UNAVAILABLE: ErrorSeverity.CRITICAL,
    ErrorCode.RESOURCE_EXHAUSTED: ErrorSeverity.CRITICAL,
    ErrorCode.MEMORY_ERROR: ErrorSeverity.CRITICAL,
    ErrorCode.DISK_FULL: ErrorSeverity.CRITICAL,
    ErrorCode.CIRCUIT_OPEN: ErrorSeverity.CRITICAL,

    # 一般错误
    ErrorCode.INTERNAL_ERROR: ErrorSeverity.ERROR,
    ErrorCode.NETWORK_ERROR: ErrorSeverity.ERROR,
    ErrorCode.AUTH_FAILED: ErrorSeverity.ERROR,
    ErrorCode.PARSE_FAILED: ErrorSeverity.ERROR,
    ErrorCode.MODEL_LOAD_ERROR: ErrorSeverity.ERROR,
    ErrorCode.MODEL_SIZE_EXCEEDED: ErrorSeverity.ERROR,
    ErrorCode.MODEL_ROLLBACK: ErrorSeverity.WARNING,
    ErrorCode.CONFIGURATION_ERROR: ErrorSeverity.ERROR,

    # 警告
    ErrorCode.RATE_LIMIT: ErrorSeverity.WARNING,
    ErrorCode.PROVIDER_DEGRADED: ErrorSeverity.WARNING,
    ErrorCode.RETRY_EXHAUSTED: ErrorSeverity.WARNING,
    ErrorCode.BULKHEAD_FULL: ErrorSeverity.WARNING,
    ErrorCode.QUOTA_EXCEEDED: ErrorSeverity.WARNING,

    # 信息
    ErrorCode.INPUT_ERROR: ErrorSeverity.INFO,
    ErrorCode.VALIDATION_FAILED: ErrorSeverity.INFO,
    ErrorCode.UNSUPPORTED_FORMAT: ErrorSeverity.INFO,
    ErrorCode.JSON_PARSE_ERROR: ErrorSeverity.INFO,
    ErrorCode.GONE: ErrorSeverity.INFO,
    ErrorCode.MOVED_PERMANENTLY: ErrorSeverity.INFO,
}


def get_error_source(error_code: ErrorCode) -> ErrorSource:
    """获取错误码对应的来源"""
    return ERROR_SOURCE_MAPPING.get(error_code, ErrorSource.SYSTEM)


def get_error_severity(error_code: ErrorCode) -> ErrorSeverity:
    """获取错误码对应的严重程度"""
    return ERROR_SEVERITY_MAPPING.get(error_code, ErrorSeverity.ERROR)


def create_extended_error(
    error_code: ErrorCode,
    message: str,
    provider: Optional[str] = None,
    stage: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> ExtendedError:
    """创建扩展错误信息"""
    return ExtendedError(
        code=error_code,
        source=get_error_source(error_code),
        severity=get_error_severity(error_code),
        message=message,
        provider=provider,
        stage=stage,
        context=context
    )


def build_error(
    error_code: ErrorCode,
    stage: str,
    message: str,
    **context: Any,
) -> Dict[str, Any]:
    """Unified error dict builder for API responses.

    Returns a dict suitable for direct inclusion under `detail` or `error` fields.
    """
    return create_extended_error(
        error_code=error_code,
        message=message,
        stage=stage,
        context=context or None,
    ).to_dict()


def create_migration_error(
    old_path: str,
    new_path: str,
    method: str = "GET",
    migration_date: str = "2024-11-24"
) -> Dict[str, Any]:
    """创建标准的端点迁移错误响应

    Args:
        old_path: 已废弃的路径
        new_path: 新的路径
        method: HTTP方法
        migration_date: 迁移日期

    Returns:
        标准化的410错误响应
    """
    return build_error(
        error_code=ErrorCode.GONE,
        stage="routing",
        message=f"Endpoint moved. Please use {method} {new_path}",
        deprecated_path=old_path,
        new_path=new_path,
        method=method,
        migration_date=migration_date
    )


# 错误码兼容性映射（旧码到新码）
LEGACY_ERROR_MAPPING: Dict[str, ErrorCode] = {
    "TIMEOUT": ErrorCode.PROVIDER_TIMEOUT,
    "PROVIDER_TIMEOUT": ErrorCode.PROVIDER_TIMEOUT,
    "RATE_LIMIT": ErrorCode.UPSTREAM_RATE_LIMIT,
    "PARSE_FAILED": ErrorCode.PARSE_FAILED,
    "DECODE_FAILURE": ErrorCode.DECODE_FAILURE,
}


def map_legacy_error_code(legacy_code: str) -> ErrorCode:
    """映射旧错误码到新错误码"""
    if legacy_code in LEGACY_ERROR_MAPPING:
        return LEGACY_ERROR_MAPPING[legacy_code]

    # 尝试直接转换
    try:
        return ErrorCode(legacy_code)
    except ValueError:
        return ErrorCode.INTERNAL_ERROR


__all__ = [
    "ErrorCode",
    "ErrorSource",
    "ErrorSeverity",
    "ExtendedError",
    "get_error_source",
    "get_error_severity",
    "create_extended_error",
    "map_legacy_error_code",
    "build_error",
    "create_migration_error",
]
