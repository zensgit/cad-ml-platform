"""Configuration Center for CAD ML Platform.

Provides:
- Dynamic configuration management
- Hot reload without restart
- Environment-specific overrides
- Configuration versioning
- Watch and callback support
"""

from src.core.config.manager import (
    ConfigManager,
    ConfigSource,
    ConfigValue,
    ConfigChangeEvent,
    get_config_manager,
    config_value,
)
from src.core.config.sources import (
    EnvConfigSource,
    FileConfigSource,
    ConsulConfigSource,
    EtcdConfigSource,
)
from src.core.config.watcher import (
    ConfigWatcher,
    WatchCallback,
)

# Re-export legacy Settings and get_settings for backward compatibility
# These are defined in the sibling config.py module
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings for OCR integration."""
    DEBUG: bool = True
    HOST: str = "0.0.0.0"  # nosec B104
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "INFO"

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = True

    CORS_ORIGINS: list[str] = ["*"]
    ALLOWED_HOSTS: list[str] = ["*"]

    INTEGRATION_AUTH_MODE: str = "disabled"
    INTEGRATION_JWT_SECRET: str = ""
    INTEGRATION_JWT_ALG: str = "HS256"
    INTEGRATION_TENANT_HEADER: str = "x-tenant-id"
    INTEGRATION_ORG_HEADER: str = "x-org-id"
    INTEGRATION_USER_HEADER: str = "x-user-id"

    OCR_PROVIDER_DEFAULT: str = "auto"
    CONFIDENCE_FALLBACK: float = 0.85
    OCR_TIMEOUT_MS: int = 30000
    VISION_MAX_BASE64_BYTES: int = 1 * 1024 * 1024
    ERROR_EMA_ALPHA: float = 0.2
    TELEMETRY_STORE_BACKEND: str = "memory"

    model_config = {
        "env_file": ".env.dev",
        "case_sensitive": False,
    }


_settings_cache: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


__all__ = [
    # Manager
    "ConfigManager",
    "ConfigSource",
    "ConfigValue",
    "ConfigChangeEvent",
    "get_config_manager",
    "config_value",
    # Sources
    "EnvConfigSource",
    "FileConfigSource",
    "ConsulConfigSource",
    "EtcdConfigSource",
    # Watcher
    "ConfigWatcher",
    "WatchCallback",
    # Legacy settings
    "Settings",
    "get_settings",
]
