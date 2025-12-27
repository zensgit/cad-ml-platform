"""Runtime settings (minimal) for OCR integration.

Added because main.py imports get_settings; this scaffolds required fields.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "INFO"

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = True

    # Web settings
    CORS_ORIGINS: list[str] = ["*"]
    ALLOWED_HOSTS: list[str] = ["*"]

    # Integration auth (optional)
    INTEGRATION_AUTH_MODE: str = "disabled"  # disabled|optional|required
    INTEGRATION_JWT_SECRET: str = ""
    INTEGRATION_JWT_ALG: str = "HS256"
    INTEGRATION_TENANT_HEADER: str = "x-tenant-id"
    INTEGRATION_ORG_HEADER: str = "x-org-id"
    INTEGRATION_USER_HEADER: str = "x-user-id"

    OCR_PROVIDER_DEFAULT: str = "auto"  # auto|paddle|deepseek_hf
    CONFIDENCE_FALLBACK: float = 0.85
    OCR_TIMEOUT_MS: int = 30000
    VISION_MAX_BASE64_BYTES: int = 1 * 1024 * 1024  # 1MB default limit
    # Error EMA smoothing factor (0..1], higher = more reactive
    ERROR_EMA_ALPHA: float = 0.2

    # Telemetry store backend (memory|redis|none)
    TELEMETRY_STORE_BACKEND: str = "memory"

    model_config = {
        "env_file": ".env.dev",
        "case_sensitive": False,
    }


_settings_cache: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache
