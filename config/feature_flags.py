"""Feature flags configuration for CAD ML Platform.

Controls experimental features, security modes, and version rollout.
"""

import os
from typing import Dict, Any, Literal

# Feature version control
V4_ENABLED = os.getenv("FEATURE_V4_ENABLE", "0") == "1"
V4_SURFACE_ALGORITHM: Literal["simple", "advanced"] = os.getenv(
    "FEATURE_V4_SURFACE_ALGORITHM", "simple"
)  # type: ignore
V4_STRICT_MODE = os.getenv("FEATURE_V4_ENABLE_STRICT", "0") == "1"

# Security and model loading
OPCODE_MODE: Literal["audit", "blocklist", "whitelist"] = os.getenv(
    "MODEL_OPCODE_MODE", "blocklist"
)  # type: ignore
OPCODE_SCAN_ENABLED = os.getenv("MODEL_OPCODE_SCAN_ENABLED", "1") == "1"
MAGIC_NUMBER_CHECK = os.getenv("MODEL_MAGIC_NUMBER_CHECK", "1") == "1"
HASH_WHITELIST_ENABLED = bool(os.getenv("ALLOWED_MODEL_HASHES", "").strip())

# Cache tuning
CACHE_TUNING_EXPERIMENTAL = os.getenv("CACHE_TUNING_EXPERIMENTAL", "1") == "1"
CACHE_AUTO_ADJUST_TTL = os.getenv("CACHE_AUTO_ADJUST_TTL", "0") == "1"

# Drift baseline management
DRIFT_AUTO_REFRESH = os.getenv("DRIFT_BASELINE_AUTO_REFRESH", "1") == "1"
DRIFT_BASELINE_EXPORT_ENABLED = os.getenv("DRIFT_BASELINE_EXPORT_ENABLED", "0") == "1"

# Backend management
BACKEND_RELOAD_AUTH_REQUIRED = os.getenv("BACKEND_RELOAD_AUTH_REQUIRED", "1") == "1"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Batch operations
BATCH_SIMILARITY_MAX_IDS = int(os.getenv("BATCH_SIMILARITY_MAX_IDS", "200"))
BATCH_EMPTY_RESULT_TRACKING = os.getenv("BATCH_EMPTY_RESULT_TRACKING", "1") == "1"

# Migration and compatibility
MIGRATION_DOWNGRADE_TRACKING = os.getenv("MIGRATION_DOWNGRADE_TRACKING", "1") == "1"
MIGRATION_PREVIEW_ENABLED = os.getenv("MIGRATION_PREVIEW_ENABLED", "1") == "1"


def get_feature_flags() -> Dict[str, Any]:
    """Get all feature flags as a dictionary for inspection."""
    return {
        "v4_enabled": V4_ENABLED,
        "v4_surface_algorithm": V4_SURFACE_ALGORITHM,
        "v4_strict_mode": V4_STRICT_MODE,
        "opcode_mode": OPCODE_MODE,
        "opcode_scan_enabled": OPCODE_SCAN_ENABLED,
        "magic_number_check": MAGIC_NUMBER_CHECK,
        "hash_whitelist_enabled": HASH_WHITELIST_ENABLED,
        "cache_tuning_experimental": CACHE_TUNING_EXPERIMENTAL,
        "cache_auto_adjust_ttl": CACHE_AUTO_ADJUST_TTL,
        "drift_auto_refresh": DRIFT_AUTO_REFRESH,
        "drift_baseline_export_enabled": DRIFT_BASELINE_EXPORT_ENABLED,
        "backend_reload_auth_required": BACKEND_RELOAD_AUTH_REQUIRED,
        "batch_similarity_max_ids": BATCH_SIMILARITY_MAX_IDS,
        "batch_empty_result_tracking": BATCH_EMPTY_RESULT_TRACKING,
        "migration_downgrade_tracking": MIGRATION_DOWNGRADE_TRACKING,
        "migration_preview_enabled": MIGRATION_PREVIEW_ENABLED,
    }


def validate_feature_flags() -> None:
    """Validate feature flag combinations for conflicts."""
    issues = []

    # V4 strict mode requires V4 enabled
    if V4_STRICT_MODE and not V4_ENABLED:
        issues.append("V4_STRICT_MODE requires FEATURE_V4_ENABLE=1")

    # Whitelist mode requires hash whitelist
    if OPCODE_MODE == "whitelist" and not HASH_WHITELIST_ENABLED:
        issues.append("OPCODE_MODE=whitelist requires ALLOWED_MODEL_HASHES to be set")

    # Backend reload auth requires admin token
    if BACKEND_RELOAD_AUTH_REQUIRED and not ADMIN_TOKEN:
        issues.append("BACKEND_RELOAD_AUTH_REQUIRED=1 requires ADMIN_TOKEN to be set")

    if issues:
        import warnings

        for issue in issues:
            warnings.warn(f"Feature flag conflict: {issue}")


# Validate on import
validate_feature_flags()


__all__ = [
    "V4_ENABLED",
    "V4_SURFACE_ALGORITHM",
    "V4_STRICT_MODE",
    "OPCODE_MODE",
    "OPCODE_SCAN_ENABLED",
    "MAGIC_NUMBER_CHECK",
    "HASH_WHITELIST_ENABLED",
    "CACHE_TUNING_EXPERIMENTAL",
    "CACHE_AUTO_ADJUST_TTL",
    "DRIFT_AUTO_REFRESH",
    "DRIFT_BASELINE_EXPORT_ENABLED",
    "BACKEND_RELOAD_AUTH_REQUIRED",
    "ADMIN_TOKEN",
    "BATCH_SIMILARITY_MAX_IDS",
    "BATCH_EMPTY_RESULT_TRACKING",
    "MIGRATION_DOWNGRADE_TRACKING",
    "MIGRATION_PREVIEW_ENABLED",
    "get_feature_flags",
    "validate_feature_flags",
]
