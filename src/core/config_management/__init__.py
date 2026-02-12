"""Configuration Management Module.

Provides centralized configuration:
- Multi-source configuration
- Type-safe access
- Validation
- Hot reload
"""

from src.core.config_management.core import (
    ConfigValueType,
    ConfigSchema,
    ConfigValue,
    ConfigSource,
    DictConfigSource,
    EnvironmentConfigSource,
    FileConfigSource,
    ValidationResult,
    ConfigValidator,
    min_value,
    max_value,
    in_range,
    regex_match,
    not_empty,
)
from src.core.config_management.manager import (
    ConfigChange,
    ConfigManager,
    ConfigWatcher,
    AsyncConfigWatcher,
    create_config_manager,
)

__all__ = [
    # Core
    "ConfigValueType",
    "ConfigSchema",
    "ConfigValue",
    "ConfigSource",
    "DictConfigSource",
    "EnvironmentConfigSource",
    "FileConfigSource",
    "ValidationResult",
    "ConfigValidator",
    # Validators
    "min_value",
    "max_value",
    "in_range",
    "regex_match",
    "not_empty",
    # Manager
    "ConfigChange",
    "ConfigManager",
    "ConfigWatcher",
    "AsyncConfigWatcher",
    "create_config_manager",
]
