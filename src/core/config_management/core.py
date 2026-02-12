"""Configuration Management Core.

Provides centralized configuration:
- Type-safe configuration
- Validation
- Environment support
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

T = TypeVar("T")


class ConfigValueType(Enum):
    """Types of configuration values."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SECRET = "secret"


@dataclass
class ConfigSchema:
    """Schema for a configuration value."""
    key: str
    value_type: ConfigValueType
    default: Any = None
    required: bool = False
    description: str = ""
    validators: List[Callable[[Any], bool]] = field(default_factory=list)
    secret: bool = False
    env_var: Optional[str] = None
    choices: Optional[List[Any]] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this schema."""
        if value is None:
            if self.required and self.default is None:
                return False, f"Required config '{self.key}' is missing"
            return True, None

        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # Choices validation
        if self.choices and value not in self.choices:
            return False, f"Value must be one of {self.choices}"

        # Custom validators
        for validator in self.validators:
            try:
                if not validator(value):
                    return False, f"Custom validation failed for '{self.key}'"
            except Exception as e:
                return False, f"Validator error: {e}"

        return True, None

    def _validate_type(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate value type."""
        if self.value_type == ConfigValueType.STRING:
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Expected integer, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Expected float, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Expected boolean, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.LIST:
            if not isinstance(value, list):
                return False, f"Expected list, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.DICT:
            if not isinstance(value, dict):
                return False, f"Expected dict, got {type(value).__name__}"

        return True, None


@dataclass
class ConfigValue(Generic[T]):
    """A configuration value with metadata."""
    key: str
    value: T
    source: str = "default"
    schema: Optional[ConfigSchema] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def as_type(self, target_type: Type[T]) -> T:
        """Convert value to target type."""
        if isinstance(self.value, target_type):
            return self.value

        if target_type == str:
            return str(self.value)
        elif target_type == int:
            return int(self.value)
        elif target_type == float:
            return float(self.value)
        elif target_type == bool:
            if isinstance(self.value, str):
                return self.value.lower() in ("true", "1", "yes", "on")
            return bool(self.value)

        return self.value


class ConfigSource(ABC):
    """Abstract configuration source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Source name."""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority (higher = overrides lower)."""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all values."""
        pass


class DictConfigSource(ConfigSource):
    """Dictionary-based config source."""

    def __init__(
        self,
        data: Dict[str, Any],
        source_name: str = "dict",
        source_priority: int = 0,
    ):
        self._data = data
        self._name = source_name
        self._priority = source_priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def get(self, key: str) -> Optional[Any]:
        # Support nested keys with dot notation
        parts = key.split(".")
        current = self._data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def get_all(self) -> Dict[str, Any]:
        return self._data.copy()


class EnvironmentConfigSource(ConfigSource):
    """Environment variable config source."""

    def __init__(
        self,
        prefix: str = "",
        source_priority: int = 100,
    ):
        self._prefix = prefix
        self._priority = source_priority

    @property
    def name(self) -> str:
        return "environment"

    @property
    def priority(self) -> int:
        return self._priority

    def get(self, key: str) -> Optional[Any]:
        env_key = self._to_env_key(key)
        value = os.environ.get(env_key)

        if value is not None:
            return self._parse_value(value)
        return None

    def get_all(self) -> Dict[str, Any]:
        result = {}
        prefix = self._prefix.upper()

        for key, value in os.environ.items():
            if prefix and key.startswith(prefix):
                config_key = self._from_env_key(key)
                result[config_key] = self._parse_value(value)
            elif not prefix:
                result[key.lower()] = self._parse_value(value)

        return result

    def _to_env_key(self, key: str) -> str:
        """Convert config key to env var name."""
        env_key = key.upper().replace(".", "_")
        if self._prefix:
            return f"{self._prefix.upper()}_{env_key}"
        return env_key

    def _from_env_key(self, env_key: str) -> str:
        """Convert env var name to config key."""
        key = env_key.lower()
        if self._prefix:
            prefix = f"{self._prefix.lower()}_"
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key.replace("_", ".")

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Try JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        return value


class FileConfigSource(ConfigSource):
    """File-based config source."""

    def __init__(
        self,
        path: Union[str, Path],
        source_priority: int = 50,
    ):
        self._path = Path(path)
        self._priority = source_priority
        self._data: Dict[str, Any] = {}
        self._load()

    @property
    def name(self) -> str:
        return f"file:{self._path.name}"

    @property
    def priority(self) -> int:
        return self._priority

    def _load(self) -> None:
        """Load config from file."""
        if not self._path.exists():
            return

        suffix = self._path.suffix.lower()

        try:
            content = self._path.read_text()

            if suffix == ".json":
                self._data = json.loads(content)
            elif suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    self._data = yaml.safe_load(content) or {}
                except ImportError:
                    pass
            elif suffix == ".toml":
                try:
                    import tomllib
                    self._data = tomllib.loads(content)
                except ImportError:
                    pass
        except Exception:
            pass

    def get(self, key: str) -> Optional[Any]:
        parts = key.split(".")
        current = self._data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def get_all(self) -> Dict[str, Any]:
        return self._data.copy()

    def reload(self) -> None:
        """Reload config from file."""
        self._load()


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigValidator:
    """Validates configuration against schemas."""

    def __init__(self, schemas: Optional[List[ConfigSchema]] = None):
        self._schemas: Dict[str, ConfigSchema] = {}
        if schemas:
            for schema in schemas:
                self._schemas[schema.key] = schema

    def add_schema(self, schema: ConfigSchema) -> None:
        """Add a schema."""
        self._schemas[schema.key] = schema

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration."""
        errors = []
        warnings = []

        # Check all schemas
        for key, schema in self._schemas.items():
            value = self._get_nested(config, key)

            if value is None:
                if schema.required:
                    errors.append(f"Required config '{key}' is missing")
                continue

            valid, error = schema.validate(value)
            if not valid:
                errors.append(error or f"Validation failed for '{key}'")

        # Check for unknown keys (warning only)
        known_keys = set(self._schemas.keys())
        config_keys = self._flatten_keys(config)

        for key in config_keys:
            if key not in known_keys:
                # Check if it's a prefix of known keys
                if not any(k.startswith(f"{key}.") for k in known_keys):
                    warnings.append(f"Unknown config key: '{key}'")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _get_nested(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested value by dotted key."""
        parts = key.split(".")
        current = config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _flatten_keys(
        self,
        config: Dict[str, Any],
        prefix: str = "",
    ) -> List[str]:
        """Flatten nested config to dotted keys."""
        keys = []

        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))
            else:
                keys.append(full_key)

        return keys


# Common validators
def min_value(min_val: float) -> Callable[[Any], bool]:
    """Validator for minimum value."""
    return lambda x: x >= min_val


def max_value(max_val: float) -> Callable[[Any], bool]:
    """Validator for maximum value."""
    return lambda x: x <= max_val


def in_range(min_val: float, max_val: float) -> Callable[[Any], bool]:
    """Validator for value in range."""
    return lambda x: min_val <= x <= max_val


def regex_match(pattern: str) -> Callable[[Any], bool]:
    """Validator for regex match."""
    compiled = re.compile(pattern)
    return lambda x: bool(compiled.match(str(x)))


def not_empty() -> Callable[[Any], bool]:
    """Validator for non-empty values."""
    return lambda x: bool(x)
