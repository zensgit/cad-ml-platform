"""
Plugin System for Vision Provider.

This module provides an extensible plugin architecture including:
- Plugin discovery and loading
- Plugin lifecycle management
- Hook system for extensibility
- Plugin sandboxing and isolation
- Plugin dependency management
- Plugin marketplace patterns

Phase 10 Feature.
"""

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


# ============================================================================
# Plugin Enums
# ============================================================================


class PluginState(Enum):
    """Plugin lifecycle states."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    FAILED = "failed"
    UNLOADED = "unloaded"


class PluginType(Enum):
    """Types of plugins."""

    PROVIDER = "provider"  # Vision provider plugins
    PREPROCESSOR = "preprocessor"  # Image preprocessing
    POSTPROCESSOR = "postprocessor"  # Result postprocessing
    FILTER = "filter"  # Content filtering
    ANALYTICS = "analytics"  # Analytics plugins
    STORAGE = "storage"  # Storage backends
    CACHE = "cache"  # Caching plugins
    MIDDLEWARE = "middleware"  # Request/response middleware
    EXTENSION = "extension"  # General extensions


class HookPriority(Enum):
    """Hook execution priority."""

    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100
    CRITICAL = 150


class PluginCapability(Enum):
    """Plugin capabilities for sandboxing."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    ENVIRONMENT = "environment"
    FULL_ACCESS = "full_access"


# ============================================================================
# Plugin Data Classes
# ============================================================================


@dataclass
class PluginMetadata:
    """Plugin metadata and manifest."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    plugin_type: PluginType = PluginType.EXTENSION
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[PluginCapability] = field(default_factory=list)
    entry_point: str = ""
    config_schema: Optional[Dict[str, Any]] = None
    min_system_version: str = "1.0.0"
    max_system_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    homepage: str = ""
    license: str = ""
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "capabilities": [c.value for c in self.capabilities],
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "min_system_version": self.min_system_version,
            "max_system_version": self.max_system_version,
            "tags": self.tags,
            "homepage": self.homepage,
            "license": self.license,
            "checksum": self.checksum,
        }


@dataclass
class PluginInstance:
    """Runtime plugin instance."""

    metadata: PluginMetadata
    state: PluginState = PluginState.DISCOVERED
    instance: Optional[Any] = None
    module: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[datetime] = None
    error: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookRegistration:
    """Hook registration details."""

    hook_name: str
    callback: Callable[..., Any]
    priority: HookPriority = HookPriority.NORMAL
    plugin_name: Optional[str] = None
    async_handler: bool = False
    once: bool = False
    called_count: int = 0


@dataclass
class PluginEvent:
    """Plugin system event."""

    event_type: str
    plugin_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Plugin Base Class
# ============================================================================


class Plugin(ABC):
    """Abstract base class for plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def activate(self) -> None:
        """Activate the plugin."""
        pass

    def deactivate(self) -> None:
        """Deactivate the plugin."""
        pass

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass

    def get_hooks(self) -> Dict[str, Callable[..., Any]]:
        """Return hooks provided by this plugin."""
        return {}


class VisionProviderPlugin(Plugin):
    """Base class for vision provider plugins."""

    @abstractmethod
    def get_provider(self) -> VisionProvider:
        """Return the vision provider instance."""
        pass


class PreprocessorPlugin(Plugin):
    """Base class for preprocessor plugins."""

    @abstractmethod
    def preprocess(self, image_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Preprocess image data."""
        pass


class PostprocessorPlugin(Plugin):
    """Base class for postprocessor plugins."""

    @abstractmethod
    def postprocess(
        self, description: VisionDescription, metadata: Dict[str, Any]
    ) -> VisionDescription:
        """Postprocess vision description."""
        pass


class MiddlewarePlugin(Plugin):
    """Base class for middleware plugins."""

    @abstractmethod
    def process_request(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process incoming request."""
        pass

    @abstractmethod
    def process_response(
        self, response: VisionDescription, request: Dict[str, Any]
    ) -> VisionDescription:
        """Process outgoing response."""
        pass


# ============================================================================
# Hook System
# ============================================================================


class HookManager:
    """Manages hooks for plugin extensibility."""

    def __init__(self) -> None:
        """Initialize hook manager."""
        self._hooks: Dict[str, List[HookRegistration]] = {}
        self._lock = asyncio.Lock()

    def register(
        self,
        hook_name: str,
        callback: Callable[..., Any],
        priority: HookPriority = HookPriority.NORMAL,
        plugin_name: Optional[str] = None,
        once: bool = False,
    ) -> HookRegistration:
        """Register a hook handler."""
        is_async = asyncio.iscoroutinefunction(callback)

        registration = HookRegistration(
            hook_name=hook_name,
            callback=callback,
            priority=priority,
            plugin_name=plugin_name,
            async_handler=is_async,
            once=once,
        )

        if hook_name not in self._hooks:
            self._hooks[hook_name] = []

        self._hooks[hook_name].append(registration)
        # Sort by priority (highest first)
        self._hooks[hook_name].sort(key=lambda h: h.priority.value, reverse=True)

        logger.debug(
            f"Registered hook '{hook_name}' with priority {priority.name}"
            f" from plugin '{plugin_name}'"
        )

        return registration

    def unregister(
        self, hook_name: str, plugin_name: Optional[str] = None
    ) -> int:
        """Unregister hooks by name and optionally plugin."""
        if hook_name not in self._hooks:
            return 0

        original_count = len(self._hooks[hook_name])

        if plugin_name:
            self._hooks[hook_name] = [
                h for h in self._hooks[hook_name] if h.plugin_name != plugin_name
            ]
        else:
            self._hooks[hook_name] = []

        removed = original_count - len(self._hooks[hook_name])
        logger.debug(f"Unregistered {removed} hooks for '{hook_name}'")

        return removed

    def trigger(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """Trigger a hook synchronously."""
        if hook_name not in self._hooks:
            return []

        results: List[Any] = []
        to_remove: List[HookRegistration] = []

        for registration in self._hooks[hook_name]:
            if registration.async_handler:
                logger.warning(
                    f"Async handler for hook '{hook_name}' called synchronously"
                )
                continue

            try:
                result = registration.callback(*args, **kwargs)
                results.append(result)
                registration.called_count += 1

                if registration.once:
                    to_remove.append(registration)

            except Exception as e:
                logger.error(
                    f"Error in hook '{hook_name}' from plugin "
                    f"'{registration.plugin_name}': {e}"
                )

        # Remove one-time hooks
        for reg in to_remove:
            self._hooks[hook_name].remove(reg)

        return results

    async def trigger_async(
        self, hook_name: str, *args: Any, **kwargs: Any
    ) -> List[Any]:
        """Trigger a hook asynchronously."""
        if hook_name not in self._hooks:
            return []

        results: List[Any] = []
        to_remove: List[HookRegistration] = []

        async with self._lock:
            for registration in self._hooks[hook_name]:
                try:
                    if registration.async_handler:
                        result = await registration.callback(*args, **kwargs)
                    else:
                        result = registration.callback(*args, **kwargs)

                    results.append(result)
                    registration.called_count += 1

                    if registration.once:
                        to_remove.append(registration)

                except Exception as e:
                    logger.error(
                        f"Error in async hook '{hook_name}' from plugin "
                        f"'{registration.plugin_name}': {e}"
                    )

            # Remove one-time hooks
            for reg in to_remove:
                self._hooks[hook_name].remove(reg)

        return results

    def get_registered_hooks(self) -> Dict[str, int]:
        """Get all registered hooks and their handler counts."""
        return {name: len(handlers) for name, handlers in self._hooks.items()}

    def clear_all(self) -> None:
        """Clear all hooks."""
        self._hooks.clear()


# ============================================================================
# Plugin Discovery
# ============================================================================


class PluginDiscovery:
    """Discovers plugins from various sources."""

    def __init__(self, plugin_dirs: Optional[List[str]] = None) -> None:
        """Initialize plugin discovery."""
        self._plugin_dirs = plugin_dirs or []
        self._discovered: Dict[str, PluginMetadata] = {}

    def add_directory(self, directory: str) -> None:
        """Add a directory to search for plugins."""
        if directory not in self._plugin_dirs:
            self._plugin_dirs.append(directory)

    def discover(self) -> List[PluginMetadata]:
        """Discover all plugins in configured directories."""
        discovered: List[PluginMetadata] = []

        for plugin_dir in self._plugin_dirs:
            path = Path(plugin_dir)
            if not path.exists():
                logger.warning(f"Plugin directory does not exist: {plugin_dir}")
                continue

            discovered.extend(self._discover_in_directory(path))

        return discovered

    def _discover_in_directory(self, directory: Path) -> List[PluginMetadata]:
        """Discover plugins in a specific directory."""
        discovered: List[PluginMetadata] = []

        for item in directory.iterdir():
            if item.is_dir():
                # Check for plugin package
                init_file = item / "__init__.py"
                manifest_file = item / "plugin.json"

                if init_file.exists():
                    metadata = self._load_plugin_metadata(item, manifest_file)
                    if metadata:
                        discovered.append(metadata)
                        self._discovered[metadata.name] = metadata

            elif item.suffix == ".py" and item.name != "__init__.py":
                # Single file plugin
                metadata = self._load_single_file_plugin(item)
                if metadata:
                    discovered.append(metadata)
                    self._discovered[metadata.name] = metadata

        return discovered

    def _load_plugin_metadata(
        self, plugin_dir: Path, manifest_file: Path
    ) -> Optional[PluginMetadata]:
        """Load plugin metadata from manifest or module."""
        try:
            if manifest_file.exists():
                import json

                with open(manifest_file) as f:
                    data = json.load(f)

                return PluginMetadata(
                    name=data.get("name", plugin_dir.name),
                    version=data.get("version", "1.0.0"),
                    description=data.get("description", ""),
                    author=data.get("author", ""),
                    plugin_type=PluginType(data.get("type", "extension")),
                    dependencies=data.get("dependencies", []),
                    capabilities=[
                        PluginCapability(c) for c in data.get("capabilities", [])
                    ],
                    entry_point=str(plugin_dir / "__init__.py"),
                    config_schema=data.get("config_schema"),
                    tags=data.get("tags", []),
                )
            else:
                # Infer metadata from directory name
                return PluginMetadata(
                    name=plugin_dir.name,
                    version="1.0.0",
                    entry_point=str(plugin_dir / "__init__.py"),
                )

        except Exception as e:
            logger.error(f"Error loading plugin metadata from {plugin_dir}: {e}")
            return None

    def _load_single_file_plugin(self, plugin_file: Path) -> Optional[PluginMetadata]:
        """Load metadata from a single-file plugin."""
        try:
            # Calculate checksum
            with open(plugin_file, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            return PluginMetadata(
                name=plugin_file.stem,
                version="1.0.0",
                entry_point=str(plugin_file),
                checksum=checksum,
            )

        except Exception as e:
            logger.error(f"Error loading single file plugin {plugin_file}: {e}")
            return None


# ============================================================================
# Plugin Sandbox
# ============================================================================


class PluginSandbox:
    """Sandboxes plugin execution for isolation."""

    def __init__(
        self,
        allowed_capabilities: Optional[List[PluginCapability]] = None,
    ) -> None:
        """Initialize sandbox."""
        self._allowed = set(allowed_capabilities or [])
        self._restricted_modules: Set[str] = {
            "os",
            "subprocess",
            "shutil",
            "sys",
            "importlib",
        }

    def check_capability(self, capability: PluginCapability) -> bool:
        """Check if capability is allowed."""
        if PluginCapability.FULL_ACCESS in self._allowed:
            return True
        return capability in self._allowed

    def validate_plugin(self, metadata: PluginMetadata) -> List[str]:
        """Validate plugin capabilities against sandbox restrictions."""
        violations: List[str] = []

        for cap in metadata.capabilities:
            if not self.check_capability(cap):
                violations.append(
                    f"Plugin requires capability '{cap.value}' which is not allowed"
                )

        return violations

    def create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace for plugin execution."""
        restricted: Dict[str, Any] = {
            "__builtins__": {
                # Allow safe builtins
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "type": type,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "print": print,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
            }
        }

        return restricted


# ============================================================================
# Plugin Manager
# ============================================================================


class PluginManager:
    """
    Central plugin management system.

    Handles plugin lifecycle, dependency resolution, and execution.
    """

    def __init__(
        self,
        plugin_dirs: Optional[List[str]] = None,
        sandbox: Optional[PluginSandbox] = None,
    ) -> None:
        """Initialize plugin manager."""
        self._discovery = PluginDiscovery(plugin_dirs)
        self._sandbox = sandbox or PluginSandbox()
        self._hooks = HookManager()
        self._plugins: Dict[str, PluginInstance] = {}
        self._event_handlers: List[Callable[[PluginEvent], None]] = []
        self._lock = asyncio.Lock()

    @property
    def hooks(self) -> HookManager:
        """Get hook manager."""
        return self._hooks

    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins."""
        discovered = self._discovery.discover()

        for metadata in discovered:
            if metadata.name not in self._plugins:
                self._plugins[metadata.name] = PluginInstance(
                    metadata=metadata,
                    state=PluginState.DISCOVERED,
                )

        self._emit_event("plugins_discovered", "", {"count": len(discovered)})
        return discovered

    def load_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> PluginInstance:
        """Load a discovered plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found")

        instance = self._plugins[name]

        if instance.state not in [PluginState.DISCOVERED, PluginState.UNLOADED]:
            logger.warning(f"Plugin '{name}' is already loaded (state: {instance.state})")
            return instance

        # Validate against sandbox
        violations = self._sandbox.validate_plugin(instance.metadata)
        if violations:
            instance.state = PluginState.FAILED
            instance.error = "; ".join(violations)
            self._emit_event("plugin_load_failed", name, {"error": instance.error})
            raise PermissionError(f"Plugin sandbox violations: {violations}")

        # Load dependencies first
        for dep in instance.metadata.dependencies:
            if dep not in self._plugins:
                instance.state = PluginState.FAILED
                instance.error = f"Missing dependency: {dep}"
                raise ValueError(f"Missing dependency '{dep}' for plugin '{name}'")

            dep_instance = self._plugins[dep]
            if dep_instance.state not in [PluginState.ACTIVE, PluginState.INITIALIZED]:
                self.load_plugin(dep)

        # Load the plugin module
        try:
            module = self._load_module(instance.metadata)
            instance.module = module

            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if plugin_class:
                instance.instance = plugin_class()
                instance.config = config or {}
                instance.state = PluginState.LOADED
                instance.loaded_at = datetime.now()

                self._emit_event("plugin_loaded", name)
                logger.info(f"Loaded plugin: {name}")

        except Exception as e:
            instance.state = PluginState.FAILED
            instance.error = str(e)
            self._emit_event("plugin_load_failed", name, {"error": str(e)})
            raise

        return instance

    def initialize_plugin(self, name: str) -> PluginInstance:
        """Initialize a loaded plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found")

        instance = self._plugins[name]

        if instance.state != PluginState.LOADED:
            raise ValueError(
                f"Plugin '{name}' must be loaded before initialization "
                f"(current state: {instance.state})"
            )

        try:
            if instance.instance:
                instance.instance.initialize(instance.config)
                instance.state = PluginState.INITIALIZED

                self._emit_event("plugin_initialized", name)
                logger.info(f"Initialized plugin: {name}")

        except Exception as e:
            instance.state = PluginState.FAILED
            instance.error = str(e)
            self._emit_event("plugin_init_failed", name, {"error": str(e)})
            raise

        return instance

    def activate_plugin(self, name: str) -> PluginInstance:
        """Activate an initialized plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found")

        instance = self._plugins[name]

        if instance.state != PluginState.INITIALIZED:
            raise ValueError(
                f"Plugin '{name}' must be initialized before activation "
                f"(current state: {instance.state})"
            )

        try:
            if instance.instance:
                instance.instance.activate()

                # Register hooks from plugin
                hooks = instance.instance.get_hooks()
                for hook_name, callback in hooks.items():
                    self._hooks.register(
                        hook_name=hook_name,
                        callback=callback,
                        plugin_name=name,
                    )

                instance.state = PluginState.ACTIVE
                self._emit_event("plugin_activated", name)
                logger.info(f"Activated plugin: {name}")

        except Exception as e:
            instance.state = PluginState.FAILED
            instance.error = str(e)
            self._emit_event("plugin_activate_failed", name, {"error": str(e)})
            raise

        return instance

    def deactivate_plugin(self, name: str) -> PluginInstance:
        """Deactivate an active plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found")

        instance = self._plugins[name]

        if instance.state != PluginState.ACTIVE:
            logger.warning(f"Plugin '{name}' is not active")
            return instance

        try:
            # Unregister hooks
            for hook_name in self._hooks.get_registered_hooks():
                self._hooks.unregister(hook_name, plugin_name=name)

            if instance.instance:
                instance.instance.deactivate()

            instance.state = PluginState.SUSPENDED
            self._emit_event("plugin_deactivated", name)
            logger.info(f"Deactivated plugin: {name}")

        except Exception as e:
            logger.error(f"Error deactivating plugin '{name}': {e}")

        return instance

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin completely."""
        if name not in self._plugins:
            return

        instance = self._plugins[name]

        # Deactivate first if active
        if instance.state == PluginState.ACTIVE:
            self.deactivate_plugin(name)

        try:
            if instance.instance:
                instance.instance.cleanup()

            instance.instance = None
            instance.module = None
            instance.state = PluginState.UNLOADED

            self._emit_event("plugin_unloaded", name)
            logger.info(f"Unloaded plugin: {name}")

        except Exception as e:
            logger.error(f"Error unloading plugin '{name}': {e}")

    def get_plugin(self, name: str) -> Optional[PluginInstance]:
        """Get a plugin instance by name."""
        return self._plugins.get(name)

    def get_active_plugins(self) -> List[PluginInstance]:
        """Get all active plugins."""
        return [p for p in self._plugins.values() if p.state == PluginState.ACTIVE]

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInstance]:
        """Get plugins by type."""
        return [
            p for p in self._plugins.values()
            if p.metadata.plugin_type == plugin_type
        ]

    def on_event(self, handler: Callable[[PluginEvent], None]) -> None:
        """Register event handler."""
        self._event_handlers.append(handler)

    def _emit_event(
        self, event_type: str, plugin_name: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit a plugin event."""
        event = PluginEvent(
            event_type=event_type,
            plugin_name=plugin_name,
            data=data or {},
        )

        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def _load_module(self, metadata: PluginMetadata) -> Any:
        """Load plugin module from file."""
        entry_point = Path(metadata.entry_point)

        if not entry_point.exists():
            raise FileNotFoundError(f"Plugin entry point not found: {entry_point}")

        spec = importlib.util.spec_from_file_location(metadata.name, entry_point)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin module: {metadata.name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[metadata.name] = module
        spec.loader.exec_module(module)

        return module

    def _find_plugin_class(self, module: Any) -> Optional[Type[Plugin]]:
        """Find the plugin class in a module."""
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and not inspect.isabstract(obj)
            ):
                return obj  # type: ignore

        return None


# ============================================================================
# Plugin Registry
# ============================================================================


class PluginRegistry:
    """
    Registry for plugin types and factories.

    Allows dynamic registration and lookup of plugin implementations.
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._providers: Dict[str, Type[VisionProviderPlugin]] = {}
        self._preprocessors: Dict[str, Type[PreprocessorPlugin]] = {}
        self._postprocessors: Dict[str, Type[PostprocessorPlugin]] = {}
        self._middleware: Dict[str, Type[MiddlewarePlugin]] = {}
        self._extensions: Dict[str, Type[Plugin]] = {}

    def register_provider(
        self, name: str, plugin_class: Type[VisionProviderPlugin]
    ) -> None:
        """Register a vision provider plugin."""
        self._providers[name] = plugin_class
        logger.debug(f"Registered provider plugin: {name}")

    def register_preprocessor(
        self, name: str, plugin_class: Type[PreprocessorPlugin]
    ) -> None:
        """Register a preprocessor plugin."""
        self._preprocessors[name] = plugin_class
        logger.debug(f"Registered preprocessor plugin: {name}")

    def register_postprocessor(
        self, name: str, plugin_class: Type[PostprocessorPlugin]
    ) -> None:
        """Register a postprocessor plugin."""
        self._postprocessors[name] = plugin_class
        logger.debug(f"Registered postprocessor plugin: {name}")

    def register_middleware(
        self, name: str, plugin_class: Type[MiddlewarePlugin]
    ) -> None:
        """Register a middleware plugin."""
        self._middleware[name] = plugin_class
        logger.debug(f"Registered middleware plugin: {name}")

    def register_extension(self, name: str, plugin_class: Type[Plugin]) -> None:
        """Register a general extension plugin."""
        self._extensions[name] = plugin_class
        logger.debug(f"Registered extension plugin: {name}")

    def get_provider(self, name: str) -> Optional[Type[VisionProviderPlugin]]:
        """Get a provider plugin class."""
        return self._providers.get(name)

    def get_preprocessor(self, name: str) -> Optional[Type[PreprocessorPlugin]]:
        """Get a preprocessor plugin class."""
        return self._preprocessors.get(name)

    def get_postprocessor(self, name: str) -> Optional[Type[PostprocessorPlugin]]:
        """Get a postprocessor plugin class."""
        return self._postprocessors.get(name)

    def get_middleware(self, name: str) -> Optional[Type[MiddlewarePlugin]]:
        """Get a middleware plugin class."""
        return self._middleware.get(name)

    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self._providers.keys())

    def list_preprocessors(self) -> List[str]:
        """List all registered preprocessors."""
        return list(self._preprocessors.keys())

    def list_postprocessors(self) -> List[str]:
        """List all registered postprocessors."""
        return list(self._postprocessors.keys())

    def list_middleware(self) -> List[str]:
        """List all registered middleware."""
        return list(self._middleware.keys())


# ============================================================================
# Plugin Pipeline
# ============================================================================


class PluginPipeline:
    """
    Pipeline for executing plugins in sequence.

    Supports preprocessing, provider selection, and postprocessing.
    """

    def __init__(
        self,
        preprocessors: Optional[List[PreprocessorPlugin]] = None,
        postprocessors: Optional[List[PostprocessorPlugin]] = None,
        middleware: Optional[List[MiddlewarePlugin]] = None,
    ) -> None:
        """Initialize pipeline."""
        self._preprocessors = preprocessors or []
        self._postprocessors = postprocessors or []
        self._middleware = middleware or []

    def add_preprocessor(self, plugin: PreprocessorPlugin) -> None:
        """Add a preprocessor to the pipeline."""
        self._preprocessors.append(plugin)

    def add_postprocessor(self, plugin: PostprocessorPlugin) -> None:
        """Add a postprocessor to the pipeline."""
        self._postprocessors.append(plugin)

    def add_middleware(self, plugin: MiddlewarePlugin) -> None:
        """Add middleware to the pipeline."""
        self._middleware.append(plugin)

    def preprocess(self, image_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Run all preprocessors on image data."""
        result = image_data

        for preprocessor in self._preprocessors:
            try:
                result = preprocessor.preprocess(result, metadata)
            except Exception as e:
                logger.error(f"Preprocessor error: {e}")

        return result

    def postprocess(
        self, description: VisionDescription, metadata: Dict[str, Any]
    ) -> VisionDescription:
        """Run all postprocessors on description."""
        result = description

        for postprocessor in self._postprocessors:
            try:
                result = postprocessor.postprocess(result, metadata)
            except Exception as e:
                logger.error(f"Postprocessor error: {e}")

        return result

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run middleware request processing."""
        result = request

        for mw in self._middleware:
            try:
                result = mw.process_request(result)
            except Exception as e:
                logger.error(f"Middleware request error: {e}")

        return result

    def process_response(
        self, response: VisionDescription, request: Dict[str, Any]
    ) -> VisionDescription:
        """Run middleware response processing."""
        result = response

        # Process in reverse order for response
        for mw in reversed(self._middleware):
            try:
                result = mw.process_response(result, request)
            except Exception as e:
                logger.error(f"Middleware response error: {e}")

        return result


# ============================================================================
# Pluggable Vision Provider
# ============================================================================


class PluggableVisionProvider(VisionProvider):
    """
    Vision provider with plugin support.

    Integrates with the plugin system for extensibility.
    """

    def __init__(
        self,
        base_provider: VisionProvider,
        plugin_manager: PluginManager,
        pipeline: Optional[PluginPipeline] = None,
    ) -> None:
        """Initialize pluggable provider."""
        self._base_provider = base_provider
        self._plugin_manager = plugin_manager
        self._pipeline = pipeline or PluginPipeline()

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"pluggable_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, context: Optional[str] = None
    ) -> VisionDescription:
        """Analyze image with plugin processing."""
        metadata: Dict[str, Any] = {
            "context": context,
            "provider": self._base_provider.provider_name,
        }

        # Trigger pre-analysis hooks
        self._plugin_manager.hooks.trigger(
            "pre_analysis", image_data=image_data, metadata=metadata
        )

        # Preprocess image
        processed_image = self._pipeline.preprocess(image_data, metadata)

        # Build request
        request: Dict[str, Any] = {
            "image_data": processed_image,
            "context": context,
            "metadata": metadata,
        }

        # Process request through middleware
        request = self._pipeline.process_request(request)

        # Analyze with base provider
        description = await self._base_provider.analyze_image(
            request["image_data"],
            request.get("context"),
        )

        # Postprocess result
        description = self._pipeline.postprocess(description, metadata)

        # Process response through middleware
        description = self._pipeline.process_response(description, request)

        # Trigger post-analysis hooks
        self._plugin_manager.hooks.trigger(
            "post_analysis", description=description, metadata=metadata
        )

        return description


# ============================================================================
# Factory Functions
# ============================================================================


def create_plugin_manager(
    plugin_dirs: Optional[List[str]] = None,
    allowed_capabilities: Optional[List[PluginCapability]] = None,
) -> PluginManager:
    """Create a configured plugin manager."""
    sandbox = PluginSandbox(allowed_capabilities=allowed_capabilities)
    return PluginManager(plugin_dirs=plugin_dirs, sandbox=sandbox)


def create_plugin_pipeline(
    plugin_manager: PluginManager,
) -> PluginPipeline:
    """Create a plugin pipeline from active plugins."""
    pipeline = PluginPipeline()

    for instance in plugin_manager.get_plugins_by_type(PluginType.PREPROCESSOR):
        if instance.state == PluginState.ACTIVE and instance.instance:
            if isinstance(instance.instance, PreprocessorPlugin):
                pipeline.add_preprocessor(instance.instance)

    for instance in plugin_manager.get_plugins_by_type(PluginType.POSTPROCESSOR):
        if instance.state == PluginState.ACTIVE and instance.instance:
            if isinstance(instance.instance, PostprocessorPlugin):
                pipeline.add_postprocessor(instance.instance)

    for instance in plugin_manager.get_plugins_by_type(PluginType.MIDDLEWARE):
        if instance.state == PluginState.ACTIVE and instance.instance:
            if isinstance(instance.instance, MiddlewarePlugin):
                pipeline.add_middleware(instance.instance)

    return pipeline
