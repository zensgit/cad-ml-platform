"""Plugin Manager Module.

Provides plugin lifecycle management, dependency injection, and extensibility.
"""

import importlib
import importlib.util
import inspect
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from .base import VisionDescription, VisionProvider

T = TypeVar("T")


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

    PROVIDER = "provider"
    PROCESSOR = "processor"
    FILTER = "filter"
    TRANSFORMER = "transformer"
    ANALYTICS = "analytics"
    STORAGE = "storage"
    AUTHENTICATION = "authentication"
    CUSTOM = "custom"


class HookType(Enum):
    """Types of plugin hooks."""

    PRE_ANALYZE = "pre_analyze"
    POST_ANALYZE = "post_analyze"
    ON_ERROR = "on_error"
    ON_SUCCESS = "on_success"
    ON_LOAD = "on_load"
    ON_UNLOAD = "on_unload"
    ON_CONFIG_CHANGE = "on_config_change"


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    plugin_type: PluginType = PluginType.CUSTOM
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    hooks: List[HookType] = field(default_factory=list)
    priority: int = 100  # Lower = higher priority
    tags: List[str] = field(default_factory=list)


@dataclass
class PluginInfo:
    """Information about an installed plugin."""

    metadata: PluginMetadata
    state: PluginState
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None
    instance: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Result of a hook execution."""

    hook_type: HookType
    plugin_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class Plugin(ABC):
    """Abstract base class for plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        pass

    def on_config_change(self, new_config: Dict[str, Any]) -> None:
        """Handle configuration changes."""
        pass


class VisionPlugin(Plugin):
    """Base class for vision-related plugins."""

    @abstractmethod
    def process(self, image_data: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process image data."""
        pass


class ProviderPlugin(VisionPlugin):
    """Plugin that provides vision analysis."""

    @abstractmethod
    def analyze(self, image_data: bytes, **kwargs: Any) -> VisionDescription:
        """Analyze an image."""
        pass

    def process(self, image_data: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process by analyzing."""
        result = self.analyze(image_data, **context)
        return {"description": result}


class FilterPlugin(VisionPlugin):
    """Plugin that filters image data."""

    @abstractmethod
    def should_process(self, image_data: bytes, context: Dict[str, Any]) -> bool:
        """Determine if image should be processed."""
        pass

    def process(self, image_data: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process by filtering."""
        return {"should_process": self.should_process(image_data, context)}


class TransformerPlugin(VisionPlugin):
    """Plugin that transforms image data or results."""

    @abstractmethod
    def transform(self, data: Any, context: Dict[str, Any]) -> Any:
        """Transform data."""
        pass

    def process(self, image_data: bytes, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process by transforming."""
        return {"transformed": self.transform(image_data, context)}


class DependencyContainer:
    """Simple dependency injection container."""

    def __init__(self):
        self._singletons: Dict[type, Any] = {}
        self._factories: Dict[type, Callable[[], Any]] = {}
        self._lock = threading.Lock()

    def register_singleton(self, interface: type, instance: Any) -> None:
        """Register a singleton instance."""
        with self._lock:
            self._singletons[interface] = instance

    def register_factory(self, interface: type, factory: Callable[[], Any]) -> None:
        """Register a factory function."""
        with self._lock:
            self._factories[interface] = factory

    def resolve(self, interface: type) -> Any:
        """Resolve a dependency."""
        with self._lock:
            if interface in self._singletons:
                return self._singletons[interface]

            if interface in self._factories:
                instance = self._factories[interface]()
                self._singletons[interface] = instance
                return instance

        raise KeyError(f"No registration found for {interface}")

    def has(self, interface: type) -> bool:
        """Check if a dependency is registered."""
        with self._lock:
            return interface in self._singletons or interface in self._factories

    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._singletons.clear()
            self._factories.clear()


class PluginRegistry:
    """Registry for plugin discovery and storage."""

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._by_type: Dict[PluginType, Set[str]] = {t: set() for t in PluginType}
        self._by_hook: Dict[HookType, Set[str]] = {h: set() for h in HookType}
        self._lock = threading.Lock()

    def register(self, plugin: Plugin) -> bool:
        """Register a plugin."""
        with self._lock:
            name = plugin.metadata.name
            if name in self._plugins:
                return False

            info = PluginInfo(
                metadata=plugin.metadata,
                state=PluginState.DISCOVERED,
                instance=plugin,
            )
            self._plugins[name] = info
            self._by_type[plugin.metadata.plugin_type].add(name)

            for hook in plugin.metadata.hooks:
                self._by_hook[hook].add(name)

            return True

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        with self._lock:
            if name not in self._plugins:
                return False

            info = self._plugins.pop(name)
            self._by_type[info.metadata.plugin_type].discard(name)

            for hook in info.metadata.hooks:
                self._by_hook[hook].discard(name)

            return True

    def get(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        with self._lock:
            return self._plugins.get(name)

    def get_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get plugins by type."""
        with self._lock:
            names = self._by_type.get(plugin_type, set())
            return [self._plugins[n] for n in names if n in self._plugins]

    def get_by_hook(self, hook_type: HookType) -> List[PluginInfo]:
        """Get plugins that implement a hook."""
        with self._lock:
            names = self._by_hook.get(hook_type, set())
            return [self._plugins[n] for n in names if n in self._plugins]

    def get_all(self) -> List[PluginInfo]:
        """Get all plugins."""
        with self._lock:
            return list(self._plugins.values())

    def update_state(self, name: str, state: PluginState) -> None:
        """Update plugin state."""
        with self._lock:
            if name in self._plugins:
                self._plugins[name].state = state


class HookExecutor:
    """Executes plugin hooks."""

    def __init__(self, registry: PluginRegistry):
        self._registry = registry

    def execute(
        self,
        hook_type: HookType,
        context: Dict[str, Any],
        stop_on_error: bool = False,
    ) -> List[HookResult]:
        """Execute all plugins for a hook."""
        results = []
        plugins = self._registry.get_by_hook(hook_type)

        # Sort by priority
        plugins.sort(key=lambda p: p.metadata.priority)

        for plugin_info in plugins:
            if plugin_info.state != PluginState.ACTIVE:
                continue

            start_time = time.time()
            try:
                hook_method = getattr(plugin_info.instance, hook_type.value, None)
                if hook_method and callable(hook_method):
                    result = hook_method(context)
                    results.append(
                        HookResult(
                            hook_type=hook_type,
                            plugin_name=plugin_info.metadata.name,
                            success=True,
                            result=result,
                            execution_time_ms=(time.time() - start_time) * 1000,
                        )
                    )
                else:
                    results.append(
                        HookResult(
                            hook_type=hook_type,
                            plugin_name=plugin_info.metadata.name,
                            success=True,
                            result=None,
                            execution_time_ms=(time.time() - start_time) * 1000,
                        )
                    )
            except Exception as e:
                results.append(
                    HookResult(
                        hook_type=hook_type,
                        plugin_name=plugin_info.metadata.name,
                        success=False,
                        error=str(e),
                        execution_time_ms=(time.time() - start_time) * 1000,
                    )
                )
                if stop_on_error:
                    break

        return results


class PluginLoader:
    """Loads plugins from various sources."""

    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}

    def load_from_class(self, plugin_class: Type[Plugin]) -> Plugin:
        """Load a plugin from a class."""
        return plugin_class()

    def load_from_module(self, module_path: str, class_name: str) -> Plugin:
        """Load a plugin from a module."""
        if module_path in self._loaded_modules:
            module = self._loaded_modules[module_path]
        else:
            module = importlib.import_module(module_path)
            self._loaded_modules[module_path] = module

        plugin_class = getattr(module, class_name)
        return plugin_class()

    def load_from_file(self, file_path: str, class_name: str) -> Plugin:
        """Load a plugin from a Python file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plugin from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        plugin_class = getattr(module, class_name)
        return plugin_class()

    def discover_plugins(self, directory: str) -> List[Plugin]:
        """Discover plugins in a directory."""
        plugins = []
        path = Path(directory)

        if not path.exists():
            return plugins

        for file_path in path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find Plugin subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, Plugin)
                        and obj is not Plugin
                        and obj is not VisionPlugin
                        and obj is not ProviderPlugin
                        and obj is not FilterPlugin
                        and obj is not TransformerPlugin
                    ):
                        plugins.append(obj())

            except Exception:
                continue

        return plugins


class PluginManager:
    """Main plugin manager coordinating all plugin operations."""

    def __init__(self):
        self._registry = PluginRegistry()
        self._loader = PluginLoader()
        self._hook_executor = HookExecutor(self._registry)
        self._container = DependencyContainer()
        self._lock = threading.Lock()

    def register_plugin(self, plugin: Plugin) -> bool:
        """Register and initialize a plugin."""
        if not self._registry.register(plugin):
            return False

        name = plugin.metadata.name

        # Check dependencies
        for dep in plugin.metadata.dependencies:
            dep_info = self._registry.get(dep)
            if not dep_info or dep_info.state != PluginState.ACTIVE:
                self._registry.update_state(name, PluginState.FAILED)
                info = self._registry.get(name)
                if info:
                    info.error_message = f"Missing dependency: {dep}"
                return False

        # Initialize plugin
        try:
            config = self._registry.get(name)
            if config:
                plugin.initialize(config.config)
                self._registry.update_state(name, PluginState.INITIALIZED)
                config.load_time = datetime.now()
        except Exception as e:
            self._registry.update_state(name, PluginState.FAILED)
            info = self._registry.get(name)
            if info:
                info.error_message = str(e)
            return False

        return True

    def activate_plugin(self, name: str) -> bool:
        """Activate a plugin."""
        info = self._registry.get(name)
        if not info:
            return False

        if info.state not in (PluginState.INITIALIZED, PluginState.SUSPENDED):
            return False

        self._registry.update_state(name, PluginState.ACTIVE)

        # Execute on_load hook
        self._hook_executor.execute(
            HookType.ON_LOAD,
            {"plugin_name": name},
        )

        return True

    def deactivate_plugin(self, name: str) -> bool:
        """Deactivate a plugin."""
        info = self._registry.get(name)
        if not info or info.state != PluginState.ACTIVE:
            return False

        # Execute on_unload hook
        self._hook_executor.execute(
            HookType.ON_UNLOAD,
            {"plugin_name": name},
        )

        self._registry.update_state(name, PluginState.SUSPENDED)
        return True

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin."""
        info = self._registry.get(name)
        if not info:
            return False

        if info.state == PluginState.ACTIVE:
            self.deactivate_plugin(name)

        if info.instance:
            try:
                info.instance.shutdown()
            except Exception:
                pass

        return self._registry.unregister(name)

    def load_from_directory(self, directory: str) -> int:
        """Load plugins from a directory."""
        plugins = self._loader.discover_plugins(directory)
        loaded = 0

        for plugin in plugins:
            if self.register_plugin(plugin):
                if self.activate_plugin(plugin.metadata.name):
                    loaded += 1

        return loaded

    def execute_hook(
        self,
        hook_type: HookType,
        context: Dict[str, Any],
    ) -> List[HookResult]:
        """Execute a hook on all active plugins."""
        return self._hook_executor.execute(hook_type, context)

    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin by name."""
        return self._registry.get(name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get plugins by type."""
        return self._registry.get_by_type(plugin_type)

    def get_active_plugins(self) -> List[PluginInfo]:
        """Get all active plugins."""
        return [p for p in self._registry.get_all() if p.state == PluginState.ACTIVE]

    def get_all_plugins(self) -> List[PluginInfo]:
        """Get all plugins."""
        return self._registry.get_all()

    def update_config(self, name: str, config: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
        info = self._registry.get(name)
        if not info:
            return False

        info.config = config

        if info.instance and info.state == PluginState.ACTIVE:
            try:
                info.instance.on_config_change(config)
            except Exception:
                pass

        return True

    @property
    def container(self) -> DependencyContainer:
        """Get the dependency container."""
        return self._container


class PluginVisionProvider(VisionProvider):
    """Vision provider with plugin support."""

    def __init__(
        self,
        wrapped_provider: VisionProvider,
        manager: Optional[PluginManager] = None,
    ):
        self._wrapped = wrapped_provider
        self.manager = manager or PluginManager()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"pluggable_{self._wrapped.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with plugin hooks."""
        context = {
            "image_data": image_data,
            "include_description": include_description,
            "kwargs": kwargs,
        }

        # Pre-analyze hooks
        self.manager.execute_hook(HookType.PRE_ANALYZE, context)

        try:
            result = await self._wrapped.analyze_image(image_data, include_description, **kwargs)

            # Post-analyze hooks
            context["result"] = result
            self.manager.execute_hook(HookType.POST_ANALYZE, context)

            # On success hooks
            self.manager.execute_hook(HookType.ON_SUCCESS, context)

            return result

        except Exception as e:
            # On error hooks
            context["error"] = str(e)
            self.manager.execute_hook(HookType.ON_ERROR, context)
            raise


# Factory functions
def create_plugin_manager() -> PluginManager:
    """Create a plugin manager."""
    return PluginManager()


def create_plugin_provider(
    provider: VisionProvider,
    manager: Optional[PluginManager] = None,
) -> PluginVisionProvider:
    """Create a pluggable vision provider."""
    return PluginVisionProvider(provider, manager)


def create_dependency_container() -> DependencyContainer:
    """Create a dependency container."""
    return DependencyContainer()


def create_plugin_metadata(
    name: str,
    version: str,
    plugin_type: PluginType = PluginType.CUSTOM,
    description: str = "",
    dependencies: Optional[List[str]] = None,
    hooks: Optional[List[HookType]] = None,
) -> PluginMetadata:
    """Create plugin metadata."""
    return PluginMetadata(
        name=name,
        version=version,
        description=description,
        plugin_type=plugin_type,
        dependencies=dependencies or [],
        hooks=hooks or [],
    )
