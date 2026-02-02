"""Configuration Sources Implementation.

Provides various configuration source backends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from src.core.config.manager import ConfigSource, ConfigPriority, ConfigChangeEvent

logger = logging.getLogger(__name__)


class EnvConfigSource(ConfigSource):
    """Environment variable configuration source."""

    def __init__(
        self,
        prefix: str = "",
        priority: int = ConfigPriority.ENVIRONMENT,
    ):
        super().__init__("env", priority)
        self.prefix = prefix.upper() + "_" if prefix else ""

    def _env_key(self, key: str) -> str:
        """Convert config key to environment variable name."""
        return self.prefix + key.upper().replace(".", "_")

    async def get(self, key: str) -> Optional[Any]:
        env_key = self._env_key(key)
        value = os.getenv(env_key)
        if value is not None:
            # Try to parse as JSON for complex types
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def get_all(self, prefix: str = "") -> Dict[str, Any]:
        result = {}
        search_prefix = self._env_key(prefix) if prefix else self.prefix

        for key, value in os.environ.items():
            if key.startswith(search_prefix):
                # Convert back to config key format
                config_key = key[len(self.prefix):].lower().replace("_", ".")
                try:
                    result[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    result[config_key] = value

        return result


class FileConfigSource(ConfigSource):
    """File-based configuration source (JSON/YAML)."""

    def __init__(
        self,
        file_path: str,
        priority: int = ConfigPriority.FILE,
        watch_changes: bool = True,
    ):
        super().__init__(f"file:{file_path}", priority)
        self.file_path = Path(file_path)
        self.watch_changes = watch_changes
        self._config: Dict[str, Any] = {}
        self._last_modified: float = 0
        self._watchers: Dict[str, list] = {}
        self._watch_task: Optional[asyncio.Task] = None

    async def _load(self) -> None:
        """Load configuration from file."""
        if not self.file_path.exists():
            logger.warning(f"Config file not found: {self.file_path}")
            return

        try:
            content = self.file_path.read_text()

            if self.file_path.suffix in (".yaml", ".yml"):
                self._config = yaml.safe_load(content) or {}
            elif self.file_path.suffix == ".json":
                self._config = json.loads(content)
            else:
                # Try JSON first, then YAML
                try:
                    self._config = json.loads(content)
                except json.JSONDecodeError:
                    self._config = yaml.safe_load(content) or {}

            self._last_modified = self.file_path.stat().st_mtime
            logger.info(f"Loaded config from {self.file_path}")

        except Exception as e:
            logger.error(f"Error loading config from {self.file_path}: {e}")

    def _get_nested(self, key: str) -> Optional[Any]:
        """Get nested value using dot notation."""
        parts = key.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value

    async def get(self, key: str) -> Optional[Any]:
        if not self._config:
            await self._load()
        return self._get_nested(key)

    async def get_all(self, prefix: str = "") -> Dict[str, Any]:
        if not self._config:
            await self._load()

        if not prefix:
            return self._flatten(self._config)

        # Get nested section
        section = self._get_nested(prefix)
        if isinstance(section, dict):
            return self._flatten(section, prefix)
        return {}

    def _flatten(self, d: Dict, parent_key: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary with dot notation."""
        items: list = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    async def watch(self, key: str, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Watch for changes to a key."""
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)

        # Start file watcher if not running
        if self.watch_changes and self._watch_task is None:
            self._watch_task = asyncio.create_task(self._watch_file())

    async def _watch_file(self) -> None:
        """Watch file for changes."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                if not self.file_path.exists():
                    continue

                current_mtime = self.file_path.stat().st_mtime
                if current_mtime > self._last_modified:
                    old_config = self._config.copy()
                    await self._load()

                    # Find changed keys and notify
                    await self._notify_changes(old_config, self._config)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"File watch error: {e}")

    async def _notify_changes(self, old: Dict, new: Dict, prefix: str = "") -> None:
        """Notify watchers of changes."""
        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)

            if isinstance(old_val, dict) and isinstance(new_val, dict):
                await self._notify_changes(old_val, new_val, full_key)
            elif old_val != new_val:
                event = ConfigChangeEvent(
                    key=full_key,
                    old_value=old_val,
                    new_value=new_val,
                    source=self.name,
                )

                # Notify specific watchers
                for callback in self._watchers.get(full_key, []):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.error(f"Watcher callback error: {e}")

                # Notify wildcard watchers
                for callback in self._watchers.get("*", []):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.error(f"Watcher callback error: {e}")

    async def close(self) -> None:
        """Close the file watcher."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None


class ConsulConfigSource(ConfigSource):
    """Consul KV store configuration source."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8500,
        token: Optional[str] = None,
        prefix: str = "config/",
        priority: int = ConfigPriority.CONSUL,
    ):
        super().__init__("consul", priority)
        self.host = host or os.getenv("CONSUL_HOST", "localhost")
        self.port = port or int(os.getenv("CONSUL_PORT", "8500"))
        self.token = token or os.getenv("CONSUL_TOKEN")
        self.prefix = prefix
        self._client: Optional[Any] = None
        self._watch_tasks: Dict[str, asyncio.Task] = {}

    async def _get_client(self) -> Any:
        """Get or create Consul client."""
        if self._client is None:
            try:
                import consul.aio
                self._client = consul.aio.Consul(
                    host=self.host,
                    port=self.port,
                    token=self.token,
                )
            except ImportError:
                raise RuntimeError("python-consul package not installed")
        return self._client

    def _full_key(self, key: str) -> str:
        """Get full Consul key with prefix."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            index, data = await client.kv.get(self._full_key(key))

            if data is None:
                return None

            value = data["Value"]
            if value:
                value = value.decode("utf-8")
                # Try to parse as JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None

        except Exception as e:
            logger.error(f"Consul get error: {e}")
            return None

    async def get_all(self, prefix: str = "") -> Dict[str, Any]:
        try:
            client = await self._get_client()
            full_prefix = self._full_key(prefix)
            index, data = await client.kv.get(full_prefix, recurse=True)

            if data is None:
                return {}

            result = {}
            for item in data:
                key = item["Key"][len(self.prefix):]  # Remove prefix
                value = item["Value"]
                if value:
                    value = value.decode("utf-8")
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value

            return result

        except Exception as e:
            logger.error(f"Consul get_all error: {e}")
            return {}

    async def set(self, key: str, value: Any) -> bool:
        try:
            client = await self._get_client()
            if not isinstance(value, str):
                value = json.dumps(value)
            return await client.kv.put(self._full_key(key), value)
        except Exception as e:
            logger.error(f"Consul set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return await client.kv.delete(self._full_key(key))
        except Exception as e:
            logger.error(f"Consul delete error: {e}")
            return False

    async def watch(self, key: str, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Watch for changes using Consul's blocking queries."""
        if key in self._watch_tasks:
            return

        async def watch_loop():
            index = None
            last_value = None

            while True:
                try:
                    client = await self._get_client()
                    new_index, data = await client.kv.get(
                        self._full_key(key),
                        index=index,
                        wait="30s",
                    )

                    if new_index != index:
                        new_value = None
                        if data and data["Value"]:
                            new_value = data["Value"].decode("utf-8")
                            try:
                                new_value = json.loads(new_value)
                            except json.JSONDecodeError:
                                pass

                        if new_value != last_value:
                            event = ConfigChangeEvent(
                                key=key,
                                old_value=last_value,
                                new_value=new_value,
                                source=self.name,
                            )
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event)
                            else:
                                callback(event)

                            last_value = new_value

                        index = new_index

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Consul watch error: {e}")
                    await asyncio.sleep(5)

        self._watch_tasks[key] = asyncio.create_task(watch_loop())

    async def close(self) -> None:
        """Close watchers and client."""
        for task in self._watch_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._watch_tasks.clear()

        if self._client:
            await self._client.close()
            self._client = None


class EtcdConfigSource(ConfigSource):
    """etcd configuration source."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2379,
        prefix: str = "/config/",
        priority: int = ConfigPriority.ETCD,
    ):
        super().__init__("etcd", priority)
        self.host = host or os.getenv("ETCD_HOST", "localhost")
        self.port = port or int(os.getenv("ETCD_PORT", "2379"))
        self.prefix = prefix
        self._client: Optional[Any] = None
        self._watch_tasks: Dict[str, asyncio.Task] = {}

    async def _get_client(self) -> Any:
        """Get or create etcd client."""
        if self._client is None:
            try:
                import etcd3
                self._client = etcd3.client(
                    host=self.host,
                    port=self.port,
                )
            except ImportError:
                raise RuntimeError("etcd3 package not installed")
        return self._client

    def _full_key(self, key: str) -> str:
        """Get full etcd key with prefix."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            value, _ = client.get(self._full_key(key))

            if value is None:
                return None

            value = value.decode("utf-8")
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        except Exception as e:
            logger.error(f"etcd get error: {e}")
            return None

    async def get_all(self, prefix: str = "") -> Dict[str, Any]:
        try:
            client = await self._get_client()
            full_prefix = self._full_key(prefix)

            result = {}
            for value, metadata in client.get_prefix(full_prefix):
                key = metadata.key.decode("utf-8")[len(self.prefix):]
                if value:
                    value = value.decode("utf-8")
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value

            return result

        except Exception as e:
            logger.error(f"etcd get_all error: {e}")
            return {}

    async def set(self, key: str, value: Any) -> bool:
        try:
            client = await self._get_client()
            if not isinstance(value, str):
                value = json.dumps(value)
            client.put(self._full_key(key), value)
            return True
        except Exception as e:
            logger.error(f"etcd set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return client.delete(self._full_key(key))
        except Exception as e:
            logger.error(f"etcd delete error: {e}")
            return False

    async def close(self) -> None:
        """Close watchers and client."""
        for task in self._watch_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._watch_tasks.clear()

        if self._client:
            self._client.close()
            self._client = None
