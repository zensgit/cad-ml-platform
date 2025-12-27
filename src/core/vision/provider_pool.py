"""Provider connection pooling for Vision Provider system.

This module provides connection pooling including:
- Provider instance pooling
- Connection lifecycle management
- Pool sizing and scaling
- Health-based pool management
- Resource cleanup
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from .base import VisionDescription, VisionProvider


class PoolState(Enum):
    """State of the pool."""

    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"


class ConnectionState(Enum):
    """State of a pooled connection."""

    IDLE = "idle"
    IN_USE = "in_use"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass
class PoolConfig:
    """Configuration for provider pool."""

    min_size: int = 1
    max_size: int = 10
    max_idle_time_seconds: float = 300.0  # 5 minutes
    max_lifetime_seconds: float = 3600.0  # 1 hour
    acquire_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 60.0
    validation_on_acquire: bool = True
    validation_on_return: bool = False


@dataclass
class PooledConnection:
    """Pooled provider connection."""

    connection_id: str
    provider: VisionProvider
    state: ConnectionState = ConnectionState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None
    use_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0

    def mark_in_use(self) -> None:
        """Mark connection as in use."""
        self.state = ConnectionState.IN_USE
        self.use_count += 1

    def mark_idle(self, duration_ms: float = 0.0) -> None:
        """Mark connection as idle."""
        self.state = ConnectionState.IDLE
        self.last_used_at = datetime.now()
        self.total_duration_ms += duration_ms

    def mark_unhealthy(self) -> None:
        """Mark connection as unhealthy."""
        self.state = ConnectionState.UNHEALTHY
        self.error_count += 1

    def mark_closed(self) -> None:
        """Mark connection as closed."""
        self.state = ConnectionState.CLOSED

    def is_expired(self, max_lifetime: float) -> bool:
        """Check if connection is expired."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > max_lifetime

    def is_idle_too_long(self, max_idle: float) -> bool:
        """Check if connection has been idle too long."""
        idle_time = (datetime.now() - self.last_used_at).total_seconds()
        return idle_time > max_idle

    @property
    def average_duration_ms(self) -> float:
        """Get average operation duration."""
        if self.use_count == 0:
            return 0.0
        return self.total_duration_ms / self.use_count


@dataclass
class PoolStats:
    """Statistics for provider pool."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_acquisitions: int = 0
    successful_acquisitions: int = 0
    failed_acquisitions: int = 0
    total_wait_time_ms: float = 0.0
    connections_created: int = 0
    connections_closed: int = 0
    health_checks_performed: int = 0
    health_check_failures: int = 0

    @property
    def average_wait_time_ms(self) -> float:
        """Get average acquisition wait time."""
        if self.total_acquisitions == 0:
            return 0.0
        return self.total_wait_time_ms / self.total_acquisitions


class ProviderFactory(ABC):
    """Abstract factory for creating providers."""

    @abstractmethod
    def create(self) -> VisionProvider:
        """Create a new provider instance."""
        pass

    @abstractmethod
    async def validate(self, provider: VisionProvider) -> bool:
        """Validate a provider instance."""
        pass

    @abstractmethod
    def destroy(self, provider: VisionProvider) -> None:
        """Destroy a provider instance."""
        pass


class SimpleProviderFactory(ProviderFactory):
    """Simple provider factory using a creation function."""

    def __init__(
        self,
        create_fn: Callable[[], VisionProvider],
        validate_fn: Optional[Callable[[VisionProvider], bool]] = None,
        destroy_fn: Optional[Callable[[VisionProvider], None]] = None,
    ) -> None:
        """Initialize factory.

        Args:
            create_fn: Function to create provider
            validate_fn: Optional validation function
            destroy_fn: Optional destruction function
        """
        self._create_fn = create_fn
        self._validate_fn = validate_fn
        self._destroy_fn = destroy_fn

    def create(self) -> VisionProvider:
        """Create a new provider."""
        return self._create_fn()

    async def validate(self, provider: VisionProvider) -> bool:
        """Validate a provider."""
        if self._validate_fn:
            return self._validate_fn(provider)
        return True

    def destroy(self, provider: VisionProvider) -> None:
        """Destroy a provider."""
        if self._destroy_fn:
            self._destroy_fn(provider)


class ProviderPool:
    """Pool of provider connections."""

    def __init__(
        self,
        factory: ProviderFactory,
        config: Optional[PoolConfig] = None,
    ) -> None:
        """Initialize provider pool.

        Args:
            factory: Provider factory
            config: Pool configuration
        """
        self._factory = factory
        self._config = config or PoolConfig()
        self._connections: List[PooledConnection] = []
        self._state = PoolState.RUNNING
        self._stats = PoolStats()
        self._lock = threading.Lock()
        self._available = asyncio.Semaphore(self._config.max_size)
        self._connection_counter = 0

    @property
    def config(self) -> PoolConfig:
        """Return pool configuration."""
        return self._config

    @property
    def stats(self) -> PoolStats:
        """Return pool statistics."""
        return self._stats

    @property
    def state(self) -> PoolState:
        """Return pool state."""
        return self._state

    @property
    def size(self) -> int:
        """Return current pool size."""
        with self._lock:
            return len([c for c in self._connections if c.state != ConnectionState.CLOSED])

    async def initialize(self) -> None:
        """Initialize pool with minimum connections."""
        for _ in range(self._config.min_size):
            await self._create_connection()

    async def acquire(self) -> PooledConnection:
        """Acquire a connection from pool.

        Returns:
            Pooled connection

        Raises:
            TimeoutError: If acquisition times out
            RuntimeError: If pool is not running
        """
        if self._state != PoolState.RUNNING:
            raise RuntimeError(f"Pool is {self._state.value}")

        start_time = time.time()
        self._stats.total_acquisitions += 1

        try:
            await asyncio.wait_for(
                self._available.acquire(),
                timeout=self._config.acquire_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._stats.failed_acquisitions += 1
            raise TimeoutError("Connection acquisition timed out")

        wait_time = (time.time() - start_time) * 1000
        self._stats.total_wait_time_ms += wait_time

        # Try to get an idle connection
        connection = self._get_idle_connection()

        if connection is None:
            # Create new connection if under max
            if self.size < self._config.max_size:
                connection = await self._create_connection()
            else:
                self._available.release()
                self._stats.failed_acquisitions += 1
                raise RuntimeError("No connections available")

        # Validate if configured
        if self._config.validation_on_acquire:
            if not await self._validate_connection(connection):
                await self._close_connection(connection)
                connection = await self._create_connection()

        connection.mark_in_use()
        self._stats.successful_acquisitions += 1
        self._update_stats()

        return connection

    async def release(self, connection: PooledConnection, duration_ms: float = 0.0) -> None:
        """Release a connection back to pool.

        Args:
            connection: Connection to release
            duration_ms: Operation duration
        """
        if self._config.validation_on_return:
            if not await self._validate_connection(connection):
                await self._close_connection(connection)
                self._available.release()
                return

        # Check if connection should be closed
        if connection.is_expired(self._config.max_lifetime_seconds):
            await self._close_connection(connection)
        elif connection.state == ConnectionState.UNHEALTHY:
            await self._close_connection(connection)
        else:
            connection.mark_idle(duration_ms)

        self._available.release()
        self._update_stats()

    async def invalidate(self, connection: PooledConnection) -> None:
        """Invalidate and close a connection.

        Args:
            connection: Connection to invalidate
        """
        connection.mark_unhealthy()
        await self._close_connection(connection)
        self._available.release()
        self._update_stats()

    def _get_idle_connection(self) -> Optional[PooledConnection]:
        """Get an idle connection from pool."""
        with self._lock:
            for conn in self._connections:
                if conn.state == ConnectionState.IDLE:
                    if not conn.is_idle_too_long(self._config.max_idle_time_seconds):
                        return conn
            return None

    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        provider = self._factory.create()

        with self._lock:
            self._connection_counter += 1
            connection = PooledConnection(
                connection_id=f"conn_{self._connection_counter}",
                provider=provider,
            )
            self._connections.append(connection)
            self._stats.connections_created += 1

        return connection

    async def _close_connection(self, connection: PooledConnection) -> None:
        """Close a connection."""
        connection.mark_closed()
        self._factory.destroy(connection.provider)

        with self._lock:
            if connection in self._connections:
                self._connections.remove(connection)
            self._stats.connections_closed += 1

    async def _validate_connection(self, connection: PooledConnection) -> bool:
        """Validate a connection."""
        self._stats.health_checks_performed += 1

        try:
            is_valid = await self._factory.validate(connection.provider)
            connection.last_validated_at = datetime.now()

            if not is_valid:
                self._stats.health_check_failures += 1

            return is_valid

        except Exception:
            self._stats.health_check_failures += 1
            return False

    def _update_stats(self) -> None:
        """Update pool statistics."""
        with self._lock:
            active = sum(1 for c in self._connections if c.state == ConnectionState.IN_USE)
            idle = sum(1 for c in self._connections if c.state == ConnectionState.IDLE)
            total = len([c for c in self._connections if c.state != ConnectionState.CLOSED])

            self._stats.total_connections = total
            self._stats.active_connections = active
            self._stats.idle_connections = idle

    async def cleanup_idle(self) -> int:
        """Cleanup idle connections.

        Returns:
            Number of connections cleaned up
        """
        cleaned = 0

        with self._lock:
            to_close = []
            for conn in self._connections:
                if conn.state == ConnectionState.IDLE:
                    if conn.is_idle_too_long(self._config.max_idle_time_seconds):
                        if self.size > self._config.min_size:
                            to_close.append(conn)

        for conn in to_close:
            await self._close_connection(conn)
            cleaned += 1

        self._update_stats()
        return cleaned

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Shutdown the pool.

        Args:
            timeout_seconds: Timeout for draining
        """
        self._state = PoolState.DRAINING

        # Wait for active connections to be released
        start = time.time()
        while self._stats.active_connections > 0:
            if time.time() - start > timeout_seconds:
                break
            await asyncio.sleep(0.1)

        # Close all connections
        self._state = PoolState.STOPPED

        with self._lock:
            for conn in list(self._connections):
                conn.mark_closed()
                self._factory.destroy(conn.provider)
            self._connections.clear()

        self._update_stats()


class PooledVisionProvider(VisionProvider):
    """Vision provider using connection pool."""

    def __init__(self, pool: ProviderPool) -> None:
        """Initialize pooled provider.

        Args:
            pool: Provider pool
        """
        self._pool = pool

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "pooled"

    @property
    def pool(self) -> ProviderPool:
        """Return provider pool."""
        return self._pool

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using pooled connection.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        connection = await self._pool.acquire()
        start_time = time.time()

        try:
            result = await connection.provider.analyze_image(image_data, include_description)
            duration_ms = (time.time() - start_time) * 1000
            await self._pool.release(connection, duration_ms)
            return result

        except Exception as e:
            await self._pool.invalidate(connection)
            raise


async def create_pooled_provider(
    factory: ProviderFactory,
    config: Optional[PoolConfig] = None,
    initialize: bool = True,
) -> PooledVisionProvider:
    """Create a pooled vision provider.

    Args:
        factory: Provider factory
        config: Pool configuration
        initialize: Whether to initialize pool

    Returns:
        PooledVisionProvider instance
    """
    pool = ProviderPool(factory=factory, config=config)

    if initialize:
        await pool.initialize()

    return PooledVisionProvider(pool=pool)
