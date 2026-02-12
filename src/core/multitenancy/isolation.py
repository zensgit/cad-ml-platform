"""Data Isolation Strategies for Multi-tenancy.

Provides different levels of tenant data isolation:
- Row-level: Same table, filtered by tenant_id
- Schema-level: Separate schema per tenant
- Database-level: Separate database per tenant
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from src.core.multitenancy.context import TenantContext, get_current_tenant

logger = logging.getLogger(__name__)


class IsolationLevel(str, Enum):
    """Data isolation levels."""

    ROW_LEVEL = "row_level"      # Same DB, same schema, filter by tenant_id
    SCHEMA_LEVEL = "schema"      # Same DB, separate schema per tenant
    DATABASE_LEVEL = "database"  # Separate database per tenant


@dataclass
class IsolationConfig:
    """Configuration for data isolation."""

    level: IsolationLevel
    tenant_column: str = "tenant_id"
    schema_prefix: str = "tenant_"
    database_prefix: str = "tenant_"
    default_schema: str = "public"
    default_database: str = "cad_ml_platform"


class DataIsolationStrategy(ABC):
    """Abstract base class for data isolation strategies."""

    def __init__(self, config: IsolationConfig):
        self.config = config

    @abstractmethod
    def get_connection_params(self, tenant: TenantContext) -> Dict[str, Any]:
        """Get database connection parameters for tenant."""
        pass

    @abstractmethod
    def apply_filter(self, query: Any, tenant: TenantContext) -> Any:
        """Apply tenant filter to a query."""
        pass

    @abstractmethod
    def inject_tenant(self, data: Dict[str, Any], tenant: TenantContext) -> Dict[str, Any]:
        """Inject tenant identifier into data."""
        pass

    @abstractmethod
    def validate_access(self, resource_tenant_id: str, tenant: TenantContext) -> bool:
        """Validate tenant has access to resource."""
        pass


class RowLevelIsolation(DataIsolationStrategy):
    """Row-level isolation using tenant_id column.

    All tenants share the same tables, data is filtered by tenant_id.
    """

    def get_connection_params(self, tenant: TenantContext) -> Dict[str, Any]:
        """Same connection for all tenants."""
        return {
            "database": self.config.default_database,
            "schema": self.config.default_schema,
        }

    def apply_filter(self, query: Any, tenant: TenantContext) -> Any:
        """Add WHERE tenant_id = ? clause.

        Note: This is a simplified example. In practice, you'd integrate
        with your ORM (SQLAlchemy, Django ORM, etc.)
        """
        # For SQLAlchemy-style queries
        if hasattr(query, "filter"):
            # Assuming query model has tenant_id column
            return query.filter_by(**{self.config.tenant_column: tenant.tenant_id})

        # For raw SQL (returns modified query string)
        if isinstance(query, str):
            if "WHERE" in query.upper():
                return f"{query} AND {self.config.tenant_column} = '{tenant.tenant_id}'"
            else:
                return f"{query} WHERE {self.config.tenant_column} = '{tenant.tenant_id}'"

        return query

    def inject_tenant(self, data: Dict[str, Any], tenant: TenantContext) -> Dict[str, Any]:
        """Add tenant_id to data."""
        data[self.config.tenant_column] = tenant.tenant_id
        return data

    def validate_access(self, resource_tenant_id: str, tenant: TenantContext) -> bool:
        """Check if tenant owns the resource."""
        return resource_tenant_id == tenant.tenant_id


class SchemaIsolation(DataIsolationStrategy):
    """Schema-level isolation.

    Each tenant gets their own database schema.
    """

    def get_connection_params(self, tenant: TenantContext) -> Dict[str, Any]:
        """Get schema-specific connection params."""
        schema_name = tenant.schema_name or f"{self.config.schema_prefix}{tenant.tenant_id}"
        return {
            "database": self.config.default_database,
            "schema": schema_name,
            "search_path": schema_name,
        }

    def apply_filter(self, query: Any, tenant: TenantContext) -> Any:
        """No additional filter needed - schema provides isolation."""
        return query

    def inject_tenant(self, data: Dict[str, Any], tenant: TenantContext) -> Dict[str, Any]:
        """Tenant ID still stored for audit purposes."""
        data[self.config.tenant_column] = tenant.tenant_id
        return data

    def validate_access(self, resource_tenant_id: str, tenant: TenantContext) -> bool:
        """Schema isolation handles access control."""
        return resource_tenant_id == tenant.tenant_id

    async def create_tenant_schema(self, tenant: TenantContext, connection: Any) -> bool:
        """Create schema for new tenant.

        Args:
            tenant: TenantContext
            connection: Database connection

        Returns:
            True if schema created successfully
        """
        schema_name = tenant.schema_name or f"{self.config.schema_prefix}{tenant.tenant_id}"

        try:
            # Create schema
            await connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

            # Copy structure from template (if using PostgreSQL)
            # This would clone tables from a template schema
            logger.info(f"Created schema {schema_name} for tenant {tenant.tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False

    async def drop_tenant_schema(self, tenant: TenantContext, connection: Any) -> bool:
        """Drop schema for tenant (use with caution!)."""
        schema_name = tenant.schema_name or f"{self.config.schema_prefix}{tenant.tenant_id}"

        try:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
            logger.info(f"Dropped schema {schema_name} for tenant {tenant.tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to drop schema: {e}")
            return False


class DatabaseIsolation(DataIsolationStrategy):
    """Database-level isolation.

    Each tenant gets their own database.
    """

    def __init__(self, config: IsolationConfig):
        super().__init__(config)
        self._connection_pools: Dict[str, Any] = {}

    def get_connection_params(self, tenant: TenantContext) -> Dict[str, Any]:
        """Get database-specific connection params."""
        database_name = tenant.database_name or f"{self.config.database_prefix}{tenant.tenant_id}"
        return {
            "database": database_name,
            "schema": self.config.default_schema,
        }

    def apply_filter(self, query: Any, tenant: TenantContext) -> Any:
        """No filter needed - database provides complete isolation."""
        return query

    def inject_tenant(self, data: Dict[str, Any], tenant: TenantContext) -> Dict[str, Any]:
        """Tenant ID stored for cross-database operations."""
        data[self.config.tenant_column] = tenant.tenant_id
        return data

    def validate_access(self, resource_tenant_id: str, tenant: TenantContext) -> bool:
        """Database isolation handles access control."""
        return resource_tenant_id == tenant.tenant_id

    async def create_tenant_database(
        self,
        tenant: TenantContext,
        admin_connection: Any,
        template_db: Optional[str] = None,
    ) -> bool:
        """Create database for new tenant.

        Args:
            tenant: TenantContext
            admin_connection: Admin database connection
            template_db: Template database to clone (optional)

        Returns:
            True if database created successfully
        """
        database_name = tenant.database_name or f"{self.config.database_prefix}{tenant.tenant_id}"

        try:
            if template_db:
                await admin_connection.execute(
                    f'CREATE DATABASE "{database_name}" TEMPLATE "{template_db}"'
                )
            else:
                await admin_connection.execute(f'CREATE DATABASE "{database_name}"')

            logger.info(f"Created database {database_name} for tenant {tenant.tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False


class HybridIsolation(DataIsolationStrategy):
    """Hybrid isolation combining multiple strategies.

    Example: Schema isolation for large tenants, row-level for small ones.
    """

    def __init__(
        self,
        config: IsolationConfig,
        threshold_documents: int = 10000,
    ):
        super().__init__(config)
        self.threshold_documents = threshold_documents
        self._row_level = RowLevelIsolation(config)
        self._schema_level = SchemaIsolation(config)

    def _get_strategy(self, tenant: TenantContext) -> DataIsolationStrategy:
        """Determine which strategy to use for tenant."""
        if tenant.current_documents >= self.threshold_documents:
            return self._schema_level
        return self._row_level

    def get_connection_params(self, tenant: TenantContext) -> Dict[str, Any]:
        return self._get_strategy(tenant).get_connection_params(tenant)

    def apply_filter(self, query: Any, tenant: TenantContext) -> Any:
        return self._get_strategy(tenant).apply_filter(query, tenant)

    def inject_tenant(self, data: Dict[str, Any], tenant: TenantContext) -> Dict[str, Any]:
        return self._get_strategy(tenant).inject_tenant(data, tenant)

    def validate_access(self, resource_tenant_id: str, tenant: TenantContext) -> bool:
        return self._get_strategy(tenant).validate_access(resource_tenant_id, tenant)


# Strategy registry
_strategies: Dict[IsolationLevel, Type[DataIsolationStrategy]] = {
    IsolationLevel.ROW_LEVEL: RowLevelIsolation,
    IsolationLevel.SCHEMA_LEVEL: SchemaIsolation,
    IsolationLevel.DATABASE_LEVEL: DatabaseIsolation,
}

# Global strategy instance
_isolation_strategy: Optional[DataIsolationStrategy] = None


def get_isolation_strategy(
    level: Optional[IsolationLevel] = None,
    config: Optional[IsolationConfig] = None,
) -> DataIsolationStrategy:
    """Get data isolation strategy.

    Args:
        level: Isolation level (default: ROW_LEVEL)
        config: Isolation configuration

    Returns:
        DataIsolationStrategy instance
    """
    global _isolation_strategy

    if _isolation_strategy is None:
        level = level or IsolationLevel.ROW_LEVEL
        config = config or IsolationConfig(level=level)
        strategy_class = _strategies.get(level, RowLevelIsolation)
        _isolation_strategy = strategy_class(config)

    return _isolation_strategy


def set_isolation_strategy(strategy: DataIsolationStrategy) -> None:
    """Set global isolation strategy."""
    global _isolation_strategy
    _isolation_strategy = strategy
