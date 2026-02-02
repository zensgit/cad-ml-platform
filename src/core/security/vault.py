"""HashiCorp Vault Integration for Secret Management.

Provides:
- Dynamic secret retrieval
- Automatic secret rotation
- Lease management
- Kubernetes auth integration
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Conditional import
try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False
    hvac = None  # type: ignore


@dataclass
class VaultConfig:
    """Vault client configuration."""

    url: str = ""
    token: str = ""
    namespace: str = ""

    # Kubernetes auth
    k8s_role: str = ""
    k8s_mount_point: str = "kubernetes"
    service_account_token_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"

    # AppRole auth
    role_id: str = ""
    secret_id: str = ""
    approle_mount_point: str = "approle"

    # TLS
    verify_tls: bool = True
    ca_cert: str = ""

    # Secret paths
    kv_mount_point: str = "secret"
    kv_version: int = 2

    # Rotation
    rotation_buffer_seconds: int = 300  # Rotate 5 min before expiry

    @classmethod
    def from_env(cls) -> "VaultConfig":
        """Create config from environment variables."""
        return cls(
            url=os.getenv("VAULT_ADDR", "http://localhost:8200"),
            token=os.getenv("VAULT_TOKEN", ""),
            namespace=os.getenv("VAULT_NAMESPACE", ""),
            k8s_role=os.getenv("VAULT_K8S_ROLE", ""),
            k8s_mount_point=os.getenv("VAULT_K8S_MOUNT", "kubernetes"),
            role_id=os.getenv("VAULT_ROLE_ID", ""),
            secret_id=os.getenv("VAULT_SECRET_ID", ""),
            verify_tls=os.getenv("VAULT_SKIP_VERIFY", "false").lower() != "true",
            ca_cert=os.getenv("VAULT_CACERT", ""),
            kv_mount_point=os.getenv("VAULT_KV_MOUNT", "secret"),
            kv_version=int(os.getenv("VAULT_KV_VERSION", "2")),
            rotation_buffer_seconds=int(os.getenv("VAULT_ROTATION_BUFFER", "300")),
        )


@dataclass
class SecretLease:
    """Tracks a secret lease from Vault."""

    lease_id: str
    path: str
    secret_data: Dict[str, Any]
    lease_duration: int
    renewable: bool
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def expires_at(self) -> datetime:
        """Get expiration time."""
        return self.created_at + timedelta(seconds=self.lease_duration)

    def is_expired(self, buffer_seconds: int = 0) -> bool:
        """Check if lease is expired or about to expire."""
        return datetime.utcnow() >= self.expires_at - timedelta(seconds=buffer_seconds)


class VaultClient:
    """Client for HashiCorp Vault secret management."""

    def __init__(self, config: Optional[VaultConfig] = None):
        self.config = config or VaultConfig.from_env()
        self._client: Optional[Any] = None
        self._leases: Dict[str, SecretLease] = {}
        self._rotation_callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._rotation_task: Optional[asyncio.Task] = None

    def _get_client(self) -> Any:
        """Get or create Vault client."""
        if not HVAC_AVAILABLE:
            raise RuntimeError("hvac library not installed. Install with: pip install hvac")

        if self._client is None:
            self._client = hvac.Client(
                url=self.config.url,
                token=self.config.token or None,
                namespace=self.config.namespace or None,
                verify=self.config.ca_cert if self.config.ca_cert else self.config.verify_tls,
            )

            # Authenticate if needed
            if not self._client.is_authenticated():
                self._authenticate()

        return self._client

    def _authenticate(self) -> None:
        """Authenticate to Vault."""
        client = self._client

        # Try Kubernetes auth first
        if self.config.k8s_role:
            try:
                with open(self.config.service_account_token_path) as f:
                    jwt = f.read()
                client.auth.kubernetes.login(
                    role=self.config.k8s_role,
                    jwt=jwt,
                    mount_point=self.config.k8s_mount_point,
                )
                logger.info("Authenticated to Vault using Kubernetes auth")
                return
            except Exception as e:
                logger.warning(f"Kubernetes auth failed: {e}")

        # Try AppRole auth
        if self.config.role_id and self.config.secret_id:
            try:
                client.auth.approle.login(
                    role_id=self.config.role_id,
                    secret_id=self.config.secret_id,
                    mount_point=self.config.approle_mount_point,
                )
                logger.info("Authenticated to Vault using AppRole")
                return
            except Exception as e:
                logger.warning(f"AppRole auth failed: {e}")

        raise RuntimeError("No valid Vault authentication method configured")

    def get_secret(self, path: str, mount_point: Optional[str] = None) -> Dict[str, Any]:
        """Get a secret from Vault KV store.

        Args:
            path: Secret path
            mount_point: KV mount point (defaults to config)

        Returns:
            Secret data
        """
        client = self._get_client()
        mount = mount_point or self.config.kv_mount_point

        try:
            if self.config.kv_version == 2:
                response = client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=mount,
                )
                return response["data"]["data"]
            else:
                response = client.secrets.kv.v1.read_secret(
                    path=path,
                    mount_point=mount,
                )
                return response["data"]
        except Exception as e:
            logger.error(f"Failed to get secret {path}: {e}")
            raise

    def put_secret(
        self,
        path: str,
        data: Dict[str, Any],
        mount_point: Optional[str] = None,
    ) -> None:
        """Store a secret in Vault KV store.

        Args:
            path: Secret path
            data: Secret data
            mount_point: KV mount point
        """
        client = self._get_client()
        mount = mount_point or self.config.kv_mount_point

        try:
            if self.config.kv_version == 2:
                client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=data,
                    mount_point=mount,
                )
            else:
                client.secrets.kv.v1.create_or_update_secret(
                    path=path,
                    secret=data,
                    mount_point=mount,
                )
            logger.info(f"Stored secret at {path}")
        except Exception as e:
            logger.error(f"Failed to store secret {path}: {e}")
            raise

    def delete_secret(self, path: str, mount_point: Optional[str] = None) -> None:
        """Delete a secret from Vault.

        Args:
            path: Secret path
            mount_point: KV mount point
        """
        client = self._get_client()
        mount = mount_point or self.config.kv_mount_point

        try:
            if self.config.kv_version == 2:
                client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=path,
                    mount_point=mount,
                )
            else:
                client.secrets.kv.v1.delete_secret(
                    path=path,
                    mount_point=mount,
                )
            logger.info(f"Deleted secret at {path}")
        except Exception as e:
            logger.error(f"Failed to delete secret {path}: {e}")
            raise

    def get_database_credentials(
        self,
        role: str,
        mount_point: str = "database",
    ) -> Dict[str, Any]:
        """Get dynamic database credentials.

        Args:
            role: Database role name
            mount_point: Database secrets mount point

        Returns:
            Database credentials with username and password
        """
        client = self._get_client()

        try:
            response = client.secrets.database.generate_credentials(
                name=role,
                mount_point=mount_point,
            )

            # Track lease
            lease = SecretLease(
                lease_id=response["lease_id"],
                path=f"{mount_point}/creds/{role}",
                secret_data=response["data"],
                lease_duration=response["lease_duration"],
                renewable=response["renewable"],
            )
            self._leases[lease.lease_id] = lease

            logger.info(f"Generated database credentials for role {role}")
            return response["data"]
        except Exception as e:
            logger.error(f"Failed to get database credentials: {e}")
            raise

    def renew_lease(self, lease_id: str) -> bool:
        """Renew a secret lease.

        Args:
            lease_id: Lease ID to renew

        Returns:
            True if renewal successful
        """
        client = self._get_client()

        try:
            response = client.sys.renew_lease(lease_id=lease_id)

            # Update tracked lease
            if lease_id in self._leases:
                self._leases[lease_id].lease_duration = response["lease_duration"]
                self._leases[lease_id].created_at = datetime.utcnow()

            logger.info(f"Renewed lease {lease_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to renew lease {lease_id}: {e}")
            return False

    def revoke_lease(self, lease_id: str) -> bool:
        """Revoke a secret lease.

        Args:
            lease_id: Lease ID to revoke

        Returns:
            True if revocation successful
        """
        client = self._get_client()

        try:
            client.sys.revoke_lease(lease_id=lease_id)
            self._leases.pop(lease_id, None)
            logger.info(f"Revoked lease {lease_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke lease {lease_id}: {e}")
            return False

    def register_rotation_callback(
        self,
        path: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a callback for secret rotation.

        Args:
            path: Secret path to watch
            callback: Function to call with new secret data
        """
        if path not in self._rotation_callbacks:
            self._rotation_callbacks[path] = []
        self._rotation_callbacks[path].append(callback)
        logger.info(f"Registered rotation callback for {path}")

    async def start_rotation_monitor(self) -> None:
        """Start background task to monitor and rotate secrets."""
        if self._rotation_task is not None:
            return

        async def monitor_loop():
            while True:
                try:
                    await self._check_and_rotate()
                except Exception as e:
                    logger.error(f"Error in rotation monitor: {e}")
                await asyncio.sleep(60)  # Check every minute

        self._rotation_task = asyncio.create_task(monitor_loop())
        logger.info("Started secret rotation monitor")

    async def _check_and_rotate(self) -> None:
        """Check leases and trigger rotation if needed."""
        buffer = self.config.rotation_buffer_seconds

        for lease_id, lease in list(self._leases.items()):
            if lease.is_expired(buffer):
                if lease.renewable:
                    # Try to renew
                    if self.renew_lease(lease_id):
                        continue

                # Rotation needed - get fresh credentials
                if lease.path in self._rotation_callbacks:
                    try:
                        # Get new secret
                        parts = lease.path.split("/")
                        if "database" in parts[0]:
                            role = parts[-1]
                            new_data = self.get_database_credentials(role, parts[0])
                        else:
                            new_data = self.get_secret(lease.path)

                        # Notify callbacks
                        for callback in self._rotation_callbacks.get(lease.path, []):
                            callback(new_data)

                        logger.info(f"Rotated secret at {lease.path}")
                    except Exception as e:
                        logger.error(f"Failed to rotate secret {lease.path}: {e}")

    def stop_rotation_monitor(self) -> None:
        """Stop the rotation monitor task."""
        if self._rotation_task:
            self._rotation_task.cancel()
            self._rotation_task = None
            logger.info("Stopped secret rotation monitor")

    def health_check(self) -> bool:
        """Check Vault connectivity.

        Returns:
            True if Vault is healthy
        """
        try:
            client = self._get_client()
            return client.sys.is_initialized() and not client.sys.is_sealed()
        except Exception as e:
            logger.error(f"Vault health check failed: {e}")
            return False


# Global Vault client
_vault_client: Optional[VaultClient] = None


def get_vault_client() -> VaultClient:
    """Get global Vault client."""
    global _vault_client
    if _vault_client is None:
        _vault_client = VaultClient()
    return _vault_client


def get_secret(path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to get a secret.

    Args:
        path: Secret path
        default: Default value if secret not found

    Returns:
        Secret data
    """
    try:
        return get_vault_client().get_secret(path)
    except Exception:
        if default is not None:
            return default
        raise
