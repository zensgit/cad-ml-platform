"""File Storage Module.

Provides complete file storage capabilities:
- S3-compatible object storage
- Presigned URL generation
- Multipart upload support
- Time-series storage for telemetry
"""

from src.core.storage.object_store import (
    StorageBackend,
    StorageConfig,
    StorageClient,
    ObjectMetadata,
    InMemoryStorage,
    LocalFileStorage,
    S3Storage,
    create_storage_client,
    get_storage_client,
    configure_storage,
)
from src.core.storage.presigned import (
    URLMethod,
    PresignedURL,
    URLSigner,
    MultipartUploadURL,
    MultipartURLGenerator,
    get_url_signer,
    configure_url_signer,
)
from src.core.storage.multipart import (
    UploadStatus,
    PartInfo,
    UploadProgress,
    MultipartUploadState,
    MultipartUploadManager,
    get_upload_manager,
)
from src.core.storage.timeseries import (
    TimeSeriesStore,
    InMemoryTimeSeriesStore,
    NullTimeSeriesStore,
)

__all__ = [
    # Object Storage
    "StorageBackend",
    "StorageConfig",
    "StorageClient",
    "ObjectMetadata",
    "InMemoryStorage",
    "LocalFileStorage",
    "S3Storage",
    "create_storage_client",
    "get_storage_client",
    "configure_storage",
    # Presigned URLs
    "URLMethod",
    "PresignedURL",
    "URLSigner",
    "MultipartUploadURL",
    "MultipartURLGenerator",
    "get_url_signer",
    "configure_url_signer",
    # Multipart Upload
    "UploadStatus",
    "PartInfo",
    "UploadProgress",
    "MultipartUploadState",
    "MultipartUploadManager",
    "get_upload_manager",
    # Time Series
    "TimeSeriesStore",
    "InMemoryTimeSeriesStore",
    "NullTimeSeriesStore",
]
