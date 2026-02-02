"""Presigned URL Service.

Provides secure, time-limited URLs for file access:
- Upload URLs for direct browser uploads
- Download URLs for secure file access
- URL signing and validation
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class URLMethod(str, Enum):
    """HTTP methods for presigned URLs."""
    GET = "GET"
    PUT = "PUT"
    POST = "POST"
    DELETE = "DELETE"


@dataclass
class PresignedURL:
    """A presigned URL with metadata."""
    url: str
    method: URLMethod
    key: str
    bucket: str
    expires_at: datetime
    headers: Dict[str, str]
    metadata: Dict[str, Any]


class URLSigner:
    """Signs and validates presigned URLs."""

    def __init__(
        self,
        secret_key: str,
        base_url: str = "http://localhost:8000",
        default_expiry: int = 3600,
    ):
        """Initialize the URL signer.

        Args:
            secret_key: Secret key for signing.
            base_url: Base URL for generated URLs.
            default_expiry: Default expiry in seconds.
        """
        self.secret_key = secret_key.encode()
        self.base_url = base_url.rstrip("/")
        self.default_expiry = default_expiry

    def _compute_signature(
        self,
        method: str,
        key: str,
        expires: int,
        content_type: Optional[str] = None,
    ) -> str:
        """Compute HMAC signature for URL."""
        string_to_sign = f"{method}\n{key}\n{expires}"
        if content_type:
            string_to_sign += f"\n{content_type}"

        signature = hmac.new(
            self.secret_key,
            string_to_sign.encode(),
            hashlib.sha256
        ).digest()

        return base64.urlsafe_b64encode(signature).decode()

    def generate_download_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = None,
        filename: Optional[str] = None,
    ) -> PresignedURL:
        """Generate a presigned download URL.

        Args:
            bucket: Bucket name.
            key: Object key.
            expires_in: Expiry time in seconds.
            filename: Optional download filename.

        Returns:
            PresignedURL object.
        """
        expires_in = expires_in or self.default_expiry
        expires = int(time.time()) + expires_in
        signature = self._compute_signature("GET", key, expires)

        params = {
            "expires": str(expires),
            "signature": signature,
        }
        if filename:
            params["filename"] = filename

        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}/storage/{bucket}/{key}?{query}"

        return PresignedURL(
            url=url,
            method=URLMethod.GET,
            key=key,
            bucket=bucket,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
            headers={},
            metadata={"filename": filename} if filename else {},
        )

    def generate_upload_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = None,
        content_type: str = "application/octet-stream",
        max_size: Optional[int] = None,
    ) -> PresignedURL:
        """Generate a presigned upload URL.

        Args:
            bucket: Bucket name.
            key: Object key.
            expires_in: Expiry time in seconds.
            content_type: Expected content type.
            max_size: Maximum file size in bytes.

        Returns:
            PresignedURL object with required headers.
        """
        expires_in = expires_in or self.default_expiry
        expires = int(time.time()) + expires_in
        signature = self._compute_signature("PUT", key, expires, content_type)

        params = {
            "expires": str(expires),
            "signature": signature,
            "content_type": content_type,
        }
        if max_size:
            params["max_size"] = str(max_size)

        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}/storage/{bucket}/{key}?{query}"

        headers = {
            "Content-Type": content_type,
        }

        return PresignedURL(
            url=url,
            method=URLMethod.PUT,
            key=key,
            bucket=bucket,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
            headers=headers,
            metadata={
                "content_type": content_type,
                "max_size": max_size,
            },
        )

    def validate_signature(
        self,
        method: str,
        key: str,
        expires: int,
        signature: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """Validate a presigned URL signature.

        Args:
            method: HTTP method.
            key: Object key.
            expires: Expiry timestamp.
            signature: Signature to validate.
            content_type: Content type (for uploads).

        Returns:
            True if signature is valid and not expired.
        """
        # Check expiry
        if time.time() > expires:
            logger.warning(f"URL expired: key={key}")
            return False

        # Compute expected signature
        expected = self._compute_signature(method, key, expires, content_type)

        # Constant-time comparison
        if not hmac.compare_digest(expected, signature):
            logger.warning(f"Invalid signature: key={key}")
            return False

        return True


@dataclass
class MultipartUploadURL:
    """URLs for multipart upload."""
    upload_id: str
    key: str
    bucket: str
    part_urls: Dict[int, str]  # part_number -> presigned URL
    complete_url: str
    abort_url: str
    expires_at: datetime


class MultipartURLGenerator:
    """Generates URLs for multipart uploads."""

    def __init__(self, signer: URLSigner):
        self.signer = signer

    def generate_multipart_urls(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        num_parts: int,
        expires_in: int = 3600,
    ) -> MultipartUploadURL:
        """Generate URLs for multipart upload parts.

        Args:
            bucket: Bucket name.
            key: Object key.
            upload_id: Multipart upload ID.
            num_parts: Number of parts.
            expires_in: Expiry time in seconds.

        Returns:
            MultipartUploadURL with all part URLs.
        """
        part_urls = {}
        expires = int(time.time()) + expires_in

        for part_num in range(1, num_parts + 1):
            signature = self.signer._compute_signature(
                "PUT", f"{key}?partNumber={part_num}&uploadId={upload_id}", expires
            )
            params = {
                "uploadId": upload_id,
                "partNumber": str(part_num),
                "expires": str(expires),
                "signature": signature,
            }
            query = urllib.parse.urlencode(params)
            part_urls[part_num] = f"{self.signer.base_url}/storage/{bucket}/{key}?{query}"

        # Complete URL
        complete_signature = self.signer._compute_signature(
            "POST", f"{key}?uploadId={upload_id}", expires
        )
        complete_url = (
            f"{self.signer.base_url}/storage/{bucket}/{key}/complete"
            f"?uploadId={upload_id}&expires={expires}&signature={complete_signature}"
        )

        # Abort URL
        abort_signature = self.signer._compute_signature(
            "DELETE", f"{key}?uploadId={upload_id}", expires
        )
        abort_url = (
            f"{self.signer.base_url}/storage/{bucket}/{key}/abort"
            f"?uploadId={upload_id}&expires={expires}&signature={abort_signature}"
        )

        return MultipartUploadURL(
            upload_id=upload_id,
            key=key,
            bucket=bucket,
            part_urls=part_urls,
            complete_url=complete_url,
            abort_url=abort_url,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
        )


# Global URL signer
_url_signer: Optional[URLSigner] = None


def get_url_signer() -> URLSigner:
    """Get the global URL signer."""
    global _url_signer
    if _url_signer is None:
        # Default configuration - should be overridden in production
        import os
        secret = os.getenv("STORAGE_SECRET_KEY", "default-dev-secret-key")
        base_url = os.getenv("STORAGE_BASE_URL", "http://localhost:8000")
        _url_signer = URLSigner(secret_key=secret, base_url=base_url)
    return _url_signer


def configure_url_signer(secret_key: str, base_url: str) -> URLSigner:
    """Configure the global URL signer."""
    global _url_signer
    _url_signer = URLSigner(secret_key=secret_key, base_url=base_url)
    return _url_signer
