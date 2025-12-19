"""Unit tests for S3Dedup2DFileStorage using botocore.stub.Stubber."""

from __future__ import annotations

import io
import os
from typing import Any
from unittest import mock

import pytest

# Skip entire module if boto3/botocore not installed
boto3 = pytest.importorskip("boto3")
botocore = pytest.importorskip("botocore")

from botocore.stub import ANY, Stubber


class TestS3Dedup2DFileStorageConfig:
    """Tests for S3 file storage configuration."""

    def test_s3_config_from_env(self) -> None:
        """Load S3 config from environment variables."""
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        env = {
            "DEDUP2D_FILE_STORAGE": "s3",
            "DEDUP2D_S3_BUCKET": "my-bucket",
            "DEDUP2D_S3_PREFIX": "dedup2d/uploads",
            "DEDUP2D_S3_ENDPOINT": "http://localhost:9000",
            "DEDUP2D_S3_REGION": "us-east-1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        assert cfg.backend == "s3"
        assert cfg.s3_bucket == "my-bucket"
        assert cfg.s3_prefix == "dedup2d/uploads"
        assert cfg.s3_endpoint == "http://localhost:9000"
        assert cfg.s3_region == "us-east-1"

    def test_s3_bucket_required(self) -> None:
        """S3 backend requires bucket to be set."""
        from src.core.dedup2d_file_storage import (
            Dedup2DFileStorageConfig,
            S3Dedup2DFileStorage,
        )

        env = {
            "DEDUP2D_FILE_STORAGE": "s3",
            "DEDUP2D_S3_BUCKET": "",  # Empty bucket
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        with pytest.raises(ValueError, match="DEDUP2D_S3_BUCKET is required"):
            S3Dedup2DFileStorage(cfg)


class TestS3Dedup2DFileStorageWithStubber:
    """Tests for S3 file storage operations using botocore Stubber."""

    @pytest.fixture
    def s3_config(self) -> "Dedup2DFileStorageConfig":
        """Create S3 config for testing."""
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        env = {
            "DEDUP2D_FILE_STORAGE": "s3",
            "DEDUP2D_S3_BUCKET": "test-bucket",
            "DEDUP2D_S3_PREFIX": "test-prefix",
            "DEDUP2D_S3_ENDPOINT": "http://localhost:9000",
            "DEDUP2D_S3_REGION": "us-east-1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            return Dedup2DFileStorageConfig.from_env()

    @pytest.fixture
    def s3_storage(self, s3_config: Any) -> "S3Dedup2DFileStorage":
        """Create S3 storage instance for testing."""
        from src.core.dedup2d_file_storage import S3Dedup2DFileStorage

        return S3Dedup2DFileStorage(s3_config)

    @pytest.mark.asyncio
    async def test_save_bytes_calls_put_object(self, s3_storage: Any) -> None:
        """save_bytes should call S3 put_object with correct parameters."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        test_data = b"test file content for S3"
        job_id = "job-123"
        file_name = "drawing.dxf"
        content_type = "application/dxf"

        with Stubber(s3_storage.client) as stubber:
            # Add expected put_object call
            stubber.add_response(
                "put_object",
                {},  # put_object returns empty response
                expected_params={
                    "Bucket": "test-bucket",
                    "Key": ANY,  # Key includes UUID, hard to predict
                    "Body": test_data,
                    "ContentType": content_type,
                },
            )

            file_ref = await s3_storage.save_bytes(
                job_id=job_id,
                file_name=file_name,
                content_type=content_type,
                data=test_data,
            )

            stubber.assert_no_pending_responses()

        assert isinstance(file_ref, Dedup2DFileRef)
        assert file_ref.backend == "s3"
        assert file_ref.bucket == "test-bucket"
        assert file_ref.key is not None
        assert job_id in file_ref.key
        assert "drawing.dxf" in file_ref.key

    @pytest.mark.asyncio
    async def test_save_bytes_sanitizes_filename(self, s3_storage: Any) -> None:
        """save_bytes should sanitize unsafe characters in filename."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        test_data = b"data"
        unsafe_filename = "my file <with> bad:chars?.dxf"

        with Stubber(s3_storage.client) as stubber:
            # Use expected_params with ANY for dynamic values
            stubber.add_response(
                "put_object",
                {},
                expected_params={
                    "Bucket": "test-bucket",
                    "Key": ANY,  # Key includes UUID, hard to predict
                    "Body": test_data,
                    "ContentType": "application/octet-stream",
                },
            )

            file_ref = await s3_storage.save_bytes(
                job_id="job-456",
                file_name=unsafe_filename,
                content_type="application/octet-stream",
                data=test_data,
            )

            stubber.assert_no_pending_responses()

        # Key should not contain special characters
        assert "<" not in file_ref.key
        assert ">" not in file_ref.key
        assert ":" not in file_ref.key
        assert "?" not in file_ref.key

    @pytest.mark.asyncio
    async def test_load_bytes_calls_get_object(self, s3_storage: Any) -> None:
        """load_bytes should call S3 get_object and return file content."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        expected_content = b"file content from S3"
        file_ref = Dedup2DFileRef(
            backend="s3",
            bucket="test-bucket",
            key="test-prefix/job-123/abc_file.dxf",
        )

        # Create a mock streaming body
        body_stream = io.BytesIO(expected_content)

        with Stubber(s3_storage.client) as stubber:
            stubber.add_response(
                "get_object",
                {
                    "Body": body_stream,
                    "ContentType": "application/dxf",
                    "ContentLength": len(expected_content),
                },
                expected_params={
                    "Bucket": "test-bucket",
                    "Key": "test-prefix/job-123/abc_file.dxf",
                },
            )

            content = await s3_storage.load_bytes(file_ref)

            stubber.assert_no_pending_responses()

        assert content == expected_content

    @pytest.mark.asyncio
    async def test_load_bytes_backend_mismatch(self, s3_storage: Any) -> None:
        """load_bytes should raise ValueError for non-S3 file_ref."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        local_ref = Dedup2DFileRef(backend="local", path="some/path.dxf")

        with pytest.raises(ValueError, match="backend mismatch"):
            await s3_storage.load_bytes(local_ref)

    @pytest.mark.asyncio
    async def test_delete_calls_delete_object(self, s3_storage: Any) -> None:
        """delete should call S3 delete_object."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        file_ref = Dedup2DFileRef(
            backend="s3",
            bucket="test-bucket",
            key="test-prefix/job-123/file.dxf",
        )

        with Stubber(s3_storage.client) as stubber:
            stubber.add_response(
                "delete_object",
                {},
                expected_params={
                    "Bucket": "test-bucket",
                    "Key": "test-prefix/job-123/file.dxf",
                },
            )

            await s3_storage.delete(file_ref)

            stubber.assert_no_pending_responses()

    @pytest.mark.asyncio
    async def test_delete_ignores_non_s3_ref(self, s3_storage: Any) -> None:
        """delete should silently ignore non-S3 file_ref."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        local_ref = Dedup2DFileRef(backend="local", path="some/path.dxf")

        # Should not raise, just return silently
        await s3_storage.delete(local_ref)

    @pytest.mark.asyncio
    async def test_delete_handles_error_gracefully(self, s3_storage: Any) -> None:
        """delete should handle S3 errors gracefully (log, don't raise)."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        file_ref = Dedup2DFileRef(
            backend="s3",
            bucket="test-bucket",
            key="test-prefix/job-123/file.dxf",
        )

        with Stubber(s3_storage.client) as stubber:
            # Simulate S3 error
            stubber.add_client_error(
                "delete_object",
                service_error_code="NoSuchKey",
                service_message="The specified key does not exist.",
            )

            # Should not raise
            await s3_storage.delete(file_ref)


class TestS3FileRefSerialization:
    """Tests for S3 file reference serialization."""

    def test_file_ref_to_dict(self) -> None:
        """Dedup2DFileRef.to_dict() should serialize S3 refs correctly."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        ref = Dedup2DFileRef(backend="s3", bucket="my-bucket", key="path/to/file.dxf")
        d = ref.to_dict()

        assert d == {
            "backend": "s3",
            "bucket": "my-bucket",
            "key": "path/to/file.dxf",
        }

    def test_file_ref_from_dict(self) -> None:
        """Dedup2DFileRef.from_dict() should deserialize S3 refs correctly."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        d = {"backend": "s3", "bucket": "my-bucket", "key": "path/to/file.dxf"}
        ref = Dedup2DFileRef.from_dict(d)

        assert ref.backend == "s3"
        assert ref.bucket == "my-bucket"
        assert ref.key == "path/to/file.dxf"

    def test_file_ref_from_dict_missing_bucket(self) -> None:
        """S3 file_ref requires bucket."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        d = {"backend": "s3", "key": "path/to/file.dxf"}  # Missing bucket

        with pytest.raises(ValueError, match="s3 file_ref requires bucket and key"):
            Dedup2DFileRef.from_dict(d)

    def test_file_ref_from_dict_missing_key(self) -> None:
        """S3 file_ref requires key."""
        from src.core.dedup2d_file_storage import Dedup2DFileRef

        d = {"backend": "s3", "bucket": "my-bucket"}  # Missing key

        with pytest.raises(ValueError, match="s3 file_ref requires bucket and key"):
            Dedup2DFileRef.from_dict(d)


class TestCreateFileStorageFactory:
    """Tests for create_dedup2d_file_storage factory function."""

    def test_create_s3_storage(self) -> None:
        """Factory should create S3 storage when backend is 's3'."""
        from src.core.dedup2d_file_storage import (
            S3Dedup2DFileStorage,
            create_dedup2d_file_storage,
        )

        env = {
            "DEDUP2D_FILE_STORAGE": "s3",
            "DEDUP2D_S3_BUCKET": "test-bucket",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            storage = create_dedup2d_file_storage()

        assert isinstance(storage, S3Dedup2DFileStorage)

    def test_create_local_storage(self) -> None:
        """Factory should create local storage when backend is 'local'."""
        from src.core.dedup2d_file_storage import (
            LocalDedup2DFileStorage,
            create_dedup2d_file_storage,
        )

        env = {
            "DEDUP2D_FILE_STORAGE": "local",
            "DEDUP2D_FILE_STORAGE_DIR": "/tmp/test-uploads",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            storage = create_dedup2d_file_storage()

        assert isinstance(storage, LocalDedup2DFileStorage)

    def test_create_unknown_backend_raises(self) -> None:
        """Factory should raise for unknown backend."""
        from src.core.dedup2d_file_storage import create_dedup2d_file_storage

        env = {
            "DEDUP2D_FILE_STORAGE": "unknown",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Unknown DEDUP2D_FILE_STORAGE"):
                create_dedup2d_file_storage()
