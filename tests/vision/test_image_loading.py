"""Image loading tests - URL download and validation.

Tests the image_url downloading functionality:
1. Valid URL downloads successfully
2. Invalid URL returns 400 error
3. Timeout handling
4. Large file rejection
5. HTTP error codes (404, 403, 500)
6. Invalid URL schemes (file://, ftp://)
7. Empty image rejection
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.core.vision import VisionManager, VisionAnalyzeRequest, create_stub_provider, VisionInputError
import httpx


# ========== Test Fixtures ==========


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return sample image bytes (1x1 PNG)."""
    # Minimal 1x1 PNG (black pixel)
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    return png_bytes


@pytest.fixture
def mock_httpx_response_success(sample_image_bytes):
    """Mock successful httpx response."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {'content-length': str(len(sample_image_bytes))}
    response.content = sample_image_bytes
    return response


@pytest.fixture
def mock_httpx_response_404():
    """Mock 404 httpx response."""
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    response.content = b''
    return response


@pytest.fixture
def mock_httpx_response_403():
    """Mock 403 httpx response."""
    response = MagicMock()
    response.status_code = 403
    response.headers = {}
    response.content = b''
    return response


@pytest.fixture
def mock_httpx_response_large_file():
    """Mock response with large content-length header."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {'content-length': str(60 * 1024 * 1024)}  # 60MB
    response.content = b''
    return response


# ========== URL Download Tests ==========


@pytest.mark.asyncio
async def test_image_url_download_success(mock_httpx_response_success):
    """
    Test: Valid URL downloads successfully.

    Expected behavior:
    - HTTP request made to URL
    - Image bytes returned
    - Vision analysis proceeds normally
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://example.com/test.png",
        include_description=True,
        include_ocr=False
    )

    # Mock httpx.AsyncClient
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_httpx_response_success)
        mock_client_class.return_value = mock_client

        # Execute analysis
        response = await manager.analyze(request)

        # Assertions
        assert response.success is True
        assert response.description is not None
        assert "cylindrical part" in response.description.summary.lower()

        # Verify HTTP request was made
        mock_client.get.assert_called_once_with(
            "https://example.com/test.png",
            follow_redirects=True
        )


@pytest.mark.asyncio
async def test_image_url_invalid_scheme():
    """
    Test: Invalid URL scheme returns 400 error.

    Expected behavior:
    - file:// and ftp:// schemes rejected
    - VisionInputError raised with clear message
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    # Test file:// scheme
    request = VisionAnalyzeRequest(
        image_url="file:///path/to/image.png",
        include_description=True,
        include_ocr=False
    )

    with pytest.raises(VisionInputError) as exc_info:
        await manager.analyze(request)

    assert "Invalid URL scheme 'file'" in str(exc_info.value)
    assert "Only http:// and https://" in str(exc_info.value)

    # Test ftp:// scheme
    request_ftp = VisionAnalyzeRequest(
        image_url="ftp://example.com/image.png",
        include_description=True,
        include_ocr=False
    )

    with pytest.raises(VisionInputError) as exc_info:
        await manager.analyze(request_ftp)

    assert "Invalid URL scheme 'ftp'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_image_url_404_error(mock_httpx_response_404):
    """
    Test: HTTP 404 returns appropriate error.

    Expected behavior:
    - VisionInputError raised
    - Error message mentions 404
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://example.com/nonexistent.png",
        include_description=True,
        include_ocr=False
    )

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_httpx_response_404)
        mock_client_class.return_value = mock_client

        with pytest.raises(VisionInputError) as exc_info:
            await manager.analyze(request)

        assert "HTTP 404" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_image_url_403_error(mock_httpx_response_403):
    """
    Test: HTTP 403 returns appropriate error.

    Expected behavior:
    - VisionInputError raised
    - Error message mentions 403 and forbidden
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://example.com/forbidden.png",
        include_description=True,
        include_ocr=False
    )

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_httpx_response_403)
        mock_client_class.return_value = mock_client

        with pytest.raises(VisionInputError) as exc_info:
            await manager.analyze(request)

        assert "HTTP 403" in str(exc_info.value)
        assert "forbidden" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_image_url_timeout():
    """
    Test: Timeout handling (>5s).

    Expected behavior:
    - httpx.TimeoutException raised by client
    - VisionInputError raised with timeout message
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://example.com/slow.png",
        include_description=True,
        include_ocr=False
    )

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))
        mock_client_class.return_value = mock_client

        with pytest.raises(VisionInputError) as exc_info:
            await manager.analyze(request)

        assert "Timeout" in str(exc_info.value)
        assert ">5s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_image_url_large_file_rejection(mock_httpx_response_large_file):
    """
    Test: Large file (>50MB) rejected via content-length header.

    Expected behavior:
    - VisionInputError raised before download
    - Error message mentions file size and 50MB limit
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://example.com/huge.png",
        include_description=True,
        include_ocr=False
    )

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_httpx_response_large_file)
        mock_client_class.return_value = mock_client

        with pytest.raises(VisionInputError) as exc_info:
            await manager.analyze(request)

        error_msg = str(exc_info.value)
        assert "too large" in error_msg.lower()
        assert "50MB" in error_msg or "50 MB" in error_msg


@pytest.mark.asyncio
async def test_image_url_empty_image():
    """
    Test: Empty image (0 bytes) rejected.

    Expected behavior:
    - VisionInputError raised
    - Error message mentions empty image
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://example.com/empty.png",
        include_description=True,
        include_ocr=False
    )

    # Mock response with 0 bytes
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-length': '0'}
    mock_response.content = b''

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        with pytest.raises(VisionInputError) as exc_info:
            await manager.analyze(request)

        assert "empty" in str(exc_info.value).lower()
        assert "0 bytes" in str(exc_info.value)


@pytest.mark.asyncio
async def test_image_url_network_error():
    """
    Test: Network error (DNS failure, connection refused).

    Expected behavior:
    - httpx.RequestError raised by client
    - VisionInputError raised with network error message
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://nonexistent-domain-12345.com/image.png",
        include_description=True,
        include_ocr=False
    )

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection refused"))
        mock_client_class.return_value = mock_client

        with pytest.raises(VisionInputError) as exc_info:
            await manager.analyze(request)

        assert "Network error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_image_url_follows_redirects(sample_image_bytes):
    """
    Test: URL redirects are followed.

    Expected behavior:
    - follow_redirects=True passed to httpx
    - Final destination image downloaded successfully
    """
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    request = VisionAnalyzeRequest(
        image_url="https://example.com/redirect-to-image",
        include_description=True,
        include_ocr=False
    )

    # Mock response (after redirect)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-length': str(len(sample_image_bytes))}
    mock_response.content = sample_image_bytes

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        response = await manager.analyze(request)

        assert response.success is True

        # Verify follow_redirects=True was used
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs.get('follow_redirects') is True
